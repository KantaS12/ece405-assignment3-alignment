"""
This is the training of GRPO w/ standard deviation normalization without normalization. 
"""
import os
import sys
import json
import random
from typing import Literal

import torch
import torch.nn.functional as F
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer
from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drgrpo_grader import r1_zero_reward_fn
from grpo_implementation import compute_group_normalized_rewards, grpo_microbatch_train_step_mean_normalized
from sft_helper import tokenize_prompt_and_output

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Hyperparameters
N_GRPO_STEPS: int = 200
LEARNING_RATE: float = 1e-5
ADVANTAGE_EPS: float = 1e-6
ROLLOUT_BATCH_SIZE: int = 256
GROUP_SIZE: int = 8
SAMPLING_TEMPERATURE: float = 1.0
SAMPLING_MIN_TOKENS: int = 4
SAMPLING_MAX_TOKENS: int = 512
EPOCHS_PER_ROLLOUT_BATCH: int = 1
TRAIN_BATCH_SIZE: int = 256
GRADIENT_ACCUMULATION_STEPS: int = 128
GPU_MEMORY_UTILIZATION: float = 0.70
GRAD_CLIP: float = 1.0
CLIPRANGE: float = 0.2
VAL_SIZE: int = 1024
VAL_EVERY: int = 5

LOSS_TYPE: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline"
USE_STD_NORMALIZATION: bool = False

# Derived constants (validated at import time)
assert TRAIN_BATCH_SIZE % GRADIENT_ACCUMULATION_STEPS == 0
MICRO_TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS
assert ROLLOUT_BATCH_SIZE % MICRO_TRAIN_BATCH_SIZE == 0
assert TRAIN_BATCH_SIZE >= GROUP_SIZE
assert ROLLOUT_BATCH_SIZE % GROUP_SIZE == 0
N_PROMPTS_PER_ROLLOUT = ROLLOUT_BATCH_SIZE // GROUP_SIZE
N_MICROBATCHES = ROLLOUT_BATCH_SIZE // MICRO_TRAIN_BATCH_SIZE
assert N_MICROBATCHES == GRADIENT_ACCUMULATION_STEPS

# Prompt template
with open(os.path.join(SCRIPT_DIR, "prompts/r1_zero.prompt")) as _f:
    _R1_TEMPLATE = _f.read().strip()
STOP_STR = "</answer>"


def build_prompt(question: str) -> str:
    return _R1_TEMPLATE.format(question=question)


def _get_field(item: dict, *keys: str) -> str:
    for k in keys:
        if k in item:
            return item[k]
    raise KeyError(f"None of {keys} found in item keys={list(item.keys())}")


# vLLM Helpers

def init_vllm(model_path: str, device: str, seed: int,
              gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION) -> LLM:
    vllm_set_random_seed(seed)
    world_patch = patch("torch.distributed.get_world_size", return_value=1)
    prof_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_patch, prof_patch:
        return LLM(
            model=model_path,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=SAMPLING_MAX_TOKENS + 512,
        )


def load_policy_into_vllm(policy: torch.nn.Module, llm: LLM) -> None:
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


# Helper Functions

def compute_policy_log_probs(
    policy: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return per-token log-probs (B, T) under current policy."""
    outputs = policy(input_ids=input_ids)
    logits = outputs.logits  # (B, T, V)
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def compute_policy_log_probs_and_entropy(
    policy: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """
    Single forward pass returning per-token log-probs (B, T) and mean
    per-token entropy (scalar float) averaged over response tokens.
    """
    outputs = policy(input_ids=input_ids)
    logits = outputs.logits  # (B, T, V)

    # Compute entropy first under no_grad so we can free probs immediately
    with torch.no_grad():
        log_probs_d = F.log_softmax(logits.detach(), dim=-1)
        probs_d = log_probs_d.exp()
        token_entropy = -(probs_d * log_probs_d).sum(dim=-1)  # (B, T)
        del probs_d, log_probs_d
        mask_f = response_mask.float()
        mean_entropy = (token_entropy * mask_f).sum() / mask_f.sum().clamp(min=1)
        mean_entropy = mean_entropy.item()
        del token_entropy, mask_f

    # Compute log_probs for gradient; delete logits to free (B,T,V) from Python scope
    log_probs = F.log_softmax(logits, dim=-1)
    del logits
    token_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, T)
    del log_probs  # autograd still holds ref until backward; Python ref freed

    return token_lp, mean_entropy


# Rollout

def sample_and_generate_rollout_batch(
    policy: torch.nn.Module,
    llm: LLM,
    train_items: list[dict],
    seed_offset: int,
) -> tuple[list[dict], dict]:
    """
    Sample N_PROMPTS_PER_ROLLOUT questions, generate GROUP_SIZE completions each,
    compute group-normalised advantages, and return per-item rollout dicts.
    """
    batch_items = random.sample(train_items, N_PROMPTS_PER_ROLLOUT)
    prompts = [build_prompt(_get_field(it, "problem", "question")) for it in batch_items]
    ground_truths = [_get_field(it, "answer", "solution") for it in batch_items]

    load_policy_into_vllm(policy, llm)

    params = SamplingParams(
        temperature=SAMPLING_TEMPERATURE,
        min_tokens=SAMPLING_MIN_TOKENS,
        max_tokens=SAMPLING_MAX_TOKENS,
        n=GROUP_SIZE,
        stop=[STOP_STR],
        seed=seed_offset,
    )
    outputs = llm.generate(prompts, params)

    rollout_prompts: list[str] = []
    rollout_responses: list[str] = []
    rollout_gts: list[str] = []

    for out, gt, prompt in zip(outputs, ground_truths, prompts):
        for gen in out.outputs:
            rollout_prompts.append(prompt)
            rollout_responses.append(gen.text + STOP_STR)
            rollout_gts.append(gt)

    # compute_group_normalized_rewards calls reward_fn once per response and
    # now returns mean_format_reward / mean_answer_reward in its metadata.
    advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
        reward_fn=r1_zero_reward_fn,
        rollout_reponses=rollout_responses,
        repeated_ground_truths=rollout_gts,
        group_size=GROUP_SIZE,
        advantage_eps=ADVANTAGE_EPS,
        normalize_by_std=USE_STD_NORMALIZATION,
    )

    rollout_batch = [
        {
            "prompt": rollout_prompts[i],
            "response": rollout_responses[i],
            "ground_truth": rollout_gts[i],
            "advantage": advantages[i].item(),
            "raw_reward": raw_rewards[i].item(),
        }
        for i in range(len(rollout_responses))
    ]
    return rollout_batch, reward_metadata


# Validation

def evaluate_validation(
    policy: torch.nn.Module,
    llm: LLM,
    val_items: list[dict],
    seed: int,
) -> dict:
    """
    Greedy decode on val_items, score with r1_zero_reward_fn.
    Returns reward stats and a few example rollouts.
    """
    load_policy_into_vllm(policy, llm)

    prompts = [build_prompt(_get_field(it, "problem", "question")) for it in val_items]
    ground_truths = [_get_field(it, "answer", "solution") for it in val_items]

    params = SamplingParams(
        temperature=0.0,
        max_tokens=SAMPLING_MAX_TOKENS,
        n=1,
        stop=[STOP_STR],
        seed=seed,
    )
    outputs = llm.generate(prompts, params)

    total_r, format_r, answer_r = [], [], []
    examples = []

    for i, (out, gt, prompt) in enumerate(zip(outputs, ground_truths, prompts)):
        response = out.outputs[0].text + STOP_STR
        rd = r1_zero_reward_fn(response, gt)
        total_r.append(rd["reward"])
        format_r.append(rd["format_reward"])
        answer_r.append(rd["answer_reward"])
        if i < 5:
            examples.append({
                "prompt_tail": prompt[-300:],
                "response": response[:600],
                "ground_truth": gt,
                "reward": rd["reward"],
                "format_reward": rd["format_reward"],
                "answer_reward": rd["answer_reward"],
            })

    n = len(total_r)
    return {
        "val_mean_reward": sum(total_r) / n,
        "val_mean_format_reward": sum(format_r) / n,
        "val_mean_answer_reward": sum(answer_r) / n,
        "val_examples": examples,
    }


# Training Loop

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model_path: str = typer.Argument(..., help="SFT checkpoint to initialise from"),
    data_path: str = typer.Option(None, help="JSONL training data (default: data/math/train.jsonl)"),
    val_data_path: str = typer.Option(None, help="JSONL validation data; if absent, hold out first VAL_SIZE from train"),
    output_dir: str = typer.Option(None, help="Where to save checkpoints and logs"),
    policy_device: str = typer.Option("cuda:0"),
    vllm_device: str = typer.Option("cuda:1"),
    seed: int = typer.Option(42),
    save_every: int = typer.Option(10, help="Checkpoint every N GRPO steps"),
    val_every: int = typer.Option(VAL_EVERY, help="Validate every N GRPO steps"),
    n_grpo_steps: int = typer.Option(N_GRPO_STEPS, help="Total GRPO steps"),
    loss_type: str = typer.Option(LOSS_TYPE, help="no_baseline | reinforce_with_baseline | grpo_clip"),
    lr: float = typer.Option(LEARNING_RATE, help="AdamW learning rate"),
):
    random.seed(seed)
    torch.manual_seed(seed)

    # Resolve paths
    data_path = data_path or os.path.join(SCRIPT_DIR, "../data/math/train.jsonl")
    model_tag = os.path.basename(model_path.rstrip("/"))
    output_dir = output_dir or os.path.join(
        SCRIPT_DIR, f"../models/{model_tag}_grpo_{loss_type}_lr{lr:.0e}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    all_items: list[dict] = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                all_items.append(json.loads(line))

    if val_data_path and os.path.exists(val_data_path):
        val_items: list[dict] = []
        with open(val_data_path) as f:
            for line in f:
                if line.strip():
                    val_items.append(json.loads(line))
        val_items = val_items[:VAL_SIZE]
        train_items = all_items
    else:
        # Split Data (Shuffle with seed then hold the val_size)
        rng = random.Random(seed)
        idx = list(range(len(all_items)))
        rng.shuffle(idx)
        val_items = [all_items[i] for i in idx[:VAL_SIZE]]
        train_items = [all_items[i] for i in idx[VAL_SIZE:]]

    print(f"Train: {len(train_items)}  Val: {len(val_items)}")
    print(f"loss_type={loss_type}  n_grpo_steps={n_grpo_steps}  "
          f"rollout_batch={ROLLOUT_BATCH_SIZE}  group_size={GROUP_SIZE}")

    # Model
    print(f"Loading policy from {model_path} …")
    policy = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(policy_device)
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.95),
    )

    print(f"Initialising vLLM on {vllm_device} …")
    llm = init_vllm(model_path, vllm_device, seed)

    train_log: list[dict] = []
    val_log: list[dict] = []

    for grpo_step in range(n_grpo_steps):
        sep = "=" * 60
        print(f"\n{sep}\nGRPO Step {grpo_step + 1}/{n_grpo_steps}  loss={loss_type}\n{sep}")

        # Rollout
        policy.eval()
        rollout_batch, reward_meta = sample_and_generate_rollout_batch(
            policy=policy,
            llm=llm,
            train_items=train_items,
            seed_offset=seed + grpo_step,
        )
        print(f"  [train rewards]  total={reward_meta['mean_reward']:.4f}  "
              f"format={reward_meta['mean_format_reward']:.4f}  "
              f"answer={reward_meta['mean_answer_reward']:.4f}")

        # Old log-probs for clipping, if needed
        old_log_probs_per_mb: list[torch.Tensor] | None = None
        if loss_type == "grpo_clip":
            old_log_probs_per_mb = []
            policy.eval()
            with torch.no_grad():
                for mb_idx in range(N_MICROBATCHES):
                    mb = rollout_batch[mb_idx * MICRO_TRAIN_BATCH_SIZE:
                                       (mb_idx + 1) * MICRO_TRAIN_BATCH_SIZE]
                    tokens = tokenize_prompt_and_output(
                        [it["prompt"] for it in mb],
                        [it["response"] for it in mb],
                        tokenizer,
                    )
                    old_lp = compute_policy_log_probs(
                        policy,
                        tokens["input_ids"].to(policy_device),
                        tokens["labels"].to(policy_device),
                    )
                    old_log_probs_per_mb.append(old_lp.cpu())

        # Training
        policy.train()
        step_log = {
            "grpo_step": grpo_step + 1,
            "lr": lr,
            "train_mean_reward": reward_meta["mean_reward"],
            "train_mean_format_reward": reward_meta["mean_format_reward"],
            "train_mean_answer_reward": reward_meta["mean_answer_reward"],
        }

        # One optimizer step per rollout batch (EPOCHS_PER_ROLLOUT_BATCH may be > 1 for off-policy)
        for epoch in range(EPOCHS_PER_ROLLOUT_BATCH):
            optimizer.zero_grad()
            epoch_loss = 0.0
            epoch_entropy = 0.0
            epoch_clip_frac = 0.0

            for mb_idx in range(N_MICROBATCHES):
                mb = rollout_batch[mb_idx * MICRO_TRAIN_BATCH_SIZE:
                                   (mb_idx + 1) * MICRO_TRAIN_BATCH_SIZE]
                tokens = tokenize_prompt_and_output(
                    [it["prompt"] for it in mb],
                    [it["response"] for it in mb],
                    tokenizer,
                )
                input_ids = tokens["input_ids"].to(policy_device)
                labels = tokens["labels"].to(policy_device)
                response_mask = tokens["response_mask"].float().to(policy_device)

                mb_advantages = torch.tensor(
                    [it["advantage"] for it in mb], dtype=torch.float32
                ).unsqueeze(-1).to(policy_device)
                mb_raw_rewards = torch.tensor(
                    [it["raw_reward"] for it in mb], dtype=torch.float32
                ).unsqueeze(-1).to(policy_device)

                # Forward pass — also compute token entropy for logging
                policy_log_probs, mb_entropy = compute_policy_log_probs_and_entropy(
                    policy, input_ids, labels, response_mask
                )
                epoch_entropy += mb_entropy / N_MICROBATCHES

                old_lp = None
                if loss_type == "grpo_clip":
                    old_lp = old_log_probs_per_mb[mb_idx].to(policy_device)

                loss, meta = grpo_microbatch_train_step_mean_normalized(
                    policy_log_probs=policy_log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                    loss_type=loss_type,
                    raw_rewards=mb_raw_rewards if loss_type == "no_baseline" else None,
                    advantages=mb_advantages if loss_type != "no_baseline" else None,
                    old_log_probs=old_lp,
                    cliprange=CLIPRANGE if loss_type == "grpo_clip" else None,
                )
                epoch_loss += loss.item()
                if loss_type == "grpo_clip" and "percent_clipped" in meta:
                    epoch_clip_frac += meta["percent_clipped"] / N_MICROBATCHES

            # Gradient clipping — returns the pre-clip global norm
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
            optimizer.step()
            torch.cuda.empty_cache()

            clip_str = f"  clip_frac={epoch_clip_frac:.4f}" if loss_type == "grpo_clip" else ""
            print(f"  epoch {epoch + 1}  loss={epoch_loss:.6f}  "
                  f"grad_norm={grad_norm:.4f}  entropy={epoch_entropy:.4f}{clip_str}")

        step_log["loss"] = epoch_loss
        step_log["grad_norm"] = float(grad_norm)
        step_log["token_entropy"] = epoch_entropy
        if loss_type == "grpo_clip":
            step_log["clip_fraction"] = epoch_clip_frac

        # Validation
        if (grpo_step + 1) % val_every == 0:
            print(f"  [validation] evaluating {len(val_items)} examples …")
            policy.eval()
            val_result = evaluate_validation(policy, llm, val_items, seed=seed + grpo_step)
            step_log.update({
                "val_mean_reward": val_result["val_mean_reward"],
                "val_mean_format_reward": val_result["val_mean_format_reward"],
                "val_mean_answer_reward": val_result["val_mean_answer_reward"],
            })
            val_entry = {
                "grpo_step": grpo_step + 1,
                "val_mean_reward": val_result["val_mean_reward"],
                "val_mean_format_reward": val_result["val_mean_format_reward"],
                "val_mean_answer_reward": val_result["val_mean_answer_reward"],
                "examples": val_result["val_examples"],
            }
            val_log.append(val_entry)

            print(f"  [val rewards]  total={val_result['val_mean_reward']:.4f}  "
                  f"format={val_result['val_mean_format_reward']:.4f}  "
                  f"answer={val_result['val_mean_answer_reward']:.4f}")
            for ex in val_result["val_examples"][:3]:
                print(f"    reward={ex['reward']}  "
                      f"response={ex['response'][:200]!r}")

        train_log.append(step_log)

        # Checkpointing
        if (grpo_step + 1) % save_every == 0:
            ckpt = os.path.join(output_dir, f"step_{grpo_step + 1}")
            policy.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"  [checkpoint] saved to {ckpt}")

    # Final save
    final_dir = os.path.join(output_dir, "final")
    policy.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nFinal model -> {final_dir}")

    with open(os.path.join(output_dir, "grpo_log.json"), "w") as f:
        json.dump(train_log, f, indent=2)
    with open(os.path.join(output_dir, "val_log.json"), "w") as f:
        json.dump(val_log, f, indent=2)
    print("Logs saved.")


if __name__ == "__main__":
    app()
