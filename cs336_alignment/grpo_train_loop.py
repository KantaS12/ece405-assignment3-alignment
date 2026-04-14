#!/usr/bin/env python3
import os
import sys
import json
import argparse
import random
from typing import Literal

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drgrpo_grader import r1_zero_reward_fn
from grpo_implementation import compute_group_normalized_rewards, grpo_microbatch_train_step
from sft_helper import tokenize_prompt_and_output

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Hyperparameters
N_GRPO_STEPS: int = 200
LEARNING_RATE: float = 1e-4
ADVANTAGE_EPS: float = 1e-6
ROLLOUT_BATCH_SIZE: int = 256
GROUP_SIZE: int = 8
SAMPLING_TEMPERATURE: float = 1.0
SAMPLING_MIN_TOKENS: int = 4
SAMPLING_MAX_TOKENS: int = 1024
EPOCHS_PER_ROLLOUT_BATCH: int = 1
TRAIN_BATCH_SIZE: int = 256
GRADIENT_ACCUMULATION_STEPS: int = 128
GPU_MEMORY_UTILIZATION: float = 0.85
GRAD_CLIP: float = 1.0
CLIPRANGE: float = 0.2

LOSS_TYPE: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline"
USE_STD_NORMALIZATION: bool = True


assert TRAIN_BATCH_SIZE % GRADIENT_ACCUMULATION_STEPS == 0, (
    "TRAIN_BATCH_SIZE must be divisible by GRADIENT_ACCUMULATION_STEPS"
)
micro_train_batch_size = TRAIN_BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS

assert ROLLOUT_BATCH_SIZE % micro_train_batch_size == 0, (
    "ROLLOUT_BATCH_SIZE must be divisible by micro_train_batch_size"
)
assert TRAIN_BATCH_SIZE >= GROUP_SIZE, (
    "TRAIN_BATCH_SIZE must be >= GROUP_SIZE"
)
assert ROLLOUT_BATCH_SIZE % GROUP_SIZE == 0, (
    "ROLLOUT_BATCH_SIZE must be divisible by GROUP_SIZE"
)

n_prompts_per_rollout_batch = ROLLOUT_BATCH_SIZE // GROUP_SIZE
n_microbatches_per_rollout_batch = ROLLOUT_BATCH_SIZE // micro_train_batch_size

assert n_microbatches_per_rollout_batch == GRADIENT_ACCUMULATION_STEPS, (
    "n_microbatches_per_rollout_batch must equal GRADIENT_ACCUMULATION_STEPS "
    "(one optimizer step per rollout batch)"
)

# Prompt Template
with open(os.path.join(SCRIPT_DIR, "prompts/r1_zero.prompt")) as _f:
    _R1_TEMPLATE = _f.read().strip()
STOP_STR = "</answer>"


def build_prompt(question: str) -> str:
    return _R1_TEMPLATE.format(question=question)


def _get_field(item: dict, *keys: str) -> str:
    """Return the first key found in item, raise if none exist."""
    for k in keys:
        if k in item:
            return item[k]
    raise KeyError(f"None of {keys} found in item: {list(item.keys())}")


# Helper
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
            max_model_len=SAMPLING_MAX_TOKENS + 512,  # extra buffer for prompt tokens
        )


def load_policy_into_vllm(policy: torch.nn.Module, llm: LLM) -> None:
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def sample_and_generate_rollout_batch(
    policy: torch.nn.Module,
    llm: LLM,
    train_items: list[dict],
    seed_offset: int,
) -> tuple[list[dict], dict]:
    """
    Sample n_prompts_per_rollout_batch questions, generate GROUP_SIZE responses
    each via vLLM, compute group-normalised advantages.

    Returns
    -------
    rollout_batch : list[dict]  length == ROLLOUT_BATCH_SIZE
        Each dict has keys: prompt, response, ground_truth, advantage, raw_reward.
    reward_metadata : dict[str, float]
    """
    batch_items = random.sample(train_items, n_prompts_per_rollout_batch)
    prompts = [build_prompt(_get_field(it, "problem", "question")) for it in batch_items]
    ground_truths = [_get_field(it, "answer", "solution") for it in batch_items]

    # Sync latest policy weights into vLLM before generation
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
        for i in range(ROLLOUT_BATCH_SIZE)
    ]
    return rollout_batch, reward_metadata


# Forward Pass 
def compute_policy_log_probs(
    policy: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return per-token log-probs under the current policy (gradients enabled)."""
    outputs = policy(input_ids=input_ids)
    logits = outputs.logits  # (B, T, V)
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, T)


def main():
    parser = argparse.ArgumentParser(description="GRPO training loop")
    parser.add_argument("--model_path", required=True,
                        help="Path to the SFT checkpoint to start from")
    parser.add_argument("--data_path", default=None,
                        help="Path to JSONL training data (default: data/math/train.jsonl)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--policy_device", default="cuda:0")
    parser.add_argument("--vllm_device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save a checkpoint every N GRPO steps")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = args.data_path or os.path.join(SCRIPT_DIR, "../data/math/train.jsonl")
    model_tag = os.path.basename(args.model_path.rstrip("/"))
    output_dir = args.output_dir or os.path.join(
        SCRIPT_DIR, f"../models/{model_tag}_grpo_{LOSS_TYPE}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Data loading
    train_items = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                train_items.append(json.loads(line))
    print(f"Loaded {len(train_items)} training examples from {data_path}")

    # Policy Load
    print(f"Loading policy from {args.model_path} ...")
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(args.policy_device)
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"

    # Optimizer
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    # vLLM
    print(f"Initialising vLLM on {args.vllm_device} ...")
    llm = init_vllm(args.model_path, args.vllm_device, args.seed)

    log = []

    for grpo_step in range(N_GRPO_STEPS):
        print(f"\n{'='*60}")
        print(f"GRPO Step {grpo_step + 1}/{N_GRPO_STEPS}  loss={LOSS_TYPE}")
        print("=" * 60)

        policy.eval()
        rollout_batch, reward_meta = sample_and_generate_rollout_batch(
            policy=policy,
            llm=llm,
            train_items=train_items,
            seed_offset=args.seed + grpo_step,
        )
        print(f"  Rewards  mean={reward_meta['mean_reward']:.4f}  "
              f"std={reward_meta['std_reward']:.4f}  "
              f"max={reward_meta['max_reward']:.4f}  "
              f"min={reward_meta['min_reward']:.4f}")

        old_log_probs_per_mb = None
        if LOSS_TYPE == "grpo_clip":
            old_log_probs_per_mb = []
            policy.eval()
            with torch.no_grad():
                for mb_idx in range(n_microbatches_per_rollout_batch):
                    mb = rollout_batch[
                        mb_idx * micro_train_batch_size:
                        (mb_idx + 1) * micro_train_batch_size
                    ]
                    tokens = tokenize_prompt_and_output(
                        [item["prompt"] for item in mb],
                        [item["response"] for item in mb],
                        tokenizer,
                    )
                    old_lp = compute_policy_log_probs(
                        policy,
                        tokens["input_ids"].to(args.policy_device),
                        tokens["labels"].to(args.policy_device),
                    )
                    old_log_probs_per_mb.append(old_lp.cpu())

        policy.train()
        step_log = {"grpo_step": grpo_step + 1, **reward_meta}

        for epoch in range(EPOCHS_PER_ROLLOUT_BATCH):
            optimizer.zero_grad()
            epoch_loss = 0.0

            for mb_idx in range(n_microbatches_per_rollout_batch):
                mb = rollout_batch[
                    mb_idx * micro_train_batch_size:
                    (mb_idx + 1) * micro_train_batch_size
                ]

                # Tokenize
                tokens = tokenize_prompt_and_output(
                    [item["prompt"] for item in mb],
                    [item["response"] for item in mb],
                    tokenizer,
                )
                input_ids = tokens["input_ids"].to(args.policy_device)
                labels = tokens["labels"].to(args.policy_device)
                response_mask = tokens["response_mask"].float().to(args.policy_device)

                # Per-sample reward tensors  shape: (batch_size, 1)
                mb_advantages = torch.tensor(
                    [item["advantage"] for item in mb], dtype=torch.float32
                ).unsqueeze(-1).to(args.policy_device)
                mb_raw_rewards = torch.tensor(
                    [item["raw_reward"] for item in mb], dtype=torch.float32
                ).unsqueeze(-1).to(args.policy_device)

                # Forward pass with gradients
                policy_log_probs = compute_policy_log_probs(policy, input_ids, labels)

                # Old log probs only needed for GRPO-clip
                old_lp = None
                if LOSS_TYPE == "grpo_clip":
                    old_lp = old_log_probs_per_mb[mb_idx].to(args.policy_device)

                loss, _ = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                    loss_type=LOSS_TYPE,
                    raw_rewards=mb_raw_rewards if LOSS_TYPE == "no_baseline" else None,
                    advantages=mb_advantages if LOSS_TYPE != "no_baseline" else None,
                    old_log_probs=old_lp,
                    cliprange=CLIPRANGE if LOSS_TYPE == "grpo_clip" else None,
                )
                epoch_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
            optimizer.step()
            print(f"  Epoch {epoch + 1}/{EPOCHS_PER_ROLLOUT_BATCH}  loss={epoch_loss:.6f}")

        step_log["loss"] = epoch_loss
        log.append(step_log)

        # Periodic checkpoint
        if (grpo_step + 1) % args.save_every == 0:
            ckpt = os.path.join(output_dir, f"step_{grpo_step + 1}")
            policy.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"  Checkpoint: {ckpt}")

    # Save final model
    final_dir = os.path.join(output_dir, "final")
    policy.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nFinal model saved to {final_dir}")

    log_path = os.path.join(output_dir, "grpo_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
