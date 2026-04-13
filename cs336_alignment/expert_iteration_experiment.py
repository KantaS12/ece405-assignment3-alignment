#!/usr/bin/env python3
import os, sys, json, math, re, argparse, random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drgrpo_grader import r1_zero_reward_fn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Hyperparameters

SEQ_LENGTH     = 512
WARMUP_RATIO   = 0.03
GRAD_CLIP      = 1.0
VAL_SAMPLES    = 128
MAX_NEW_TOKENS = 512

with open(os.path.join(SCRIPT_DIR, "prompts/r1_zero.prompt")) as _f:
    _R1_TEMPLATE = _f.read().strip()

STOP_STR = "</answer>"


# Utilities

# Prompt construction and reward extraction
def build_prompt(question: str) -> str:
    return _R1_TEMPLATE.format(question=question)

# GSM8K answers have the format "#### <answer>", so we can extract the answer for grading.
def extract_gsm8k_gt(answer_str: str) -> str:
    """Pull the number after #### from a GSM8K answer string."""
    m = re.findall(r"####\s*([^\n]+)", answer_str)
    return m[-1].strip().replace(",", "") if m else answer_str.strip()

# Reward function
def reward(response: str, gt: str) -> float:
    """Binary reward via r1_zero grader. response must contain </answer>."""
    return float(r1_zero_reward_fn(response, gt)["reward"])


# vLLM helpers

def init_vllm(model_path: str, device: str, seed: int,
              gpu_memory_utilization: float = 0.85) -> LLM:
    vllm_set_random_seed(seed)
    world_patch = patch("torch.distributed.get_world_size", return_value=1)
    prof_patch  = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_patch, prof_patch:
        return LLM(
            model=model_path, device=device, dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=512,
        )


# Load policy weights into vLLM model
def load_policy_into_vllm(policy: torch.nn.Module, llm: LLM) -> None:
    state_dict = policy.state_dict()
    llm_model  = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


# In memory SFT dataset

class PairDataset(torch.utils.data.Dataset):
    """Tokenizes (prompt, response) pairs; masks prompt tokens from loss."""

    def __init__(self, pairs, tokenizer, seq_length=SEQ_LENGTH):
        self.samples = []
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        for prompt, response in pairs:
            full      = prompt + response + tokenizer.eos_token
            full_ids  = tokenizer(full, add_special_tokens=False)["input_ids"]
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            prompt_len = len(prompt_ids)

            # (input, label) are offset by 1 for causal LM
            ids    = full_ids[: seq_length + 1]
            inputs = ids[:-1]
            labels = ids[1:]

            # Mask prompt tokens
            mask_len = min(prompt_len, len(labels))
            labels   = [-100] * mask_len + labels[mask_len:]

            # Pad to seq_length
            pad = seq_length - len(inputs)
            if pad > 0:
                inputs = inputs + [pad_id] * pad
                labels = labels + [-100]   * pad

            self.samples.append({
                "input_ids": torch.tensor(inputs[:seq_length], dtype=torch.long),
                "labels":    torch.tensor(labels[:seq_length],  dtype=torch.long),
            })

    def __len__(self):              
        return len(self.samples)

    def __getitem__(self, i):       
        return self.samples[i]

# Collate fn to batch dict of tensors
def _collate(batch):
    return {k: torch.stack([s[k] for s in batch]) for k in batch[0]}


# SFT step

def sft_step(model, tokenizer, correct_pairs, lr, epochs, device):
    """
    Fine-tune `model` on `correct_pairs` for `epochs` passes.
    Returns list of per-batch losses (for logging).
    """
    if not correct_pairs:
        print("  [SFT] No correct pairs — skipping.")
        return []

    dataset = PairDataset(correct_pairs, tokenizer)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=_collate
    )

    total_steps  = len(loader) * epochs
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    optimizer    = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    model.train()
    model.config.use_cache = False
    losses = []

    for epoch in range(epochs):
        for batch in loader:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(inputs).logits
            loss   = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

    torch.cuda.empty_cache()
    print(f"  [SFT] {len(correct_pairs)} pairs | {epochs} epoch(s) | "
          f"avg loss {sum(losses)/len(losses):.4f}")
    return losses


# Evaluation Helper

def compute_val_accuracy(llm, val_items, n=VAL_SAMPLES):
    prompts = [build_prompt(it["question"]) for it in val_items[:n]]
    params  = SamplingParams(
        temperature=0, max_tokens=MAX_NEW_TOKENS, stop=[STOP_STR]
    )
    outputs = llm.generate(prompts, params)
    correct = 0
    for out, item in zip(outputs, val_items[:n]):
        # vLLM strips the stop string; re-append so the grader sees </answer>
        resp = out.outputs[0].text + STOP_STR
        gt   = extract_gsm8k_gt(item["answer"])
        correct += int(reward(resp, gt) == 1.0)
    acc = correct / len(prompts)
    print(f"  Val accuracy: {correct}/{len(prompts)} = {acc:.2%}")
    return acc


# Entropy Helper
def compute_entropy(model, tokenizer, val_items, device, n=64):
    """
    Average token-level entropy of the model's output distribution,
    measured on the first n val prompts.
    """
    model.eval()
    entropies = []
    with torch.no_grad():
        for item in val_items[:n]:
            ids = tokenizer(
                build_prompt(item["question"]),
                return_tensors="pt", truncation=True, max_length=SEQ_LENGTH,
            ).input_ids.to(device)
            logits = model(ids).logits[0]          # (T, V)
            probs  = F.softmax(logits, dim=-1)
            ent    = -(probs * probs.log().clamp(min=-1e9)).sum(dim=-1).mean().item()
            entropies.append(ent)
    model.train()
    torch.cuda.empty_cache()
    return sum(entropies) / len(entropies) if entropies else 0.0


# Main Expected Iteration Loop

def run_expert_iteration(
    model_path, data_path, output_dir,
    n_ei_steps, G, db_size, sft_epochs, lr,
    policy_device, vllm_device, seed,
):
    random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    all_items = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                all_items.append(json.loads(line))
    random.shuffle(all_items)
    n_val       = max(VAL_SAMPLES, int(len(all_items) * 0.1))
    val_items   = all_items[:n_val]
    train_items = all_items[n_val:]
    print(f"Train: {len(train_items)} | Val: {len(val_items)}")

    # Load policy
    print(f"\nLoading policy from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(policy_device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"

    # Init vLLM
    print(f"Initialising vLLM on {vllm_device} ...")
    llm = init_vllm(model_path, vllm_device, seed)

    log = []

    # Baseline eval
    print("\nStep 0 (baseline)")
    load_policy_into_vllm(model, llm)
    val_acc = compute_val_accuracy(llm, val_items)
    ent     = compute_entropy(model, tokenizer, val_items, policy_device)
    log.append({"ei_step": 0, "val_accuracy": val_acc, "entropy": ent,
                 "n_correct": 0, "n_rollouts": 0, "sft_losses": []})
    print(f"  Entropy: {ent:.4f}")

    # EI steps
    for step in range(1, n_ei_steps + 1):
        print(f"\n{'='*60}")
        print(f"EI Step {step}/{n_ei_steps}  G={G}  db={db_size}  sft_epochs={sft_epochs}")
        print("="*60)

        # Sample batch of questions
        batch = random.sample(train_items, min(db_size, len(train_items)))

        # Sync latest weights into vLLM before rollout
        load_policy_into_vllm(model, llm)

        # Generate G rollouts per question
        prompts = [build_prompt(q["question"]) for q in batch]
        params  = SamplingParams(
            temperature=1.0, max_tokens=MAX_NEW_TOKENS,
            n=G, stop=[STOP_STR], seed=seed + step,
        )
        outputs = llm.generate(prompts, params)

        # Filter to correct pairs
        correct_pairs = []
        n_rollouts    = 0
        for q_item, out in zip(batch, outputs):
            gt     = extract_gsm8k_gt(q_item["answer"])
            prompt = build_prompt(q_item["question"])
            for gen in out.outputs:
                n_rollouts += 1
                resp = gen.text + STOP_STR   # restore stop string for grader
                if reward(resp, gt) == 1.0:
                    correct_pairs.append((prompt, resp))

        pct = len(correct_pairs) / n_rollouts * 100 if n_rollouts else 0
        print(f"  Rollouts: {n_rollouts} | Correct: {len(correct_pairs)} ({pct:.1f}%)")

        # SFT on correct pairs
        losses = sft_step(model, tokenizer, correct_pairs, lr, sft_epochs, policy_device)

        # Eval after SFT
        load_policy_into_vllm(model, llm)
        val_acc = compute_val_accuracy(llm, val_items)
        ent     = compute_entropy(model, tokenizer, val_items, policy_device)
        print(f"  Entropy: {ent:.4f}")

        log.append({
            "ei_step": step, "val_accuracy": val_acc, "entropy": ent,
            "n_correct": len(correct_pairs), "n_rollouts": n_rollouts,
            "sft_losses": losses,
        })

        # Save per-step checkpoint
        ckpt = os.path.join(output_dir, f"step_{step}")
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        print(f"  Checkpoint: {ckpt}")

    # Save final
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    log_path = os.path.join(output_dir, "ei_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nLog: {log_path}")

    # Plot
    _plot(log, output_dir)

    final_acc = log[-1]["val_accuracy"]
    print(f"\nFinal val accuracy: {final_acc:.2%}")
    if final_acc >= 0.15:
        print(">=15% target achieved.")
    else:
        print("Below 15% — try larger G, more sft_epochs, or more ei_steps.")

    return log


def _plot(log, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps   = [e["ei_step"]     for e in log]
        accs    = [e["val_accuracy"] for e in log]
        entropies = [e["entropy"]   for e in log]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(steps, accs, marker="o", linewidth=2)
        axes[0].axhline(0.15, color="red", linestyle="--", label="15% target")
        axes[0].set_xlabel("EI Step")
        axes[0].set_ylabel("Val Accuracy")
        axes[0].set_title("Val Accuracy over EI Steps")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(steps, entropies, marker="o", color="orange", linewidth=2)
        axes[1].set_xlabel("EI Step")
        axes[1].set_ylabel("Avg Token Entropy (nats)")
        axes[1].set_title("Response Entropy over EI Steps")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(output_dir, "ei_curves.png")
        plt.savefig(out, dpi=150)
        print(f"Plot: {out}")
    except Exception as e:
        print(f"[plot skipped: {e}]")


# CLI

def parse_args():
    p = argparse.ArgumentParser(description="Expert Iteration on GSM8K")
    p.add_argument("--model_path",    required=True,
                   help="SFT checkpoint or base model path")
    p.add_argument("--data_path",     default=None)
    p.add_argument("--output_dir",    default=None)
    p.add_argument("--n_ei_steps",    type=int,   default=5)
    p.add_argument("--G",             type=int,   default=8,
                   help="Rollouts per question")
    p.add_argument("--db_size",       type=int,   default=512,
                   help="Questions sampled per EI step {512,1024,2048}")
    p.add_argument("--sft_epochs",    type=int,   default=1,
                   help="SFT epochs per EI step")
    p.add_argument("--lr",            type=float, default=2e-5)
    p.add_argument("--policy_device", default="cuda:0")
    p.add_argument("--vllm_device",   default="cuda:1")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    data_path = args.data_path or os.path.abspath(
        os.path.join(SCRIPT_DIR, "../data/gsm8k/train.jsonl")
    )

    model_tag  = os.path.basename(args.model_path.rstrip("/"))
    run_name   = (f"{model_tag}_ei{args.n_ei_steps}"
                  f"_G{args.G}_db{args.db_size}_ep{args.sft_epochs}"
                  f"_lr{args.lr:.0e}")
    output_dir = args.output_dir or os.path.abspath(
        os.path.join(SCRIPT_DIR, f"../models/{run_name}")
    )

    print("=" * 65)
    print(f"Run          : {run_name}")
    print(f"Model        : {args.model_path}")
    print(f"n_ei_steps   : {args.n_ei_steps}")
    print(f"G (rollouts) : {args.G}")
    print(f"db_size      : {args.db_size}")
    print(f"sft_epochs   : {args.sft_epochs}")
    print(f"lr           : {args.lr}")
    print(f"Policy device: {args.policy_device}")
    print(f"vLLM device  : {args.vllm_device}")
    print(f"Output       : {output_dir}")
    print("=" * 65)

    run_expert_iteration(
        model_path    = args.model_path,
        data_path     = data_path,
        output_dir    = output_dir,
        n_ei_steps    = args.n_ei_steps,
        G             = args.G,
        db_size       = args.db_size,
        sft_epochs    = args.sft_epochs,
        lr            = args.lr,
        policy_device = args.policy_device,
        vllm_device   = args.vllm_device,
        seed          = args.seed,
    )


if __name__ == "__main__":
    main()
