import os
import sys
import json
import math
import re
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_script import SFTTrainer
from data_loading import SFTDataLoading
from implementation import iterate_batches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

VAL_SPLIT       = 0.1
WARMUP_RATIO    = 0.03
LOG_EVERY       = 10
SEQ_LENGTH      = 512
VAL_ACC_SAMPLES = 64
MAX_NEW_TOKENS  = 300

INFERENCE_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{question}\n\n### Response:"
)


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=512,
        )


def load_policy_into_vllm_instance(policy, llm):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def _get_question(item: dict) -> str:
    return item.get('problem') or item.get('question', '')


def _get_answer(item: dict) -> str:
    return item.get('answer') or item.get('solution', '')


def extract_answer(text: str) -> str | None:
    # GSM8K style
    matches = re.findall(r'####\s*([^\n]+)', text)
    if matches:
        return matches[-1].strip().replace(',', '')
    # MATH / boxed style
    import re as _re
    m = _re.search(r'\\boxed\{([^}]*)\}', text)
    if m:
        return m.group(1).strip()
    return None


def compute_val_accuracy(llm, val_items, n=VAL_ACC_SAMPLES):
    prompts = [INFERENCE_PROMPT.format(question=_get_question(item)) for item in val_items[:n]]
    params = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS)
    outputs = llm.generate(prompts, params)
    correct = sum(
        extract_answer(o.outputs[0].text) == extract_answer(_get_answer(item))
        for o, item in zip(outputs, val_items[:n])
    )
    acc = correct / len(prompts) if prompts else 0.0
    print(f"  Val accuracy: {correct}/{len(prompts)} = {acc:.2%}")
    return acc


class GSM8KSFTTrainer(SFTTrainer):
    def __init__(self, model_path, dataset_path, output_model_path,
                 num_examples=None, lr=2e-5, batch_size=1, grad_accum=32, epochs=3,
                 device='cuda'):
        self.device            = device
        self.output_model_path = output_model_path
        self._batch_size       = batch_size
        self._grad_accum       = grad_accum
        self._epochs           = epochs
        self.train_log         = []

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device)
        self.model.config.use_cache = False  # required for gradient checkpointing
        self.model.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'

        full_dataset = SFTDataLoading(
            self.tokenizer, dataset_path, seq_length=SEQ_LENGTH,
            shuffle=True, num_examples=num_examples,
        )
        n_total = len(full_dataset)
        n_val   = max(1, int(n_total * VAL_SPLIT))
        n_train = n_total - n_val
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val]
        )
        print(f"Dataset: {n_train} train / {n_val} val sequences")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        steps_per_epoch = math.ceil(n_train / (batch_size * grad_accum))
        total_steps  = steps_per_epoch * epochs
        warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )
        print(f"LR={lr} | batch={batch_size}×{grad_accum}={batch_size*grad_accum} | "
              f"epochs={epochs} | steps={total_steps} | warmup={warmup_steps}")

        raw_all = []
        with open(dataset_path) as f:
            for line in f:
                if line.strip():
                    raw_all.append(json.loads(line))
        if num_examples is not None:
            raw_all = raw_all[:num_examples]
        n_raw_val = max(VAL_ACC_SAMPLES, int(len(raw_all) * VAL_SPLIT))
        self._val_items = raw_all[-n_raw_val:]
        self._num_examples = num_examples  # stored for log metadata

    def _eval(self):
        self.model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in iterate_batches(self.val_dataset, batch_size=self._batch_size, shuffle=False):
                total += self._compute_loss(batch).item()
                n += 1
        self.model.train()
        torch.cuda.empty_cache()
        return total / n if n else 0.0

    def train(self):
        self.model.train()
        global_step  = 0
        running_loss = 0.0
        self.optimizer.zero_grad()

        # Log metadata so runs with different num_examples are identifiable
        self.train_log.append({
            'meta': True,
            'num_examples': self._num_examples or 'full',
        })

        for epoch in range(self._epochs):
            print(f"\n── Epoch {epoch + 1}/{self._epochs} ──")
            microbatch_idx = 0

            for batch in iterate_batches(self.train_dataset, batch_size=self._batch_size, shuffle=True):
                loss = self._compute_loss(batch) / self._grad_accum
                loss.backward()
                running_loss += loss.item()
                microbatch_idx += 1

                if microbatch_idx % self._grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % LOG_EVERY == 0:
                        val_loss  = self._eval()
                        avg_train = running_loss / LOG_EVERY
                        lr_now    = self.scheduler.get_last_lr()[0]
                        print(f"[E{epoch+1} step={global_step:4d}] "
                              f"train_loss={avg_train:.4f} | val_loss={val_loss:.4f} | lr={lr_now:.2e}")
                        self.train_log.append({
                            'step': global_step, 'epoch': epoch + 1,
                            'train_loss': avg_train, 'val_loss': val_loss, 'lr': lr_now,
                        })
                        running_loss = 0.0

            if microbatch_idx % self._grad_accum != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

        print("\n── Final val loss ──")
        final_val_loss = self._eval()
        print(f"Final val loss: {final_val_loss:.4f}")
        self.train_log.append({'step': 'final', 'val_loss': final_val_loss})

        os.makedirs(self.output_model_path, exist_ok=True)
        self.model.save_pretrained(self.output_model_path)
        self.tokenizer.save_pretrained(self.output_model_path)
        log_path = os.path.join(self.output_model_path, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(self.train_log, f, indent=2)
        print(f"Model saved : {self.output_model_path}")
        print(f"Log saved   : {log_path}")


def parse_args():
    p = argparse.ArgumentParser(description="SFT on GSM8K")
    p.add_argument('--model_path', required=True)
    p.add_argument('--data_path', default=None)
    p.add_argument('--num_examples', default=None,
                   help="int or 'full' (default: full dataset)")
    p.add_argument('--lr',           type=float, default=2e-5)
    p.add_argument('--batch_size',   type=int,   default=1)
    p.add_argument('--grad_accum',   type=int,   default=32)
    p.add_argument('--epochs',       type=int,   default=3)
    p.add_argument('--output_dir',   default=None)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--policy_device', default='cuda:0',
                   help="Device for the training policy (default: cuda:0)")
    p.add_argument('--vllm_device',   default='cuda:1',
                   help="Device for the vLLM inference instance (default: cuda:1)")
    return p.parse_args()


def main():
    args = parse_args()

    data_path = args.data_path or os.path.abspath(
        os.path.join(SCRIPT_DIR, '../data/gsm8k/train.jsonl')
    )
    num_examples = None
    if args.num_examples and args.num_examples.lower() != 'full':
        num_examples = int(args.num_examples)

    model_tag  = os.path.basename(args.model_path.rstrip('/'))
    n_tag      = f"n{num_examples}" if num_examples else "full"
    run_name   = f"{model_tag}_math_{n_tag}_lr{args.lr:.0e}_bs{args.batch_size*args.grad_accum}_ep{args.epochs}"
    output_dir = args.output_dir or os.path.abspath(os.path.join(SCRIPT_DIR, f'../models/{run_name}'))

    print("=" * 65)
    print(f"Run          : {run_name}")
    print(f"Policy device: {args.policy_device}")
    print(f"vLLM device  : {args.vllm_device}")
    print(f"Model        : {args.model_path}")
    print(f"Examples     : {num_examples or 'full (7473)'}")
    print(f"Output       : {output_dir}")
    print("=" * 65)

    trainer = GSM8KSFTTrainer(
        model_path=args.model_path,
        dataset_path=data_path,
        output_model_path=output_dir,
        num_examples=num_examples,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        device=args.policy_device,
    )
    trainer.train()

    # Load trained model into vLLM (on separate GPU) for fast batch accuracy evaluation
    print("\n── Val accuracy (vLLM) ──")
    llm = init_vllm(output_dir, args.vllm_device, seed=args.seed)
    load_policy_into_vllm_instance(trainer.model, llm)
    val_acc = compute_val_accuracy(llm, trainer._val_items)

    trainer.train_log.append({'step': 'final_vllm', 'val_accuracy': val_acc})
    log_path = os.path.join(output_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(trainer.train_log, f, indent=2)

    print("\n" + "=" * 65)
    print(f"Final val accuracy: {val_acc:.2%}")
    if val_acc < 0.15:
        print("Below 15% target. Try: --lr 5e-5  --grad_accum 16  --epochs 3")
    else:
        print(">=15% target achieved.")
    print("=" * 65)


if __name__ == '__main__':
    main()
