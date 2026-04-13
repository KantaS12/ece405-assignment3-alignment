import os
import json
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from implementation import iterate_batches
from data_loading import SFTDataLoading
from torch.nn import functional as F

LEARNING_RATE        = 2e-5
EPOCHS               = 1
BATCH_SIZE           = 1      # sequences per forward pass
GRADIENT_ACCUM_STEPS = 32     # effective batch size = 1 * 32 = 32 sequences
SEQ_LENGTH           = 512
LOG_EVERY            = 10     # log every N optimizer steps
VAL_SPLIT            = 0.1
WARMUP_RATIO         = 0.03   # fraction of total steps used for linear warmup


class SFTTrainer:
    def __init__(self, model_path, dataset_path, output_model_path, device="cuda"):
        self.device = device
        self.output_model_path = output_model_path

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"

        full_dataset = SFTDataLoading(self.tokenizer, dataset_path, seq_length=SEQ_LENGTH, shuffle=True)

        # Train / val split
        n_total = len(full_dataset)
        n_val = max(1, int(n_total * VAL_SPLIT))
        n_train = n_total - n_val
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(full_dataset, [n_train, n_val])
        print(f"Dataset: {n_train} train / {n_val} val sequences")

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)

        # Cosine LR schedule with linear warmup
        steps_per_epoch = math.ceil(n_train / (BATCH_SIZE * GRADIENT_ACCUM_STEPS))
        total_steps = steps_per_epoch * EPOCHS
        warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        print(f"Total optimizer steps: {total_steps} | Warmup steps: {warmup_steps}")

        self.train_log = []  # (step, train_loss, val_loss, lr)

    def _compute_loss(self, batch):
        inputs = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        logits = self.model(inputs).logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

    def _eval(self):
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        with torch.no_grad():
            for batch in iterate_batches(self.val_dataset, batch_size=BATCH_SIZE, shuffle=False):
                total_loss += self._compute_loss(batch).item()
                n_batches += 1
        self.model.train()
        return total_loss / n_batches if n_batches else 0.0

    def train(self):
        self.model.train()
        global_step = 0
        running_loss = 0.0
        self.optimizer.zero_grad()

        for epoch in range(EPOCHS):
            batches = iterate_batches(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True)

            for idx, batch in enumerate(batches):
                loss = self._compute_loss(batch) / GRADIENT_ACCUM_STEPS
                loss.backward()
                running_loss += loss.item()

                if (idx + 1) % GRADIENT_ACCUM_STEPS == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    if global_step % LOG_EVERY == 0:
                        val_loss = self._eval()
                        avg_train = running_loss / LOG_EVERY
                        lr_now = self.scheduler.get_last_lr()[0]
                        print(f"Epoch {epoch+1}/{EPOCHS} | Step {global_step} | "
                              f"Train loss: {avg_train:.4f} | Val loss: {val_loss:.4f} | "
                              f"LR: {lr_now:.2e}")
                        self.train_log.append({
                            "step": global_step,
                            "train_loss": avg_train,
                            "val_loss": val_loss,
                            "lr": lr_now,
                        })
                        running_loss = 0.0

            # Flush leftover gradients at epoch end
            if (len(batches) % GRADIENT_ACCUM_STEPS) != 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

        # Final validation loss
        final_val_loss = self._eval()
        print(f"\nFinal val loss: {final_val_loss:.4f}")

        # Save model + tokenizer
        self.model.save_pretrained(self.output_model_path)
        self.tokenizer.save_pretrained(self.output_model_path)
        print(f"Model saved to {self.output_model_path}")

        # Save learning curve
        curve_path = os.path.join(self.output_model_path, "learning_curve.json")
        with open(curve_path, "w") as f:
            json.dump(self.train_log, f, indent=2)
        print(f"Learning curve saved to {curve_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path        = os.path.abspath(os.path.join(script_dir, "../models/Qwen_Qwen2.5-0.5B"))
    dataset_path      = os.path.abspath(os.path.join(script_dir, "../data/alpaca_eval/alpaca_eval.jsonl"))
    output_model_path = os.path.abspath(os.path.join(script_dir, "../models/Qwen_Qwen2.5-0.5B-sft"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    trainer = SFTTrainer(model_path=model_path, dataset_path=dataset_path,
                         output_model_path=output_model_path, device=device)
    trainer.train()


if __name__ == "__main__":
    main()
