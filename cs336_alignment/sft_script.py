import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from implementation import iterate_batches
from data_loading import SFTDataLoading
from torch.nn import functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
small_model_path = os.path.abspath(os.path.join(script_dir, "../models/Qwen_Qwen2.5-0.5B"))
output_model_path = os.path.abspath(os.path.join(script_dir, "../models/Qwen_Qwen2.5-0.5B-finetuned"))

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the small model and tokenizer
small_model = AutoModelForCausalLM.from_pretrained(small_model_path, trust_remote_code=True).to(device)
small_tokenizer = AutoTokenizer.from_pretrained(small_model_path, trust_remote_code=True)
small_tokenizer.padding_side = "left"

# Language modeling loss
dataset_path = os.path.abspath(os.path.join(script_dir, "../data/alpaca_eval/alpaca_eval.jsonl"))
dataset = SFTDataLoading(small_tokenizer, dataset_path, seq_length=32, shuffle=True)

train_batch = next(iter(iterate_batches(dataset, batch_size=8, shuffle=True)))

inputs = train_batch["input_ids"].to(device)
labels = train_batch["labels"].to(device)

logits = small_model(inputs).logits
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=small_tokenizer.eos_token_id)
print(f"Language modeling loss: {loss.item()}")


# Save Trained model
small_model.save_pretrained(save_directory=output_model_path)
small_tokenizer.save_pretrained(save_directory=output_model_path)


# Gradient Accumulation
