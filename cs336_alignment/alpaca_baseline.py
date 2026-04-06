import os
import json
import pandas as pd
import string
import re
import time
import torch
import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Alpaca examples
script_dir = os.path.dirname(os.path.abspath(__file__))
alpaca_eval_jsonl_path = os.path.abspath(os.path.join(script_dir, "../data/alpaca_eval/alpaca_eval.jsonl"))


small_model_path = os.path.abspath(os.path.join(script_dir, "../models/Qwen_Qwen2.5-0.5B"))
medium_model_path = os.path.abspath(os.path.join(script_dir, "../models/Qwen_Qwen2.5-3B-Instruct"))

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the small model and tokenizer
small_model = AutoModelForCausalLM.from_pretrained(small_model_path, trust_remote_code=True).to(device)
small_tokenizer = AutoTokenizer.from_pretrained(small_model_path, trust_remote_code=True)
small_tokenizer.padding_side = "left"

# Load the medium model and tokenizer
medium_model = AutoModelForCausalLM.from_pretrained(medium_model_path, trust_remote_code=True).to(device)
medium_tokenizer = AutoTokenizer.from_pretrained(medium_model_path, trust_remote_code=True)
medium_tokenizer.padding_side = "left"

alpaca_examples = []
with open(alpaca_eval_jsonl_path, "r") as f:
    for line in f:
        if line.strip():
            alpaca_examples.append(json.loads(line))


all_results = []
generation_times = []


for example in alpaca_examples:
    
    instruction = example["instruction"]
    dataset_name = example.get("dataset", "alpaca_eval")

    # Format as a chat message for the Instruct model
    messages = [{"role": "user", "content": instruction}]
    string_prompt = small_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize
    inputs = small_tokenizer(string_prompt, return_tensors="pt", padding=True).to(device)
    input_length = inputs.input_ids.shape[1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Generate
    prediction_tensors = small_model.generate(
        **inputs, 
        max_new_tokens=2048, 
        pad_token_id=small_tokenizer.eos_token_id,
        do_sample=False # Standard for zero-shot baselines
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    time_taken = end_time - start_time
    generation_times.append(time_taken)

    # Decode
    generated_tokens = prediction_tensors[0][input_length:]
    model_output = small_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    all_results.append({
        "instruction": instruction,
        "output": model_output,
        "generator": "Qwen_2.5_0.5B",
        "dataset": dataset_name
    })

# Calculate average generation time
average_time = sum(generation_times) / len(generation_times) if generation_times else 0
print(f"\nFinished processing {len(all_results)} instructions.")
print(f"Average Generation Time: {average_time:.2f} seconds per prompt")

# Save outputs to your output directory
output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)

output_filename = os.path.join(output_dir, "Qwen_0.5B_alpaca_eval_results.json")

with open(output_filename, "w") as f:
    json.dump(all_results, f, indent=4)


"""
Winrate: 24.41% (Beats the reference GPT-4 baseline about 1 in 4 times)
Length-Controlled wWin: 24.67%
Std Error: 1.77
n: 590
"""