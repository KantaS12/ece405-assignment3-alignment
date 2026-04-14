import os
import json
import pandas as pd
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.abspath(os.path.join(script_dir, "../data/simple_safety_tests/simple_safety_tests.csv"))
small_model_path = os.path.abspath(os.path.join(script_dir, "../models/Qwen_Qwen2.5-0.5B"))

device = "cuda" if torch.cuda.is_available() else "cpu"

small_model = AutoModelForCausalLM.from_pretrained(small_model_path, trust_remote_code=True).to(device)
small_tokenizer = AutoTokenizer.from_pretrained(small_model_path, trust_remote_code=True)
small_tokenizer.padding_side = "left"

df = pd.read_csv(data_path)

generation_times = []
all_results = []

for _, row in df.iterrows():
    prompt_id = row["id"]
    harm_area = row["harm_area"]
    category = row["category"]
    prompt = row["prompts_final"]

    if not prompt:
        continue

    inputs = small_tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    input_length = inputs.input_ids.shape[1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    output_tensors = small_model.generate(**inputs, max_new_tokens=256, pad_token_id=small_tokenizer.eos_token_id)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    generation_times.append(end_time - start_time)

    generated_tokens = output_tensors[0][input_length:]
    model_output = small_tokenizer.decode(generated_tokens, skip_special_tokens=True)

    all_results.append({
        "id": prompt_id,
        "harm_area": harm_area,
        "category": category,
        "prompt": prompt,
        "model_output": model_output,
    })

total = len(all_results)
average_time = sum(generation_times) / len(generation_times) if generation_times else 0

print(f"Total examples: {total}")
print(f"Average Generation Time: {average_time:.2f} seconds per example")

output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)
output_filename = os.path.join(output_dir, "Qwen_0.5B_sst_results.json")

with open(output_filename, "w") as f:
    json.dump(all_results, f, indent=4)

print(f"Results saved to {output_filename}")
