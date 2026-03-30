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
from implementation import gsm8k_baseline

# Load GSM8k examples
script_dir = os.path.dirname(os.path.abspath(__file__))
test_jsonl_path = os.path.abspath(os.path.join(script_dir, "../data/gsm8k/test.jsonl"))
train_jsonl_path = os.path.abspath(os.path.join(script_dir, "../data/gsm8k/train.jsonl"))

small_model_path = os.path.abspath(os.path.join(script_dir, "../models/Qwen_Qwen2.5-0.5B"))
medium_model_path = os.path.abspath(os.path.join(script_dir, "../models/Qwen_Qwen2.5-3B-Instruct"))

# Load the small model and tokenizer
small_model = AutoModelForCausalLM.from_pretrained(small_model_path, trust_remote_code=True)
small_tokenizer = AutoTokenizer.from_pretrained(small_model_path, trust_remote_code=True)

# Load the medium model and tokenizer
medium_model = AutoModelForCausalLM.from_pretrained(medium_model_path, trust_remote_code=True)
medium_tokenizer = AutoTokenizer.from_pretrained(medium_model_path, trust_remote_code=True)
medium_tokenizer.padding_side = "left"

# Read jsonl files
gsm8k_examples = []
with open(train_jsonl_path, "r") as f:
    for line in f:
        train_data = json.loads(line)
        question = train_data["question"]
        raw_answer = train_data["answer"]

        ground_truth_number = raw_answer.split("####")[-1].strip()
        string_prompt = f"Question: {question}\nAnswer:"
        gsm8k_examples.append((string_prompt, ground_truth_number))


# String matching evaluation
tp_count = 0
fp_count = 0
failed_parse_count = 0
generation_times = []
all_results = []
incorrect_examples = []

for string_prompt, ground in gsm8k_examples:
    
    if not string_prompt:
        continue

    # Tokenize
    prediction = medium_tokenizer(string_prompt, return_tensors="pt", padding=True)
    prediction_length = prediction.input_ids.shape[1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Generate
    prediction_tensors = medium_model.generate(**prediction, max_new_tokens=256, pad_token_id=medium_tokenizer.eos_token_id)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    time_taken = end_time - start_time
    generation_times.append(time_taken)

    # Decode
    generated_tokens = prediction_tensors[0][prediction_length:]
    model_output = medium_tokenizer.decode(generated_tokens, skip_special_tokens=True)

    upper_model_output = model_output.upper()

    prediction = gsm8k_baseline(None, upper_model_output)

    is_correct = prediction == ground

    if is_correct:
        tp_count += 1
    else:
        fp_count += 1
        if prediction is None:
            failed_parse_count += 1

        incorrect_examples.append({
            "prompt": string_prompt,
            "ground_truth": ground,
            "parsed_prediction": prediction,
            "raw_output": model_output
        })

    all_results.append({
        "prompt": string_prompt,
        "ground_truth": ground,
        "raw_model_generation": model_output,
        "parsed_prediction": prediction,
        "is_correct": is_correct
    })

# Aggregate results
total_questions = tp_count + fp_count

# Accuracy
accuracy_score = tp_count / total_questions if total_questions > 0 else 0.0

# Calcuate average generation time
average_time = sum(generation_times) / len(generation_times) if generation_times else 0

print(f"True Positives: {tp_count}")
print(f"False Positives: {fp_count}")
print(f"Failed to Parse: {failed_parse_count}")
print(f"Total Questions: {total_questions}")
print(f"Accuracy: {accuracy_score:.4f}")
print(f"Average Generation Time: {average_time:.2f} seconds per question")

output_filename = "Qwen_3B_gsm8k_results.json"
with open(output_filename, "w") as f:
    json.dump(all_results, f, indent=4)

num_to_sample = min(10, len(incorrect_examples))
if num_to_sample > 0:
    sampled_errors = random.sample(incorrect_examples, num_to_sample)
    for i, error in enumerate(sampled_errors):
        print(f"\n[Error {i+1}]")
        print(f"Ground Truth: {error['ground_truth']} | Parsed As: {error['parsed_prediction']}")
        print(f"Raw Output:\n{error['raw_output']}")
        print("-" * 40)