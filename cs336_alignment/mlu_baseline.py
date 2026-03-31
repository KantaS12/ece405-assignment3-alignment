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
from implementation import mmlu_baseline

# Load MMLU examples
script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.abspath(os.path.join(script_dir, "../data/mmlu/dev"))
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

mmlu_examples = []
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        data = pd.read_csv(os.path.join(folder_path, filename))
        # Format example as string prompts

        for _, row in data.iterrows():
            question = row.iloc[0]
            options = [row.iloc[1], row.iloc[2], row.iloc[3], row.iloc[4]]
            answer = row.iloc[5]

            string_prompt = f"Question: {question}\nOptions:\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\nAnswer:"        
            
            mmlu_examples.append((string_prompt, answer)) 
        

# String matching evaluation
tp_count = 0
fp_count = 0
failed_parse_count = 0

generation_times = []
all_results = []
incorrect_examples = []

for string_prompt, ground in mmlu_examples:

    if not string_prompt:
        continue

    # Tokenize
    prediction = small_tokenizer(string_prompt, return_tensors="pt", padding=True).to(device)
    prediction_length = prediction.input_ids.shape[1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Generate
    prediction_tensors = small_model.generate(**prediction, max_new_tokens=256, pad_token_id=small_tokenizer.eos_token_id)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    time_taken = end_time - start_time
    generation_times.append(time_taken)

    # Decode
    generated_tokens = prediction_tensors[0][prediction_length:]
    model_output = small_tokenizer.decode(generated_tokens, skip_special_tokens=True)

    upper_model_output = model_output.upper()

    prediction = mmlu_baseline(None, upper_model_output)

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

output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)

output_filename = os.path.join(output_dir, "Qwen_0.5B_mlu_results.json")

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