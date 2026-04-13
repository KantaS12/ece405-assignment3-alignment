import os
import json
import time
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from drgrpo_grader import r1_zero_reward_fn


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    math_model_path = os.path.abspath(os.path.join(script_dir, "../models/Qwen_Qwen2.5-Math-1.5B"))
    test_jsonl_path = os.path.abspath(os.path.join(script_dir, "../data/gsm8k/test.jsonl"))
    r1_zero_prompt_path = os.path.abspath(os.path.join(script_dir, "prompts/r1_zero.prompt"))

    # Load r1_zero prompt template
    with open(r1_zero_prompt_path, "r") as f:
        r1_zero_template = f.read()

    # Load math validation examples from gsm8k test split
    math_examples = []
    with open(test_jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                question = data["question"]
                raw_answer = data["answer"]
                ground_truth = raw_answer.split("####")[-1].strip()
                math_examples.append((question, ground_truth))

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the math model and tokenizer
    math_model = AutoModelForCausalLM.from_pretrained(math_model_path, trust_remote_code=True).to(device)
    math_tokenizer = AutoTokenizer.from_pretrained(math_model_path, trust_remote_code=True)
    math_tokenizer.padding_side = "left"

    all_results = []
    generation_times = []

    for question, ground_truth in math_examples:
        # Format prompt using r1_zero template
        string_prompt = r1_zero_template.format(question=question)

        # Tokenize
        inputs = math_tokenizer(string_prompt, return_tensors="pt", padding=True).to(device)
        input_length = inputs.input_ids.shape[1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        # Generate
        prediction_tensors = math_model.generate(
            **inputs,
            max_new_tokens=2048,
            pad_token_id=math_tokenizer.eos_token_id,
            do_sample=False
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        generation_times.append(end_time - start_time)

        # Decode only new tokens
        generated_tokens = prediction_tensors[0][input_length:]
        model_output = math_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Prompt ends with "Assistant: <think>" so prepend to reconstruct full response
        full_response = "<think>" + model_output

        # Evaluate using r1_zero_reward_fn
        scores = r1_zero_reward_fn(full_response, ground_truth)

        all_results.append({
            "question": question,
            "ground_truth": ground_truth,
            "model_generation": model_output,
            "format_reward": scores["format_reward"],
            "answer_reward": scores["answer_reward"],
            "reward": scores["reward"],
        })

    # Compute aggregate metrics
    total = len(all_results)
    avg_reward = sum(r["reward"] for r in all_results) / total if total > 0 else 0.0
    avg_format = sum(r["format_reward"] for r in all_results) / total if total > 0 else 0.0
    avg_answer = sum(r["answer_reward"] for r in all_results) / total if total > 0 else 0.0
    avg_time = sum(generation_times) / len(generation_times) if generation_times else 0.0

    print(f"Total examples: {total}")
    print(f"Accuracy (answer_reward): {avg_answer:.4f}")
    print(f"Format compliance: {avg_format:.4f}")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Average generation time: {avg_time:.2f}s per example")

    # Serialize results
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, "Qwen_Math_1.5B_gsm8k_r1zero_results.json")
    with open(output_filename, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nResults saved to {output_filename}")


if __name__ == "__main__":
    main()
