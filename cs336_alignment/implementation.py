"""
This is from Spring 2024
"""
from typing import Any
import re
import random
import torch
from cs336_alignment.data_loading import SFTDataLoading

def mmlu_baseline(mmlu_example: dict[str, Any], model_output: str) -> str | None:
    match = re.search(r'\b([A-D])\b', model_output)
    try:
        if match:
            return match.group(1)
        return None
    except Exception as e:
        print(f"Error parsing model output: {e}")
    return None

def gsm8k_baseline(model_output: str) -> str | None:
    clean_output = model_output.replace(',', '')
    numbers = re.findall(r'[-+]?\d*\.?\d+', clean_output)
    try:
        if numbers:
            return numbers[-1]
        return None
    except Exception as e:
        print(f"Error parsing model output: {e}")
    return None


def iterate_batches(dataset: SFTDataLoading, batch_size: int, shuffle: bool):
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)

    batches = []
    for start in range(0, len(dataset), batch_size):
        batch = [dataset[i] for i in indices[start:start + batch_size]]
        batches.append({key: torch.stack([s[key] for s in batch]) for key in batch[0]})
    return batches


