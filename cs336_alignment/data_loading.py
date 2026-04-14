import torch
import json
import random

class SFTDataLoading:
    def __init__(self, tokenizer, dataset_path, seq_length, shuffle, num_examples=None):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.packed_data = []

        # Load the raw JSONL
        raw_data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    raw_data.append(json.loads(line))

        # Shuffle the dataset if required
        if shuffle:
            random.shuffle(raw_data)

        # Optionally limit to a fixed number of examples
        if num_examples is not None:
            raw_data = raw_data[:num_examples]

        # Tokenize and flatten into 1D list of tokens
        all_tokens = []
        for item in raw_data:
            # Support Alpaca-style keys and GSM8K-style keys
            prompt = item.get('prompt') or item.get('instruction') or item.get('question') or item.get('problem', '')
            response = item.get('response') or item.get('output') or item.get('answer') or item.get('solution', '')
            text_string = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
            )
            
            tokens = self.tokenizer(text_string, add_special_tokens=True)["input_ids"]
            tokens.append(self.tokenizer.eos_token_id)
            all_tokens.extend(tokens)

        # Pack the tokens into chunks
        for i in range(0, len(all_tokens) - self.seq_length, self.seq_length):
            chunk = all_tokens[i : i + self.seq_length + 1]
            
            if len(chunk) == self.seq_length + 1:
                self.packed_data.append({
                    "input_ids": chunk[:-1],
                    "labels": chunk[1:]
                })

    def __len__(self):
        return len(self.packed_data)

    def __getitem__(self, i):
        # Return the pre-packed sequence as tensors
        item = self.packed_data[i]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long)
        }