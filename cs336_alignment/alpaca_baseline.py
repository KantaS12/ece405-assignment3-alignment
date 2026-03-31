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

