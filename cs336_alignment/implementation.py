from typing import Any
import re

def mmlu_baseline(mmlu_example: dict[str, Any], model_output: str) -> str | None:
    # Parse the model output to extract the predicted answer choice
    
    # Check if the model output contains the expected format
    match = re.search(r'\b([A-D])\b', model_output)                                          
    try:
      if match:   
          return match.group(1)                                                            
      return None
                        
    except Exception as e:
        print(f"Error parsing model output: {e}")

    return None

def gsm8k_baseline(model_output: str) -> str | None:
    # Parse the model output to extract the predicted answer
    
    # Strip commas first
    clean_output = model_output.replace(',', '')

    # Find all numbers in the output
    numbers = re.findall(r'[-+]?\d*\.?\d+', clean_output)
    try:
      if numbers:
          return numbers[-1]
      return None
      
    except Exception as e:
        print(f"Error parsing model output: {e}")

    return None

# def alpaca_eval_baseline

