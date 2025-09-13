import argparse
import json
import os
import random
from typing import List, Dict, Tuple

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm


def load_model_and_tokenizer(model_name: str, device: str, quant: bool = False) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a language model and its tokenizer from Hugging Face."""
    print(f"Loading model: {model_name}...")

    quantization_config = None
    if quant:
        if not torch.cuda.is_available():
            raise ValueError("Quantization requires a CUDA-enabled GPU.")
        print("Loading model in 4-bit precision (quantized)...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto" if quant else None
    )

    if not quant:
        model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def load_prompts_and_test_data(task: str, data_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """Load prompts and test data for the given task."""
    task_dir = os.path.join(data_dir, task)
    prompts_file = os.path.join(task_dir, f"{task}_prompts.json")
    test_file = os.path.join(task_dir, f"{task}_test.jsonl")
    
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    print(f"Loading prompts from {prompts_file}")
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    
    print(f"Loading test data from {test_file}")
    test_data = []
    with open(test_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                test_data.append(json.loads(line))
    
    return prompts, test_data


def get_label_tokens(tokenizer: AutoTokenizer, options: List[str]) -> Dict[str, int]:
    """Get token IDs for the possible labels."""
    label_tokens = {}
    for option in options:
        # Add space before for better tokenization
        token_id = tokenizer.encode(f" {option}", add_special_tokens=False)[0]
        label_tokens[option] = token_id
    return label_tokens


def run_inference_on_query(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    query: Dict,
    device: str
) -> Tuple[float, float]:
    """
    Run inference on a single query with the given prompt.
    Returns (loss, margin) where margin = log_prob_correct - log_prob_incorrect.
    """
    # Construct input: prompt + query input
    full_input = prompt + "\n" + query["input"]
    
    # Tokenize input
    inputs = tokenizer(full_input, return_tensors="pt").to(device)
    
    # Get label tokens
    options = query["options"]
    label_tokens = get_label_tokens(tokenizer, options)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Get logits for the next token
        next_token_logits = outputs.logits[:, -1, :]
    
    # Get logits for the possible labels
    option_token_ids = [label_tokens[opt] for opt in options]
    logits_for_options = next_token_logits[:, option_token_ids]
    
    # Calculate loss
    correct_label = query["output"]
    correct_idx = options.index(correct_label)
    target = torch.tensor([correct_idx], device=device)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(logits_for_options, target).item()
    
    # Calculate margin (log_prob_correct - log_prob_incorrect)
    log_probs = torch.log_softmax(next_token_logits, dim=-1)
    correct_logit = log_probs[:, label_tokens[correct_label]].item()
    incorrect_logit = log_probs[:, label_tokens[options[1 - correct_idx]]].item()
    margin = correct_logit - incorrect_logit
    
    return loss, margin


def run_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[Dict],
    test_data: List[Dict],
    device: str,
    max_queries: int = None
) -> List[Dict]:
    """
    Run inference on all queries with all prompts.
    Returns results for each (prompt, query) pair.
    """
    print("Running inference...")
    results = []
    
    # Limit number of queries if specified
    if max_queries is not None:
        test_data = test_data[:max_queries]
    
    total_pairs = len(prompts) * len(test_data)
    pbar = tqdm(total=total_pairs, desc="Inference")
    
    for prompt_idx, prompt_data in enumerate(prompts):
        prompt_text = prompt_data["prompt"]
        
        for query_idx, query in enumerate(test_data):
            try:
                loss, margin = run_inference_on_query(
                    model, tokenizer, prompt_text, query, device
                )
                
                results.append({
                    "prompt_index": prompt_idx,
                    "query_index": query_idx,
                    "loss": loss,
                    "margin": margin,
                    "correct_label": query["output"],
                    "options": query["options"]
                })
                
            except Exception as e:
                print(f"Error processing prompt {prompt_idx}, query {query_idx}: {e}")
                results.append({
                    "prompt_index": prompt_idx,
                    "query_index": query_idx,
                    "loss": float('inf'),
                    "margin": float('-inf'),
                    "correct_label": query["output"],
                    "options": query["options"],
                    "error": str(e)
                })
            
            pbar.update(1)
    
    pbar.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference on prompts and test queries")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-7B",
        help="Model name from Hugging Face"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="coin_flip",
        help="Task name matching a folder in data_dir"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Base directory containing task subfolders"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file for results (default: {task}_inference_results.json)"
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Maximum number of test queries to process"
    )
    parser.add_argument(
        "--quant",
        action="store_true",
        help="Load the model in 4-bit precision (quantized)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name, device, args.quant)
    
    # Load prompts and test data
    prompts, test_data = load_prompts_and_test_data(args.task, args.data_dir)
    
    print(f"Loaded {len(prompts)} prompts and {len(test_data)} test queries")
    
    # Run inference
    results = run_inference(model, tokenizer, prompts, test_data, device, args.max_queries)
    
    # Save results
    if args.output_file is None:
        args.output_file = os.path.join(args.data_dir, args.task, f"{args.task}_inference_results.json")
    
    print(f"Saving results to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary statistics
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        avg_loss = np.mean([r["loss"] for r in valid_results])
        avg_margin = np.mean([r["margin"] for r in valid_results])
        print(f"\n--- Inference Complete ---")
        print(f"Valid results: {len(valid_results)}/{len(results)}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Margin: {avg_margin:.4f}")
    else:
        print("No valid results obtained.")


if __name__ == "__main__":
    main()
