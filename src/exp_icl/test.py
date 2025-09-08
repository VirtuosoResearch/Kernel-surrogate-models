import argparse
import json
import random
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import numpy as np

def load_model_and_tokenizer(model_name, device, quant=False):
    """
    Loads a language model and its tokenizer from Hugging Face.

    Args:
        model_name (str): The name of the model to load (e.g., 'gpt2').
        device (str): The device to load the model onto ('cuda' or 'cpu').
        quant (bool): Whether to load the model in 4-bit precision.

    Returns:
        tuple: A tuple containing the model and tokenizer.
    """
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
        # device_map="auto" is recommended for quantized models and handles device placement
        device_map="auto" if quant else None
    )

    if not quant:
        model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set padding token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_and_split_dataset(dataset):
    """
    Loads a dataset from Hugging Face and splits it into train and test sets.

    Args:
        dataset (str): The name of the dataset (e.g., 'coin_flip').

    Returns:
        tuple: A tuple containing the train_set and test_set.
    """
    print(f"Loading dataset: {dataset}...")
    dataset = load_dataset(dataset)
    # Example for coin_flip which only has a 'train' split
    if 'test' not in dataset:
        print("No 'test' split found. Splitting 'train' into train/test (90/10).")
        train_test_split = dataset['train'].train_test_split(test_size=0.1, seed=42)
        train_set = train_test_split['train']
        test_set = train_test_split['test']
    else:
        train_set = dataset['train']
        test_set = dataset['test']
    return train_set, test_set

def construct_prompts(train_set, test_set, k, output_file):
    """
    Constructs k-shot prompts and saves them to a JSON file.

    Args:
        train_set (Dataset): The training dataset.
        test_set (Dataset): The testing dataset.
        k (int): The number of few-shot examples (shots) to include in the prompt.
        output_file (str): Path to save the generated prompts.
    """
    print(f"Constructing {k}-shot prompts...")
    prompts_data = []
    train_indices = list(range(len(train_set)))

    for test_idx, test_sample in enumerate(tqdm(test_set, desc="Generating Prompts")):
        # Ensure we don't sample the same training example multiple times for one prompt
        sampled_indices = random.sample(train_indices, k)
        
        prompt_text = ""
        # Add k-shot examples from the training set
        for train_idx in sampled_indices:
            train_sample = train_set[train_idx]
            # This formatting is specific to 'coin_flip', adapt for other datasets
            prompt_text += f"Coin flip outcome: {train_sample['result']}\n"

        # Add the test sample query
        # We append the query part but not the answer, which the model should predict
        prompt_text += f"Coin flip outcome:"
        
        prompts_data.append({
            "test_sample_index": test_idx,
            "prompt": prompt_text,
            "correct_label": test_sample['result'], # The ground truth
            "sampled_train_indices": sampled_indices
        })

    print(f"Saving prompts to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(prompts_data, f, indent=4)
    print("Prompts saved successfully.")


def run_inference(model, tokenizer, prompts_file, device):
    """
    Runs inference using the generated prompts and calculates metrics.

    Args:
        model (PreTrainedModel): The loaded language model.
        tokenizer (PreTrainedTokenizer): The loaded tokenizer.
        prompts_file (str): The path to the JSON file with prompts.
        device (str): The device to run inference on.

    Returns:
        list: A list of dictionaries, each containing results for a sample.
    """
    print(f"Running inference with prompts from {prompts_file}...")
    with open(prompts_file, 'r') as f:
        prompts_data = json.load(f)

    results = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    for sample in tqdm(prompts_data, desc="Inference"):
        prompt = sample['prompt']
        correct_label = sample['correct_label']
        
        # Tokenize the prompt and the potential labels
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        # Note: 'Heads' and 'Tails' are specific to coin_flip
        # Add a space before for better tokenization with some models
        heads_token_id = tokenizer.encode(" Heads", add_special_tokens=False)[0]
        tails_token_id = tokenizer.encode(" Tails", add_special_tokens=False)[0]

        with torch.no_grad():
            outputs = model(**inputs)
            # We are interested in the logits for the very next token
            next_token_logits = outputs.logits[:, -1, :]

        # Calculate loss for the two possible outcomes
        # The shape for loss_fct needs to be (N, C) and (N)
        logits_for_outcomes = next_token_logits[:, [heads_token_id, tails_token_id]]
        
        # Determine the target index (0 for Heads, 1 for Tails)
        target_idx = 0 if correct_label == "Heads" else 1
        target = torch.tensor([target_idx], device=device)
        
        loss = loss_fct(logits_for_outcomes, target).item()

        # Calculate margin (difference in logits)
        log_probs = torch.log_softmax(next_token_logits, dim=-1)
        prob_correct = log_probs[:, heads_token_id if correct_label == "Heads" else tails_token_id].item()
        prob_incorrect = log_probs[:, tails_token_id if correct_label == "Heads" else heads_token_id].item()
        margin = prob_correct - prob_incorrect

        results.append({
            "test_sample_index": sample['test_sample_index'],
            "loss": loss,
            "margin": margin,
            "correct_label": correct_label
        })
        
    return results


def main():
    parser = argparse.ArgumentParser(description="In-Context Learning Inference Script")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", help="Model name from Hugging Face.")
    parser.add_argument("--dataset", type=str, default="coin_flip", help="Dataset name from Hugging Face.")
    parser.add_argument("--k", type=int, default=4, help="Number of examples for k-shot prompting.")
    parser.add_argument("--prompts_file", type=str, default="prompts.json", help="File to save/load prompts.")
    parser.add_argument("--results_file", type=str, default="results.json", help="File to save inference results.")
    parser.add_argument("--quant", action="store_true", help="Load the model in 4-bit precision (quantized).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Model and Tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name, device, args.quant)

    # 2. Load and split dataset
    train_set, test_set = load_and_split_dataset(args.dataset)

    # 3. Construct and save prompts
    construct_prompts(train_set, test_set, args.k, args.prompts_file)
    
    # 4. Run Inference
    inference_results = run_inference(model, tokenizer, args.prompts_file, device)

    # 5. Collate and display results
    avg_loss = np.mean([res['loss'] for res in inference_results])
    avg_margin = np.mean([res['margin'] for res in inference_results])
    
    print("\n--- Inference Complete ---")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average Margin (log_prob_correct - log_prob_incorrect): {avg_margin:.4f}")
    
    # Save detailed results
    with open(args.results_file, 'w') as f:
        json.dump(inference_results, f, indent=4)
    print(f"Detailed results saved to {args.results_file}")

if __name__ == "__main__":
    main()

