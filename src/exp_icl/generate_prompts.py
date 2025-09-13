import argparse
import json
import random
from typing import Tuple, List, Dict
import os

from tqdm import tqdm


def _read_jsonl(path: str) -> List[Dict]:
    data: List[Dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def load_local_dataset(task: str, data_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Load dev and test splits from local JSONL files located at:
    {data_dir}/{task}/{task}_dev.jsonl and {data_dir}/{task}/{task}_test.jsonl
    """
    task_dir = os.path.join(data_dir, task)
    dev_path = os.path.join(task_dir, f"{task}_dev.jsonl")
    test_path = os.path.join(task_dir, f"{task}_test.jsonl")
    if not os.path.exists(dev_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Expected files not found for task '{task}'. Looked for: {dev_path}, {test_path}"
        )
    print(f"Loading local data from {task_dir}")
    train_set = _read_jsonl(dev_path)
    test_set = _read_jsonl(test_path)
    return train_set, test_set


def construct_prompts(train_set: List[Dict], test_set: List[Dict], k: int, num_prompts: int) -> list:
    """
    Construct a fixed number of k-shot prompts by concatenating k
    train examples as input-output pairs. Includes the gold label and options when available.
    """
    print(f"Constructing {num_prompts} prompts with {k}-shot examples...")
    prompts_data = []
    train_indices = list(range(len(train_set)))

    for prompt_idx in tqdm(range(num_prompts), desc="Generating Prompts"):
        sampled_indices = random.sample(train_indices, k)

        prompt_lines: List[str] = []
        for train_idx in sampled_indices:
            train_example = train_set[train_idx]
            # Expect keys: 'input', 'output'
            prompt_lines.append(f"{train_example['input']}{train_example['output']}")

        prompt_text = "\n".join(prompt_lines)

        sample_record: Dict = {
            "prompt_index": prompt_idx,
            "prompt": prompt_text,
            "sampled_train_indices": sampled_indices,
        }

        prompts_data.append(sample_record)

    return prompts_data


def main():
    parser = argparse.ArgumentParser(
        description="Construct k-shot prompts from a dataset and save to JSON"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="coin_flip",
        help="Task name matching a folder in data_dir (e.g., coin_flip, sst2)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Base directory containing task subfolders with JSONL files",
    )
    parser.add_argument(
        "--k", type=int, default=4, help="Number of examples for k-shot prompting"
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=200,
        help="Number of prompts to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling k-shot examples",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    train_set, test_set = load_local_dataset(args.task, args.data_dir)
    train_set = train_set[:50]
    prompts = construct_prompts(train_set, test_set, args.k, args.num_prompts)

    output_file = os.path.join(args.data_dir, args.task, f"{args.task}_prompts.json")
    print(f"Saving prompts to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(prompts, f, indent=4)
    print("Prompts saved successfully.")


if __name__ == "__main__":
    main()


