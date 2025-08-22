import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import yaml
import logging
import random

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

def get_device():
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{get_free_gpu()}")
    else:
        device = torch.device("cpu")
    return device

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.task not in configs:
        cfg_label = 'default'
    else:
        cfg_label = args.task
    
    print(f"load config of task: {cfg_label}")

    logging.info("Using best configs")
    configs = configs[cfg_label]

    for k, v in configs.items():
        if "lr" in k or "decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args

def get_device():
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{get_free_gpu()}")
    else:
        device = torch.device("cpu")
    return device

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_wandb_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    return configs["wandb"]

def compute_margin(logits, targets):
    """
    Computes the margin between the true class and the highest other class for each example.
    For correctly classified examples, computes the margin between the true class and the second-highest class.
    For incorrectly classified examples, computes the margin between the true class and the top predicted class.

    Args:
    - logits (torch.Tensor): Tensor of shape (batch_size, num_classes) containing logits.
    - targets (torch.Tensor): Tensor of shape (batch_size,) containing the true class indices.

    Returns:
    - margin_prob (torch.Tensor): Average margin, computed in probability space.
    - num_examples (int): Number of examples considered.
    """

    # Compute probabilities using softmax
    probabilities = F.softmax(logits, dim=-1)
    
    # Get the top logits and their corresponding indices
    top2_logits, top2_indices = torch.topk(logits, 2, dim=-1)

    # Extract the top logits
    top1_logits = top2_logits[:, 0]  # Logit for the highest class
    top2_logits = top2_logits[:, 1]  # Logit for the second-highest class

    # Compute the probabilities for the top 2 classes
    top1_probs = probabilities.gather(1, top2_indices[:, 0:1]).squeeze()  # Probability of the highest class
    top2_probs = probabilities.gather(1, top2_indices[:, 1:2]).squeeze()  # Probability of the second-highest class
    
    # Get the probability of the true class
    true_class_probs = probabilities.gather(1, targets.unsqueeze(1)).squeeze()

    # Mask to check if the true class is the highest logit
    true_class_is_highest = (logits.gather(1, targets.unsqueeze(1)) == top1_logits.unsqueeze(1)).squeeze()

    # Compute margins in probability space
    # For correctly classified examples, margin is (true_class - second_highest)
    # For incorrectly classified examples, margin is (true_class - highest_other)
    margin_prob = torch.where(true_class_is_highest, true_class_probs - top2_probs, top1_probs - true_class_probs)

    return margin_prob.mean().item(), len(margin_prob)


# Example usage
if __name__ == "__main__":
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 1.5, 0.2]], dtype=torch.float32)
    targets = torch.tensor([0, 2], dtype=torch.long)
    margin_prob, num_examples = compute_margin(logits, targets)
    print("Logits:", logits)
    print("Targets:", targets)
    print("Margin in Probability Space:", margin_prob)
    print("Number of Examples:", num_examples)
