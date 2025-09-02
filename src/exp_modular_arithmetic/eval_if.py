from argparse import ArgumentParser
import math
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import os
import re

from src.models import *
from src.data import *
from src.configs.config import load_config
from src.utils.util import get_save_file_paths

from scipy.stats import spearmanr
from taskHessian.data import get_subset_dataloader
from taskHessian.datamodels import datamodels, get_subset_scores
from train_subset import prepare_subsets_by_group_sampling

def load_batch_fn(batch, device='cpu'):
    """Load batch for modular arithmetic task - following compute_if.py pattern"""
    # Following compute_if.py: batch is wrapped in tuple, extract batch[0]
    batch = batch[0].to(device)
    # For modular arithmetic, inputs are batch[:, :-1] and targets are batch[:, -1]
    inputs = batch[:, :-1]
    targets = batch[:, -1]
    return inputs, targets

def update_config(config, args):
    config.sam = args.sam
    config.nsm = args.nsm
    config.reg = args.reg
    config.swa = args.swa
    config.mark = args.mark
    config.model_type = args.model_type

    # update data generation
    config.task.task_kwargs.p = args.p
    config.task.task_kwargs.train_ratio = args.train_ratio
    config.task.task_kwargs.valid_ratio = args.valid_ratio

    # training
    config.optimizer.device = args.device

    return config

def load_influence_function_results(if_path, name):
    """Load influence function results from saved files"""
    # Load influence function scores
    if_scores_path = if_path
    if os.path.exists(if_scores_path):
        if_scores = np.load(if_scores_path)
        print(f"Loaded influence function scores with shape: {if_scores.shape}")
        return torch.from_numpy(if_scores)
    else:
        # Try to load from individual files if the combined file doesn't exist
        print(f"Influence scores file not found at {if_scores_path}")
        print("Trying to load from individual files...")
        
        # Look for influence function result files in the directory
        if_path_dir = os.path.dirname(if_path)
        if_files = []
        if os.path.exists(if_path_dir):
            for file in os.listdir(if_path_dir):
                if file.startswith('phase') and file.endswith('.npy') and 'influence_matrix' in file:
                    if_files.append(file)
        
        if not if_files:
            raise FileNotFoundError(f"No influence function files found in {if_path_dir}")
        
        # Load and combine influence function results
        if_scores_list = []
        for file in sorted(if_files):
            file_path = os.path.join(if_path_dir, file)
            scores = np.load(file_path)
            if_scores_list.append(scores)
        
        if_scores = np.concatenate(if_scores_list, axis=0)
        print(f"Combined influence function scores with shape: {if_scores.shape}")
        return torch.from_numpy(if_scores)

def evaluate_influence_function(if_scores, subset_scores, train_ids_list, test_batch_num):
    """Evaluate influence function performance by computing correlations with subset scores"""
    results_list = []
    if_scores = if_scores.T
    
    print(f"Evaluating influence function on {test_batch_num} test batches...")
    
    for j in tqdm(range(test_batch_num), desc="Evaluating test batches"):
        scores_list = []
        if_group_scores_list = []
        
        for i in range(len(train_ids_list)):
            # Get influence function scores for this training subset
            if_group_score = if_scores[torch.tensor(train_ids_list[i], dtype=torch.long), :]
            if_group_score = if_group_score[:, torch.tensor(j, dtype=torch.long)]
            if_group_scores_list.append(if_group_score.mean())
            
            # Get actual subset scores
            scores_list.append(torch.tensor(subset_scores[i, j]))
        
        scores = torch.stack(scores_list, dim=0)
        if_group_scores = torch.stack(if_group_scores_list, dim=0)
        
        # Compute Spearman correlation
        spearman_corr, p_value = spearmanr(scores.cpu().numpy(), if_group_scores.cpu().numpy())
        
        print(f"Batch {j}: Spearman correlation = {spearman_corr:.4f}, p-value = {p_value:.4f}")
        
        results_list.append({
            'batch_idx': j,
            'influence_scores': if_group_scores.tolist(),
            'actual_scores': scores.tolist(),
            'spearman_corr': spearman_corr,
            'p_value': p_value,
        })
    
    return results_list

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default='addition')
    parser.add_argument("--sam", action="store_true")
    parser.add_argument("--nsm", action="store_true")
    parser.add_argument("--reg", action="store_true")
    parser.add_argument("--swa", action="store_true")
    parser.add_argument("--mark", type=str, default='')
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--model_type", type=str, default='tf')

    # data generation
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--valid_ratio", type=float, default=0.2)

    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--test_batch_num", type=int, default=100)
    parser.add_argument("--subset_ratio", type=float, default=0.6)
    parser.add_argument("--subset_num", type=int, default=50)

    args = parser.parse_args()

    # Load configs
    config = load_config(args.task)
    config = update_config(config, args)

    print("Configuration:")
    print(config)
    
    device = torch.device(f"cuda:{int(config.optimizer.device)}")
    task = ModularArithmetic(config.task.task_kwargs)
    
    # Setup paths
    model_path = f'./checkpoints/{config.task.task_kwargs.task}/phase{args.phase}/'
    if config.nsm:
        model_name = 'nso_model'
    else:
        model_name = 'model'
    name = f"{model_name}_phase_{args.phase}"
    
    # Paths for results
    results_path = f'./results/influence_function/{config.task.task_kwargs.task}_p_{config.task.task_kwargs.p}_split_{config.task.task_kwargs.train_ratio}'
    if config.nsm:
        model_name = 'nso_model'
    else:
        model_name = 'model'
    if_path = os.path.join(results_path, f'phase{args.phase}_{model_name}_influence_matrix_{config.mark}.npy')
    save_file = f'./results/influence_function_eval/{name}.json'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    
    # Load datasets
    data_path = f'./dataset/{config.task.task_kwargs.task}_p_{config.task.task_kwargs.p}_split_{config.task.task_kwargs.train_ratio}'
    task = ModularArithmetic(config.task.task_kwargs)
    
    print(f"Using existing dataset at {data_path}")
    train_dataset = torch.load(os.path.join(data_path, 'train_dataset.pt'), weights_only=False)
    valid_dataset = torch.load(os.path.join(data_path, 'valid_dataset.pt'), weights_only=False)
    sub_valid_dataset = Subset(valid_dataset, list(range(100)))  # 只使用前100个验证样本以节省计算时间
    test_loader = DataLoader(sub_valid_dataset, batch_size=1, shuffle=False, num_workers=8)

    def loss_fn(logits, sequence, reduction='mean'):
        return F.cross_entropy(logits[:, -1], sequence, reduction=reduction)

    # Model
    if config.model_type == 'mlp':
        model = MLP_arithmetic(
            dim=128, num_layers=config.model.num_layers, num_tokens=task.p + 2, seq_len=task.seq_len, activation_name=config.model.activation, dropout_p=config.optimizer.dropout_p
        ).to(device)
    else:
        model = Decoder(
            dim=config.model.dim, num_layers=config.model.num_layers, num_heads=config.model.num_heads, num_tokens=task.p + 2, seq_len=task.seq_len, activation_name=config.model.activation, dropout_p=config.optimizer.dropout_p
        ).to(device)
    
    # Load pre-trained model
    checkpoint_path = os.path.join(model_path, f'all/{model_name}_all_1.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading pre-trained model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}")
        print("Please train the model first using train_all.py")
        exit(1)

    task_name = config.task.task_kwargs.task
    bar = 20
    subset_ratio = 0.9
    subset_num = 50
    datamodels_num = 40
    p = config.task.task_kwargs.p
    train_ratio = config.task.task_kwargs.train_ratio
    paths = get_save_file_paths(task_name, bar, subset_ratio, subset_num, p, train_ratio)
    model_path = paths['model_path']
    model_path = os.path.join(model_path, f'phase{args.phase}')
    dataset_path = paths['dataset_path']
    groups_file = paths['groups_file']
    chosen_groups_file = paths['chosen_groups_file']
    run_name = f'{paths['run_name']}_{name}'
    result_path = paths['result_path']
    save_file = os.path.join(result_path, f'{name}.json')
    print(f'model_path: {model_path}')
    print(f'dataset_path: {dataset_path}')
    print(f'groups_file: {groups_file}')
    print(f'chosen_groups_file: {chosen_groups_file}')

    subsets, train_ids_list, chosen_groups_list, groups = prepare_subsets_by_group_sampling(
        train_dataset,
        alpha=subset_ratio,
        m=subset_num,
        bar=bar,
        seed=42,
        groups_path=groups_file,
        chosen_path=chosen_groups_file,
    )

    train_loaders = [
        DataLoader(s, batch_size=config.train.batch_size, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
        for s in subsets
    ]
    # Load influence function results
    print("Loading influence function results...")
    print(f"Looking for influence function file at: {if_path}")
    try:
        if_scores = load_influence_function_results(if_path, name)
        print(f"Influence function scores shape: {if_scores.shape}")
    except Exception as e:
        print(f"Error loading influence function results: {e}")
        print(f"Expected path: {if_path}")
        print("Please run compute_if.py first to generate the influence function scores")
        exit(1)

    subset_scores = get_subset_scores(
        model=model,
        model_path=model_path,
        model_name=model_name,
        run_name=run_name,
        use_nso=config.nsm,
        train_ids_list=train_ids_list,
        loss_fn=loss_fn,
        load_batch_fn=load_batch_fn,
        test_dataset=valid_dataset,
        test_batch_size=1,
        test_batch_num=args.test_batch_num,
        device=device
    )
    print(f"Subset scores shape: {subset_scores.shape}")

    

    # Evaluate influence function
    print("Evaluating influence function performance...")
    results_list = evaluate_influence_function(
        if_scores, subset_scores, train_ids_list, args.test_batch_num
    )
    
    # Compute summary statistics
    spearman_array = np.array([r['spearman_corr'] for r in results_list])
    p_values_array = np.array([r['p_value'] for r in results_list])
    
    print("\n" + "="*50)
    print("INFLUENCE FUNCTION EVALUATION RESULTS")
    print("="*50)
    print(f"Average Spearman correlation: {spearman_array.mean():.4f}")
    print(f"Std Spearman correlation: {spearman_array.std():.4f}")
    print(f"Min Spearman correlation: {spearman_array.min():.4f}")
    print(f"Max Spearman correlation: {spearman_array.max():.4f}")
    print(f"Number of significant correlations (p < 0.05): {(p_values_array < 0.05).sum()}/{len(p_values_array)}")
    print("="*50)
    
    # Save results
    final_results = {
        'config': {
            'task': args.task,
            'sam': args.sam,
            'nsm': args.nsm,
            'reg': args.reg,
            'swa': args.swa,
            'mark': args.mark,
            'model_type': args.model_type,
            'p': args.p,
            'train_ratio': args.train_ratio,
            'valid_ratio': args.valid_ratio,
            'phase': args.phase,
            'subset_ratio': args.subset_ratio,
            'subset_num': args.subset_num,
            'test_batch_num': args.test_batch_num,
        },
        'summary_stats': {
            'mean_spearman': spearman_array.mean().item(),
            'std_spearman': spearman_array.std().item(),
            'min_spearman': spearman_array.min().item(),
            'max_spearman': spearman_array.max().item(),
            'num_significant': (p_values_array < 0.05).sum().item(),
            'total_batches': len(p_values_array),
        },
        'detailed_results': results_list
    }
    
    with open(save_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"Results saved to: {save_file}")
