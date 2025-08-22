from argparse import ArgumentParser
import math
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import json
import os
import re

from src.models import *
from src.data import *
from src.configs.config import load_config

from scipy.stats import spearmanr
from taskHessian.data import get_subset_dataloader
from taskHessian.datamodels import datamodels, get_subset_scores
from taskHessian.trak import TRAKer
from taskHessian.trak.gradient_computers import IterativeGradientComputer

def load_batch_fn(batch, device='cpu'):
    batch = batch[0].to(device)
    inputs = batch[:, :-1]
    targets = batch
    batch_size = batch.shape[0]
    return inputs, targets, batch_size

def update_config(config, args):
    config.sam = args.sam
    config.nsm = args.nsm
    config.reg = args.reg
    config.swa = args.swa
    config.mark = args.mark
    config.model_type = args.model_type

    # update data generation
    config.task.task_kwargs.p = args.p
    # config.task.task_kwargs.num_input_numbers = args.num_input_numbers
    # config.task.task_kwargs.num_total_samples = args.num_total_samples
    config.task.task_kwargs.train_ratio = args.train_ratio
    config.task.task_kwargs.valid_ratio = args.valid_ratio

    # training
    config.optimizer.device = args.device

    return config

if __name__ == "__main__":
    parser = ArgumentParser()
    #parser.add_argument("--config", type=str, required=True)
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
    # parser.add_argument("--num_input_numbers", type=int, default=2)
    # parser.add_argument("--num_total_samples", type=int, default=10000)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.2)

    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--test_batch_num", type=int, default=100)

    args = parser.parse_args()

    # Load configs
    config = load_config(args.task)

    config = update_config(config, args)

    print(config)
    device = torch.device(f"cuda:{int(config.optimizer.device)}") # get_device()
    #device = 'cpu'
    task = ModularArithmetic(config.task.task_kwargs)
    model_path = f'./checkpoints/phase{args.phase}/'
    if config.nsm:
        model_name = 'nso_model'
    else:
        model_name = 'model'
    name = f"{model_name}_phase_{args.phase}"
    save_file = f'./results/datamodels/{name}.json'
    data_path = './dataset/'
    train_dataset = torch.load(os.path.join(data_path, 'train_dataset.pt'), weights_only=False)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
    valid_dataset = torch.load(os.path.join(data_path, 'valid_dataset.pt'), weights_only=False)
    train_loaders, train_ids_list, test_loader = get_subset_dataloader(
        train_dataset, valid_dataset, 
        subset_ratio=0.6, 
        subset_num=50, 
        batch_size=512, 
        num_workers=8
    )

    def loss_fn(logits, sequence, reduction='mean'):
        return F.cross_entropy(logits[:, -1], sequence[:, -1], reduction=reduction)

    model = Decoder(
        dim=config.model.dim, num_layers=config.model.num_layers, num_heads=config.model.num_heads, num_tokens=task.p + 2, seq_len=task.seq_len, activation_name=config.model.activation, dropout_p=config.optimizer.dropout_p
    ).to(device)

    subset_scores = get_subset_scores(
        model=model,
        model_path=model_path,
        model_name=model_name,
        train_ids_list=train_ids_list,
        loss_fn=loss_fn,
        load_batch_fn=load_batch_fn,
        test_dataset=valid_dataset,
        test_batch_size=1,
        test_batch_num=args.test_batch_num,
        device=device
    )

    mm = np.load('./results/trak_results/scores/quickstart.mmap', mmap_mode='r')
    trak_scores = torch.from_numpy(mm)

    results_list = []
    test_batch_num = 2
    for j in range(args.test_batch_num):
        scores_list = []
        trak_group_scores_list = []
        for i in range(len(train_ids_list)):
            trak_group_score = trak_scores[torch.tensor(train_ids_list[i], dtype=torch.long), :]
            trak_group_score = trak_group_score[:, torch.tensor(j, dtype=torch.long)]
            trak_group_scores_list.append(trak_group_score.mean())

            scores_list.append(torch.tensor(subset_scores[i, j]))
        scores = torch.stack(scores_list, dim=0)
        trak_group_scores = torch.stack(trak_group_scores_list, dim=0)

        # phi_col = torch.linalg.lstsq(w, scores).solution
        # print(phi_col.shape)
        # print(phi_col)
        # residual = w @ phi_col - scores
        # print("||residual|| =", residual.norm().item())

        # Compute Spearman correlation
        spearman_corr, _ = spearmanr(scores.cpu().numpy(), trak_group_scores.cpu().numpy())
        print("Spearman correlation =", spearman_corr)
        results_list.append({
            'phi': trak_group_scores.tolist(),
            'spearman_corr': spearman_corr,
        })
    
    spearman_array = np.array([r['spearman_corr'] for r in results_list]) 
    print("Average spearman correlation:", spearman_array.mean())
