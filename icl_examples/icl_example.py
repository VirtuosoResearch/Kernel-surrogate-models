from argparse import ArgumentParser
import math
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import re


from scipy.stats import spearmanr
from kernelsm.kernelsm import kernelsm_score

if __name__ == "__main__":
    parser = ArgumentParser()
    #parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, default='coin_flip')
    parser.add_argument("--solver", type=str, default='lstsq')

    args = parser.parse_args()

    subset_scores_loss_file = f'data/{args.task}/{args.task}_subset_scores_loss.json'
    train_ids_list_file = f'data/{args.task}/{args.task}_train_ids_list.json'

    subset_scores_loss = json.load(open(subset_scores_loss_file, 'r'))
    train_ids_list = json.load(open(train_ids_list_file, 'r'))
    subset_scores_loss = torch.tensor(subset_scores_loss)
    train_ids_list = torch.tensor(train_ids_list)
    
    runs = 5
    all_errors_list = []
    all_spearmans_list = []
    for run in range(runs):
        permutation = np.random.permutation(len(train_ids_list))
        train_ids_list = [train_ids_list[i] for i in permutation]
        subset_scores_loss = subset_scores_loss[permutation]
        results = kernelsm_score(subset_scores_loss, train_ids_list, test_batch_num=1000, num_train=80, solver=args.solver)
        all_errors_list.append(np.mean(results['error']))
        all_spearmans_list.append(np.mean(results['spearman_corr']))
    print(f"Average error: {np.mean(all_errors_list)}+-{np.std(all_errors_list)}")
    print(f"Average spearman correlation: {np.mean(all_spearmans_list)}+-{np.std(all_spearmans_list)}")
    
    
    