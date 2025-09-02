from argparse import ArgumentParser
import math
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader, Dataset
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
from taskHessian.datamodels import datamodels, get_subset_scores, filter_insensitive_samples
from taskHessian.utils import get_training_hessian, get_validation_hessian
from taskHessian.plot import analyze_phi_long_tail
from train_subset import prepare_subsets_by_group_sampling
from phase_indices_utils import get_intersected_indices_across_phases

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
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--valid_ratio", type=float, default=0.2)

    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--test_batch_num", type=int, default=1000)
    parser.add_argument("--hessian", action="store_true")

    args = parser.parse_args()

    # Load configs
    config = load_config(args.task)

    config = update_config(config, args)

    print(config)
    device = torch.device(f"cuda:{int(config.optimizer.device)}") # get_device()
    task = ModularArithmetic(config.task.task_kwargs)
    model_path = f'./checkpoints/{config.task.task_kwargs.task}/phase{args.phase}/'
    if config.nsm:
        model_name = 'nso_model'
    elif config.sam:
        model_name = 'sam_model'
    else:
        model_name = 'model'
    name = f"{model_name}_phase_{args.phase}"

    def loss_fn(logits, sequence, reduction='mean'):
        return F.cross_entropy(logits[:, -1], sequence[:, -1], reduction=reduction)

    model = Decoder(
            dim=config.model.dim, num_layers=config.model.num_layers, num_heads=config.model.num_heads, num_tokens=task.p + 2, seq_len=task.seq_len, activation_name=config.model.activation, dropout_p=config.optimizer.dropout_p
        ).to(device)

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

    train_dataset = torch.load(os.path.join(dataset_path, 'train_dataset.pt'), weights_only=False)
    valid_dataset = torch.load(os.path.join(dataset_path, 'valid_dataset.pt'), weights_only=False)

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

    if args.hessian:
        train_hessian = get_training_hessian(
            model=model,
            use_nso=config.nsm,
            model_path=model_path,
            model_name=model_name,
            train_loaders=train_loaders,
            datamodels_num=datamodels_num,
            loss_fn=loss_fn,
            load_batch_fn=load_batch_fn,
            device=device
        )
        print("Training Hessian:", train_hessian)

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

    # compute data affinity matrix, by the average of the scores of the same sample
    # data_affinity_matrix = torch.zeros(len(train_ids_list), len(train_ids_list))
    # data_affinity_count = torch.zeros(len(train_ids_list), len(train_ids_list))
    # # for col in tqdm(range(subset_scores.shape[1])):   
    # for i in tqdm(range(len(train_ids_list))):
    #     for j in range(len(train_ids_list)):
    #         data_affinity_matrix[i, j] += subset_scores[i]
    # data_affinity_matrix = data_affinity_matrix / data_affinity_count
    # print("Data affinity matrix shape:", data_affinity_matrix.shape)
    # print("Data affinity matrix:", data_affinity_matrix)

    # Use utility function to compute and intersect indices across phases
    sensitive_indices, subset_scores = get_intersected_indices_across_phases(
        subset_scores=subset_scores,
        current_phase=args.phase,
        model_name=model_name,
        result_path=f'{result_path}_{model_name}',
        phases_to_check=[1, 2, 3],
        variance_threshold=1e-2,
        datamodels_num=datamodels_num,
        index_type="sensitive"
    )
    
    select_valid_dataset = Subset(valid_dataset, sensitive_indices)
    select_valid_loader = DataLoader(select_valid_dataset, batch_size=256, shuffle=False, num_workers=8)
    if args.hessian:
        valid_hessian, valid_loss = get_validation_hessian(
            model=model,
            use_nso=config.nsm,
            model_path=model_path,
            model_name=model_name,
            valid_loader=select_valid_loader,
            datamodels_num=datamodels_num,
            loss_fn=loss_fn,
            load_batch_fn=load_batch_fn,
            device=device
        )
        print("Validation Hessian:", valid_hessian)
        print("Validation Loss:", valid_loss)


    results = datamodels(
        subset_scores=subset_scores,
        model_path=model_path,
        model_name=model_name,
        train_ids_list=train_ids_list,
        test_batch_num=args.test_batch_num,
        num_train=datamodels_num
    )

    #phi_array = results['phi']

    # df_metrics, overall_metrics, topk_idx = analyze_phi_long_tail(
    #     phi_array,
    #     save_dir="phi_longtail_report",
    #     export_csv=True,
    #     make_per_array_plots=True,
    #     topk_examples=3,
    # )

    error_array = results['error']
    spearman_array = results['spearman_corr']
    print("Average error:", np.mean(error_array))
    print("Average spearman correlation:", np.mean(spearman_array))

    # Save results
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    with open(save_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {save_file}")


