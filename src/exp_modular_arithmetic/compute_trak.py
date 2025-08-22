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
from taskHessian.datamodels import datamodels
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
    parser.add_argument("--test_batch_num", type=int, default=1000)

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

    pattern = re.compile(rf"^{re.escape(model_name)}_\w+\.pth$")
    model_file_list = []
    for file in os.listdir(model_path):
        if pattern.match(file):
            model_file = os.path.join(model_path, file)
            model_file_list.append(model_file)
    num_models = len(model_file_list)
    ckpts = [torch.load(model_file, map_location='cpu') for model_file in model_file_list]

    traker = TRAKer(model=model,
                task='modular_arithmetic',
                save_dir=f'./results/trak_results/{name}/',
                gradient_computer=IterativeGradientComputer,
                train_set_size=len(train_dataset))

    for model_id, ckpt in enumerate(tqdm(ckpts)):
        # TRAKer loads the provided checkpoint and also associates
        # the provided (unique) model_id with the checkpoint.
        traker.load_checkpoint(ckpt, model_id=model_id)

        for batch in train_loader:
            batch = [x.to(device) for x in batch]
            # TRAKer computes features corresponding to the batch of examples,
            # using the checkpoint loaded above.
            traker.featurize(batch=batch, num_samples=batch[0].shape[0])

    # Tells TRAKer that we've given it all the information, at which point
    # TRAKer does some post-processing to get ready for the next step
    # (scoring target examples).
    traker.finalize_features()

    test_loader = DataLoader(valid_dataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=torch.cuda.is_available())

    for model_id, ckpt in enumerate(tqdm(ckpts)):
        traker.start_scoring_checkpoint(exp_name='quickstart',
                                        checkpoint=ckpt,
                                        model_id=model_id,
                                        num_targets=len(valid_dataset))
        for batch in test_loader:
            batch = [x.cuda() for x in batch]
            traker.score(batch=batch, num_samples=batch[0].shape[0])

    scores = traker.finalize_scores(exp_name='quickstart')
    print(scores.shape)



