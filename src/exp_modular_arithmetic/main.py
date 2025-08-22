from argparse import ArgumentParser
import math
from tqdm import tqdm
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Pytorch's efficient attention doesn't allow Hessian computation by default
from torch.nn.attention import SDPBackend, sdpa_kernel

# Config
from src.configs.config import load_config
# Util
from src.utils.util import seed_everything, get_device
from src.utils.plot import plot
# Logger
from src.utils.logger import Logger
# Task
from src.data import *
# Model
from src.models import *
# Optimizer
from src.optimizers import get_optimizer
from src.optimizers.nsm import NSM
from src.optimizers.sam import SAM
# Hessian
from src.hessian.hessian import Hessian_Calculator
from src.hessian.plot import hessian_plot
# SWA
from src.utils.swa import *

from taskHessian.data import get_subset_dataloader


def remove_outliers(val_acc, threshold=0.1):
    val_acc = np.array(val_acc)
    diff = np.abs(np.diff(val_acc)) 
    outliers = diff > threshold 

    valid_indices = [0]  
    for i in range(1, len(val_acc)):
        if not outliers[i - 1]:  
            valid_indices.append(i)
    cleaned_val_acc = val_acc[valid_indices]
    return cleaned_val_acc

def detect_rise_with_window(val_acc, i, window_size=5, rise_threshold=0.01):
    i -= window_size
    val_acc = np.array(val_acc)

    avg_before = np.mean(val_acc[max(0, i - window_size):i])

    avg_after = np.mean(val_acc[i:])

    if avg_after - avg_before > rise_threshold:
        return i 
    
    return -1 

def eval(task, dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_margin = 0
    total_num_correct_margin = 0
    
    for batch in dataloader:
        
        # Per batch, both train and val
        batch = batch[0].to(device)
        #print(batch.shape)
        
        logits = model(batch[:, :-1])
        #print(logits.shape)
        # Loss
        loss = loss_fn(logits, batch)
        #print(batch[:, -1])
        total_loss += loss.item() * batch.shape[0]
        # Accuracy
        acc = task.get_eval_metrics()(logits, batch)["accuracy"]
        total_acc += acc * batch.shape[0]
    
    mean_loss = total_loss / valid_num
    mean_acc = total_acc / valid_num

    return mean_acc, mean_loss

def train(config, task, model, optimizer, loss_fn, train_dataloader, valid_dataloader, device, model_id=0, epochs=10, logger=None):
    pbar = tqdm(range(epochs))
    train_num = len(train_dataloader.dataset)
    valid_num = len(valid_dataloader.dataset)
    phase1_folder = './checkpoints/phase1'
    phase2_folder = './checkpoints/phase2'
    phase3_folder = './checkpoints/phase3'
    os.makedirs(phase1_folder, exist_ok=True)
    os.makedirs(phase2_folder, exist_ok=True)
    os.makedirs(phase3_folder, exist_ok=True)

    if config.nsm:
        model_reg = 'nso_'
    else:
        model_reg = ''

    phase1_path = os.path.join(phase1_folder, f'{model_reg}model_{model_id}.pth')
    phase2_path = os.path.join(phase2_folder, f'{model_reg}model_{model_id}.pth')
    phase3_path = os.path.join(phase3_folder, f'{model_reg}model_{model_id}.pth')

    # State vars
    i = 0 # Step
    grads = None # Gradfilter
    # Record whether train/val acc is above threshold
    train_converge = -1
    val_converge = -1
    grok_start = -1
    # For loss gap
    train_loss = None
    val_loss = None
    # Best accuarcy
    train_acc = 0
    val_acc = 0
    best_train_acc = 0
    best_val_acc = 0
    
    # Train loop
    init = True
    phase = 'init'
    swa_n = 0
    for e in pbar:
        do_log = (e+1) % config.logger.save_every == 0 or e<1
        """
        if train_converge > 0:
            do_log = (e+1) % 300 == 0 or e<1
        else:
            do_log = (e+1) % config.logger.save_every == 0 or e<1
        if weight_decay > 0:
            do_log = (e+1) % config.logger.save_every == 0 or e<1
        """
        data_loaders = [(train_dataloader, True), (valid_dataloader, False)]
        for data_loader_idx, (dataloader, is_train) in enumerate(data_loaders):

            model.train(is_train)

            # Per epoch stats
            total_loss = 0
            total_acc = 0
            total_margin = 0
            total_num_correct_margin = 0
            
            for batch in dataloader:
                
                # Per batch, both train and val
                batch = batch[0].to(device)                
                logits = model(batch[:, :-1])

                # Loss
                loss = loss_fn(logits, batch)
                total_loss += loss.item() * batch.shape[0]
                # Accuracy
                #acc = task.get_eval_metrics()(logits, batch)["accuracy"]
                #total_acc += acc * batch.shape[0]

                if is_train:
                    i += 1
                    #if set_zero_loss:
                    #    loss = torch.tensor(0.0, requires_grad=True, device=device).backward() 
                    #else:
                    #    loss.backward() 
                    loss.backward()
                    # Update model parameters
                    if config.nsm:
                        nsm_lam = 0
                        nsm_num_perturbs = 1
                        nsm_use_neg = False
                        # 1st forward-backward step
                        optimizer.store_gradients(
                            zero_grad=True,
                            store_weights=True,
                            update_weight=nsm_lam,
                        )
                        update_weight = (1-nsm_lam) / (2*nsm_num_perturbs) if nsm_use_neg else (1-nsm_lam) / nsm_num_perturbs
                        #update_weight = 1

                        # 2nd forward-backward step
                        for _ in range(nsm_num_perturbs):
                            optimizer.first_step(
                                zero_grad=True,
                                store_perturb=True,
                            )
                            logits = model(batch[:, :-1])
                            loss_fn(logits, batch).backward()
                            optimizer.store_gradients(
                                zero_grad=True,
                                store_weights=False,
                                update_weight=update_weight,
                            )
                            if nsm_use_neg:
                                optimizer.first_step(
                                    zero_grad=True,
                                    store_perturb=False,
                                )
                                logits = model(batch[:, :-1])
                                loss_fn(logits, batch).backward()
                                optimizer.store_gradients(
                                    zero_grad=True,
                                    store_weights=False,
                                    update_weight=update_weight,
                                )
                            optimizer.second_step(zero_grad=True)

                    elif config.sam:
                        optimizer.first_step(zero_grad=True)
                        logits = model(batch[:, :-1])
                        loss_fn(logits, batch).backward()
                        optimizer.second_step(zero_grad=True)

                    else:
                        
                        optimizer.step()
                        optimizer.zero_grad()
                
                if is_train and (config.nsm or config.sam):
                    # recompute the logits for evaluations
                    logits = model(batch[:, :-1])
                acc = task.get_eval_metrics()(logits.detach(), batch)["accuracy"]
                total_acc += acc * batch.shape[0]

            if is_train:
                train_loss = total_loss / train_num
                train_acc = total_acc / train_num
                if train_converge < 0 and train_acc >= config.train.converge_threshold:
                #if train_converge < 0 and train_loss <= config.train.loss_converge_threshold:
                    train_converge = i
                    torch.save(model.state_dict(), phase1_path)
                    logger.log_value("save_spectrum", 2)
                    #model.save_init_state()
                best_train_acc = max(best_train_acc, train_acc)
                logger.log("train_loss", train_loss, i)
                logger.log("train_acc", train_acc, i)
                logger.log("train_margin", total_margin / train_num, i)
                logger.log("train_num_correct_margin", total_num_correct_margin, i)
                logger.log_value("train_converge", train_converge)
            elif data_loader_idx==1:
                val_loss = total_loss / valid_num
                val_acc = total_acc / valid_num
                if val_converge < 0 and val_acc >= config.train.converge_threshold:
                    val_converge = i
                    torch.save(model.state_dict(), phase3_path)

                if train_converge > 0:
                    cleaned_val_acc = remove_outliers(logger["val_acc"]['value'])
                    #print(cleaned_val_acc)
                    #if val_converge < 0 and grok_start < 0 and np.mean(logger["val_acc"]['value'][-10:]) - logger["val_acc"]['value'][-10] > 0.03 and abs(np.mean(logger["val_acc"]['value'][-20:-10]) - logger["val_acc"]['value'][-10]) < 0.01:
                    #if val_converge < 0 and grok_start < 0:
                    #    grok_start = detect_rise_with_window(cleaned_val_acc, i)
                    if val_acc > 0.6 and grok_start<0:
                        torch.save(model.state_dict(), phase2_path)
                        grok_start = i

                best_val_acc = max(best_val_acc, val_acc)
                logger.log("val_loss", val_loss, i)
                logger.log("val_acc", val_acc, i)
                logger.log("val_margin", total_margin / valid_num, i)
                logger.log("val_num_correct_margin", total_num_correct_margin, i)
                logger.log_value("val_converge", val_converge)
                logger.log_value("grok_start", grok_start)
        
        if do_log:
            
            logger.log("loss_gap", val_loss - train_loss, i)
            logger.log("acc_gap", train_acc - val_acc, i)
            if config.train.hessian_log_every > 0:

                hessian_calculator = Hessian_Calculator(model=model, loss_fn=loss_fn, p=task.p, dataloader=train_dataloader, valid_dataloader=valid_dataloader, device=device)
                
                #hessian_calculator.compute_spectrum(train_num=train_num, n_iter=100, n_v=1, method=3)
                #hessian_calculator.collect(train_num)
                #hessian_calculator.log(logger, i)
                #hessian_plot(logger, config)
                #pac_bound_result, pac_part, hessian_part, real_result  = hessian_calculator.pac_bound(logger=logger, log_i=i, train_num=train_num, valid_num=valid_num, n_iter=100, n_v=1, delta=0.1)
                #logger.log("train_hessianmeasurement", pac_bound_result, i)

                #hessian_calculator.compare_bound(logger=logger, log_i=i, train_num=train_num, valid_num=valid_num, train_loss=train_loss, n_iter=100, n_v=1)
                #hessian_calculator.noisy_loss(logger=logger, log_i=i, train_loss=train_loss, val_loss=val_loss, train_num=train_num, valid_num=valid_num)
                #hessian_calculator.compute_compression_bound(logger=logger, log_i=i, train_num=train_num, valid_num=valid_num, train_loss=train_loss)

            
            plot(logger, config)
            logger.save()
            if val_converge > 0:
                break
            
        pbar.set_description(f"Train acc: {train_acc:.3f} / Val acc: {val_acc:.3f} / Best train acc: {best_train_acc:.3f} / Best val acc: {best_val_acc:.3f}")

def run_experiment(config):
    
    # Seed
    seed_everything(config.train.seed)

    # Device
    device = torch.device(f"cuda:{int(config.optimizer.device)}") # get_device()

    # Task
    #task = get_task(config.task)
    data_path = './dataset/'
    task = ModularArithmetic(config.task.task_kwargs)
    if os.path.exists(data_path):
        print(f"Using existing dataset at {data_path}")
        train_dataset = torch.load(os.path.join(data_path, 'train_dataset.pt'), weights_only=False)
        valid_dataset = torch.load(os.path.join(data_path, 'valid_dataset.pt'), weights_only=False)
    else:
        
        train_dataset = task.get_train_dataset()
        valid_dataset = task.get_valid_dataset()
        os.mkdir(data_path)
        torch.save(train_dataset, os.path.join(data_path, 'train_dataset.pt'))
        torch.save(valid_dataset, os.path.join(data_path, 'valid_dataset.pt'))
    
    train_loaders, train_ids_list, test_loader = get_subset_dataloader(
        train_dataset, valid_dataset, 
        subset_ratio=0.6, 
        subset_num=50, 
        batch_size=config.train.batch_size, 
        num_workers=8
    )

    # Logger
    data_str = "_numbers_" + str(config.task.task_kwargs.num_input_numbers) + \
            "_samples_" + str(config.task.task_kwargs.num_total_samples) + \
            "_train_" + str(config.task.task_kwargs.train_ratio) + \
            "_valid_" + str(config.task.task_kwargs.valid_ratio)

    if config.task.task_kwargs.p != 97:
        p_suffix = f'_p{config.task.task_kwargs.p}'
    else:
        p_suffix = ''

    optim_suffix = ''
    if config.nsm:
        optim_suffix = optim_suffix + f'_nsm'
    elif config.sam:
        optim_suffix = optim_suffix + '_sam'
    else:
        optim_suffix = optim_suffix + f'_{config.optimizer.base_optimizer}'
    optim_suffix = optim_suffix + f'_wd{format(config.optimizer.wd, ".0e")}'.replace('.', 'd')
    optim_suffix = optim_suffix + f'_lr{format(config.optimizer.lr, ".0e")}'
    optim_suffix = optim_suffix + f'_dp{format(config.optimizer.dropout_p, ".0e")}'.replace('.', 'd')

    model_suffix = f'_{config.model.type}_{config.model.activation}_{config.model.num_layers}'

    if config.reg:
        reg_suffix = '_reg'
    else:
        reg_suffix = ''
    
    if config.mark != '':
        mark = "_" + config.mark
    else:
        mark = ''

    label = task.task + data_str + optim_suffix + "_" + config.optimizer.loss + model_suffix + reg_suffix + p_suffix + mark
    if config.mark == 'standard':
        log_dir = "./results/standard/results_logs"
    else:
        log_dir = config.logger.dir
    logger = Logger(label, log_dir)

    for i, train_loader in enumerate(train_loaders):
        print(f"Training on subset {i} with {len(train_loader.dataset)} samples")
        train_num = len(train_loader.dataset)
        valid_num = len(test_loader.dataset)
        # Model
        model = Decoder(
            dim=config.model.dim, num_layers=config.model.num_layers, num_heads=config.model.num_heads, num_tokens=task.p + 2, seq_len=task.seq_len, activation_name=config.model.activation, dropout_p=config.optimizer.dropout_p
        ).to(device)

        if config.model_type == 'mlp':
            model = MLP_arithmetic(
                dim=128, num_layers=config.model.num_layers, num_tokens=task.p + 2, seq_len=task.seq_len, activation_name=config.model.activation, dropout_p=config.optimizer.dropout_p
            ).to(device)
        model = model.to(device)

        print(model)
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"Device: {device}")

        def loss_fn(logits, sequence, reduction='mean'):
            return F.cross_entropy(logits[:, -1], sequence[:, -1], reduction=reduction)

        # Optimizer
        #optimizer = get_optimizer(config.optimizer, model)
        lr = config.optimizer.lr
        weight_decay = config.optimizer.wd
        betas = (config.optimizer.beta1, config.optimizer.beta2)
        if config.nsm:
            # NSM optimizer
            print(f"Using NSM optimizer with sigma={config.optimizer.optimizer_kwargs.nsm_sigma}, distribution={config.optimizer.optimizer_kwargs.distribution}")
            base_optimizer = getattr(torch.optim, config.optimizer.base_optimizer)
            optimizer = NSM(
                model.parameters(),
                base_optimizer,
                sigma=config.optimizer.optimizer_kwargs.nsm_sigma,
                distribution=config.optimizer.optimizer_kwargs.distribution,
                lr=lr,
                weight_decay=weight_decay,
                betas=betas
            )
            #  linear learning rate warmup over the first 10 updates
            # scheduler = torch.optim.lr_scheduler.LambdaLR(
            #     optimizer.base_optimizer, lambda update: 1 if update > 10 else update / 10
            # )
        elif config.sam:
            # SAM optimizer
            print(f"Using SAM optimizer")
            base_optimizer = getattr(torch.optim, config.optimizer.base_optimizer)
            optimizer = SAM(
                model.parameters(),
                base_optimizer,
                rho=config.optimizer.optimizer_kwargs.rho,
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
            )
            #  linear learning rate warmup over the first 10 updates
            # scheduler = torch.optim.lr_scheduler.LambdaLR(
            #     optimizer.base_optimizer, lambda update: 1 if update > 10 else update / 10
            # )
        else:
            optimizer = getattr(torch.optim, config.optimizer.base_optimizer)(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
                #betas=(args.beta1, args.beta2),
            )

    

        # Compute/save hessian condition
        if config.train.hessian_log_every == 0 or config.train.hessian_log_every is None:
            def log_hessian(e):
                return False
        else:
            def log_hessian(e):
                return (e < 10 or (e+1) % config.train.hessian_log_every == 0)

        steps_per_epoch = math.ceil(train_num / config.train.batch_size)
        total_epochs = config.train.budget // steps_per_epoch
        print(f"Epochs: {total_epochs}, steps per epoch: {steps_per_epoch}, train num: {train_num}")

        train(config, task, model, optimizer, loss_fn, train_loader, test_loader, device, model_id=i, epochs=total_epochs, logger=logger)


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

    args = parser.parse_args()

    # Load configs
    config = load_config(args.task)

    config = update_config(config, args)

    print(config)

    # Run the experiment
    run_experiment(config)