from argparse import ArgumentParser
import math
from tqdm import tqdm
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset, Subset
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


def calculate_influence_function(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    load_batch_fn,
    loss_fn,
    cg_iterations: int = 10,
    cg_damping: float = 1e-4
):
    """
    计算训练集中每个样本对测试集中每个样本的影响分数 (修正版)。
    
    遵循 train_all.py 的模式:
    - model(inputs) -> logits (inputs = batch[:, :-1])
    - loss_fn(logits, targets) -> loss (targets = batch[:, -1])
    
    Args:
        model (nn.Module): 已经训练好的PyTorch模型。
        train_loader (DataLoader): 训练数据的DataLoader。
                                  注意：为了计算精确的HVP，该函数假设
                                  整个训练集可以一次性加载到内存/显存中。
        test_loader (DataLoader): 测试数据的DataLoader。
        load_batch_fn: 一个函数，输入(batch, device)，返回移动到设备上的 (inputs, labels)。
        loss_fn: 损失函数 (例如 nn.BCELoss())。
        cg_iterations (int): 共轭梯度法的迭代次数。
        cg_damping (float): HVP计算中的阻尼项，用于增加稳定性。

    Returns:
        np.ndarray: 一个形状为 (N_test, N_train) 的numpy数组。
    """
    
    # --- 内部辅助函数定义 ---
    
    def _flatten_tensors(tensors):
        """将一个张量列表/元组展平成一个单一向量。"""
        if not tensors:
            return torch.tensor([])
        return torch.cat([t.view(-1) for t in tensors])

    def _hvp(loss, model_params, v):
        """
        计算海森-向量乘积 (Hessian-Vector Product)。
        修正：直接对 model.parameters() 求导。
        """
        # 第一步: 计算 grad(loss, model_params)
        grad_tuple = grad(loss, model_params, create_graph=True, retain_graph=True)
        grad_flat = _flatten_tensors(grad_tuple)
        
        # 第二步: 计算 (grad_flat * v).sum()
        dot_product = torch.dot(grad_flat, v)
        
        # 第三步: 计算 dot_product 对 model_params 的梯度
        hvp_tuple = grad(dot_product, model_params, retain_graph=True)
        hvp_flat = _flatten_tensors(hvp_tuple)
        
        return hvp_flat + cg_damping * v # 增加阻尼项

    def _get_inverse_hvp_cg(avg_train_loss, model_params, v):
        """使用共轭梯度法近似计算 H^{-1}v。"""
        def get_hvp_func(vec):
            return _hvp(avg_train_loss, model_params, vec)

        x = torch.zeros_like(v)
        r = v.clone()
        p = v.clone()
        rs_old = torch.dot(r, r)
        
        for _ in range(cg_iterations):
            Ap = get_hvp_func(p)
            alpha = rs_old / (torch.dot(p, Ap) + 1e-10)
            x += alpha * p
            r -= alpha * Ap
            rs_new = torch.dot(r, r)
            if torch.sqrt(rs_new) < 1e-8:
                break
            p = r + (rs_new / (rs_old + 1e-10)) * p
            rs_old = rs_new
        return x

    # --- 函数主体开始 ---
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 获取模型参数列表 (用于求导)
    model_params = list(model.parameters())

    try:
        all_train_inputs, all_train_labels = zip(*[load_batch_fn(b, device) for b in train_loader])
        all_train_inputs = torch.cat(all_train_inputs)
        all_train_labels = torch.cat(all_train_labels)
    except Exception as e:
        print(f"无法一次性加载整个训练集: {e}")
        return None

    with sdpa_kernel(SDPBackend.MATH):
        # Following train_all.py pattern: model(inputs) -> logits, loss_fn(logits, targets)
        print(all_train_inputs)
        print(all_train_labels)
        avg_train_loss = loss_fn(model(all_train_inputs), all_train_labels)
        
        n_train = len(train_loader.dataset)
        n_test = len(test_loader.dataset)
        all_influences = np.zeros((n_test, n_train))

        for test_idx, test_batch in enumerate(tqdm(test_loader, desc="处理测试样本")):
            test_inputs, test_labels = load_batch_fn(test_batch, device)
            
            # Following train_all.py: logits = model(inputs), loss = loss_fn(logits, targets)
            test_loss = loss_fn(model(test_inputs), test_labels)
            
            # 修正: 对 model.parameters() 求导，然后展平
            v_tuple = grad(test_loss, model_params, retain_graph=True)
            v_flat = _flatten_tensors(v_tuple)
            
            s_test = _get_inverse_hvp_cg(avg_train_loss, model_params, v_flat)
            
            for train_idx in range(n_train):
                train_inputs = all_train_inputs[train_idx].unsqueeze(0)
                train_labels = all_train_labels[train_idx].unsqueeze(0)
                
                # Following train_all.py: logits = model(inputs), loss = loss_fn(logits, targets)
                train_sample_loss = loss_fn(model(train_inputs), train_labels)

                # 修正: 对 model.parameters() 求导，然后展平
                grad_z_tuple = grad(train_sample_loss, model_params)
                grad_z_flat = _flatten_tensors(grad_z_tuple)
                
                influence_score = -torch.dot(s_test, grad_z_flat) / n_train
                all_influences[test_idx, train_idx] = influence_score.item()

    return all_influences

def run_experiment(config):
    
    # Seed
    #seed_everything(config.train.seed)

    # Device
    device = torch.device(f"cuda:{int(config.optimizer.device)}") # get_device()

    # Task
    #task = get_task(config.task)
    data_path = f'./dataset/{config.task.task_kwargs.task}_p_{config.task.task_kwargs.p}_split_{config.task.task_kwargs.train_ratio}'
    task = ModularArithmetic(config.task.task_kwargs)
    
    print(f"Using existing dataset at {data_path}")
    train_dataset = torch.load(os.path.join(data_path, 'train_dataset.pt'), weights_only=False)
    valid_dataset = torch.load(os.path.join(data_path, 'valid_dataset.pt'), weights_only=False)
    sub_valid_dataset = Subset(valid_dataset, list(range(100)))  # 只使用前100个验证样本以节省计算时间

    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(sub_valid_dataset, batch_size=1, shuffle=False, num_workers=8)

    # Model
    if config.model_type == 'mlp':
        model = MLP_arithmetic(
            dim=128, num_layers=config.model.num_layers, num_tokens=task.p + 2, seq_len=task.seq_len, activation_name=config.model.activation, dropout_p=config.optimizer.dropout_p
        ).to(device)
    else:
        model = Decoder(
            dim=config.model.dim, num_layers=config.model.num_layers, num_heads=config.model.num_heads, num_tokens=task.p + 2, seq_len=task.seq_len, activation_name=config.model.activation, dropout_p=config.optimizer.dropout_p
        ).to(device)

    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Device: {device}")

    # Load pre-trained model if checkpoint exists
    model_name = 'nso_model' if config.nsm else 'model'
    
    checkpoint_path = f'./checkpoints/{config.task.task_kwargs.task}/phase{config.phase}/all/{model_name}_all_1.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading pre-trained model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}")
        print("Please train the model first using train_all.py")

    # Following train_all.py: same loss function definition
    def loss_fn(logits, sequence, reduction='mean'):
        return F.cross_entropy(logits[:, -1], sequence, reduction=reduction)

    # Define load_batch_fn for the modular arithmetic task
    def load_batch_fn(batch, device):
        """Load batch for modular arithmetic task - following train_all.py pattern"""
        # Following train_all.py: batch is wrapped in tuple, extract batch[0]
        batch = batch[0].to(device)
        # For modular arithmetic, inputs are batch[:, :-1] and targets are batch[:, -1]
        # This matches train_all.py: logits = model(batch[:, :-1])
        inputs = batch[:, :-1]
        targets = batch[:, -1]
        return inputs, targets

    print("\n开始计算影响函数...")
    influences = calculate_influence_function(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        load_batch_fn=load_batch_fn,
        loss_fn=loss_fn
    )
    print("影响函数计算完成。")
    
    # Save results
    if influences is not None:
        results_path = f'./results/influence_function/{config.task.task_kwargs.task}_p_{config.task.task_kwargs.p}_split_{config.task.task_kwargs.train_ratio}'
        os.makedirs(results_path, exist_ok=True)
        if config.nsm:
            model_name = 'nso_model'
        else:
            model_name = 'model'
        results_file = os.path.join(results_path, f'phase{config.phase}_{model_name}_influence_matrix_{config.mark}.npy')
        np.save(results_file, influences)
        print(f"影响矩阵已保存到: {results_file}")
        
        print(f"\n影响矩阵的形状: {influences.shape}")
        
        # 以第一个测试样本为例进行分析
        test_sample_index = 0
        influences_for_sample_0 = influences[test_sample_index]
        print(valid_dataset[test_sample_index])
        
        # 找出最有帮助的训练样本
        most_helpful_indices = np.argsort(influences_for_sample_0)[-5:][::-1].copy()
        print(f"\n对测试样本 {test_sample_index} 最有帮助的5个训练样本索引:")
        print(most_helpful_indices)
        print(train_dataset[most_helpful_indices])
        print("对应影响分数:", influences_for_sample_0[most_helpful_indices])
        
        # 找出最有害的训练样本
        most_harmful_indices = np.argsort(influences_for_sample_0)[:5].copy()
        print(f"\n对测试样本 {test_sample_index} 最有害的5个训练样本索引:")
        print(most_harmful_indices)
        print(train_dataset[most_harmful_indices])
        print("对应影响分数:", influences_for_sample_0[most_harmful_indices])
    else:
        print("影响函数计算失败")

def update_config(config, args):
    config.sam = args.sam   
    config.nsm = args.nsm
    config.reg = args.reg
    config.swa = args.swa
    config.mark = args.mark
    config.model_type = args.model_type
    config.phase = args.phase

    # update data generation
    config.task.task_kwargs.p = args.p
    # config.task.task_kwargs.num_input_numbers = args.num_input_numbers
    # config.task.task_kwargs.num_total_samples = args.num_total_samples
    config.task.task_kwargs.train_ratio = args.train_ratio
    config.task.task_kwargs.valid_ratio = args.valid_ratio

    # training
    config.optimizer.device = args.device

    return config

if __name__ == '__main__':
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
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--phase", type=int, default=1)

    args = parser.parse_args()

    # Load configs
    config = load_config(args.task)

    config = update_config(config, args)

    print(config)

    # Run the experiment
    run_experiment(config)