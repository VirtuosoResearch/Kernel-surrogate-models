import torch
import numpy as np
import json
import os
import re
from scipy.stats import spearmanr

def eval_test_loss(model, loss_fn, load_batch_fn, test_dataset, batch_size, batch_num, device='cuda'):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=torch.cuda.is_available())

    batch_loss_list = []
    total_loss = 0.0
    with torch.no_grad():
        for i, (batch) in enumerate(test_loader):
            inputs, targets, batch_size = load_batch_fn(batch, device)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            batch_loss_list.append(loss.item())
            if i >= batch_num - 1:
                break
    return batch_loss_list

def get_subset_scores(model, model_path, model_name, train_ids_list, loss_fn, load_batch_fn, test_dataset, test_batch_size=1, test_batch_num=50, device='cuda'):
    num_samples = np.max(train_ids_list) + 1
    pattern = re.compile(rf"^{re.escape(model_name)}_\w+\.pth$")
    model_file_list = []
    for file in os.listdir(model_path):
        if pattern.match(file):
            model_file = os.path.join(model_path, file)
            model_file_list.append(model_file)
    num_models = len(model_file_list)
    print(f"Found {num_models} models matching the pattern '{model_name}' in '{model_path}'.")

    subset_scores = []
    for i in range(num_models):
        model_file = model_file_list[i]
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
        model.eval()
        scores = eval_test_loss(model, loss_fn, load_batch_fn, test_dataset, test_batch_size, test_batch_num, device)
        subset_scores.append(scores)
    subset_scores = torch.tensor(subset_scores)
    print("Subset scores shape:", subset_scores.shape)

    return subset_scores

def datamodels(subset_scores, model_path, model_name, train_ids_list, test_batch_num=50):
    num_samples = np.max(train_ids_list) + 1
    pattern = re.compile(rf"^{re.escape(model_name)}_\w+\.pth$")
    model_file_list = []
    for file in os.listdir(model_path):
        if pattern.match(file):
            model_file = os.path.join(model_path, file)
            model_file_list.append(model_file)
    num_models = len(model_file_list)
    print(f"Found {num_models} models matching the pattern '{model_name}' in '{model_path}'.")

    w_list = []
    for subset_index in range(num_models):
        w_i = torch.zeros(num_samples)
        w_i[train_ids_list[subset_index]] = 1
        w_list.append(w_i)
    w = torch.stack(w_list, dim=0)

    results_list = []
    for j in range(test_batch_num):
        scores = subset_scores[:, j]
        phi_col = torch.linalg.lstsq(w, scores).solution
        print(f"Phi column for batch {j}:", phi_col)

        residual = w @ phi_col - scores
        print("||residual|| =", residual.norm().item())

        # Compute Spearman correlation
        spearman_corr, _ = spearmanr(scores.cpu().numpy(), (w @ phi_col).cpu().numpy())
        print("Spearman correlation =", spearman_corr)
        results_list.append({
            'error': residual.norm().item(),
            'phi': phi_col.tolist(),
            'spearman_corr': spearman_corr,
        })
    return results_list