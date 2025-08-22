import torch

# Data = (size, ...)
def split_data(data, split_ratio=0.5):
    perm = torch.randperm(data.shape[0])
    train_idx = perm[:int(data.shape[0] * split_ratio)]
    valid_idx = perm[int(data.shape[0] * split_ratio):]
    train_data, valid_data = data[train_idx, ...], data[valid_idx, ...]
    return train_data, valid_data

# Data = (:, size)
def old_split_data(data, split_ratio=0.5):
    perm = torch.randperm(data.shape[1])
    train_idx = perm[:int(data.shape[1] * split_ratio)]
    valid_idx = perm[int(data.shape[1] * split_ratio):]
    train_data, valid_data = data[:, train_idx], data[:, valid_idx]
    print(f"Train data: {train_data.shape}")
    print(f"Valid data: {valid_data.shape}")
    return train_data, valid_data