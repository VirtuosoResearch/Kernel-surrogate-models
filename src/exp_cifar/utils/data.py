import torch
from torch.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler
from torch.utils.data import Subset, DataLoader
import torchvision
import os
import json
from typing import List

def get_dataloader(batch_size=256, num_workers=8, split='train', shuffle=False, augment=True):
    if augment:
        transforms = torchvision.transforms.Compose(
                        [torchvision.transforms.RandomHorizontalFlip(),
                         torchvision.transforms.RandomAffine(0),
                         torchvision.transforms.ToTensor(),
                         torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                          (0.2023, 0.1994, 0.201))])
    else:
        transforms = torchvision.transforms.Compose([
                         torchvision.transforms.ToTensor(),
                         torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                          (0.2023, 0.1994, 0.201))])

    is_train = (split == 'train')
    dataset = torchvision.datasets.CIFAR10(root='/data/shared/cifar/',
                                           download=True,
                                           train=is_train,
                                           transform=transforms)

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         shuffle=shuffle,
                                         batch_size=batch_size,
                                         num_workers=num_workers)

    return loader

def _load_groups(path: str):
    with open(path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "groups" in payload:
        return payload["groups"]
    raise ValueError(f"Unsupported JSON format in {path}. Expected a dict with 'groups'.")


def _load_subset_gids(path: str):
    with open(path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "subsets" in payload:
        return payload["subsets"]
    raise ValueError(f"Unsupported JSON format in {path}. Expected a dict with 'subsets'.")


def _concat_indices(groups: List[List[int]], gids: List[int]) -> List[int]:
    out: List[int] = []
    for g in gids:
        out.extend(groups[g])
    return out

def get_subset_dataloader(batch_size=256, num_workers=8, shuffle=False, augment=True):
    pin_memory = torch.cuda.is_available()
    if augment:
        transforms = torchvision.transforms.Compose(
                        [torchvision.transforms.RandomHorizontalFlip(),
                         torchvision.transforms.RandomAffine(0),
                         torchvision.transforms.ToTensor(),
                         torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                          (0.2023, 0.1994, 0.201))])
    else:
        transforms = torchvision.transforms.Compose([
                         torchvision.transforms.ToTensor(),
                         torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                          (0.2023, 0.1994, 0.201))])

    train_dataset = torchvision.datasets.CIFAR10(root='/data/shared/cifar/',
                                           download=True,
                                           train=True,
                                           transform=transforms)
    test_dataset = torchvision.datasets.CIFAR10(root='/data/shared/cifar/',
                                           download=True,
                                           train=False,
                                           transform=transforms)

    train_groups_path = './indices/train_groups_data_indices.json'
    test_groups_path = './indices/test_groups_data_indices.json'
    train_subsets_gid_path = './indices/train_subsets_group_indices.json'
    test_subsets_gid_path = './indices/test_subsets_group_indices.json'
    train_groups = _load_groups(train_groups_path)
    test_groups = _load_groups(test_groups_path)
    train_subsets_gids = _load_subset_gids(train_subsets_gid_path)
    test_indices_per_group = test_groups

    train_indices_per_subset = [_concat_indices(train_groups, gids) for gids in train_subsets_gids]

    train_subsets = [Subset(train_dataset, idxs) for idxs in train_indices_per_subset]
    test_subsets = [Subset(test_dataset, idxs) for idxs in test_indices_per_group]

    train_loaders = [
        DataLoader(s, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        for s in train_subsets
    ]
    test_loaders = [
        DataLoader(s, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        for s in test_subsets
    ]

    return train_loaders, test_loaders