import argparse
import json
import os
from typing import List

import torch
import torchvision


def make_group_partition(n: int, num_groups: int, seed: int) -> List[List[int]]:
    """Partition [0..n-1] into num_groups disjoint groups (roughly equal sizes)."""
    assert 1 <= num_groups <= n, "num_groups must be in [1, n]"
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    base = n // num_groups
    sizes = [base] * num_groups
    for i in range(n - base * num_groups):
        sizes[i] += 1
    groups: List[List[int]] = []
    offset = 0
    for sz in sizes:
        groups.append(perm[offset: offset + sz])
        offset += sz
    return groups


def sample_train_subsets_from_groups(num_groups: int, m: int, groups_per_subset: int, seed: int) -> List[List[int]]:
    """For each TRAIN subset, sample K distinct group IDs from [0..num_groups-1]."""
    assert 1 <= groups_per_subset <= num_groups, "groups_per_subset must be in [1, num_groups]"
    g = torch.Generator().manual_seed(seed)
    all_subsets: List[List[int]] = []
    for _ in range(m):
        choice = torch.randperm(num_groups, generator=g)[:groups_per_subset].tolist()
        all_subsets.append(choice)
    return all_subsets


def main():
    p = argparse.ArgumentParser(description="Generate grouped indices for CIFAR-10 (train subsets; test uses groups directly).")
    p.add_argument("--data_root", type=str, default="/data/shared/cifar/", help="CIFAR root dir")
    p.add_argument("--output_dir", type=str, default="indices", help="Directory to write JSON files")

    # grouping
    p.add_argument("--num_train_groups", type=int, default=50, help="Number of disjoint TRAIN groups (G_train)")
    p.add_argument("--num_test_groups", type=int, default=5, help="Number of disjoint TEST groups (G_test = m_test)")
    p.add_argument("--group_seed", type=int, default=1234, help="Base RNG seed for forming groups (train uses seed, test uses seed+1)")

    # train subsets
    p.add_argument("--m", type=int, default=10, help="Number of TRAIN subsets (M)")
    p.add_argument("--groups_per_subset", type=int, default=30, help="Number of TRAIN groups per subset (K)")
    p.add_argument("--subset_seed", type=int, default=2025, help="RNG seed for selecting TRAIN groups for subsets")

    p.add_argument("--overwrite", action="store_true", help="Overwrite existing JSON files if present")

    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load CIFAR-10 just to get dataset lengths
    train_ds = torchvision.datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=None)
    test_ds = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=None)

    n_train = len(train_ds)
    n_test = len(test_ds)

    # Build disjoint groups
    train_groups = make_group_partition(n_train, args.num_train_groups, seed=args.group_seed)
    test_groups = make_group_partition(n_test, args.num_test_groups, seed=args.group_seed + 1)

    # For TRAIN subsets, pick group IDs
    subset_group_ids_train = sample_train_subsets_from_groups(
        args.num_train_groups, args.m, args.groups_per_subset, seed=args.subset_seed
    )

    # Paths
    train_groups_out = os.path.join(args.output_dir, "train_groups_data_indices.json")
    test_groups_out = os.path.join(args.output_dir, "test_groups_data_indices.json")
    subsets_train_out = os.path.join(args.output_dir, "train_subsets_group_indices.json")

    # Guard against overwrites
    if (not args.overwrite) and any(
        os.path.exists(pth) for pth in [train_groups_out, test_groups_out, subsets_train_out]
    ):
        raise FileExistsError("Refusing to overwrite existing files. Remove them or pass --overwrite.")

    # Save groups
    with open(train_groups_out, "w") as f:
        json.dump({
            "dataset": "CIFAR10",
            "split": "train",
            "type": "groups",
            "num_examples": n_train,
            "num_groups": args.num_train_groups,
            "seed": args.group_seed,
            "groups": train_groups,
        }, f)

    with open(test_groups_out, "w") as f:
        json.dump({
            "dataset": "CIFAR10",
            "split": "test",
            "type": "groups",
            "num_examples": n_test,
            "num_groups": args.num_test_groups,
            "seed": args.group_seed + 1,
            "groups": test_groups,
        }, f)

    # Save TRAIN subset group selections
    with open(subsets_train_out, "w") as f:
        json.dump({
            "dataset": "CIFAR10",
            "split": "train",
            "type": "subsets_from_groups",
            "num_subsets": args.m,
            "groups_per_subset": args.groups_per_subset,
            "seed": args.subset_seed,
            "subsets": subset_group_ids_train,
        }, f)

    # Human-friendly summary
    def _sizes(groups: List[List[int]]):
        return [len(g) for g in groups]

    def _subset_sizes(groups: List[List[int]], subsets_gid: List[List[int]]):
        return [sum(len(groups[g]) for g in gids) for gids in subsets_gid]

    print("Saved:")
    print(f"  {train_groups_out}  (TRAIN group sizes: {_sizes(train_groups)[:10]}... total groups={len(train_groups)})")
    print(f"  {test_groups_out}   (TEST  group sizes: {_sizes(test_groups)[:10]}... total groups={len(test_groups)})")
    print(f"  {subsets_train_out} (TRAIN subset sizes ~ {_subset_sizes(train_groups, subset_group_ids_train)})")

if __name__ == "__main__":
    main()