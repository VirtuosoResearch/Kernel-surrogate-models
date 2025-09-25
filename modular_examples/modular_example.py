from argparse import ArgumentParser
import json
import os
from typing import List

import torch
import numpy as np
from scipy.stats import spearmanr

from taskHessian.datamodels import make_krr_predictor

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default="addition")
    parser.add_argument("--solver", type=str, default="krr")
    return parser.parse_args()


def main():
    args = parse_args()
    task = args.task
    with open(f"modular_{task}.json", "r") as f:
        data = json.load(f)

    train_subset_scores = torch.tensor(data["train_subset_scores"], dtype=torch.float32)
    test_subset_scores = torch.tensor(data["test_subset_scores"], dtype=torch.float32)
    train_w = torch.tensor(data["train_w"], dtype=torch.float32)
    test_w = torch.tensor(data["test_w"], dtype=torch.float32)
    solver = args.solver
    if solver == "krr":
        predict = make_krr_predictor(train_w, train_subset_scores, alpha=1e-2, kernel="rbf")
        pred_test_scores = predict(test_w)
    elif solver == "lstsq":
        Phi = torch.linalg.lstsq(train_w, train_subset_scores).solution
        pred_test_scores = test_w @ Phi
    else:
        raise ValueError(f"Invalid solver: {solver}")
    spearmans_list = []
    for j in range(test_subset_scores.shape[1]):
        true_scores = test_subset_scores[:, j]
        pred_scores = pred_test_scores[:, j]
        rho, _ = spearmanr(
            true_scores.detach().cpu().numpy(),
            pred_scores.detach().cpu().numpy()
        )
        spearmans_list.append(float(rho))
    results = {
        'spearman_corr': spearmans_list,
    }

    print(f"Average spearman correlation: {np.mean(results['spearman_corr'])}")


if __name__ == "__main__":
    main()
