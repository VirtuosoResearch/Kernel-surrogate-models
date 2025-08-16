import torch
import json
from utils.data import _load_groups, _load_subset_gids, _concat_indices
from scipy.stats import spearmanr

if __name__ == "__main__":
    num_samples = 50
    num_models = 10
    num_scores = 5
    results_list = []
    # build group samples
    train_groups_path = './indices/train_groups_data_indices.json'
    test_groups_path = './indices/test_groups_data_indices.json'
    train_subsets_gid_path = './indices/train_subsets_group_indices.json'
    test_subsets_gid_path = './indices/test_subsets_group_indices.json'
    train_groups = _load_groups(train_groups_path)
    test_groups = _load_groups(test_groups_path)
    train_subsets_gids = _load_subset_gids(train_subsets_gid_path)
    print(train_subsets_gids)
    n=50
    w_list = []
    for subset in train_subsets_gids:
        w_i = torch.zeros(n)
        w_i[subset] = 1
        w_list.append(w_i)
    w = torch.stack(w_list, dim=0)

    for j in range(num_scores):
        scores_list = []
        for i in range(num_models):
            result_file = 'results/real/results_{}.json'.format(i)
            with open(result_file, 'r') as f:
                results = json.load(f)
            scores_list.append(torch.tensor(results['test_loss'][j]))
        scores = torch.stack(scores_list, dim=0)

        phi_col = torch.linalg.lstsq(w, scores).solution
        print(phi_col.shape)
        print(phi_col)
        residual = w @ phi_col - scores
        print("||residual|| =", residual.norm().item())

        # Compute Spearman correlation
        spearman_corr, _ = spearmanr(scores.cpu().numpy(), (w @ phi_col).cpu().numpy())
        print("Spearman correlation =", spearman_corr)
        results_list.append({
            'phi': phi_col.tolist(),
            'spearman_corr': spearman_corr,
        })