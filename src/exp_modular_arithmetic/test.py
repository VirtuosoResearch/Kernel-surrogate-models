import torch
from typing import Dict, Any

def create_quadratic_features(features: torch.Tensor) -> torch.Tensor:
    """
    将原始特征矩阵增广，为其加入二次项（常数项和交叉项）。
    
    这个函数假设输入特征是二元的 (0/1)，因此只添加 w_i * w_j (i < j) 形式的
    交叉项，而省略了与线性项冗余的平方项 w_i^2。
    
    转换逻辑: [w_1, ..., w_n] -> [1, w_1, ..., w_n, w_1*w_2, w_1*w_3, ...]

    Args:
        features (torch.Tensor): 形状为 (m, n) 的原始特征矩阵，m为样本数，n为特征数。

    Returns:
        torch.Tensor: 形状为 (m, 1 + n + n*(n-1)/2) 的增广特征矩阵。
    """
    m, n = features.shape

    # 1. 创建常数项 (Bias term)
    bias = torch.ones(m, 1, device=features.device, dtype=features.dtype)

    # 2. 线性项就是原始特征
    linear_terms = features

    # 3. 创建二次交叉项
    if n < 2:
        # 如果特征数小于2，则没有交叉项
        return torch.cat([bias, linear_terms], dim=1)
        
    all_pairs = features.unsqueeze(2) * features.unsqueeze(1)
    # 获取上三角（不含对角线）的索引
    triu_indices = torch.triu_indices(n, n, offset=1, device=features.device)
    cross_terms = all_pairs[:, triu_indices[0], triu_indices[1]]

    # 4. 拼接所有部分
    features_quad = torch.cat([bias, linear_terms, cross_terms], dim=1)
    
    return features_quad


def solve_quadratic_lstsq(
    train_features: torch.Tensor, 
    train_targets: torch.Tensor, 
    driver: str = 'gelss'
) -> Dict[str, Any]:
    """
    使用增广的二次特征求解最小二乘回归问题。

    Args:
        train_features (torch.Tensor): 形状为 (m, n) 的原始训练特征矩阵。
        train_targets (torch.Tensor): 形状为 (m,) 或 (m, k) 的训练目标值。
        driver (str, optional): 传递给 torch.linalg.lstsq 的 LAPACK 驱动。默认为 'gelss'。

    Returns:
        Dict[str, Any]: 一个包含求解结果的字典，包括：
            'solution' (torch.Tensor): 模型的系数 (Theta)。
            'condition_number' (float): 增广矩阵的条件数。
    """
    # 1. 调用辅助函数创建增广特征矩阵
    train_features_quad = create_quadratic_features(train_features)
    
    # 2. 计算条件数以评估问题的数值稳定性
    cond_num = torch.linalg.cond(train_features_quad.to(torch.float64)).item()
    
    # 3. 求解最小二乘问题
    solution = torch.linalg.lstsq(
        train_features_quad, train_targets, driver=driver
    ).solution
    
    results = {
        'solution': solution,
        'condition_number': cond_num,
    }
    
    return results

import torch

# --- 1. 使用与上次完全相同的设置准备数据 ---
# 为了结果可复现，我们设置一个随机种子
torch.manual_seed(42)

num_train = 1000
num_test = 20
num_samples = 15  # n, 原始特征的数量

# 训练数据
train_w = torch.randint(0, 2, (num_train, num_samples)).float()
train_scores = torch.randn(num_train)

# 测试数据
test_w = torch.randint(0, 2, (num_test, num_samples)).float()
test_scores_true = torch.randn(num_test) # 目标值也是随机的


# --- 2. 重新运行二次模型（结果应与您报告的相似）---
print("--- 二次模型重新验证 ---")
# a. 创建二次特征
train_w_quad = create_quadratic_features(train_w)
test_w_quad = create_quadratic_features(test_w)
# b. 求解
Theta_quad = torch.linalg.lstsq(train_w_quad, train_scores).solution
# c. 预测与评估
predictions_quad = test_w_quad @ Theta_quad
mse_quad = torch.mean((predictions_quad - test_scores_true) ** 2)
print(f"二次模型的特征数量: {train_w_quad.shape[1]}")
print(f"二次模型的测试MSE: {mse_quad.item():.4f}\n")


# --- 3. 运行线性模型进行对比 ---
print("--- 线性模型验证 ---")
# a. 为线性模型创建特征矩阵（原始特征 + 偏置项）
bias_train = torch.ones(num_train, 1)
train_w_linear = torch.cat([bias_train, train_w], dim=1)

bias_test = torch.ones(num_test, 1)
test_w_linear = torch.cat([bias_test, test_w], dim=1)

# b. 求解
Theta_linear = torch.linalg.lstsq(train_w_linear, train_scores).solution

# c. 预测与评估
predictions_linear = test_w_linear @ Theta_linear
mse_linear = torch.mean((predictions_linear - test_scores_true) ** 2)
print(f"线性模型的特征数量: {train_w_linear.shape[1]}")
print(f"线性模型的测试MSE: {mse_linear.item():.4f}\n")

# --- 4. 结论对比 ---
print("--- 结论 ---")
if mse_linear < mse_quad:
    print(f"线性模型的表现更好 (MSE: {mse_linear:.4f} < {mse_quad:.4f})。")
else:
    print(f"二次模型的表现更好 (MSE: {mse_quad:.4f} < {mse_linear:.4f})。")