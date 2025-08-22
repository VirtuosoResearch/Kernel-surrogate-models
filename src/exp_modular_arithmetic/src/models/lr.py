import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearRegression(nn.Module):
    """
    Linear regression model: y = w^T x (+ b)
    """
    def __init__(self, num_features, constant=False):
        super().__init__()
        self.w = nn.Parameter(torch.randn(num_features, 1))
        if constant:
            self.b = nn.Parameter(torch.randn(1))
        else:
            self.register_parameter('b', None)

    def forward(self, x):
        y = x @ self.w
        if self.b is not None:
            y += self.b
        return y