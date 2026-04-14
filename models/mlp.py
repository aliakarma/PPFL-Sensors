"""
models/mlp.py
=============
Multi-layer perceptron for tabular sensor features (e.g., HAR 561-dim vectors).

BatchNorm is intentionally replaced with LayerNorm.  BatchNorm stores
running_mean / running_var in state_dict, which FedAvg incorrectly averages
across non-IID clients (FedBN, Li et al. 2021).  LayerNorm has no running
statistics and is safe for federated aggregation.

Architecture
------------
    Input → [Linear → LayerNorm → ReLU → Dropout] × len(hidden_dims) → Linear → logits
"""

import torch
import torch.nn as nn
from typing import List

from models.base_model import BaseModel


class MLP(BaseModel):
    """
    Flexible MLP with configurable hidden layers using LayerNorm (FL-safe).

    Parameters
    ----------
    input_dim   : int
    hidden_dims : list  e.g. [256, 128]
    num_classes : int
    dropout     : float
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),   # FL-safe: no running statistics in state_dict
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def __repr__(self) -> str:
        return (
            f"MLP(input_dim={self.input_dim}, "
            f"params={self.count_parameters():,}, "
            f"classes={self.num_classes})"
        )
