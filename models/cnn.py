"""
models/cnn.py
=============
1-D Convolutional Neural Network for raw time-series sensor windows.

Architecture
------------
    Input (batch, C, L)
      → Conv1d 32 k=7  → BN → ReLU → MaxPool
      → Conv1d 64 k=5  → BN → ReLU → MaxPool
      → Conv1d 128 k=3 → BN → ReLU → AdaptiveAvgPool → flatten
      → Linear(256) → ReLU → Dropout
      → Linear(num_classes)
"""

import torch
import torch.nn as nn

from models.base_model import BaseModel


class CNN1D(BaseModel):
    """
    1-D CNN for variable-length time-series inputs.

    Parameters
    ----------
    in_channels : int  — number of sensor channels (e.g. 1 for HAR features)
    seq_len     : int  — length of the input sequence
    num_classes : int  — number of output classes
    dropout     : float
    """

    def __init__(
        self,
        in_channels: int = 1,
        seq_len: int = 561,
        num_classes: int = 6,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.num_classes = num_classes

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(output_size=4),  # → (batch, 128, 4)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                           # → (batch, 512)
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, in_channels, seq_len)
            If a 2-D tensor (batch, seq_len) is passed, a channel dim is added.

        Returns
        -------
        logits : (batch, num_classes)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # add channel dim
        x = self.features(x)
        return self.classifier(x)

    def __repr__(self) -> str:
        return (
            f"CNN1D(in_channels={self.in_channels}, seq_len={self.seq_len}, "
            f"params={self.count_parameters():,}, classes={self.num_classes})"
        )
