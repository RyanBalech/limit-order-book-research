from __future__ import annotations

import torch
from torch import nn


class DeepLOBStyleClassifier(nn.Module):
    """Compact CNN-LSTM baseline for sequences of engineered LOB features."""

    def __init__(
        self,
        n_features: int,
        n_classes: int = 3,
        conv_channels: int = 32,
        hidden_size: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.temporal = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("expected [batch, sequence, features]")
        z = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        z, _ = self.temporal(z)
        return self.head(self.dropout(z[:, -1]))


def make_sequence_windows(
    x: torch.Tensor, y: torch.Tensor, sequence_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align each label with a strictly backward-looking feature window."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if len(x) != len(y):
        raise ValueError("features and labels must have equal length")
    if len(x) < sequence_length:
        raise ValueError("not enough observations for one sequence")
    windows = x.unfold(0, sequence_length, 1).permute(0, 2, 1)
    labels = y[sequence_length - 1 :]
    return windows.contiguous(), labels
