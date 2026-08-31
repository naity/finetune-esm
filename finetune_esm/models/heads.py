"""Prediction heads mapping a fused vector to a scalar pKd."""

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int | None = None, dropout: float = 0.1
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
