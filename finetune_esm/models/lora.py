"""Low-rank adaptation, implemented from scratch and pinned against PEFT.

Two departures from the previous implementation in this repo:
  * scaling is alpha / rank, the standard formulation, not alpha;
  * wrapped layers can keep their base weights trainable, so a randomly
    initialised head is never frozen behind a rank-r bottleneck.
"""

import math
from functools import partial

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.W_a = nn.Parameter(torch.randn(in_dim, rank) / math.sqrt(rank))
        self.W_b = nn.Parameter(torch.zeros(rank, out_dim))
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaling * ((x @ self.W_a) @ self.W_b)


class LinearWithLoRA(nn.Module):
    def __init__(
        self, linear: nn.Linear, rank: int, alpha: float, freeze_base: bool = True
    ) -> None:
        super().__init__()
        self.linear = linear
        for param in self.linear.parameters():
            param.requires_grad = not freeze_base
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


def _replace(parent: nn.Module, name: str, wrap) -> None:
    setattr(parent, name, wrap(getattr(parent, name)))


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16,
    target_layers: list[str] | None = None,
    head_layers: list[str] | None = None,
) -> int:
    """Wrap named `nn.Linear` attributes of `model` with LoRA.

    `target_layers` are frozen behind the adapter; `head_layers` keep their
    base weights trainable. Returns the number of modules wrapped.
    """
    frozen = partial(LinearWithLoRA, rank=rank, alpha=alpha, freeze_base=True)
    trainable = partial(LinearWithLoRA, rank=rank, alpha=alpha, freeze_base=False)

    wrapped = 0
    for name in target_layers or []:
        _replace(model, name, frozen)
        wrapped += 1
    for name in head_layers or []:
        _replace(model, name, trainable)
        wrapped += 1
    return wrapped


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
