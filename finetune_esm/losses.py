"""Training objectives.

Every loss takes (pred, target, group_id) so that group-aware and ranking
objectives can be added later without changing any call site.
"""

from typing import Callable

import torch
import torch.nn.functional as F

LossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]


def mse(
    pred: torch.Tensor, target: torch.Tensor, group_id: torch.Tensor | None = None
) -> torch.Tensor:
    return F.mse_loss(pred, target)


LOSSES: dict[str, LossFn] = {"mse": mse}


def get_loss(name: str) -> LossFn:
    if name not in LOSSES:
        raise KeyError(f"unknown loss {name!r}; available: {sorted(LOSSES)}")
    return LOSSES[name]
