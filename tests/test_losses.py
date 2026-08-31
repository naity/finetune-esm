import pytest
import torch

from finetune_esm import losses


def test_registry_contains_mse():
    assert "mse" in losses.LOSSES


def test_mse_matches_torch():
    pred, target = torch.tensor([1.0, 2.0]), torch.tensor([1.0, 4.0])
    loss = losses.get_loss("mse")(pred, target, None)
    assert torch.allclose(loss, torch.tensor(2.0))


def test_loss_accepts_group_id_and_ignores_it_for_mse():
    pred, target = torch.randn(8), torch.randn(8)
    groups = torch.randint(0, 3, (8,))
    fn = losses.get_loss("mse")
    assert torch.allclose(fn(pred, target, groups), fn(pred, target, None))


def test_unknown_loss_raises_with_available_names():
    with pytest.raises(KeyError, match="mse"):
        losses.get_loss("nonexistent")
