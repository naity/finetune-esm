import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from finetune_esm.data.batching import collate
from finetune_esm.models import fusion, heads, module

DIM = 8


def _items(n=16):
    torch.manual_seed(0)
    out = []
    for i in range(n):
        out.append(
            {
                "ligand": torch.randn(4, DIM).numpy(),
                "ligand_chain_id": torch.zeros(4, dtype=torch.long).numpy(),
                "receptor": torch.randn(3, DIM).numpy(),
                "receptor_chain_id": torch.zeros(3, dtype=torch.long).numpy(),
                "target": float(i),
                "group_id": i % 4,
            }
        )
    return out


def _model(total_steps=8):
    fuse = fusion.FUSIONS["hp"](DIM)
    return module.AffinityModule(fuse, heads.MLPHead(fuse.out_dim), total_steps=total_steps)


def test_forward_returns_one_prediction_per_row():
    batch = collate(_items(5))
    assert _model()(batch).shape == (5,)


def test_configure_optimizers_returns_optimizer_and_step_scheduler():
    cfg = _model().configure_optimizers()
    assert isinstance(cfg["optimizer"], torch.optim.Optimizer)
    assert cfg["lr_scheduler"]["interval"] == "step"


def test_one_training_run_completes_with_finite_val_loss():
    # 16 items / batch_size 4 = 4 optimizer steps per epoch * 4 epochs = 16 steps,
    # so total_steps must be >= 16 or OneCycleLR raises once stepped past its budget.
    model = _model(total_steps=16)
    loader = DataLoader(_items(16), batch_size=4, collate_fn=collate)
    trainer = pl.Trainer(
        max_epochs=4, accelerator="cpu", logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
    )
    trainer.fit(model, loader, loader)
    assert torch.isfinite(torch.tensor(trainer.callback_metrics["val_loss"].item()))


def test_validation_metrics_are_logged_over_the_whole_epoch():
    model = _model()
    loader = DataLoader(_items(16), batch_size=4, collate_fn=collate)
    trainer = pl.Trainer(
        max_epochs=1, accelerator="cpu", logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
    )
    trainer.fit(model, loader, loader)
    for key in ("val_loss", "val_spearman", "val_pearson", "val_rmse"):
        assert key in trainer.callback_metrics
