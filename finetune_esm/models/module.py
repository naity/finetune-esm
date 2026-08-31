"""LightningModule wiring fusion, head, loss and metrics together."""

import lightning.pytorch as pl
import torch

from finetune_esm import losses, metrics
from finetune_esm.data.batching import EmbeddingBatch


class AffinityModule(pl.LightningModule):
    def __init__(
        self,
        fusion: torch.nn.Module,
        head: torch.nn.Module,
        loss_name: str = "mse",
        learning_rate: float = 1e-3,
        total_steps: int = 1000,
        warmup_steps: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["fusion", "head"])
        self.fusion = fusion
        self.head = head
        self.loss_fn = losses.get_loss(loss_name)
        self.learning_rate = learning_rate
        self.total_steps = total_steps
        self.warmup_steps = min(warmup_steps, max(total_steps - 1, 1))
        self._val_buffer: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._test_buffer: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, batch: EmbeddingBatch) -> torch.Tensor:
        return self.head(self.fusion(batch))

    def training_step(self, batch: EmbeddingBatch, batch_idx: int) -> torch.Tensor:
        pred = self(batch)
        loss = self.loss_fn(pred, batch.target, batch.group_id)
        self.log("train_loss", loss, prog_bar=True, batch_size=len(batch.target))
        return loss

    def _eval_step(self, batch, buffer, prefix):
        pred = self(batch)
        loss = self.loss_fn(pred, batch.target, batch.group_id)
        self.log(f"{prefix}_loss", loss, prog_bar=True, batch_size=len(batch.target))
        buffer.append((pred.detach().cpu(), batch.target.detach().cpu()))

    def _eval_epoch_end(self, buffer, prefix):
        if not buffer:
            return
        pred = torch.cat([p for p, _ in buffer])
        target = torch.cat([t for _, t in buffer])
        for name, value in metrics.all_metrics(pred, target).items():
            self.log(f"{prefix}_{name}", value, prog_bar=(name == "spearman"))
        buffer.clear()

    def validation_step(self, batch: EmbeddingBatch, batch_idx: int) -> None:
        self._eval_step(batch, self._val_buffer, "val")

    def on_validation_epoch_end(self) -> None:
        self._eval_epoch_end(self._val_buffer, "val")

    def test_step(self, batch: EmbeddingBatch, batch_idx: int) -> None:
        self._eval_step(batch, self._test_buffer, "test")

    def on_test_epoch_end(self) -> None:
        self._eval_epoch_end(self._test_buffer, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.total_steps,
            pct_start=self.warmup_steps / self.total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
