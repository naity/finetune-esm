import torch
import torch.nn as nn
import lightning.pytorch as pl
from typing import Optional
from transformers import EsmModel
from torchmetrics.functional.classification import multilabel_f1_score


class FinetuneESM(nn.Module):
    """
    Finetune ESM model for protein sequence classification.

    Args:
        esm_model (str): Pre-trained ESM model name (e.g., "facebook/esm2_t6_8M_UR50D").
        dropout_p (float): Dropout probability for regularization.
        num_classes (int): Number of output dimensions for the last linear layer.
    """

    def __init__(self, esm_model: str, dropout_p: float, num_classes: int) -> None:
        super().__init__()
        self.llm = EsmModel.from_pretrained(esm_model)
        embedding_dim = self.llm.config.hidden_size
        self.dropout_p = dropout_p
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout_p)
        self.pre_classifier = nn.Linear(embedding_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Average the embedding of all amino acids in a sequence.

        Args:
            token_embeddings (torch.Tensor): Token embeddings from the ESM model (bs, seq_len, embedding_dim).
            attention_mask (torch.Tensor): Attention mask (bs, seq_len).

        Returns:
            torch.Tensor: Mean-pooled embeddings (bs, embedding_dim).
        """

        # expand the mask
        expanded_mask = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.shape).float()
        )

        # sum unmasked token embeddings
        sum_embeddings = torch.sum(token_embeddings * expanded_mask, dim=1)

        # number of unmasked tokens for each sequence
        # set a min value to avoid divide by zero
        num_tokens = torch.clamp(expanded_mask.sum(1), min=1e-9)

        # divide
        mean_embeddings = sum_embeddings / num_tokens
        return mean_embeddings

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            batch (Dict[str, torch.Tensor]): Input batch containing:
                - input_ids (torch.Tensor): Token IDs (bs, seq_len).
                - attention_mask (torch.Tensor): Attention mask (bs, seq_len).

        Returns:
            torch.Tensor: Predicted logits for each class (bs, num_classes).
        """

        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]

        # per token representations from the last layer
        token_embeddings = self.llm(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        # average per token representations
        mean_embeddings = self.mean_pooling(token_embeddings, attention_mask)

        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/distilbert/modeling_distilbert.py
        mean_embeddings = self.pre_classifier(mean_embeddings)  # (bs, embedding_dim)
        mean_embeddings = nn.ReLU()(mean_embeddings)
        mean_embeddings = self.dropout(mean_embeddings)

        logits = self.classifier(mean_embeddings)  # (bs, num_classes)
        return logits


class ESMLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training and evaluating a FinetuneESM model.

    Args:
        model: The FinetuneESM model to be trained.
        learning_rate: The learning rate for the optimizer (default: 1e-3).
        loss_fn: The loss function used for training (default: nn.BCEWithLogitsLoss()).
        score_name: The scoring function name used for evaluation (default: multilabel_f1_score).
    """

    def __init__(
        self,
        model: FinetuneESM,
        learning_rate: float = 1e-3,
        loss_fn: nn.Module = nn.BCEWithLogitsLoss(),
        score_name: Optional[str] = "multilabel_f1_score",
    ):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.score_name = score_name

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            batch: A dictionary containing the input data and targets.

        Returns:
            The logits (unnormalized outputs) of the model.
        """

        return self.model(batch)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Performs a training step, calculating and logging loss.

        Args:
            batch: A dictionary containing the input data and targets.
            batch_idx: The index of the current batch.

        Returns:
            The training loss.
        """

        logits = self(batch)
        loss = self.loss_fn(logits, batch["targets"])
        self.log("train_loss", loss)
        return loss  # this is passed to the optimizer for training

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Performs a validation step, calculating and logging loss and score.

        Args:
            batch: A dictionary containing the input data and targets.
            batch_idx: The index of the current batch.
        """

        logits = self(batch)
        loss = self.loss_fn(logits, batch["targets"])
        self.log("val_loss", loss, prog_bar=True)

        if self.score_name == "multilabel_f1_score":
            f1_score = multilabel_f1_score(
                logits, batch["targets"].type(torch.int), self.model.num_classes
            )
            self.log("val_f1_score", f1_score, prog_bar=True)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Performs a test step, calculating and logging score.

        Args:
            batch: A dictionary containing the input data and targets.
            batch_idx: The index of the current batch.
        """

        logits = self(batch)

        if self.score_name == "multilabel_f1_score":
            f1_score = multilabel_f1_score(
                logits, batch["targets"].type(torch.int), self.model.num_classes
            )
            self.log("f1_score", f1_score, prog_bar=True)

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
            The optimizer instance.
        """

        if self.global_rank == 0:
            print(self.trainer.model)
        optimizer = torch.optim.Adam(
            self.trainer.model.parameters(), lr=self.learning_rate
        )
        return optimizer
