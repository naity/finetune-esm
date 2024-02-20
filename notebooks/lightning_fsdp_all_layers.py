import time
import torch
import pandas as pd
import numpy as np
import datasets
import torch.nn as nn
import lightning as L
import lightning.pytorch as pl

from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, EsmModel
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics.functional.classification import multilabel_f1_score
from lightning.pytorch.strategies import FSDPStrategy
from transformers.models.esm.modeling_esm import EsmLayer

L.seed_everything(0)

# data
data_path = Path("../data/cafa5")
df = pd.read_parquet(data_path / "top100_train_split.parquet")
dataset = datasets.Dataset.from_pandas(df, preserve_index=False)
dataset = dataset.train_test_split(test_size=0.25, seed=0)

# tokenize
model_name = "facebook/esm2_t36_3B_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer input max length:", tokenizer.model_max_length)
print("Tokenizer vocabulary size:", tokenizer.vocab_size)


def tokenize_seqs(batch):
    return tokenizer(
        batch["Sequence"],
        padding="longest",
        truncation=True,
        max_length=min(1024, tokenizer.model_max_length),
    )


tokenized_dataset = dataset.map(tokenize_seqs, batched=True)
tokenized_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "Index"]
)


class ProteinDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.data = dataset[split]

    def __len__(self):
        return self.data.num_rows

    def __getitem__(self, i):
        return self.data[i]


train_dataset = ProteinDataset(tokenized_dataset)
val_dataset = ProteinDataset(tokenized_dataset, split="test")

# targets
targets = np.load(data_path / "train_bp_top100_targets.npy")
targets.shape


def collate_fn(batch):
    batch = tokenizer.pad(batch)
    batch["targets"] = torch.as_tensor(
        targets[batch["Index"].tolist()], dtype=torch.float
    )

    return batch


batch_size = 8
num_workers = 4

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    collate_fn=collate_fn,
)

llm = EsmModel.from_pretrained(model_name)
embedding_dim = llm.config.hidden_size


class FinetunedESM(nn.Module):
    def __init__(self, llm, dropout_p, embedding_dim, num_classes):
        super().__init__()
        self.llm = llm
        self.dropout_p = dropout_p
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout_p)
        self.pre_classifier = nn.Linear(embedding_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def mean_pooling(self, token_embeddings, attention_mask):
        """Average the embedding of all amino acids in a sequence"""

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

    def forward(self, batch):
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


num_classes = 100

model = FinetunedESM(
    llm=llm, dropout_p=0.1, embedding_dim=embedding_dim, num_classes=num_classes
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"# Trainable Parameters: {count_parameters(model)}")


class ESMLightningModule(L.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.loss_fn(logits, batch["targets"])
        self.log("train_loss", loss)
        return loss  # this is passed to the optimizer for training

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.loss_fn(logits, batch["targets"])
        self.log("val_loss", loss, prog_bar=True)

        f1_score = multilabel_f1_score(
            logits, batch["targets"].type(torch.int), num_classes
        )
        self.log("val_f1_score", f1_score, prog_bar=True)

    def test_step(self, batch, batch_idx):
        logits = self(batch)

        f1_score = multilabel_f1_score(
            logits, batch["targets"].type(torch.int), num_classes
        )
        self.log("f1_score", f1_score, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


lightning_model = ESMLightningModule(model, learning_rate=1e-3)

layers = {
    EsmLayer,
}
strategy = FSDPStrategy(
    auto_wrap_policy=layers,
    activation_checkpointing_policy=layers,
)

callbacks = [ModelCheckpoint(save_top_k=1, mode="max", monitor="val_f1_score")]
logger = CSVLogger(save_dir="logs/", name="esm_model_all_fsdp")

trainer = pl.Trainer(
    max_epochs=3,
    callbacks=callbacks,
    accelerator="cuda",
    precision="16-mixed",
    devices=8,
    logger=logger,
    strategy=strategy,
)

start = time.time()

trainer.fit(
    model=lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
)
end = time.time()
trainer.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
print(f"Training Time: {(end-start)/60:.2f} min")
