"""Command line entrypoints: build the embedding cache, then train."""

import math
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import typer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from finetune_esm import config
from finetune_esm.data import batching, cache, dataset, splits
from finetune_esm.models import backbones, fusion, heads, module

app = typer.Typer(add_completion=False)

SPLIT_BUILDERS = {
    "author": lambda df, seed: splits.author_split(df),
    "cold_pdb": lambda df, seed: splits.cold_pdb_split(df, seed=seed),
    "random": lambda df, seed: splits.random_split(df, seed=seed),
}


def resolve_split(df: pd.DataFrame, split_name: str, seed: int) -> dict[str, np.ndarray]:
    if split_name not in SPLIT_BUILDERS:
        raise KeyError(
            f"unknown split {split_name!r}; available: {sorted(SPLIT_BUILDERS)}"
        )
    return SPLIT_BUILDERS[split_name](df, seed)


def compute_total_steps(loader: DataLoader, num_epochs: int) -> int:
    """Steps the scheduler will actually take, derived from the loader itself.

    Floored at 3, not 1: `OneCycleLR(total_steps=...)` raises
    `ZeroDivisionError` internally when `total_steps` is 2 (its
    `step_size_up` computes to 0); 1, 3 and 5 are all fine. A 1-epoch run
    over 2 batches produces exactly the failing value of 2, so the floor
    has to skip it.
    """
    return max(3, num_epochs * math.ceil(len(loader.dataset) / loader.batch_size))


def _loader(df, idx, store, batch_size, shuffle):
    return DataLoader(
        batching.ComplexDataset(df, idx, store),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=batching.collate,
    )


def run_training(
    df: pd.DataFrame,
    store,
    hidden_dim: int,
    fusion_name: str = "hp",
    split_name: str = "author",
    seed: int = 0,
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    dropout: float = 0.1,
    loss_name: str = "mse",
    patience: int = 5,
    accelerator: str = "auto",
) -> dict[str, float]:
    """Train one configuration and return metrics on its test split."""
    pl.seed_everything(seed, workers=True)
    idx = resolve_split(df, split_name, seed)
    splits.assert_no_pdb_leakage(df, idx)

    train_loader = _loader(df, idx["train"], store, batch_size, shuffle=True)
    val_loader = _loader(df, idx["val"], store, batch_size, shuffle=False)
    test_loader = _loader(df, idx["test"], store, batch_size, shuffle=False)

    fuse = fusion.FUSIONS[fusion_name](hidden_dim)
    model = module.AffinityModule(
        fuse,
        heads.MLPHead(fuse.out_dim, dropout=dropout),
        loss_name=loss_name,
        learning_rate=learning_rate,
        total_steps=compute_total_steps(train_loader, num_epochs),
    )

    checkpoint = ModelCheckpoint(monitor="val_spearman", mode="max", save_top_k=1)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator=accelerator,
        logger=False,
        enable_progress_bar=False,
        callbacks=[
            checkpoint,
            # check_finite=False: metrics.spearman deliberately returns NaN
            # when predictions are constant, a normal state early in
            # training. EarlyStopping's default check_finite=True treats a
            # non-finite monitored value as an immediate stop signal
            # (should_stop=True on the very first NaN, ignoring patience
            # entirely), which would halt a run after one epoch and report
            # a garbage "best" checkpoint. ModelCheckpoint needs no such
            # override -- it substitutes +/-inf for NaN internally.
            EarlyStopping(
                monitor="val_spearman",
                mode="max",
                patience=patience,
                check_finite=False,
            ),
        ],
    )
    trainer.fit(model, train_loader, val_loader)
    (results,) = trainer.test(model, test_loader, ckpt_path=checkpoint.best_model_path)
    return {
        "spearman": results["test_spearman"],
        "pearson": results["test_pearson"],
        "rmse": results["test_rmse"],
    }


@app.command("build-cache")
def build_cache_command(
    backbone: str = typer.Option("esm2-650m", help="Key in models.backbones.BACKBONES"),
    device: str = typer.Option("cuda", help="Device for the encoder forward pass"),
    batch_size: int = typer.Option(8),
) -> Path:
    """Encode every unique chain in the dataset once."""
    df = dataset.load_complexes()
    seqs = [s for row in df["ligand_seqs"] for s in row]
    seqs += [s for row in df["receptor_seqs"] for s in row]

    spec = backbones.BACKBONES[backbone]
    encoder = backbones.Backbone(spec, device=device, dtype=torch.float16)
    out_dir = config.CACHE_DIR / backbone
    meta = cache.build_cache(seqs, encoder, out_dir, batch_size=batch_size)
    typer.echo(f"cached {meta.n_chains} unique chains to {out_dir}")
    return out_dir


@app.command("train")
def train_command(
    backbone: str = typer.Option("esm2-650m"),
    fusion_name: str = typer.Option("hp", "--fusion"),
    split_name: str = typer.Option("author", "--split"),
    seed: int = typer.Option(0),
    num_epochs: int = typer.Option(20),
    batch_size: int = typer.Option(32),
    learning_rate: float = typer.Option(1e-3),
) -> None:
    """Train one configuration on the cached embeddings."""
    df = dataset.load_complexes()
    spec = backbones.BACKBONES[backbone]
    # Arm the cache's metadata guard: `expect` catches loading a cache that
    # was built with a *different* backbone than the one being trained
    # (e.g. a 1280-dim ESM2-650M cache read while sizing a fusion module
    # for 320-dim ESM2-8M). `CacheMeta.identity()` excludes `n_chains`, so
    # it can be built from `spec` alone without knowing the chain count;
    # the value passed here for `n_chains` is never compared.
    expect = cache.CacheMeta(
        backbone=spec.name,
        hf_id=spec.hf_id,
        hidden_dim=spec.hidden_dim,
        max_length=config.MAX_CHAIN_LENGTH,
        dtype="float16",
        n_chains=-1,
    )
    store = cache.EmbeddingCache(config.CACHE_DIR / backbone, expect=expect)
    out = run_training(
        df=df,
        store=store,
        hidden_dim=spec.hidden_dim,
        fusion_name=fusion_name,
        split_name=split_name,
        seed=seed,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    typer.echo(
        f"{backbone}/{fusion_name}/{split_name} seed={seed}: "
        f"spearman={out['spearman']:.4f} pearson={out['pearson']:.4f} "
        f"rmse={out['rmse']:.4f}"
    )


if __name__ == "__main__":
    app()
