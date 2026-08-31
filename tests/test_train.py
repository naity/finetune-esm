import math

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from finetune_esm import config, train
from finetune_esm.data import batching, cache
from finetune_esm.models import backbones


class FakeStore:
    def residues(self, seq: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(seq)) % (2**32))
        return rng.normal(size=(len(seq), 8)).astype(np.float32)


def _df(n=64) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "pdb": [f"P{i // 2}" for i in range(n)],
            "ligand_seqs": [["MK" * (1 + i % 3)] for i in range(n)],
            "receptor_seqs": [["GG" * (1 + i % 2)] for i in range(n)],
            "pkd": rng.normal(7.0, 2.0, size=n),
        }
    )


def test_total_steps_uses_loader_length_not_row_count():
    loader = DataLoader(
        batching.ComplexDataset(_df(64), np.arange(64), FakeStore()),
        batch_size=8,
        collate_fn=batching.collate,
    )
    assert train.compute_total_steps(loader, num_epochs=3) == 3 * math.ceil(64 / 8)


def test_total_steps_has_a_safe_floor():
    # OneCycleLR(total_steps=2) raises ZeroDivisionError internally (its
    # step_size_up computes to 0); 1, 3 and 5 are fine. A 1-epoch run over
    # 2 batches produces exactly the failing value, so the floor must be 3.
    loader = DataLoader(
        batching.ComplexDataset(_df(2), np.arange(2), FakeStore()),
        batch_size=8,
        collate_fn=batching.collate,
    )
    assert train.compute_total_steps(loader, num_epochs=1) >= 3


def test_end_to_end_training_returns_all_three_metrics():
    out = train.run_training(
        df=_df(64),
        store=FakeStore(),
        hidden_dim=8,
        fusion_name="hp",
        split_name="cold_pdb",
        seed=0,
        num_epochs=2,
        batch_size=8,
        accelerator="cpu",
    )
    assert set(out) == {"spearman", "pearson", "rmse"}
    assert np.isfinite(out["rmse"])


def test_split_choice_is_honoured_and_leak_free():
    df = _df(64)
    idx = train.resolve_split(df, "cold_pdb", seed=0)
    from finetune_esm.data import splits

    splits.assert_no_pdb_leakage(df, idx)


def test_unknown_split_name_raises():
    with pytest.raises(KeyError, match="cold_pdb"):
        train.resolve_split(_df(8), "not_a_split", seed=0)


def test_cache_expectation_rejects_mismatched_backbone(tmp_path):
    """`EmbeddingCache`'s `expect` guard must catch loading a cache built
    with a different backbone than the one being trained against -- e.g.
    reading a 1280-dim ESM2-650M cache while expecting a 320-dim ESM2-8M
    cache. `CacheMeta.identity()` excludes `n_chains`, so an `expect` built
    without knowing the chain count is still a meaningful comparison.
    """
    built_spec = backbones.BACKBONES["esm2-8m"]
    other_spec = backbones.BACKBONES["esm2-650m"]

    meta = cache.CacheMeta(
        backbone=built_spec.name,
        hf_id=built_spec.hf_id,
        hidden_dim=built_spec.hidden_dim,
        max_length=config.MAX_CHAIN_LENGTH,
        dtype="float16",
        n_chains=2,
    )
    out_dir = tmp_path / "cache"
    out_dir.mkdir()
    np.lib.format.open_memmap(
        out_dir / cache.RESIDUES_FILE,
        mode="w+",
        dtype=np.float16,
        shape=(4, built_spec.hidden_dim),
    ).flush()
    np.save(
        out_dir / cache.POOLED_FILE,
        np.zeros((2, built_spec.hidden_dim), dtype=np.float16),
    )
    pd.DataFrame(
        {"chain_key": ["a", "b"], "offset": [0, 2], "length": [2, 2], "row": [0, 1]}
    ).to_parquet(out_dir / cache.INDEX_FILE, index=False)
    meta.to_json(out_dir / cache.META_FILE)

    expect = cache.CacheMeta(
        backbone=other_spec.name,
        hf_id=other_spec.hf_id,
        hidden_dim=other_spec.hidden_dim,
        max_length=config.MAX_CHAIN_LENGTH,
        dtype="float16",
        n_chains=-1,
    )
    with pytest.raises(ValueError, match="cache metadata mismatch"):
        cache.EmbeddingCache(out_dir, expect=expect)
