import numpy as np
import pandas as pd
import pytest

from finetune_esm.data import cache
from finetune_esm.models import backbones

SEQS = ["MKTAYIAKQR", "MVLSPADKTNVKAAW", "GGGGSGGGGS"]


def test_chain_key_is_stable_and_distinct():
    assert cache.chain_key("ABC") == cache.chain_key("ABC")
    assert cache.chain_key("ABC") != cache.chain_key("ABD")
    assert len(cache.chain_key("ABC")) == 40


@pytest.mark.slow
def test_cache_round_trips_and_matches_live_forward_pass(tmp_path):
    bb = backbones.Backbone(backbones.BACKBONES["esm2-8m"])
    meta = cache.build_cache(SEQS, bb, tmp_path)
    store = cache.EmbeddingCache(tmp_path, expect=meta)

    live = bb.embed(SEQS)
    for seq, expected in zip(SEQS, live):
        got = store.residues(seq)
        assert got.shape == expected.shape
        # fp16 storage: compare at half-precision tolerance
        assert np.allclose(got, expected, atol=1e-2)


@pytest.mark.slow
def test_pooled_equals_mean_over_residues(tmp_path):
    bb = backbones.Backbone(backbones.BACKBONES["esm2-8m"])
    meta = cache.build_cache(SEQS, bb, tmp_path)
    store = cache.EmbeddingCache(tmp_path, expect=meta)
    for seq in SEQS:
        assert np.allclose(
            store.pooled(seq), store.residues(seq).mean(axis=0), atol=1e-2
        )


@pytest.mark.slow
def test_duplicate_sequences_are_encoded_once(tmp_path):
    bb = backbones.Backbone(backbones.BACKBONES["esm2-8m"])
    meta = cache.build_cache(SEQS + SEQS, bb, tmp_path)
    assert meta.n_chains == len(SEQS)


@pytest.mark.slow
def test_mismatched_metadata_is_rejected(tmp_path):
    bb = backbones.Backbone(backbones.BACKBONES["esm2-8m"])
    meta = cache.build_cache(SEQS, bb, tmp_path)
    wrong = cache.CacheMeta(
        backbone="esm2-650m",
        hf_id="facebook/esm2_t33_650M_UR50D",
        hidden_dim=1280,
        max_length=meta.max_length,
        dtype=meta.dtype,
        n_chains=meta.n_chains,
    )
    with pytest.raises(ValueError, match="mismatch"):
        cache.EmbeddingCache(tmp_path, expect=wrong)


@pytest.mark.slow
def test_unknown_sequence_raises(tmp_path):
    bb = backbones.Backbone(backbones.BACKBONES["esm2-8m"])
    meta = cache.build_cache(SEQS, bb, tmp_path)
    store = cache.EmbeddingCache(tmp_path, expect=meta)
    assert "MKTAYIAKQR" in store
    assert "WWWWWWWW" not in store
    with pytest.raises(KeyError):
        store.residues("WWWWWWWW")


@pytest.mark.slow
def test_corrupt_index_is_rejected(tmp_path):
    """A stale/corrupt index whose offset+length runs past the residues
    memmap must be rejected at construction time, not silently truncated
    into a short, plausible-looking read from the wrong region.
    """
    bb = backbones.Backbone(backbones.BACKBONES["esm2-8m"])
    cache.build_cache(SEQS, bb, tmp_path)

    residues = np.load(tmp_path / cache.RESIDUES_FILE, mmap_mode="r")
    index = pd.read_parquet(tmp_path / cache.INDEX_FILE)
    # Push one row's offset+length past the end of the residues memmap.
    index.loc[0, "offset"] = residues.shape[0] + 100
    index.loc[0, "length"] = 5
    index.to_parquet(tmp_path / cache.INDEX_FILE, index=False)

    with pytest.raises(ValueError):
        cache.EmbeddingCache(tmp_path)


@pytest.mark.slow
def test_expect_ignores_n_chains(tmp_path):
    """`expect` validates identity (backbone/hf_id/hidden_dim/max_length/
    dtype), not how many chains happen to be cached, so a caller can build
    an expectation without first knowing the cache's chain count.
    """
    bb = backbones.Backbone(backbones.BACKBONES["esm2-8m"])
    meta = cache.build_cache(SEQS, bb, tmp_path)
    expect = cache.CacheMeta(
        backbone=meta.backbone,
        hf_id=meta.hf_id,
        hidden_dim=meta.hidden_dim,
        max_length=meta.max_length,
        dtype=meta.dtype,
        n_chains=meta.n_chains + 999,
    )
    store = cache.EmbeddingCache(tmp_path, expect=expect)
    assert store.meta.n_chains == meta.n_chains


@pytest.mark.slow
def test_crashed_rebuild_leaves_no_valid_meta(tmp_path, monkeypatch):
    """A rebuild that dies after index.parquet is rewritten but before
    meta.json is rewritten must not leave the OLD meta.json in place
    describing the NEW (possibly incomplete) data files. meta.json is
    deleted up front, so a crashed rebuild is indistinguishable from a
    crashed fresh build: both fail loudly on the next load.
    """
    bb = backbones.Backbone(backbones.BACKBONES["esm2-8m"])
    cache.build_cache(SEQS, bb, tmp_path)  # a valid cache exists first

    def boom(self, path):
        raise RuntimeError("simulated crash before meta.json write")

    monkeypatch.setattr(cache.CacheMeta, "to_json", boom)
    with pytest.raises(RuntimeError):
        cache.build_cache(SEQS, bb, tmp_path)

    assert not (tmp_path / cache.META_FILE).exists()
    with pytest.raises(FileNotFoundError):
        cache.EmbeddingCache(tmp_path)
