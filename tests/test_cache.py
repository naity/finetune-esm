import numpy as np
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
