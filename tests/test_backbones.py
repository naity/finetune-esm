import numpy as np
import pytest

from finetune_esm import config
from finetune_esm.models import backbones


def test_registry_contains_expected_specs():
    assert backbones.BACKBONES["esm2-8m"].hf_id == "facebook/esm2_t6_8M_UR50D"
    assert backbones.BACKBONES["esm2-650m"].hidden_dim == 1280


@pytest.mark.slow
def test_embed_returns_one_array_per_sequence_without_special_tokens():
    bb = backbones.Backbone(backbones.BACKBONES["esm2-8m"])
    seqs = ["MKTAYIAKQR", "MVLSPADKTNVKAAW"]
    out = bb.embed(seqs)
    assert len(out) == 2
    for seq, arr in zip(seqs, out):
        assert arr.shape == (len(seq), bb.spec.hidden_dim)
        assert arr.dtype == np.float32
        assert np.isfinite(arr).all()


@pytest.mark.slow
def test_embed_truncates_at_max_chain_length():
    bb = backbones.Backbone(backbones.BACKBONES["esm2-8m"])
    long_seq = "A" * (config.MAX_CHAIN_LENGTH + 500)
    (arr,) = bb.embed([long_seq])
    assert arr.shape[0] == config.MAX_CHAIN_LENGTH


@pytest.mark.slow
def test_embedding_is_independent_of_batch_composition():
    """A chain's embedding must not depend on what it was batched with."""
    bb = backbones.Backbone(backbones.BACKBONES["esm2-8m"])
    target = "MKTAYIAKQRQISFVK"
    (alone,) = bb.embed([target])
    together = bb.embed(["MVLSPADKTNVKAAWGKVGAHAGEYGAEAL", target])[1]
    assert np.allclose(alone, together, atol=1e-4)
