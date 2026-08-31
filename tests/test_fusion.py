import pytest
import torch

from finetune_esm.data.batching import EmbeddingBatch
from finetune_esm.models import fusion

DIM = 8


def _batch(batch_size=3, ligand_len=5, receptor_len=4) -> EmbeddingBatch:
    torch.manual_seed(0)
    ligand_mask = torch.ones(batch_size, ligand_len, dtype=torch.bool)
    ligand_mask[0, -2:] = False  # exercise padding
    receptor_mask = torch.ones(batch_size, receptor_len, dtype=torch.bool)
    return EmbeddingBatch(
        ligand=torch.randn(batch_size, ligand_len, DIM),
        ligand_mask=ligand_mask,
        ligand_chain_id=torch.zeros(batch_size, ligand_len, dtype=torch.long),
        receptor=torch.randn(batch_size, receptor_len, DIM),
        receptor_mask=receptor_mask,
        receptor_chain_id=torch.zeros(batch_size, receptor_len, dtype=torch.long),
        target=torch.zeros(batch_size),
        group_id=torch.arange(batch_size),
    )


@pytest.mark.parametrize("key", ["ec", "sc", "hp", "pad"])
def test_output_shape_matches_declared_out_dim(key):
    model = fusion.FUSIONS[key](DIM)
    out = model(_batch())
    assert out.shape == (3, model.out_dim)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("key", ["ec", "sc", "hp", "pad"])
def test_padding_does_not_affect_output(key):
    """Garbage in padded positions must not change the result."""
    torch.manual_seed(0)
    model = fusion.FUSIONS[key](DIM).eval()
    clean = _batch()
    dirty = _batch()
    dirty.ligand[~dirty.ligand_mask] = 1e3

    with torch.no_grad():
        assert torch.allclose(model(clean), model(dirty), atol=1e-4)


def test_attention_pool_ignores_masked_positions():
    torch.manual_seed(0)
    pool = fusion.AttentionPool(DIM).eval()
    x = torch.randn(1, 6, DIM)
    mask = torch.tensor([[True, True, True, False, False, False]])
    with torch.no_grad():
        full = pool(x, mask)
        trimmed = pool(x[:, :3], mask[:, :3])
    assert torch.allclose(full, trimmed, atol=1e-5)


def test_hierarchical_pooling_uses_chain_structure():
    """Splitting one partner into two chains must change HP's output."""
    torch.manual_seed(0)
    model = fusion.HierarchicalPooling(DIM).eval()
    one_chain = _batch(batch_size=1, ligand_len=4)
    one_chain.ligand_mask[:] = True
    two_chains = _batch(batch_size=1, ligand_len=4)
    two_chains.ligand_mask[:] = True
    two_chains.ligand = one_chain.ligand.clone()
    two_chains.receptor = one_chain.receptor.clone()
    two_chains.ligand_chain_id[0, 2:] = 1

    with torch.no_grad():
        assert not torch.allclose(model(one_chain), model(two_chains), atol=1e-4)


def test_registry_is_complete():
    assert set(fusion.FUSIONS) == {"ec", "sc", "hp", "pad"}
