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


@pytest.mark.parametrize("key", ["ec", "sc", "hp", "pad"])
def test_runs_under_fp16_autocast(key):
    """Regression: a fixed -1e9 mask fill overflows fp16 and raises under autocast."""
    torch.manual_seed(0)
    model = fusion.FUSIONS[key](DIM).eval()
    batch = _batch()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
        out = model(batch)
    assert torch.isfinite(out).all()


def test_hierarchical_pooling_mixed_chain_counts_per_batch():
    """A row's HP output must not depend on how many chains its batch-mates have.

    ``_partner`` sizes n_chains off the batch-wide max chain id, so a row with
    fewer chains than its batch-mates gets all-False selectors for the extra
    chain slots. That must be a no-op: each row's output should match running
    it alone in a batch of one.
    """
    torch.manual_seed(0)
    model = fusion.HierarchicalPooling(DIM).eval()
    ligand_len, receptor_len = 6, 4

    ligand = torch.randn(2, ligand_len, DIM)
    ligand_mask = torch.ones(2, ligand_len, dtype=torch.bool)
    ligand_chain_id = torch.zeros(2, ligand_len, dtype=torch.long)
    ligand_chain_id[1] = torch.tensor([0, 0, 1, 1, 2, 2])  # row 1: three chains

    receptor = torch.randn(2, receptor_len, DIM)
    receptor_mask = torch.ones(2, receptor_len, dtype=torch.bool)
    receptor_chain_id = torch.zeros(2, receptor_len, dtype=torch.long)

    def _row(i: int) -> EmbeddingBatch:
        return EmbeddingBatch(
            ligand=ligand[i : i + 1],
            ligand_mask=ligand_mask[i : i + 1],
            ligand_chain_id=ligand_chain_id[i : i + 1],
            receptor=receptor[i : i + 1],
            receptor_mask=receptor_mask[i : i + 1],
            receptor_chain_id=receptor_chain_id[i : i + 1],
            target=torch.zeros(1),
            group_id=torch.zeros(1, dtype=torch.long),
        )

    batch = EmbeddingBatch(
        ligand=ligand,
        ligand_mask=ligand_mask,
        ligand_chain_id=ligand_chain_id,
        receptor=receptor,
        receptor_mask=receptor_mask,
        receptor_chain_id=receptor_chain_id,
        target=torch.zeros(2),
        group_id=torch.arange(2),
    )

    with torch.no_grad():
        batched = model(batch)
        alone0 = model(_row(0))
        alone1 = model(_row(1))

    assert torch.allclose(batched[0], alone0[0], atol=1e-4)
    assert torch.allclose(batched[1], alone1[0], atol=1e-4)
