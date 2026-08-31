import numpy as np
import pandas as pd
import torch

from finetune_esm.data import batching


class FakeStore:
    """Deterministic stand-in for EmbeddingCache: embedding = index * ones(4)."""

    def residues(self, seq: str) -> np.ndarray:
        return np.full((len(seq), 4), float(len(seq)), dtype=np.float32)


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pdb": ["A", "B"],
            "ligand_seqs": [["MK"], ["MKT", "GG"]],
            "receptor_seqs": [["MKTA"], ["M"]],
            "pkd": [7.0, 9.0],
        }
    )


def test_collate_pads_to_longest_and_masks_correctly():
    ds = batching.ComplexDataset(_df(), np.array([0, 1]), FakeStore())
    batch = batching.collate([ds[0], ds[1]])

    assert batch.ligand.shape == (2, 5, 4)          # row 1 has 3 + 2 = 5 residues
    assert batch.ligand_mask[0].sum().item() == 2
    assert batch.ligand_mask[1].sum().item() == 5
    assert batch.receptor.shape == (2, 4, 4)
    assert batch.receptor_mask[0].sum().item() == 4
    assert batch.receptor_mask[1].sum().item() == 1


def test_padded_positions_are_zero_and_chain_id_is_negative_one():
    ds = batching.ComplexDataset(_df(), np.array([0, 1]), FakeStore())
    batch = batching.collate([ds[0], ds[1]])
    assert torch.all(batch.ligand[0, 2:] == 0)
    assert torch.all(batch.ligand_chain_id[0, 2:] == -1)


def test_chain_ids_distinguish_chains_within_a_partner():
    ds = batching.ComplexDataset(_df(), np.array([1]), FakeStore())
    batch = batching.collate([ds[0]])
    ids = batch.ligand_chain_id[0][batch.ligand_mask[0]]
    assert ids.tolist() == [0, 0, 0, 1, 1]


def test_targets_and_groups_are_carried_through():
    ds = batching.ComplexDataset(_df(), np.array([0, 1]), FakeStore())
    batch = batching.collate([ds[0], ds[1]])
    assert torch.allclose(batch.target, torch.tensor([7.0, 9.0]))
    assert batch.group_id.tolist() == [0, 1]  # distinct PDBs get distinct ids


def test_to_device_is_a_noop_on_cpu():
    ds = batching.ComplexDataset(_df(), np.array([0]), FakeStore())
    batch = batching.collate([ds[0]]).to("cpu")
    assert batch.ligand.device.type == "cpu"
