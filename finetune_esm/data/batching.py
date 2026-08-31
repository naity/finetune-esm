"""Dataset and collation over cached embeddings."""

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class EmbeddingBatch:
    ligand: torch.Tensor
    ligand_mask: torch.Tensor
    ligand_chain_id: torch.Tensor
    receptor: torch.Tensor
    receptor_mask: torch.Tensor
    receptor_chain_id: torch.Tensor
    target: torch.Tensor
    group_id: torch.Tensor

    def to(self, device) -> "EmbeddingBatch":
        return EmbeddingBatch(
            **{f.name: getattr(self, f.name).to(device) for f in fields(self)}
        )


class ComplexDataset(Dataset):
    """One item per complex; embeddings are read from the cache on access."""

    def __init__(self, df: pd.DataFrame, indices: np.ndarray, store) -> None:
        self.rows = df.iloc[indices].reset_index(drop=True)
        self.store = store
        self.group_codes = pd.Categorical(self.rows["pdb"]).codes
        """Integer codes for `self.rows["pdb"]`, assigned by `pd.Categorical`
        over just this dataset instance's rows. Scoped to this instance only
        -- not comparable across splits/instances (e.g. group_id 0 in a
        train-split dataset and group_id 0 in a val-split dataset need not
        be the same PDB entry)."""

    def __len__(self) -> int:
        return len(self.rows)

    def _stack(self, seqs: list[str]) -> tuple[np.ndarray, np.ndarray]:
        parts = [self.store.residues(s) for s in seqs]
        chain_ids = np.concatenate(
            [np.full(len(p), i, dtype=np.int64) for i, p in enumerate(parts)]
        )
        return np.concatenate(parts, axis=0), chain_ids

    def __getitem__(self, i: int) -> dict:
        row = self.rows.iloc[i]
        ligand, ligand_ids = self._stack(row["ligand_seqs"])
        receptor, receptor_ids = self._stack(row["receptor_seqs"])
        return {
            "ligand": ligand,
            "ligand_chain_id": ligand_ids,
            "receptor": receptor,
            "receptor_chain_id": receptor_ids,
            "target": float(row["pkd"]),
            "group_id": int(self.group_codes[i]),
        }


def _pad(arrays: list[np.ndarray], ids: list[np.ndarray]):
    batch, dim = len(arrays), arrays[0].shape[1]
    longest = max(a.shape[0] for a in arrays)

    padded = torch.zeros(batch, longest, dim, dtype=torch.float32)
    mask = torch.zeros(batch, longest, dtype=torch.bool)
    chain_id = torch.full((batch, longest), -1, dtype=torch.long)

    for i, (arr, cid) in enumerate(zip(arrays, ids)):
        n = arr.shape[0]
        padded[i, :n] = torch.from_numpy(arr)
        mask[i, :n] = True
        chain_id[i, :n] = torch.from_numpy(cid)
    return padded, mask, chain_id


def collate(items: list[dict]) -> EmbeddingBatch:
    ligand, ligand_mask, ligand_chain_id = _pad(
        [it["ligand"] for it in items], [it["ligand_chain_id"] for it in items]
    )
    receptor, receptor_mask, receptor_chain_id = _pad(
        [it["receptor"] for it in items], [it["receptor_chain_id"] for it in items]
    )
    return EmbeddingBatch(
        ligand=ligand,
        ligand_mask=ligand_mask,
        ligand_chain_id=ligand_chain_id,
        receptor=receptor,
        receptor_mask=receptor_mask,
        receptor_chain_id=receptor_chain_id,
        target=torch.tensor([it["target"] for it in items], dtype=torch.float32),
        group_id=torch.tensor([it["group_id"] for it in items], dtype=torch.long),
    )
