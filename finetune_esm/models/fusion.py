"""Strategies for combining ligand and receptor embeddings into one vector.

EC, SC, HP and PAD follow the definitions in arXiv:2505.20036.
"""

import torch
import torch.nn as nn

from finetune_esm.data.batching import EmbeddingBatch

NEG_INF = -1e9


class AttentionPool(nn.Module):
    """Single-query attention pooling over the residue axis."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(x).squeeze(-1)
        logits = logits.masked_fill(~mask, NEG_INF)
        empty = ~mask.any(dim=-1)
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
        pooled = (x * weights * mask.unsqueeze(-1)).sum(dim=1)
        return pooled.masked_fill(empty.unsqueeze(-1), 0.0)


class FusionBase(nn.Module):
    out_dim: int

    def forward(self, batch: EmbeddingBatch) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class EmbeddingConcat(FusionBase):
    """Pool each partner independently, then concatenate."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.pool = AttentionPool(dim)
        self.out_dim = 2 * dim

    def forward(self, batch: EmbeddingBatch) -> torch.Tensor:
        return torch.cat(
            [
                self.pool(batch.ligand, batch.ligand_mask),
                self.pool(batch.receptor, batch.receptor_mask),
            ],
            dim=-1,
        )


class SequenceConcat(FusionBase):
    """Concatenate both partners along the residue axis, then pool once."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.pool = AttentionPool(dim)
        self.out_dim = dim

    def forward(self, batch: EmbeddingBatch) -> torch.Tensor:
        joined = torch.cat([batch.ligand, batch.receptor], dim=1)
        mask = torch.cat([batch.ligand_mask, batch.receptor_mask], dim=1)
        return self.pool(joined, mask)


class HierarchicalPooling(FusionBase):
    """Pool residues within each chain, then pool across chains, per partner."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.residue_pool = AttentionPool(dim)
        self.chain_pool = AttentionPool(dim)
        self.out_dim = 2 * dim

    def _partner(
        self, x: torch.Tensor, mask: torch.Tensor, chain_id: torch.Tensor
    ) -> torch.Tensor:
        n_chains = int(chain_id.max().item()) + 1 if mask.any() else 1
        chain_vectors, chain_mask = [], []
        for c in range(n_chains):
            selector = mask & (chain_id == c)
            chain_vectors.append(self.residue_pool(x, selector))
            chain_mask.append(selector.any(dim=-1))
        return self.chain_pool(
            torch.stack(chain_vectors, dim=1), torch.stack(chain_mask, dim=1)
        )

    def forward(self, batch: EmbeddingBatch) -> torch.Tensor:
        return torch.cat(
            [
                self._partner(batch.ligand, batch.ligand_mask, batch.ligand_chain_id),
                self._partner(
                    batch.receptor, batch.receptor_mask, batch.receptor_chain_id
                ),
            ],
            dim=-1,
        )


class PooledAttentionAddition(FusionBase):
    """Cross-attend the partners, pool each, then add element-wise."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.cross = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.pool = AttentionPool(dim)
        self.out_dim = dim

    def _attend(self, q, q_mask, kv, kv_mask):
        attended, _ = self.cross(q, kv, kv, key_padding_mask=~kv_mask)
        attended = attended.masked_fill(~q_mask.unsqueeze(-1), 0.0)
        return self.norm(q + attended)

    def forward(self, batch: EmbeddingBatch) -> torch.Tensor:
        ligand = self._attend(
            batch.ligand, batch.ligand_mask, batch.receptor, batch.receptor_mask
        )
        receptor = self._attend(
            batch.receptor, batch.receptor_mask, batch.ligand, batch.ligand_mask
        )
        return self.pool(ligand, batch.ligand_mask) + self.pool(
            receptor, batch.receptor_mask
        )


FUSIONS: dict[str, type] = {
    "ec": EmbeddingConcat,
    "sc": SequenceConcat,
    "hp": HierarchicalPooling,
    "pad": PooledAttentionAddition,
}
