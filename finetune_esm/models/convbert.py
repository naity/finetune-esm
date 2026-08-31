"""A ConvBERT block over per-residue embeddings, as the benchmark paper used.

Spec §5.3 called for "MLP and ConvBERT-style heads"; only the MLP was built, which
left the E0 reproduction gap (0.4284 against a published 0.47) attributed to a
control that did not exist. This supplies it.

Placement matters. In the Ankh/ProtT5 downstream lineage the benchmark paper follows,
a ConvBERT head operates on the *residue sequence* produced by the frozen PLM and is
followed by pooling — it is not a head in our sense of "maps a pooled vector to a
scalar". So this sits between the embedding cache and the fusion module, transforming
per-residue features that the fusion then pools.

It also down-projects (1280 → 256 by default), following the same convention. That
happens to make the resulting model *smaller* than the plain-MLP configuration, which
the final review flagged as over-parameterised at ~6.6M parameters on 6,485 training
rows.
"""

from dataclasses import replace

import torch
import torch.nn as nn
from transformers import ConvBertConfig
from transformers.models.convbert.modeling_convbert import ConvBertLayer

from finetune_esm.data.batching import EmbeddingBatch
from finetune_esm.models.fusion import FUSIONS, FusionBase


class ConvBertResidueEncoder(nn.Module):
    """Down-project per-residue embeddings, then apply one ConvBERT layer."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.project = nn.Linear(in_dim, hidden_dim)
        self.layer = ConvBertLayer(
            ConvBertConfig(
                hidden_size=hidden_dim,
                num_attention_heads=num_heads,
                intermediate_size=hidden_dim * 2,
                conv_kernel_size=kernel_size,
                head_ratio=2,
                num_hidden_layers=1,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
            )
        )
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, L, in_dim) and a bool (B, L) mask -> (B, L, hidden_dim)."""
        hidden = self.project(x)
        # Zero the padded positions BEFORE the layer. The attention mask below
        # suppresses attention, but ConvBERT's span-based convolution mixes
        # neighbouring positions regardless of it, so with a kernel of 7 a
        # padded position contaminates up to three real residues on either
        # side. Zeroing makes the convolution see ordinary zero-padding.
        hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0.0)
        # ConvBERT wants an additive (B, 1, 1, L) mask: 0 to keep, very negative
        # to suppress. Use the dtype's own minimum so this survives fp16 autocast,
        # for the same reason fusion.AttentionPool does.
        additive = torch.zeros(
            mask.shape[0], 1, 1, mask.shape[1], dtype=hidden.dtype, device=hidden.device
        )
        additive = additive.masked_fill(
            ~mask[:, None, None, :], torch.finfo(hidden.dtype).min
        )
        out = self.layer(hidden, attention_mask=additive)
        if isinstance(out, tuple):  # older transformers returned a tuple
            out = out[0]
        # Zero the padded positions so nothing downstream can read them.
        return out.masked_fill(~mask.unsqueeze(-1), 0.0)


class ConvBertFusion(FusionBase):
    """Apply a shared ConvBERT encoder to both partners, then an inner fusion.

    The encoder is shared between ligand and receptor, matching the paper's
    "repeat for receptor" phrasing for the pooling stages.
    """

    def __init__(
        self,
        dim: int,
        inner: str = "hp",
        hidden_dim: int = 256,
        num_heads: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = ConvBertResidueEncoder(
            dim, hidden_dim=hidden_dim, num_heads=num_heads,
            kernel_size=kernel_size, dropout=dropout,
        )
        self.inner = FUSIONS[inner](self.encoder.out_dim)
        self.out_dim = self.inner.out_dim

    def forward(self, batch: EmbeddingBatch) -> torch.Tensor:
        encoded = replace(
            batch,
            ligand=self.encoder(batch.ligand, batch.ligand_mask),
            receptor=self.encoder(batch.receptor, batch.receptor_mask),
        )
        return self.inner(encoded)
