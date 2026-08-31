"""Protein language model backbones behind one interface."""

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from finetune_esm import config


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    hf_id: str
    hidden_dim: int


BACKBONES: dict[str, BackboneSpec] = {
    "esm2-8m": BackboneSpec("esm2-8m", "facebook/esm2_t6_8M_UR50D", 320),
    "esm2-35m": BackboneSpec("esm2-35m", "facebook/esm2_t12_35M_UR50D", 480),
    "esm2-150m": BackboneSpec("esm2-150m", "facebook/esm2_t30_150M_UR50D", 640),
    "esm2-650m": BackboneSpec("esm2-650m", "facebook/esm2_t33_650M_UR50D", 1280),
}


class Backbone:
    """Wraps a frozen PLM and returns per-residue embeddings."""

    def __init__(
        self,
        spec: BackboneSpec,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.spec = spec
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        self.model = AutoModel.from_pretrained(spec.hf_id, torch_dtype=dtype)
        self.model.eval().to(device)
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def embed(self, seqs: list[str]) -> list[np.ndarray]:
        """Per-residue embeddings, special tokens stripped, one array per sequence."""
        truncated = [s[: config.MAX_CHAIN_LENGTH] for s in seqs]
        encoded = self.tokenizer(
            truncated,
            padding=True,
            truncation=True,
            max_length=config.MAX_CHAIN_LENGTH + 2,  # room for CLS and EOS
            return_tensors="pt",
        ).to(self.device)

        hidden = self.model(**encoded).last_hidden_state.float().cpu().numpy()

        out = []
        for i, seq in enumerate(truncated):
            # ESM prepends CLS and appends EOS; residue i lives at position i + 1.
            out.append(hidden[i, 1 : len(seq) + 1].astype(np.float32))
        return out
