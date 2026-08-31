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


def _check_special_token_layout(tokenizer, spec: BackboneSpec) -> None:
    """Verify the tokenizer wraps sequences with exactly one leading and one
    trailing special token, as `Backbone.embed()` assumes when it slices
    `hidden[i, 1 : len(seq) + 1]`.
    """
    probe = tokenizer("MKTAYIAK", return_special_tokens_mask=True)
    mask = probe["special_tokens_mask"]
    layout_ok = (
        len(mask) >= 2
        and mask[0] == 1
        and mask[-1] == 1
        and all(m == 0 for m in mask[1:-1])
    )
    if not layout_ok:
        raise ValueError(
            f"Backbone {spec.name!r} ({spec.hf_id!r}) has a special-token "
            f"layout {mask} that is not a single leading and single "
            "trailing special token. embed() assumes a single leading and "
            "single trailing special token, and this tokenizer does not "
            "match, so the slicing in embed() needs generalising before "
            "this backbone can be used."
        )


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
        _check_special_token_layout(self.tokenizer, spec)
        self.model = AutoModel.from_pretrained(spec.hf_id, dtype=dtype)
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
