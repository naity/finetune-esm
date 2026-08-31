"""Disk cache of frozen per-residue embeddings, keyed by sequence hash."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from finetune_esm import config
from finetune_esm.models.backbones import Backbone

RESIDUES_FILE = "residues.f16.npy"
POOLED_FILE = "pooled.f16.npy"
INDEX_FILE = "index.parquet"
META_FILE = "meta.json"


def chain_key(seq: str) -> str:
    """SHA-1 of the sequence. Deduplicates repeated chains across complexes."""
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheMeta:
    backbone: str
    hf_id: str
    hidden_dim: int
    max_length: int
    dtype: str
    n_chains: int

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "CacheMeta":
        return cls(**json.loads(path.read_text()))


def build_cache(
    seqs: Iterable[str],
    backbone: Backbone,
    out_dir: Path,
    batch_size: int = 8,
) -> CacheMeta:
    """Encode every distinct sequence once and write the cache to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unique: dict[str, str] = {}
    for seq in seqs:
        unique.setdefault(chain_key(seq), seq)
    keys = sorted(unique)
    ordered = [unique[k] for k in keys]

    dim = backbone.spec.hidden_dim
    lengths = [min(len(s), config.MAX_CHAIN_LENGTH) for s in ordered]
    total = int(sum(lengths))

    residues = np.lib.format.open_memmap(
        out_dir / RESIDUES_FILE, mode="w+", dtype=np.float16, shape=(total, dim)
    )
    pooled = np.zeros((len(ordered), dim), dtype=np.float16)

    offsets, cursor = [], 0
    for start in range(0, len(ordered), batch_size):
        chunk = ordered[start : start + batch_size]
        for arr in backbone.embed(chunk):
            residues[cursor : cursor + len(arr)] = arr.astype(np.float16)
            pooled[len(offsets)] = arr.mean(axis=0).astype(np.float16)
            offsets.append(cursor)
            cursor += len(arr)
    residues.flush()

    np.save(out_dir / POOLED_FILE, pooled)
    pd.DataFrame(
        {"chain_key": keys, "offset": offsets, "length": lengths, "row": range(len(keys))}
    ).to_parquet(out_dir / INDEX_FILE, index=False)

    meta = CacheMeta(
        backbone=backbone.spec.name,
        hf_id=backbone.spec.hf_id,
        hidden_dim=dim,
        max_length=config.MAX_CHAIN_LENGTH,
        dtype="float16",
        n_chains=len(keys),
    )
    meta.to_json(out_dir / META_FILE)
    return meta


class EmbeddingCache:
    """Read-only memmapped view over a built cache."""

    def __init__(self, out_dir: Path, expect: CacheMeta | None = None) -> None:
        out_dir = Path(out_dir)
        self.meta = CacheMeta.from_json(out_dir / META_FILE)
        if expect is not None and expect != self.meta:
            raise ValueError(
                f"cache metadata mismatch: on disk {self.meta}, expected {expect}"
            )
        self._residues = np.load(out_dir / RESIDUES_FILE, mmap_mode="r")
        self._pooled = np.load(out_dir / POOLED_FILE)
        index = pd.read_parquet(out_dir / INDEX_FILE)
        self._index = {
            row.chain_key: (row.offset, row.length, row.row)
            for row in index.itertuples()
        }

    def __contains__(self, seq: str) -> bool:
        return chain_key(seq) in self._index

    def _lookup(self, seq: str) -> tuple[int, int, int]:
        key = chain_key(seq)
        if key not in self._index:
            raise KeyError(f"sequence not in cache (key {key})")
        return self._index[key]

    def residues(self, seq: str) -> np.ndarray:
        offset, length, _ = self._lookup(seq)
        return np.asarray(self._residues[offset : offset + length], dtype=np.float32)

    def pooled(self, seq: str) -> np.ndarray:
        _, _, row = self._lookup(seq)
        return self._pooled[row].astype(np.float32)
