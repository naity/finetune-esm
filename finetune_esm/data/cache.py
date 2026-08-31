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

# Fields that identify *which backbone/config* produced a cache, as opposed
# to how many chains happen to be in it. Used by `EmbeddingCache`'s `expect`
# guard so a caller can validate against a cache without first knowing its
# chain count.
_IDENTITY_FIELDS = ("backbone", "hf_id", "hidden_dim", "max_length", "dtype")


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

    def identity(self) -> tuple:
        """The subset of fields that identify the backbone/config that
        produced a cache, excluding `n_chains`. Two `CacheMeta` with equal
        `identity()` describe caches built with the same backbone, hf_id,
        hidden_dim, max_length, and storage dtype -- regardless of how many
        distinct chains they happen to contain.
        """
        return tuple(getattr(self, f) for f in _IDENTITY_FIELDS)


def build_cache(
    seqs: Iterable[str],
    backbone: Backbone,
    out_dir: Path,
    batch_size: int = 8,
) -> CacheMeta:
    """Encode every distinct sequence once and write the cache to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # meta.json is the completion marker (written last, below). Remove any
    # existing one up front so a rebuild that dies partway through leaves
    # the directory with no valid meta.json, rather than a stale one that
    # lies about freshly (and possibly only partially) rewritten data.
    (out_dir / META_FILE).unlink(missing_ok=True)

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

        if expect is not None and expect.identity() != self.meta.identity():
            diffs = [
                f"{field}: expected {getattr(expect, field)!r}, "
                f"got {getattr(self.meta, field)!r}"
                for field in _IDENTITY_FIELDS
                if getattr(expect, field) != getattr(self.meta, field)
            ]
            raise ValueError(
                f"cache metadata mismatch at {out_dir}: " + "; ".join(diffs)
            )

        self._residues = np.load(out_dir / RESIDUES_FILE, mmap_mode="r")
        self._pooled = np.load(out_dir / POOLED_FILE)
        index = pd.read_parquet(out_dir / INDEX_FILE)

        # Always-on internal consistency checks: a cache that loads without
        # error must have a residues memmap, pooled array, and index that
        # all genuinely agree with meta.json and with each other. This is
        # what stops a stale/corrupt index from silently truncating a
        # slice into a short, plausible-looking, wrong-region read.
        if self._residues.shape[1] != self.meta.hidden_dim:
            raise ValueError(
                f"cache at {out_dir}: residues.shape[1]={self._residues.shape[1]} "
                f"!= meta.hidden_dim={self.meta.hidden_dim}"
            )
        if self._pooled.shape != (self.meta.n_chains, self.meta.hidden_dim):
            raise ValueError(
                f"cache at {out_dir}: pooled.shape={self._pooled.shape} != "
                f"(meta.n_chains, meta.hidden_dim)="
                f"({self.meta.n_chains}, {self.meta.hidden_dim})"
            )
        if len(index) != self.meta.n_chains:
            raise ValueError(
                f"cache at {out_dir}: index has {len(index)} rows != "
                f"meta.n_chains={self.meta.n_chains}"
            )
        if len(index) > 0:
            if (index["offset"] < 0).any() or (index["length"] < 1).any():
                raise ValueError(
                    f"cache at {out_dir}: index contains a negative offset "
                    "or a non-positive length"
                )
            max_end = int((index["offset"] + index["length"]).max())
            if max_end > self._residues.shape[0]:
                raise ValueError(
                    f"cache at {out_dir}: index reaches offset+length="
                    f"{max_end} but residues has only "
                    f"{self._residues.shape[0]} rows"
                )

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
