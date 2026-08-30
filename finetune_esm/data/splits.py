"""Construct and validate train/val/test splits."""

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "val", "test")


def assert_no_pdb_leakage(df: pd.DataFrame, splits: dict[str, np.ndarray]) -> None:
    """Raise if any PDB entry appears in more than one split."""
    seen: dict[str, str] = {}
    for name, idx in splits.items():
        for pdb in df.iloc[idx]["pdb"].unique():
            if pdb in seen and seen[pdb] != name:
                raise AssertionError(
                    f"PDB {pdb} appears in both {seen[pdb]!r} and {name!r}"
                )
            seen[pdb] = name


def author_split(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """The split shipped in the dataset's own `split` column."""
    return {
        name: np.flatnonzero((df["split"] == name).to_numpy())
        for name in SPLIT_NAMES
    }


def _assign_groups(
    groups: np.ndarray, seed: int, fracs: tuple[float, float, float]
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rng.shuffle(unique)
    n_train = int(round(fracs[0] * len(unique)))
    n_val = int(round(fracs[1] * len(unique)))
    chunks = {
        "train": set(unique[:n_train]),
        "val": set(unique[n_train : n_train + n_val]),
        "test": set(unique[n_train + n_val :]),
    }
    return {
        name: np.flatnonzero(np.isin(groups, list(members)))
        for name, members in chunks.items()
    }


def cold_pdb_split(
    df: pd.DataFrame, seed: int, fracs: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> dict[str, np.ndarray]:
    """Split so that a PDB entry never spans two splits."""
    return _assign_groups(df["pdb"].to_numpy(), seed, fracs)


def random_split(
    df: pd.DataFrame, seed: int, fracs: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> dict[str, np.ndarray]:
    """Row-level random split. Leaks by construction; reported for contrast."""
    return _assign_groups(np.arange(len(df)), seed, fracs)
