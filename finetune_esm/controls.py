"""Ablation controls that establish what a result actually measures.

Spec §6 lists four mandatory controls. Each one destroys a specific source of
information while leaving the split structure, the PDB grouping, the row count
and the label distribution intact. A control that scores close to the
uncontrolled run is evidence that the uncontrolled run was not using the
information the control removed.

Transformations are applied WITHIN each split independently, and to every
split, so the model both trains and is evaluated under the same corruption.
That answers "is this task learnable without the removed information?", which
is a claim about the benchmark rather than about one trained model.

Every chain sequence that survives a transformation is one that already
appeared in the dataset, so the embedding cache always has it.
"""

import numpy as np
import pandas as pd

CONTROLS = (
    "none",
    "scramble_receptor",
    "fixed_receptor",
    "fixed_ligand",
    "shuffle_labels",
)


def _derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation of range(n) with as few fixed points as possible.

    A plain shuffle leaves ~1 row in e correctly paired, which would blunt the
    control. Fixed points are repaired by swapping each with its neighbour.
    n < 2 cannot be deranged and is returned unchanged.
    """
    if n < 2:
        return np.arange(n)
    perm = rng.permutation(n)
    # Repair in a single forward pass, re-checking each position as we reach
    # it. Collecting the fixed points up front and swapping them afterwards is
    # wrong: two adjacent fixed points swap with each other twice and end up
    # exactly where they started.
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    return perm


def _permute_within_splits(
    values: pd.Series, splits: dict[str, np.ndarray], rng: np.random.Generator
) -> pd.Series:
    out = values.copy()
    for idx in splits.values():
        if len(idx) < 2:
            continue
        out.iloc[idx] = values.iloc[idx[_derangement(len(idx), rng)]].to_numpy()
    return out


def _most_common(values: pd.Series, idx: np.ndarray):
    """The modal value over `idx`, as a stand-in constant."""
    keys = values.iloc[idx].map(tuple)
    return list(keys.value_counts().idxmax())


def apply_control(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    control: str,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a copy of `df` with `control` applied. `none` returns a copy unchanged."""
    if control not in CONTROLS:
        raise KeyError(f"unknown control {control!r}; available: {list(CONTROLS)}")

    out = df.copy()
    rng = np.random.default_rng(seed)

    if control == "none":
        return out

    if control == "scramble_receptor":
        # Each complex keeps its ligand and its label but is paired with a
        # different complex's receptor. If the score survives this, receptor
        # identity was not contributing.
        out["receptor_seqs"] = _permute_within_splits(
            df["receptor_seqs"], splits, rng
        )
        return out

    if control in ("fixed_receptor", "fixed_ligand"):
        # Every row gets the SAME partner, so that partner carries no
        # per-example information at all -- a stricter test than scrambling,
        # which at least preserves the marginal distribution of partners.
        column = "receptor_seqs" if control == "fixed_receptor" else "ligand_seqs"
        constant = _most_common(df[column], splits["train"])
        out[column] = pd.Series([list(constant)] * len(df), index=df.index)
        return out

    # shuffle_labels: the floor. Anything meaningfully above zero here means
    # the evaluation itself is leaking.
    out["pkd"] = _permute_within_splits(df["pkd"], splits, rng)
    return out
