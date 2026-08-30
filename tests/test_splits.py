import numpy as np
import pandas as pd
import pytest

from finetune_esm.data import dataset, splits


def _toy() -> pd.DataFrame:
    # 3 PDBs, 6 rows: A appears 3x, B 2x, C 1x
    return pd.DataFrame(
        {
            "pdb": ["A", "A", "A", "B", "B", "C"],
            "split": ["train"] * 5 + ["test"],
            "pkd": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


def test_assert_no_pdb_leakage_accepts_disjoint_groups():
    df = _toy()
    splits.assert_no_pdb_leakage(df, {"a": np.array([0, 1, 2]), "b": np.array([3, 4, 5])})


def test_assert_no_pdb_leakage_rejects_shared_pdb():
    df = _toy()
    with pytest.raises(AssertionError, match="A"):
        splits.assert_no_pdb_leakage(df, {"a": np.array([0, 1]), "b": np.array([2, 3])})


def test_cold_pdb_split_keeps_groups_intact():
    rng_df = pd.DataFrame({"pdb": ["P%d" % (i // 4) for i in range(400)]})
    out = splits.cold_pdb_split(rng_df, seed=0)
    splits.assert_no_pdb_leakage(rng_df, out)
    assert sum(len(v) for v in out.values()) == 400


def test_random_split_partitions_every_row_exactly_once():
    df = pd.DataFrame({"pdb": ["X"] * 100})
    out = splits.random_split(df, seed=0)
    joined = np.sort(np.concatenate(list(out.values())))
    assert np.array_equal(joined, np.arange(100))


def test_splits_are_deterministic_given_a_seed():
    df = pd.DataFrame({"pdb": ["P%d" % (i // 3) for i in range(300)]})
    assert np.array_equal(
        splits.cold_pdb_split(df, seed=7)["train"],
        splits.cold_pdb_split(df, seed=7)["train"],
    )


@pytest.mark.slow
def test_author_split_sizes_and_purity():
    df = dataset.load_complexes()
    out = splits.author_split(df)
    assert {k: len(v) for k, v in out.items()} == {
        "train": 6485, "val": 965, "test": 757
    }
    splits.assert_no_pdb_leakage(df, out)
