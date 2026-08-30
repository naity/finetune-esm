import numpy as np
import pytest

from finetune_esm.data import dataset


def test_kd_to_pkd_is_negative_log10():
    kd = np.array([1e-9, 1e-6, 1.0])
    assert np.allclose(dataset.kd_to_pkd(kd), [9.0, 6.0, 0.0])


@pytest.mark.slow
def test_load_complexes_schema_and_counts():
    df = dataset.load_complexes()
    assert len(df) == 8207
    assert list(df.columns) == [
        "complex_id", "pdb", "ligand_seqs", "receptor_seqs",
        "pkd", "split", "has_mutation", "source",
    ]
    assert df["split"].value_counts().to_dict() == {
        "train": 6485, "val": 965, "test": 757
    }


@pytest.mark.slow
def test_pkd_range_matches_measurements():
    df = dataset.load_complexes()
    assert df["pkd"].min() == pytest.approx(1.39, abs=0.01)
    assert df["pkd"].max() == pytest.approx(15.70, abs=0.01)
    assert df["pkd"].isna().sum() == 0


@pytest.mark.slow
def test_sequences_are_split_on_commas_and_nonempty():
    df = dataset.load_complexes()
    assert df["ligand_seqs"].map(len).min() >= 1
    assert df["receptor_seqs"].map(len).min() >= 1
    for seqs in df["ligand_seqs"].head(200):
        for s in seqs:
            assert s == s.strip() and len(s) > 0 and "," not in s


@pytest.mark.slow
def test_mutation_flag_matches_split_composition():
    df = dataset.load_complexes()
    by_split = df.groupby("split")["has_mutation"].sum().to_dict()
    assert by_split["train"] == 4766
    assert by_split["test"] == 21
