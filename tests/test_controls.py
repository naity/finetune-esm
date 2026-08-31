import numpy as np
import pandas as pd
import pytest

from finetune_esm import controls


def _df(n=60):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "pdb": [f"P{i // 2}" for i in range(n)],
            "ligand_seqs": [[f"L{i}"] for i in range(n)],
            "receptor_seqs": [[f"R{i}"] for i in range(n)],
            "pkd": rng.normal(7.0, 2.0, size=n),
        }
    )


def _splits(n=60):
    return {"train": np.arange(0, 40), "val": np.arange(40, 50), "test": np.arange(50, n)}


def test_unknown_control_raises_naming_available():
    with pytest.raises(KeyError, match="scramble_receptor"):
        controls.apply_control(_df(), _splits(), "nonexistent")


def test_none_is_a_faithful_copy_not_the_same_object():
    df = _df()
    out = controls.apply_control(df, _splits(), "none")
    assert out is not df
    pd.testing.assert_frame_equal(out, df)


@pytest.mark.parametrize("control", controls.CONTROLS)
def test_every_control_preserves_shape_pdb_and_label_distribution(control):
    df = _df()
    out = controls.apply_control(df, _splits(), control, seed=1)
    assert len(out) == len(df)
    assert list(out.columns) == list(df.columns)
    # PDB grouping must survive so splits and group_id stay meaningful
    assert out["pdb"].tolist() == df["pdb"].tolist()
    # labels are permuted at most, never invented
    assert sorted(out["pkd"].round(9)) == sorted(df["pkd"].round(9))


def test_scramble_receptor_keeps_ligands_and_labels_but_repairs_pairing():
    df = _df()
    out = controls.apply_control(df, _splits(), "scramble_receptor", seed=1)
    assert out["ligand_seqs"].tolist() == df["ligand_seqs"].tolist()
    assert out["pkd"].tolist() == df["pkd"].tolist()
    # the multiset of receptors is unchanged...
    assert sorted(map(tuple, out["receptor_seqs"])) == sorted(
        map(tuple, df["receptor_seqs"])
    )
    # ...but essentially nothing is still correctly paired
    same = sum(a == b for a, b in zip(out["receptor_seqs"], df["receptor_seqs"]))
    assert same == 0


def test_scramble_stays_within_split_boundaries():
    df, sp = _df(), _splits()
    out = controls.apply_control(df, sp, "scramble_receptor", seed=2)
    for idx in sp.values():
        before = sorted(map(tuple, df["receptor_seqs"].iloc[idx]))
        after = sorted(map(tuple, out["receptor_seqs"].iloc[idx]))
        assert before == after, "a receptor crossed a split boundary"


def test_fixed_partner_collapses_that_side_to_one_sequence():
    df, sp = _df(), _splits()
    r = controls.apply_control(df, sp, "fixed_receptor", seed=0)
    assert r["receptor_seqs"].map(tuple).nunique() == 1
    assert r["ligand_seqs"].tolist() == df["ligand_seqs"].tolist()

    lig = controls.apply_control(df, sp, "fixed_ligand", seed=0)
    assert lig["ligand_seqs"].map(tuple).nunique() == 1
    assert lig["receptor_seqs"].tolist() == df["receptor_seqs"].tolist()


def test_fixed_partner_uses_a_sequence_that_exists_in_the_data():
    """The constant must be a real chain, so the embedding cache contains it."""
    df, sp = _df(), _splits()
    out = controls.apply_control(df, sp, "fixed_receptor", seed=0)
    constant = tuple(out["receptor_seqs"].iloc[0])
    assert constant in set(df["receptor_seqs"].map(tuple))


def test_shuffle_labels_breaks_the_pairing_but_keeps_sequences():
    df, sp = _df(), _splits()
    out = controls.apply_control(df, sp, "shuffle_labels", seed=3)
    assert out["ligand_seqs"].tolist() == df["ligand_seqs"].tolist()
    assert out["receptor_seqs"].tolist() == df["receptor_seqs"].tolist()
    assert not np.allclose(out["pkd"].to_numpy(), df["pkd"].to_numpy())


def test_controls_are_deterministic_given_a_seed():
    df, sp = _df(), _splits()
    a = controls.apply_control(df, sp, "scramble_receptor", seed=7)
    b = controls.apply_control(df, sp, "scramble_receptor", seed=7)
    assert a["receptor_seqs"].tolist() == b["receptor_seqs"].tolist()


def test_derangement_has_no_fixed_points():
    rng = np.random.default_rng(0)
    for n in (2, 3, 5, 50, 500):
        perm = controls._derangement(n, rng)
        assert sorted(perm) == list(range(n))
        assert not (perm == np.arange(n)).any()
