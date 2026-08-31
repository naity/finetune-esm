import pytest

from finetune_esm import baselines
from finetune_esm.data import dataset


@pytest.mark.slow
def test_trivial_features_shape_and_finiteness():
    df = dataset.load_complexes().head(500)
    feats = baselines.trivial_features(df)
    assert len(feats) == 500
    assert feats.notna().all().all()
    for col in ("ligand_len", "receptor_len", "n_ligand_chains", "total_len"):
        assert col in feats.columns
    # 20 amino-acid composition features
    assert sum(c.startswith("frac_") for c in feats.columns) == 20


@pytest.mark.slow
def test_trivial_baseline_reproduces_measured_floor():
    out = baselines.run_trivial_baseline(seed=0)
    # Measured from sequence-only feature set on authors' train/test split.
    assert out["spearman"] == pytest.approx(0.229, abs=0.015)
    assert out["rmse"] == pytest.approx(1.896, abs=0.05)
