import numpy as np
import pytest
import torch
from scipy import stats

from finetune_esm import metrics


def test_matches_scipy_on_random_data():
    rng = np.random.default_rng(0)
    pred, target = rng.normal(size=200), rng.normal(size=200)
    assert abs(metrics.spearman(pred, target) - stats.spearmanr(pred, target).statistic) < 1e-9
    assert abs(metrics.pearson(pred, target) - stats.pearsonr(pred, target).statistic) < 1e-9


def test_rmse_is_root_mean_square_error():
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([1.0, 2.0, 5.0])
    assert abs(metrics.rmse(pred, target) - np.sqrt(4.0 / 3.0)) < 1e-12


def test_perfect_rank_agreement_is_one():
    target = np.arange(50, dtype=float)
    assert abs(metrics.spearman(2 * target + 7, target) - 1.0) < 1e-12


def test_handles_ties_without_error():
    pred = np.array([1.0, 1.0, 2.0, 2.0])
    target = np.array([1.0, 2.0, 1.0, 2.0])
    assert np.isfinite(metrics.spearman(pred, target))


def test_constant_prediction_is_nan_not_crash():
    # Test with exactly constant (ones)
    pred = np.ones(10)
    target = np.arange(10, dtype=float)
    assert np.isnan(metrics.spearman(pred, target))
    assert np.isnan(metrics.pearson(pred, target))
    assert np.isfinite(metrics.rmse(pred, target))

    # Test with realistic "constant" values (floating point near-constant)
    pred = np.full(7, 1.1)
    target = np.arange(7, dtype=float)
    assert np.isnan(metrics.spearman(pred, target))
    assert np.isnan(metrics.pearson(pred, target))
    assert np.isfinite(metrics.rmse(pred, target))

    pred = np.full(10, 1234.5678)
    target = np.arange(10, dtype=float)
    assert np.isnan(metrics.spearman(pred, target))
    assert np.isnan(metrics.pearson(pred, target))
    assert np.isfinite(metrics.rmse(pred, target))


def test_near_constant_prediction_is_nan_via_isclose_guard():
    # std ~= 2.9e-10, far from exactly 0 but well within floating-point
    # noise. scipy itself does NOT return NaN here -- it happily reports a
    # spurious near-1.0 correlation driven entirely by rounding, and a
    # guard written as `np.std(x) == 0` would fail to catch it (the std is
    # nonzero). Only `np.isclose(np.std(x), 0.0)` correctly recognizes this
    # as effectively constant and returns NaN. This test fails if the guard
    # is reverted from `np.isclose` to an exact `== 0` comparison.
    pred = np.linspace(1.0, 1.0 + 1e-9, 50)
    target = np.arange(50, dtype=float)
    assert np.std(pred) != 0.0  # not exactly constant -- an == 0 guard misses it
    assert not np.isnan(stats.spearmanr(pred, target).statistic)  # scipy alone: spurious
    assert np.isnan(metrics.spearman(pred, target))
    assert np.isnan(metrics.pearson(pred, target))
    assert np.isfinite(metrics.rmse(pred, target))


def test_accepts_torch_tensors():
    pred = torch.arange(20, dtype=torch.float32)
    target = torch.arange(20, dtype=torch.float32)
    assert abs(metrics.spearman(pred, target) - 1.0) < 1e-9


def test_all_metrics_keys():
    rng = np.random.default_rng(1)
    out = metrics.all_metrics(rng.normal(size=30), rng.normal(size=30))
    assert set(out) == {"spearman", "pearson", "rmse"}


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.rmse(np.zeros((2, 3)), np.zeros((5,)))
    with pytest.raises(ValueError):
        metrics.spearman(np.zeros((2, 3)), np.zeros((5,)))
    with pytest.raises(ValueError):
        metrics.pearson(np.zeros((2, 3)), np.zeros((5,)))
