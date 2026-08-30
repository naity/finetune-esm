import numpy as np
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
    pred = np.ones(10)
    target = np.arange(10, dtype=float)
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
