"""Evaluation metrics. Spearman is primary; Pearson and RMSE are reported alongside."""

import numpy as np
from scipy import stats


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x, dtype=np.float64).ravel()


def _to_numpy_pair(pred, target) -> tuple[np.ndarray, np.ndarray]:
    """Convert both inputs to numpy and check shape agreement."""
    p, t = _to_numpy(pred), _to_numpy(target)
    if len(p) != len(t):
        raise ValueError(
            f"pred and target must have the same length: got {len(p)} and {len(t)}"
        )
    return p, t


def spearman(pred, target) -> float:
    """Spearman rank correlation. NaN when either input is constant."""
    p, t = _to_numpy_pair(pred, target)
    if np.isclose(np.std(p), 0.0) or np.isclose(np.std(t), 0.0):
        return float("nan")
    return float(stats.spearmanr(p, t).statistic)


def pearson(pred, target) -> float:
    """Pearson correlation. NaN when either input is constant."""
    p, t = _to_numpy_pair(pred, target)
    if np.isclose(np.std(p), 0.0) or np.isclose(np.std(t), 0.0):
        return float("nan")
    return float(stats.pearsonr(p, t).statistic)


def rmse(pred, target) -> float:
    """Root mean squared error, in pKd units."""
    p, t = _to_numpy_pair(pred, target)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def all_metrics(pred, target) -> dict[str, float]:
    return {
        "spearman": spearman(pred, target),
        "pearson": pearson(pred, target),
        "rmse": rmse(pred, target),
    }
