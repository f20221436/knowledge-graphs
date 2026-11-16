"""Small statistics helpers for EDA and pattern validation."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def wilson_ci(k, n, z=1.96):
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def default_rate_table(x: pd.Series, y: pd.Series, bins=None, q=None, labels=None) -> pd.DataFrame:
    """Default rate per bucket with bucket size and a 95% Wilson confidence interval."""
    if q is not None:
        groups = pd.qcut(x, q, duplicates="drop")
    elif bins is not None:
        groups = pd.cut(x, bins, labels=labels)
    else:
        groups = x
    g = y.groupby(groups, observed=True).agg(n="size", defaults="sum")
    g["default_rate"] = g["defaults"] / g["n"]
    lo, hi = wilson_ci(g["defaults"], g["n"])
    g["ci_low"], g["ci_high"] = lo, hi
    g.index = g.index.astype(str)
    g.index.name = "bucket"
    return g.round(4)


def cochran_armitage(table: pd.DataFrame) -> float:
    """Two-sided p-value of the Cochran-Armitage trend test across ordered buckets."""
    n = table["n"].to_numpy(float)
    k = table["defaults"].to_numpy(float)
    t = np.arange(len(n), dtype=float)
    N, K = n.sum(), k.sum()
    p = K / N
    stat = (t * (k - n * p)).sum()
    var = p * (1 - p) * ((n * t ** 2).sum() - (n * t).sum() ** 2 / N)
    if var <= 0:
        return np.nan
    z = stat / np.sqrt(var)
    return float(2 * stats.norm.sf(abs(z)))


def single_feature_auc(x: pd.Series, y: pd.Series) -> float:
    """AUC of one feature alone, folded so 0.5 = no signal and direction is ignored."""
    auc = roc_auc_score(y, x.fillna(x.median()))
    return max(auc, 1 - auc)


def is_monotone(rates) -> str:
    d = np.diff(np.asarray(rates, float))
    if (d >= -0.005).all():
        return "increasing"
    if (d <= 0.005).all():
        return "decreasing"
    return "non-monotone"
