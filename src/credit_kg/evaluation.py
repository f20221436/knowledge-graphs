"""Phase 9: confidence intervals, calibration and fairness-audit helpers for the final evaluation."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import config as C
from .stats import wilson_ci

EDUCATION = {1: "graduate school", 2: "university", 3: "high school", 4: "other / unknown"}
MARRIAGE = {1: "married", 2: "single", 3: "other / unknown"}
SEX = {1: "male", 2: "female"}
AGE_BINS = [0, 24, 34, 44, 54, 200]
AGE_LABELS = ["21-24", "25-34", "35-44", "45-54", "55+"]


# ---------- confidence intervals ----------
def cv_mean_ci(values, n_train: int, n_test: int, level: float = 0.95) -> tuple:
    """Mean of k CV-fold scores and a Nadeau-Bengio corrected t interval.
    Folds share training data, so the plain t interval is too narrow; the (1/k + n_test/n_train) factor widens it."""
    v = np.asarray(values, float)
    k = len(v)
    half = stats.t.ppf(0.5 + level / 2, k - 1) * np.sqrt((1 / k + n_test / n_train) * v.var(ddof=1))
    return float(v.mean()), float(v.mean() - half), float(v.mean() + half)


def bootstrap_metrics(y, preds: dict, n_boot: int = 2000, seed: int = C.SEED) -> pd.DataFrame:
    """ROC-AUC, PR-AUC and Brier score with 95% percentile intervals for several models.
    Every model is scored on the same resamples, so their intervals are directly comparable."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    idx = [i for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot)) if y[i].min() != y[i].max()]
    rows = []
    for name, p in preds.items():
        p = np.asarray(p, float)
        for metric, fn in [("roc_auc", roc_auc_score), ("pr_auc", average_precision_score), ("brier", brier_score_loss)]:
            boot = np.array([fn(y[i], p[i]) for i in idx])
            rows.append({"model": name, "metric": metric, "value": fn(y, p),
                         "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5)})
    return pd.DataFrame(rows)


# ---------- calibration ----------
def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Customers grouped into equal-size bins of predicted probability: mean prediction vs observed rate (Wilson CI)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, float)})
    df["bin"] = pd.qcut(df.p.rank(method="first"), n_bins, labels=False)
    g = df.groupby("bin").agg(n=("y", "size"), events=("y", "sum"), predicted=("p", "mean"))
    g["observed"] = g.events / g.n
    g["ci_low"], g["ci_high"] = wilson_ci(g.events, g.n)
    return g


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """Average gap between predicted and observed rate across equal-size bins, weighted by bin size."""
    t = reliability_table(y, p, n_bins)
    return float((t.n * (t.predicted - t.observed).abs()).sum() / t.n.sum())


def calibration_slope_intercept(y, p) -> tuple:
    """Logistic recalibration fit: observed ~ a + b * logit(p). Perfect calibration gives a = 0, b = 1."""
    from sklearn.linear_model import LogisticRegression

    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lg = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lg, np.asarray(y))
    return float(m.intercept_[0]), float(m.coef_[0][0])


def calibration_summary(y, p) -> dict:
    y = np.asarray(y)
    base = np.full(len(y), y.mean())
    a, b = calibration_slope_intercept(y, p)
    brier, brier_base = brier_score_loss(y, p), brier_score_loss(y, base)
    return {"brier": brier, "brier_base_rate": brier_base, "brier_skill": 1 - brier / brier_base,
            "ece": expected_calibration_error(y, p), "mean_predicted": float(np.mean(p)),
            "observed_rate": float(y.mean()), "calibration_intercept": a, "calibration_slope": b}


# ---------- fairness audit ----------
def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Readable protected-attribute groups. Used only to audit outcomes, never as inputs."""
    out = df.copy()
    out["sex_group"] = out["sex"].map(SEX)
    out["age_group"] = pd.cut(out["age"], AGE_BINS, labels=AGE_LABELS).astype(str)
    out["education_group"] = out["education"].map(EDUCATION)
    out["marriage_group"] = out["marriage"].map(MARRIAGE)
    return out


def group_audit(df: pd.DataFrame, group: str, band_col: str = "band", target: str = C.TARGET,
                favourable: str = "Supportable in this scenario") -> pd.DataFrame:
    """Per group: size, label shares, observed miss rate, miss rate among favourable labels,
    favourable share among customers who did not miss, and mean predicted vs observed risk."""
    fav = df[band_col] == favourable
    g = df.groupby(group, observed=True)
    t = pd.DataFrame({"n": g.size()})
    shares = pd.crosstab(df[group], df[band_col], normalize="index")
    t = t.join(shares)
    t["selection_rate"] = fav.groupby(df[group]).mean()
    t["selection_ratio_vs_max"] = t.selection_rate / t.selection_rate.max()
    t["observed_miss_rate"] = g[target].mean()
    t["mean_pd"] = g["pd"].mean()
    k = df[fav].groupby(group, observed=True)[target].agg(["sum", "size"])
    t["miss_rate_if_supportable"] = k["sum"] / k["size"]
    t["miss_if_sup_ci_low"], t["miss_if_sup_ci_high"] = wilson_ci(k["sum"], k["size"])
    t["supportable_among_non_missers"] = fav[df[target] == 0].groupby(df.loc[df[target] == 0, group]).mean()
    return t


def disparity_summary(audit: pd.DataFrame) -> dict:
    """Largest gaps across the groups of one attribute."""
    return {"min_selection_ratio": float(audit.selection_ratio_vs_max.min()),
            "selection_rate_range": (float(audit.selection_rate.min()), float(audit.selection_rate.max())),
            "miss_if_supportable_range": (float(audit.miss_rate_if_supportable.min()), float(audit.miss_rate_if_supportable.max())),
            "equal_opportunity_gap": float(audit.supportable_among_non_missers.max() - audit.supportable_among_non_missers.min())}


def proxy_strength(df: pd.DataFrame, inputs: list, targets: dict, cv: int = 5) -> pd.DataFrame:
    """How well the score's own inputs predict each protected attribute (cross-validated ROC-AUC).
    0.5 = the inputs carry no information about the attribute; higher = the inputs act partly as a proxy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    rows = [{"attribute": name, "positive_share": float(np.mean(y)),
             "roc_auc_from_score_inputs": float(cross_val_score(model, df[inputs], y, cv=cv, scoring="roc_auc").mean())}
            for name, y in targets.items()]
    return pd.DataFrame(rows)
