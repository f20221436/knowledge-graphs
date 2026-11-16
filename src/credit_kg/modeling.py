"""Phases 6-7: feature matrices, Random Forest, repeated CV, OOF thresholds, metrics, paired bootstrap.

Experiments A and B share this code, the same split, the same folds, the same hyper-parameters and
the same threshold procedure, so any difference comes from the features alone.
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, accuracy_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict

from . import config as C
from .features import GRAPH_FEATURES

POPULATION_FEATURES = ["g_markov_next_late_prob", "g_path_typicality"]


def feature_sets() -> dict:
    spec = json.loads((C.FEATURES / "feature_spec.json").read_text())
    raw, eng = spec["raw_features"], spec["engineered_features"]
    return {
        "A0_raw": raw,
        "A": raw + eng,
        "A_demo": raw + eng + C.DEMOGRAPHICS,
        "B": raw + eng + GRAPH_FEATURES,
        "B_no_population": raw + eng + [g for g in GRAPH_FEATURES if g not in POPULATION_FEATURES],
        "graph_only": ["credit_limit"] + GRAPH_FEATURES,
    }


def load_table() -> pd.DataFrame:
    """One row per customer: raw + baseline + graph features + target + split, sorted by customer_id."""
    wide = pd.read_parquet(C.PROCESSED / "customers.parquet")
    split = pd.read_csv(C.PROCESSED / "split.csv")
    t = wide.merge(split).merge(pd.read_parquet(C.FEATURES / "baseline.parquet"))
    gpath = C.FEATURES / "graph.parquet"
    if gpath.exists():
        t = t.merge(pd.read_parquet(gpath), how="left")
    return t.sort_values("customer_id", ignore_index=True)


def matrices(table: pd.DataFrame, name: str):
    cols = feature_sets()[name]
    forbidden = {C.TARGET, "split"} | {c for c in table.columns if c.startswith("scenario_")}
    assert not forbidden & set(cols), f"leaky columns in {name}: {forbidden & set(cols)}"
    if any(c.startswith("g_") for c in cols):
        g = [c for c in cols if c.startswith("g_")]
        assert table[g].notna().all().all(), "graph features missing - run notebook 05"
        assert (table[g].nunique() > 1).all(), "a graph feature is constant"
    tr, te = table[table.split == "train"], table[table.split == "test"]
    return tr[cols], tr[C.TARGET].to_numpy(), te[cols], te[C.TARGET].to_numpy(), tr["customer_id"].to_numpy(), te["customer_id"].to_numpy()


def make_rf(min_samples_leaf: int = 20, n_estimators: int = 300) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features="sqrt",
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=C.SEED)


def repeated_cv(X, y, model_fn, n_splits=5, n_repeats=5) -> pd.DataFrame:
    """Per-fold ROC-AUC and PR-AUC on identical folds for any feature set (folds depend only on y and the seed)."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=C.SEED)
    rows = []
    for i, (a, b) in enumerate(cv.split(X, y)):
        m = model_fn().fit(X.iloc[a], y[a])
        p = m.predict_proba(X.iloc[b])[:, 1]
        rows.append({"fold": i, "roc_auc": roc_auc_score(y[b], p), "pr_auc": average_precision_score(y[b], p)})
    return pd.DataFrame(rows)


def oof_proba(X, y, model_fn) -> np.ndarray:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.SEED)
    return cross_val_predict(model_fn(), X, y, cv=cv, method="predict_proba")[:, 1]


def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(thr[np.argmax(f1[:-1])])


def evaluate(y, p, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"threshold": round(threshold, 4), "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "precision": precision_score(y, pred), "recall": recall_score(y, pred), "f1": f1_score(y, pred),
            "accuracy": accuracy_score(y, pred), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def corrected_resampled_ttest(diffs, n_train: int, n_test: int, n_splits: int = 5) -> dict:
    """Nadeau & Bengio (2003) corrected repeated k-fold t-test on per-fold differences.
    Plain t-tests on overlapping CV folds are over-confident; the (1/k + n_test/n_train) term corrects that."""
    from scipy import stats

    d = np.asarray(diffs, float)
    k = len(d)
    var = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var) if var > 0 else np.inf
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return {"mean_diff": d.mean(), "folds_b_better": int((d > 0).sum()), "folds": k, "t": t, "p_value": p}


def paired_bootstrap(y, p_a, p_b, n_boot=2000, seed=C.SEED) -> dict:
    """Bootstrap the test set; recompute both models' AUCs on the same resample; CI of the difference B - A."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d_auc, d_pr = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        d_auc.append(roc_auc_score(y[i], p_b[i]) - roc_auc_score(y[i], p_a[i]))
        d_pr.append(average_precision_score(y[i], p_b[i]) - average_precision_score(y[i], p_a[i]))
    d_auc, d_pr = np.array(d_auc), np.array(d_pr)
    return {"delta_roc_auc": roc_auc_score(y, p_b) - roc_auc_score(y, p_a),
            "roc_auc_ci": (np.percentile(d_auc, 2.5), np.percentile(d_auc, 97.5)),
            "p_b_not_better_roc": float((d_auc <= 0).mean()),
            "delta_pr_auc": average_precision_score(y, p_b) - average_precision_score(y, p_a),
            "pr_auc_ci": (np.percentile(d_pr, 2.5), np.percentile(d_pr, 97.5))}
