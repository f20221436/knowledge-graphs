"""Item 2: learn from every month with sliding windows.

v1 trains on one example per customer (6 months -> October default). Here every customer also gives
three "practice" examples from inside the six months: months 1-3 -> late in month 4, 2-4 -> 5, 3-5 -> 6.
No default label is used; the practice label is the customer's own observed lateness.
A model trained on those windows is then applied to months 4-6 to estimate "late next month" (October).

Repayment codes are inconsistent across months (code 1 almost only appears in September), so windows use
robust per-month flags (late / paid in full / no use) instead of raw codes.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

WINDOW = 3
TRAIN_ENDS = (3, 4, 5)   # windows ending in month 3/4/5 have a known next month (4/5/6)
SCORE_END = 6            # window ending in September is used to predict October


def window_features(wide: pd.DataFrame, end: int) -> pd.DataFrame:
    lim = wide["credit_limit"].to_numpy()
    f = pd.DataFrame({"customer_id": wide["customer_id"].to_numpy(), "window_end": end,
                      "log_limit": np.log(lim)})
    for k, m in enumerate(range(end - WINDOW + 1, end + 1), start=1):
        st = wide[f"status_m{m}"].to_numpy()
        bill = wide[f"bill_m{m}"].to_numpy()
        pay = wide[f"pay_m{m}"].to_numpy()
        f[f"late_{k}"] = (st >= 1).astype(int)
        f[f"paid_full_{k}"] = (st == -1).astype(int)
        f[f"no_use_{k}"] = (st == -2).astype(int)
        f[f"util_{k}"] = np.clip(bill, 0, None) / lim
        f[f"pay_rel_{k}"] = pay / lim
        if m > 1:  # coverage of the previous month's bill by this month's payment
            prev = wide[f"bill_m{m - 1}"].to_numpy()
            f[f"cov_{k}"] = np.where(prev > 0, np.minimum(pay / np.where(prev > 0, prev, 1), 1.0), 1.0)
            f[f"due_{k}"] = (prev > 0).astype(int)
    f["late_count"] = f[[f"late_{k}" for k in range(1, WINDOW + 1)]].sum(axis=1)
    return f


def training_windows(wide: pd.DataFrame, ends=TRAIN_ENDS) -> pd.DataFrame:
    parts = []
    for e in ends:
        w = window_features(wide, e)
        w["label_late_next"] = (wide[f"status_m{e + 1}"].to_numpy() >= 1).astype(int)
        parts.append(w)
    return pd.concat(parts, ignore_index=True)


def feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in ("customer_id", "window_end", "label_late_next")]


def make_model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=50, random_state=42)


def next_late_feature(wide: pd.DataFrame, split: pd.DataFrame, n_folds: int = 5) -> pd.Series:
    """'Probability of being late next month (October)' for every customer.
    Test customers: model trained on all training customers' windows.
    Training customers: out-of-fold by customer, so nobody is scored by a model that saw their own windows."""
    wide = wide.merge(split[["customer_id", "split"]]).reset_index(drop=True)
    train_ids = wide.loc[wide.split == "train", "customer_id"].to_numpy()
    tw = training_windows(wide[wide.split == "train"])
    cols = feature_cols(tw)
    score_w = window_features(wide, SCORE_END)
    out = pd.Series(np.nan, index=wide["customer_id"].to_numpy(), name="sw_next_late_prob")

    full = make_model().fit(tw[cols], tw["label_late_next"])
    test_mask = (wide.split == "test").to_numpy()
    out.iloc[np.where(test_mask)[0]] = full.predict_proba(score_w.loc[test_mask, cols])[:, 1]

    gkf = GroupKFold(n_splits=n_folds)
    folds = np.zeros(len(train_ids), dtype=int)
    for k, (_, b) in enumerate(gkf.split(train_ids, groups=train_ids)):
        folds[b] = k
    fold_of = dict(zip(train_ids, folds))
    tw_fold = tw["customer_id"].map(fold_of).to_numpy()
    for k in range(n_folds):
        m = make_model().fit(tw.loc[tw_fold != k, cols], tw.loc[tw_fold != k, "label_late_next"])
        ids_k = train_ids[folds == k]
        rows = np.where(np.isin(wide["customer_id"].to_numpy(), ids_k))[0]
        out.iloc[rows] = m.predict_proba(score_w.iloc[rows][cols])[:, 1]
    assert out.notna().all()
    return out
