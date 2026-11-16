"""Phase 3 baseline features (order-insensitive aggregates + latest value + slope).

Anything that needs month-to-month adjacency (streaks, transitions, relapses, population
transition probabilities) is deliberately NOT here: those come from the Neo4j graph (Phase 5).
`graph_features_pandas` re-implements the graph features only for the parity test.
"""
import warnings

import numpy as np
import pandas as pd

from . import config as C
from .data import coverage_matrix, status_to_state

RAW_MONTHLY = (
    [f"status_m{m}" for m in C.MONTHS]
    + [f"bill_m{m}" for m in C.MONTHS]
    + [f"pay_m{m}" for m in C.MONTHS]
)

_T = np.arange(1, 7) - 3.5  # centred month index for OLS slopes


def _slope(mat: np.ndarray) -> np.ndarray:
    return (mat * _T).sum(axis=1) / (_T ** 2).sum()


def baseline_features(wide: pd.DataFrame) -> pd.DataFrame:
    lim = wide["credit_limit"].to_numpy()
    bill = wide[[f"bill_m{m}" for m in C.MONTHS]].to_numpy()
    pay = wide[[f"pay_m{m}" for m in C.MONTHS]].to_numpy()
    status = wide[[f"status_m{m}" for m in C.MONTHS]].to_numpy()
    util = np.clip(bill, 0, None) / lim[:, None]
    cov = coverage_matrix(wide)
    late = status >= 1

    f = pd.DataFrame({"customer_id": wide["customer_id"].to_numpy()})
    # A. utilization
    f["util_avg"] = util.mean(1)
    f["util_latest"] = util[:, -1]
    f["util_std"] = util.std(1)
    f["util_slope"] = _slope(util)
    f["high_util_months"] = (util >= C.HIGH_UTIL).sum(1)
    f["inactive_flag"] = (util <= C.INACTIVE_UTIL).all(1).astype(int)
    f["credit_balance_months"] = (bill < 0).sum(1)
    # B. payment coverage (5 aligned pairs; months with nothing due are NaN)
    has_due = ~np.isnan(cov)
    with warnings.catch_warnings():  # customers with no statement ever due give all-NaN rows (handled below)
        warnings.simplefilter("ignore", RuntimeWarning)
        cov_med = np.nanmedian(np.where(has_due, cov, np.nan), axis=1)
        cov_min = np.nanmin(np.where(has_due, cov, np.nan), axis=1)
    # No statement ever due -> nothing was left unpaid: coverage 1.0 (flagged by no_balance_months).
    f["coverage_median"] = np.where(np.isnan(cov_med), 1.0, cov_med)
    f["coverage_min"] = np.where(np.isnan(cov_min), 1.0, cov_min)
    f["coverage_latest"] = np.where(np.isnan(cov[:, -1]), 1.0, cov[:, -1])
    f["weak_cov_months"] = (np.nan_to_num(cov, nan=1.0) < C.WEAK_COVERAGE).sum(1)
    f["full_pay_months"] = (np.nan_to_num(cov, nan=0.0) >= C.FULL_COVERAGE).sum(1)
    f["no_balance_months"] = (~has_due).sum(1)
    # C. repayment / delinquency
    f["late_months"] = late.sum(1)
    f["late_recent3"] = late[:, 3:].sum(1)
    f["max_status"] = status.max(1)
    f["status_latest"] = status[:, -1]
    f["paid_full_months"] = (status == -1).sum(1)
    f["no_use_months"] = (status == -2).sum(1)
    # D. bills
    f["bill_latest"] = bill[:, -1]
    f["bill_slope_rel"] = _slope(bill) / lim
    # E. payments (median = typical demonstrated payment; robust to one-off lump sums)
    f["pay_median"] = np.median(pay, 1)
    f["pay_latest"] = pay[:, -1]
    f["pay_slope_rel"] = _slope(pay) / lim
    f["zero_pay_months"] = (pay == 0).sum(1)
    # F. stability
    pay_mean = pay.mean(1)
    f["pay_cv"] = np.where(pay_mean > 0, pay.std(1) / np.where(pay_mean > 0, pay_mean, 1), 0.0)
    return f


ENGINEERED = [
    "util_avg", "util_latest", "util_std", "util_slope", "high_util_months", "inactive_flag",
    "credit_balance_months", "coverage_median", "coverage_min", "coverage_latest", "weak_cov_months",
    "full_pay_months", "no_balance_months", "late_months", "late_recent3", "max_status",
    "status_latest", "paid_full_months", "no_use_months", "bill_latest", "bill_slope_rel",
    "pay_median", "pay_latest", "pay_slope_rel", "zero_pay_months", "pay_cv",
]

GRAPH_FEATURES = [
    "g_max_late_streak", "g_late_onsets", "g_recoveries", "g_relapses", "g_latest_transition",
    "g_max_util_rise_run", "g_weak_cov_after_late", "g_markov_next_late_prob", "g_path_typicality",
]


# ---------------------------------------------------------------------------------------------
# Pandas re-implementation of the Cypher graph features. Used ONLY by the parity test to prove
# the Cypher logic is correct; the models always read the Neo4j-exported values.
# ---------------------------------------------------------------------------------------------
def _longest_run(mask: np.ndarray) -> np.ndarray:
    best = np.zeros(len(mask), dtype=int)
    cur = np.zeros(len(mask), dtype=int)
    for j in range(mask.shape[1]):
        cur = np.where(mask[:, j], cur + 1, 0)
        best = np.maximum(best, cur)
    return best


def transition_tables(wide_train: pd.DataFrame, alpha: float = 1.0):
    """First- and second-order state transition probabilities from training customers."""
    order =["NO_USE", "PAID_FULL", "REVOLVING", "LATE_1", "LATE_2PLUS"]
    s = status_to_state(wide_train[[f"status_m{m}" for m in C.MONTHS]].to_numpy())
    first = pd.crosstab(pd.Series(s[:, :-1].ravel(), name="from"), pd.Series(s[:, 1:].ravel(), name="to"))
    first = first.reindex(index=order, columns=order, fill_value=0) + alpha
    p1 = first.div(first.sum(axis=1), axis=0)
    late_cols = ["LATE_1", "LATE_2PLUS"]
    ctx = pd.Series(s[:, :-2].ravel()).str.cat(pd.Series(s[:, 1:-1].ravel()), sep="|")
    nxt_late = pd.Series(np.isin(s[:, 2:].ravel(), late_cols))
    g = nxt_late.groupby(ctx).agg(["sum", "size"])
    p2 = ((g["sum"] + alpha * 0.5) / (g["size"] + alpha)).to_dict()
    return p1, p2


def graph_features_pandas(wide: pd.DataFrame, p1: pd.DataFrame, p2: dict) -> pd.DataFrame:
    status = wide[[f"status_m{m}" for m in C.MONTHS]].to_numpy()
    bill = wide[[f"bill_m{m}" for m in C.MONTHS]].to_numpy()
    util = np.clip(bill, 0, None) / wide["credit_limit"].to_numpy()[:, None]
    cov = coverage_matrix(wide)
    late = status >= 1
    risk = np.clip(status, 0, None)
    s = status_to_state(status)

    onset = ~late[:, :-1] & late[:, 1:]
    recover = late[:, :-1] & ~late[:, 1:]
    # relapse = each onset that happens after an earlier recovery
    relapses = np.zeros(len(wide), dtype=int)
    seen_recovery = np.zeros(len(wide), dtype=bool)
    for j in range(5):
        relapses += (onset[:, j] & seen_recovery).astype(int)
        seen_recovery |= recover[:, j]
    rise = np.diff(util, axis=1) > C.UTIL_RISE
    last = np.sign(risk[:, 5] - risk[:, 4]).astype(int)
    weak_after_late = (late[:, :-1] & (np.nan_to_num(cov, nan=1.0) < C.WEAK_COVERAGE)).sum(1)

    logp = np.log(p1.to_numpy())
    idx = {k: i for i, k in enumerate(p1.index)}
    si = np.vectorize(idx.get)(s)
    typ = logp[si[:, :-1], si[:, 1:]].mean(1)
    fallback = float(np.mean(list(p2.values())))
    ctx = pd.Series(s[:, 4]).str.cat(pd.Series(s[:, 5]), sep="|")
    nxt = ctx.map(p2).fillna(fallback).to_numpy()

    return pd.DataFrame({
        "customer_id": wide["customer_id"].to_numpy(),
        "g_max_late_streak": _longest_run(late),
        "g_late_onsets": onset.sum(1),
        "g_recoveries": recover.sum(1),
        "g_relapses": relapses,
        "g_latest_transition": last,
        "g_max_util_rise_run": _longest_run(rise),
        "g_weak_cov_after_late": weak_after_late,
        "g_markov_next_late_prob": nxt,
        "g_path_typicality": typ,
    })
