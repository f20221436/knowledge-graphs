"""Item 3: expected extra loss of a hypothetical increase + 'what would change the decision' explanations.

Assumptions (stated, not hidden):
* The default probability (PD) does not change because of the increase - the data cannot tell us that.
* Expected use of the new headroom = the customer's own typical (median) utilization, capped to [0, 1].
* LGD (share lost if default) is a configurable assumption; the data has no recovery information.
"""
import numpy as np
import pandas as pd

from credit_kg import scoring as S

from . import load_config


def typical_utilization(wide: pd.DataFrame) -> pd.Series:
    bills = wide[[f"bill_m{m}" for m in range(1, 7)]].clip(lower=0).to_numpy()
    util = bills / wide["credit_limit"].to_numpy()[:, None]
    return pd.Series(np.clip(np.median(util, axis=1), 0, 1), index=wide.index, name="typical_util")


def expected_loss(df: pd.DataFrame, increase, lgd: float | None = None) -> pd.DataFrame:
    """df needs: customer_id, pd, credit_limit, pay_median, typical_util, bill_latest."""
    lgd = load_config()["expected_loss"]["lgd"] if lgd is None else lgd
    inc = pd.Series(np.broadcast_to(np.asarray(increase, float), len(df)), index=df.index)
    drawn = inc * df["typical_util"]
    out = pd.DataFrame({"customer_id": df["customer_id"]}, index=df.index)
    out["el_requested_increase"] = inc
    out["el_expected_extra_drawn"] = drawn
    out["el_current_expected_loss"] = df["pd"] * df["bill_latest"].clip(lower=0) * lgd
    out["el_extra_expected_loss"] = df["pd"] * drawn * lgd
    has_pay = df["pay_median"] > 0
    out["el_extra_loss_in_payment_months"] = np.where(has_pay, out["el_extra_expected_loss"] / df["pay_median"].where(has_pay, 1), np.nan)
    return out


def _bisect(lo, hi, ok, iters):
    """Vectorised binary search for the boundary where ok(x) switches; ok must be monotone."""
    lo, hi = lo.astype(float).copy(), hi.astype(float).copy()
    for _ in range(iters):
        mid = (lo + hi) / 2
        good = ok(mid)
        lo = np.where(good, mid, lo)
        hi = np.where(good, hi, mid)
    return lo


def max_supportable_increase(df: pd.DataFrame, cfg=None, upper=None) -> pd.Series:
    """Largest hypothetical increase that still lands in 'Supportable' (score is non-increasing in the request).
    NaN when even a tiny request is not supportable (gate or weak profile)."""
    cfg = cfg or S.load_config()
    iters = load_config()["counterfactual"]["search_iterations"]
    upper = df["credit_limit"].to_numpy() * 3 if upper is None else np.asarray(upper, float)
    supp = lambda inc: (S.score(df, inc, cfg)["band"] == S.BAND_SUPPORT).to_numpy()
    possible = supp(np.full(len(df), 1.0))
    best = _bisect(np.zeros(len(df)), upper, supp, iters)
    best = np.where(supp(upper), upper, best)          # supportable even at the upper bound
    return pd.Series(np.where(possible, best, np.nan), index=df.index, name="max_supportable_increase")


def coverage_needed(df: pd.DataFrame, increase, cfg=None) -> pd.Series:
    """Smallest median payment coverage (share of last bill paid) that would make this request 'Supportable',
    keeping everything else fixed. NaN = already supportable; inf = coverage alone cannot get there."""
    cfg = cfg or S.load_config()
    iters = load_config()["counterfactual"]["search_iterations"]
    base = S.score(df, increase, cfg)
    supp_at = lambda c: (S.score(df.assign(coverage_median=c), increase, cfg)["band"] == S.BAND_SUPPORT).to_numpy()
    already = (base["band"] == S.BAND_SUPPORT).to_numpy()
    reachable = supp_at(np.ones(len(df)))
    # bisect for the smallest coverage that is supportable: search on 'not supportable' boundary
    lo, hi = df["coverage_median"].to_numpy(float), np.ones(len(df))
    for _ in range(iters):
        mid = (lo + hi) / 2
        ok = supp_at(mid)
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    out = np.where(already, np.nan, np.where(reachable, hi, np.inf))
    return pd.Series(out, index=df.index, name="coverage_needed")


def what_would_change(row_df: pd.DataFrame, increase, cfg=None) -> list[str]:
    """Plain-language levers for one customer (row_df has one row)."""
    cfg = cfg or S.load_config()
    r = S.score(row_df, increase, cfg).iloc[0]
    if r["band"] == S.BAND_SUPPORT:
        return ["Already supportable in this scenario."]
    if r["gate"]:
        return [f"Blocked by a gate ({r['gate']}); request size or payment coverage alone cannot change this."]
    msgs = []
    mx = max_supportable_increase(row_df, cfg).iloc[0]
    if np.isfinite(mx) and mx > 0:
        msgs.append(f"A request of up to {mx:,.0f} (instead of {float(np.asarray(increase).ravel()[0]):,.0f}) would be supportable.")
    cov = coverage_needed(row_df, increase, cfg).iloc[0]
    if np.isfinite(cov):
        msgs.append(f"Paying about {cov:.0%} of each previous bill (now {row_df['coverage_median'].iloc[0]:.0%}) would make this request supportable.")
    return msgs or ["No single lever (request size or payment coverage) is enough; the risk profile dominates."]
