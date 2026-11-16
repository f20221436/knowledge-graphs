"""Validation of the credit-increase score, which has no ground-truth label.

We cannot prove the score is "right", so we prove it is well-behaved: it moves in the intended direction,
ignores demographics, matches agreed expectations on hand-built personas, never breaks on edge cases,
and its explanation adds up exactly.
"""
import numpy as np
import pandas as pd
import pytest

from credit_kg import config as C
from credit_kg import scoring as S

CFG = S.load_config()
RANK = {S.BAND_NOT: 0, S.BAND_REVIEW: 1, S.BAND_SUPPORT: 2}


@pytest.fixture(scope="module")
def real():
    if not (C.MODELS / "pd_calibrated.parquet").exists():
        pytest.skip("run notebooks 01-07 first")
    return S.load_scoring_inputs().sample(3000, random_state=C.SEED).reset_index(drop=True)


def _band_rank(res):
    return res["band"].map(RANK)  # NaN for "insufficient evidence", which does not depend on these inputs


def _not_better(before, after):
    """after must have score <= before and band rank <= before (where both are ranked)."""
    assert (after["score"] <= before["score"] + 1e-9).all()
    rb, ra = _band_rank(before), _band_rank(after)
    both = rb.notna() & ra.notna()
    assert (ra[both] <= rb[both]).all()


# ---------- 1. direction (metamorphic) tests on 3,000 real customers ----------
def test_weights_sum_to_one():
    assert abs(sum(c["weight"] for c in CFG["components"].values()) - 1) < 1e-9


def test_larger_request_never_helps(real):
    prev = S.score(real, 10000, CFG)
    for amount in [20000, 30000, 50000, 100000]:
        cur = S.score(real, amount, CFG)
        _not_better(prev, cur)
        prev = cur


def test_higher_pd_never_helps(real):
    worse = real.assign(pd=(real["pd"] + 0.1).clip(upper=1))
    _not_better(S.score(real, 30000, CFG), S.score(worse, 30000, CFG))


def test_more_late_months_never_helps(real):
    worse = real.assign(late_months=(real["late_months"] + 1).clip(upper=6))
    _not_better(S.score(real, 30000, CFG), S.score(worse, 30000, CFG))


def test_better_coverage_never_hurts(real):
    better = real.assign(coverage_median=(real["coverage_median"] + 0.1).clip(upper=1))
    _not_better(S.score(better, 30000, CFG), S.score(real, 30000, CFG))


# ---------- 2. invariance: demographics must not matter ----------
def test_demographics_have_no_effect(real):
    rng = np.random.default_rng(0)
    a = real.assign(sex=1, education=1, marriage=1, age=30)
    b = real.assign(sex=rng.integers(1, 3, len(real)), education=rng.integers(1, 5, len(real)),
                    marriage=rng.integers(1, 4, len(real)), age=rng.integers(21, 80, len(real)))
    pd.testing.assert_frame_equal(S.score(a, 30000, CFG), S.score(b, 30000, CFG))


# ---------- 2b. relativity: only the customer's own scale matters, not absolute units ----------
@pytest.mark.parametrize("k", [0.1, 0.5, 2, 10])
def test_scale_invariance(real, k):
    """Multiply a customer's limit, typical payment AND the request by k (utilization and coverage are ratios,
    so they are unchanged). With the same default probability, the score must be identical."""
    scaled = real.assign(credit_limit=real["credit_limit"] * k, pay_median=real["pay_median"] * k)
    a = S.score(real, real["credit_limit"] * 0.25, CFG)
    b = S.score(scaled, scaled["credit_limit"] * 0.25, CFG)
    assert np.allclose(a["score"], b["score"]) and (a["band"] == b["band"]).all()


def test_same_absolute_request_differs_by_own_scale():
    """Senior's example in spirit: same absolute request, same everything else, different historical payment scale."""
    two = pd.DataFrame([{**BASE, "customer_id": 1, "pay_median": 20000}, {**BASE, "customer_id": 2, "pay_median": 10000}])
    res = S.score(two, 30000, CFG)
    assert res["score"].iloc[0] > res["score"].iloc[1]
    assert res["scenario_payback_months"].iloc[1] == pytest.approx(2 * res["scenario_payback_months"].iloc[0])


def test_relative_request_modes(real):
    r = S.relative_increase(real, "payment_multiple", 3)
    assert np.allclose(r, 3 * real["pay_median"])
    res = S.score(real, r, CFG)
    paying = real["pay_median"] > 0
    assert np.allclose(res.loc[paying, "scenario_payback_months"], 3)


# ---------- 3. golden personas (expected bands written BEFORE running the score) ----------
BASE = dict(customer_id=0, credit_limit=100000, pd=0.10, late_months=0, status_latest=0, coverage_median=0.5,
            pay_median=5000, util_latest=0.3, inactive_flag=0)
PERSONAS = [
    ("clean full payer, small request", dict(credit_limit=200000, pd=0.05, status_latest=-1, coverage_median=1.0, pay_median=20000), 20000, S.BAND_SUPPORT),
    ("chronic late payer", dict(credit_limit=50000, pd=0.70, late_months=6, status_latest=2, coverage_median=0.02, pay_median=1000, util_latest=0.95), 30000, S.BAND_NOT),
    ("inactive account", dict(pd=0.20, status_latest=-2, coverage_median=1.0, pay_median=0, util_latest=0.0, inactive_flag=1), 30000, S.BAND_INSUFFICIENT),
    ("good record but request = 50 months of payments", dict(credit_limit=50000, pd=0.12, coverage_median=0.1, pay_median=1000, util_latest=0.5), 50000, S.BAND_REVIEW),
    ("one recent late month, moderate profile", dict(credit_limit=150000, pd=0.35, late_months=1, status_latest=1, coverage_median=0.3, pay_median=5000, util_latest=0.6), 30000, S.BAND_REVIEW),
    ("uses card but never pays anything", dict(pd=0.30, coverage_median=0.0, pay_median=0, util_latest=0.8), 30000, S.BAND_INSUFFICIENT),
    ("over the limit, weak coverage, not late", dict(pd=0.30, coverage_median=0.0, pay_median=2000, util_latest=1.2), 10000, S.BAND_REVIEW),
    ("clean history but model PD above gate", dict(pd=0.55, coverage_median=1.0, pay_median=20000), 10000, S.BAND_NOT),
]


@pytest.mark.parametrize("name,overrides,increase,expected", PERSONAS, ids=[p[0] for p in PERSONAS])
def test_golden_persona(name, overrides, increase, expected):
    row = pd.DataFrame([{**BASE, **overrides}])
    assert S.score(row, increase, CFG)["band"].iloc[0] == expected


# ---------- 4. edge cases ----------
def test_edge_cases_do_not_break():
    rows = pd.DataFrame([
        {**BASE, "pay_median": 0},                                 # no demonstrated payment
        {**BASE, "util_latest": 0.0, "coverage_median": 1.0},      # credit balance (bill < 0 clipped to util 0)
        {**BASE, "pd": 0.0, "util_latest": 10.7},                  # extreme utilization
        {**BASE, "credit_limit": 10000, "pay_median": 1},          # tiny payment -> huge payback
    ])
    res = S.score(rows, 30000, CFG)
    num = res.select_dtypes("number").drop(columns=["scenario_payback_months"])
    assert np.isfinite(num.to_numpy()).all()
    assert np.isnan(res.loc[0, "scenario_payback_months"]) and res.loc[0, "scenario_no_demonstrated_payment"] == 1
    assert res["score"].between(0, 100).all()


# ---------- 5. traceability ----------
def test_contributions_add_up_to_score(real):
    res = S.score(real, 30000, CFG)
    total = 50 + res.filter(like="contrib_").sum(axis=1)
    assert np.allclose(total.round(1), res["score"])


def test_scenario_columns_never_enter_ml_features():
    from credit_kg.modeling import feature_sets
    for cols in feature_sets().values():
        assert not any(c.startswith("scenario_") or c == "pd" for c in cols)
