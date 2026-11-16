"""Phase 9 helpers: interval width, calibration metrics and the fairness audit table."""
import numpy as np
import pandas as pd
import pytest

from credit_kg import evaluation as E


def test_corrected_cv_interval_is_wider_than_naive():
    v = np.random.default_rng(0).normal(0.78, 0.007, 25)
    mean, lo, hi = E.cv_mean_ci(v, n_train=19200, n_test=4800)
    naive_half = 2.064 * v.std(ddof=1) / np.sqrt(25)
    assert lo < mean < hi
    assert (hi - lo) / 2 > naive_half


def test_perfectly_calibrated_predictions_score_well():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.02, 0.9, 200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    s = E.calibration_summary(y, p)
    assert s["ece"] < 0.01
    assert s["calibration_slope"] == pytest.approx(1, abs=0.05)
    assert s["calibration_intercept"] == pytest.approx(0, abs=0.05)
    assert s["brier_skill"] > 0


def test_overconfident_predictions_are_caught():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.6, 100_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    stretched = np.clip(0.5 + (p - 0.5) * 2.5, 0.001, 0.999)
    assert E.calibration_summary(y, stretched)["calibration_slope"] < 0.7
    assert E.expected_calibration_error(y, stretched) > E.expected_calibration_error(y, p)


def test_group_audit_counts_and_rates():
    df = pd.DataFrame({
        "g": ["a"] * 4 + ["b"] * 4,
        "band": ["Supportable in this scenario", "Supportable in this scenario", "Review", "Review",
                 "Supportable in this scenario", "Review", "Review", "Review"],
        "default_next_month": [0, 1, 0, 1, 0, 0, 1, 1],
        "pd": [0.1] * 8,
    })
    t = E.group_audit(df, "g")
    assert t.loc["a", "n"] == 4 and t.loc["b", "n"] == 4
    assert t.loc["a", "selection_rate"] == 0.5 and t.loc["b", "selection_rate"] == 0.25
    assert t.loc["b", "selection_ratio_vs_max"] == 0.5
    assert t.loc["a", "miss_rate_if_supportable"] == 0.5
    assert t.loc["a", "supportable_among_non_missers"] == 0.5
    assert t.loc["b", "supportable_among_non_missers"] == 0.5
