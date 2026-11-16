"""Checks for v2: expected loss, counterfactual levers and the scorecard."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from credit_kg_v2 import load_config, scenario_plus as SP, scorecard as SC  # noqa: E402
from credit_kg import scoring as S, modeling as M, config as C  # noqa: E402

BASE = dict(customer_id=0, credit_limit=100000, pd=0.10, late_months=0, status_latest=0, coverage_median=0.5,
            pay_median=5000, util_latest=0.3, inactive_flag=0, typical_util=0.4, bill_latest=30000)


def test_expected_loss_formula_and_monotone():
    df = pd.DataFrame([BASE])
    a, b = SP.expected_loss(df, 10000, lgd=1.0), SP.expected_loss(df, 20000, lgd=1.0)
    assert a["el_extra_expected_loss"].iloc[0] == pytest.approx(0.10 * 10000 * 0.4)
    assert b["el_extra_expected_loss"].iloc[0] == pytest.approx(2 * a["el_extra_expected_loss"].iloc[0])


def test_max_supportable_increase_is_a_real_boundary():
    df = pd.DataFrame([{**BASE, "pd": 0.2, "coverage_median": 0.2, "pay_median": 3000}])
    mx = SP.max_supportable_increase(df).iloc[0]
    assert np.isfinite(mx)
    assert S.score(df, mx * 0.99)["band"].iloc[0] == S.BAND_SUPPORT
    assert S.score(df, mx * 1.05)["band"].iloc[0] != S.BAND_SUPPORT


def test_coverage_needed_reaches_supportable():
    # Review-band customer (score ~57) who would reach Supportable by paying more of each bill.
    df = pd.DataFrame([{**BASE, "pd": 0.3, "late_months": 1, "coverage_median": 0.0, "pay_median": 2000}])
    inc = 30000
    assert S.score(df, inc)["band"].iloc[0] == S.BAND_REVIEW
    cov = SP.coverage_needed(df, inc).iloc[0]
    assert np.isfinite(cov) and 0 < cov <= 1
    assert S.score(df.assign(coverage_median=cov), inc)["band"].iloc[0] == S.BAND_SUPPORT
    assert S.score(df.assign(coverage_median=max(cov - 0.02, 0)), inc)["band"].iloc[0] != S.BAND_SUPPORT


def test_gated_customer_gets_gate_message():
    df = pd.DataFrame([{**BASE, "pd": 0.7}])
    assert "gate" in SP.what_would_change(df, 30000)[0]


@pytest.fixture(scope="module")
def sc_fit():
    t = M.load_table()
    tr = t[t.split == "train"]
    cfg = load_config()["scorecard"]
    feats = cfg["features_without_graph"] + cfg["graph_features"]
    return SC.Scorecard(feats).fit(tr[feats], tr[C.TARGET]), tr, feats, cfg


def test_scorecard_bins_monotone_except_allowed(sc_fit):
    sc, _, feats, cfg = sc_fit
    for f in feats:
        if f in cfg["allow_non_monotone"]:
            continue
        d = np.diff(sc.tables[f]["bad_rate"].to_numpy())
        assert (d >= 0).all() or (d <= 0).all(), f


def test_scorecard_points_add_up(sc_fit):
    sc, tr, feats, _ = sc_fit
    row = tr.iloc[[0]]
    assert sc.explain(row[feats])["points"].sum() == pytest.approx(sc.score(row[feats]).iloc[0])


def test_rich_states_and_motif_tokens():
    from credit_kg_v2 import rich_states as RS
    from credit_kg import data as D
    w = D.load_processed()[0].head(200)
    st = RS.rich_states(w)
    assert st.shape == (200, 6) and all(len(s.split("|")) == 3 for s in st.ravel())
    assert (np.char.endswith(st[:, 0].astype(str), "|NA")).all()        # month 1 has no previous bill
    cm = RS.customer_motifs(w)
    assert len(cm) == 200 * 3 and cm.start_month.isin([2, 3, 4]).all()


def test_bh_correction_is_monotone_and_bounded():
    from credit_kg_v2.rich_states import _bh
    p = np.array([0.001, 0.01, 0.02, 0.04, 0.5])
    q = _bh(p)
    assert (q >= p).all() and (q <= 1).all() and (np.diff(q[np.argsort(p)]) >= 0).all()


def test_sliding_windows_have_no_default_label():
    from credit_kg_v2 import sliding as SW
    from credit_kg import data as D
    w = D.load_processed()[0].head(100)
    tw = SW.training_windows(w)
    assert C.TARGET not in tw.columns and len(tw) == 300
    assert set(tw.window_end) == {3, 4, 5}


def test_v2_graph_refuses_v1_database():
    from credit_kg_v2 import graph_v2 as GV
    from dotenv import dotenv_values
    v1 = dotenv_values(C.ROOT / ".env")
    assert GV._same_as_v1(v1.get("NEO4J_URI", "bolt://localhost:7687"), v1.get("NEO4J_DATABASE") or None)
    assert not GV._same_as_v1("bolt://localhost:7688", None)


def test_higher_points_mean_lower_default(sc_fit):
    sc, tr, feats, _ = sc_fit
    pts = sc.score(tr[feats])
    q = pd.qcut(pts, 5, labels=False, duplicates="drop")
    rates = tr[C.TARGET].groupby(q).mean().to_numpy()
    assert rates[0] > rates[-1]
