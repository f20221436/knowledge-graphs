"""Layer 1-5 checks: month mapping, payment alignment, feature sanity, and graph/pandas parity."""
import numpy as np
import pandas as pd
import pytest

from credit_kg import config as C
from credit_kg import data as D
from credit_kg import features as F


@pytest.fixture(scope="module")
def raw():
    return D.load_raw()


def test_month_mapping(raw):
    w = D.to_wide(raw)
    assert (w["status_m6"] == raw["PAY_0"]).all() and (w["status_m1"] == raw["PAY_6"]).all()
    assert (w["bill_m6"] == raw["BILL_AMT1"]).all() and (w["pay_m1"] == raw["PAY_AMT6"]).all()


def test_payment_pays_previous_bill(raw):
    ev = D.alignment_evidence(raw)
    assert (ev["matches_previous_month_bill"] > 4 * ev["matches_same_month_bill"]).all()


def test_monthly_table_shape_and_coverage(raw):
    w = D.to_wide(raw)
    m = D.to_monthly(w)
    assert len(m) == 180000 and m.groupby("customer_id").size().eq(6).all()
    c1 = m[(m.customer_id == 1) & (m.month_index == 5)].iloc[0]   # Aug: paid 689 against July bill 689
    assert c1.coverage_of_prev_bill == 1.0
    assert np.isnan(m.loc[m.month_index == 1, "coverage_of_prev_bill"]).all()


def test_baseline_features_finite(raw):
    f = F.baseline_features(D.to_wide(raw))
    assert np.isfinite(f.drop(columns="customer_id").to_numpy()).all()


def test_graph_features_match_pandas():
    path = C.FEATURES / "graph.parquet"
    if not path.exists():
        pytest.skip("run notebook 05 first")
    g = pd.read_parquet(path).sort_values("customer_id", ignore_index=True)
    wide = pd.read_parquet(C.PROCESSED / "customers.parquet").merge(pd.read_csv(C.PROCESSED / "split.csv"))
    p1, p2 = F.transition_tables(wide[wide.split == "train"])
    pg = F.graph_features_pandas(wide, p1, p2).sort_values("customer_id", ignore_index=True)
    for c in F.GRAPH_FEATURES:
        assert np.allclose(g[c].astype(float), pg[c].astype(float)), c
