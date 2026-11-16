"""Phase 8: hypothetical credit-increase scenario features, suitability score and explanations.

Everything here is a DEMONSTRATIVE scenario layer on top of the default-risk model. `requested_increase`
is a scenario parameter, never an observation, and the score is not an underwriting decision.
"""
import numpy as np
import pandas as pd

from . import config as C

BAND_SUPPORT = "Supportable in this scenario"
BAND_REVIEW = "Review"
BAND_NOT = "Not supportable in this scenario"
BAND_INSUFFICIENT = "Insufficient behavioural evidence"

REQUIRED = ["customer_id", "credit_limit", "pd", "late_months", "status_latest", "coverage_median",
            "pay_median", "util_latest", "inactive_flag"]


def load_config() -> dict:
    cfg = C.load_yaml("scoring.yaml")
    total = sum(c["weight"] for c in cfg["components"].values())
    assert abs(total - 1) < 1e-9, f"weights sum to {total}, not 1"
    return cfg


def relative_increase(df: pd.DataFrame, mode: str, factor: float) -> pd.Series:
    """Hypothetical request expressed in each customer's own historical units:
    'payment_multiple' -> factor x typical (median) monthly payment; 'limit_share' -> factor x credit limit."""
    base = {"payment_multiple": df["pay_median"], "limit_share": df["credit_limit"]}[mode]
    return (factor * base).astype(float)


def scenario_features(df: pd.DataFrame, increase) -> pd.DataFrame:
    """Relative features for a hypothetical increase (one amount, or one amount per customer).
    All columns are prefixed scenario_ so they never leak into the historical ML features."""
    out = pd.DataFrame({"customer_id": df["customer_id"].to_numpy()}, index=df.index)
    has_pay = df["pay_median"] > 0
    increase = pd.Series(np.broadcast_to(np.asarray(increase, float), len(df)), index=df.index)
    out["scenario_requested_increase"] = increase
    out["scenario_increase_to_limit"] = increase / df["credit_limit"]
    out["scenario_payback_months"] = np.where(has_pay, increase / df["pay_median"].where(has_pay, 1), np.nan)
    out["scenario_typical_payment_to_increase"] = (df["pay_median"] / increase.where(increase > 0)).astype(float)
    out["scenario_projected_limit"] = df["credit_limit"] + increase
    out["scenario_current_util"] = df["util_latest"]
    # Static-bill assumption: the latest bill stays the same, so projected utilization mechanically falls.
    out["scenario_projected_util"] = df["util_latest"] * df["credit_limit"] / (df["credit_limit"] + increase)
    out["scenario_util_change"] = out["scenario_projected_util"] - out["scenario_current_util"]
    out["scenario_no_demonstrated_payment"] = (~has_pay).astype(int)
    return out


def _interp(x, comp):
    return np.interp(np.asarray(x, float), comp["x"], comp["y"])


def component_scores(df: pd.DataFrame, sc: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    comp = cfg["components"]
    s = pd.DataFrame(index=df.index)
    s["model_risk"] = _interp(df["pd"], comp["model_risk"])
    s["repayment"] = (1 - df["late_months"] / 6) * np.where(df["status_latest"] >= 1, 0.5, 1.0)
    s["coverage"] = _interp(df["coverage_median"], comp["coverage"])
    s["payback"] = np.where(sc["scenario_no_demonstrated_payment"] == 1, 0.0,
                            _interp(sc["scenario_payback_months"].fillna(np.inf), comp["payback"]))
    s["request_vs_limit"] = _interp(sc["scenario_increase_to_limit"], comp["request_vs_limit"])
    s["utilization"] = np.where(df["inactive_flag"] == 1, comp["utilization"]["inactive_value"],
                                _interp(df["util_latest"], comp["utilization"]))
    return s.clip(0, 1)


def score(df: pd.DataFrame, increase, cfg: dict | None = None, weights: dict | None = None) -> pd.DataFrame:
    """Score every row of df (must contain REQUIRED columns) for a hypothetical increase
    (a single amount, or a per-customer array such as relative_increase(...))."""
    cfg = cfg or load_config()
    missing = set(REQUIRED) - set(df.columns)
    assert not missing, f"missing inputs: {missing}"
    w = weights or {k: v["weight"] for k, v in cfg["components"].items()}
    sc = scenario_features(df, increase)
    sub = component_scores(df, sc, cfg)
    contrib = pd.DataFrame({k: w[k] * (sub[k] - 0.5) * 100 for k in w}, index=df.index)
    total = 50 + contrib.sum(axis=1)

    g, b = cfg["gates"], cfg["bands"]
    band = np.select([total >= b["supportable"], total >= b["review"]], [BAND_SUPPORT, BAND_REVIEW], BAND_NOT)
    gate = np.full(len(df), "", dtype=object)
    risk_gate = (df["pd"] >= g["high_risk_pd"]) | (df["status_latest"] >= g["severe_latest_status"])
    evid_gate = (df["inactive_flag"] == 1) | (sc["scenario_no_demonstrated_payment"] == 1)
    gate[evid_gate.to_numpy()] = "insufficient evidence (inactive account or no payments)"
    gate[risk_gate.to_numpy()] = "high risk (PD >= %.2f or latest month %d+ months late)" % (g["high_risk_pd"], g["severe_latest_status"])
    band = np.where(risk_gate, BAND_NOT, np.where(evid_gate, BAND_INSUFFICIENT, band))

    out = pd.concat([sc, sub.add_prefix("sub_"), contrib.add_prefix("contrib_")], axis=1)
    out["score"] = total.round(1)
    out["band"] = band
    out["gate"] = gate
    return out


def explain(row: pd.Series, res: pd.Series, cfg: dict, graph_evidence: dict | None = None) -> str:
    """Plain-text explanation; every number is read from a computed column."""
    comp = cfg["components"]
    months = lambda v: ", ".join(v) if v else "none"
    lines = [
        f"Customer {int(row.customer_id)}  |  SCENARIO: hypothetical credit increase of {res.scenario_requested_increase:,.0f} (not an observed request)",
        f"Current limit {row.credit_limit:,.0f} -> projected {res.scenario_projected_limit:,.0f}",
        "",
        "Historical evidence (Apr-Sep 2005):",
        f"  average utilization {row.util_avg:.2f} | latest utilization {row.util_latest:.2f} | utilization trend {row.util_slope:+.3f}/month",
        f"  median coverage of previous statement {row.coverage_median:.2f} | weak-coverage months {int(row.weak_cov_months)}/5",
        f"  late months {int(row.late_months)}/6 (last 3 months: {int(row.late_recent3)}) | latest status code {int(row.status_latest)}",
        f"  typical (median) demonstrated monthly payment {row.pay_median:,.0f}",
    ]
    if graph_evidence:
        lines += ["Graph evidence (Neo4j):",
                  f"  late in: {months(graph_evidence.get('late_months'))}",
                  f"  weak payment against previous statement in: {months(graph_evidence.get('weak_payment_months'))}",
                  f"  longest late streak {graph_evidence.get('streak', 0)} | relapses {graph_evidence.get('relapses', 0)} | "
                  f"population P(late next month) {graph_evidence.get('p_next_late', float('nan')):.2f}"]
    payback = "n/a (no demonstrated payments)" if np.isnan(res.scenario_payback_months) else f"{res.scenario_payback_months:.1f} months of typical payment"
    lines += [
        "Scenario-relative evidence:",
        f"  request / current limit {res.scenario_increase_to_limit:.2f} | payback {payback}",
        f"  projected utilization {res.scenario_projected_util:.2f} (mechanical, assumes the bill stays the same; not scored)",
        f"Model: default probability {row.pd:.2f} (calibrated, out-of-sample)",
        "",
        f"Suitability score {res.score:.0f}/100  ->  {res.band}" + (f"  [gate: {res.gate}]" if res.gate else ""),
    ]
    contrib = {k: res[f"contrib_{k}"] for k in comp}
    pos = sorted([(v, k) for k, v in contrib.items() if v > 0.05], reverse=True)
    neg = sorted([(v, k) for k, v in contrib.items() if v < -0.05])
    lines += ["  supports:       " + ("; ".join(f"{comp[k]['label']} (+{v:.1f})" for v, k in pos) or "-"),
              "  weighs against: " + ("; ".join(f"{comp[k]['label']} ({v:.1f})" for v, k in neg) or "-"),
              "", "Note: " + cfg["disclaimer"]]
    return "\n".join(lines)


def load_scoring_inputs() -> pd.DataFrame:
    """Baseline features + calibrated out-of-sample PD + graph features, one row per customer."""
    wide = pd.read_parquet(C.PROCESSED / "customers.parquet")[["customer_id", "credit_limit", C.TARGET]]
    base = pd.read_parquet(C.FEATURES / "baseline.parquet")
    pdc = pd.read_parquet(C.MODELS / "pd_calibrated.parquet")
    t = wide.merge(base).merge(pdc)
    gpath = C.FEATURES / "graph.parquet"
    if gpath.exists():
        t = t.merge(pd.read_parquet(gpath), how="left")
    return t.sort_values("customer_id", ignore_index=True)


def graph_evidence(driver, customer_id: int, graph_row: pd.Series | None = None) -> dict:
    from .graph import run_file

    ev = run_file(driver, C.QUERIES / "patterns" / "customer_evidence.cypher", cid=int(customer_id),
                  weak=C.WEAK_COVERAGE).iloc[0].to_dict()
    if graph_row is not None:
        ev.update(streak=int(graph_row.g_max_late_streak), relapses=int(graph_row.g_relapses),
                  p_next_late=float(graph_row.g_markov_next_late_prob))
    return ev
