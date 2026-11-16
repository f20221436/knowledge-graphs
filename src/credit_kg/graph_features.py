"""Phase 5: extract graph-derived features from Neo4j with Cypher.

Pattern queries double as feature queries: with their thresholds lowered to 1 they return the raw
per-customer measurement (streak length, relapse count, rising run). Population probabilities come from
the training customers' NEXT_MONTH/RepaymentState structure only.
"""
import numpy as np
import pandas as pd

from . import config as C
from .graph import run_file

P = C.QUERIES / "patterns"
FQ = C.QUERIES / "features"
STATE_ORDER = ["NO_USE", "PAID_FULL", "REVOLVING", "LATE_1", "LATE_2PLUS"]


def transition_probs(driver, alpha: float = 1.0):
    """First-order P(next state | state) from Q9 and second-order P(late next | two states) from F2."""
    t = run_file(driver, P / "Q09_transition_network.cypher")
    counts = t.pivot(index="from_state", columns="to_state", values="transitions")
    counts = counts.reindex(index=STATE_ORDER, columns=STATE_ORDER).fillna(0) + alpha
    p1 = counts.div(counts.sum(axis=1), axis=0)
    ctx = run_file(driver, FQ / "F02_context_next_late.cypher")
    p2 = dict(zip(ctx["context"], (ctx["next_late"] + alpha * 0.5) / (ctx["n"] + alpha)))
    return p1, p2


def extract(driver) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    base = run_file(driver, FQ / "F01_transition_counts.cypher", weak=C.WEAK_COVERAGE)
    streak = run_file(driver, P / "Q02_late_streak.cypher", min_streak=1)[["customer_id", "longest_streak"]]
    relapse = run_file(driver, P / "Q10_relapse.cypher", min_relapses=1)[["customer_id", "relapses"]]
    rise = run_file(driver, P / "Q04_rising_utilization.cypher", rise=C.UTIL_RISE, min_rises=1)[["customer_id", "rising_months"]]

    p1, p2 = transition_probs(driver)
    logp = {f"{a}->{b}": float(np.log(p1.loc[a, b])) for a in STATE_ORDER for b in STATE_ORDER}
    pop = run_file(driver, FQ / "F03_population_features.cypher", p2=p2, logp=logp,
                   fallback=float(np.mean(list(p2.values()))))

    g = (base.merge(streak, how="left").merge(relapse, how="left").merge(rise, how="left").merge(pop, how="left")
         .rename(columns={"longest_streak": "g_max_late_streak", "relapses": "g_relapses", "rising_months": "g_max_util_rise_run"}))
    for c in ["g_max_late_streak", "g_relapses", "g_max_util_rise_run"]:
        g[c] = g[c].fillna(0).astype(int)
    cols = ["customer_id", "g_max_late_streak", "g_late_onsets", "g_recoveries", "g_relapses", "g_latest_transition",
            "g_max_util_rise_run", "g_weak_cov_after_late", "g_markov_next_late_prob", "g_path_typicality"]
    return g[cols].sort_values("customer_id", ignore_index=True), p1, p2
