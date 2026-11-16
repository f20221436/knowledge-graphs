"""Item 1 (graph part): load the v2 knowledge graph into a SEPARATE Neo4j database.

The v2 graph = the full v1 graph (built with v1's tested, idempotent loader) + new structure:
    (:BehaviourState {code, repayment, usage, payment, p_late_next})       60 shared rich states
    (:MonthlyCreditRecord)-[:IN_BEHAVIOUR_STATE]->(:BehaviourState)        180,000 links
    (:Motif {pattern, odds_ratio_within_late_groups, ...})                  mined 3-month patterns
    (:CreditAccount)-[:SHOWS_MOTIF {start_month}]->(:Motif)                 found by walking NEXT_MONTH paths

Credentials come from v2/.env.v2 (NEO4J_V2_URI / NEO4J_V2_USERNAME / NEO4J_V2_PASSWORD / NEO4J_V2_DATABASE).
Loading refuses to run against the v1 database, so the v1 graph can never be modified by v2.
"""
import os

import numpy as np
import pandas as pd
from dotenv import dotenv_values
from neo4j import GraphDatabase

from credit_kg import config as C1
from credit_kg import graph as G1

from . import V2
from . import rich_states as RS

Q = V2 / "queries"
V2_LABELS = {"BehaviourState", "Motif"}


def v2_settings() -> dict:
    vals = dotenv_values(V2 / ".env.v2") if (V2 / ".env.v2").exists() else {}
    vals = {k: os.environ.get(k, v) for k, v in vals.items()}
    missing = [k for k in ("NEO4J_V2_URI", "NEO4J_V2_USERNAME", "NEO4J_V2_PASSWORD") if not vals.get(k)]
    if missing:
        raise RuntimeError(f"v2 graph not configured: set {missing} in v2/.env.v2 (see v2/.env.v2.example)")
    return vals


def _same_as_v1(v2uri: str, v2db: str | None) -> bool:
    v1 = dotenv_values(C1.ROOT / ".env")
    norm = lambda u: (u or "").replace("neo4j://", "bolt://").replace("127.0.0.1", "localhost").rstrip("/")
    return norm(v1.get("NEO4J_URI")) == norm(v2uri) and (v1.get("NEO4J_DATABASE") or None) == (v2db or None)


def get_v2_driver():
    s = v2_settings()
    if _same_as_v1(s["NEO4J_V2_URI"], s.get("NEO4J_V2_DATABASE")):
        raise RuntimeError("NEO4J_V2_URI points at the v1 database. v2 must use a separate Neo4j instance/database.")
    drv = GraphDatabase.driver(s["NEO4J_V2_URI"], auth=(s["NEO4J_V2_USERNAME"], s["NEO4J_V2_PASSWORD"]),
                               notifications_min_severity="OFF")
    drv.verify_connectivity()
    drv._v2_db = s.get("NEO4J_V2_DATABASE") or None
    return drv


def runner(driver):
    """run(query, **params) -> DataFrame against the v2 database (auto-commit)."""
    def run(query, **params):
        recs, _, keys = driver.execute_query(query, params, database_=driver._v2_db)
        return pd.DataFrame([r.values() for r in recs], columns=keys)
    return run


def tx_runner(tx):
    """Same interface on an explicit transaction (used by the rollback-only test)."""
    def run(query, **params):
        res = tx.run(query, params)
        keys = res.keys()
        return pd.DataFrame([r.values() for r in res], columns=keys)
    return run


def read(name: str) -> str:
    return (Q / name).read_text(encoding="utf-8").strip().rstrip(";")


# ------------------------------------------------------------------ row builders
def state_rows(p_late: dict) -> list:
    rows = []
    for code, p in p_late.items():
        if code == "__overall__":
            continue
        repay, usage, payb = code.split("|")
        rows.append({"code": code, "repay": repay, "usage": usage, "payband": payb, "p_late_next": float(p)})
    return rows


def record_rows(wide: pd.DataFrame) -> list:
    st = RS.rich_states(wide)
    r, _, p = RS.month_parts(wide)
    tok = np.char.add(np.char.add(r.astype(str), "/"), p.astype(str))
    ids = wide["customer_id"].to_numpy()
    out = []
    for j in range(6):
        out += [{"record_id": f"{c}-{j + 1}", "code": s, "token": t} for c, s, t in zip(ids, st[:, j], tok[:, j])]
    return out


def motif_rows(mined: pd.DataFrame) -> list:
    return [{"motif": r.motif, "customers": int(r.customers), "default_rate": float(r.default_rate),
             "or_within": float(r.or_within_late_groups), "q_within": float(r.q_within),
             "selected": bool(r.adds_beyond_late_count),
             "direction": "riskier" if r.or_within_late_groups > 1 else "safer"} for r in mined.itertuples()]


def load_extras(run, p_late: dict, wide: pd.DataFrame, mined: pd.DataFrame, batch: int = 5000) -> dict:
    """Load the v2-only structure via any run(query, **params) callable."""
    srows, rrows = state_rows(p_late), record_rows(wide)
    # every code seen in the records must exist as a node (states unseen in training get the overall rate)
    known = {s["code"] for s in srows}
    for code in {r["code"] for r in rrows} - known:
        repay, usage, payb = code.split("|")
        srows.append({"code": code, "repay": repay, "usage": usage, "payband": payb, "p_late_next": float(p_late["__overall__"])})
    run(read("load_01_behaviour_states.cypher"), rows=srows)
    for i in range(0, len(rrows), batch):
        run(read("load_02_record_states.cypher"), rows=rrows[i:i + batch])
    run(read("load_03_motifs.cypher"), rows=motif_rows(mined))
    links = run(read("load_04_shows_motif.cypher"))["links"].iloc[0]
    return {"behaviour_states": len(srows), "record_links": len(rrows), "motifs": len(mined), "shows_motif_links": int(links)}


def load_v2_graph(driver, wide, monthly, split, p_late, mined) -> dict:
    """Full v2 load into the separate database: v1 base graph (idempotent) + v2 extras."""
    run = runner(driver)
    labels = set(run("CALL db.labels() YIELD label RETURN label")["label"])
    foreign = labels - G1.OUR_LABELS - V2_LABELS
    if foreign:
        raise RuntimeError(f"v2 database contains other labels {foreign}; use an empty database.")
    old_db = os.environ.get("NEO4J_DATABASE")
    os.environ["NEO4J_DATABASE"] = driver._v2_db or ""      # v1 helpers read the database name from here
    try:
        G1.load_graph(driver, G1.build_rows(wide, monthly, split))
    finally:
        if old_db is None:
            os.environ.pop("NEO4J_DATABASE", None)
        else:
            os.environ["NEO4J_DATABASE"] = old_db
    for stmt in [s.strip() for s in read("constraints_v2.cypher").split(";") if s.strip() and not s.strip().startswith("//")]:
        driver.execute_query("\n".join(l for l in stmt.splitlines() if not l.strip().startswith("//")), database_=driver._v2_db)
    return load_extras(run, p_late, wide, mined)
