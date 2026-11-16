"""Neo4j plumbing: connection from environment variables, query helpers, batched idempotent loader."""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

from . import config as C

load_dotenv(C.ROOT / ".env")


def get_driver():
    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USERNAME"]
    pwd = os.environ["NEO4J_PASSWORD"]
    # Server notifications (e.g. "null value eliminated in set function") are expected here: we sum over
    # CASE expressions that are null for edges without a bill due. Silence them to keep notebooks readable.
    driver = GraphDatabase.driver(uri, auth=(user, pwd), notifications_min_severity="OFF")
    driver.verify_connectivity()
    return driver


def database():
    return os.environ.get("NEO4J_DATABASE") or None


def run(driver, query: str, **params) -> pd.DataFrame:
    records, _, keys = driver.execute_query(query, params, database_=database())
    return pd.DataFrame([r.values() for r in records], columns=keys)


def read_cypher(path) -> str:
    """Read a .cypher file (comments kept; the driver ignores them)."""
    return Path(path).read_text(encoding="utf-8").strip().rstrip(";")


def run_file(driver, path, **params) -> pd.DataFrame:
    return run(driver, read_cypher(path), **params)


def run_statements(driver, path):
    """Run a file containing several ';'-separated statements (schema setup)."""
    text = Path(path).read_text(encoding="utf-8")
    body = "\n".join(l for l in text.splitlines() if not l.strip().startswith("//"))
    for stmt in [s.strip() for s in body.split(";") if s.strip()]:
        driver.execute_query(stmt, database_=database())


def run_batched(driver, query: str, rows: list, batch_size: int = 5000):
    """UNWIND a list of dicts in batches, one transaction per batch."""
    for i in range(0, len(rows), batch_size):
        driver.execute_query(query, {"rows": rows[i:i + batch_size]}, database_=database())


OUR_LABELS = {"Customer", "CreditAccount", "MonthlyCreditRecord", "RepaymentState"}


def foreign_labels(driver) -> set:
    """Labels in the database that this project did not create (guard against loading into another project's DB)."""
    labels = set(run(driver, "CALL db.labels() YIELD label RETURN label")["label"])
    return labels - OUR_LABELS


def _records(df: pd.DataFrame) -> list:
    """DataFrame -> list of dicts with plain Python types and NaN -> None (Neo4j-safe)."""
    out = df.astype(object).where(df.notna(), None).to_dict("records")
    return [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in out]


def build_rows(wide: pd.DataFrame, monthly: pd.DataFrame, split: pd.DataFrame) -> dict:
    from .data import STATES

    states = [{"code": k, "ordinal": v[0], "description": v[1]} for k, v in STATES.items()]
    cust = wide.merge(split)[["customer_id", "credit_limit", "age", "sex", "education", "marriage", "split"]]
    rec_cols = ["record_id", "customer_id", "month_index", "month", "bill_amount", "payment_amount",
                "repayment_status", "state", "utilization", "is_late", "credit_balance"]
    rec = monthly[rec_cols]

    m = monthly.sort_values(["customer_id", "month_index"])
    nxt = m.groupby("customer_id").shift(-1)
    edges = pd.DataFrame({
        "from_id": m["record_id"],
        "to_id": nxt["record_id"],
        "coverage": nxt["coverage_of_prev_bill"],  # payment in t+1 against the bill of t
        "util_delta": nxt["utilization"] - m["utilization"],
        "risk_from": m["repayment_status"].clip(lower=0),
        "risk_to": nxt["repayment_status"].clip(lower=0),
        "transition": m["state"] + "->" + nxt["state"].astype(str),
    }).dropna(subset=["to_id"])
    edges["status_change"] = pd.Series(
        pd.cut(edges["risk_to"] - edges["risk_from"], [-99, -0.5, 0.5, 99], labels=["improving", "stable", "worsening"]),
        index=edges.index).astype(str)
    edges = edges.drop(columns=["risk_from", "risk_to"])
    return {"states": states, "customers": _records(cust), "records": _records(rec), "next_month": _records(edges)}


def load_graph(driver, rows: dict, batch_size: int = 5000) -> dict:
    """Idempotent load: constraints first, then MERGE by unique key in batches. Returns seconds per pass."""
    import time

    timings = {}
    t = time.time(); run_statements(driver, C.QUERIES / "schema" / "constraints.cypher"); timings["constraints"] = time.time() - t
    passes = [("states", "01_states.cypher"), ("customers", "02_customers_accounts.cypher"),
              ("records", "03_monthly_records.cypher"), ("next_month", "04_next_month.cypher")]
    for key, fname in passes:
        t = time.time()
        run_batched(driver, read_cypher(C.QUERIES / "load" / fname), rows[key], batch_size)
        timings[key] = round(time.time() - t, 1)
    return timings


def graph_counts(driver) -> dict:
    counts = {}
    for label, key in [("Customer", "customers"), ("CreditAccount", "accounts"),
                       ("MonthlyCreditRecord", "monthly_records"), ("RepaymentState", "states")]:
        counts[key] = int(run(driver, f"MATCH (n:{label}) RETURN count(n) AS n")["n"][0])
    for rel in ["HAS_ACCOUNT", "HAS_MONTHLY_RECORD", "NEXT_MONTH", "IN_STATE"]:
        counts[rel] = int(run(driver, f"MATCH ()-[r:{rel}]->() RETURN count(r) AS n")["n"][0])
    return counts
