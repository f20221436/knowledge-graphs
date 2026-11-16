"""The agent's tools. Each wraps existing, tested project code; the LLM only chooses which to call.

Every result passes through strip_protected before the LLM sees it, so protected attributes, the outcome label
and internal fields cannot leak even if a function returned them by mistake.
"""
import json
import pickle
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, model_validator

from .. import config as C
from .. import modeling as M
from .. import scoring as S
from .guardrails import CypherRejected, check_cypher_schema, run_guarded_cypher, strip_protected
from .settings import AgentSettings

N_CUSTOMERS = 30000


# ---------------------------------------------------------------- argument schemas
class CustomerArgs(BaseModel):
    customer_id: int = Field(..., description="Customer ID, an integer from 1 to 30000")


class ScoreArgs(BaseModel):
    customer_id: int = Field(..., description="Customer ID, an integer from 1 to 30000")
    requested_amount: float | None = Field(None, gt=0, description="Requested increase in currency units, e.g. 20000")
    times_typical_payment: float | None = Field(None, gt=0, description="Request as a multiple of the customer's typical monthly payment, e.g. 3")
    share_of_limit: float | None = Field(None, gt=0, description="Request as a share of the current limit, e.g. 0.25")

    @model_validator(mode="after")
    def exactly_one_request(self):
        # An explicit amount is unambiguous, so it takes precedence (eval v3: qwen2.5:3b sent requested_amount together
        # with share_of_limit, got a validation error, and gave up). The amount guard checks it matches the question.
        if self.requested_amount is not None:
            self.times_typical_payment = self.share_of_limit = None
            return self
        given = [v for v in (self.times_typical_payment, self.share_of_limit) if v is not None]
        if len(given) != 1:
            raise ValueError("give exactly one of requested_amount, times_typical_payment, share_of_limit")
        return self


class CypherArgs(BaseModel):
    query: str = Field(..., max_length=2000, description="One read-only Cypher query")


# ---------------------------------------------------------------- data access (loaded once, lazily)
class DataStore:
    @cached_property
    def monthly(self) -> pd.DataFrame:
        return pd.read_parquet(C.PROCESSED / "monthly.parquet").set_index("customer_id")

    @cached_property
    def scoring_inputs(self) -> pd.DataFrame:
        return S.load_scoring_inputs().set_index("customer_id", drop=False)

    @cached_property
    def model_table(self) -> pd.DataFrame:
        return M.load_table().set_index("customer_id", drop=False)

    @cached_property
    def model_a(self):
        return pickle.load(open(C.MODELS / "model_A.pkl", "rb"))

    @cached_property
    def features_a(self) -> list:
        return M.feature_sets()["A"]

    @cached_property
    def cfg(self) -> dict:
        return S.load_config()

    @cached_property
    def base_rate(self) -> float:
        return float(self.model_table[C.TARGET].mean())


def _exists(customer_id: int):
    if not 1 <= customer_id <= N_CUSTOMERS:
        raise LookupError(f"customer {customer_id} does not exist (valid IDs are 1 to {N_CUSTOMERS})")


def _r(x, d=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


# Plain-language meanings computed by code, so a small model does not have to infer them (smoke tests showed it
# mis-reading status codes and the direction of numbers).
def status_meaning(code: int) -> str:
    return {-2: "card not used, nothing to pay", -1: "paid in full on time", 0: "paid the minimum, revolving credit"}.get(
        int(code), "about 1 month late" if int(code) == 1 else "2 or more months late")


_MONTH = {k: C.MONTH_LABELS[k] for k in C.MONTHS}
_ENGINEERED = {
    "credit_limit": "credit limit", "util_avg": "average share of the limit used", "util_slope": "trend in share of limit used",
    "high_util_months": "months using 90%+ of the limit", "inactive_flag": "card essentially unused",
    "coverage_median": "typical share of the previous bill paid", "coverage_min": "lowest share of the previous bill paid",
    "coverage_latest": "share of the previous bill paid in the latest month", "weak_cov_months": "months paying under 10% of the previous bill",
    "full_pay_months": "months paying the previous bill in full", "late_months": "number of late months (of 6)",
    "late_recent3": "late months among the last 3", "max_status": "worst repayment status in the 6 months",
    "status_latest": "repayment status in the latest month", "paid_full_months": "months marked paid in full",
    "no_use_months": "months with no card use", "pay_median": "typical (median) monthly payment",
    "pay_latest": "payment in the latest month", "zero_pay_months": "months with no payment",
}


def feature_meaning(name: str) -> str:
    """Plain-language name for a model input, so a model does not confuse e.g. pay_m5 with the typical payment."""
    for prefix, what in [("status_m", "repayment status in"), ("bill_m", "bill in"), ("pay_m", "payment in")]:
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            return f"{what} {_MONTH[int(name[len(prefix):])]}"
    return _ENGINEERED.get(name, name)


def vs_average(p: float, avg: float) -> str:
    return "above the population average (higher risk than typical)" if p > avg else "below the population average (lower risk than typical)"


# ---------------------------------------------------------------- tool functions
def get_customer_profile(ctx: "ToolContext", customer_id: int) -> dict:
    _exists(customer_id)
    m = ctx.data.monthly.loc[[customer_id]].sort_values("month_index")
    row = ctx.data.scoring_inputs.loc[customer_id]
    months = [{"month": r.month, "bill": _r(r.bill_amount, 0), "payment": _r(r.payment_amount, 0),
               "status_code": int(r.repayment_status), "state": r.state, "late": bool(r.is_late),
               "utilization": _r(r.utilization, 3), "share_of_previous_bill_paid": _r(r.coverage_of_prev_bill, 3)}
              for r in m.itertuples()]
    return {"customer_id": customer_id, "credit_limit": _r(row.credit_limit, 0), "months_observed": len(months),
            "months": months,
            "summary": {"late_months": int(row.late_months), "late_months_last_3": int(ctx.data.model_table.loc[customer_id, "late_recent3"]),
                        "latest_status_code": int(row.status_latest), "latest_status_meaning": status_meaning(row.status_latest),
                        "typical_monthly_payment": _r(row.pay_median, 1),
                        "median_share_of_previous_bill_paid": _r(row.coverage_median, 3),
                        "latest_utilization": _r(row.util_latest, 3), "average_utilization": _r(row.util_avg, 3),
                        "inactive_account": bool(row.inactive_flag)},
            "note": "Payments are payment behaviour, not income. Demographic attributes are withheld by design."}


def _forest_contributions(forest, x: np.ndarray):
    """Saabas decomposition: walk each tree's path and credit every change in the class-1 share to the split feature.
    bias + sum(contributions) equals the forest's predicted probability exactly."""
    contrib, bias = np.zeros(x.shape[1]), 0.0
    for est in forest.estimators_:
        t = est.tree_
        v = t.value[:, 0, :]
        p = v[:, 1] / v.sum(axis=1)
        node = 0
        bias += p[0]
        while t.children_left[node] != -1:
            f = t.feature[node]
            nxt = t.children_left[node] if x[0, f] <= t.threshold[node] else t.children_right[node]
            contrib[f] += p[nxt] - p[node]
            node = nxt
    n = len(forest.estimators_)
    return bias / n, contrib / n


def get_risk_prediction(ctx: "ToolContext", customer_id: int) -> dict:
    _exists(customer_id)
    row = ctx.data.scoring_inputs.loc[customer_id]
    x = ctx.data.model_table.loc[[customer_id], ctx.data.features_a].to_numpy(dtype=np.float32)
    bias, contrib = _forest_contributions(ctx.data.model_a, x)
    top = np.argsort(-np.abs(contrib))[:5]
    return {"customer_id": customer_id,
            "probability_of_missing_next_payment": _r(row["pd"]),
            "calibrated": True, "out_of_sample": True,
            "population_average": _r(ctx.data.base_rate),
            "compared_with_average": vs_average(row["pd"], ctx.data.base_rate),
            "top_contributions": [{"feature": ctx.data.features_a[i], "meaning": feature_meaning(ctx.data.features_a[i]),
                                   "customer_value": _r(x[0, i], 3),
                                   "contribution": _r(contrib[i]),
                                   "effect": "raises risk" if contrib[i] > 0 else "lowers risk"} for i in top],
            "note": "Contributions explain the uncalibrated forest (model A) score, which starts from "
                    f"{bias:.4f} and adds them up; positive = pushes towards missing a payment. The probability above "
                    "comes from a calibrated, cross-validated version of the same model."}


def score_credit_increase(ctx: "ToolContext", customer_id: int, requested_amount=None, times_typical_payment=None,
                          share_of_limit=None) -> dict:
    _exists(customer_id)
    cfg = ctx.data.cfg
    row = ctx.data.scoring_inputs.loc[[customer_id]].reset_index(drop=True)
    if requested_amount is not None:
        inc, basis = float(requested_amount), "absolute amount"
    elif times_typical_payment is not None:
        inc, basis = float(S.relative_increase(row, "payment_multiple", times_typical_payment).iloc[0]), \
            f"{times_typical_payment:g} x typical monthly payment"
    else:
        inc, basis = float(S.relative_increase(row, "limit_share", share_of_limit).iloc[0]), f"{share_of_limit:g} x current limit"
    res = S.score(row, inc, cfg).iloc[0]
    r0 = row.iloc[0]
    pts = sorted(((cfg["components"][k]["label"], _r(res[f"contrib_{k}"], 1)) for k in cfg["components"]), key=lambda t: -t[1])
    return {"customer_id": customer_id, "requested_increase": _r(inc, 0), "request_basis": basis,
            "score": _r(res.score, 1), "band": res.band, "gate": res.gate or None,
            "band_thresholds": {"supportable_from": cfg["bands"]["supportable"], "review_from": cfg["bands"]["review"]},
            "how_the_score_is_built": "starts at a neutral 50; each factor adds or removes points",
            "factors_that_added_points": [{"factor": f, "points": p} for f, p in pts if p > 0.05],
            "factors_that_removed_points": [{"factor": f, "points": p} for f, p in pts if p < -0.05],
            "probability_compared_with_average": vs_average(r0["pd"], ctx.data.base_rate),
            "population_average_probability": _r(ctx.data.base_rate),
            "payback_months": _r(res.scenario_payback_months, 1), "request_to_limit": _r(res.scenario_increase_to_limit, 3),
            "current_limit": _r(r0.credit_limit, 0), "projected_limit": _r(res.scenario_projected_limit, 0),
            "inputs": {"probability_of_missing_next_payment": _r(r0["pd"]), "late_months": int(r0.late_months),
                       "latest_status_code": int(r0.status_latest), "latest_status_meaning": status_meaning(r0.status_latest),
                       "median_share_of_previous_bill_paid": _r(r0.coverage_median, 3),
                       "typical_monthly_payment": _r(r0.pay_median, 1), "latest_utilization": _r(r0.util_latest, 3)},
            "disclaimer": cfg["disclaimer"]}


def get_graph_evidence(ctx: "ToolContext", customer_id: int) -> dict:
    _exists(customer_id)
    g = ctx.data.model_table.loc[customer_id]
    ev = S.graph_evidence(ctx.driver, customer_id)   # parameterised query from queries/patterns/customer_evidence.cypher
    return {"customer_id": customer_id, "late_in": ev.get("late_months") or [],
            "weak_payment_against_previous_statement_in": ev.get("weak_payment_months") or [],
            "improved_in": ev.get("improved_in") or [], "worsened_in": ev.get("worsened_in") or [],
            "longest_late_streak": int(g.g_max_late_streak), "relapses": int(g.g_relapses),
            "recoveries": int(g.g_recoveries),
            "population_probability_late_next_month": _r(g.g_markov_next_late_prob, 3),
            "note": "Read from the customer's month-by-month chain in the Neo4j knowledge graph."}


# v3 (follow-up to eval v2): the schema used to sit inside run_cypher's description, which every request carries.
# With the tool present, a 3B model got 3 unrelated factual questions wrong that it answered correctly without it.
# The schema now comes from its own tool, fetched only when a Cypher question needs it, with worked examples.
# The examples teach syntax and are deliberately not the evaluation questions (see evals/tasks_v1.jsonl).
GRAPH_SCHEMA = {
    "paths": ["(:Customer {customer_id})-[:HAS_ACCOUNT]->(:CreditAccount {account_id, credit_limit})",
              "(:CreditAccount)-[:HAS_MONTHLY_RECORD]->(:MonthlyCreditRecord)",
              "(:MonthlyCreditRecord)-[:NEXT_MONTH {coverage, util_delta, status_change, transition}]->(:MonthlyCreditRecord)",
              "(:MonthlyCreditRecord)-[:IN_STATE]->(:RepaymentState {code, ordinal, description})"],
    "MonthlyCreditRecord_properties": "record_id, customer_id, month_index (1..6), month ('2005-04' .. '2005-09'), bill_amount, "
                                      "payment_amount, repayment_status, state, utilization (bill / limit), is_late (true/false), credit_balance",
    "values": {"state / RepaymentState.code": ["NO_USE", "PAID_FULL", "REVOLVING", "LATE_1", "LATE_2PLUS"],
               "NEXT_MONTH.status_change": ["stable", "worsening", "improving"],
               "NEXT_MONTH.transition": "e.g. 'REVOLVING->LATE_2PLUS'",
               "NEXT_MONTH.coverage": "share of the previous month's bill paid"},
    "rules": ["Cypher, not SQL: no SELECT, FROM or sub-selects.",
              "A Customer reaches its months only through its CreditAccount; MonthlyCreditRecord also has customer_id.",
              "Properties belong to the node or relationship listed above (status_change is on NEXT_MONTH, not IN_STATE).",
              "One read-only query; aggregate (count, avg, sum) or use LIMIT. Demographics and the outcome are not queryable."],
    "examples": [
        {"question": "Average credit limit", "cypher": "MATCH (a:CreditAccount) RETURN avg(a.credit_limit) AS avg_limit"},
        {"question": "Records per repayment state in 2005-07",
         "cypher": "MATCH (m:MonthlyCreditRecord {month: '2005-07'})-[:IN_STATE]->(s:RepaymentState) RETURN s.code AS state, count(*) AS n"},
        {"question": "Kinds of change from 2005-04 to 2005-05",
         "cypher": "MATCH (a:MonthlyCreditRecord {month: '2005-04'})-[r:NEXT_MONTH]->(:MonthlyCreditRecord) "
                   "RETURN r.status_change AS change, count(*) AS n"},
        {"question": "Accounts using 90%+ of the limit in at least 3 months",
         "cypher": "MATCH (a:CreditAccount)-[:HAS_MONTHLY_RECORD]->(m:MonthlyCreditRecord) "
                   "WITH a, sum(CASE WHEN m.utilization >= 0.9 THEN 1 ELSE 0 END) AS high WHERE high >= 3 RETURN count(a) AS n"},
    ],
}


class NoArgs(BaseModel):
    pass


def get_graph_schema(ctx: "ToolContext") -> dict:
    return GRAPH_SCHEMA


def run_cypher(ctx: "ToolContext", query: str) -> dict:
    s = ctx.settings
    check_cypher_schema(query)
    return run_guarded_cypher(ctx.driver, query, database=ctx.database, row_cap=s.cypher_row_cap,
                              timeout_s=s.cypher_timeout_s, guardrails=s.guardrails)


# ---------------------------------------------------------------- registry
@dataclass
class ToolSpec:
    name: str
    description: str
    args: type
    fn: Callable


TOOLS = {t.name: t for t in [
    ToolSpec("get_customer_profile", "Credit limit and the six monthly bills, payments and repayment statuses "
             "(Apr-Sep 2005) for one customer, plus summary behaviour (late months, typical payment, share of bill paid).",
             CustomerArgs, get_customer_profile),
    ToolSpec("get_risk_prediction", "The model's calibrated probability that the customer misses next month's payment, "
             "and the five inputs that pushed the model's score most.", CustomerArgs, get_risk_prediction),
    ToolSpec("score_credit_increase", "Run the credit-increase suitability score for a hypothetical request: score 0-100, "
             "label, safety-check result and the points each factor adds or removes. Give exactly one of requested_amount, "
             "times_typical_payment or share_of_limit.", ScoreArgs, score_credit_increase),
    ToolSpec("get_graph_evidence", "Evidence from the knowledge graph for one customer: which months were late, weak "
             "payments, improvements, longest late streak, relapses (late, recovered, late again).", CustomerArgs, get_graph_evidence),
    ToolSpec("get_graph_schema", "The knowledge graph's structure, allowed values and example Cypher queries. Call this "
             "before writing a Cypher query.", NoArgs, get_graph_schema),
    ToolSpec("run_cypher", "Run ONE read-only Cypher query on the knowledge graph for questions the other tools cannot "
             "answer, e.g. counts across many customers. Call get_graph_schema first.", CypherArgs, run_cypher),
]}
CYPHER_TOOLS = {"run_cypher", "get_graph_schema"}


@dataclass
class ToolContext:
    settings: AgentSettings
    data: DataStore = field(default_factory=DataStore)
    _driver: object = None

    @property
    def database(self):
        from ..graph import database
        return database()

    @property
    def driver(self):
        if self._driver is None:
            if not self.settings.guardrails:
                # Guardrails off is only for the ON/OFF comparison, and only against a separate, disposable database.
                from neo4j import GraphDatabase
                import os
                uri = self.settings.disposable_neo4j_uri
                if not uri or uri == os.environ.get("NEO4J_URI"):
                    raise RuntimeError("guardrails are off: set AGENT_DISPOSABLE_NEO4J_URI to a separate, disposable database")
                self._driver = GraphDatabase.driver(uri, auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
            else:
                from ..graph import get_driver
                self._driver = get_driver()
        return self._driver

    def close(self):
        if self._driver is not None:
            self._driver.close()
            self._driver = None


def tool_schemas(enable_cypher: bool = True) -> list:
    """Tool definitions in the JSON-schema function format that Ollama (and most providers) accept."""
    out = []
    for t in TOOLS.values():
        if t.name in CYPHER_TOOLS and not enable_cypher:
            continue
        schema = t.args.model_json_schema()
        schema.pop("title", None)
        for p in schema.get("properties", {}).values():
            p.pop("title", None)
            if "anyOf" in p:                      # Optional[float] -> plain number for simpler models
                types = [a for a in p.pop("anyOf") if a.get("type") != "null"]
                p.update(types[0] if types else {})
                p.pop("default", None)
        out.append({"type": "function", "function": {"name": t.name, "description": t.description, "parameters": schema}})
    return out


def call_tool(ctx: ToolContext, name: str, raw_args, enable_cypher: bool = True) -> tuple[dict, dict]:
    """Validate and execute one tool call. Never raises: errors go back to the LLM as {"error": ...}.
    Returns (result_for_llm, meta_for_trace)."""
    t0 = time.perf_counter()
    meta = {"tool": name, "args": raw_args, "guardrail": None}
    try:
        if isinstance(raw_args, str):
            raw_args = json.loads(raw_args or "{}")
        if name not in TOOLS or (name in CYPHER_TOOLS and not enable_cypher):
            raise LookupError(f"unknown tool '{name}'")
        spec = TOOLS[name]
        args = spec.args.model_validate(raw_args or {})
        result = spec.fn(ctx, **args.model_dump())
        if name == "run_cypher":
            meta["guardrail"] = {"decision": "allowed", "guardrails_on": ctx.settings.guardrails}
    except CypherRejected as e:
        result = {"error": f"query rejected by guardrail ({e.category}): {e.detail}"}
        meta["guardrail"] = {"decision": "rejected", "category": e.category, "detail": e.detail}
    except ValidationError as e:
        result = {"error": "invalid arguments: " + "; ".join(err["msg"] for err in e.errors())}
    except Exception as e:  # noqa: BLE001 - every failure is reported to the LLM and the trace
        result = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    if name == "run_cypher" and "error" in result:
        # Small models skip "call get_graph_schema first" and give up after one failed query (observed with
        # qwen2.5:3b). Every failed query therefore comes back with the schema and examples attached.
        result.update({"what_to_do": "Fix the query using this schema and examples, then call run_cypher once more.",
                       "schema": GRAPH_SCHEMA})
    result = strip_protected(result)
    meta.update({"latency_s": round(time.perf_counter() - t0, 3), "result_chars": len(json.dumps(result, default=str)),
                 "error": result.get("error") if isinstance(result, dict) else None})
    return result, meta
