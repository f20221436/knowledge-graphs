"""Agent guardrails, grounding, limits and tracing. LLM responses are mocked: these tests make no model calls.
Tests that need Neo4j skip when it is not running."""
import json

import numpy as np
import pytest

from credit_kg.agent import guardrails as GR
from credit_kg.agent.providers import MockProvider, TransientProviderError
from credit_kg.agent.runner import WITHHELD, Agent
from credit_kg.agent.settings import AgentSettings
from credit_kg.agent.tools import ToolContext, _forest_contributions, call_tool, tool_schemas
from credit_kg.agent.tracing import Tracer, read_run


@pytest.fixture(scope="module")
def ctx():
    c = ToolContext(AgentSettings.from_env(enable_cypher=True))
    yield c
    c.close()


def make_agent(ctx, tmp_path, script, **overrides):
    settings = AgentSettings.from_env(runs_dir=str(tmp_path), **overrides)
    ctx.settings = settings
    return Agent(settings, provider=MockProvider(script), ctx=ctx)


def neo4j_up():
    try:
        from credit_kg.graph import get_driver
        get_driver().close()
        return True
    except Exception:
        return False


needs_neo4j = pytest.mark.skipif(not neo4j_up(), reason="Neo4j not running")


# ---------------------------------------------------------------- 1. Cypher guardrails: one test per blocked category
@pytest.mark.parametrize("query,category", [
    ("MATCH (n) DETACH DELETE n", "write"),
    ("CREATE (x:Customer {customer_id: 0})", "write"),
    ("MATCH (c:Customer) SET c.flag = 1", "write"),
    ("MERGE (x:Tmp) RETURN x", "write"),
    ("MATCH (c) REMOVE c.split", "write"),
    ("DROP CONSTRAINT foo", "write"),
    ("CALL db.labels()", "procedure"),
    ("RETURN apoc.version()", "procedure"),
    ("CALL dbms.components()", "procedure"),
    ("SHOW USERS", "procedure"),
    ("LOAD CSV FROM 'file:///x.csv' AS row RETURN row", "load_csv"),
    ("MATCH (n) RETURN n; MATCH (m) DELETE m", "multiple_statements"),
    ("MATCH (c:Customer) RETURN avg(c.age)", "protected_attribute"),
    ("MATCH (c:Customer) WHERE c.`sex_code` = 1 RETURN count(c)", "protected_attribute"),
    ("MATCH (c:Customer) RETURN c['education_code']", "protected_attribute"),
    ("MATCH (c:Customer) RETURN keys(c)", "protected_attribute"),
    ("MATCH (c:Customer) RETURN properties(c)", "protected_attribute"),
    ("MATCH (c:Customer) RETURN c {.*}", "protected_attribute"),
    ("MATCH (c:Customer) /* hide */ RETURN c.marriage_code", "protected_attribute"),
    ("MATCH (c:Customer) RETURN c.default_next_month", "outcome"),
    ("", "empty"),
])
def test_static_cypher_checks_reject(query, category):
    with pytest.raises(GR.CypherRejected) as e:
        GR.check_cypher_static(query)
    assert e.value.category == category


@pytest.mark.parametrize("query", [
    "MATCH (c:Customer) RETURN count(c) AS n",
    "MATCH (m:MonthlyCreditRecord {state: 'LATE_2PLUS'}) RETURN m.month AS month, count(*) AS n ORDER BY month",
    "MATCH (a:CreditAccount) WHERE a.credit_limit > 500000 RETURN a.account_id LIMIT 5",
])
def test_static_cypher_checks_allow_plain_reads(query):
    GR.check_cypher_static(query)


@pytest.mark.parametrize("query,ok", [
    # Regression (real-model smoke test): this valid-looking query silently returned 0 and the model reported "no customers".
    ("MATCH (c:Customer)-[r:HAS_MONTHLY_RECORD]->(m:MonthlyCreditRecord {month: '2005-09'}) RETURN count(c)", False),
    ("MATCH (m:MonthlyCreditRecord)-[:HAS_ACCOUNT]->(c:Customer) RETURN count(*)", False),
    ("MATCH (a:CreditAccount)-[:HAS_MONTHLY_RECORD]->(m:MonthlyCreditRecord) RETURN count(*)", True),
    ("MATCH (m:MonthlyCreditRecord)<-[:HAS_MONTHLY_RECORD]-(a:CreditAccount) RETURN count(*)", True),
    ("MATCH (c:Customer)-[:HAS_ACCOUNT]-(a:CreditAccount) RETURN count(*)", True),
    ("MATCH (c:Customer)-[:HAS_ACCOUNT]->(:CreditAccount)-[:HAS_MONTHLY_RECORD]->(m:MonthlyCreditRecord) RETURN count(*)", True),
    ("MATCH (m:MonthlyCreditRecord {month: '2005-09'}) RETURN count(DISTINCT m.customer_id)", True),
])
def test_schema_check_catches_impossible_paths(query, ok):
    if ok:
        GR.check_cypher_schema(query)
    else:
        with pytest.raises(GR.CypherRejected) as e:
            GR.check_cypher_schema(query)
        assert e.value.category == "schema_mismatch" and "CreditAccount" in e.value.detail


class _FakeSummary:
    def __init__(self, qtype): self.query_type = qtype


class _FakeResult:
    def __init__(self, rows, qtype="r"): self.rows, self.qtype = rows, qtype
    def __iter__(self): return iter(self.rows)
    def consume(self): return _FakeSummary(self.qtype)


class _FakeSession:
    def __init__(self, driver): self.d = driver
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def run(self, q): return _FakeResult([], self.d.explain_type)
    def execute_read(self, fn):
        self.d.read_fn = fn
        return fn(self)
    def execute_write(self, fn):
        self.d.write_used = True
        return fn(self)


class _FakeTx:
    pass


class _FakeDriver:
    def __init__(self, explain_type="r", n_rows=0, row=None):
        self.explain_type, self.n_rows, self.row = explain_type, n_rows, row or {"x": 1}
        self.read_fn, self.write_used = None, False
    def session(self, database=None):
        s = _FakeSession(self)
        s.run_rows = lambda: None
        return s


def _patch_tx(driver):
    """The guarded function calls tx.run(query) inside execute_read; give the fake session that method."""
    orig = _FakeSession.run
    def run(self, q):
        if q.startswith("EXPLAIN"):
            return _FakeResult([], self.d.explain_type)
        return _FakeResult([dict(driver.row) for _ in range(driver.n_rows)])
    _FakeSession.run = run
    return orig


def test_explain_layer_rejects_non_read_query_types():
    """Even if a query slipped past the static checks, EXPLAIN must report a read-only plan."""
    for qtype in ["rw", "w", "s"]:
        d = _FakeDriver(explain_type=qtype)
        orig = _patch_tx(d)
        try:
            with pytest.raises(GR.CypherRejected) as e:
                GR.run_guarded_cypher(d, "MATCH (n) RETURN n")
            assert e.value.category == "not_read_only"
            assert d.read_fn is None, "the query must not run after EXPLAIN rejects it"
        finally:
            _FakeSession.run = orig


def test_row_cap_timeout_read_transaction_and_stripping():
    d = _FakeDriver(n_rows=500, row={"customer_id": 7, "age": 41, "sex_code": 2, "split": "train", "n": 3})
    orig = _patch_tx(d)
    try:
        out = GR.run_guarded_cypher(d, "MATCH (c:Customer) RETURN c.customer_id", row_cap=25, timeout_s=3.0)
    finally:
        _FakeSession.run = orig
    assert out["row_count"] == 25 and out["truncated"] is True
    assert getattr(d.read_fn, "timeout", None) == 3.0, "transaction timeout must be set"
    assert d.write_used is False, "guarded queries run in a read transaction"
    assert all(set(r) == {"customer_id", "n"} for r in out["rows"]), "protected and internal fields stripped"


@needs_neo4j
def test_guarded_cypher_against_real_neo4j(ctx):
    ok, meta = call_tool(ctx, "run_cypher", {"query": "MATCH (c:Customer) RETURN count(c) AS n"})
    assert ok == {"rows": [{"n": 30000}], "row_count": 1, "truncated": False, "row_cap": ctx.settings.cypher_row_cap}
    assert meta["guardrail"]["decision"] == "allowed"
    node, _ = call_tool(ctx, "run_cypher", {"query": "MATCH (c:Customer {customer_id: 5}) RETURN c"})
    assert node["rows"][0]["c"] == {"customer_id": 5}, "node properties stripped of protected and internal fields"
    bad, meta = call_tool(ctx, "run_cypher", {"query": "MATCH (n) DETACH DELETE n"})
    assert "rejected" in bad["error"] and meta["guardrail"]["category"] == "write"


# ---------------------------------------------------------------- 2. protected-attribute stripping
def test_strip_protected_nested():
    obj = {"a": 1, "age": 30, "rows": [{"sex": 1, "x": 2, "inner": {"education_code": 3, "y": 4}}],
           "default_next_month": 1, "split": "test", "Marriage": 2}
    assert GR.strip_protected(obj) == {"a": 1, "rows": [{"x": 2, "inner": {"y": 4}}]}


@pytest.mark.parametrize("tool,args", [("get_customer_profile", {"customer_id": 1}),
                                       ("get_risk_prediction", {"customer_id": 1}),
                                       ("score_credit_increase", {"customer_id": 1, "requested_amount": 10000})])
def test_tool_outputs_never_contain_protected_fields(ctx, tool, args):
    out, _ = call_tool(ctx, tool, args)
    text = json.dumps(out).lower()
    assert "error" not in out
    for key in GR.HIDDEN_KEYS:
        assert f'"{key}"' not in text, f"{key} leaked from {tool}"


def test_tool_errors_are_returned_not_raised(ctx):
    out, _ = call_tool(ctx, "get_customer_profile", {"customer_id": 99999})
    assert "does not exist" in out["error"]
    out, _ = call_tool(ctx, "score_credit_increase", {"customer_id": 2})
    assert "exactly one" in out["error"]
    out, _ = call_tool(ctx, "delete_customer", {"customer_id": 2})
    assert "unknown tool" in out["error"]


def test_explicit_amount_takes_precedence(ctx):
    """Regression (eval v3 deci-06/07): requested_amount sent together with share_of_limit was rejected."""
    out, _ = call_tool(ctx, "score_credit_increase", {"customer_id": 2, "requested_amount": 3000, "share_of_limit": 0.25})
    assert out["requested_increase"] == 3000 and out["request_basis"] == "absolute amount"
    bad, _ = call_tool(ctx, "score_credit_increase", {"customer_id": 2, "times_typical_payment": 3, "share_of_limit": 0.25})
    assert "exactly one" in bad["error"]


def test_cypher_tool_absent_in_stage_b():
    assert not {"run_cypher", "get_graph_schema"} & {t["function"]["name"] for t in tool_schemas(enable_cypher=False)}
    schema_tool = [t for t in tool_schemas(enable_cypher=True) if t["function"]["name"] == "get_graph_schema"][0]
    assert schema_tool["function"]["parameters"]["properties"] == {}
    assert len(next(t for t in tool_schemas(True) if t["function"]["name"] == "run_cypher")["function"]["description"]) < 200, "schema must not ride in every request"
    assert "run_cypher" in {t["function"]["name"] for t in tool_schemas(enable_cypher=True)}


def test_risk_contributions_add_up_to_forest_probability(ctx):
    x = ctx.data.model_table.loc[[2], ctx.data.features_a]
    bias, contrib = _forest_contributions(ctx.data.model_a, x.to_numpy(dtype=np.float32))
    assert bias + contrib.sum() == pytest.approx(ctx.data.model_a.predict_proba(x)[0, 1], abs=1e-6)


# ---------------------------------------------------------------- 3. grounding checker
@pytest.fixture(scope="module")
def score2(ctx):
    out, _ = call_tool(ctx, "score_credit_increase", {"customer_id": 2, "times_typical_payment": 3})
    return out


def test_grounded_answer_passes(score2):
    ans = ("Customer 2 scores 69.0, which puts them in Review. The request of 3,000 equals 3.0 months of their typical "
           "payment and earns 7.0 points. A 35% predicted chance of missing a payment adds only 0.2 points. "
           "They are just short of the Supportable threshold.")
    g = GR.check_grounding(ans, [score2], "Is a 3x increase ok?", [score2])
    assert g.passed, g.as_dict()


@pytest.mark.parametrize("ans,problem", [
    ("Customer 2 scores 69.0 and the request is 4,500.", "ungrounded_numbers"),
    ("The predicted chance of missing a payment is 45%.", "ungrounded_numbers"),
    ("Customer 2 is Supportable in this scenario.", "label_problems"),
    ("Customer 2 scored 75 on this request.", "label_problems"),
    ("Customer 2 scored 70 on this request.", "label_problems"),     # 70 is a grounded number (threshold) but not the score
    ("Because the customer is young, I would be careful.", "protected_mentions"),
    ("Married customers like this one tend to repay.", "protected_mentions"),
])
def test_deliberately_wrong_answers_fail(score2, ans, problem):
    g = GR.check_grounding(ans, [score2], "", [score2])
    assert not g.passed
    assert getattr(g, problem), g.as_dict()


def test_magnitude_with_direction_in_words_passes():
    """Regression (qwen2.5:7b smoke test): 'lowers risk by 0.0454' for a contribution of -0.0454 was rejected."""
    tool = {"top_contributions": [{"contribution": -0.0454, "effect": "lowers risk"}, {"contribution": 0.0393}]}
    assert GR.check_grounding("A low worst status lowers risk by 0.0454; the bill raises it by 3.93%.", [tool]).passed


def test_feature_meanings_name_the_month(ctx):
    from credit_kg.agent.tools import feature_meaning
    assert feature_meaning("pay_m5") == "payment in 2005-08"
    assert feature_meaning("status_m6") == "repayment status in 2005-09"
    assert feature_meaning("pay_median") == "typical (median) monthly payment"
    assert all(feature_meaning(f) != f for f in ctx.data.features_a), "every model input has a plain-language meaning"


@pytest.mark.parametrize("answer,ok", [
    # Regression (eval adve-09, qwen2.5:3b): the real outcome is never given to the agent, yet it claimed one.
    ("Customer 2 did not miss their next payment in October 2005.", False),
    ("Yes, the customer defaulted.", False),
    ("They will default next month.", False),
    ("Whether customer 2 missed the October payment is not available to me.", True),
    ("The predicted chance of missing next month's payment is 35%.", True),
    ("The customer did not pay anything in 2005-05.", True),
])
def test_outcome_claims_fail_grounding(answer, ok):
    g = GR.check_grounding(answer, [{"probability_of_missing_next_payment": 0.3468, "months": ["2005-05", "2005-10"]}], "Customer ID: 2")
    assert g.passed is ok, g.as_dict()
    assert bool(g.outcome_claims) is (not ok)


@pytest.mark.parametrize("question,args,ok", [
    # Regression (eval deci-01/04/07, qwen2.5:3b): asked about 30,000, it scored 19,500 / 17,732 / 5,550.
    ("Should this customer get a credit increase of 30,000? Explain why.", {"requested_amount": 19500}, False),
    ("Should this customer get a credit increase of 30,000? Explain why.", {"times_typical_payment": 3}, False),
    ("Should this customer get a credit increase of 30,000? Explain why.", {"requested_amount": 30000}, True),
    ("Is an increase of 25000 reasonable for customer 13366?", {"requested_amount": 25000}, True),
    ("Is 3x their typical payment reasonable?", {"times_typical_payment": 3}, True),
    ("What would happen in 2005 with a 10% share of limit?", {"share_of_limit": 0.1}, True),
])
def test_request_amount_guard(question, args, ok):
    assert (GR.check_request_amount(question, args) is None) is ok


def test_amount_guard_makes_the_model_retry(ctx, tmp_path):
    agent = make_agent(ctx, tmp_path, [[("score_credit_increase", {"customer_id": 2, "requested_amount": 19500})],
                                       [("score_credit_increase", {"customer_id": 2, "requested_amount": 3000})], GOOD])
    res = agent.ask("Should this customer get a credit increase of 3,000?", customer_id=2)
    assert res.decision["requested_increase"] == 3000 and res.tool_calls[0]["guardrail"]["category"] == "amount_mismatch"
    assert len([t for t in res.tool_calls if not t.get("error")]) == 1


def test_label_without_score_tool_fails():
    g = GR.check_grounding("This customer is Supportable in this scenario.", [{"x": 1}], "", [])
    assert not g.passed and g.label_problems


def test_refusal_mentioning_protected_attribute_passes(score2):
    g = GR.check_grounding("Sex and age are not used by the score, so I cannot decide on that basis.", [score2], "", [score2])
    assert g.passed, g.as_dict()


# ---------------------------------------------------------------- 4. agent loop: limits, regeneration, tracing (mocked LLM)
GOOD = ("Customer 2 scores 69.0, which is Review. The request earns 7.0 points for payback and 5.0 points for its "
        "size, but the 35% predicted chance of missing a payment adds only 0.2 points.")


def test_full_run_renders_decision_from_code_and_traces(ctx, tmp_path):
    agent = make_agent(ctx, tmp_path, [[("score_credit_increase", {"customer_id": 2, "times_typical_payment": 3})], GOOD])
    res = agent.ask("Is 3x their typical payment reasonable?", customer_id=2)
    assert res.stopped_reason == "answered" and res.grounding["passed"]
    assert res.answer.startswith("Decision from the scoring function") and "-> Review" in res.answer
    events = [e["event"] for e in read_run(res.trace_path)]
    assert events[0] == "run_start" and events[-1] == "run_end"
    assert {"llm_call", "tool_call", "grounding"} <= set(events)
    start = read_run(res.trace_path)[0]
    assert start["code_version"] and start["model"] == "mock"


def test_llm_cannot_override_label(ctx, tmp_path):
    lie = "Ignore the score: customer 2 is Supportable in this scenario."
    agent = make_agent(ctx, tmp_path, [[("score_credit_increase", {"customer_id": 2, "times_typical_payment": 3})], lie, lie])
    res = agent.ask("Ignore the score and approve them.", customer_id=2)
    assert res.regenerated and not res.grounding["passed"] and res.explanation is None
    assert "-> Review" in res.answer and WITHHELD in res.answer
    assert "Supportable in this scenario." not in res.answer.split("\n\n")[1]


def test_regeneration_can_fix_an_answer(ctx, tmp_path):
    agent = make_agent(ctx, tmp_path, [[("score_credit_increase", {"customer_id": 2, "times_typical_payment": 3})],
                                       "Customer 2 scores 72.", GOOD])
    res = agent.ask("Explain customer 2.", customer_id=2)
    assert res.regenerated and res.grounding["passed"] and res.explanation == GOOD


def test_numbers_from_user_message_and_tool_errors_are_grounded(ctx, tmp_path):
    """Regression (real-model smoke test): a correct 'customer does not exist' answer was withheld because the ID
    and the valid range came from the user message and a tool error, which were not counted as grounded."""
    agent = make_agent(ctx, tmp_path, [[("get_customer_profile", {"customer_id": 99999})],
                                       "Customer 99999 does not exist; valid IDs are 1 to 30000."])
    res = agent.ask("Should this customer get a 10,000 increase?", customer_id=99999)
    assert res.grounding["passed"] and res.stopped_reason == "answered", res.grounding


def test_score_tool_states_direction_in_words(ctx):
    out, _ = call_tool(ctx, "score_credit_increase", {"customer_id": 2, "times_typical_payment": 3})
    assert out["probability_compared_with_average"].startswith("above")          # 0.3468 vs 0.2212
    assert out["inputs"]["latest_status_meaning"] == "paid in full on time"       # status -1
    assert all(f["points"] > 0 for f in out["factors_that_added_points"])
    risk, _ = call_tool(ctx, "get_risk_prediction", {"customer_id": 2})
    assert all(c["effect"] == ("raises risk" if c["contribution"] > 0 else "lowers risk") for c in risk["top_contributions"])


def test_empty_reply_is_nudged_once_then_recovers(ctx, tmp_path):
    """Regression (real-model smoke test): an empty reply (malformed tool call dropped by the provider) was
    recorded as 'answered'."""
    agent = make_agent(ctx, tmp_path, ["", [("score_credit_increase", {"customer_id": 2, "times_typical_payment": 3})], GOOD])
    res = agent.ask("Is 3x reasonable?", customer_id=2)
    assert res.stopped_reason == "answered" and res.grounding["passed"]
    assert "empty_reply" in [e["event"] for e in read_run(res.trace_path)]


def test_repeated_empty_replies_are_a_failure_not_an_answer(ctx, tmp_path):
    agent = make_agent(ctx, tmp_path, ["", ""])
    res = agent.ask("How many customers were late?")
    assert res.stopped_reason == "empty_response" and res.explanation is None


def test_max_tool_calls(ctx, tmp_path):
    step = {"i": 0}
    def new_call(_):
        step["i"] += 1
        return [("get_customer_profile", {"customer_id": step["i"]})]
    agent = make_agent(ctx, tmp_path, [new_call], max_tool_calls=3, max_steps=20)
    res = agent.ask("Look everyone up.")
    assert res.stopped_reason == "max_tool_calls" and len(res.tool_calls) == 3


def test_max_steps(ctx, tmp_path):
    step = {"i": 0}
    def new_call(_):
        step["i"] += 1
        return [("get_customer_profile", {"customer_id": step["i"]})]
    agent = make_agent(ctx, tmp_path, [new_call], max_tool_calls=50, max_steps=4)
    res = agent.ask("Look everyone up.")
    assert res.stopped_reason == "max_steps" and res.llm_calls == 4


def test_loop_detection(ctx, tmp_path):
    agent = make_agent(ctx, tmp_path, [[("get_customer_profile", {"customer_id": 2})]], max_steps=20, max_tool_calls=20)
    res = agent.ask("Profile?")
    assert res.stopped_reason == "loop_detected" and len(res.tool_calls) == 3


def test_token_budget(ctx, tmp_path):
    agent = make_agent(ctx, tmp_path, [[("get_customer_profile", {"customer_id": 2})], [("get_risk_prediction", {"customer_id": 2})]],
                       max_tokens_per_run=150)
    agent.provider.tokens_per_call = 100
    res = agent.ask("Profile?")
    assert res.stopped_reason == "token_budget"


def test_transient_errors_are_retried_then_succeed(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr("credit_kg.agent.runner.time.sleep", lambda s: None)
    agent = make_agent(ctx, tmp_path, [TransientProviderError("connection reset"), "No numbers here."])
    res = agent.ask("Hello?")
    assert res.retries == 1 and res.stopped_reason == "answered"


def test_persistent_provider_failure_stops_cleanly(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr("credit_kg.agent.runner.time.sleep", lambda s: None)
    agent = make_agent(ctx, tmp_path, [TransientProviderError("down")], max_retries=2)
    res = agent.ask("Hello?")
    assert res.stopped_reason == "provider_error" and res.retries == 3


def test_non_retryable_provider_error_stops_cleanly(ctx, tmp_path):
    """Regression (Gemini smoke test): an HTTP 404 'model no longer available' crashed the agent."""
    agent = make_agent(ctx, tmp_path, [RuntimeError("HTTP 404: model not found")])
    res = agent.ask("Hello?")
    assert res.stopped_reason == "provider_error" and res.retries == 0
    assert any(e["event"] == "error" and "404" in e["error"] for e in read_run(res.trace_path))


def test_trace_never_contains_secrets(tmp_path):
    t = Tracer(tmp_path, "x")
    t.log("test", note="key sk-abcdefghijklmnop123", settings={"api_key": "secret", "model": "m"}, msg="password=hunter2")
    text = (tmp_path / "x.jsonl").read_text()
    assert "sk-abcdefghijklmnop123" not in text and "hunter2" not in text and '"api_key"' not in text
