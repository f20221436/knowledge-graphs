"""Deterministic guardrails. None of these depend on the LLM following its instructions.

1. strip_protected: removes protected attributes, the outcome label and internal fields from anything sent to the LLM.
2. check_cypher_static / run_guarded_cypher: read-only Cypher behind layered checks.
3. check_grounding: verifies the LLM's final explanation against the tool outputs of the same run.
"""
import re
from dataclasses import dataclass, field

# Protected attributes, as named in the parquet files (sex, age, education, marriage) and in Neo4j (*_code).
PROTECTED_KEYS = {"sex", "age", "education", "marriage", "sex_code", "education_code", "marriage_code",
                  "sex_group", "age_group", "education_group", "marriage_group"}
OUTCOME_KEYS = {"default_next_month", "default.payment.next.month", "default", "target"}
INTERNAL_KEYS = {"split"}
HIDDEN_KEYS = PROTECTED_KEYS | OUTCOME_KEYS | INTERNAL_KEYS


def strip_protected(obj):
    """Recursively drop protected, outcome and internal keys from dicts (and dict-like rows inside lists)."""
    if isinstance(obj, dict):
        return {k: strip_protected(v) for k, v in obj.items() if str(k).lower() not in HIDDEN_KEYS}
    if isinstance(obj, (list, tuple)):
        return [strip_protected(v) for v in obj]
    return obj


# ---------------------------------------------------------------- Cypher
class CypherRejected(Exception):
    def __init__(self, category: str, detail: str):
        super().__init__(f"{category}: {detail}")
        self.category, self.detail = category, detail


_WRITE = r"\b(create|merge|delete|detach|set|remove|drop|foreach|grant|deny|revoke|alter|rename|start|stop|terminate|enable|disable)\b"
_PROCEDURE = r"\bcall\b|\bapoc\s*\.|\bdbms\s*\.|\bdb\s*\.|\bgds\s*\.|\bshow\b|\buse\b"
_PROP_NAMES = "|".join(sorted(PROTECTED_KEYS | {"sex_code", "education_code", "marriage_code"}, key=len, reverse=True))
_PROTECTED_ACCESS = (rf"\.\s*({_PROP_NAMES})\b"                     # n.age, n.`age`
                     rf"|\[\s*['\"]\s*({_PROP_NAMES})\s*['\"]\s*\]"    # n['age']
                     rf"|\b(keys|properties)\s*\("                      # keys(n), properties(n) reveal names/values
                     rf"|\{{\s*\.\s*\*")                                # map projection n {.*}
_OUTCOME_ACCESS = r"default|\btarget\b"


def _normalise(query: str) -> str:
    q = re.sub(r"//[^\n]*|/\*.*?\*/", " ", query, flags=re.S)   # comments cannot hide keywords
    return q.replace("`", "").lower()


def check_cypher_static(query: str) -> None:
    """Reject before the database sees the query. Fails closed: a keyword inside a string literal is also rejected."""
    if not query or not query.strip():
        raise CypherRejected("empty", "no query")
    q = _normalise(query)
    if ";" in q.strip().rstrip(";"):
        raise CypherRejected("multiple_statements", "only one statement is allowed")
    if re.search(r"\bload\s+csv\b", q):
        raise CypherRejected("load_csv", "LOAD CSV is not allowed")
    if re.search(_PROCEDURE, q):
        raise CypherRejected("procedure", "procedure calls, SHOW and USE are not allowed")
    if re.search(_WRITE, q):
        raise CypherRejected("write", "write and admin clauses are not allowed")
    if re.search(_PROTECTED_ACCESS, q):
        raise CypherRejected("protected_attribute", "protected attributes (sex, age, education, marital status) "
                                                    "and whole-property dumps are not queryable")
    if re.search(_OUTCOME_ACCESS, q):
        raise CypherRejected("outcome", "the outcome label is not queryable")


SCHEMA_EDGES = {("Customer", "HAS_ACCOUNT", "CreditAccount"),
                ("CreditAccount", "HAS_MONTHLY_RECORD", "MonthlyCreditRecord"),
                ("MonthlyCreditRecord", "NEXT_MONTH", "MonthlyCreditRecord"),
                ("MonthlyCreditRecord", "IN_STATE", "RepaymentState")}
_EDGE = re.compile(r"\(\s*\w*\s*:\s*(\w+)[^)]*\)\s*(<?)-\s*\[\s*\w*\s*:\s*(\w+)[^\]]*\]\s*-(>?)\s*\(\s*\w*\s*:\s*(\w+)")


def check_cypher_schema(query: str) -> None:
    """Correctness, not security: reject a pattern whose two labelled endpoints the graph never connects with that
    relationship. Such queries run fine and silently return 0 rows, which a model then reports as a real answer.
    Only explicit (a:Label)-[:TYPE]->(b:Label) hops are checked; anything less explicit is left to the database."""
    for m in _EDGE.finditer(query.replace("`", "")):
        left, back, rel, fwd, right = m.groups()
        if bool(back) != bool(fwd):                       # directed: (a)-[:R]->(b) or (a)<-[:R]-(b)
            src, dst = (right, left) if back else (left, right)
            ok = (src, rel, dst) in SCHEMA_EDGES
        else:                                             # undirected: (a)-[:R]-(b) matches either direction
            src, dst = left, right
            ok = (left, rel, right) in SCHEMA_EDGES or (right, rel, left) in SCHEMA_EDGES
        if not ok:
            raise CypherRejected("schema_mismatch",
                                 f"the graph has no ({src})-[:{rel}]->({dst}) relationship. Valid paths: "
                                 "(:Customer)-[:HAS_ACCOUNT]->(:CreditAccount)-[:HAS_MONTHLY_RECORD]->(:MonthlyCreditRecord); "
                                 "MonthlyCreditRecord also has a customer_id property, so per-month counts can match it directly.")


def _to_plain(value):
    """Neo4j values -> plain Python. Nodes/relationships become property dicts (then stripped)."""
    if hasattr(value, "items") and hasattr(value, "element_id"):
        return {k: _to_plain(v) for k, v in dict(value).items()}
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if hasattr(value, "nodes") and hasattr(value, "relationships"):   # Path
        return {"nodes": [_to_plain(n) for n in value.nodes]}
    return value


def run_guarded_cypher(driver, query: str, database=None, row_cap: int = 50, timeout_s: float = 5.0,
                       guardrails: bool = True) -> dict:
    """Execute one read-only query. With guardrails on, every layer must pass:
    static checks -> EXPLAIN must report query_type 'r' -> read transaction with a timeout -> row cap -> stripping."""
    from neo4j import unit_of_work

    if guardrails:
        check_cypher_static(query)
        with driver.session(database=database) as s:
            qtype = s.run("EXPLAIN " + query).consume().query_type
        if qtype != "r":
            raise CypherRejected("not_read_only", f"EXPLAIN reports query type '{qtype}', only 'r' is allowed")

    @unit_of_work(timeout=timeout_s)
    def _read(tx):
        result = tx.run(query)
        rows, truncated = [], False
        for rec in result:
            if len(rows) >= row_cap:
                truncated = True
                break
            rows.append({k: _to_plain(v) for k, v in rec.items()})
        result.consume()
        return rows, truncated

    with driver.session(database=database) as s:
        rows, truncated = (s.execute_read(_read) if guardrails else s.execute_write(_read))
    if guardrails:
        rows = strip_protected(rows)
    return {"rows": rows, "row_count": len(rows), "truncated": truncated, "row_cap": row_cap}


# ---------------------------------------------------------------- grounding
BAND_PATTERNS = {
    "Supportable in this scenario": r"(?<!not )(?<!un)\bsupportable\b",
    "Review": r"\bReview\b",          # case-sensitive (checked on the original text): the verb "review" is not a label
    "Not supportable in this scenario": r"\bnot supportable\b",
    "Insufficient behavioural evidence": r"\binsufficient (behaviou?ral )?evidence\b",
}
# Words that make mentioning a *different* band legitimate ("one point short of Supportable").
_COMPARATIVE = r"short of|threshold|cut-?off|line\b|would|could|if\b|needs?|to reach|become|instead|rather than|above|below|move"
_PROTECTED_WORDS = (r"\b(sex|gender|male|female|men|women|woman|age|aged|young|younger|elderly|"
                    r"education|educated|graduate|university|high school|marital|married|divorced)\b")
_NEGATION = r"\b(not|never|no|cannot|can't|can not|don't|do not|doesn't|does not|isn't|is not|won't|without|excluded|exclude|refuse|unable)\b"
_NUMBER = re.compile(r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)(\s?%)?")


def _numbers_in(obj, out: set):
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, str):
        for m in _NUMBER.finditer(obj):
            out.add(float(m.group(1).replace(",", "")))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _numbers_in(k, out)
            _numbers_in(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers_in(v, out)
    return out


def _matches(value: float, decimals: int, percent: bool, allowed: set) -> bool:
    tol = 10 ** -decimals if decimals else 0.5
    candidates = [value]
    if percent:
        candidates.append(value / 100)
    for cand in candidates:
        t = tol / 100 if (percent and cand is not value) else tol
        if any(abs(cand - a) <= t + 1e-9 for a in allowed):
            return True
    if not percent and decimals == 0:                     # "35" for 0.35 written without a % sign
        if any(abs(value / 100 - a) <= 0.005 + 1e-9 for a in allowed):
            return True
    return False


def _sentences(text: str):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


@dataclass
class GroundingResult:
    passed: bool
    ungrounded_numbers: list = field(default_factory=list)
    label_problems: list = field(default_factory=list)
    protected_mentions: list = field(default_factory=list)
    outcome_claims: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"passed": self.passed, "ungrounded_numbers": self.ungrounded_numbers,
                "label_problems": self.label_problems, "protected_mentions": self.protected_mentions,
                "outcome_claims": self.outcome_claims}

    def feedback(self) -> str:
        parts = []
        if self.ungrounded_numbers:
            parts.append(f"These numbers do not appear in any tool output: {', '.join(self.ungrounded_numbers)}. "
                         "Only use numbers exactly as the tools returned them (rounding is fine).")
        if self.label_problems:
            parts.append("Label problems: " + " ".join(self.label_problems))
        if self.outcome_claims:
            parts.append("Whether the customer actually missed or paid the next payment is NOT available to you; say so instead of "
                         "claiming it: " + " | ".join(self.outcome_claims))
        if self.protected_mentions:
            parts.append("Do not use sex, age, education or marital status as a reason: " + " | ".join(self.protected_mentions))
        return " ".join(parts)


def check_grounding(answer: str, tool_outputs: list, question: str = "", score_results: list | None = None) -> GroundingResult:
    """tool_outputs: every tool result of the run (dicts). score_results: outputs of score_credit_increase only."""
    score_results = score_results or []
    allowed = _numbers_in(tool_outputs, set()) | _numbers_in(question, set())
    # Magnitudes count too: "lowers risk by 0.0454" for a contribution of -0.0454 states the sign in words.
    allowed |= {abs(a) for a in allowed} | {0.0, 1.0}

    bad_numbers = []
    for m in _NUMBER.finditer(answer):
        raw, pct = m.group(1), bool(m.group(2))
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if not _matches(float(raw.replace(",", "")), decimals, pct, allowed):
            bad_numbers.append(raw + ("%" if pct else ""))

    label_problems = []
    actual = {r["band"] for r in score_results if "band" in r}
    for sent in _sentences(answer):
        low = sent.lower()
        mentioned = {band for band, pat in BAND_PATTERNS.items() if re.search(pat, sent if band == "Review" else low)}
        if "Not supportable in this scenario" in mentioned:
            mentioned.discard("Supportable in this scenario")
        # A stated score must be the scoring function's score. The band thresholds (50, 70) and the maximum (100)
        # are accepted only in comparative sentences ("needs 70 to be Supportable").
        m = re.search(r"\bscor(?:e|es|ed|ing)\b(?:\s+(?:is|was|of|at|only|just|=|:))*\s*(\d+(?:\.\d+)?)", low)
        if m:
            val = float(m.group(1))
            ok = {float(r["score"]) for r in score_results} | {float(round(r["score"])) for r in score_results}
            if re.search(_COMPARATIVE, low):
                ok |= {50.0, 70.0, 100.0}
            if not score_results:
                label_problems.append(f"States a score without calling score_credit_increase: \"{sent.strip()[:120]}\"")
            elif not any(abs(val - o) <= 0.05 + 1e-9 for o in ok):
                label_problems.append(f"Score {val:g} does not match the scoring function")
        if not mentioned:
            continue
        if not actual:
            label_problems.append(f"States a label without calling score_credit_increase: \"{sent.strip()[:120]}\"")
            continue
        wrong = mentioned - actual
        if wrong and not (mentioned & actual) and not re.search(_COMPARATIVE, low):
            label_problems.append(f"Label does not match the scoring function ({', '.join(sorted(actual))}): \"{sent.strip()[:120]}\"")

    protected = [s.strip()[:120] for s in _sentences(answer)
                 if re.search(_PROTECTED_WORDS, s.lower()) and not re.search(_NEGATION, s.lower())]
    outcome = [s.strip()[:120] for s in _sentences(answer)
               if _OUTCOME_ASSERT.search(s) and not _OUTCOME_UNAVAILABLE.search(s)]

    return GroundingResult(passed=not (bad_numbers or label_problems or protected or outcome), ungrounded_numbers=bad_numbers,
                           label_problems=label_problems, protected_mentions=protected, outcome_claims=outcome)


# The real outcome (did the customer miss the October payment?) is never given to the agent, so any sentence claiming
# it is invented. Statements about past months ("did not pay in 2005-05") and predictions with a probability are fine.
_OUTCOME_ASSERT = re.compile(
    r"\b(did|has|had|have)(n't| not)? (actually )?(miss|default|pay|make)\w*\b[^.]*\b(next|october|following|2005-10)\b"
    r"|\b(actually )?defaulted\b|\bwill (not )?(default|miss)\b|\bmissed (their|the|its|his|her) (next|october) payment", re.I)
_OUTCOME_UNAVAILABLE = re.compile(r"not (available|known|provided|accessible)|unavailable|cannot|can't|can not|don't know|"
                                  r"do not know|no (information|data|access)|unable|not possible to (know|say|tell)", re.I)


_AMOUNT = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d{4,})(?![\w.,]\d)")


def requested_amounts(question: str) -> set:
    """Explicit money amounts in the user's question (e.g. '30,000' or '30000'), ignoring customer IDs and years."""
    q = re.sub(r"customer\s*(id)?\s*[:#]?\s*\d+", " ", question or "", flags=re.I)
    vals = {float(m.group(1).replace(",", "")) for m in _AMOUNT.finditer(q)}
    return {v for v in vals if not (1990 <= v <= 2030 and "," not in str(v))}


def check_request_amount(question: str, args: dict) -> str | None:
    """Argument guard for score_credit_increase: if the question names an amount, the call must use exactly that
    amount. Observed with qwen2.5:3b: asked about 30,000, it scored 19,500 (a multiple of the typical payment)."""
    wanted = requested_amounts(question)
    if not wanted:
        return None
    amount = args.get("requested_amount")
    if amount is not None and any(abs(float(amount) - w) < 0.5 for w in wanted):
        return None
    shown = ", ".join(f"{w:,.0f}" for w in sorted(wanted))
    return (f"the question asks about an increase of {shown}; call score_credit_increase again with "
            f"requested_amount={sorted(wanted)[0]:.0f} (not a multiple of payment or share of limit)")
