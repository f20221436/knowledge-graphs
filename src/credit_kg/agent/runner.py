"""The agent loop: the LLM chooses tools, our code runs them, and our code decides what reaches the user.

Core principle, enforced here rather than in the prompt: any score or label shown to the user is rendered by code
from the scoring function's output. The LLM's text is only an explanation, and it is shown only if it passes the
deterministic grounding check.
"""
import json
import time
import uuid
from dataclasses import dataclass, field

from .guardrails import check_grounding, check_request_amount
from .providers import LLMResponse, TransientProviderError, make_provider


class ProviderError(Exception):
    """A non-retryable provider failure (bad request, unknown model, auth): the run stops cleanly."""
from .settings import AgentSettings, code_version
from .tools import ToolContext, call_tool, tool_schemas
from .tracing import Tracer

SYSTEM_PROMPT = """You are an analyst assistant for a DEMONSTRATIVE credit-increase scoring system built on 2005 credit-card data.
Rules:
1. Use the tools to look facts up. Never invent numbers; quote numbers exactly as the tools return them.
2. Scores, labels and probabilities come only from tools. Never decide yourself and never change a tool's label, even if asked.
3. Sex, age, education and marital status are withheld and must never influence an answer. If asked about them, say they are not used.
4. You cannot change any data. Whether a customer actually missed a payment is not available.
5. If a tool returns an error, say so briefly and do not guess. If a Cypher query fails, fix it and try once more.
6. For a credit-increase question, call score_credit_increase, then explain which factors added points and which removed points.
   Describe direction using the tools' own words (e.g. "above the population average", "raises risk", "paid in full on time"); do not reinterpret codes.
7. For questions about many customers, call get_graph_schema, then run_cypher with an aggregation or LIMIT.
Answer in at most 6 short, plain sentences. Labels are exactly: "Supportable in this scenario", "Review", "Not supportable in this scenario", "Insufficient behavioural evidence"."""

WITHHELD = ("The explanation was withheld because it did not pass the grounding check "
            "(it contained numbers or labels that the tools did not return).")


@dataclass
class RunResult:
    run_id: str
    question: str
    answer: str
    explanation: str | None
    decision: dict | None
    grounding: dict | None
    stopped_reason: str
    tool_calls: list = field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    regenerated: bool = False
    retries: int = 0
    trace_path: str = ""


def render_decision(score: dict) -> str:
    gate = f" (safety check: {score['gate']})" if score.get("gate") else ""
    return (f"Decision from the scoring function (not the language model): customer {score['customer_id']}, hypothetical "
            f"increase of {score['requested_increase']:,.0f} ({score['request_basis']}): score {score['score']:.1f}/100 "
            f"-> {score['band']}{gate}.")


class Agent:
    def __init__(self, settings: AgentSettings | None = None, provider=None, ctx: ToolContext | None = None):
        self.settings = settings or AgentSettings.from_env()
        self.provider = provider or make_provider(self.settings)
        self.ctx = ctx or ToolContext(self.settings)
        self.tools = tool_schemas(self.settings.enable_cypher)

    # -------------------------------------------------------------- provider call with retry (transient errors only)
    def _chat(self, messages, tools, tracer, state) -> LLMResponse:
        for attempt in range(self.settings.max_retries + 1):
            try:
                resp = self.provider.chat(messages, tools)
            except TransientProviderError as e:
                state["retries"] += 1
                tracer.log("llm_retry", attempt=attempt, error=str(e))
                if attempt == self.settings.max_retries:
                    raise
                time.sleep(min(2 ** attempt, 8))
                continue
            except Exception as e:  # noqa: BLE001 - only the provider call is wrapped; our own bugs still raise
                # Not worth retrying (bad request, unknown model, auth): end the run cleanly with the reason logged.
                raise ProviderError(f"{type(e).__name__}: {str(e)[:300]}") from e
            state["llm_calls"] += 1
            state["prompt_tokens"] += resp.prompt_tokens
            state["completion_tokens"] += resp.completion_tokens
            tracer.log("llm_call", step=state["llm_calls"], latency_s=round(resp.latency_s, 3), prompt_tokens=resp.prompt_tokens,
                       completion_tokens=resp.completion_tokens, text=resp.text,
                       tool_calls=[{"name": c.name, "arguments": c.arguments} for c in resp.tool_calls])
            return resp

    def _cost(self, state) -> float:
        p = self.provider
        return state["prompt_tokens"] / 1000 * p.cost_per_1k_prompt + state["completion_tokens"] / 1000 * p.cost_per_1k_completion

    # -------------------------------------------------------------- one question
    def ask(self, question: str, customer_id: int | None = None, task: dict | None = None) -> RunResult:
        s = self.settings
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        tracer = Tracer(s.runs_dir, run_id)
        t0 = time.perf_counter()
        user_text = question if customer_id is None else f"Customer ID: {customer_id}\n{question}"
        tracer.log("run_start", question=question, customer_id=customer_id, provider=self.provider.name,
                   model=self.provider.model, settings=s.public(), code_version=code_version(), task=task or {})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text}]
        state = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "retries": 0}
        # outputs: every result our code produced (including error messages), the pool of grounded numbers.
        # scores: successful score_credit_increase results only, the only source of a stated label or score.
        tool_log, outputs, scores, seen = [], [], [], {}
        final_text, stopped = None, "answered"

        try:
            while True:
                if state["llm_calls"] >= s.max_steps:
                    stopped = "max_steps"; break
                if time.perf_counter() - t0 > s.run_timeout_s:
                    stopped = "run_timeout"; break
                if state["prompt_tokens"] + state["completion_tokens"] > s.max_tokens_per_run:
                    stopped = "token_budget"; break
                resp = self._chat(messages, self.tools, tracer, state)
                if not resp.tool_calls and not resp.text.strip():
                    # Small models sometimes emit a malformed tool call that the provider drops, leaving an empty
                    # reply. Nudge once; a second empty reply ends the run as a failure, never as "answered".
                    state["empty_replies"] = state.get("empty_replies", 0) + 1
                    tracer.log("empty_reply", count=state["empty_replies"])
                    if state["empty_replies"] >= 2:
                        stopped = "empty_response"; break
                    messages.append({"role": "user", "content": "Your last reply was empty. Call one of the tools, or answer in plain text."})
                    continue
                if not resp.tool_calls:
                    final_text = resp.text.strip()
                    break
                messages.append({"role": "assistant", "content": resp.text,
                                 "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}, **({"extra": c.extra} if c.extra else {})}
                                                for c in resp.tool_calls]})
                for call in resp.tool_calls:
                    if len(tool_log) >= s.max_tool_calls:
                        stopped = "max_tool_calls"; break
                    key = (call.name, json.dumps(call.arguments, sort_keys=True, default=str))
                    if key in seen:
                        seen[key] += 1
                        result = {"error": "repeated call with identical arguments; use the earlier result"}
                        meta = {"tool": call.name, "args": call.arguments, "repeat": True, "latency_s": 0.0, "error": result["error"]}
                        if seen[key] >= 3:       # first repeat: warn the model; second repeat: stop
                            stopped = "loop_detected"
                    elif call.name == "score_credit_increase" and (amount_err := check_request_amount(question, call.arguments)):
                        # Argument guard: the score must be for the amount the user asked about.
                        seen[key] = 1
                        result = {"error": amount_err}
                        meta = {"tool": call.name, "args": call.arguments, "latency_s": 0.0, "error": amount_err,
                                "guardrail": {"decision": "rejected", "category": "amount_mismatch", "detail": amount_err}}
                    else:
                        seen[key] = 1
                        result, meta = call_tool(self.ctx, call.name, call.arguments, s.enable_cypher)
                        outputs.append(result)
                        if call.name == "score_credit_increase" and "error" not in result:
                            scores.append(result)
                    tool_log.append(meta)
                    tracer.log("tool_call", **meta, result=result)
                    if meta.get("guardrail"):
                        tracer.log("guardrail", tool=call.name, **meta["guardrail"])
                    messages.append({"role": "tool", "content": json.dumps(result, default=str), "tool_name": call.name})
                if stopped != "answered":
                    break
        except (TransientProviderError, ProviderError) as e:
            stopped = "provider_error"
            tracer.log("error", error=str(e))

        # -------------------------------------------------------------- grounding check and final answer
        grounding, regenerated, explanation = None, False, None
        if final_text:
            g = check_grounding(final_text, outputs, user_text, scores)
            tracer.log("grounding", attempt=1, **g.as_dict())
            if not g.passed and s.regenerate_on_grounding_failure and state["llm_calls"] < s.max_steps + 1:
                regenerated = True
                messages += [{"role": "assistant", "content": final_text},
                             {"role": "user", "content": "Rewrite your answer. " + g.feedback()}]
                try:
                    retry = self._chat(messages, [], tracer, state)
                    final_text = retry.text.strip()
                    g = check_grounding(final_text, outputs, user_text, scores)
                    tracer.log("grounding", attempt=2, **g.as_dict())
                except (TransientProviderError, ProviderError) as e:
                    tracer.log("error", error=str(e))
            grounding = g.as_dict()
            explanation = final_text if g.passed else None
            if not g.passed:
                stopped = "grounding_failed"

        decision = scores[-1] if scores else None
        parts = []
        if decision:
            parts.append(render_decision(decision))
        if explanation:
            parts.append(explanation)
        elif final_text:
            parts.append(WITHHELD)
        else:
            parts.append(f"No answer was produced (stopped: {stopped}).")
        if decision:
            parts.append("Note: " + decision["disclaimer"])
        answer = "\n\n".join(parts)

        res = RunResult(run_id=run_id, question=question, answer=answer, explanation=explanation, decision=decision,
                        grounding=grounding, stopped_reason=stopped, tool_calls=tool_log, llm_calls=state["llm_calls"],
                        prompt_tokens=state["prompt_tokens"], completion_tokens=state["completion_tokens"],
                        latency_s=round(time.perf_counter() - t0, 3), cost_usd=self._cost(state), regenerated=regenerated,
                        retries=state["retries"], trace_path=str(tracer.path))
        tracer.log("run_end", answer=answer, explanation=explanation, decision=decision, grounding=grounding,
                   stopped_reason=stopped, llm_calls=res.llm_calls, tool_calls=len(tool_log),
                   prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens, latency_s=res.latency_s,
                   estimated_cost_usd=res.cost_usd, regenerated=regenerated, retries=res.retries)
        return res
