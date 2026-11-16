"""Hosted (OpenAI-compatible) provider: message conversion, error handling, and that the API key never leaks.
Uses a fake HTTP transport, so these tests never contact Gemini, Groq or OpenRouter."""
import json

import httpx
import pytest

from credit_kg.agent.providers import OpenAICompatibleProvider, TransientProviderError, to_openai_messages
from credit_kg.agent.runner import Agent
from credit_kg.agent.settings import AgentSettings
from credit_kg.agent.tools import ToolContext
from credit_kg.agent.tracing import read_run

FAKE_KEY = "AIzaSyFAKE0123456789abcdefghijklmnop"


def _reply(content="", tool=None):
    msg = {"role": "assistant", "content": content}
    if tool:
        msg["tool_calls"] = [{"id": "x1", "type": "function", "function": {"name": tool[0], "arguments": json.dumps(tool[1])}}]
    return {"choices": [{"message": msg}], "usage": {"prompt_tokens": 50, "completion_tokens": 5}}


def _provider(monkeypatch, handler, name="gemini"):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    return OpenAICompatibleProvider(name, "test-model", 0.0, 42, 10.0, transport=httpx.MockTransport(handler))


def test_message_conversion_pairs_tool_results_with_ids():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "a", "arguments": {"customer_id": 2}}},
                                                                {"function": {"name": "b", "arguments": {}}}]},
            {"role": "tool", "content": "{}", "tool_name": "a"}, {"role": "tool", "content": "{}", "tool_name": "b"}]
    out = to_openai_messages(msgs)
    ids = [c["id"] for c in out[2]["tool_calls"]]
    assert [m["tool_call_id"] for m in out[3:]] == ids
    assert json.loads(out[2]["tool_calls"][0]["function"]["arguments"]) == {"customer_id": 2}
    assert out[2]["content"] is None


def test_tool_call_parsing_and_auth_header(monkeypatch):
    seen = {}

    def handler(req):
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=_reply(tool=("get_customer_profile", {"customer_id": 2})))
    r = _provider(monkeypatch, handler).chat([{"role": "user", "content": "hi"}], [{"type": "function", "function": {"name": "x"}}])
    assert r.tool_calls[0].name == "get_customer_profile" and r.tool_calls[0].arguments == {"customer_id": 2}
    assert seen["auth"] == f"Bearer {FAKE_KEY}" and seen["body"]["model"] == "test-model" and seen["body"]["tools"]
    assert r.prompt_tokens == 50


def test_gemini_thought_signature_is_echoed_back(monkeypatch, tmp_path):
    """Regression (gemini-3.8-flash): 'Function call is missing a thought_signature' (HTTP 400) on step 2, because
    the provider's per-call extra data was dropped from the history."""
    sig = {"google": {"thought_signature": "opaque-sig-123"}}
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        if len(bodies) == 1:
            msg = {"role": "assistant", "content": None, "tool_calls": [{"id": "fc_1", "type": "function", "extra_content": sig,
                   "function": {"name": "score_credit_increase", "arguments": json.dumps({"customer_id": 2, "times_typical_payment": 3})}}]}
            return httpx.Response(200, json={"choices": [{"message": msg}], "usage": {}})
        return httpx.Response(200, json=_reply(content="Customer 2 scores 69.0, which is Review."))
    p = _provider(monkeypatch, handler)
    settings = AgentSettings.from_env(runs_dir=str(tmp_path), provider="gemini")
    ctx = ToolContext(settings)
    try:
        res = Agent(settings, provider=p, ctx=ctx).ask("Is 3x reasonable?", customer_id=2)
    finally:
        ctx.close()
    echoed = [m for m in bodies[1]["messages"] if m["role"] == "assistant" and m.get("tool_calls")][0]["tool_calls"][0]
    assert echoed["id"] == "fc_1" and echoed["extra_content"] == sig
    assert [m for m in bodies[1]["messages"] if m["role"] == "tool"][0]["tool_call_id"] == "fc_1"
    assert res.stopped_reason == "answered"


@pytest.mark.parametrize("status,transient", [(429, True), (503, True), (400, False), (401, False)])
def test_rate_limits_and_server_errors_are_transient_others_are_not(monkeypatch, status, transient):
    p = _provider(monkeypatch, lambda req: httpx.Response(status, text="nope"))
    with pytest.raises(TransientProviderError if transient else RuntimeError):
        p.chat([{"role": "user", "content": "hi"}], [])


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY is not set"):
        OpenAICompatibleProvider("groq", "m", 0.0, 42, 10.0)


def test_api_key_never_reaches_results_or_traces(monkeypatch, tmp_path):
    script = iter([_reply(tool=("score_credit_increase", {"customer_id": 2, "times_typical_payment": 3})),
                   _reply(content=f"Customer 2 scores 69.0, which is Review. (debug: key {FAKE_KEY})")])
    p = _provider(monkeypatch, lambda req: httpx.Response(200, json=next(script)))
    settings = AgentSettings.from_env(runs_dir=str(tmp_path), provider="gemini", regenerate_on_grounding_failure=False)
    ctx = ToolContext(settings)
    try:
        res = Agent(settings, provider=p, ctx=ctx).ask("Is 3x reasonable?", customer_id=2)
    finally:
        ctx.close()
    trace_text = (tmp_path / f"{res.run_id}.jsonl").read_text(encoding="utf-8")
    assert FAKE_KEY not in trace_text, "API key leaked into the trace"
    assert FAKE_KEY not in json.dumps(read_run(res.trace_path)[0]["settings"])
    assert "-> Review" in res.answer
