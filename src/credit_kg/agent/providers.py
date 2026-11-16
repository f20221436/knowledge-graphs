"""LLM providers behind one small interface. Ollama runs locally (no key, no cost, no data leaves the machine);
MockProvider replays scripted responses so unit tests never call a real model."""
import json
import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    name: str
    arguments: dict
    extra: dict = field(default_factory=dict)   # provider data to echo back unchanged (call id, Gemini thought_signature)


@dataclass
class LLMResponse:
    text: str
    tool_calls: list = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    raw_error: str | None = None


class TransientProviderError(Exception):
    """Errors worth retrying: connection problems, timeouts, server overload."""


class Provider(Protocol):
    name: str
    model: str
    cost_per_1k_prompt: float
    cost_per_1k_completion: float

    def chat(self, messages: list, tools: list) -> LLMResponse: ...


class OllamaProvider:
    name = "ollama"
    cost_per_1k_prompt = 0.0
    cost_per_1k_completion = 0.0

    def __init__(self, model: str, host: str, temperature: float, seed: int, num_ctx: int, timeout_s: float):
        import ollama

        self.model = model
        self._client = ollama.Client(host=host, timeout=timeout_s)
        self._options = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx}

    def chat(self, messages: list, tools: list) -> LLMResponse:
        import httpx
        import ollama

        t0 = time.perf_counter()
        try:
            r = self._client.chat(model=self.model, messages=messages, tools=tools or None, options=self._options)
        except (httpx.ConnectError, httpx.TimeoutException, ConnectionError) as e:
            raise TransientProviderError(f"{type(e).__name__}: {e}") from e
        except ollama.ResponseError as e:
            if getattr(e, "status_code", 0) >= 500 or getattr(e, "status_code", 0) == 429:
                raise TransientProviderError(f"HTTP {e.status_code}: {e.error}") from e
            raise
        msg = r.message
        calls = []
        for tc in msg.tool_calls or []:
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_unparseable": args}
            calls.append(ToolCall(tc.function.name, dict(args or {})))
        return LLMResponse(text=msg.content or "", tool_calls=calls, prompt_tokens=r.prompt_eval_count or 0,
                           completion_tokens=r.eval_count or 0, latency_s=time.perf_counter() - t0)


class MockProvider:
    """Replays a script. Each item is either a string (final answer) or a list of (tool_name, args) tool calls.
    A callable item receives the message history, so tests can react to tool results."""
    name = "mock"
    model = "mock"
    cost_per_1k_prompt = 0.0
    cost_per_1k_completion = 0.0

    def __init__(self, script: list, tokens_per_call: int = 100):
        self.script = list(script)
        self.calls = 0
        self.tokens_per_call = tokens_per_call

    def chat(self, messages: list, tools: list) -> LLMResponse:
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if callable(item):
            item = item(messages)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return LLMResponse(text=item, prompt_tokens=self.tokens_per_call, completion_tokens=10)
        return LLMResponse(text="", tool_calls=[ToolCall(n, a) for n, a in item],
                           prompt_tokens=self.tokens_per_call, completion_tokens=10)


# Hosted providers that accept the OpenAI chat-completions format. Free tiers: data leaves the machine and may be
# used by the provider; acceptable here only because the dataset is public.
HOSTED = {
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "key_env": "GEMINI_API_KEY"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
}


def to_openai_messages(messages: list) -> list:
    """Our history uses Ollama's shape (tool calls without IDs, arguments as dicts). The OpenAI format needs an ID on
    every tool call, the matching tool_call_id on each tool result, and arguments as a JSON string."""
    out, pending = [], []
    for i, m in enumerate(messages):
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = []
            for j, tc in enumerate(m["tool_calls"]):
                extra = tc.get("extra") or {}
                cid = extra.get("id") or f"call_{i}_{j}"
                pending.append(cid)
                call = {"id": cid, "type": "function", "function": {
                    "name": tc["function"]["name"], "arguments": json.dumps(tc["function"]["arguments"], default=str)}}
                if "extra_content" in extra:          # Gemini requires its thought_signature to come back unchanged
                    call["extra_content"] = extra["extra_content"]
                calls.append(call)
            out.append({"role": "assistant", "content": m.get("content") or None, "tool_calls": calls})
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": pending.pop(0) if pending else "call_unknown", "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


class OpenAICompatibleProvider:
    """Gemini, Groq or OpenRouter through their OpenAI-compatible endpoints. The API key is read from the environment
    at construction and kept only in the HTTP client's headers: never in settings, results or traces."""
    cost_per_1k_prompt = 0.0          # free tier; the trace records the provider so paid use would be visible
    cost_per_1k_completion = 0.0

    def __init__(self, name: str, model: str, temperature: float, seed: int, timeout_s: float, min_interval_s: float = 0.0,
                 transport=None, reasoning_effort: str = ""):
        import os

        import httpx

        if name not in HOSTED:
            raise ValueError(f"unknown hosted provider '{name}' (available: {', '.join(HOSTED)})")
        key = os.environ.get(HOSTED[name]["key_env"], "").strip()
        if not key:
            raise RuntimeError(f"{HOSTED[name]['key_env']} is not set in .env")
        self.name, self.model = name, model
        self._temperature, self._seed = temperature, seed
        self._min_interval, self._last_call = min_interval_s, 0.0
        self._reasoning_effort = reasoning_effort
        self._client = httpx.Client(base_url=HOSTED[name]["base_url"], timeout=timeout_s,
                                    headers={"Authorization": f"Bearer {key}"}, transport=transport)

    def list_models(self) -> list:
        r = self._client.get("/models")
        r.raise_for_status()
        return sorted(m.get("id", "") for m in r.json().get("data", []))

    def chat(self, messages: list, tools: list) -> LLMResponse:
        import httpx

        wait = self._min_interval - (time.monotonic() - self._last_call)
        if wait > 0:                                   # pace requests to stay inside free-tier rate limits
            time.sleep(wait)
        body = {"model": self.model, "messages": to_openai_messages(messages), "temperature": self._temperature}
        if self.name != "gemini":
            body["seed"] = self._seed
        if self._reasoning_effort:                     # thinking models: less thinking = faster, cheaper replies
            body["reasoning_effort"] = self._reasoning_effort
        if tools:
            body["tools"] = tools
        t0 = time.perf_counter()
        self._last_call = time.monotonic()
        try:
            r = self._client.post("/chat/completions", json=body)
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise TransientProviderError(f"{type(e).__name__}: {e}") from e
        if r.status_code == 429 or r.status_code >= 500:
            raise TransientProviderError(f"HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        msg = data["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            args = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except json.JSONDecodeError:
                args = {"_unparseable": args}
            calls.append(ToolCall(tc["function"]["name"], dict(args or {}),
                                  {k: tc[k] for k in ("id", "extra_content") if k in tc}))
        usage = data.get("usage") or {}
        return LLMResponse(text=msg.get("content") or "", tool_calls=calls, prompt_tokens=usage.get("prompt_tokens", 0),
                           completion_tokens=usage.get("completion_tokens", 0), latency_s=time.perf_counter() - t0)


def make_provider(settings) -> Provider:
    if settings.provider == "ollama":
        return OllamaProvider(settings.model, settings.ollama_host, settings.temperature, settings.seed,
                              settings.num_ctx, settings.llm_timeout_s)
    if settings.provider in HOSTED:
        return OpenAICompatibleProvider(settings.provider, settings.model, settings.temperature, settings.seed,
                                        settings.llm_timeout_s, settings.min_seconds_between_calls,
                                        reasoning_effort=settings.reasoning_effort)
    raise ValueError(f"unknown provider '{settings.provider}' (available: ollama, {', '.join(HOSTED)})")
