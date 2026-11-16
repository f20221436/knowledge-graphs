"""CLI: python -m credit_kg.agent ask --customer 2 "Is a 3,000 increase reasonable?" """
import argparse
import sys

from .runner import Agent
from .settings import AgentSettings


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m credit_kg.agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="ask one question")
    a.add_argument("question")
    a.add_argument("--customer", type=int, default=None, help="customer ID the question is about")
    a.add_argument("--provider", default=None, help="ollama (default, local) | gemini | groq | openrouter (key in .env)")
    a.add_argument("--model", default=None, help="model name (default from AGENT_MODEL or qwen2.5:7b)")
    a.add_argument("--no-cypher", action="store_true", help="stage B: fixed tools only")
    a.add_argument("--verbose", action="store_true", help="also print tool calls, grounding result and trace path")
    args = p.parse_args(argv)

    settings = AgentSettings.from_env(provider=args.provider, model=args.model, enable_cypher=False if args.no_cypher else None)
    agent = Agent(settings)
    try:
        res = agent.ask(args.question, customer_id=args.customer)
    finally:
        agent.ctx.close()
    print(res.answer)
    if args.verbose:
        print("\n--- run details ---")
        for t in res.tool_calls:
            g = t.get("guardrail")
            print(f"tool {t['tool']}({t['args']}) {t['latency_s']}s" + (f" error={t['error']}" if t.get("error") else "")
                  + (f" guardrail={g['decision']}" if g else ""))
        print(f"grounding: {res.grounding} | regenerated: {res.regenerated} | stopped: {res.stopped_reason}")
        print(f"llm calls {res.llm_calls} | tokens {res.prompt_tokens}+{res.completion_tokens} | {res.latency_s}s | "
              f"cost ${res.cost_usd:.4f} | trace {res.trace_path}")
    return 0 if res.stopped_reason == "answered" else 1


if __name__ == "__main__":
    sys.exit(main())
