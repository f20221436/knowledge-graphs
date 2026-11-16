"""Runs the agent on an eval task set and stores one JSON line per task in evals/runs/<run_id>.jsonl.

    .venv\\Scripts\\python.exe evals\\run_eval.py --provider ollama --model qwen2.5:3b
    .venv\\Scripts\\python.exe evals\\run_eval.py --model qwen2.5:3b --limit 5          # quick subset
    .venv\\Scripts\\python.exe evals\\run_eval.py --model qwen2.5:3b --no-cypher        # stage B (fixed tools only)

Never point --tasks at evals/heldout/: that set is the user's own and is run only on request.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from credit_kg.agent import Agent, AgentSettings
from credit_kg.agent.evals import grade
from credit_kg.agent.settings import code_version
from credit_kg.agent.tracing import read_run

HERE = Path(__file__).parent


def graph_counts():
    try:
        from credit_kg.graph import get_driver, run
        d = get_driver()
        n = int(run(d, "MATCH (n) RETURN count(n) AS n")["n"][0])
        r = int(run(d, "MATCH ()-[r]->() RETURN count(r) AS n")["n"][0])
        lim = float(run(d, "MATCH (a:CreditAccount) RETURN sum(a.credit_limit) AS s")["s"][0])
        d.close()
        return (n, r, lim)
    except Exception as e:  # noqa: BLE001
        return ("unavailable", str(e)[:80])


def run_task(agent: Agent, task: dict) -> dict:
    watch = task["category"] == "adversarial" and task["check"]["kind"] in {"write", "injection"}
    before = graph_counts() if watch else None
    res = agent.ask(task["question"], customer_id=task.get("customer_id"), task={"id": task["id"], "version": task["version"]})
    after = graph_counts() if watch else None
    events = read_run(res.trace_path)
    tool_results = [e.get("result") for e in events if e["event"] == "tool_call"]
    run = {"answer": res.answer, "explanation": res.explanation, "decision": res.decision, "stopped_reason": res.stopped_reason,
           "tool_calls": [{k: t.get(k) for k in ("tool", "args", "guardrail", "error", "latency_s", "repeat")} for t in res.tool_calls],
           "tool_results": tool_results, "graph_changed": bool(watch and before != after)}
    return {"task_id": task["id"], "category": task["category"], "question": task["question"], "customer_id": task.get("customer_id"),
            "check": task["check"], "answer": res.answer, "explanation": res.explanation,
            "decision": {k: res.decision[k] for k in ("band", "score", "requested_increase")} if res.decision else None,
            "tool_calls": run["tool_calls"], "graph_changed": run["graph_changed"], "grounding": res.grounding,
            "stopped_reason": res.stopped_reason, "llm_calls": res.llm_calls, "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens, "latency_s": res.latency_s, "cost_usd": res.cost_usd,
            "retries": res.retries, "regenerated": res.regenerated, "agent_run_id": res.run_id, "grade": grade(task, run)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default=str(HERE / "tasks_v1.jsonl"))
    p.add_argument("--provider", default="ollama")
    p.add_argument("--model", default="qwen2.5:3b")
    p.add_argument("--no-cypher", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="run only the first N tasks")
    p.add_argument("--ids", default=None, help="comma-separated task IDs to run")
    p.add_argument("--label", default="", help="free-text label for this run (e.g. 'subset: time limit')")
    p.add_argument("--allow-heldout", action="store_true", help="only when the owner of evals/heldout/ asks for it")
    args = p.parse_args(argv)
    if "heldout" in Path(args.tasks).parts and not args.allow_heldout:
        sys.exit("refusing to run evals/heldout/: pass --allow-heldout only when its owner asks")

    tasks = [json.loads(line) for line in Path(args.tasks).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.ids:
        keep = set(args.ids.split(","))
        tasks = [t for t in tasks if t["id"] in keep]
    if args.limit:
        tasks = tasks[: args.limit]
    settings = AgentSettings.from_env(provider=args.provider, model=args.model, enable_cypher=not args.no_cypher,
                                      run_timeout_s=240.0, llm_timeout_s=180.0)
    agent = Agent(settings)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + args.model.replace(":", "-").replace("/", "-")
    out = HERE / "runs" / f"{run_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = {"record": "run", "run_id": run_id, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "code_version": code_version(),
              "provider": args.provider, "model": args.model, "task_set": Path(args.tasks).name,
              "task_set_version": tasks[0]["version"] if tasks else None, "tasks_planned": len(tasks),
              "guardrails": settings.guardrails, "cypher_enabled": settings.enable_cypher, "label": args.label,
              "settings": settings.public()}
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
    try:
        for i, task in enumerate(tasks, 1):
            rec = run_task(agent, task)
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"record": "task", "run_id": run_id, **rec}, default=str) + "\n")
            print(f"[{i}/{len(tasks)}] {task['id']} correct={rec['grade']['correct']} {rec['latency_s']:.0f}s "
                  f"{rec['grade']['fail_reason']}", flush=True)
    finally:
        agent.ctx.close()
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
