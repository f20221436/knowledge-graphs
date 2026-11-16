"""Step 5: LLM-as-predictor experiment. Stores every call in evals/predictor_runs/<run_id>.jsonl and a summary in
evals/predictor_runs/<run_id>_summary.json (read by scripts/make_results.py).

    .venv\\Scripts\\python.exe evals\\llm_predictor.py --model qwen2.5:3b            # 400 + 20 + 300 calls, ~1 h on CPU
    .venv\\Scripts\\python.exe evals\\llm_predictor.py --model qwen2.5:3b --n 20 --subset 10   # quick check
"""
import argparse
import json
import sys
import time
from pathlib import Path

import ollama

from credit_kg import config as C
from credit_kg import llm_predictor as P
from credit_kg import modeling as M
from credit_kg.agent.settings import code_version

OUT = Path(__file__).parent / "predictor_runs"


def main(argv=None) -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--model", default="qwen2.5:3b")
    a.add_argument("--n", type=int, default=400, help="stratified test customers")
    a.add_argument("--subset", type=int, default=100, help="customers repeated for run-to-run consistency")
    a.add_argument("--repeats", type=int, default=3)
    a.add_argument("--consistency-temperature", type=float, default=0.8)
    a.add_argument("--determinism", type=int, default=20, help="customers re-asked at temperature 0")
    a.add_argument("--resummarise", default=None, help="run_id: recompute its summary from the stored calls (no model calls)")
    args = a.parse_args(argv)

    table = M.load_table()
    if args.resummarise:
        lines = [json.loads(x) for x in (OUT / f"{args.resummarise}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        head, records = lines[0], [x for x in lines if x.get("record") == "call"]
        sample = P.build_sample(table, head["customers"])
        summary = P.summarise(records, sample, P.baselines_for(sample, table))
        summary.update({"run_id": head["run_id"], "model": head["model"], "started": head["started"], "code_version": head["code_version"]})
        (OUT / f"{args.resummarise}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("resummarised", args.resummarise)
        return 0
    feats = M.feature_sets()["A"]
    sample = P.build_sample(table, args.n)
    client = ollama.Client(host="http://localhost:11434", timeout=300)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + args.model.replace(":", "-")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{run_id}.jsonl"
    plan = ([("main", 0, 0.0, C.SEED, r) for r in sample.itertuples()]
            + [("determinism", 0, 0.0, C.SEED, r) for r in sample.head(args.determinism).itertuples()]
            + [("consistency", k + 1, args.consistency_temperature, k + 1, r)
               for k in range(args.repeats) for r in sample.head(args.subset).itertuples()])
    head = {"record": "run", "run_id": run_id, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "code_version": code_version(),
            "provider": "ollama", "model": args.model, "customers": len(sample), "calls_planned": len(plan),
            "features": feats, "instructions": P.INSTRUCTIONS, "args": vars(args)}
    path.write_text(json.dumps(head) + "\n", encoding="utf-8")
    records = []
    for i, (phase, run, temp, seed, r) in enumerate(plan, 1):
        row = sample.loc[r.Index]
        t0 = time.perf_counter()
        try:
            resp = client.chat(model=args.model, messages=[{"role": "user", "content": P.make_prompt(row, feats)}],
                               format=P.JSON_SCHEMA, options={"temperature": temp, "seed": seed, "num_ctx": 2048})
            text, err = resp.message.content, None
            pt, ct = resp.prompt_eval_count or 0, resp.eval_count or 0
        except Exception as e:  # noqa: BLE001 - recorded, counted against coverage
            text, err, pt, ct = "", f"{type(e).__name__}: {str(e)[:200]}", 0, 0
        rec = {"record": "call", "customer_id": int(row.customer_id), "phase": phase, "run": run, "temperature": temp, "seed": seed,
               "raw": text[:200], "probability": P.parse_probability(text), "error": err,
               "latency_s": round(time.perf_counter() - t0, 3), "prompt_tokens": pt, "completion_tokens": ct}
        records.append(rec)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        if i % 25 == 0:
            print(f"[{i}/{len(plan)}] {phase}", flush=True)

    summary = P.summarise(records, sample, P.baselines_for(sample, table))
    summary.update({"run_id": run_id, "model": args.model, "started": head["started"], "code_version": head["code_version"]})
    (OUT / f"{run_id}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("coverage", "models", "paired_differences")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
