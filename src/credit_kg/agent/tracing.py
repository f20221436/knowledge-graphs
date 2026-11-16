"""Structured run traces: one JSON object per line in runs/<run_id>.jsonl. No secrets are ever logged."""
import json
import re
import time
from pathlib import Path

# Key shapes: OpenAI/OpenRouter (sk-...), Google (AIza...), Groq (gsk_...), bearer headers, key=/password= pairs.
_SECRET = re.compile(r"(sk-[A-Za-z0-9_\-]{10,}|AIza[0-9A-Za-z_\-]{20,}|gsk_[A-Za-z0-9]{20,}|bearer\s+\S+|"
                     r"api[_-]?key\s*[:=]\s*\S+|password\s*[:=]\s*\S+)", re.I)


def _scrub(obj):
    if isinstance(obj, str):
        return _SECRET.sub("[REDACTED]", obj)
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items() if "password" not in str(k).lower() and "api_key" not in str(k).lower()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    return obj


class Tracer:
    def __init__(self, runs_dir, run_id: str):
        self.path = Path(runs_dir) / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.t0 = time.perf_counter()

    def log(self, event: str, **data) -> None:
        rec = {"run_id": self.run_id, "event": event, "t": round(time.perf_counter() - self.t0, 3), **_scrub(data)}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")


def read_run(path) -> list:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
