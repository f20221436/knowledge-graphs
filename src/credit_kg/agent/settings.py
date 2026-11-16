"""Agent configuration. Every limit lives here and can be overridden from .env (prefix AGENT_)."""
import hashlib
import os
import subprocess

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .. import config as C

load_dotenv(C.ROOT / ".env")


def _env(name: str, default):
    raw = os.environ.get(f"AGENT_{name.upper()}")
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return type(default)(raw)


class AgentSettings(BaseModel):
    provider: str = Field("ollama", description="ollama (local) | gemini | groq | openrouter (hosted, key in .env)")
    min_seconds_between_calls: float = 0.0   # pacing for hosted free tiers (rate limits)
    reasoning_effort: str = ""              # hosted thinking models, e.g. "low"; empty = provider default
    model: str = "qwen2.5:7b"
    ollama_host: str = "http://localhost:11434"
    temperature: float = 0.0
    seed: int = C.SEED
    num_ctx: int = 8192

    # loop limits
    max_steps: int = 8                 # LLM calls per run (including the final answer)
    max_tool_calls: int = 8
    max_tokens_per_run: int = 30000    # prompt + completion tokens summed over the run
    llm_timeout_s: float = 300.0       # one LLM call (CPU inference is slow)
    run_timeout_s: float = 900.0       # whole run, wall clock
    max_retries: int = 2               # transient provider errors only
    regenerate_on_grounding_failure: bool = True

    # tools and guardrails
    enable_cypher: bool = True         # stage C: guarded read-only Cypher tool
    guardrails: bool = True            # False only for the ON/OFF comparison, against a disposable DB
    cypher_row_cap: int = 50
    cypher_timeout_s: float = 5.0
    disposable_neo4j_uri: str = ""     # required when guardrails are off

    runs_dir: str = str(C.ROOT / "runs")

    @classmethod
    def from_env(cls, **overrides) -> "AgentSettings":
        base = cls()
        values = {k: _env(k, v) for k, v in base.model_dump().items()}
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def public(self) -> dict:
        """Settings safe to log (there are no secrets in this model, but keep the boundary explicit)."""
        return self.model_dump()


def code_version() -> str:
    """Git commit if available; otherwise a fingerprint of the source and config files (git is not used here)."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=C.ROOT, capture_output=True, text=True, timeout=5)
        if sha.returncode == 0 and sha.stdout.strip():
            return sha.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    h = hashlib.sha256()
    for p in sorted([*(C.ROOT / "src").rglob("*.py"), *(C.CONFIG_DIR).glob("*.yaml"), *(C.QUERIES).rglob("*.cypher")]):
        h.update(p.relative_to(C.ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return "nogit-" + h.hexdigest()[:12]
