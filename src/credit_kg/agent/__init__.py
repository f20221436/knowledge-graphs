"""A guarded LLM agent that explains credit decisions using evidence from the knowledge graph.

The model and the scoring function make every decision; the LLM only chooses what to look up and writes the
explanation. See docs/THREAT_MODEL.md for what the guardrails stop and what they do not.
"""
from .runner import Agent, RunResult
from .settings import AgentSettings

__all__ = ["Agent", "AgentSettings", "RunResult"]
