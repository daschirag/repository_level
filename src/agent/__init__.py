"""Layer 5: LangGraph refactor agent over the debt graph + RAG context."""

from __future__ import annotations

from typing import Any

__all__ = [
    "generate_refactor_proposals",
    "StarCoder2Client",
    "MockLLMClient",
    "get_llm_client",
    "TOOLS",
]


def __getattr__(name: str) -> Any:
    """Lazy re-exports."""
    if name in {
        "generate_refactor_proposals",
        "StarCoder2Client",
        "MockLLMClient",
        "get_llm_client",
        "TOOLS",
    }:
        from src.agent import refactor_agent

        return getattr(refactor_agent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
