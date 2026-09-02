"""Layer 2: Neo4j call-graph construction."""

from __future__ import annotations

from typing import Any

__all__ = [
    "GraphClient",
    "initialize_schema",
    "compute_radon_scores",
]


def __getattr__(name: str) -> Any:
    """Lazy re-exports."""
    if name == "GraphClient":
        from src.graph.neo4j_client import GraphClient

        return GraphClient
    if name == "compute_radon_scores":
        from src.graph.neo4j_client import compute_radon_scores

        return compute_radon_scores
    if name == "initialize_schema":
        from src.graph.schema import initialize_schema

        return initialize_schema
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
