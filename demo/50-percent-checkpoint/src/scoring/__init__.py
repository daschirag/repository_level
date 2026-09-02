"""Layer 3: causal DIV (Debt Impact Vector) scoring."""

from __future__ import annotations

from typing import Any

__all__ = [
    "compute_div_scores",
    "write_div_scores",
    "get_top_debt_nodes",
]


def __getattr__(name: str) -> Any:
    """Lazy re-exports."""
    if name in {"compute_div_scores", "write_div_scores", "get_top_debt_nodes"}:
        from src.scoring import div_propagation

        return getattr(div_propagation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
