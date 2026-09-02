"""Layer 1: repository ingestion (AST parsing and git history)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "parse_repository",
    "crawl_repo_history",
    "get_change_frequency",
    "get_git_blame_summary",
    "get_last_modified_dates",
]


def __getattr__(name: str) -> Any:
    """Lazy re-exports to keep ``python -m src.ingestion.ast_parser`` clean."""
    if name == "parse_repository":
        from src.ingestion.ast_parser import parse_repository

        return parse_repository
    if name in {
        "crawl_repo_history",
        "get_change_frequency",
        "get_git_blame_summary",
        "get_last_modified_dates",
    }:
        from src.ingestion import git_crawler

        return getattr(git_crawler, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
