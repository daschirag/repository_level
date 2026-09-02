"""Layer 4: semantic RAG over code documentation."""

from __future__ import annotations

from typing import Any

__all__ = [
    "embed_texts",
    "extract_embeddable_content",
    "CodeRAGClient",
    "MODEL_NAME",
]


def __getattr__(name: str) -> Any:
    """Lazy re-exports."""
    if name in {"embed_texts", "extract_embeddable_content", "MODEL_NAME"}:
        from src.rag import embeddings

        return getattr(embeddings, name)
    if name == "CodeRAGClient":
        from src.rag.qdrant_client import CodeRAGClient

        return CodeRAGClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
