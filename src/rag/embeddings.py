"""Code-oriented text embeddings for Layer-4 semantic RAG.

Model choice
------------
We use ``microsoft/codebert-base`` (768-d) via HuggingFace ``transformers``
with mean-pooling + L2 normalization.

Why not the alternatives:
- ``sentence-transformers`` crashes on import in this Windows/CUDA env
  (access violation), so we cannot rely on the ST high-level API here.
- ``flax-sentence-embeddings/st-codesearch-distilroberta-base`` is ST-native
  and would be ideal for code search, but its default weight files are
  pickle-based and blocked by the current transformers/torch CVE gate
  unless safetensors are available; ``microsoft/codebert-base`` loads
  cleanly with ``use_safetensors=True``.

If ``sentence_transformers`` imports successfully in another environment,
``embed_texts`` will prefer that path with the same model id when possible.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

MODEL_NAME = "microsoft/codebert-base"
EMBEDDING_DIM = 768
BATCH_SIZE = 32


def _select_device() -> str:
    """Return ``cuda`` when available, otherwise ``cpu``."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _load_backend() -> tuple[str, Any, Any, str]:
    """Load the embedding backend once.

    Returns:
        ``(backend_kind, model, tokenizer_or_none, device)`` where
        ``backend_kind`` is ``"transformers"`` (sentence-transformers is
        intentionally not imported here — it hard-crashes on this Windows/CUDA
        install rather than raising a catchable exception).
    """
    device = _select_device()
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, use_safetensors=True)
    model.to(device)
    model.eval()
    logger.info("Loaded %s via transformers on %s (dim=%d)", MODEL_NAME, device, EMBEDDING_DIM)
    return "transformers", model, tokenizer, device


def get_embedding_dim() -> int:
    """Return the output vector dimensionality for the active model."""
    _kind, model, _tok, _device = _load_backend()
    return int(getattr(model.config, "hidden_size", EMBEDDING_DIM))


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings with attention masking, then L2-normalize."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    pooled = summed / counts
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts in batches of 32 on GPU when available, else CPU.

    Args:
        texts: Input strings (docstrings / comment blocks / queries).

    Returns:
        List of embedding vectors (each a list of floats).
    """
    if not texts:
        return []

    _kind, model, tokenizer, device = _load_backend()
    assert tokenizer is not None

    all_vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        all_vectors.extend(pooled.detach().cpu().tolist())
    return all_vectors


def _extract_preceding_comments(source_lines: list[str], start_line: int) -> list[str]:
    """Collect comment lines immediately above ``start_line`` (1-indexed)."""
    idx = start_line - 2  # line above the definition
    block: list[str] = []
    while idx >= 0:
        raw = source_lines[idx]
        stripped = raw.strip()
        if stripped == "":
            # Allow a single blank separator inside a comment run, but stop if
            # we already have comments and hit a second blank below code.
            if block:
                block.append("")
            idx -= 1
            continue
        if stripped.startswith("#"):
            block.append(stripped.lstrip("#").strip())
            idx -= 1
            continue
        break
    # Walked upward; reverse to top-down order and drop leading blanks.
    block.reverse()
    while block and block[0] == "":
        block.pop(0)
    while block and block[-1] == "":
        block.pop()
    return block


def _extract_docstring(source_lines: list[str], start_line: int, end_line: int) -> Optional[str]:
    """Best-effort docstring extraction from the body of a def/class block."""
    body_end = min(end_line, len(source_lines))
    # First body line is the line after the def/class header (1-indexed start).
    i = start_line  # 0-indexed index into source_lines
    limit = min(body_end, start_line + 40)

    while i < limit:
        line = source_lines[i]
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            i += 1
            continue

        # Match optional string prefix + opening quotes.
        match = re.match(r'^\s*[rRuUbBfF]*("""|\'\'\'|"|\')', line)
        if not match:
            return None

        delim = match.group(1)
        after = line[match.end() :]

        # One-line docstring.
        end_pos = after.find(delim)
        if end_pos >= 0:
            return after[:end_pos].strip() or None

        # Multi-line triple-quoted docstring.
        if delim not in {'"""', "'''"}:
            return None
        parts = [after]
        i += 1
        while i < limit:
            if delim in source_lines[i]:
                parts.append(source_lines[i].split(delim, 1)[0])
                text = "\n".join(parts).strip()
                return text or None
            parts.append(source_lines[i])
            i += 1
        return None
    return None


def extract_embeddable_content(ast_node: dict, source_lines: list[str]) -> str | None:
    """Extract docstring + preceding comments for a function/class AST node.

    Bare code with neither docs nor comments returns ``None`` so we do not
    index implementation-only bodies into the semantic store.

    Args:
        ast_node: AST dict with at least ``node_type``, ``start_line``,
            ``end_line`` (and ideally ``name``).
        source_lines: Full file content split into lines (no trailing
            requirement on newlines).

    Returns:
        A single string suitable for embedding, or ``None`` to skip.
    """
    kind = ast_node.get("node_type")
    if kind not in {"function", "class"}:
        return None

    start_line = int(ast_node.get("start_line") or 0)
    end_line = int(ast_node.get("end_line") or 0)
    if start_line <= 0 or not source_lines:
        return None

    comments = _extract_preceding_comments(source_lines, start_line)
    docstring = _extract_docstring(source_lines, start_line, end_line or start_line)

    parts: list[str] = []
    if comments:
        parts.append("\n".join(comments))
    if docstring:
        parts.append(docstring)

    text = "\n\n".join(parts).strip()
    if len(text) < 8:
        # Too short to be meaningful semantic context.
        return None
    return text
