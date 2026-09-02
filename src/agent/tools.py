"""LangChain tools for the Layer-5 refactor agent.

Each tool wraps an existing Layer 2–4 capability (DIV ranking, Neo4j
neighborhood, Qdrant semantic neighbors, or on-disk source slices).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import tool
from neo4j import Driver, GraphDatabase

from src.rag.qdrant_client import CodeRAGClient
from src.scoring.div_propagation import get_top_debt_nodes

logger = logging.getLogger(__name__)

NEO4J_URI = os.environ.get("TECHDEBT_NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.environ.get("TECHDEBT_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("TECHDEBT_NEO4J_PASSWORD", "techdebt123")

# Repository root used to resolve relative ``file_path`` values from the graph.
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "flask-test-repo"
REPO_ROOT = Path(os.environ.get("TECHDEBT_REPO_PATH", str(DEFAULT_REPO_ROOT))).resolve()


def _neo4j_driver() -> Driver:
    """Open a short-lived Neo4j driver for tool calls."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _fetch_call_neighborhood(file_path: str, name: str) -> dict[str, list[dict[str, Any]]]:
    """Return 1-hop callers and callees for a ``:Function`` node."""
    query = """
    MATCH (f:Function {file_path: $file_path, name: $name})
    OPTIONAL MATCH (caller:Function)-[:CALLS]->(f)
    OPTIONAL MATCH (f)-[:CALLS]->(callee:Function)
    RETURN
      collect(DISTINCT {
        file_path: caller.file_path,
        name: caller.name,
        div_score: caller.div_score
      }) AS callers,
      collect(DISTINCT {
        file_path: callee.file_path,
        name: callee.name,
        div_score: callee.div_score
      }) AS callees
    """
    driver = _neo4j_driver()
    try:
        with driver.session() as session:
            record = session.run(
                query,
                {"file_path": file_path, "name": name},
            ).single()
        if record is None:
            return {"callers": [], "callees": []}

        def _clean(rows: list[Any]) -> list[dict[str, Any]]:
            cleaned: list[dict[str, Any]] = []
            for row in rows or []:
                if not row or row.get("name") is None:
                    continue
                cleaned.append(
                    {
                        "file_path": row.get("file_path"),
                        "name": row.get("name"),
                        "div_score": row.get("div_score"),
                    }
                )
            return cleaned

        return {
            "callers": _clean(list(record["callers"])),
            "callees": _clean(list(record["callees"])),
        }
    finally:
        driver.close()


def _fetch_function_span(file_path: str, name: str) -> dict[str, Any]:
    """Lookup ``start_line`` / ``end_line`` / metrics for a function in Neo4j."""
    driver = _neo4j_driver()
    try:
        with driver.session() as session:
            record = session.run(
                """
                MATCH (f:Function {file_path: $file_path, name: $name})
                RETURN f.file_path AS file_path,
                       f.name AS name,
                       f.start_line AS start_line,
                       f.end_line AS end_line,
                       f.div_score AS div_score,
                       coalesce(f.cyclomatic_complexity, 0) AS cyclomatic_complexity,
                       coalesce(f.change_frequency, 0) AS change_frequency
                LIMIT 1
                """,
                {"file_path": file_path, "name": name},
            ).single()
        return dict(record) if record is not None else {}
    finally:
        driver.close()


@tool
def get_top_debt_nodes_tool(k: int = 10) -> str:
    """Return the top-k functions ranked by DIV debt score from Neo4j.

    Args:
        k: Number of highest-DIV functions to return (default 10).
    """
    driver = _neo4j_driver()
    try:
        nodes = get_top_debt_nodes(driver, k=int(k))
    finally:
        driver.close()
    # JSON string keeps LangChain tool I/O simple and serializable.
    return json.dumps(nodes, default=str)


@tool
def get_function_context_tool(file_path: str, name: str) -> str:
    """Gather architectural + semantic context for a debt function.

    Combines:
    - 1-hop Neo4j callers and callees (CALLS edges)
    - Qdrant semantic neighbors (docstring/comment similarity), when available
    """
    file_path = file_path.replace("\\", "/")
    neighborhood = _fetch_call_neighborhood(file_path, name)
    span = _fetch_function_span(file_path, name)

    semantic: list[dict[str, Any]] = []
    try:
        rag = CodeRAGClient()
        semantic = rag.get_context_for_function(file_path, name, top_k=3)
    except Exception as exc:  # noqa: BLE001 — Qdrant may be offline
        logger.warning("Semantic context unavailable for %s::%s (%s)", file_path, name, exc)
        semantic = []

    payload = {
        "file_path": file_path,
        "name": name,
        "metrics": {
            "div_score": span.get("div_score"),
            "cyclomatic_complexity": span.get("cyclomatic_complexity"),
            "change_frequency": span.get("change_frequency"),
            "start_line": span.get("start_line"),
            "end_line": span.get("end_line"),
        },
        "callers": neighborhood["callers"],
        "callees": neighborhood["callees"],
        "semantic_neighbors": semantic,
    }
    return json.dumps(payload, default=str)


@tool
def get_function_source_tool(
    file_path: str,
    name: str,
    start_line: int,
    end_line: int,
) -> str:
    """Read the source lines for a function from disk.

    Args:
        file_path: Repository-relative path (as stored in Neo4j).
        name: Function name (included in the returned header for clarity).
        start_line: Inclusive 1-indexed start line.
        end_line: Inclusive 1-indexed end line.
    """
    file_path = file_path.replace("\\", "/")
    full_path = REPO_ROOT / file_path
    if not full_path.is_file():
        return json.dumps(
            {
                "error": f"File not found under REPO_ROOT={REPO_ROOT}: {file_path}",
                "file_path": file_path,
                "name": name,
            }
        )

    try:
        lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return json.dumps({"error": str(exc), "file_path": file_path, "name": name})

    start = max(int(start_line), 1)
    end = min(int(end_line), len(lines))
    if end < start:
        return json.dumps(
            {
                "error": f"Invalid span {start_line}-{end_line}",
                "file_path": file_path,
                "name": name,
            }
        )

    snippet = "\n".join(lines[start - 1 : end])
    return json.dumps(
        {
            "file_path": file_path,
            "name": name,
            "start_line": start,
            "end_line": end,
            "source": snippet,
        }
    )


# Convenience non-tool wrappers used by the LangGraph nodes directly
# (avoid round-tripping JSON when we already control the caller).
def fetch_top_debt_nodes(k: int = 10) -> list[dict[str, Any]]:
    """Typed helper mirroring ``get_top_debt_nodes_tool``."""
    driver = _neo4j_driver()
    try:
        return get_top_debt_nodes(driver, k=int(k))
    finally:
        driver.close()


def fetch_function_context(file_path: str, name: str) -> dict[str, Any]:
    """Typed helper mirroring ``get_function_context_tool``."""
    return json.loads(get_function_context_tool.invoke({"file_path": file_path, "name": name}))


def fetch_function_source(
    file_path: str,
    name: str,
    start_line: int,
    end_line: int,
) -> dict[str, Any]:
    """Typed helper mirroring ``get_function_source_tool``."""
    return json.loads(
        get_function_source_tool.invoke(
            {
                "file_path": file_path,
                "name": name,
                "start_line": start_line,
                "end_line": end_line,
            }
        )
    )


def fetch_function_span(file_path: str, name: str) -> dict[str, Any]:
    """Typed helper to resolve line spans from Neo4j."""
    return _fetch_function_span(file_path, name)
