"""Layer 6: FastAPI backend for the tech-debt heatmap and refactor API.

Serves Neo4j DIV graph data, top-debt rankings, RAG/graph context drill-down,
and async StarCoder2 refactor proposal jobs.
"""

from __future__ import annotations

import logging
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from neo4j import Driver, GraphDatabase
from pydantic import BaseModel, Field

from src.agent.refactor_agent import generate_refactor_proposals
from src.agent.tools import (
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    fetch_function_context,
)
from src.rag.qdrant_client import CodeRAGClient
from src.scoring.div_propagation import get_top_debt_nodes

logger = logging.getLogger(__name__)

# In-memory async job store (single-process demo; not durable across restarts).
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()

# Shared clients, initialized in lifespan.
_neo4j_driver: Optional[Driver] = None
_rag_client: Optional[CodeRAGClient] = None


class RefactorRequest(BaseModel):
    """Body for ``POST /api/refactor-proposals``."""

    top_k: int = Field(default=3, ge=1, le=20)
    # Optional single-function target; when both are set, ``top_k`` is ignored.
    file_path: Optional[str] = None
    name: Optional[str] = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open Neo4j + Qdrant clients once at startup; close on shutdown."""
    global _neo4j_driver, _rag_client

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("Starting techdebt API (Neo4j=%s)", NEO4J_URI)

    _neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        _neo4j_driver.verify_connectivity()
        logger.info("Neo4j connected")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Neo4j not reachable at startup: %s", exc)

    try:
        _rag_client = CodeRAGClient()
        _rag_client._client.get_collections()
        logger.info("Qdrant connected")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Qdrant not reachable at startup: %s", exc)
        _rag_client = None

    yield

    if _neo4j_driver is not None:
        _neo4j_driver.close()
        _neo4j_driver = None
    _rag_client = None
    logger.info("techdebt API shut down")


app = FastAPI(
    title="Tech Debt Agent API",
    description="DIV debt graph, semantic context, and refactor proposals",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_driver() -> Driver:
    if _neo4j_driver is None:
        raise HTTPException(status_code=503, detail="Neo4j driver not initialized")
    return _neo4j_driver


def _node_id(file_path: str, name: str) -> str:
    return f"{file_path}::{name}"


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Liveness/readiness snapshot for local demos."""
    neo4j_ok = False
    if _neo4j_driver is not None:
        try:
            _neo4j_driver.verify_connectivity()
            neo4j_ok = True
        except Exception:  # noqa: BLE001
            neo4j_ok = False
    qdrant_ok = _rag_client is not None
    return {
        "status": "ok",
        "neo4j": neo4j_ok,
        "qdrant": qdrant_ok,
        "jobs_tracked": len(_jobs),
    }


@app.get("/api/debt-graph")
def debt_graph() -> dict[str, Any]:
    """Return Function nodes + CALLS/IMPORTS edges for D3 visualization."""
    driver = _require_driver()
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()

    with driver.session() as session:
        func_rows = session.run(
            """
            MATCH (f:Function)
            RETURN f.file_path AS file_path,
                   f.name AS name,
                   coalesce(f.div_score, 0.0) AS div_score,
                   coalesce(f.cyclomatic_complexity, 0) AS cyclomatic_complexity,
                   coalesce(f.change_frequency, 0) AS change_frequency,
                   f.start_line AS start_line,
                   f.end_line AS end_line
            """
        )
        for record in func_rows:
            file_path = record["file_path"]
            name = record["name"]
            if not file_path or not name:
                continue
            nid = _node_id(file_path, name)
            seen_nodes.add(nid)
            nodes.append(
                {
                    "id": nid,
                    "file_path": file_path,
                    "name": name,
                    "div_score": float(record["div_score"] or 0.0),
                    "cyclomatic_complexity": int(record["cyclomatic_complexity"] or 0),
                    "change_frequency": int(record["change_frequency"] or 0),
                    "start_line": record["start_line"],
                    "end_line": record["end_line"],
                    "node_kind": "Function",
                }
            )

        call_rows = session.run(
            """
            MATCH (a:Function)-[:CALLS]->(b:Function)
            RETURN a.file_path AS source_file, a.name AS source_name,
                   b.file_path AS target_file, b.name AS target_name
            """
        )
        for record in call_rows:
            source = _node_id(record["source_file"], record["source_name"])
            target = _node_id(record["target_file"], record["target_name"])
            if source in seen_nodes and target in seen_nodes:
                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "type": "CALLS",
                    }
                )

        import_rows = session.run(
            """
            MATCH (f:File)-[:IMPORTS]->(m:Module)
            RETURN f.path AS file_path, m.name AS module_name
            """
        )
        for record in import_rows:
            file_path = record["file_path"]
            module_name = record["module_name"]
            if not file_path or not module_name:
                continue
            file_id = f"file::{file_path}"
            mod_id = f"module::{module_name}"
            if file_id not in seen_nodes:
                seen_nodes.add(file_id)
                nodes.append(
                    {
                        "id": file_id,
                        "file_path": file_path,
                        "name": file_path.rsplit("/", 1)[-1],
                        "div_score": 0.0,
                        "cyclomatic_complexity": 0,
                        "change_frequency": 0,
                        "start_line": None,
                        "end_line": None,
                        "node_kind": "File",
                    }
                )
            if mod_id not in seen_nodes:
                seen_nodes.add(mod_id)
                nodes.append(
                    {
                        "id": mod_id,
                        "file_path": "",
                        "name": module_name[:80],
                        "div_score": 0.0,
                        "cyclomatic_complexity": 0,
                        "change_frequency": 0,
                        "start_line": None,
                        "end_line": None,
                        "node_kind": "Module",
                    }
                )
            edges.append(
                {
                    "source": file_id,
                    "target": mod_id,
                    "type": "IMPORTS",
                }
            )

    return {
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


@app.get("/api/top-debt")
def top_debt(k: int = Query(default=10, ge=1, le=100)) -> dict[str, Any]:
    """Return the top-``k`` functions by DIV score."""
    driver = _require_driver()
    nodes = [
        {"id": _node_id(n["file_path"], n["name"]), **n}
        for n in get_top_debt_nodes(driver, k=k)
    ]
    return {"k": k, "nodes": nodes}


@app.get("/api/function/{file_path}/{name}/context")
def function_context(file_path: str, name: str) -> dict[str, Any]:
    """RAG neighbors + 1-hop callers/callees for a function.

    ``file_path`` should be URL-encoded by the client (slashes as ``%2F``) so it
    arrives as a single path segment, e.g.
    ``/api/function/src%2Fflask%2Fapp.py/ensure_sync/context``.
    """
    # Starlette decodes %2F in path params on most setups; also accept ``::``.
    file_path = file_path.replace("::", "/").replace("\\", "/")
    try:
        return fetch_function_context(file_path, name)
    except Exception as exc:  # noqa: BLE001
        logger.exception("context lookup failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/function/context")
def function_context_query(
    file_path: str = Query(..., description="Repository-relative file path"),
    name: str = Query(..., description="Function name"),
) -> dict[str, Any]:
    """Same as the path-param route, but with query params (slash-safe)."""
    try:
        return fetch_function_context(file_path.replace("\\", "/"), name)
    except Exception as exc:  # noqa: BLE001
        logger.exception("context lookup failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _fetch_debt_node(driver: Driver, file_path: str, name: str) -> Optional[dict[str, Any]]:
    """Return DIV metrics for one ``:Function`` node, or ``None`` if absent."""
    with driver.session() as session:
        record = session.run(
            """
            MATCH (f:Function {file_path: $file_path, name: $name})
            RETURN f.file_path AS file_path,
                   f.name AS name,
                   coalesce(f.div_score, 0.0) AS div_score,
                   coalesce(f.cyclomatic_complexity, 0) AS cyclomatic_complexity,
                   coalesce(f.change_frequency, 0) AS change_frequency
            LIMIT 1
            """,
            {"file_path": file_path, "name": name},
        ).single()
    return dict(record) if record else None


def _run_refactor_job(
    job_id: str, top_k: int, nodes: Optional[list[dict[str, Any]]] = None
) -> None:
    """Background worker: generate proposals and update job status."""
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = _utc_now()
    try:
        proposals = generate_refactor_proposals(top_k=top_k, nodes=nodes)
        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["finished_at"] = _utc_now()
            _jobs[job_id]["result"] = {"top_k": top_k, "proposals": proposals}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Refactor job %s failed", job_id)
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["finished_at"] = _utc_now()
            _jobs[job_id]["error"] = str(exc)


@app.post("/api/refactor-proposals")
def start_refactor_proposals(
    body: RefactorRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Enqueue a StarCoder2 refactor job; poll with GET ``/api/refactor-proposals/{job_id}``.

    Targets the top-``k`` DIV functions, or a single function when ``file_path``
    and ``name`` are both given.
    """
    nodes: Optional[list[dict[str, Any]]] = None
    if body.file_path and body.name:
        node = _fetch_debt_node(_require_driver(), body.file_path.replace("\\", "/"), body.name)
        if node is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown function: {body.file_path}::{body.name}",
            )
        nodes = [node]
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "top_k": body.top_k,
            "created_at": _utc_now(),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }
    background_tasks.add_task(_run_refactor_job, job_id, body.top_k, nodes)
    return {"job_id": job_id, "status": "queued", "top_k": body.top_k}


@app.get("/api/refactor-proposals/{job_id}")
def get_refactor_job(job_id: str) -> dict[str, Any]:
    """Poll status/results for a background refactor proposal job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
        return dict(job)
