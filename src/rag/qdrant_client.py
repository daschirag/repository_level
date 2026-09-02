"""Qdrant-backed Code RAG client for Layer-4 semantic retrieval.

Requires a local Qdrant instance on ``localhost:6333``. If it is not running,
start it with::

    docker run -d -p 6333:6333 -p 6334:6334 --name techdebt-qdrant \\
      -v D:\\techdebt-agent\\data\\qdrant:/qdrant/storage qdrant/qdrant
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.ingestion.ast_parser import SKIP_DIRS, parse_repository
from src.rag.embeddings import (
    MODEL_NAME,
    embed_texts,
    extract_embeddable_content,
    get_embedding_dim,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 6333
DEFAULT_COLLECTION = "techdebt_code_context"

NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "techdebt123"


def _point_id(file_path: str, name: str, node_type: str) -> str:
    """Stable UUID for a code entity (deterministic across re-indexes)."""
    key = f"{file_path}::{name}::{node_type}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _fetch_div_scores() -> dict[tuple[str, str], float]:
    """Load ``div_score`` from Neo4j ``:Function`` nodes when available."""
    scores: dict[tuple[str, str], float] = {}
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Neo4j driver init failed; div_score omitted (%s)", exc)
        return scores

    try:
        driver.verify_connectivity()
        with driver.session() as session:
            result = session.run(
                """
                MATCH (f:Function)
                WHERE f.div_score IS NOT NULL
                RETURN f.file_path AS file_path, f.name AS name, f.div_score AS div_score
                """
            )
            for record in result:
                file_path = record["file_path"]
                name = record["name"]
                if file_path and name:
                    scores[(str(file_path), str(name))] = float(record["div_score"])
        logger.info("Loaded %d div_score values from Neo4j", len(scores))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch div_scores from Neo4j (%s)", exc)
    finally:
        driver.close()
    return scores


def _load_source_lines(repo_path: Path, file_path: str) -> Optional[list[str]]:
    """Read a repo-relative source file as a list of lines."""
    full = repo_path / file_path
    if not full.is_file():
        return None
    try:
        text = full.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Failed to read %s: %s", full, exc)
        return None
    return text.splitlines()


class CodeRAGClient:
    """Thin Qdrant wrapper for docstring/comment semantic search."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        """Connect to a local Qdrant instance.

        Args:
            host: Qdrant host (default ``localhost``).
            port: Qdrant HTTP port (default ``6333``).
            collection_name: Collection used for code-context vectors.
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self._client = QdrantClient(host=host, port=port, timeout=60.0)

    def ensure_collection(self) -> None:
        """Create the collection if missing (cosine distance, model dim)."""
        dim = get_embedding_dim()
        if self._client.collection_exists(self.collection_name):
            info = self._client.get_collection(self.collection_name)
            logger.info(
                "Collection %s already exists (points=%s)",
                self.collection_name,
                getattr(info, "points_count", "?"),
            )
            return

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=dim,
                distance=qmodels.Distance.COSINE,
            ),
        )
        logger.info(
            "Created collection %s (dim=%d, distance=cosine, model=%s)",
            self.collection_name,
            dim,
            MODEL_NAME,
        )

    def index_repository(self, repo_path: str, ast_nodes: list[dict]) -> dict[str, int]:
        """Embed docstring/comment text for function/class nodes and upsert.

        Args:
            repo_path: Repository root path.
            ast_nodes: Output of ``parse_repository``.

        Returns:
            Counts: ``considered``, ``embedded``, ``skipped``, ``upserted``.
        """
        self.ensure_collection()
        root = Path(repo_path).resolve()
        div_scores = _fetch_div_scores()

        source_cache: dict[str, Optional[list[str]]] = {}
        pending_texts: list[str] = []
        pending_meta: list[dict[str, Any]] = []
        considered = 0
        skipped = 0

        for node in ast_nodes:
            kind = node.get("node_type")
            if kind not in {"function", "class"}:
                continue
            considered += 1
            file_path = str(node.get("file_path") or "").replace("\\", "/")
            name = node.get("name")
            if not file_path or not name:
                skipped += 1
                continue

            if file_path not in source_cache:
                source_cache[file_path] = _load_source_lines(root, file_path)
            source_lines = source_cache[file_path]
            if not source_lines:
                skipped += 1
                continue

            content = extract_embeddable_content(node, source_lines)
            if content is None:
                skipped += 1
                continue

            pending_texts.append(content)
            pending_meta.append(
                {
                    "file_path": file_path,
                    "name": str(name),
                    "node_type": str(kind),
                    "start_line": int(node.get("start_line") or 0),
                    "end_line": int(node.get("end_line") or 0),
                    "div_score": div_scores.get((file_path, str(name))),
                    "text": content,
                }
            )

        if not pending_texts:
            logger.warning("No embeddable function/class docs found in %s", root)
            return {
                "considered": considered,
                "embedded": 0,
                "skipped": skipped,
                "upserted": 0,
            }

        logger.info("Embedding %d doc/comment blocks with %s …", len(pending_texts), MODEL_NAME)
        vectors = embed_texts(pending_texts)

        points: list[qmodels.PointStruct] = []
        for meta, vector in zip(pending_meta, vectors, strict=True):
            points.append(
                qmodels.PointStruct(
                    id=_point_id(meta["file_path"], meta["name"], meta["node_type"]),
                    vector=vector,
                    payload={
                        "file_path": meta["file_path"],
                        "name": meta["name"],
                        "node_type": meta["node_type"],
                        "start_line": meta["start_line"],
                        "end_line": meta["end_line"],
                        "div_score": meta["div_score"],
                        "text": meta["text"],
                    },
                )
            )

        upserted = 0
        batch_size = 64
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            self._client.upsert(collection_name=self.collection_name, points=batch)
            upserted += len(batch)

        logger.info(
            "Indexed %d/%d entities into %s (skipped %d)",
            upserted,
            considered,
            self.collection_name,
            skipped,
        )
        return {
            "considered": considered,
            "embedded": len(pending_texts),
            "skipped": skipped,
            "upserted": upserted,
        }

    def query_context(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Embed ``query_text`` and return the top-k semantic matches.

        Args:
            query_text: Natural-language or code-flavored query.
            top_k: Maximum number of hits.

        Returns:
            List of dicts with payload fields plus ``score`` (cosine similarity).
        """
        if not query_text.strip():
            return []

        vector = embed_texts([query_text])[0]
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=int(top_k),
            with_payload=True,
        )
        hits: list[dict[str, Any]] = []
        for point in response.points:
            payload = dict(point.payload or {})
            hits.append(
                {
                    "file_path": payload.get("file_path"),
                    "name": payload.get("name"),
                    "node_type": payload.get("node_type"),
                    "start_line": payload.get("start_line"),
                    "end_line": payload.get("end_line"),
                    "div_score": payload.get("div_score"),
                    "text": payload.get("text"),
                    "score": float(point.score) if point.score is not None else 0.0,
                }
            )
        return hits

    def get_context_for_function(
        self,
        file_path: str,
        name: str,
        top_k: int = 3,
    ) -> list[dict]:
        """Return a function's own embedding (if any) plus k nearest neighbors.

        Useful for agent prompts: "what similar documented code exists?"

        Args:
            file_path: Repository-relative path.
            name: Function/class name.
            top_k: Neighbor count (excluding the function itself when possible).

        Returns:
            List of match dicts (own hit first when present, then neighbors).
        """
        file_path = file_path.replace("\\", "/")
        # Locate the entity's stored vector via filtered scroll/retrieve by payload.
        own_points, _next = self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="file_path",
                        match=qmodels.MatchValue(value=file_path),
                    ),
                    qmodels.FieldCondition(
                        key="name",
                        match=qmodels.MatchValue(value=name),
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=True,
        )

        results: list[dict[str, Any]] = []
        query_vector: Optional[list[float]] = None
        if own_points:
            own = own_points[0]
            payload = dict(own.payload or {})
            results.append(
                {
                    "file_path": payload.get("file_path"),
                    "name": payload.get("name"),
                    "node_type": payload.get("node_type"),
                    "start_line": payload.get("start_line"),
                    "end_line": payload.get("end_line"),
                    "div_score": payload.get("div_score"),
                    "text": payload.get("text"),
                    "score": 1.0,
                    "is_self": True,
                }
            )
            vec = own.vector
            if isinstance(vec, dict):
                # Named vector collections — take the first.
                vec = next(iter(vec.values()), None)
            if isinstance(vec, list):
                query_vector = vec

        if query_vector is None:
            # Fall back: semantic search using the bare function name as query.
            return self.query_context(f"{name} in {file_path}", top_k=top_k)

        response = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=int(top_k) + 1,
            with_payload=True,
        )
        for point in response.points:
            payload = dict(point.payload or {})
            if payload.get("file_path") == file_path and payload.get("name") == name:
                continue
            results.append(
                {
                    "file_path": payload.get("file_path"),
                    "name": payload.get("name"),
                    "node_type": payload.get("node_type"),
                    "start_line": payload.get("start_line"),
                    "end_line": payload.get("end_line"),
                    "div_score": payload.get("div_score"),
                    "text": payload.get("text"),
                    "score": float(point.score) if point.score is not None else 0.0,
                    "is_self": False,
                }
            )
            if len([r for r in results if not r.get("is_self")]) >= top_k:
                break
        return results


def _print_hits(hits: list[dict]) -> None:
    """Pretty-print retrieval hits."""
    if not hits:
        print("No matches.")
        return
    print(f"{'score':>7}  {'name':<28}  file_path")
    print("-" * 80)
    for hit in hits:
        print(
            f"{float(hit.get('score', 0)):.4f}  "
            f"{str(hit.get('name', ''))[:28]:<28}  "
            f"{hit.get('file_path')}"
        )
        text = (hit.get("text") or "").replace("\n", " ")
        if text:
            print(f"         {text[:120]}{'…' if len(text) > 120 else ''}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    default_repo = (
        Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "flask-test-repo"
    )
    repo = sys.argv[1] if len(sys.argv) > 1 else str(default_repo)
    query = sys.argv[2] if len(sys.argv) > 2 else "handle HTTP request routing"

    print(f"Model: {MODEL_NAME}")
    print(f"Parsing repository: {repo}")
    # Restrict walk noise from SKIP_DIRS already handled inside parse_repository.
    _ = SKIP_DIRS
    nodes = parse_repository(repo)
    stats = getattr(parse_repository, "last_stats", {})
    print(
        f"AST: files={stats.get('files_parsed')} "
        f"functions={stats.get('total_functions')} "
        f"classes={stats.get('total_classes')}"
    )

    try:
        client = CodeRAGClient(host=DEFAULT_HOST, port=DEFAULT_PORT)
        # Touch the server early for a clear error message.
        client._client.get_collections()
    except Exception as exc:  # noqa: BLE001
        print(
            "\nQdrant is not reachable at localhost:6333.\n"
            "Start it with:\n"
            "  docker run -d -p 6333:6333 -p 6334:6334 --name techdebt-qdrant "
            "-v D:\\techdebt-agent\\data\\qdrant:/qdrant/storage qdrant/qdrant\n"
            f"\nUnderlying error: {exc}"
        )
        raise SystemExit(1) from exc

    counts = client.index_repository(repo, nodes)
    print(f"Index stats: {counts}")

    print(f'\nQuery: "{query}"')
    hits = client.query_context(query, top_k=5)
    _print_hits(hits)
