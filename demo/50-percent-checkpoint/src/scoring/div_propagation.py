"""Causal Debt Impact Vector (DIV) propagation.

==========================================================================
ALGORITHM (plain-language — for patent disclosure / thesis writeup)
--------------------------------------------------------------------------
Technical debt is not evenly distributed: a complex, frequently-changed
function that many other functions depend on is more dangerous than an
equally complex leaf function that nothing calls.

DIV propagation formalizes this. Each function receives a *base debt
weight* combining its own complexity and change churn:

    base_weight = cyclomatic_complexity * log(1 + change_frequency)

The logarithmic dampening prevents raw commit volume from dominating.

We then propagate debt *along the call graph in the upstream direction*
(from a function toward everything that transitively calls it). A
function's DIV score is its own base weight plus a decaying contribution
from every transitive caller:

    DIV(v) = base(v) + Σ_{u ∈ ancestors(v)}  base(u) * δ^{dist(u,v)}

where δ = 0.7 is the decay factor and dist(u,v) is the shortest-path
caller-depth from ancestor u to v (at most 6 hops). Cycles (recursion,
mutual recursion) are broken by visiting each ancestor at most once, at
its shortest-path depth.

Intuition: debt in a widely-depended-upon module cascades to every
caller, but influence fades with distance so distant, incidental callers
do not inflate the score unboundedly.
==========================================================================

Implementation note
-------------------
We pull the CALLS subgraph into NetworkX and compute DIV in Python rather
than via Cypher variable-length paths. Reasons:

1. Shortest-path ancestor depth with a hard hop cap and cycle breaking is
   a straightforward BFS on the reverse call graph; Cypher path matching
   does not natively return "shortest depth only" while also applying a
   per-hop decay aggregation without expensive post-filters.
2. NetworkX BFS visits each node once (natural cycle break) and scales
   fine for repo-sized call graphs that already fit in Neo4j memory.
3. Scores are written back with batched parameterized Cypher UNWIND.
"""

from __future__ import annotations

import logging
import math
import sys
from collections import deque
from typing import Any, Optional

import networkx as nx
from neo4j import Driver, GraphDatabase

logger = logging.getLogger(__name__)

DECAY_FACTOR: float = 0.7
MAX_DEPTH: int = 6
BATCH_SIZE: int = 500

DEFAULT_URI = "bolt://127.0.0.1:7687"
DEFAULT_USER = "neo4j"
DEFAULT_PASSWORD = "techdebt123"


def _node_key(file_path: str, name: str) -> str:
    """Stable unique key for a function: ``file_path::name``."""
    return f"{file_path}::{name}"


def _base_weight(cyclomatic_complexity: float, change_frequency: float) -> float:
    """Compute the local debt weight for a single function."""
    cc = max(float(cyclomatic_complexity or 0.0), 0.0)
    cf = max(float(change_frequency or 0.0), 0.0)
    return cc * math.log(1.0 + cf)


def _load_call_graph(driver: Driver) -> tuple[nx.DiGraph, dict[str, dict[str, Any]]]:
    """Load ``:Function`` nodes and ``CALLS`` edges into a NetworkX digraph.

    Edge direction mirrors Neo4j: ``caller -> callee``.

    Returns:
        graph: Directed call graph keyed by ``file_path::name``.
        attrs: Per-key property dicts (file_path, name, cc, cf, base_weight).
    """
    attrs: dict[str, dict[str, Any]] = {}
    graph = nx.DiGraph()

    with driver.session() as session:
        node_result = session.run(
            """
            MATCH (f:Function)
            RETURN f.file_path AS file_path,
                   f.name AS name,
                   coalesce(f.cyclomatic_complexity, 0) AS cc,
                   coalesce(f.change_frequency, 0) AS cf
            """
        )
        for record in node_result:
            file_path = record["file_path"]
            name = record["name"]
            if not file_path or not name:
                continue
            key = _node_key(file_path, name)
            cc = float(record["cc"] or 0)
            cf = float(record["cf"] or 0)
            attrs[key] = {
                "file_path": file_path,
                "name": name,
                "cyclomatic_complexity": cc,
                "change_frequency": cf,
                "base_weight": _base_weight(cc, cf),
            }
            graph.add_node(key)

        edge_result = session.run(
            """
            MATCH (caller:Function)-[:CALLS]->(callee:Function)
            RETURN caller.file_path AS caller_file,
                   caller.name AS caller_name,
                   callee.file_path AS callee_file,
                   callee.name AS callee_name
            """
        )
        for record in edge_result:
            caller_key = _node_key(record["caller_file"], record["caller_name"])
            callee_key = _node_key(record["callee_file"], record["callee_name"])
            if caller_key in attrs and callee_key in attrs:
                graph.add_edge(caller_key, callee_key)

    return graph, attrs


def _div_for_node(
    target: str,
    reverse_graph: nx.DiGraph,
    base_weights: dict[str, float],
    decay: float = DECAY_FACTOR,
    max_depth: int = MAX_DEPTH,
) -> float:
    """Propagate upstream caller debt onto ``target`` via BFS.

    Traverses the *reverse* CALLS graph (callee -> caller) so neighbors of
    ``target`` are its direct callers. Each ancestor is visited once at its
    shortest-path depth (cycle-safe). Contributions beyond ``max_depth``
    are ignored.
    """
    score = base_weights.get(target, 0.0)
    if target not in reverse_graph:
        return score

    visited: set[str] = {target}
    queue: deque[tuple[str, int]] = deque()

    for caller in reverse_graph.successors(target):
        if caller not in visited:
            visited.add(caller)
            queue.append((caller, 1))

    while queue:
        ancestor, depth = queue.popleft()
        if depth > max_depth:
            continue

        score += base_weights.get(ancestor, 0.0) * (decay**depth)

        if depth == max_depth:
            continue

        for next_caller in reverse_graph.successors(ancestor):
            if next_caller not in visited:
                visited.add(next_caller)
                queue.append((next_caller, depth + 1))

    return score


def compute_div_scores(driver: Driver) -> dict[str, float]:
    """Compute DIV scores for every ``:Function`` in the Neo4j graph.

    Args:
        driver: Open Neo4j driver connected to a populated call graph.

    Returns:
        Mapping of ``file_path::name`` -> DIV score.
    """
    graph, attrs = _load_call_graph(driver)
    if not attrs:
        logger.warning("No :Function nodes found; returning empty DIV scores")
        return {}

    base_weights = {key: data["base_weight"] for key, data in attrs.items()}
    # Reverse edges: callee -> caller, so BFS walks "upstream" toward callers.
    reverse_graph = graph.reverse(copy=True)

    scores: dict[str, float] = {}
    for key in attrs:
        scores[key] = _div_for_node(key, reverse_graph, base_weights)

    logger.info("Computed DIV scores for %d functions", len(scores))
    return scores


def write_div_scores(driver: Driver, scores: dict[str, float]) -> int:
    """Persist DIV scores onto ``:Function`` nodes as ``div_score``.

    Args:
        driver: Open Neo4j driver.
        scores: Mapping from ``file_path::name`` to DIV score.

    Returns:
        Number of score rows written.
    """
    rows: list[dict[str, Any]] = []
    for key, score in scores.items():
        if "::" not in key:
            logger.warning("Skipping malformed score key: %s", key)
            continue
        file_path, name = key.split("::", 1)
        rows.append(
            {
                "file_path": file_path,
                "name": name,
                "div_score": float(score),
            }
        )

    if not rows:
        return 0

    query = """
    UNWIND $rows AS row
    MATCH (f:Function {
        file_path: row.file_path,
        name: row.name,
        node_type: 'function'
    })
    SET f.div_score = row.div_score
    """
    with driver.session() as session:
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start : start + BATCH_SIZE]
            session.run(query, {"rows": batch})

    return len(rows)


def get_top_debt_nodes(driver: Driver, k: int = 10) -> list[dict]:
    """Return the top-``k`` functions ranked by ``div_score`` descending.

    Args:
        driver: Open Neo4j driver.
        k: Maximum number of nodes to return.

    Returns:
        List of dicts with ``file_path``, ``name``, ``div_score``,
        ``cyclomatic_complexity``, and ``change_frequency``.
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (f:Function)
            WHERE f.div_score IS NOT NULL
            RETURN f.file_path AS file_path,
                   f.name AS name,
                   f.div_score AS div_score,
                   coalesce(f.cyclomatic_complexity, 0) AS cyclomatic_complexity,
                   coalesce(f.change_frequency, 0) AS change_frequency
            ORDER BY f.div_score DESC
            LIMIT $k
            """,
            {"k": int(k)},
        )
        return [dict(record) for record in result]


def _print_top_table(rows: list[dict]) -> None:
    """Print top debt nodes as a fixed-width table."""
    if not rows:
        print("No DIV scores found in Neo4j.")
        return

    headers = ("rank", "div_score", "cc", "chg", "name", "file_path")
    print(
        f"{headers[0]:>4}  {headers[1]:>10}  {headers[2]:>4}  "
        f"{headers[3]:>4}  {headers[4]:<24}  {headers[5]}"
    )
    print("-" * 88)
    for i, row in enumerate(rows, start=1):
        print(
            f"{i:>4}  {float(row['div_score']):>10.4f}  "
            f"{int(row['cyclomatic_complexity']):>4}  "
            f"{int(row['change_frequency']):>4}  "
            f"{str(row['name'])[:24]:<24}  {row['file_path']}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    uri = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URI
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    driver = GraphDatabase.driver(uri, auth=(DEFAULT_USER, DEFAULT_PASSWORD))
    try:
        driver.verify_connectivity()
        print(f"Connected to Neo4j at {uri}")

        scores = compute_div_scores(driver)
        written = write_div_scores(driver, scores)
        print(f"Wrote div_score onto {written} :Function nodes")

        top = get_top_debt_nodes(driver, k=top_k)
        print(f"\nTop {len(top)} debt nodes by DIV:")
        _print_top_table(top)
    finally:
        driver.close()
