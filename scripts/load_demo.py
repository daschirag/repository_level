"""Load the Flask demo repository into Neo4j + Qdrant with one command.

    python scripts/load_demo.py              # library code only (src/)
    python scripts/load_demo.py --include-tests

What it does
------------
1. Clones ``pallets/flask`` (full history, ~13 MB) into ``data/demo/flask`` and
   checks out ``FLASK_COMMIT`` -- the exact commit that
   ``tests/fixtures/flask-test-repo`` was copied from. The fixture has no
   ``.git`` of its own, so the clone supplies real commit history for churn.
2. Verifies the fixture is byte-identical to that commit (via git blob ids).
3. Runs the real pipeline on the fixture: AST parse -> radon complexity ->
   Neo4j call graph -> DIV propagation -> Qdrant embeddings.
4. Writes an ``(:Analysis {id: 'latest'})`` node with the repo name, commit,
   counts and per-stage timings, which the dashboard header reads.

Requires Neo4j + Qdrant running (``docker compose up -d --wait``).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.neo4j_client import (  # noqa: E402
    DEFAULT_PASSWORD,
    DEFAULT_URI,
    DEFAULT_USER,
    GraphClient,
    compute_radon_scores,
)
from src.graph.schema import initialize_schema  # noqa: E402
from src.ingestion.ast_parser import parse_repository  # noqa: E402
from src.ingestion.git_crawler import get_change_frequency  # noqa: E402
from src.scoring.div_propagation import compute_div_scores, write_div_scores  # noqa: E402

logger = logging.getLogger("load_demo")

FLASK_URL = "https://github.com/pallets/flask"
FLASK_REPO_NAME = "pallets/flask"
# Commit the fixture was copied from (Flask 3.2.0.dev, 2026-05-31).
FLASK_COMMIT = "36e4a824f340fdee7ed50937ba8e7f6bc7d17f81"
FIXTURE_REL = "tests/fixtures/flask-test-repo"
CLONE_REL = "data/demo/flask"
# Paths excluded from the default (library-only) scope.
NON_LIBRARY_PREFIXES = ("tests/", "examples/", "docs/")


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout.strip()


def ensure_clone(clone_dir: Path) -> None:
    """Clone Flask with full history (if missing) and check out FLASK_COMMIT."""
    if not (clone_dir / ".git").exists():
        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Cloning %s into %s ...", FLASK_URL, clone_dir)
        subprocess.run(["git", "clone", "-q", FLASK_URL, str(clone_dir)], check=True)
    try:
        _git("cat-file", "-e", f"{FLASK_COMMIT}^{{commit}}", cwd=clone_dir)
    except subprocess.CalledProcessError:
        _git("fetch", "-q", "origin", cwd=clone_dir)
    _git("checkout", "-q", "--detach", FLASK_COMMIT, cwd=clone_dir)


def verify_fixture_matches(fixture_dir: Path, clone_dir: Path) -> bool:
    """Compare the fixture's tracked blob ids with the pinned Flask commit."""
    try:
        listing = _git("ls-files", "-s", "--", FIXTURE_REL, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError:
        return False
    prefix = FIXTURE_REL + "/"
    fixture = {
        line.split("\t", 1)[1][len(prefix):]: line.split()[1]
        for line in listing.splitlines()
        if "\t" in line
    }
    commit = {
        line.split("\t", 1)[1]: line.split()[2]
        for line in _git("ls-tree", "-r", FLASK_COMMIT, cwd=clone_dir).splitlines()
    }
    mismatched = [p for p, blob in fixture.items() if commit.get(p) != blob]
    if mismatched:
        logger.warning("Fixture differs from %s in %d files, e.g. %s", FLASK_COMMIT[:8], len(mismatched), mismatched[:3])
    return bool(fixture) and not mismatched


def reindex_qdrant(repo_path: Path, nodes: list[dict]) -> dict[str, int]:
    """Drop and rebuild the code-context collection for this repo."""
    from src.rag.qdrant_client import CodeRAGClient

    client = CodeRAGClient()
    if client._client.collection_exists(client.collection_name):
        client._client.delete_collection(client.collection_name)
    return client.index_repository(str(repo_path), nodes)


def write_analysis_node(client: GraphClient, props: dict[str, Any]) -> None:
    with client.driver.session() as session:
        session.run(
            "MERGE (a:Analysis {id: 'latest'}) SET a = $props, a.id = 'latest'",
            {"props": props},
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--include-tests", action="store_true", help="also analyse tests/, examples/ and docs/")
    parser.add_argument("--skip-qdrant", action="store_true", help="skip the (slow) embedding step")
    parser.add_argument("--uri", default=DEFAULT_URI, help="Neo4j bolt URI")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("neo4j").setLevel(logging.WARNING)  # schema "already exists" notices
    fixture_dir = PROJECT_ROOT / FIXTURE_REL
    clone_dir = PROJECT_ROOT / CLONE_REL
    timings: dict[str, float] = {}
    started = time.perf_counter()

    def timed(stage: str):
        class _Timer:
            def __enter__(self):
                self.t0 = time.perf_counter()
                logger.info("== %s", stage)

            def __exit__(self, *exc):
                timings[stage] = round(time.perf_counter() - self.t0, 2)
                logger.info("   %s took %.1fs", stage, timings[stage])

        return _Timer()

    with timed("history"):
        ensure_clone(clone_dir)
        verified = verify_fixture_matches(fixture_dir, clone_dir)
        commits_in_history = int(_git("rev-list", "--count", FLASK_COMMIT, cwd=clone_dir))
        commit_date = _git("show", "-s", "--format=%cI", FLASK_COMMIT, cwd=clone_dir)
        commit_subject = _git("show", "-s", "--format=%s", FLASK_COMMIT, cwd=clone_dir)
        change_freq = get_change_frequency(str(clone_dir))

    with timed("parse"):
        nodes = parse_repository(str(fixture_dir))
        if not args.include_tests:
            nodes = [n for n in nodes if not str(n.get("file_path", "")).replace("\\", "/").startswith(NON_LIBRARY_PREFIXES)]
        radon_scores = compute_radon_scores(str(fixture_dir))

    with timed("graph"):
        with GraphClient(uri=args.uri, user=DEFAULT_USER, password=DEFAULT_PASSWORD) as client:
            initialize_schema(client.driver)
            client.clear_graph()
            client.write_ast_nodes(nodes)
            client.write_call_edges(nodes)
            call_stats = dict(getattr(client, "last_call_edge_stats", {}) or {})
            client.write_import_edges(nodes)
            client.attach_debt_attributes(str(fixture_dir), change_freq, radon_scores)

    with timed("scoring"):
        with GraphClient(uri=args.uri, user=DEFAULT_USER, password=DEFAULT_PASSWORD) as client:
            scores = compute_div_scores(client.driver)
            write_div_scores(client.driver, scores)

    rag_counts: dict[str, int] = {}
    if not args.skip_qdrant:
        with timed("embeddings"):
            rag_counts = reindex_qdrant(fixture_dir, nodes)

    with GraphClient(uri=args.uri, user=DEFAULT_USER, password=DEFAULT_PASSWORD) as client:
        summary = client.graph_summary()
        props: dict[str, Any] = {
            "repo_name": FLASK_REPO_NAME,
            "repo_url": FLASK_URL,
            "commit": FLASK_COMMIT,
            "commit_date": commit_date,
            "commit_subject": commit_subject,
            "source_path": FIXTURE_REL,
            "history_path": CLONE_REL,
            "fixture_verified": verified,
            "scope": "all" if args.include_tests else "library (src/)",
            "commits_in_history": commits_in_history,
            "files_with_churn": len(change_freq),
            "call_sites": int(call_stats.get("call_sites", 0) or 0),
            "unresolved_calls": int(call_stats.get("unresolved", 0) or 0),
            "rag_embedded": int(rag_counts.get("embedded", 0)),
            "rag_considered": int(rag_counts.get("considered", 0)),
            "analysed_at": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(time.perf_counter() - started, 2),
            **{f"t_{k}_s": v for k, v in timings.items()},
        }
        write_analysis_node(client, props)

    print("\nDemo loaded:")
    print(f"  repo      : {FLASK_REPO_NAME} @ {FLASK_COMMIT[:8]} ({commit_subject})")
    print(f"  verified  : fixture matches commit = {verified}")
    print(f"  history   : {commits_in_history} commits, churn for {len(change_freq)} files")
    print(f"  graph     : {summary['nodes']} / {summary['edges']}")
    print(f"  embeddings: {rag_counts or 'skipped'}")
    print(f"  total     : {props['duration_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
