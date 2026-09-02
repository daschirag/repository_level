"""Git history metrics via a single-pass repository crawl.

Performance diagnosis
---------------------
``get_change_frequency`` was already a single ``traverse_commits()`` loop
(not O(files × commits)). The real costs on large repos were:

1. **Duplicate full-history walks** — ``get_last_modified_dates`` repeated the
   same PyDriller traversal, roughly doubling wall time when both were used.
2. **PyDriller ``commit.modified_files``** — each access parses the full commit
   diff (~70 ms/commit on Flask). At ~5.5k commits that is ~6–7 minutes *per*
   pass, which matches the observed 15+ minute runs when both helpers ran.

Fix: one shared ``crawl_repo_history`` pass driven by ``git log --name-only``
(O(commits), sub-second on Flask). Thin wrappers preserve the old APIs.
``get_git_blame_summary`` stays on-demand / per-file via PyDriller.
"""

from __future__ import annotations

import logging
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydriller import Repository

logger = logging.getLogger(__name__)

_PROGRESS_EVERY = 500

# Cache the most recent crawl so get_change_frequency + get_last_modified_dates
# back-to-back still share one pass when called on the same repo path.
_LAST_CRAWL: dict[str, Any] = {"path": None, "data": None, "stats": None}


def _normalize_repo_path(repo_path: str) -> Path:
    """Resolve and validate a local git repository path."""
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {root}")
    if not (root / ".git").exists():
        raise ValueError(f"Not a git repository (missing .git): {root}")
    return root


def _rel_path(path: Optional[str]) -> Optional[str]:
    """Normalize a path to forward-slash relative form."""
    if not path:
        return None
    return path.replace("\\", "/")


def _parse_author_date(value: str) -> Optional[datetime]:
    """Parse an ISO-8601 author date from ``git log`` (``%aI``)."""
    text = value.strip()
    if not text:
        return None
    # datetime.fromisoformat handles ``2024-01-15T12:34:56+05:30``; accept ``Z``.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        logger.debug("Unparseable author date from git log: %r", value)
        return None


def crawl_repo_history(repo_path: str) -> dict[str, dict[str, Any]]:
    """Single-pass crawl of full git history for per-file churn metrics.

    Uses ``git log --name-only --pretty=format:%aI`` so each commit is visited
    exactly once without PyDriller's per-commit diff parse.

    Args:
        repo_path: Path to a local git repository.

    Returns:
        Mapping of repository-relative file path to a dict with:
        - ``change_frequency`` (``int``): number of commits touching the file
        - ``last_modified_date`` (``datetime | None``): newest author date
    """
    root = _normalize_repo_path(repo_path)
    counts: dict[str, int] = defaultdict(int)
    last_dates: dict[str, datetime] = {}

    cmd = [
        "git",
        "-C",
        str(root),
        "log",
        "--name-only",
        "--pretty=format:%aI",
        "--no-renames",
    ]
    logger.info("Crawling git history (single pass): %s", root)
    started = time.perf_counter()

    # Stream stdout so large repos do not require holding the full log in RAM
    # before parsing starts. ``text`` + line iteration is enough for Flask-scale.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None

    current_date: Optional[datetime] = None
    commits_seen = 0
    in_commit = False

    try:
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")

            # Blank line separates the pretty header from the name list, and
            # also appears between commits. A non-empty line that parses as a
            # date starts a new commit; a non-empty non-date line is a path.
            if not line:
                continue

            # %aI lines always look like 2024-01-15T12:34:56+00:00 (contain 'T'
            # and start with a digit). File paths almost never match this shape.
            if line[:1].isdigit() and "T" in line:
                parsed_date = _parse_author_date(line)
                if parsed_date is not None:
                    current_date = parsed_date
                    commits_seen += 1
                    in_commit = True
                    if commits_seen % _PROGRESS_EVERY == 0:
                        elapsed = time.perf_counter() - started
                        logger.info(
                            "Processed %d commits (%.1fs elapsed)",
                            commits_seen,
                            elapsed,
                        )
                    continue

            if not in_commit:
                continue

            path = _rel_path(line)
            if path is None:
                continue
            counts[path] += 1
            if current_date is not None:
                prev = last_dates.get(path)
                if prev is None or current_date > prev:
                    last_dates[path] = current_date

        stderr = proc.stderr.read() if proc.stderr is not None else ""
        returncode = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    if returncode != 0:
        raise RuntimeError(
            f"git log failed (exit {returncode}) for {root}: {stderr.strip()}"
        )

    elapsed = time.perf_counter() - started
    result = {
        path: {
            "change_frequency": count,
            "last_modified_date": last_dates.get(path),
        }
        for path, count in counts.items()
    }
    stats = {
        "commits": commits_seen,
        "files": len(result),
        "elapsed_seconds": elapsed,
    }
    logger.info(
        "History crawl complete: %d commits, %d files, %.2fs",
        commits_seen,
        len(result),
        elapsed,
    )
    crawl_repo_history.last_stats = stats  # type: ignore[attr-defined]
    _LAST_CRAWL["path"] = str(root)
    _LAST_CRAWL["data"] = result
    _LAST_CRAWL["stats"] = stats
    return result


def _cached_crawl(repo_path: str) -> dict[str, dict[str, Any]]:
    """Return cached crawl for ``repo_path`` when still valid, else recompute."""
    root = str(_normalize_repo_path(repo_path))
    if _LAST_CRAWL["path"] == root and _LAST_CRAWL["data"] is not None:
        crawl_repo_history.last_stats = _LAST_CRAWL["stats"]  # type: ignore[attr-defined]
        return _LAST_CRAWL["data"]
    return crawl_repo_history(repo_path)


def get_change_frequency(repo_path: str) -> dict[str, int]:
    """Count how many commits touch each file over the full git history.

    Thin wrapper around :func:`crawl_repo_history` (single shared pass).

    Args:
        repo_path: Path to a local git repository.

    Returns:
        Mapping of repository-relative file path -> commit count.
    """
    history = _cached_crawl(repo_path)
    return {path: int(info["change_frequency"]) for path, info in history.items()}


def get_last_modified_dates(repo_path: str) -> dict[str, datetime]:
    """Return the most recent author date for every file touched in history.

    Thin wrapper around :func:`crawl_repo_history` (single shared pass).

    Args:
        repo_path: Path to a local git repository.

    Returns:
        Mapping of repository-relative file path -> last commit author date.
        Files with an unparseable date are omitted.
    """
    history = _cached_crawl(repo_path)
    return {
        path: info["last_modified_date"]
        for path, info in history.items()
        if info.get("last_modified_date") is not None
    }


def get_git_blame_summary(repo_path: str, file_path: str) -> dict:
    """Summarize authorship and recency for a single file (on-demand).

    Intentionally per-file via PyDriller ``filepath=`` filtering — fine for
    top-k reporting, not for whole-repo precomputation.

    Args:
        repo_path: Path to a local git repository.
        file_path: Repository-relative path of the file to summarize.

    Returns:
        Dict with ``file_path``, ``author_count``, ``authors``, ``commit_count``,
        ``last_commit_date``, and ``last_author``.
    """
    root = _normalize_repo_path(repo_path)
    target = _rel_path(file_path)
    if target is None:
        raise ValueError("file_path must be a non-empty string")

    authors: set[str] = set()
    commit_count = 0
    last_commit_date: Optional[datetime] = None
    last_author: Optional[str] = None

    for commit in Repository(str(root), filepath=target).traverse_commits():
        touched = False
        for modified in commit.modified_files:
            new_p = _rel_path(modified.new_path)
            old_p = _rel_path(modified.old_path)
            if target in {new_p, old_p} or (
                new_p and new_p.endswith("/" + target)
            ) or (old_p and old_p.endswith("/" + target)):
                touched = True
                break
        if not touched and commit.modified_files:
            touched = True
        if not touched:
            continue

        commit_count += 1
        author_id = (commit.author.email or commit.author.name or "unknown").strip()
        authors.add(author_id)

        if last_commit_date is None or commit.author_date > last_commit_date:
            last_commit_date = commit.author_date
            last_author = author_id

    return {
        "file_path": target,
        "author_count": len(authors),
        "authors": sorted(authors),
        "commit_count": commit_count,
        "last_commit_date": last_commit_date.isoformat() if last_commit_date else None,
        "last_author": last_author,
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    default = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "flask-test-repo"
    target_repo = sys.argv[1] if len(sys.argv) > 1 else str(default)

    history = crawl_repo_history(target_repo)
    stats = getattr(crawl_repo_history, "last_stats", {})
    print(
        f"files={stats.get('files')} commits={stats.get('commits')} "
        f"elapsed={stats.get('elapsed_seconds'):.2f}s"
    )
    # Show a few hot files.
    top = sorted(
        history.items(),
        key=lambda item: item[1]["change_frequency"],
        reverse=True,
    )[:10]
    print("Top 10 by change_frequency:")
    for path, info in top:
        print(f"  {info['change_frequency']:>5}  {path}")
