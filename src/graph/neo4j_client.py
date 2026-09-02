"""Neo4j client for writing AST call graphs and debt attributes.

Uses batched ``UNWIND`` Cypher writes (500 rows) and parameterized queries.
"""

from __future__ import annotations

import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from neo4j import Driver, GraphDatabase
from radon.complexity import cc_visit
from radon.visitors import Function as RadonFunction

from src.graph.schema import initialize_schema
from src.ingestion.ast_parser import SKIP_DIRS, parse_repository
from src.ingestion.git_crawler import get_change_frequency

logger = logging.getLogger(__name__)

DEFAULT_URI = "bolt://127.0.0.1:7687"
DEFAULT_USER = "neo4j"
DEFAULT_PASSWORD = "techdebt123"
BATCH_SIZE = 500

# Names that should never become CALLS edges (builtins / dunders / common
# container methods). Without this, scoped resolution can still false-match a
# local helper named ``get`` when the call was actually ``dict.get``.
_BUILTIN_DENYLIST: set[str] = {
    # builtins / types
    "abs",
    "all",
    "any",
    "bool",
    "bytearray",
    "bytes",
    "callable",
    "classmethod",
    "dict",
    "dir",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "getattr",
    "globals",
    "hasattr",
    "hash",
    "help",
    "id",
    "input",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "locals",
    "map",
    "max",
    "min",
    "next",
    "object",
    "open",
    "ord",
    "print",
    "property",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "setattr",
    "slice",
    "sorted",
    "staticmethod",
    "str",
    "sum",
    "super",
    "tuple",
    "type",
    "vars",
    "zip",
    # common container / str methods
    "add",
    "append",
    "clear",
    "copy",
    "count",
    "decode",
    "encode",
    "endswith",
    "extend",
    "find",
    "get",
    "index",
    "insert",
    "items",
    "join",
    "keys",
    "lower",
    "pop",
    "popitem",
    "remove",
    "replace",
    "setdefault",
    "split",
    "startswith",
    "strip",
    "update",
    "upper",
    "values",
    "write",
    "read",
    "close",
    # dunder methods
    "__init__",
    "__str__",
    "__repr__",
    "__call__",
    "__enter__",
    "__exit__",
    "__getitem__",
    "__setitem__",
    "__delitem__",
    "__iter__",
    "__next__",
    "__len__",
    "__contains__",
    "__eq__",
    "__hash__",
    "__getattr__",
    "__setattr__",
    "__new__",
    "__bool__",
}

_IMPORT_RE = re.compile(
    r"^import\s+([\w.]+)(?:\s+as\s+(\w+))?",
    re.MULTILINE,
)
_FROM_IMPORT_RE = re.compile(
    r"^from\s+([.\w]+)\s+import\s+(.+)$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class _FuncRef:
    """A function/method definition used for scoped call resolution."""

    file_path: str
    name: str
    parent: Optional[str]  # enclosing class/function name from AST, if any
    start_line: int


def _batched(items: Sequence[Any], size: int = BATCH_SIZE) -> Iterator[list[Any]]:
    """Yield successive slices of ``items`` with at most ``size`` elements."""
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _normalize_callee_name(name: str) -> str:
    """Reduce a call expression to a simple identifier when possible.

    Examples: ``self.foo`` -> ``foo``, ``pkg.mod.bar`` -> ``bar``, ``foo`` -> ``foo``.
    """
    _, bare = _split_callee(name)
    return bare


def _split_callee(name: str) -> tuple[Optional[str], str]:
    """Split a call expression into ``(qualifier, bare_name)``.

    ``self.foo`` / ``cls.foo`` -> ``(None, "foo")`` (attribute on receiver).
    ``helpers.foo`` -> ``("helpers", "foo")``.
    ``foo`` -> ``(None, "foo")``.
    """
    text = name.strip()
    if not text or text.startswith("<"):
        return None, text
    if "(" in text:
        text = text.split("(", 1)[0].strip()
    if not text:
        return None, text
    if "." not in text:
        return None, text
    parts = text.split(".")
    bare = parts[-1].strip()
    head = parts[0].strip()
    if head in {"self", "cls", "cls_"}:
        return None, bare
    if len(parts) == 2:
        return head, bare
    # pkg.mod.func -> treat last qualifier segment as the import alias/module tip
    return parts[-2].strip(), bare


def _module_candidates(module: str, importer_file: str) -> list[str]:
    """Expand a Python import module path into possible repo-relative file paths."""
    module = module.strip()
    if not module:
        return []

    # Relative import: from .helpers / from ..pkg.mod
    if module.startswith("."):
        dots = len(module) - len(module.lstrip("."))
        remainder = module.lstrip(".")
        importer = Path(importer_file)
        base = importer.parent
        for _ in range(dots - 1):
            base = base.parent if base.parent != base else base
        if remainder:
            rel = (base / Path(*remainder.split("."))).as_posix()
        else:
            rel = base.as_posix()
        return [
            f"{rel}.py",
            f"{rel}/__init__.py",
        ]

    dotted = module.replace(".", "/")
    return [
        f"{dotted}.py",
        f"{dotted}/__init__.py",
        f"src/{dotted}.py",
        f"src/{dotted}/__init__.py",
    ]


def _resolve_module_file(
    module: str,
    importer_file: str,
    defined_files: set[str],
) -> Optional[str]:
    """Map an import module string to a repo file that defines symbols, if any."""
    for candidate in _module_candidates(module, importer_file):
        norm = candidate.replace("\\", "/")
        if norm in defined_files:
            return norm
        # Soft match: endswith (handles flask/app.py vs src/flask/app.py)
        matches = [f for f in defined_files if f.endswith("/" + norm) or f == norm]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # Prefer the shortest path (usually the canonical package path).
            return sorted(matches, key=len)[0]
    return None


def _parse_import_statement(text: str) -> list[tuple[str, Optional[str]]]:
    """Parse an import node name into ``(module, imported_symbol_or_None)`` pairs.

    ``imported_symbol`` is None for ``import pkg.mod`` (whole module binding).
    """
    label = " ".join(text.split())
    results: list[tuple[str, Optional[str]]] = []

    for match in _IMPORT_RE.finditer(label):
        module = match.group(1)
        results.append((module, None))

    for match in _FROM_IMPORT_RE.finditer(label):
        module = match.group(1)
        names_blob = match.group(2)
        if names_blob.strip().startswith("("):
            names_blob = names_blob.strip()[1:]
        if names_blob.strip().endswith(")"):
            names_blob = names_blob.strip()[:-1]
        for part in names_blob.split(","):
            part = part.strip()
            if not part or part == "*":
                results.append((module, None))
                continue
            # ``name as alias`` -> alias is what appears at call sites
            if " as " in part:
                _original, alias = part.split(" as ", 1)
                results.append((module, alias.strip()))
            else:
                results.append((module, part.split(".", 1)[0].strip()))

    return results


def _build_resolution_indexes(
    nodes: list[dict],
) -> tuple[
    dict[str, list[_FuncRef]],
    dict[str, set[str]],
    dict[str, dict[str, list[_FuncRef]]],
    dict[str, dict[str, list[str]]],
]:
    """Build indexes for scoped call resolution.

    Returns:
        funcs_by_file: file -> list of function refs
        classes_by_file: file -> set of class names
        funcs_by_file_name: file -> name -> list of refs (overloads / multi-class)
        import_name_to_files: importer_file -> symbol_or_module_alias -> [def files]
    """
    funcs_by_file: dict[str, list[_FuncRef]] = defaultdict(list)
    classes_by_file: dict[str, set[str]] = defaultdict(set)
    funcs_by_file_name: dict[str, dict[str, list[_FuncRef]]] = defaultdict(
        lambda: defaultdict(list)
    )
    defined_files: set[str] = set()

    for node in nodes:
        kind = node.get("node_type")
        file_path = node.get("file_path")
        name = node.get("name")
        if not file_path or not name:
            continue
        file_path = str(file_path).replace("\\", "/")
        if kind == "class":
            classes_by_file[file_path].add(str(name))
            defined_files.add(file_path)
        elif kind == "function":
            ref = _FuncRef(
                file_path=file_path,
                name=str(name),
                parent=node.get("parent"),
                start_line=int(node.get("start_line") or 0),
            )
            funcs_by_file[file_path].append(ref)
            funcs_by_file_name[file_path][ref.name].append(ref)
            defined_files.add(file_path)

    # Also index every path that appears on any node (imports may target packages).
    for node in nodes:
        fp = node.get("file_path")
        if fp:
            defined_files.add(str(fp).replace("\\", "/"))

    import_name_to_files: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for node in nodes:
        if node.get("node_type") != "import":
            continue
        file_path = str(node.get("file_path") or "").replace("\\", "/")
        import_text = node.get("name")
        if not file_path or not import_text:
            continue
        for module, symbol in _parse_import_statement(str(import_text)):
            resolved = _resolve_module_file(module, file_path, defined_files)
            if resolved is None:
                continue
            # Whole-module binding: ``import flask.helpers`` or ``from . import helpers``
            alias = symbol or module.rsplit(".", 1)[-1]
            bucket = import_name_to_files[file_path][alias]
            if resolved not in bucket:
                bucket.append(resolved)
            if symbol:
                # ``from mod import symbol`` — symbol may itself be the callee name
                sym_bucket = import_name_to_files[file_path][symbol]
                if resolved not in sym_bucket:
                    sym_bucket.append(resolved)

    return funcs_by_file, classes_by_file, funcs_by_file_name, import_name_to_files


def _enclosing_class(
    file_path: str,
    parent_name: Optional[str],
    funcs_by_file_name: dict[str, dict[str, list[_FuncRef]]],
    classes_by_file: dict[str, set[str]],
) -> Optional[str]:
    """Infer the class enclosing a call site from the AST parent chain."""
    if not parent_name:
        return None
    classes = classes_by_file.get(file_path, set())
    if parent_name in classes:
        return parent_name
    # parent is a function/method — use that function's parent if it is a class
    for ref in funcs_by_file_name.get(file_path, {}).get(parent_name, []):
        if ref.parent and ref.parent in classes:
            return ref.parent
    return None


def _resolve_callee(
    *,
    caller_file: str,
    caller_parent: str,
    qualifier: Optional[str],
    callee_name: str,
    funcs_by_file_name: dict[str, dict[str, list[_FuncRef]]],
    classes_by_file: dict[str, set[str]],
    import_name_to_files: dict[str, dict[str, list[str]]],
) -> Optional[_FuncRef]:
    """Resolve a callee using same-class > same-file > import-scoped priority.

    Returns None when unresolved (no global name fallback).
    """
    if callee_name in _BUILTIN_DENYLIST or callee_name.startswith("__"):
        return None

    class_name = _enclosing_class(
        caller_file, caller_parent, funcs_by_file_name, classes_by_file
    )

    # 1) Same-class (most specific within a multi-class file)
    if class_name:
        class_hits = [
            ref
            for ref in funcs_by_file_name.get(caller_file, {}).get(callee_name, [])
            if ref.parent == class_name
        ]
        if len(class_hits) == 1:
            return class_hits[0]
        if len(class_hits) > 1:
            # Same class shouldn't redefine the same method name; take earliest.
            return sorted(class_hits, key=lambda r: r.start_line)[0]

    # 2) Same-file (module-level or unique definition)
    file_hits = list(funcs_by_file_name.get(caller_file, {}).get(callee_name, []))
    if class_name:
        # Prefer non-other-class matches when we already missed same-class.
        file_hits = [r for r in file_hits if r.parent == class_name or r.parent is None
                     or r.parent not in classes_by_file.get(caller_file, set())]
    if len(file_hits) == 1:
        return file_hits[0]
    if len(file_hits) > 1:
        module_level = [r for r in file_hits if r.parent is None
                        or r.parent not in classes_by_file.get(caller_file, set())]
        if len(module_level) == 1:
            return module_level[0]
        # Ambiguous within file — do not guess.
        return None

    # 3) Import-scoped
    imports = import_name_to_files.get(caller_file, {})
    candidate_files: list[str] = []
    if qualifier and qualifier in imports:
        candidate_files.extend(imports[qualifier])
    if callee_name in imports:
        candidate_files.extend(imports[callee_name])

    # De-dupe while preserving order
    seen: set[str] = set()
    ordered_files: list[str] = []
    for fpath in candidate_files:
        if fpath not in seen:
            seen.add(fpath)
            ordered_files.append(fpath)

    import_hits: list[_FuncRef] = []
    for fpath in ordered_files:
        import_hits.extend(funcs_by_file_name.get(fpath, {}).get(callee_name, []))

    if len(import_hits) == 1:
        return import_hits[0]
    if len(import_hits) > 1:
        # Prefer module-level definitions in the imported file.
        module_level = [
            r
            for r in import_hits
            if r.parent is None
            or r.parent not in classes_by_file.get(r.file_path, set())
        ]
        if len(module_level) == 1:
            return module_level[0]
        # Still ambiguous across imports — skip.
        return None

    # 4) No global fallback
    return None


def compute_radon_scores(repo_path: str) -> dict[str, dict[str, int]]:
    """Compute cyclomatic complexity per function via ``radon.complexity.cc_visit``.

    Only Python source files are scored (radon's supported surface).

    Args:
        repo_path: Repository root path.

    Returns:
        Mapping of ``file_path -> {function_name: complexity}``.
    """
    root = Path(repo_path).resolve()
    scores: dict[str, dict[str, int]] = {}

    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            blocks = cc_visit(source)
        except Exception as exc:  # noqa: BLE001
            logger.warning("radon failed for %s: %s", rel, exc)
            continue

        file_scores: dict[str, int] = {}
        for block in blocks:
            if isinstance(block, RadonFunction):
                file_scores[block.name] = int(block.complexity)
            else:
                # Class blocks expose methods via .methods
                for method in getattr(block, "methods", []) or []:
                    file_scores[method.name] = int(method.complexity)
        if file_scores:
            scores[rel] = file_scores

    return scores


class GraphClient:
    """Thin Neo4j wrapper for Layer-2 call-graph ingestion."""

    def __init__(
        self,
        uri: str = DEFAULT_URI,
        user: str = DEFAULT_USER,
        password: str = DEFAULT_PASSWORD,
    ) -> None:
        """Open a Neo4j driver.

        Args:
            uri: Bolt URI (defaults to local Docker Neo4j).
            user: Database username.
            password: Database password.
        """
        self._uri = uri
        self._user = user
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    def __enter__(self) -> GraphClient:
        """Enter the context manager, returning ``self``."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Close the driver on context exit."""
        self.close()

    def close(self) -> None:
        """Close the underlying Neo4j driver."""
        self._driver.close()

    @property
    def driver(self) -> Driver:
        """Expose the raw Neo4j driver (e.g. for ``initialize_schema``)."""
        return self._driver

    def _write_batches(self, query: str, rows: Sequence[dict[str, Any]]) -> int:
        """Run ``query`` over ``rows`` in batches of ``BATCH_SIZE``.

        Returns:
            Total number of input rows processed.
        """
        if not rows:
            return 0
        with self._driver.session() as session:
            for batch in _batched(rows):
                session.run(query, {"rows": batch})
        return len(rows)

    def write_ast_nodes(self, nodes: list[dict]) -> dict[str, int]:
        """Upsert ``:Function`` / ``:Class`` nodes from AST extraction results.

        Only entries with ``node_type`` of ``function`` or ``class`` are written.
        Properties stored: ``name``, ``file_path``, ``start_line``, ``end_line``,
        ``node_type``.

        Args:
            nodes: Output of ``parse_repository``.

        Returns:
            Counts of functions and classes written (input row counts).
        """
        functions: list[dict[str, Any]] = []
        classes: list[dict[str, Any]] = []

        for node in nodes:
            kind = node.get("node_type")
            name = node.get("name")
            file_path = node.get("file_path")
            if not name or not file_path:
                continue
            row = {
                "name": name,
                "file_path": file_path,
                "start_line": int(node.get("start_line") or 0),
                "end_line": int(node.get("end_line") or 0),
                "node_type": kind,
            }
            if kind == "function":
                functions.append(row)
            elif kind == "class":
                classes.append(row)

        func_query = """
        UNWIND $rows AS row
        MERGE (n:Function {
            file_path: row.file_path,
            name: row.name,
            node_type: row.node_type
        })
        SET n.start_line = row.start_line,
            n.end_line = row.end_line
        """
        class_query = """
        UNWIND $rows AS row
        MERGE (n:Class {
            file_path: row.file_path,
            name: row.name,
            node_type: row.node_type
        })
        SET n.start_line = row.start_line,
            n.end_line = row.end_line
        """

        # Also ensure a :File node exists for each referenced path.
        file_paths = sorted({r["file_path"] for r in functions + classes})
        file_query = """
        UNWIND $rows AS row
        MERGE (f:File {path: row.path})
        """
        self._write_batches(file_query, [{"path": p} for p in file_paths])

        return {
            "functions": self._write_batches(func_query, functions),
            "classes": self._write_batches(class_query, classes),
            "files": len(file_paths),
        }

    def write_call_edges(self, nodes: list[dict]) -> int:
        """Create ``CALLS`` edges using scoped callee resolution.

        Resolution priority (no global name fallback):

        1. Same-class method in the call site's file
        2. Unique same-file definition
        3. Import-scoped definition from a resolved local module
        4. Skip (unresolved / builtin denylist) — never link to an arbitrary
           same-named function elsewhere in the repo

        Args:
            nodes: Output of ``parse_repository``.

        Returns:
            Number of resolved ``CALLS`` rows written to Neo4j.
        """
        (
            _funcs_by_file,
            classes_by_file,
            funcs_by_file_name,
            import_name_to_files,
        ) = _build_resolution_indexes(nodes)

        rows: list[dict[str, Any]] = []
        unresolved = 0
        denied = 0
        call_sites = 0
        # Also count how many edges the old global-name matcher would create
        # (for before/after reporting without a second full ingest).
        legacy_edge_pairs: set[tuple[str, str, str, str]] = set()
        all_funcs_by_name: dict[str, list[_FuncRef]] = defaultdict(list)
        for file_map in funcs_by_file_name.values():
            for name, refs in file_map.items():
                all_funcs_by_name[name].extend(refs)

        for node in nodes:
            if node.get("node_type") != "call":
                continue
            parent = node.get("parent")
            callee_raw = node.get("name")
            file_path = node.get("file_path")
            if not parent or not callee_raw or not file_path:
                continue
            file_path = str(file_path).replace("\\", "/")
            qualifier, callee = _split_callee(str(callee_raw))
            if not callee or callee.startswith("<"):
                continue
            call_sites += 1

            # Legacy global-name collision count (diagnostic only).
            if callee not in _BUILTIN_DENYLIST and not callee.startswith("__"):
                for target in all_funcs_by_name.get(callee, []):
                    legacy_edge_pairs.add(
                        (file_path, str(parent), target.file_path, target.name)
                    )

            if callee in _BUILTIN_DENYLIST or callee.startswith("__"):
                denied += 1
                continue

            resolved = _resolve_callee(
                caller_file=file_path,
                caller_parent=str(parent),
                qualifier=qualifier,
                callee_name=callee,
                funcs_by_file_name=funcs_by_file_name,
                classes_by_file=classes_by_file,
                import_name_to_files=import_name_to_files,
            )
            if resolved is None:
                unresolved += 1
                continue

            rows.append(
                {
                    "caller_name": str(parent),
                    "caller_file": file_path,
                    "callee_name": resolved.name,
                    "callee_file": resolved.file_path,
                }
            )

        # De-dupe identical call edges before write.
        unique_rows = [
            dict(t)
            for t in {
                (
                    r["caller_name"],
                    r["caller_file"],
                    r["callee_name"],
                    r["callee_file"],
                ): r
                for r in rows
            }.values()
        ]

        query = """
        UNWIND $rows AS row
        MATCH (caller:Function {
            name: row.caller_name,
            file_path: row.caller_file,
            node_type: 'function'
        })
        MATCH (callee:Function {
            name: row.callee_name,
            file_path: row.callee_file,
            node_type: 'function'
        })
        MERGE (caller)-[:CALLS]->(callee)
        """
        written = self._write_batches(query, unique_rows)

        self.last_call_edge_stats = {
            "call_sites": call_sites,
            "resolved_edges": written,
            "unresolved": unresolved,
            "denied_builtins": denied,
            "legacy_global_edges": len(legacy_edge_pairs),
        }
        logger.info(
            "CALLS resolution: sites=%d resolved=%d unresolved=%d "
            "denied_builtins=%d (legacy global would create %d edges)",
            call_sites,
            written,
            unresolved,
            denied,
            len(legacy_edge_pairs),
        )
        return written

    def write_import_edges(self, nodes: list[dict]) -> int:
        """Create ``IMPORTS`` relationships from ``:File`` to ``:Module``.

        Args:
            nodes: Output of ``parse_repository``.

        Returns:
            Number of import rows submitted.
        """
        rows: list[dict[str, Any]] = []
        for node in nodes:
            if node.get("node_type") != "import":
                continue
            file_path = node.get("file_path")
            import_name = node.get("name")
            if not file_path or not import_name:
                continue
            # Collapse multi-line / verbose import text to a single-line label.
            label = " ".join(str(import_name).split())
            rows.append({"file_path": file_path, "import_name": label})

        query = """
        UNWIND $rows AS row
        MERGE (f:File {path: row.file_path})
        MERGE (m:Module {name: row.import_name})
        MERGE (f)-[:IMPORTS]->(m)
        """
        return self._write_batches(query, rows)

    def attach_debt_attributes(
        self,
        repo_path: str,
        change_frequency: dict,
        radon_scores: dict,
    ) -> int:
        """Set debt-related properties on every ``:Function`` node.

        Properties written:
        - ``change_frequency``: commits touching the function's file
        - ``cyclomatic_complexity``: radon score for that function (0 if unknown)

        Args:
            repo_path: Repository root (reserved for future path normalization).
            change_frequency: ``file_path -> commit_count`` from git_crawler.
            radon_scores: ``file_path -> {function_name -> complexity}`` from
                :func:`compute_radon_scores` (or equivalent).

        Returns:
            Number of function update rows submitted.
        """
        del repo_path  # available for callers that want path-aware scoring later
        freq = {str(k).replace("\\", "/"): int(v) for k, v in change_frequency.items()}
        radon: dict[str, dict[str, int]] = {}
        for file_path, funcs in radon_scores.items():
            norm = str(file_path).replace("\\", "/")
            if isinstance(funcs, dict):
                radon[norm] = {str(n): int(c) for n, c in funcs.items()}
            else:
                # Allow flat ``{(file, name): score}``-style iterables of pairs.
                logger.warning("Unexpected radon_scores entry for %s; skipping", file_path)

        rows: list[dict[str, Any]] = []
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (f:Function)
                RETURN f.file_path AS file_path, f.name AS name
                """
            )
            for record in result:
                file_path = record["file_path"]
                name = record["name"]
                file_radon = radon.get(file_path, {})
                rows.append(
                    {
                        "file_path": file_path,
                        "name": name,
                        "change_frequency": freq.get(file_path, 0),
                        "cyclomatic_complexity": file_radon.get(name, 0),
                    }
                )

        query = """
        UNWIND $rows AS row
        MATCH (f:Function {
            file_path: row.file_path,
            name: row.name,
            node_type: 'function'
        })
        SET f.change_frequency = row.change_frequency,
            f.cyclomatic_complexity = row.cyclomatic_complexity
        """
        return self._write_batches(query, rows)

    def graph_summary(self) -> dict[str, Any]:
        """Query node and relationship counts for a post-ingest summary."""
        with self._driver.session() as session:
            node_rows = session.run(
                """
                MATCH (n)
                RETURN labels(n)[0] AS label, count(*) AS count
                ORDER BY label
                """
            )
            nodes = {r["label"]: r["count"] for r in node_rows}

            edge_rows = session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(*) AS count
                ORDER BY type
                """
            )
            edges = {r["type"]: r["count"] for r in edge_rows}

        return {
            "nodes": nodes,
            "edges": edges,
            "node_total": sum(nodes.values()),
            "edge_total": sum(edges.values()),
        }

    def clear_graph(self) -> None:
        """Delete all nodes and relationships (useful before a full re-ingest)."""
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")


def _print_summary(summary: dict[str, Any]) -> None:
    """Pretty-print graph summary stats."""
    print("Neo4j graph summary")
    print(f"  total nodes : {summary['node_total']}")
    for label, count in summary["nodes"].items():
        print(f"    :{label} = {count}")
    print(f"  total edges : {summary['edge_total']}")
    for rel_type, count in summary["edges"].items():
        print(f"    -[{rel_type}]-> = {count}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    default_repo = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "mini_repo"
    repo = sys.argv[1] if len(sys.argv) > 1 else str(default_repo)
    uri = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_URI

    print(f"Parsing repository: {repo}")
    nodes = parse_repository(repo)
    stats = getattr(parse_repository, "last_stats", {})
    print(
        f"AST: {stats.get('files_parsed', '?')} files, "
        f"{stats.get('total_functions', '?')} functions, "
        f"{stats.get('total_classes', '?')} classes"
    )

    # Change frequency requires a git repo; fall back to empty map otherwise.
    try:
        change_freq = get_change_frequency(repo)
    except (ValueError, NotADirectoryError) as exc:
        logger.warning("Skipping change frequency (%s)", exc)
        change_freq = {}

    radon_scores = compute_radon_scores(repo)

    with GraphClient(uri=uri, user=DEFAULT_USER, password=DEFAULT_PASSWORD) as client:
        initialize_schema(client.driver)
        client.clear_graph()

        written = client.write_ast_nodes(nodes)
        print(f"Wrote nodes: {written}")

        call_rows = client.write_call_edges(nodes)
        call_stats = getattr(client, "last_call_edge_stats", {})
        print(f"Submitted CALLS edges: {call_rows}")
        if call_stats:
            print(
                f"  call sites={call_stats.get('call_sites')} "
                f"unresolved={call_stats.get('unresolved')} "
                f"denied_builtins={call_stats.get('denied_builtins')} "
                f"legacy_global_edges={call_stats.get('legacy_global_edges')}"
            )

        import_rows = client.write_import_edges(nodes)
        print(f"Submitted IMPORTS rows: {import_rows}")

        debt_rows = client.attach_debt_attributes(repo, change_freq, radon_scores)
        print(f"Attached debt attributes on functions: {debt_rows}")

        _print_summary(client.graph_summary())
