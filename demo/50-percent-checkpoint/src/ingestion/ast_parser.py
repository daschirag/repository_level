"""Multi-language AST extraction via tree-sitter.

Walks a repository, parses source files with the appropriate grammar, and
extracts function/method definitions, class definitions, imports, and call sites.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

from tree_sitter_languages import get_parser

logger = logging.getLogger(__name__)

# Extension -> tree-sitter language name used by tree-sitter-languages.
EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
}

# Directories skipped while walking a repository.
SKIP_DIRS: set[str] = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    "dist",
    "build",
    "target",
    "vendor",
    ".idea",
    ".vscode",
}

# Per-language grammar node types mapped to our logical categories.
# Values are sets of tree-sitter node type strings.
NODE_TYPE_MAP: dict[str, dict[str, set[str]]] = {
    "python": {
        "function": {"function_definition", "async_function_definition"},
        "class": {"class_definition"},
        "import": {"import_statement", "import_from_statement"},
        "call": {"call"},
    },
    "javascript": {
        "function": {
            "function_declaration",
            "generator_function_declaration",
            "method_definition",
            "arrow_function",
            "function_expression",
            "generator_function",
        },
        "class": {"class_declaration"},
        "import": {"import_statement"},
        "call": {"call_expression"},
    },
    "typescript": {
        "function": {
            "function_declaration",
            "generator_function_declaration",
            "method_definition",
            "arrow_function",
            "function_expression",
            "generator_function",
        },
        "class": {"class_declaration", "abstract_class_declaration"},
        "import": {"import_statement"},
        "call": {"call_expression"},
    },
    "tsx": {
        "function": {
            "function_declaration",
            "generator_function_declaration",
            "method_definition",
            "arrow_function",
            "function_expression",
            "generator_function",
        },
        "class": {"class_declaration", "abstract_class_declaration"},
        "import": {"import_statement"},
        "call": {"call_expression"},
    },
    "java": {
        "function": {
            "method_declaration",
            "constructor_declaration",
            "compact_constructor_declaration",
        },
        "class": {
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
        },
        "import": {"import_declaration"},
        "call": {"method_invocation", "object_creation_expression"},
    },
    "go": {
        "function": {"function_declaration", "method_declaration"},
        "class": {"type_declaration"},
        "import": {"import_declaration", "import_spec"},
        "call": {"call_expression"},
    },
    "rust": {
        "function": {"function_item"},
        "class": {"struct_item", "enum_item", "trait_item", "impl_item"},
        "import": {"use_declaration"},
        "call": {"call_expression"},
    },
}

# Node types that establish an enclosing scope for `parent`.
SCOPE_NODE_TYPES: set[str] = set()
for _lang_map in NODE_TYPE_MAP.values():
    SCOPE_NODE_TYPES.update(_lang_map["function"])
    SCOPE_NODE_TYPES.update(_lang_map["class"])


def _node_text(node: Any, source: bytes) -> str:
    """Return the UTF-8 source slice covered by ``node``."""
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _extract_name(node: Any, source: bytes, category: str) -> str:
    """Best-effort name extraction for a definition, import, or call node."""
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return _node_text(name_node, source)

    if category == "call":
        # Prefer the callee expression (identifier / member / attribute).
        for field in ("function", "method"):
            callee = node.child_by_field_name(field)
            if callee is not None:
                return _node_text(callee, source)
        if node.named_child_count > 0:
            return _node_text(node.named_children[0], source)

    if category == "import":
        return _node_text(node, source).strip()

    if category == "function":
        # Arrow / anonymous functions: try parent property / variable name.
        parent = node.parent
        if parent is not None and parent.type in {
            "variable_declarator",
            "pair",
            "assignment_expression",
            "public_field_definition",
        }:
            pname = parent.child_by_field_name("name") or parent.child_by_field_name("key")
            if pname is not None:
                return _node_text(pname, source)
            if parent.named_child_count > 0:
                return _node_text(parent.named_children[0], source)
        return "<anonymous>"

    if category == "class":
        # Go type_declaration / Rust impl_item often nest the named type.
        for child in node.named_children:
            child_name = child.child_by_field_name("name")
            if child_name is not None:
                return _node_text(child_name, source)
            if child.type in {"type_identifier", "identifier"}:
                return _node_text(child, source)

    return _node_text(node, source).split("\n", 1)[0][:120]


def _category_for_node(node_type: str, language: str) -> Optional[str]:
    """Return logical category (function/class/import/call) or None."""
    mapping = NODE_TYPE_MAP.get(language, {})
    for category, types in mapping.items():
        if node_type in types:
            return category
    return None


def _walk_tree(
    node: Any,
    source: bytes,
    file_path: str,
    language: str,
    results: list[dict[str, Any]],
    parent_stack: list[str],
) -> None:
    """Recursively visit ``node``, appending extracted entities to ``results``."""
    category = _category_for_node(node.type, language)
    pushed = False
    name: Optional[str] = None

    if category is not None:
        name = _extract_name(node, source, category)
        parent_name = parent_stack[-1] if parent_stack else None
        results.append(
            {
                "node_type": category,
                "name": name,
                "file_path": file_path,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "parent": parent_name,
            }
        )
        if node.type in SCOPE_NODE_TYPES and name:
            parent_stack.append(name)
            pushed = True

    for child in node.children:
        _walk_tree(child, source, file_path, language, results, parent_stack)

    if pushed:
        parent_stack.pop()


def _parse_file(file_path: Path, repo_path: Path, language: str) -> list[dict[str, Any]]:
    """Parse a single source file and return extracted nodes."""
    parser = get_parser(language)
    source = file_path.read_bytes()
    tree = parser.parse(source)
    if tree is None or tree.root_node is None:
        raise RuntimeError(f"tree-sitter returned no tree for {file_path}")

    rel_path = str(file_path.relative_to(repo_path)).replace("\\", "/")
    results: list[dict[str, Any]] = []
    _walk_tree(tree.root_node, source, rel_path, language, results, [])
    return results


def parse_repository(repo_path: str) -> list[dict]:
    """Walk ``repo_path`` and extract AST entities from supported source files.

    For each file matching a known extension, the appropriate tree-sitter
    grammar is used to extract function/method definitions, class definitions,
    import statements, and call sites.

    Args:
        repo_path: Absolute or relative path to the repository root.

    Returns:
        A list of dicts, each with keys:
        ``node_type``, ``name``, ``file_path``, ``start_line``, ``end_line``,
        ``parent``.

    Notes:
        Files that fail to parse are skipped with a warning; remaining files
        continue to be processed. Summary counters are stored on
        ``parse_repository.last_stats`` for callers (and the ``__main__`` block).
    """
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {root}")

    all_nodes: list[dict[str, Any]] = []
    files_parsed = 0
    files_skipped = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue

        language = EXTENSION_LANGUAGE_MAP.get(path.suffix.lower())
        if language is None:
            continue

        try:
            nodes = _parse_file(path, root, language)
            all_nodes.extend(nodes)
            files_parsed += 1
        except Exception as exc:  # noqa: BLE001 — skip and continue per file
            files_skipped += 1
            logger.warning("Skipping %s (%s): %s", path, language, exc)

    parse_repository.last_stats = {  # type: ignore[attr-defined]
        "files_parsed": files_parsed,
        "files_skipped": files_skipped,
        "total_functions": sum(1 for n in all_nodes if n["node_type"] == "function"),
        "total_classes": sum(1 for n in all_nodes if n["node_type"] == "class"),
        "total_imports": sum(1 for n in all_nodes if n["node_type"] == "import"),
        "total_calls": sum(1 for n in all_nodes if n["node_type"] == "call"),
        "total_nodes": len(all_nodes),
    }
    return all_nodes


def _print_summary(stats: dict[str, int]) -> None:
    """Print human-readable parse summary stats."""
    print("AST parse summary")
    print(f"  files parsed : {stats['files_parsed']}")
    print(f"  files skipped: {stats['files_skipped']}")
    print(f"  functions    : {stats['total_functions']}")
    print(f"  classes      : {stats['total_classes']}")
    print(f"  imports      : {stats['total_imports']}")
    print(f"  calls        : {stats['total_calls']}")
    print(f"  total nodes  : {stats['total_nodes']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    default_repo = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "mini_repo"
    target = sys.argv[1] if len(sys.argv) > 1 else str(default_repo)

    print(f"Parsing repository: {target}")
    nodes = parse_repository(target)
    stats = getattr(parse_repository, "last_stats", {})
    _print_summary(stats)

    # Brief sample of extracted entities for smoke-checking.
    for sample in nodes[:10]:
        print(
            f"  [{sample['node_type']}] {sample['name']} "
            f"@ {sample['file_path']}:{sample['start_line']}-{sample['end_line']} "
            f"(parent={sample['parent']})"
        )
    if len(nodes) > 10:
        print(f"  ... and {len(nodes) - 10} more")
