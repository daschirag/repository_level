#!/usr/bin/env python3
"""Generate patent/thesis figures from live Neo4j (+ Qdrant) pipeline data.

Outputs PNGs at 300 DPI under ``docs/figures/`` and prints a numeric summary
table for Section 8 cross-checks.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from neo4j import GraphDatabase

# Ensure project root is on sys.path when invoked as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.graph.neo4j_client import (  # noqa: E402
    _BUILTIN_DENYLIST,
    _build_resolution_indexes,
    _resolve_callee,
    _split_callee,
    _FuncRef,
)
from src.ingestion.ast_parser import parse_repository  # noqa: E402
from src.ingestion.git_crawler import get_change_frequency  # noqa: E402
from src.rag.qdrant_client import CodeRAGClient  # noqa: E402
from src.scoring.div_propagation import (  # noqa: E402
    compute_div_scores,
    get_top_debt_nodes,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("generate_result_graphs")

NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "techdebt123"
FLASK_REPO = _PROJECT_ROOT / "tests" / "fixtures" / "flask-test-repo"
FIGURES_DIR = _PROJECT_ROOT / "docs" / "figures"

DPI = 300


def _configure_matplotlib() -> None:
    """Print-ready style: serif font + whitegrid when available."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.style.use("ggplot")

    preferred = [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "Liberation Serif",
        "DejaVu Serif",
        "serif",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name == "serif" or name in available:
            plt.rcParams["font.family"] = name if name != "serif" else "serif"
            break
    plt.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": DPI,
            "axes.edgecolor": "#333333",
            "grid.alpha": 0.35,
        }
    )


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Wrote %s", path)
    return path


def fetch_function_metrics(driver) -> list[dict[str, Any]]:
    """Pull all :Function metric properties from Neo4j."""
    with driver.session() as session:
        rows = session.run(
            """
            MATCH (f:Function)
            RETURN f.file_path AS file_path,
                   f.name AS name,
                   coalesce(f.div_score, 0.0) AS div_score,
                   coalesce(f.cyclomatic_complexity, 0) AS cyclomatic_complexity,
                   coalesce(f.change_frequency, 0) AS change_frequency
            """
        )
        return [dict(r) for r in rows]


def fetch_edge_counts(driver) -> dict[str, int]:
    """Count CALLS / IMPORTS relationships currently in Neo4j."""
    with driver.session() as session:
        calls = session.run("MATCH ()-[r:CALLS]->() RETURN count(r) AS c").single()["c"]
        imports = session.run(
            "MATCH ()-[r:IMPORTS]->() RETURN count(r) AS c"
        ).single()["c"]
    return {"CALLS": int(calls), "IMPORTS": int(imports)}


def compute_call_resolution_stats(ast_nodes: list[dict]) -> dict[str, int]:
    """Dry-run scoped call resolution (same logic as write_call_edges)."""
    (
        _funcs_by_file,
        classes_by_file,
        funcs_by_file_name,
        import_name_to_files,
    ) = _build_resolution_indexes(ast_nodes)

    all_funcs_by_name: dict[str, list[_FuncRef]] = defaultdict(list)
    for file_map in funcs_by_file_name.values():
        for name, refs in file_map.items():
            all_funcs_by_name[name].extend(refs)

    call_sites = 0
    unresolved = 0
    denied = 0
    resolved_rows: list[tuple[str, str, str, str]] = []
    legacy_edge_pairs: set[tuple[str, str, str, str]] = set()

    for node in ast_nodes:
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
        resolved_rows.append(
            (file_path, str(parent), resolved.file_path, resolved.name)
        )

    unique_resolved = len(set(resolved_rows))
    return {
        "call_sites": call_sites,
        "resolved_edges": unique_resolved,
        "unresolved": unresolved,
        "denied_builtins": denied,
        "legacy_global_edges": len(legacy_edge_pairs),
        "neo4j_calls_edges": 0,  # filled by caller
    }


def measure_stage_timings(
    driver,
    ast_nodes: list[dict],
) -> dict[str, float]:
    """Measure wall-clock seconds for each pipeline stage (live)."""
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    _ = parse_repository(str(FLASK_REPO))
    timings["1. Ingestion — AST parse"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = get_change_frequency(str(FLASK_REPO))
    timings["2. Ingestion — git history"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = fetch_function_metrics(driver)
    _ = fetch_edge_counts(driver)
    timings["3. Graph materialization (Neo4j read)"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = compute_call_resolution_stats(ast_nodes)
    timings["4. Call resolution (scoped dry-run)"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = compute_div_scores(driver)
    timings["5. DIV propagation"] = time.perf_counter() - t0

    # RAG: time a real Qdrant query path (collection must already be indexed).
    t0 = time.perf_counter()
    try:
        rag = CodeRAGClient()
        rag._client.get_collections()
        _ = rag.query_context("handle HTTP request routing", top_k=5)
        timings["6. RAG query (Qdrant + CodeBERT)"] = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG timing skipped (%s)", exc)
        timings["6. RAG query (Qdrant + CodeBERT)"] = float("nan")

    return timings


def fig_div_distribution(rows: list[dict[str, Any]], top10: list[dict]) -> Path:
    scores = np.array([float(r["div_score"] or 0.0) for r in rows], dtype=float)
    cutoff = float(top10[-1]["div_score"]) if top10 else 0.0

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    # Grayscale-friendly histogram with dark outline.
    ax.hist(
        scores,
        bins=40,
        color="#4a4a4a",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
    )
    ax.set_yscale("log")
    ax.axvline(
        cutoff,
        color="#b22222",
        linestyle="--",
        linewidth=1.6,
        label=f"Top-10 cutoff (DIV = {cutoff:.2f})",
    )
    ax.set_xlabel("DIV score")
    ax.set_ylabel("Number of functions (log scale)")
    ax.set_title("Distribution of Debt Impact Vector (DIV) Scores Across Flask Repository")
    ax.legend(loc="upper right", frameon=True)
    return _save(fig, "div_distribution.png")


def fig_top10(top10: list[dict]) -> Path:
    labels = [
        f"{r['name']} ({Path(str(r['file_path'])).name})" for r in reversed(top10)
    ]
    values = [float(r["div_score"]) for r in reversed(top10)]
    # High DIV = red, low among top-10 = olive/green (still B&W-distinct via lightness).
    norm = Normalize(vmin=min(values), vmax=max(values))
    cmap = plt.get_cmap("RdYlGn_r")
    colors = [cmap(norm(v)) for v in values]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.barh(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("DIV score")
    ax.set_ylabel("Function (file)")
    ax.set_title("Top 10 Debt Nodes by Causal DIV Score")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("DIV (high → red)")
    return _save(fig, "top10_debt_nodes.png")


def fig_call_resolution(stats: dict[str, int]) -> Path:
    """Grouped bars: scoped outcomes vs legacy unscoped edge count."""
    categories = [
        "Resolved\n(scoped)",
        "Unresolved\n(skipped)",
        "Denied\n(builtins)",
        "Legacy unscoped\n(would-create)",
    ]
    values = [
        stats["resolved_edges"],
        stats["unresolved"],
        stats["denied_builtins"],
        stats["legacy_global_edges"],
    ]
    # Distinct grayscale + one accent for legacy comparison.
    colors = ["#2f2f2f", "#7a7a7a", "#b0b0b0", "#8b1e1e"]

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Count")
    ax.set_title(
        f"Call Site Resolution Outcomes "
        f"(Flask Repository, {stats['call_sites']:,} call sites)"
    )
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    if stats["legacy_global_edges"] > 0:
        reduction = 100.0 * (
            1.0 - stats["resolved_edges"] / stats["legacy_global_edges"]
        )
        ax.annotate(
            f"Scoped edges vs legacy: "
            f"{stats['resolved_edges']:,} / {stats['legacy_global_edges']:,} "
            f"({reduction:.1f}% fewer edges)",
            xy=(0.5, 0.95),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#444"),
        )
    return _save(fig, "call_resolution_breakdown.png")


def fig_div_vs_complexity(rows: list[dict[str, Any]]) -> Path:
    xs = np.array([float(r["cyclomatic_complexity"] or 0) for r in rows], dtype=float)
    ys = np.array([float(r["div_score"] or 0) for r in rows], dtype=float)
    cf = np.array([float(r["change_frequency"] or 0) for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sc = ax.scatter(
        xs,
        ys,
        c=cf,
        cmap="Greys",
        s=18,
        alpha=0.75,
        edgecolors="black",
        linewidths=0.25,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("change_frequency")
    ax.set_xlabel("Cyclomatic complexity")
    ax.set_ylabel("DIV score")
    ax.set_title("DIV Score vs Cyclomatic Complexity (Flask Functions)")

    # Highlight that high-DIV is not just high-CC: annotate a low-CC high-DIV point.
    # Prefer points with cc <= 3 and high DIV.
    candidates = [
        (float(r["cyclomatic_complexity"] or 0), float(r["div_score"] or 0), r)
        for r in rows
        if float(r["cyclomatic_complexity"] or 0) <= 3
        and float(r["div_score"] or 0) > 0
    ]
    if candidates:
        candidates.sort(key=lambda t: t[1], reverse=True)
        cc, div, r = candidates[0]
        ax.annotate(
            f"{r['name']} (cc={int(cc)})",
            xy=(cc, div),
            xytext=(cc + max(xs.max() * 0.08, 1.5), div),
            fontsize=7,
            arrowprops=dict(arrowstyle="->", color="#222", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#555"),
        )
    return _save(fig, "div_vs_complexity_scatter.png")


def fig_pipeline_timing(timings: dict[str, float]) -> Path:
    labels = list(timings.keys())
    values = [timings[k] for k in labels]
    # Replace NaN with 0 for plotting but annotate as N/A.
    plot_vals = [0.0 if (v != v) else v for v in values]  # NaN != NaN
    colors = ["#1a1a1a", "#3a3a3a", "#5a5a5a", "#7a7a7a", "#9a9a9a", "#b0b0b0"]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    y = np.arange(len(labels))
    bars = ax.barh(
        y,
        plot_vals,
        color=colors[: len(labels)],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Wall-clock time (seconds)")
    ax.set_title("Pipeline Stage Wall-Clock Timing (Flask Fixture, Live Measurement)")
    for bar, val in zip(bars, values):
        label = "N/A" if val != val else f"{val:.2f}s"
        ax.text(
            (0 if val != val else val) + max(plot_vals) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=8,
        )
    return _save(fig, "pipeline_stage_timing.png")


def print_summary(
    rows: list[dict[str, Any]],
    top10: list[dict],
    stats: dict[str, int],
    edge_counts: dict[str, int],
    timings: dict[str, float],
) -> None:
    scores = [float(r["div_score"] or 0) for r in rows]
    print("\n" + "=" * 88)
    print("SECTION 8 — FIGURE NUMERIC SUMMARY (live Neo4j / measured)")
    print("=" * 88)

    print("\n[1] div_distribution.png")
    print(f"  functions                 : {len(rows)}")
    print(f"  DIV min / median / max    : {min(scores):.4f} / {np.median(scores):.4f} / {max(scores):.4f}")
    print(f"  DIV mean / std            : {np.mean(scores):.4f} / {np.std(scores):.4f}")
    print(f"  top-10 cutoff (10th DIV)  : {float(top10[-1]['div_score']):.4f}")
    print(f"  functions with DIV > 0    : {sum(1 for s in scores if s > 0)}")

    print("\n[2] top10_debt_nodes.png")
    for i, r in enumerate(top10, 1):
        print(
            f"  {i:2d}. DIV={float(r['div_score']):8.4f}  "
            f"cc={int(r['cyclomatic_complexity']):3d}  "
            f"chg={int(r['change_frequency']):4d}  "
            f"{r['name']} @ {r['file_path']}"
        )

    print("\n[3] call_resolution_breakdown.png")
    print(f"  call sites                : {stats['call_sites']}")
    print(f"  resolved (scoped unique)  : {stats['resolved_edges']}")
    print(f"  unresolved                : {stats['unresolved']}")
    print(f"  denied builtins           : {stats['denied_builtins']}")
    print(f"  legacy unscoped edges     : {stats['legacy_global_edges']}")
    print(f"  Neo4j CALLS edges (live)  : {edge_counts['CALLS']}")
    print(f"  Neo4j IMPORTS edges       : {edge_counts['IMPORTS']}")
    if stats["legacy_global_edges"]:
        reduction = 100.0 * (
            1.0 - stats["resolved_edges"] / stats["legacy_global_edges"]
        )
        print(f"  edge reduction vs legacy  : {reduction:.2f}%")

    print("\n[4] div_vs_complexity_scatter.png")
    ccs = [float(r["cyclomatic_complexity"] or 0) for r in rows]
    if len(scores) > 1 and np.std(ccs) > 0 and np.std(scores) > 0:
        corr = float(np.corrcoef(ccs, scores)[0, 1])
    else:
        corr = float("nan")
    print(f"  Pearson corr(cc, DIV)     : {corr:.4f}")
    low_cc_high = sorted(
        (
            r
            for r in rows
            if float(r["cyclomatic_complexity"] or 0) <= 3
            and float(r["div_score"] or 0) > 0
        ),
        key=lambda r: float(r["div_score"]),
        reverse=True,
    )[:5]
    print("  top low-cc (<=3) high-DIV exemplars:")
    for r in low_cc_high:
        print(
            f"    DIV={float(r['div_score']):8.4f}  cc={int(r['cyclomatic_complexity'])}  "
            f"{r['name']} @ {r['file_path']}"
        )

    print("\n[5] pipeline_stage_timing.png")
    for stage, secs in timings.items():
        label = "N/A" if secs != secs else f"{secs:.3f}s"
        print(f"  {stage:42s} {label}")
    print("=" * 88)


def main() -> int:
    _configure_matplotlib()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not FLASK_REPO.is_dir():
        logger.error("Flask fixture not found: %s", FLASK_REPO)
        return 1

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()

        logger.info("Fetching Function metrics from Neo4j…")
        rows = fetch_function_metrics(driver)
        if not rows:
            logger.error("No :Function nodes found — run graph ingest + DIV first")
            return 1

        edge_counts = fetch_edge_counts(driver)
        top10 = get_top_debt_nodes(driver, k=10)

        logger.info("Parsing Flask AST for call-resolution dry-run…")
        # Ensure tree-sitter compatible import path works under uv/venv.
        ast_nodes = parse_repository(str(FLASK_REPO))
        stats = compute_call_resolution_stats(ast_nodes)
        stats["neo4j_calls_edges"] = edge_counts["CALLS"]

        logger.info("Measuring pipeline stage timings (live)…")
        timings = measure_stage_timings(driver, ast_nodes)

        logger.info("Rendering figures…")
        paths = [
            fig_div_distribution(rows, top10),
            fig_top10(top10),
            fig_call_resolution(stats),
            fig_div_vs_complexity(rows),
            fig_pipeline_timing(timings),
        ]

        print_summary(rows, top10, stats, edge_counts, timings)
        print("\nGenerated figures:")
        for p in paths:
            print(f"  - {p}")
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
