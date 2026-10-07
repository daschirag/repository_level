"""Deterministic, rule-based remediation suggestions.

Used alongside (and as the fallback for) the LLM refactor agent. Every
suggestion cites the metric and threshold that triggered it so it can be
shown verbatim in the UI, labelled "rule-based".

Inputs are graph metrics only -- there is no test-coverage data in this
pipeline, so no rule claims anything about coverage.
"""

from __future__ import annotations

from typing import Any, TypedDict

from neo4j import Driver

# Thresholds (conventional guidelines, not tuned per repo).
COMPLEXITY_SPLIT = 10  # McCabe: >10 is "complex, high risk"
LOC_EXTRACT = 50  # function longer than ~one screen
FAN_IN_CONTRACT = 5  # transitive callers that break if behaviour changes
FAN_OUT_INVERT = 5  # direct callees -> tight coupling
LOW_COMPLEXITY = 3  # "simple, but structurally critical"
CHURN_PERCENTILE = 0.75  # file churn in the repo's top quartile


class Suggestion(TypedDict):
    rule: str
    title: str
    detail: str


class NodeMetrics(TypedDict):
    cyclomatic_complexity: int
    loc: int
    change_frequency: int
    direct_callers: int
    transitive_callers: int
    affected_files: int
    direct_callees: int
    churn_threshold: int


def fetch_node_metrics(driver: Driver, file_path: str, name: str) -> NodeMetrics | None:
    """Collect the graph metrics the rules need for one ``:Function``."""
    with driver.session() as session:
        record = session.run(
            """
            MATCH (f:Function {file_path: $file_path, name: $name})
            WITH f LIMIT 1
            OPTIONAL MATCH (dc:Function)-[:CALLS]->(f)
            WITH f, count(DISTINCT dc) AS direct_callers
            OPTIONAL MATCH (f)-[:CALLS]->(ce:Function)
            WITH f, direct_callers, count(DISTINCT ce) AS direct_callees
            OPTIONAL MATCH (tc:Function)-[:CALLS*1..]->(f)
            WHERE tc <> f
            RETURN coalesce(f.cyclomatic_complexity, 0) AS cyclomatic_complexity,
                   coalesce(f.end_line - f.start_line + 1, 0) AS loc,
                   coalesce(f.change_frequency, 0) AS change_frequency,
                   direct_callers,
                   direct_callees,
                   count(DISTINCT tc) AS transitive_callers,
                   count(DISTINCT tc.file_path) AS affected_files
            """,
            {"file_path": file_path, "name": name},
        ).single()
        if record is None:
            return None
        churn = session.run(
            """
            MATCH (f:Function)
            RETURN percentileDisc(coalesce(f.change_frequency, 0), $p) AS threshold
            """,
            {"p": CHURN_PERCENTILE},
        ).single()
    metrics: dict[str, Any] = dict(record)
    metrics["churn_threshold"] = int(churn["threshold"] or 0) if churn else 0
    return metrics  # type: ignore[return-value]


def suggest(metrics: NodeMetrics) -> list[Suggestion]:
    """Return rule-based suggestions, most specific first (never empty)."""
    cc = int(metrics["cyclomatic_complexity"])
    loc = int(metrics["loc"])
    churn = int(metrics["change_frequency"])
    fan_in = int(metrics["transitive_callers"])
    fan_out = int(metrics["direct_callees"])
    files = int(metrics["affected_files"])
    out: list[Suggestion] = []

    if cc > COMPLEXITY_SPLIT:
        out.append(
            {
                "rule": "split_function",
                "title": "Split the function",
                "detail": f"Cyclomatic complexity {cc} > threshold {COMPLEXITY_SPLIT}: "
                "extract each major branch into a named helper and keep a thin orchestrator.",
            }
        )
    if loc > LOC_EXTRACT:
        out.append(
            {
                "rule": "extract_blocks",
                "title": "Extract cohesive blocks",
                "detail": f"{loc} lines > {LOC_EXTRACT}-line guideline: move self-contained "
                "steps into pure helpers that can be tested on their own.",
            }
        )
    if fan_in >= FAN_IN_CONTRACT:
        out.append(
            {
                "rule": "contract_tests",
                "title": "Pin behaviour with contract tests first",
                "detail": f"{fan_in} functions in {files} file{'s' if files != 1 else ''} transitively call this one "
                f"(threshold {FAN_IN_CONTRACT}): add characterisation tests for its "
                "public behaviour before changing it.",
            }
        )
    if fan_out >= FAN_OUT_INVERT:
        out.append(
            {
                "rule": "invert_dependencies",
                "title": "Reduce fan-out",
                "detail": f"Calls {fan_out} other functions (threshold {FAN_OUT_INVERT}): "
                "group collaborators behind a narrow interface and inject them.",
            }
        )
    if metrics["churn_threshold"] > 0 and churn >= metrics["churn_threshold"]:
        out.append(
            {
                "rule": "stabilise_hotspot",
                "title": "Stabilise a change hotspot",
                "detail": f"Its file changed in {churn} commits (top {round((1 - CHURN_PERCENTILE) * 100)}% "
                f"of this repo, >= {metrics['churn_threshold']}): prefer small, test-backed "
                "refactors and isolate the parts that keep changing.",
            }
        )
    if cc <= LOW_COMPLEXITY and fan_in >= FAN_IN_CONTRACT:
        out.append(
            {
                "rule": "inherited_risk",
                "title": "Risk comes from position, not code",
                "detail": f"Complexity is only {cc}, but {fan_in} callers depend on it: "
                "rewriting it gains little -- invest in tests and a stable signature instead.",
            }
        )
    if not out:
        out.append(
            {
                "rule": "monitor",
                "title": "No rule threshold exceeded",
                "detail": f"Complexity {cc}, {loc} lines, {fan_in} transitive callers, "
                f"file churn {churn}: monitor rather than refactor now.",
            }
        )
    return out
