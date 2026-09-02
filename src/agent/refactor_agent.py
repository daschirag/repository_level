"""Layer 5: LangGraph ReAct-style refactor proposal agent.

Uses DIV-ranked debt nodes, Neo4j/Qdrant context tools, and StarCoder2-7B
(via llama-cpp-python) to propose concrete refactors. When the GGUF weights
are missing, a labeled ``MockLLMClient`` keeps the pipeline testable.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal, Optional, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent.tools import (
    REPO_ROOT,
    fetch_function_context,
    fetch_function_source,
    fetch_function_span,
    fetch_top_debt_nodes,
    get_function_context_tool,
    get_function_source_tool,
    get_top_debt_nodes_tool,
)

logger = logging.getLogger(__name__)

# StarCoder2-7B GGUF (Q4_K_M). Anchored to project root via __file__ so the
# path is cwd-independent. Download manually if missing — multi-GB artifact:
#   https://huggingface.co/bigcode/starcoder2-7b
# or a community GGUF mirror (e.g. second-state / TheBloke-style quantizations).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = _PROJECT_ROOT / "models" / "starcoder2"
MODEL_PATH = MODEL_DIR / "starcoder2-7b-Q4_K_M.gguf"

RefactorType = Literal[
    "function_split",
    "dependency_inversion",
    "pure_function_extraction",
]


class LLMClient(Protocol):
    """Shared interface for StarCoder2 and the mock fallback."""

    def generate(self, prompt: str) -> str:
        """Generate a completion for ``prompt``."""


class MockLLMClient:
    """Deterministic placeholder LLM used when no GGUF weights are present.

    Responses are clearly labeled ``[MOCK LLM — StarCoder2 GGUF not found]``
    so downstream evaluation never confuses stubs with real model output.
    """

    def generate(self, prompt: str) -> str:
        """Return a structured mock refactor proposal derived from the prompt."""
        name_match = re.search(r"Function:\s*(\S+)", prompt)
        file_match = re.search(r"File:\s*(\S+)", prompt)
        cc_match = re.search(r"Cyclomatic complexity:\s*(\d+)", prompt)
        name = name_match.group(1) if name_match else "unknown_function"
        file_path = file_match.group(1) if file_match else "unknown_file"
        cc = int(cc_match.group(1)) if cc_match else 0

        # Count real neighborhood bullets under the callers/callees sections
        # (template headings alone must not trigger dependency_inversion).
        callers_section = re.search(
            r"Direct callers:\n(.*?)(?:\n\nDirect callees:)",
            prompt,
            flags=re.DOTALL,
        )
        callees_section = re.search(
            r"Direct callees:\n(.*?)(?:\n\nSemantically)",
            prompt,
            flags=re.DOTALL,
        )
        caller_bullets = 0
        callee_bullets = 0
        if callers_section and "(none)" not in callers_section.group(1):
            caller_bullets = callers_section.group(1).count("\n- ")
        if callees_section and "(none)" not in callees_section.group(1):
            callee_bullets = callees_section.group(1).count("\n- ")

        if cc >= 10:
            refactor_type: RefactorType = "function_split"
            rationale = (
                f"{name} has high cyclomatic complexity ({cc}); splitting into "
                "smaller helpers should isolate branching paths and shrink the "
                "DIV contribution of the parent."
            )
            proposal = (
                f"Split `{name}` in `{file_path}` into (1) a thin orchestrator "
                "that retains the public signature and (2) focused helpers for "
                "each major branch / responsibility. Keep shared state explicit "
                "via parameters rather than nested closures."
            )
        elif caller_bullets + callee_bullets >= 2:
            refactor_type = "dependency_inversion"
            rationale = (
                f"`{name}` sits on a hot call path ({caller_bullets} callers / "
                f"{callee_bullets} callees); inverting dependencies behind a "
                "narrow interface reduces coupling that amplifies DIV."
            )
            proposal = (
                f"Introduce an interface/protocol for the collaborators of "
                f"`{name}` and inject them at the call boundary so higher DIV "
                "callers no longer depend on concrete implementations."
            )
        else:
            refactor_type = "pure_function_extraction"
            rationale = (
                f"Extracting a pure core from `{name}` separates side-effecting "
                "I/O from decision logic, making the debt hotspot easier to test "
                "and safer to change."
            )
            proposal = (
                f"Extract the pure decision/transform logic inside `{name}` "
                f"(`{file_path}`) into a side-effect-free helper; leave I/O, "
                "logging, and framework calls in a thin wrapper."
            )

        return (
            "[MOCK LLM — StarCoder2 GGUF not found]\n"
            f"REFACTOR_TYPE: {refactor_type}\n"
            f"RATIONALE: {rationale}\n"
            f"PROPOSAL:\n{proposal}\n"
        )


class StarCoder2Client:
    """llama-cpp-python wrapper around a local StarCoder2-7B GGUF model.

    Loads ``MODEL_PATH`` (``models/starcoder2/starcoder2-7b-Q4_K_M.gguf``,
    resolved from project root via ``__file__``). CPU-only build: always sets
    ``n_gpu_layers=0``.
    """

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
    ) -> None:
        """Load StarCoder2 via llama-cpp (CPU).

        Args:
            model_path: Explicit ``.gguf`` path (defaults to ``MODEL_PATH``).
            n_ctx: Context window (4096 matches the Q4_K_M GGUF card).
            n_threads: CPU threads; defaults to ``min(8, os.cpu_count() or 4)``.
        """
        from llama_cpp import Llama

        resolved = _resolve_gguf_path(model_path or MODEL_PATH)
        if resolved is None:
            raise FileNotFoundError(
                f"No StarCoder2 GGUF found at {model_path or MODEL_PATH}"
            )

        if n_threads is None:
            n_threads = min(8, os.cpu_count() or 4)

        logger.info(
            "Loading StarCoder2 GGUF from %s (CPU llama-cpp, n_gpu_layers=0, "
            "n_ctx=%d, n_threads=%d)",
            resolved,
            n_ctx,
            n_threads,
        )
        self._llm = Llama(
            model_path=str(resolved),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,
            verbose=False,
        )
        self.model_path = resolved

    def generate(self, prompt: str) -> str:
        """Generate a refactor completion with StarCoder2."""
        result = self._llm(
            prompt,
            max_tokens=512,
            temperature=0.2,
            stop=[
                "</proposal>",
                "\n\n\n",
                "\n## Example",
                "\nYou are a senior engineer",
            ],
        )
        # llama-cpp returns an OpenAI-like dict.
        choices = result.get("choices") or []
        if not choices:
            return ""
        return str(choices[0].get("text") or "").strip()


def _resolve_gguf_path(path: Path | str) -> Optional[Path]:
    """Return a usable ``.gguf`` file path, or ``None`` if missing."""
    candidate = Path(path)
    if candidate.is_file() and candidate.suffix.lower() == ".gguf":
        return candidate
    if candidate.is_dir():
        ggufs = sorted(candidate.glob("*.gguf"))
        return ggufs[0] if ggufs else None
    # Also search MODEL_DIR when given a bare filename.
    alt = MODEL_DIR / candidate.name
    if alt.is_file():
        return alt
    return None


def get_llm_client() -> LLMClient:
    """Return StarCoder2 if a GGUF exists, otherwise the labeled mock client."""
    gguf = _resolve_gguf_path(MODEL_PATH)
    if gguf is not None:
        try:
            return StarCoder2Client(gguf)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load StarCoder2 from %s (%s); using MockLLMClient",
                gguf,
                exc,
            )
            return MockLLMClient()

    logger.warning(
        "StarCoder2 GGUF not found under %s — using MockLLMClient. "
        "Download a quantized .gguf manually into that directory to enable "
        "real proposals.",
        MODEL_DIR,
    )
    return MockLLMClient()


class AgentState(TypedDict, total=False):
    """LangGraph state for one debt-node refactor proposal."""

    file_path: str
    name: str
    div_score: float
    cyclomatic_complexity: int
    change_frequency: int
    start_line: int
    end_line: int
    context: dict[str, Any]
    source: str
    prompt: str
    raw_llm_output: str
    refactor_type: str
    proposal_text: str
    rationale: str
    error: str


def _build_prompt(state: AgentState) -> str:
    """Construct the StarCoder2 / mock refactor prompt."""
    context = state.get("context") or {}
    callers = context.get("callers") or []
    callees = context.get("callees") or []
    neighbors = context.get("semantic_neighbors") or []

    def _fmt_nodes(rows: list[dict[str, Any]], limit: int = 8) -> str:
        if not rows:
            return "(none)"
        lines = []
        for row in rows[:limit]:
            lines.append(
                f"- {row.get('name')} @ {row.get('file_path')} "
                f"(div={row.get('div_score')})"
            )
        return "\n".join(lines)

    neighbor_lines = []
    for row in neighbors[:5]:
        neighbor_lines.append(
            f"- {row.get('name')} @ {row.get('file_path')} "
            f"(score={row.get('score')})"
        )
    neighbors_block = "\n".join(neighbor_lines) if neighbor_lines else "(none)"

    source = state.get("source") or "(source unavailable)"
    return f"""You are a senior engineer proposing a targeted refactor for a high technical-debt function.

File: {state.get('file_path')}
Function: {state.get('name')}
DIV score: {state.get('div_score')}
Cyclomatic complexity: {state.get('cyclomatic_complexity')}
Change frequency: {state.get('change_frequency')}

Direct callers:
{_fmt_nodes(callers)}

Direct callees:
{_fmt_nodes(callees)}

Semantically similar documented code:
{neighbors_block}

Source:
```
{source}
```

Choose ONE refactor type from:
- function_split
- dependency_inversion
- pure_function_extraction

Respond in EXACTLY this format:
REFACTOR_TYPE: <one of the three types>
RATIONALE: <2-4 sentences>
PROPOSAL:
<specific, actionable refactor description referencing this function>
"""


def _parse_llm_output(text: str) -> dict[str, str]:
    """Parse REFACTOR_TYPE / RATIONALE / PROPOSAL blocks from model output."""
    refactor_type = "pure_function_extraction"
    rationale = ""
    proposal = text.strip()

    type_match = re.search(
        r"REFACTOR_TYPE:\s*(function_split|dependency_inversion|pure_function_extraction)",
        text,
        flags=re.IGNORECASE,
    )
    if type_match:
        refactor_type = type_match.group(1).lower()

    rationale_match = re.search(
        r"RATIONALE:\s*(.+?)(?:\nPROPOSAL:|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if rationale_match:
        rationale = rationale_match.group(1).strip()

    proposal_match = re.search(
        r"PROPOSAL:\s*(.+)\Z",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if proposal_match:
        proposal = proposal_match.group(1).strip()

    return {
        "refactor_type": refactor_type,
        "rationale": rationale or "No rationale parsed from model output.",
        "proposal_text": proposal or text.strip(),
    }


def _node_load_span(state: AgentState) -> AgentState:
    """Fill start/end lines and metrics from Neo4j when missing."""
    span = fetch_function_span(state["file_path"], state["name"])
    updates: AgentState = {}
    if span.get("start_line") is not None:
        updates["start_line"] = int(span["start_line"])
    if span.get("end_line") is not None:
        updates["end_line"] = int(span["end_line"])
    if span.get("div_score") is not None and state.get("div_score") is None:
        updates["div_score"] = float(span["div_score"])
    if span.get("cyclomatic_complexity") is not None:
        updates["cyclomatic_complexity"] = int(span["cyclomatic_complexity"])
    if span.get("change_frequency") is not None:
        updates["change_frequency"] = int(span["change_frequency"])
    return updates


def _node_gather_context(state: AgentState) -> AgentState:
    """Tool step: architectural + semantic context."""
    context = fetch_function_context(state["file_path"], state["name"])
    return {"context": context}


def _node_gather_source(state: AgentState) -> AgentState:
    """Tool step: read source lines from disk."""
    start = int(state.get("start_line") or 0)
    end = int(state.get("end_line") or 0)
    if start <= 0 or end <= 0:
        return {"source": "", "error": "Missing start_line/end_line for source read"}
    payload = fetch_function_source(state["file_path"], state["name"], start, end)
    if payload.get("error"):
        return {"source": "", "error": str(payload["error"])}
    return {"source": str(payload.get("source") or "")}


def _make_generate_node(llm: LLMClient):
    """Build the LLM generation node closed over ``llm``."""

    def _node_generate(state: AgentState) -> AgentState:
        prompt = _build_prompt(state)
        raw = llm.generate(prompt)
        parsed = _parse_llm_output(raw)
        return {
            "prompt": prompt,
            "raw_llm_output": raw,
            "refactor_type": parsed["refactor_type"],
            "proposal_text": parsed["proposal_text"],
            "rationale": parsed["rationale"],
        }

    return _node_generate


def build_refactor_graph(llm: Optional[LLMClient] = None) -> Any:
    """Compile the LangGraph StateGraph for one-node refactor proposals.

    Flow (simple ReAct-style tool-then-reason loop)::

        load_span -> gather_context -> gather_source -> generate -> END
    """
    client = llm or get_llm_client()
    graph: StateGraph = StateGraph(AgentState)
    graph.add_node("load_span", _node_load_span)
    graph.add_node("gather_context", _node_gather_context)
    graph.add_node("gather_source", _node_gather_source)
    graph.add_node("generate", _make_generate_node(client))

    graph.add_edge(START, "load_span")
    graph.add_edge("load_span", "gather_context")
    graph.add_edge("gather_context", "gather_source")
    graph.add_edge("gather_source", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def generate_refactor_proposals(
    top_k: int = 5,
    llm: Optional[LLMClient] = None,
) -> list[dict]:
    """Generate refactor proposals for the top-``k`` DIV debt nodes.

    Args:
        top_k: How many highest-DIV functions to propose refactors for.
        llm: Optional shared LLM client (avoids reloading a multi-GB GGUF).

    Returns:
        List of dicts with keys:
        ``file_path``, ``name``, ``div_score``, ``refactor_type``,
        ``proposal_text``, ``rationale``.
    """
    client = llm or get_llm_client()
    app = build_refactor_graph(client)
    debt_nodes = fetch_top_debt_nodes(k=top_k)
    proposals: list[dict[str, Any]] = []

    for node in debt_nodes:
        initial: AgentState = {
            "file_path": str(node.get("file_path") or ""),
            "name": str(node.get("name") or ""),
            "div_score": float(node.get("div_score") or 0.0),
            "cyclomatic_complexity": int(node.get("cyclomatic_complexity") or 0),
            "change_frequency": int(node.get("change_frequency") or 0),
        }
        logger.info(
            "Generating refactor proposal for %s::%s (DIV=%.2f)",
            initial["file_path"],
            initial["name"],
            initial["div_score"],
        )
        final_state = app.invoke(initial)
        proposals.append(
            {
                "file_path": final_state.get("file_path"),
                "name": final_state.get("name"),
                "div_score": final_state.get("div_score"),
                "refactor_type": final_state.get("refactor_type"),
                "proposal_text": final_state.get("proposal_text"),
                "rationale": final_state.get("rationale"),
                "raw_llm_output": final_state.get("raw_llm_output"),
            }
        )

    return proposals


def _print_proposals(proposals: list[dict]) -> None:
    """Pretty-print refactor proposals."""
    if not proposals:
        print("No proposals generated.")
        return
    for i, proposal in enumerate(proposals, start=1):
        print("=" * 88)
        print(
            f"[{i}] {proposal.get('name')}  @  {proposal.get('file_path')}  "
            f"(DIV={proposal.get('div_score')})"
        )
        print(f"Type     : {proposal.get('refactor_type')}")
        print(f"Rationale: {proposal.get('rationale')}")
        print("Proposal :")
        print(proposal.get("proposal_text"))
    print("=" * 88)


# Re-export tools for LangChain agent bindings / discovery.
TOOLS = [
    get_top_debt_nodes_tool,
    get_function_context_tool,
    get_function_source_tool,
]


if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    gguf = _resolve_gguf_path(MODEL_PATH)
    print(f"REPO_ROOT={REPO_ROOT}")
    print(f"MODEL_PATH={MODEL_PATH}")
    print(f"GGUF exists={gguf is not None} ({gguf})")
    client = get_llm_client()
    print(f"LLM client={type(client).__name__}")
    print(f"Generating refactor proposals for top {k} DIV nodes…\n")

    t0 = time.perf_counter()
    results = generate_refactor_proposals(top_k=k, llm=client)
    elapsed = time.perf_counter() - t0
    print(f"\nWall-clock time: {elapsed:.1f}s\n")
    _print_proposals(results)
    if results and results[0].get("raw_llm_output"):
        print("\n----- RAW LLM OUTPUT -----")
        print(results[0]["raw_llm_output"])
        print("----- END RAW LLM OUTPUT -----")
