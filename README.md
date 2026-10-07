# Tech Debt Agent

Repo-level technical debt quantification and refactor proposal system (Layers 1–6): AST ingestion, Neo4j call graph + DIV scoring, CodeBERT/Qdrant RAG, StarCoder2 refactor agent, and a FastAPI + React heatmap UI.

## Prerequisites

- Python 3.11 + `uv` (project dependencies in `pyproject.toml`)
- Neo4j at `bolt://127.0.0.1:7687` (user `neo4j` / password `techdebt123`)
- Qdrant at `localhost:6333` (optional for RAG drill-down; context endpoint degrades if down)
- Node.js 18+ and npm (frontend only)
- StarCoder2 GGUF at `models/starcoder2/starcoder2-7b-Q4_K_M.gguf` (optional; agent falls back to mock LLM)

### Start Neo4j + Qdrant

From the project root (uses `docker-compose.yml`; waits until both report healthy):

```bash
docker compose up -d --wait
```

## Load the demo repository (one command)

With Neo4j + Qdrant running, from the project root:

```bash
python scripts/load_demo.py            # library code (src/) only — the dashboard default
python scripts/load_demo.py --include-tests   # also analyse tests/, examples/, docs/
python scripts/load_demo.py --skip-qdrant     # skip the embedding step
```

What it does (~30 s after the first run):

1. Clones `pallets/flask` with full history into `data/demo/flask` (~13 MB, gitignored) and checks out
   `36e4a824`, the exact commit `tests/fixtures/flask-test-repo` was copied from. The fixture has no
   `.git` of its own, so the clone supplies real commit history for churn.
2. Verifies the fixture is byte-identical to that commit (git blob ids).
3. Parses the fixture, computes radon complexity, rebuilds the Neo4j call graph, propagates DIV,
   and re-indexes Qdrant.
4. Records repo, commit, counts and per-stage timings on an `(:Analysis {id:'latest'})` node,
   which the dashboard header and pipeline stepper read via `GET /api/analysis`.

The loader **replaces** whatever graph and Qdrant collection were loaded before.

## Layer 6 — API + heatmap UI

### Backend (FastAPI)

From the project root (`D:\techdebt-agent`):

```bash
# PowerShell
$env:PYTHONPATH = "D:\techdebt-agent"
uvicorn src.serving.api.main:app --reload --port 8000
```

API docs: http://127.0.0.1:8000/docs

Key routes:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Neo4j / Qdrant / LLM status (`llm.status`: loading → ready) |
| GET | `/api/analysis` | Last-analysis metadata + live graph counts |
| GET | `/api/debt-graph` | Nodes + CALLS/IMPORTS edges for D3 |
| GET | `/api/top-debt?k=10` | Top-k DIV functions (with graph `id`) |
| GET | `/api/function/context?file_path=...&name=...` | RAG + 1-hop neighbours + `impact` + `rule_based` suggestions |
| GET | `/api/function/{file_path}/{name}/context` | Same, path-param form (`file_path` URL-encoded) |
| POST | `/api/refactor-proposals` | `{ "file_path", "name", "force"? }` → `{ job_id, cached }` (or `{ "top_k": n }`) |
| GET | `/api/refactor-proposals/{job_id}` | Poll status, `stage` / `stages_seen`, and results |

Refactor proposals: the model loads **once** in a background thread at API startup and is shared by all
jobs. Model output must contain code (or a concrete change) for the target function; README-style or
unrelated text is rejected and retried once with a stricter code-completion prompt. If both attempts fail
the result is `source: "rule_based"` and the UI shows the deterministic suggestions from
`src/agent/rules.py` instead. Completed proposals are cached per function until `force: true`.

### Frontend (Vite + React + D3)

```bash
cd src/serving/frontend
npm install
npm run dev
```

Open http://localhost:5173 — Vite proxies `/api` to `http://127.0.0.1:8000`.

## Earlier layers (CLI)

```bash
$env:PYTHONPATH = "D:\techdebt-agent"

# Layer 1 — AST parse
python -m src.ingestion.ast_parser path\to\repo

# Layer 1 — git churn (single-pass)
python -m src.ingestion.git_crawler path\to\repo

# Layer 2 — Neo4j ingest
python -m src.graph.neo4j_client path\to\repo

# Layer 3 — DIV scores
python -m src.scoring.div_propagation

# Layer 4 — Qdrant index + sample query
python -m src.rag.qdrant_client path\to\repo

# Layer 5 — refactor proposals (uses StarCoder2 GGUF if present)
python -m src.agent.refactor_agent 1
```

## Project Status (Review 1)
Phases 1-4 implemented: ingestion, call graph, DIV propagation, RAG + agent, plus the dashboard. Evaluation and ablation study are pending.

