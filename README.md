# Tech Debt Agent

Repo-level technical debt quantification and refactor proposal system (Layers 1–6): AST ingestion, Neo4j call graph + DIV scoring, CodeBERT/Qdrant RAG, StarCoder2 refactor agent, and a FastAPI + React heatmap UI.

## Prerequisites

- Python 3.11 + `uv` (project dependencies in `pyproject.toml`)
- Neo4j at `bolt://127.0.0.1:7687` (user `neo4j` / password `techdebt123`)
- Qdrant at `localhost:6333` (optional for RAG drill-down; context endpoint degrades if down)
- Node.js 18+ and npm (frontend only)
- StarCoder2 GGUF at `models/starcoder2/starcoder2-7b-Q4_K_M.gguf` (optional; agent falls back to mock LLM)

### Start Qdrant (if not running)

```bash
docker run -d -p 6333:6333 -p 6334:6334 --name techdebt-qdrant -v D:\techdebt-agent\data\qdrant:/qdrant/storage qdrant/qdrant
```

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
| GET | `/api/debt-graph` | Nodes + CALLS/IMPORTS edges for D3 |
| GET | `/api/top-debt?k=10` | Top-k DIV functions |
| GET | `/api/function/context?file_path=...&name=...` | RAG + 1-hop neighbors (slash-safe) |
| GET | `/api/function/{file_path}/{name}/context` | Same, path-param form (`file_path` URL-encoded) |
| POST | `/api/refactor-proposals` | `{ "top_k": 1 }` → `{ job_id }` (background LLM) |
| GET | `/api/refactor-proposals/{job_id}` | Poll job status / results |

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
