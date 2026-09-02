# Repo-Level Technical Debt Quantification Agent — 50% Checkpoint Demo

**B.Tech IT Capstone Project · VIT Vellore · 2025-26**

This is a working, standalone slice of a larger 6-layer system that automatically
detects, causally ranks, and proposes fixes for technical debt across an entire
codebase — going beyond flat static-analysis tools like SonarQube, which score each
file in isolation and have no way of knowing that a simple-looking function might be
*called by* dozens of complex, high-churn modules and is therefore a much bigger risk
than its own complexity number suggests.

This checkpoint is a superset of the earlier `demo/ingestion-checkpoint/` slice: it
adds the call-graph construction and the causal debt-propagation algorithm, so the
core patentable novelty of the project — DIV propagation — actually runs end-to-end
and prints real, ranked output.

## What this checkpoint demonstrates

**Layers 1–3 of 6** in the full pipeline:

1. **Layer 1 — Ingestion.** Parses every source file with `tree-sitter` across six
   languages (Python, JavaScript, TypeScript, Java, Go, Rust), extracting every
   function, class, import, and call site. Crawls the full Git history to compute a
   change-frequency count per file.
2. **Layer 2 — Graph construction.** Loads the extracted structure into a real Neo4j
   graph database: `:Function`, `:Class`, `:File`, and `:Module` nodes, with `CALLS`
   and `IMPORTS` relationships. Call-site resolution is scoped (same-class →
   same-file → import-resolved) specifically so that common method names like `get`
   don't get spuriously linked to an unrelated same-named function elsewhere in the
   repo — this is a real graph of the codebase's actual call structure, not a flat
   file list.
3. **Layer 3 — Causal DIV propagation (the core novel contribution).** Computes a
   *Debt Impact Vector* (DIV) score for every function: each function gets a base
   weight from its own complexity and change frequency, and that weight is then
   propagated **upstream along the call graph** to every transitive caller, decaying
   with distance. A function that looks simple in isolation but is called by dozens
   of complex, frequently-changed functions ends up correctly ranked as high-risk —
   which is exactly the blind spot in tools like SonarQube that only look at one file
   at a time. The formula and full rationale are documented at the top of
   `src/scoring/div_propagation.py`.

The output you'll see is the **top-10 ranked debt nodes** for the sample repository —
this ranked list *is* the novel contribution working end-to-end, not a mockup of it.

## What's NOT in this checkpoint

Layers 4–6 of the full pipeline exist in the complete project but are **not** part of
this slice:

- **Layer 4:** embedding code documentation into a vector database (Qdrant) for
  retrieval-augmented context.
- **Layer 5:** an agentic LLM (local StarCoder2-7B) that combines the DIV ranking,
  call-graph context, and retrieved documentation to generate grounded refactoring
  suggestions.
- **Layer 6:** a web API and dashboard exposing the ranked debt report for CI/CD
  integration.

## Setup & run

Requires Python 3.11+ and Docker (for Neo4j). Everything below runs from this
`demo/50-percent-checkpoint/` folder.

### 1. Start Neo4j

```bash
docker compose up -d
```

Wait ~10-15 seconds for Neo4j to finish starting. You can check readiness with
`docker compose logs -f neo4j` (look for "Started.") or just proceed — the client
scripts below will fail fast with a clear connection error if it isn't ready yet.

### 2. Create a virtual environment and install dependencies

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Give the fixture a tiny git history (one-time)

`tests/fixtures/mini_repo` ships as plain files with no commit history of its own
(it's a fixture copied out of the main repo's test suite). The ingestion layer's
change-frequency signal comes from `git log` on the target repo, so without this step
every function's `change_frequency` reads `0` and DIV scores collapse to `0.0000`
across the board — the pipeline still runs correctly, but the ranking has nothing to
differentiate on. Give it a few commits so there's real churn to propagate:

**Windows (PowerShell)**
```powershell
cd tests/fixtures/mini_repo
git init
git config user.email "demo@example.com"
git config user.name "Demo"
git add -A; git commit -m "init"
"# churn 1" | Out-File -Append -Encoding utf8 pkg/sample.py; git add -A; git commit -m "edit py 1"
"# churn 2" | Out-File -Append -Encoding utf8 pkg/sample.py; git add -A; git commit -m "edit py 2"
"# churn 3" | Out-File -Append -Encoding utf8 pkg/sample.py; git add -A; git commit -m "edit py 3"
"// churn 1" | Out-File -Append -Encoding utf8 pkg/sample.js; git add -A; git commit -m "edit js 1"
"// churn 2" | Out-File -Append -Encoding utf8 pkg/sample.js; git add -A; git commit -m "edit js 2"
cd ../../..
```

**macOS / Linux**
```bash
cd tests/fixtures/mini_repo
git init
git config user.email "demo@example.com"
git config user.name "Demo"
git add -A && git commit -m "init"
for i in 1 2 3; do echo "# churn $i" >> pkg/sample.py; git add -A; git commit -m "edit py $i"; done
for i in 1 2; do echo "// churn $i" >> pkg/sample.js; git add -A; git commit -m "edit js $i"; done
cd ../../..
```

This is a one-time step per checkout — skip it on subsequent runs.

### 4. Run ingestion + graph construction

From `demo/50-percent-checkpoint/`:

```bash
python -m src.graph.neo4j_client tests/fixtures/mini_repo bolt://127.0.0.1:7687
```

(We use `bolt://127.0.0.1:7687` rather than `bolt://localhost:7687` — on Windows,
`localhost` can resolve to `::1` first and stall the connection; `127.0.0.1` avoids
that.)

This parses `tests/fixtures/mini_repo`, writes `:Function`/`:Class`/`:File`/`:Module`
nodes and `CALLS`/`IMPORTS` edges into Neo4j, attaches `change_frequency` and
`cyclomatic_complexity` to every function, and prints a summary of nodes/edges
written.

### 5. Run causal DIV propagation

```bash
python -m src.scoring.div_propagation bolt://127.0.0.1:7687 10
```

(First argument is the bolt URI, second is how many top debt nodes to print — both
optional, defaulting to `bolt://127.0.0.1:7687` and `10`.)

This pulls the `CALLS` subgraph out of Neo4j into NetworkX, computes each function's
DIV score (its own base weight plus decaying contributions from every transitive
caller), writes `div_score` back onto each `:Function` node, and prints the top-10
debt nodes ranked by DIV score.

### What you'll see with `mini_repo`

`tests/fixtures/mini_repo` is intentionally small (three files — Python, JS, TS — a
handful of functions each) so the whole pipeline runs in a few seconds and the
before/after of "raw complexity" vs. "propagated DIV score" is easy to eyeball in the
printed table. With the git history from step 3 in place, an actual verified run
produces:

```
rank   div_score    cc   chg  name                      file_path
----------------------------------------------------------------------------------------
   1      4.3133     1     4  format_greeting           pkg/sample.py
   2      3.8627     1     4  greet                     pkg/sample.py
   3      3.2189     2     4  main                      pkg/sample.py
   4      1.6094     1     4  __init__                  pkg/sample.py
   5      0.0000     0     3  add                       pkg/sample.js
   ...
```

This is the propagation effect in miniature: `format_greeting` has the *lowest*
cyclomatic complexity of the top four (`cc=1`) but ranks **#1** overall, because it
sits at the bottom of a call chain (`main` → `greet` → `format_greeting`) and inherits
a decaying share of every caller's debt above it. A flat, per-file tool would never
surface `format_greeting` above `main` (`cc=2`); DIV correctly does, because
`format_greeting` is the function that would actually be touched by a change
propagating through that call chain. Exact numbers will vary slightly depending on
how many churn commits you add in step 3, but the ranking behavior — low-complexity,
deeply-called functions surfacing above higher-complexity leaves — is the point being
demonstrated.

This same pipeline has also been validated against a much larger real-world
repository (Flask: ~83 files, ~1,460 functions), where the propagation produces
clearly differentiated DIV scores — functions with low individual complexity but many
upstream callers get pushed well above functions with higher raw complexity but few
callers. That run isn't included in this checkpoint (it's slower and not needed to
prove the pipeline works), but it's worth mentioning verbally when presenting this
checkpoint as evidence the algorithm holds up at real-world scale, not just on the toy
fixture.

### Resetting

`neo4j_client.py`'s `__main__` block calls `clear_graph()` before every run, so
re-running step 4 against the same Neo4j instance is safe and idempotent. To fully
reset Neo4j (wipe the Docker volume too):

```bash
docker compose down -v
docker compose up -d
```

## Why this matters (talking points for presenting)

- Existing tools (SonarQube, GitHub Copilot) either score files in isolation or have
  no repository-wide awareness at all — this project's core novelty is *causal* debt
  scoring that accounts for how risk cascades through a codebase's actual call
  structure, computed here as a real graph traversal over a real graph database, not
  a simulated or hard-coded example.
- Layers 1–2 (ingestion + graph construction) are the foundation the causal scoring
  depends on: you can't propagate debt along a call graph that doesn't accurately
  reflect the codebase's real call structure, which is why scoped call resolution
  (same-class → same-file → import-resolved, no guessing) matters as much as the
  propagation math itself.
- Layer 3's propagation formula and full algorithmic rationale — intended to double as
  patent-disclosure / thesis writeup language — are documented directly in the
  docstring at the top of `src/scoring/div_propagation.py`.
