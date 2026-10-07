import { useCallback, useEffect, useMemo, useState } from "react";
import { AnimatePresence, MotionConfig, motion } from "framer-motion";
import {
  fetchDebtGraph,
  fetchHealth,
  fetchTopDebt,
  type CallNeighbor,
  type DebtGraphResponse,
  type DebtNode,
  type HealthResponse,
  type TopDebtNode,
} from "./api";
import { GraphView } from "./components/GraphView";
import { Header } from "./components/Header";
import { NodeDrawer } from "./components/NodeDrawer";
import { StatCards, type Stats } from "./components/StatCards";
import { TopDebtList } from "./components/TopDebtList";
import { EmptyState, ErrorBanner, Skeleton } from "./components/ui";
import { makeDivColor, makeDivRadius } from "./lib/scales";
import { useRefactorJobs } from "./lib/useRefactorJobs";

const TOP_K = 15;

type LoadState =
  | { status: "loading" }
  | { status: "ready"; graph: DebtGraphResponse; top: TopDebtNode[] }
  | { status: "error"; message: string };

export default function App() {
  const [load, setLoad] = useState<LoadState>({ status: "loading" });
  const [health, setHealth] = useState<HealthResponse | null | "error">(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const { jobs, start } = useRefactorJobs();

  useEffect(() => {
    let cancelled = false;
    setLoad((prev) => (prev.status === "ready" ? prev : { status: "loading" }));
    fetchHealth()
      .then((h) => !cancelled && setHealth(h))
      .catch(() => !cancelled && setHealth("error"));
    Promise.all([fetchDebtGraph(), fetchTopDebt(TOP_K)])
      .then(([graph, top]) => !cancelled && setLoad({ status: "ready", graph, top: top.nodes }))
      .catch((err) => {
        if (cancelled) return;
        const raw = err instanceof Error ? err.message : String(err);
        const message = /Failed to fetch|NetworkError|ECONNREFUSED|^50[234]/.test(raw)
          ? "The API at /api is unreachable. Start it with `uvicorn src.serving.api.main:app --port 8000` and make sure Neo4j is running."
          : raw;
        setLoad({ status: "error", message });
      });
    return () => {
      cancelled = true;
    };
  }, [attempt]);

  const retry = useCallback(() => {
    setLoad({ status: "loading" });
    setAttempt((a) => a + 1);
  }, []);

  const graph = load.status === "ready" ? load.graph : null;

  const functions = useMemo(
    () => (graph ? graph.nodes.filter((n) => (n.node_kind ?? "Function") === "Function") : []),
    [graph],
  );
  const callEdges = useMemo(() => (graph ? graph.edges.filter((e) => e.type === "CALLS") : []), [graph]);
  const nodeById = useMemo(() => new Map(functions.map((n) => [n.id, n])), [functions]);

  const stats = useMemo<Stats | null>(() => {
    if (!graph) return null;
    let max: DebtNode | null = null;
    let sum = 0;
    for (const n of functions) {
      sum += n.div_score || 0;
      if (!max || n.div_score > max.div_score) max = n;
    }
    return {
      nodeCount: graph.node_count,
      edgeCount: graph.edge_count,
      functionCount: functions.length,
      callCount: callEdges.length,
      importCount: graph.edges.length - callEdges.length,
      maxDiv: max?.div_score ?? 0,
      maxDivName: max ? max.name : null,
      avgDiv: functions.length ? sum / functions.length : 0,
    };
  }, [graph, functions, callEdges]);

  const maxDiv = stats?.maxDiv ?? 0;
  const colorOf = useMemo(() => makeDivColor(maxDiv), [maxDiv]);
  const radiusOf = useMemo(() => makeDivRadius(maxDiv), [maxDiv]);

  const selected = selectedId ? nodeById.get(selectedId) ?? null : null;
  const close = useCallback(() => setSelectedId(null), []);

  useEffect(() => {
    if (!selected) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selected, close]);

  const selectNode = useCallback((n: { id: string }) => setSelectedId(n.id), []);
  const navigate = useCallback(
    (n: CallNeighbor) => {
      const id = `${n.file_path}::${n.name}`;
      if (nodeById.has(id)) setSelectedId(id);
    },
    [nodeById],
  );

  const qdrantOnline = health && health !== "error" ? health.qdrant : health === "error" ? false : null;
  const isEmpty = load.status === "ready" && functions.length === 0;

  return (
    <MotionConfig reducedMotion="user">
      <div className={`app ${selected ? "has-drawer" : ""}`}>
        <Header health={health} />

        <main className="content">
          <AnimatePresence>
            {load.status === "error" && (
              <ErrorBanner title="Couldn’t load debt data" message={load.message} onRetry={retry} />
            )}
          </AnimatePresence>

          <StatCards stats={stats} />

          <motion.div
            className="workspace"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
          >
            <section className="card panel panel--graph" aria-label="Call graph">
              <div className="panel__head">
                <h2>Call-graph heat-map</h2>
                <span className="muted small">Hover to trace callers/callees · drag · scroll to zoom</span>
              </div>
              {load.status !== "ready" && (
                <div className="graph-skeleton" aria-busy={load.status === "loading"}>
                  <Skeleton height="100%" radius={14} />
                  {load.status === "error" && <span className="graph-skeleton__msg">No data — backend unreachable</span>}
                </div>
              )}
              {isEmpty && (
                <EmptyState title="No functions in the graph yet">
                  <p>Ingest a repository, then compute DIV scores:</p>
                  <pre className="code">
                    <code>
                      python -m src.graph.neo4j_client path/to/repo{"\n"}python -m src.scoring.div_propagation
                    </code>
                  </pre>
                  <button type="button" className="btn btn--ghost" onClick={retry}>
                    Reload
                  </button>
                </EmptyState>
              )}
              {load.status === "ready" && !isEmpty && (
                <GraphView
                  nodes={functions}
                  edges={callEdges}
                  maxDiv={maxDiv}
                  colorOf={colorOf}
                  radiusOf={radiusOf}
                  selectedId={selectedId}
                  onSelect={selectNode}
                />
              )}
            </section>

            <aside className="card panel panel--list" aria-label="Top debt nodes">
              <TopDebtList
                nodes={load.status === "ready" ? load.top : load.status === "error" ? [] : null}
                maxDiv={maxDiv}
                colorOf={colorOf}
                selectedId={selectedId}
                onSelect={selectNode}
              />
            </aside>
          </motion.div>
        </main>

        <AnimatePresence>
          {selected && (
            <NodeDrawer
              node={selected}
              maxDiv={maxDiv}
              colorOf={colorOf}
              qdrantOnline={qdrantOnline}
              proposal={jobs[selected.id]}
              onGenerate={() => start(selected)}
              onNavigate={navigate}
              onClose={close}
            />
          )}
        </AnimatePresence>
      </div>
    </MotionConfig>
  );
}
