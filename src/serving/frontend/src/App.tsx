import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchDebtGraph,
  fetchFunctionContext,
  fetchTopDebt,
  startRefactorJob,
  fetchRefactorJob,
  type DebtNode,
  type DebtGraphResponse,
  type FunctionContext,
} from "./api";
import { DebtGraphView } from "./DebtGraphView";

type ViewMode = "graph" | "table";

export default function App() {
  const [view, setView] = useState<ViewMode>("graph");
  const [graph, setGraph] = useState<DebtGraphResponse | null>(null);
  const [topNodes, setTopNodes] = useState<DebtNode[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<DebtNode | null>(null);
  const [context, setContext] = useState<FunctionContext | null>(null);
  const [contextLoading, setContextLoading] = useState(false);
  const [topN, setTopN] = useState(80);
  const [callsOnly, setCallsOnly] = useState(true);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [proposals, setProposals] = useState<unknown[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const [g, top] = await Promise.all([fetchDebtGraph(), fetchTopDebt(15)]);
        if (cancelled) return;
        setGraph(g);
        setTopNodes(top.nodes);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const visible = useMemo(() => {
    if (!graph) return { nodes: [] as DebtNode[], edges: graph?.edges ?? [] };
    const functions = graph.nodes
      .filter((n) => (n.node_kind ?? "Function") === "Function")
      .slice()
      .sort((a, b) => (b.div_score || 0) - (a.div_score || 0));
    const keep = new Set(functions.slice(0, topN).map((n) => n.id));
    const nodes = graph.nodes.filter((n) => keep.has(n.id));
    const edges = graph.edges.filter((e) => {
      if (callsOnly && e.type !== "CALLS") return false;
      return keep.has(String(e.source)) && keep.has(String(e.target));
    });
    return { nodes, edges };
  }, [graph, topN, callsOnly]);

  const onSelect = useCallback((node: DebtNode) => {
    setSelected(node);
    setContext(null);
  }, []);

  const loadContext = async () => {
    if (!selected) return;
    setContextLoading(true);
    setError(null);
    try {
      const ctx = await fetchFunctionContext(selected.file_path, selected.name);
      setContext(ctx);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setContextLoading(false);
    }
  };

  const enqueueRefactor = async () => {
    setError(null);
    setProposals(null);
    try {
      const { job_id } = await startRefactorJob(1);
      setJobId(job_id);
      setJobStatus("queued");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    const timer = setInterval(async () => {
      try {
        const job = await fetchRefactorJob(jobId);
        if (cancelled) return;
        const status = String(job.status ?? "");
        setJobStatus(status);
        if (status === "completed") {
          const result = job.result as { proposals?: unknown[] } | undefined;
          setProposals(result?.proposals ?? []);
          clearInterval(timer);
        } else if (status === "failed") {
          setError(String(job.error ?? "Refactor job failed"));
          clearInterval(timer);
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
        clearInterval(timer);
      }
    }, 2500);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [jobId]);

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Tech Debt Heatmap</h1>
          <p className="muted">
            DIV-ranked call graph — green = low debt, red = high debt
          </p>
        </div>
        <div className="header-actions">
          <button
            className={view === "graph" ? "active" : ""}
            onClick={() => setView("graph")}
            type="button"
          >
            Graph
          </button>
          <button
            className={view === "table" ? "active" : ""}
            onClick={() => setView("table")}
            type="button"
          >
            Top debt
          </button>
        </div>
      </header>

      {error && <div className="banner error">{error}</div>}
      {loading && <div className="banner">Loading debt graph…</div>}

      <div className="layout">
        <main className="main">
          {view === "graph" && graph && (
            <>
              <div className="toolbar">
                <label>
                  Top N functions
                  <input
                    type="number"
                    min={10}
                    max={500}
                    value={topN}
                    onChange={(e) => setTopN(Number(e.target.value) || 80)}
                  />
                </label>
                <label className="check">
                  <input
                    type="checkbox"
                    checked={callsOnly}
                    onChange={(e) => setCallsOnly(e.target.checked)}
                  />
                  CALLS edges only
                </label>
                <span className="muted">
                  showing {visible.nodes.length} / {graph.node_count} nodes,{" "}
                  {visible.edges.length} edges
                </span>
              </div>
              <DebtGraphView
                nodes={visible.nodes}
                edges={visible.edges}
                selectedId={selected?.id ?? null}
                onSelect={onSelect}
              />
            </>
          )}

          {view === "table" && (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>name</th>
                    <th>file</th>
                    <th>DIV</th>
                    <th>cc</th>
                    <th>chg</th>
                  </tr>
                </thead>
                <tbody>
                  {topNodes.map((n, i) => (
                    <tr
                      key={`${n.file_path}::${n.name}`}
                      className={
                        selected?.file_path === n.file_path && selected?.name === n.name
                          ? "selected"
                          : ""
                      }
                      onClick={() =>
                        onSelect({
                          ...n,
                          id: `${n.file_path}::${n.name}`,
                          node_kind: "Function",
                        })
                      }
                    >
                      <td>{i + 1}</td>
                      <td>{n.name}</td>
                      <td className="path">{n.file_path}</td>
                      <td>{Number(n.div_score).toFixed(2)}</td>
                      <td>{n.cyclomatic_complexity}</td>
                      <td>{n.change_frequency}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </main>

        <aside className="side">
          <h2>Node details</h2>
          {!selected && <p className="muted">Click a node or table row.</p>}
          {selected && (
            <div className="details">
              <div className="kv">
                <span>Name</span>
                <strong>{selected.name}</strong>
              </div>
              <div className="kv">
                <span>File</span>
                <strong className="path">{selected.file_path}</strong>
              </div>
              <div className="kv">
                <span>DIV</span>
                <strong>{Number(selected.div_score).toFixed(3)}</strong>
              </div>
              <div className="kv">
                <span>Complexity</span>
                <strong>{selected.cyclomatic_complexity}</strong>
              </div>
              <div className="kv">
                <span>Change freq</span>
                <strong>{selected.change_frequency}</strong>
              </div>
              <button type="button" onClick={loadContext} disabled={contextLoading}>
                {contextLoading ? "Loading context…" : "Fetch RAG + graph context"}
              </button>
              <button type="button" className="secondary" onClick={enqueueRefactor}>
                Queue refactor proposal (top_k=1)
              </button>
              {jobId && (
                <p className="muted">
                  Job {jobId.slice(0, 8)}… — {jobStatus}
                </p>
              )}
            </div>
          )}

          {context && (
            <div className="context">
              <h3>Context</h3>
              <p className="muted">
                Callers: {context.callers?.length ?? 0} · Callees:{" "}
                {context.callees?.length ?? 0} · Neighbors:{" "}
                {context.semantic_neighbors?.length ?? 0}
              </p>
              <pre>{JSON.stringify(context, null, 2)}</pre>
            </div>
          )}

          {proposals && (
            <div className="context">
              <h3>Refactor proposals</h3>
              <pre>{JSON.stringify(proposals, null, 2)}</pre>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
