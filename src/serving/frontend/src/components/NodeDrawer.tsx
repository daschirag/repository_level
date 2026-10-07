import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { fetchFunctionContext, type CallNeighbor, type DebtNode, type FunctionContext } from "../api";
import { useElapsedSeconds } from "../lib/hooks";
import { formatDiv, severityOf } from "../lib/scales";
import type { ProposalState } from "../lib/useRefactorJobs";
import { Skeleton } from "./ui";

type Props = {
  node: DebtNode;
  maxDiv: number;
  colorOf: (div: number) => string;
  qdrantOnline: boolean | null;
  proposal: ProposalState | undefined;
  onGenerate: () => void;
  onNavigate: (neighbor: CallNeighbor) => void;
  onClose: () => void;
};

type ContextState =
  | { status: "loading" }
  | { status: "ready"; data: FunctionContext }
  | { status: "error"; message: string };

const SEVERITY_LABEL = { critical: "Critical", high: "High", medium: "Medium", low: "Low" } as const;

export function NodeDrawer({
  node,
  maxDiv,
  colorOf,
  qdrantOnline,
  proposal,
  onGenerate,
  onNavigate,
  onClose,
}: Props) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const [ctx, setCtx] = useState<ContextState>({ status: "loading" });
  const [reloadKey, setReloadKey] = useState(0);

  // Move focus into the drawer on open; hand it back on close.
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    closeRef.current?.focus({ preventScroll: true });
    return () => previous?.focus?.({ preventScroll: true });
  }, []);

  useEffect(() => {
    let cancelled = false;
    setCtx({ status: "loading" });
    fetchFunctionContext(node.file_path, node.name)
      .then((data) => !cancelled && setCtx({ status: "ready", data }))
      .catch(
        (err) =>
          !cancelled &&
          setCtx({ status: "error", message: err instanceof Error ? err.message : String(err) }),
      );
    return () => {
      cancelled = true;
    };
  }, [node.file_path, node.name, reloadKey]);

  const severity = severityOf(node.div_score, maxDiv);
  const share = maxDiv > 0 ? node.div_score / maxDiv : 0;
  const lines =
    node.start_line && node.end_line ? `L${node.start_line}–${node.end_line}` : node.start_line ? `L${node.start_line}` : null;

  return (
    <motion.aside
      className="drawer"
      role="dialog"
      aria-modal="false"
      aria-labelledby="drawer-title"
      initial={{ x: "100%", opacity: 0.6 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: "100%", opacity: 0.6 }}
      transition={{ type: "spring", stiffness: 380, damping: 40 }}
    >
      <header className="drawer__head">
        <div className="drawer__title">
          <span className={`chip chip--${severity}`}>{SEVERITY_LABEL[severity]} debt</span>
          <h2 id="drawer-title">{node.name}</h2>
          <p className="drawer__file">
            {node.file_path}
            {lines && <span className="muted"> · {lines}</span>}
          </p>
        </div>
        <button ref={closeRef} type="button" className="icon-btn" onClick={onClose} aria-label="Close details (Esc)">
          <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
            <path d="M5 5l10 10M15 5L5 15" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
        </button>
      </header>

      <AnimatePresence mode="wait" initial={false}>
        <motion.div
          key={node.id}
          className="drawer__body"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.2 }}
        >
          <section className="metrics">
            <div className="metric metric--div">
              <span className="metric__label">DIV score</span>
              <span className="metric__value mono" style={{ color: colorOf(node.div_score) }}>
                {formatDiv(node.div_score, 3)}
              </span>
              <span className="bar" aria-hidden="true">
                <motion.span
                  className="bar__fill"
                  style={{ background: colorOf(node.div_score), originX: 0 }}
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: Math.max(share, 0.015) }}
                  transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
                />
              </span>
              <span className="metric__hint">{Math.round(share * 100)}% of repo max</span>
            </div>
            <div className="metric">
              <span className="metric__label">Complexity</span>
              <span className="metric__value mono">{node.cyclomatic_complexity}</span>
              <span className="metric__hint">cyclomatic</span>
            </div>
            <div className="metric">
              <span className="metric__label">Churn</span>
              <span className="metric__value mono">{node.change_frequency}</span>
              <span className="metric__hint">commits to file</span>
            </div>
          </section>

          {ctx.status === "error" && (
            <div className="inline-error" role="alert">
              <span>Couldn’t load context: {ctx.message}</span>
              <button type="button" className="btn btn--ghost" onClick={() => setReloadKey((k) => k + 1)}>
                Retry
              </button>
            </div>
          )}

          <section className="neighbors">
            <NeighborList
              title="Callers"
              hint="functions that call this one"
              items={ctx.status === "ready" ? ctx.data.callers ?? [] : null}
              colorOf={colorOf}
              onNavigate={onNavigate}
            />
            <NeighborList
              title="Callees"
              hint="functions this one calls"
              items={ctx.status === "ready" ? ctx.data.callees ?? [] : null}
              colorOf={colorOf}
              onNavigate={onNavigate}
            />
          </section>

          <section className="section">
            <h3 className="section__title">
              Retrieved context <span className="muted">· Qdrant semantic neighbours</span>
            </h3>
            {ctx.status === "loading" && (
              <div className="stack">
                <Skeleton height={56} radius={10} />
                <Skeleton height={56} radius={10} />
              </div>
            )}
            {ctx.status === "ready" &&
              ((ctx.data.semantic_neighbors ?? []).length === 0 ? (
                <p className="muted small">
                  {qdrantOnline === false
                    ? "Qdrant is offline, so no semantic context was retrieved."
                    : "No semantic context indexed for this function. Only functions and classes with docstrings or comments are embedded."}
                </p>
              ) : (
                <ul className="rag">
                  {(ctx.data.semantic_neighbors ?? []).map((s, i) => (
                    <li key={`${s.file_path}::${s.name}::${i}`} className="rag__item">
                      <div className="rag__head">
                        <strong>{s.name ?? "—"}</strong>
                        {s.node_type && <span className="pill pill--quiet">{s.node_type}</span>}
                        <span className="rag__score mono" title="Cosine similarity">
                          {(s.score * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="rag__file">
                        {s.file_path}
                        {s.start_line ? `:${s.start_line}` : ""}
                      </div>
                      {s.text && <p className="rag__text">{s.text}</p>}
                    </li>
                  ))}
                </ul>
              ))}
          </section>

          <ProposalSection proposal={proposal} onGenerate={onGenerate} />
        </motion.div>
      </AnimatePresence>
    </motion.aside>
  );
}

function NeighborList({
  title,
  hint,
  items,
  colorOf,
  onNavigate,
}: {
  title: string;
  hint: string;
  items: CallNeighbor[] | null;
  colorOf: (div: number) => string;
  onNavigate: (n: CallNeighbor) => void;
}) {
  return (
    <div className="neighbors__col">
      <h3 className="section__title" title={hint}>
        {title} {items && <span className="count">{items.length}</span>}
      </h3>
      {!items && (
        <div className="stack">
          <Skeleton height={28} radius={8} />
          <Skeleton height={28} radius={8} width="80%" />
        </div>
      )}
      {items && items.length === 0 && <p className="muted small">None</p>}
      {items && items.length > 0 && (
        <ul className="neighbors__list">
          {items.map((n) => (
            <li key={`${n.file_path}::${n.name}`}>
              <button type="button" className="neighbor" onClick={() => onNavigate(n)} title={n.file_path}>
                <span className="swatch" style={{ background: colorOf(n.div_score ?? 0) }} />
                <span className="neighbor__name">{n.name}</span>
                <span className="mono muted">{formatDiv(n.div_score)}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

const STATUS_TEXT: Record<ProposalState["status"], string> = {
  starting: "Submitting job…",
  queued: "Queued…",
  running: "StarCoder2 is drafting a proposal…",
  completed: "Done",
  failed: "Failed",
};

function ProposalSection({
  proposal,
  onGenerate,
}: {
  proposal: ProposalState | undefined;
  onGenerate: () => void;
}) {
  const busy = proposal && ["starting", "queued", "running"].includes(proposal.status);
  const elapsed = useElapsedSeconds(busy ? proposal!.startedAt : null);
  const result = proposal?.status === "completed" ? proposal.proposal : null;

  return (
    <section className="section proposal" aria-live="polite" aria-busy={busy || undefined}>
      <div className="proposal__head">
        <h3 className="section__title">Refactor proposal</h3>
        <button type="button" className="btn btn--primary" onClick={onGenerate} disabled={busy}>
          {busy ? (
            <>
              <span className="spinner" aria-hidden="true" /> Generating…
            </>
          ) : result ? (
            "Regenerate"
          ) : (
            "Generate refactor proposal"
          )}
        </button>
      </div>

      {!proposal && (
        <p className="muted small">
          Runs the LangGraph refactor agent with StarCoder2 on this function’s source, metrics and
          context. Local CPU inference usually takes 1–2 minutes.
        </p>
      )}

      {busy && (
        <div className="proposal__loading">
          <div className="proposal__status">
            <span>{STATUS_TEXT[proposal!.status]}</span>
            <span className="mono muted">{elapsed}s</span>
          </div>
          <div className="stack">
            <Skeleton width={120} height={20} radius={999} />
            <Skeleton height={12} />
            <Skeleton height={12} width="85%" />
            <Skeleton height={110} radius={10} />
          </div>
        </div>
      )}

      {proposal?.status === "failed" && (
        <div className="inline-error" role="alert">
          <span>{proposal.error}</span>
          <button type="button" className="btn btn--ghost" onClick={onGenerate}>
            Retry
          </button>
        </div>
      )}

      {result && (
        <motion.div
          className="proposal__result"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          {result.refactor_type && <span className="chip chip--accent">{result.refactor_type.replace(/_/g, " ")}</span>}
          {result.rationale && <p className="proposal__rationale">{result.rationale}</p>}
          <pre className="code">
            <code>{result.proposal_text || "(empty proposal)"}</code>
          </pre>
          {result.raw_llm_output && result.raw_llm_output !== result.proposal_text && (
            <details className="raw">
              <summary>Raw model output</summary>
              <pre className="code">
                <code>{result.raw_llm_output}</code>
              </pre>
            </details>
          )}
        </motion.div>
      )}
    </section>
  );
}
