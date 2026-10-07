import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  fetchFunctionContext,
  type CallNeighbor,
  type DebtNode,
  type FunctionContext,
  type JobStage,
  type LlmState,
  type RuleSuggestion,
} from "../api";
import {
  DIV_FORMULA,
  inheritedDebt,
  locOf,
  ownDebt,
  whyFragments,
  type BlastRadius,
  type GraphIndex,
  type RepoBaselines,
} from "../lib/analysis";
import { useElapsedSeconds } from "../lib/hooks";
import { formatDiv, severityOf } from "../lib/scales";
import type { ProposalState } from "../lib/useRefactorJobs";
import { Skeleton } from "./ui";

type Props = {
  node: DebtNode;
  rank: number;
  total: number;
  maxDiv: number;
  index: GraphIndex;
  base: RepoBaselines;
  blast: BlastRadius;
  colorOf: (div: number) => string;
  qdrantOnline: boolean | null;
  llm: LlmState | null;
  proposal: ProposalState | undefined;
  onGenerate: (force: boolean) => void;
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
  rank,
  total,
  maxDiv,
  index,
  base,
  blast,
  colorOf,
  qdrantOnline,
  llm,
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
  const own = ownDebt(node);
  const inherited = inheritedDebt(node);
  const ownShare = node.div_score > 0 ? own / node.div_score : 0;
  const why = whyFragments(node, index, base, blast);
  const loc = locOf(node);
  const lines = node.start_line && node.end_line ? `L${node.start_line}–${node.end_line}` : null;
  const byDepth = useMemo(() => {
    const counts = new Map<number, number>();
    for (const d of blast.depth.values()) counts.set(d, (counts.get(d) ?? 0) + 1);
    return [...counts.entries()].sort((a, b) => a[0] - b[0]);
  }, [blast]);

  const rules = ctx.status === "ready" ? ctx.data.rule_based ?? [] : null;

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
          <span className={`chip chip--${severity}`}>
            {SEVERITY_LABEL[severity]} debt · #{rank} of {total}
          </span>
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
          {/* ---- Why this is debt ---- */}
          <section className="why" aria-labelledby="why-title">
            <h3 id="why-title" className="section__title">
              Why this is debt
            </h3>
            <p className="why__lead">
              DIV <strong className="mono" style={{ color: colorOf(node.div_score) }}>{formatDiv(node.div_score, 1)}</strong>
              {why.length > 0 && <> — {why.join(" + ")}.</>}
            </p>
            <div className="why__split" title={DIV_FORMULA}>
              <div className="why__bar" aria-hidden="true">
                <motion.span
                  className="why__own"
                  initial={{ width: 0 }}
                  animate={{ width: `${ownShare * 100}%` }}
                  transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
                />
                <motion.span
                  className="why__inh"
                  initial={{ width: 0 }}
                  animate={{ width: `${(1 - ownShare) * 100}%` }}
                  transition={{ duration: 0.7, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
                />
              </div>
              <dl className="why__legend">
                <div>
                  <dt>
                    <i className="dot dot--own" /> Own debt
                  </dt>
                  <dd className="mono">{formatDiv(own, 1)}</dd>
                  <dd className="why__how">
                    complexity {node.cyclomatic_complexity} × ln(1 + {node.change_frequency} commits to file)
                  </dd>
                </div>
                <div>
                  <dt>
                    <i className="dot dot--inh" /> Inherited from callers
                  </dt>
                  <dd className="mono">{formatDiv(inherited, 1)}</dd>
                  <dd className="why__how">
                    {blast.functions > 0
                      ? `decayed debt of ${blast.functions} transitive caller${blast.functions === 1 ? "" : "s"}`
                      : "no callers in the graph"}
                  </dd>
                </div>
              </dl>
            </div>
          </section>

          {/* ---- Blast radius ---- */}
          <section className={`blast ${blast.functions > 0 ? "" : "blast--none"}`} aria-label="Blast radius">
            <div className="blast__icon" aria-hidden="true">
              <span />
              <span />
              <span />
            </div>
            <div>
              <p className="blast__headline">
                {blast.functions > 0 ? (
                  <>
                    If this breaks: <strong>{blast.functions}</strong> function{blast.functions === 1 ? "" : "s"} in{" "}
                    <strong>{blast.files}</strong> file{blast.files === 1 ? "" : "s"} are affected
                  </>
                ) : (
                  <>Nothing in the graph calls this function — a failure here stays local.</>
                )}
              </p>
              {byDepth.length > 0 && (
                <p className="blast__depths mono">
                  {byDepth.map(([d, c]) => `${c} at ${d} hop${d === 1 ? "" : "s"}`).join(" · ")}
                </p>
              )}
            </div>
          </section>

          <section className="metrics">
            <div className="metric">
              <span className="metric__label">Complexity</span>
              <span className="metric__value mono">{node.cyclomatic_complexity}</span>
              <span className="metric__hint">cyclomatic (p75: {base.complexityP75})</span>
            </div>
            <div className="metric">
              <span className="metric__label">Churn</span>
              <span className="metric__value mono">{node.change_frequency}</span>
              <span className="metric__hint">commits to file (p75: {base.churnP75})</span>
            </div>
            <div className="metric">
              <span className="metric__label">Size</span>
              <span className="metric__value mono">{loc || "—"}</span>
              <span className="metric__hint">lines of code</span>
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

          {/* ---- Remediation ---- */}
          <section className="section remediation">
            <h3 className="section__title">Remediation</h3>
            <RuleList rules={rules} />
            <ProposalSection proposal={proposal} llm={llm} onGenerate={onGenerate} />
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
        </motion.div>
      </AnimatePresence>
    </motion.aside>
  );
}

function RuleList({ rules }: { rules: RuleSuggestion[] | null }) {
  return (
    <div className="rules">
      <div className="provenance provenance--rule">
        <span className="provenance__tag">Rule-based</span>
        <span className="muted small">deterministic checks on the graph metrics</span>
      </div>
      {!rules && (
        <div className="stack">
          <Skeleton height={44} radius={10} />
          <Skeleton height={44} radius={10} />
        </div>
      )}
      {rules && (
        <ul className="rules__list">
          {rules.map((r) => (
            <li key={r.rule} className="rules__item">
              <strong>{r.title}</strong>
              <span>{r.detail}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
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

const STAGE_LABEL: Record<JobStage, string> = {
  queued: "Queued",
  waiting_for_model: "Waiting for the model to finish loading",
  load_span: "Locating the function in the graph",
  gather_context: "Retrieving callers, callees and Qdrant context",
  gather_source: "Reading the source",
  generate_1: "Generating a proposal (attempt 1)",
  generate_2: "Rejected — retrying with a stricter code prompt",
  rules: "Applying rule checks",
  done: "Done",
};
const STAGE_ORDER: JobStage[] = [
  "queued",
  "waiting_for_model",
  "load_span",
  "gather_context",
  "gather_source",
  "generate_1",
  "generate_2",
  "rules",
  "done",
];

function ProposalSection({
  proposal,
  llm,
  onGenerate,
}: {
  proposal: ProposalState | undefined;
  llm: LlmState | null;
  onGenerate: (force: boolean) => void;
}) {
  const busy = !!proposal && ["starting", "queued", "running"].includes(proposal.status);
  const elapsed = useElapsedSeconds(busy ? proposal!.startedAt : null);
  const result = proposal?.status === "completed" ? proposal.proposal : null;
  const modelName = llm?.model ?? "the local model";
  // Stages to show: everything seen so far, plus the next expected ones (generate_2 only if it happened).
  const seen = proposal?.stagesSeen ?? [];
  const current = proposal?.stage ?? null;
  const shown = STAGE_ORDER.filter(
    (s) => s !== "done" && (seen.includes(s) || (s !== "generate_2" && s !== "waiting_for_model")),
  );
  const step = Math.max(0, shown.findIndex((s) => s === current));

  return (
    <div className="proposal" aria-live="polite" aria-busy={busy || undefined}>
      <div className="proposal__head">
        <div className="provenance provenance--ai">
          <span className="provenance__tag">AI-generated</span>
          <span className="muted small">{modelName}, validated before display</span>
        </div>
        <button
          type="button"
          className="btn btn--primary"
          onClick={() => onGenerate(!!result)}
          disabled={busy || llm?.status === "failed"}
        >
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
          Runs the LangGraph agent on this function’s source, metrics and context. Output is checked for real code
          that changes this function; README-style or unrelated text is rejected and retried once.
          {llm?.status === "loading" && " The model is still loading."}
        </p>
      )}

      {busy && (
        <div className="progress">
          <div className="progress__top">
            <span>
              Step {Math.min(step + 1, shown.length)} of {shown.length}
            </span>
            <span className="mono muted">{elapsed}s elapsed</span>
          </div>
          <div className="progress__bar" aria-hidden="true">
            <motion.span
              className="progress__fill"
              animate={{ width: `${((step + 0.5) / shown.length) * 100}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
          <ol className="progress__stages">
            {shown.map((s) => {
              const state = s === current ? "active" : seen.includes(s) ? "done" : "pending";
              return (
                <li key={s} className={`stage stage--${state}`}>
                  <span className="stage__dot" aria-hidden="true" />
                  {STAGE_LABEL[s]}
                </li>
              );
            })}
          </ol>
          <div className="stack">
            <Skeleton height={12} />
            <Skeleton height={12} width="85%" />
            <Skeleton height={90} radius={10} />
          </div>
        </div>
      )}

      {proposal?.status === "failed" && (
        <div className="inline-error" role="alert">
          <span>{proposal.error}</span>
          <button type="button" className="btn btn--ghost" onClick={() => onGenerate(true)}>
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
          <div className="proposal__meta small muted">
            {proposal?.cached ? "Cached result" : "Generated"}
            {proposal?.durationS != null && ` in ${Math.round(proposal.durationS)}s`} · {result.attempts.length}{" "}
            attempt{result.attempts.length === 1 ? "" : "s"} · {result.model}
          </div>

          {result.source === "ai" && result.proposal_text ? (
            <>
              <div className="verdict verdict--ok">
                <strong>Passed validation</strong>
                {result.refactor_type && <span className="chip chip--accent">{result.refactor_type.replace(/_/g, " ")}</span>}
              </div>
              <p className="proposal__caveat small">
                Validation confirms this is code for <code>{result.name}</code>, not that it is correct or
                behaviour-preserving — review it like any pull request before applying.
              </p>
              {result.rationale && <p className="proposal__rationale">{result.rationale}</p>}
              <pre className="code">
                <code>{result.proposal_text}</code>
              </pre>
            </>
          ) : (
            <div className="verdict verdict--none" role="status">
              <strong>No valid proposal generated</strong>
              <span>
                The model’s output was rejected {result.attempts.length === 1 ? "once" : `${result.attempts.length} times`}.
                Use the rule-based suggestions above instead.
              </span>
              <ul className="verdict__reasons">
                {result.attempts.map((a) => (
                  <li key={a.attempt}>
                    Attempt {a.attempt} ({a.prompt_style}): {a.reasons.join("; ") || "rejected"}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {result.raw_llm_output && (
            <details className="raw">
              <summary>{result.source === "ai" ? "Raw model output" : "Rejected model output"}</summary>
              <pre className="code">
                <code>{result.raw_llm_output}</code>
              </pre>
            </details>
          )}
        </motion.div>
      )}
    </div>
  );
}
