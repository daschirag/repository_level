import type { AnalysisMeta, HealthResponse } from "../api";

const TEAM = [
  { id: "23BIT0198", name: "Minmini K" },
  { id: "23BIT0185", name: "Tarunica S" },
];

function StatusDot({ label, ok, title }: { label: string; ok: boolean | null; title?: string }) {
  const state = ok == null ? "unknown" : ok ? "ok" : "down";
  const text = ok == null ? "checking" : ok ? "online" : "offline";
  return (
    <span className={`status status--${state}`} title={title ?? `${label} ${text}`}>
      <span className="status__dot" aria-hidden="true" />
      {label}
      <span className="sr-only"> {text}</span>
    </span>
  );
}

function relativeTime(iso: string): string {
  const diff = (Date.now() - Date.parse(iso)) / 1000;
  if (!Number.isFinite(diff)) return iso;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.round(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)} h ago`;
  return new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

type Props = {
  health: HealthResponse | null | "error";
  meta: AnalysisMeta | null | undefined;
  presenting: boolean;
  onTogglePresenting: () => void;
};

export function Header({ health, meta, presenting, onTogglePresenting }: Props) {
  const h = health === "error" ? null : health;
  const unreachable = health === "error";
  const llm = h?.llm;
  const llmOk = unreachable ? false : !llm ? null : llm.status === "ready" ? true : llm.status === "failed" ? false : null;
  return (
    <header className="header">
      <div className="header__brand">
        <div className="header__logo" aria-hidden="true">
          <svg viewBox="0 0 24 24" width="22" height="22">
            <circle cx="6" cy="7" r="2.6" fill="currentColor" opacity=".55" />
            <circle cx="17.5" cy="6" r="2" fill="currentColor" opacity=".75" />
            <circle cx="13" cy="17" r="3.6" fill="currentColor" />
            <path d="M7.8 8.6l3.7 5.6M16.9 7.9l-2.6 5.6M8.5 6.7l7-.6" stroke="currentColor" strokeWidth="1.4" fill="none" />
          </svg>
        </div>
        <div>
          <h1>Repo-Level Technical Debt Quantification Agent</h1>
          <p className="header__subtitle">BITE497J Project I - VIT Vellore</p>
        </div>
      </div>

      <div className="header__repo" aria-label="Last analysed repository">
        {meta === undefined ? null : meta ? (
          <>
            <span className="header__repo-label">Last analysed</span>
            <a className="header__repo-name" href={meta.repo_url} target="_blank" rel="noreferrer">
              {meta.repo_name}
            </a>
            <span className="header__repo-detail mono" title={`${meta.commit} — ${meta.commit_subject}`}>
              @{meta.commit.slice(0, 8)} · {meta.scope}
            </span>
            <span className="header__repo-detail" title={new Date(meta.analysed_at).toLocaleString()}>
              {relativeTime(meta.analysed_at)} · took {Math.round(meta.duration_s)}s
            </span>
          </>
        ) : (
          <span className="header__repo-detail">No analysis metadata — run scripts/load_demo.py</span>
        )}
      </div>

      <div className="header__meta">
        <ul className="team" aria-label="Team">
          {TEAM.map((m) => (
            <li key={m.id}>
              <span className="team__id">{m.id}</span>
              {m.name}
            </li>
          ))}
        </ul>
        <div className="header__status" aria-label="Backend services">
          <StatusDot label="API" ok={unreachable ? false : h ? true : null} />
          <StatusDot label="Neo4j" ok={unreachable ? false : h ? h.neo4j : null} />
          <StatusDot label="Qdrant" ok={unreachable ? false : h ? h.qdrant : null} />
          <StatusDot
            label="LLM"
            ok={llmOk}
            title={llm ? `${llm.model ?? "LLM"}: ${llm.status}${llm.error ? ` (${llm.error})` : ""}` : undefined}
          />
        </div>
        <button
          type="button"
          className={`btn ${presenting ? "btn--primary" : "btn--ghost"}`}
          onClick={onTogglePresenting}
          aria-pressed={presenting}
        >
          <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">
            <rect x="1.5" y="2.5" width="13" height="8.5" rx="1.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
            <path d="M8 11v2.5M5 13.5h6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
          {presenting ? "Exit presentation" : "Present"}
        </button>
      </div>
    </header>
  );
}
