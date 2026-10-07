import type { HealthResponse } from "../api";

const TEAM = [
  { id: "23BIT0198", name: "Minmini K" },
  { id: "23BIT0185", name: "Tarunica S" },
];

function StatusDot({ label, ok }: { label: string; ok: boolean | null }) {
  const state = ok == null ? "unknown" : ok ? "ok" : "down";
  const text = ok == null ? "checking" : ok ? "online" : "offline";
  return (
    <span className={`status status--${state}`} title={`${label} ${text}`}>
      <span className="status__dot" aria-hidden="true" />
      {label}
      <span className="sr-only"> {text}</span>
    </span>
  );
}

export function Header({ health }: { health: HealthResponse | null | "error" }) {
  const h = health === "error" ? null : health;
  const unreachable = health === "error";
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
        </div>
      </div>
    </header>
  );
}
