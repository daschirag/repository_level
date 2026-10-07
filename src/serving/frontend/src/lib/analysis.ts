/**
 * Pure analysis over the graph returned by /api/debt-graph.
 *
 * Every number the story UI shows is either an API value or derived here by a
 * formula written out below (and surfaced verbatim in tooltips).
 */
import type { DebtEdge, DebtNode } from "../api";

// ---------------------------------------------------------------------------
// Graph index
// ---------------------------------------------------------------------------

export type GraphIndex = {
  byId: Map<string, DebtNode>;
  /** callee id -> direct caller ids */
  callers: Map<string, Set<string>>;
  /** caller id -> direct callee ids */
  callees: Map<string, Set<string>>;
};

export function buildIndex(functions: DebtNode[], callEdges: DebtEdge[]): GraphIndex {
  const byId = new Map(functions.map((n) => [n.id, n]));
  const callers = new Map<string, Set<string>>();
  const callees = new Map<string, Set<string>>();
  for (const e of callEdges) {
    const s = String(e.source);
    const t = String(e.target);
    if (s === t || !byId.has(s) || !byId.has(t)) continue;
    if (!callers.has(t)) callers.set(t, new Set());
    if (!callees.has(s)) callees.set(s, new Set());
    callers.get(t)!.add(s);
    callees.get(s)!.add(t);
  }
  return { byId, callers, callees };
}

/** Lines of code from the AST span (0 when the span is unknown). */
export function locOf(n: DebtNode): number {
  return n.start_line && n.end_line ? Math.max(1, n.end_line - n.start_line + 1) : 0;
}

// ---------------------------------------------------------------------------
// Blast radius: everything that transitively calls a function
// ---------------------------------------------------------------------------

export type BlastRadius = {
  rootId: string;
  /** caller id -> hop distance from the root (1 = direct caller) */
  depth: Map<string, number>;
  functions: number;
  files: number;
};

/** BFS over reverse CALLS edges (unbounded depth, each caller counted once). */
export function blastRadius(index: GraphIndex, rootId: string): BlastRadius {
  const depth = new Map<string, number>();
  const queue: string[] = [rootId];
  const seen = new Set([rootId]);
  while (queue.length) {
    const id = queue.shift()!;
    const d = id === rootId ? 0 : depth.get(id)!;
    for (const caller of index.callers.get(id) ?? []) {
      if (seen.has(caller)) continue;
      seen.add(caller);
      depth.set(caller, d + 1);
      queue.push(caller);
    }
  }
  const files = new Set([...depth.keys()].map((id) => index.byId.get(id)?.file_path));
  return { rootId, depth, functions: depth.size, files: files.size };
}

// ---------------------------------------------------------------------------
// DIV decomposition (mirrors src/scoring/div_propagation.py)
// ---------------------------------------------------------------------------

/** base(v) = cyclomatic_complexity * ln(1 + change_frequency) */
export function ownDebt(n: DebtNode): number {
  return (n.cyclomatic_complexity || 0) * Math.log(1 + Math.max(0, n.change_frequency || 0));
}

/** DIV = own base weight + debt inherited (decayed) from transitive callers. */
export function inheritedDebt(n: DebtNode): number {
  return Math.max(0, (n.div_score || 0) - ownDebt(n));
}

export const DIV_FORMULA =
  "DIV(v) = base(v) + Σ base(u)·0.7^dist(u,v) over transitive callers u (≤ 6 hops); base = complexity × ln(1 + file churn)";

// ---------------------------------------------------------------------------
// Rankings
// ---------------------------------------------------------------------------

/** Rank by raw cyclomatic complexity (the "flat metric"), ties broken by LOC then name. */
export function rankByComplexity(functions: DebtNode[]): DebtNode[] {
  return [...functions].sort(
    (a, b) =>
      b.cyclomatic_complexity - a.cyclomatic_complexity ||
      locOf(b) - locOf(a) ||
      a.name.localeCompare(b.name),
  );
}

export function rankByDiv(functions: DebtNode[]): DebtNode[] {
  return [...functions].sort((a, b) => b.div_score - a.div_score || a.name.localeCompare(b.name));
}

// ---------------------------------------------------------------------------
// Refactor order: impact per effort
// ---------------------------------------------------------------------------

/** effort = complexity + LOC / 10   (an estimate of change cost, not a measurement) */
export function effortOf(n: DebtNode): number {
  return Math.max(1, (n.cyclomatic_complexity || 0) + locOf(n) / 10);
}

/** priority = DIV / effort   (debt removed per unit of estimated work) */
export function priorityOf(n: DebtNode): number {
  return (n.div_score || 0) / effortOf(n);
}

export const PRIORITY_FORMULA = "priority = DIV ÷ effort,  effort = complexity + LOC ÷ 10";

export type EffortBand = "S" | "M" | "L";

/** Rough size band for the effort estimate: S < 5, M < 15, otherwise L. */
export function effortBand(effort: number): EffortBand {
  if (effort < 5) return "S";
  if (effort < 15) return "M";
  return "L";
}

export const EFFORT_BAND_LABEL: Record<EffortBand, string> = {
  S: "small",
  M: "medium",
  L: "large",
};

// ---------------------------------------------------------------------------
// Plain-English "why"
// ---------------------------------------------------------------------------

/** p-th percentile (nearest-rank) of a numeric list. */
export function percentile(values: number[], p: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1));
  return sorted[idx];
}

export type RepoBaselines = {
  churnP75: number;
  complexityP75: number;
  locP75: number;
};

export function baselines(functions: DebtNode[]): RepoBaselines {
  return {
    churnP75: percentile(functions.map((n) => n.change_frequency || 0), 0.75),
    complexityP75: percentile(functions.map((n) => n.cyclomatic_complexity || 0), 0.75),
    locP75: percentile(functions.map(locOf), 0.75),
  };
}

/** Short reason fragments, strongest first, e.g. "high churn + 9 callers + complexity 16". */
export function whyFragments(
  n: DebtNode,
  index: GraphIndex,
  base: RepoBaselines,
  blast?: BlastRadius,
): string[] {
  const out: string[] = [];
  const direct = index.callers.get(n.id)?.size ?? 0;
  const transitive = blast?.functions ?? blastRadius(index, n.id).functions;
  if (base.churnP75 > 0 && n.change_frequency >= base.churnP75) out.push(`high churn (${n.change_frequency} commits to its file)`);
  if (transitive > 0) {
    out.push(
      transitive > direct
        ? `${direct} caller${direct === 1 ? "" : "s"} (${transitive} transitive)`
        : `${direct} caller${direct === 1 ? "" : "s"}`,
    );
  }
  if (n.cyclomatic_complexity > 0 && n.cyclomatic_complexity >= base.complexityP75)
    out.push(`complexity ${n.cyclomatic_complexity}`);
  const loc = locOf(n);
  if (loc > 0 && loc >= base.locP75 && loc > 30) out.push(`${loc} lines`);
  return out;
}
