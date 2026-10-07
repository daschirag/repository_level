import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import type { DebtNode } from "../api";
import {
  blastRadius,
  inheritedDebt,
  rankByComplexity,
  rankByDiv,
  type GraphIndex,
} from "../lib/analysis";
import { formatDiv } from "../lib/scales";

type Props = {
  functions: DebtNode[];
  index: GraphIndex;
  colorOf: (div: number) => string;
  onSelect: (node: DebtNode) => void;
  topN?: number;
};

const ROW_H = 46;
const GUTTER_W = 140;

export function ComparisonPanel({ functions, index, colorOf, onSelect, topN = 10 }: Props) {
  const [hovered, setHovered] = useState<string | null>(null);

  const data = useMemo(() => {
    const byCc = rankByComplexity(functions);
    const byDiv = rankByDiv(functions);
    // Competition rank (1 + functions strictly more complex) so ties don't invent rank gaps.
    const ccRank = new Map(
      byCc.map((n) => [n.id, 1 + functions.filter((o) => o.cyclomatic_complexity > n.cyclomatic_complexity).length]),
    );
    const divRank = new Map(byDiv.map((n, i) => [n.id, i + 1]));
    const left = byCc.slice(0, topN);
    const right = byDiv.slice(0, topN);
    const leftIds = new Set(left.map((n) => n.id));
    const newcomers = right.filter((n) => !leftIds.has(n.id));
    const callers = new Map(right.map((n) => [n.id, blastRadius(index, n.id).functions]));
    // Biggest climber: largest complexity-rank -> DIV-rank jump among the DIV top N.
    const climber = [...right].sort(
      (a, b) => ccRank.get(b.id)! - divRank.get(b.id)! - (ccRank.get(a.id)! - divRank.get(a.id)!),
    )[0];
    return { left, right, leftIds, ccRank, divRank, newcomers, callers, climber };
  }, [functions, index, topN]);

  if (functions.length < 2) return null;
  const { left, right, leftIds, ccRank, divRank, newcomers, callers, climber } = data;
  const height = topN * ROW_H;
  const links = left
    .map((n, i) => ({ n, i, j: right.findIndex((r) => r.id === n.id) }))
    .filter((l) => l.j >= 0);
  const climberJump = climber ? ccRank.get(climber.id)! - divRank.get(climber.id)! : 0;

  return (
    <div className="compare">
      <p className="compare__takeaway">
        <strong className="compare__big mono">
          {newcomers.length} of the top {right.length}
        </strong>{" "}
        functions by DIV are <em>not</em> in the top {topN} by complexity
        {climber && climberJump > 0 && (
          <>
            {" "}
            — <code>{climber.name}</code> ranks #{ccRank.get(climber.id)} by complexity but #
            {divRank.get(climber.id)} by DIV, because {callers.get(climber.id)} functions depend on it.
          </>
        )}
      </p>

      <div className="compare__grid" style={{ gridTemplateColumns: `minmax(0,1fr) ${GUTTER_W}px minmax(0,1fr)` }}>
        <h3 className="compare__head">
          Flat metric <span className="muted">· cyclomatic complexity</span>
          <span className="compare__hint">what SonarQube-style tools rank by</span>
        </h3>
        <span />
        <h3 className="compare__head">
          Causal impact <span className="muted">· DIV</span>
          <span className="compare__hint">complexity × churn, propagated through callers</span>
        </h3>

        <ol className="compare__list">
          {left.map((n, i) => {
            const dr = divRank.get(n.id)!;
            const off = dr > topN;
            return (
              <li key={n.id} style={{ height: ROW_H }}>
                <button
                  type="button"
                  className={`compare__row ${hovered === n.id ? "is-hover" : ""} ${off ? "is-off" : ""}`}
                  onMouseEnter={() => setHovered(n.id)}
                  onMouseLeave={() => setHovered(null)}
                  onFocus={() => setHovered(n.id)}
                  onBlur={() => setHovered(null)}
                  onClick={() => onSelect(n)}
                >
                  <span className="compare__rank mono">{i + 1}</span>
                  <span className="compare__name">
                    {n.name}
                    <span className="compare__file">{n.file_path}</span>
                  </span>
                  <span className="compare__val mono">cc {n.cyclomatic_complexity}</span>
                  {off && <span className="compare__away mono">DIV #{dr}</span>}
                </button>
              </li>
            );
          })}
        </ol>

        <svg className="compare__links" width={GUTTER_W} height={height} aria-hidden="true">
          {links.map(({ n, i, j }, k) => {
            const y1 = i * ROW_H + ROW_H / 2;
            const y2 = j * ROW_H + ROW_H / 2;
            const d = `M0,${y1} C${GUTTER_W / 2},${y1} ${GUTTER_W / 2},${y2} ${GUTTER_W},${y2}`;
            const dim = hovered && hovered !== n.id;
            return (
              <motion.path
                key={n.id}
                d={d}
                fill="none"
                stroke={j < i ? "var(--accent-strong)" : "rgba(148,163,184,0.45)"}
                strokeWidth={hovered === n.id ? 2.6 : 1.6}
                initial={{ pathLength: 0, opacity: 0 }}
                whileInView={{ pathLength: 1, opacity: dim ? 0.15 : 1 }}
                animate={{ opacity: dim ? 0.15 : 1 }}
                viewport={{ once: true, amount: 0.3 }}
                transition={{ duration: 0.9, delay: 0.15 + k * 0.08, ease: [0.22, 1, 0.36, 1] }}
              />
            );
          })}
        </svg>

        <ol className="compare__list">
          {right.map((n, j) => {
            const cr = ccRank.get(n.id)!;
            const isNew = !leftIds.has(n.id);
            const inherited = inheritedDebt(n);
            const share = n.div_score > 0 ? Math.round((inherited / n.div_score) * 100) : 0;
            return (
              <li key={n.id} style={{ height: ROW_H }}>
                <button
                  type="button"
                  className={`compare__row ${hovered === n.id ? "is-hover" : ""} ${isNew ? "is-riser" : ""}`}
                  onMouseEnter={() => setHovered(n.id)}
                  onMouseLeave={() => setHovered(null)}
                  onFocus={() => setHovered(n.id)}
                  onBlur={() => setHovered(null)}
                  onClick={() => onSelect(n)}
                  title={
                    isNew
                      ? `Complexity rank #${cr}. ${share}% of its DIV is inherited from ${callers.get(n.id)} transitive callers.`
                      : undefined
                  }
                >
                  <span className="compare__rank mono">{j + 1}</span>
                  <span className="compare__name">
                    {n.name}
                    <span className="compare__file">
                      {isNew
                        ? `${callers.get(n.id)} dependant${callers.get(n.id) === 1 ? "" : "s"} · ${share}% inherited`
                        : n.file_path}
                    </span>
                  </span>
                  <span className="compare__val mono" style={{ color: colorOf(n.div_score) }}>
                    {formatDiv(n.div_score, 1)}
                  </span>
                  {isNew && (
                    <span className="compare__jump mono" aria-label={`up from complexity rank ${cr}`}>
                      ▲ from #{cr}
                    </span>
                  )}
                </button>
              </li>
            );
          })}
        </ol>
      </div>
      <p className="compare__foot muted small">
        Complexity ranks are competition ranks (tied functions share a rank); the list breaks ties by length, then name. Highlighted rows rank in the DIV top {topN} but not
        the complexity top {topN}; their debt comes mostly from the functions that call them.
      </p>
    </div>
  );
}
