import { useId, useMemo } from "react";
import { motion } from "framer-motion";
import type { DebtNode } from "../api";
import {
  EFFORT_BAND_LABEL,
  PRIORITY_FORMULA,
  effortBand,
  effortOf,
  locOf,
  priorityOf,
  whyFragments,
  type GraphIndex,
  type RepoBaselines,
} from "../lib/analysis";
import { formatDiv } from "../lib/scales";

type Props = {
  functions: DebtNode[];
  index: GraphIndex;
  base: RepoBaselines;
  colorOf: (div: number) => string;
  selectedId: string | null;
  onSelect: (node: DebtNode) => void;
  limit?: number;
};

export function rankByPriority(functions: DebtNode[]): DebtNode[] {
  return [...functions]
    .filter((n) => n.div_score > 0)
    .sort((a, b) => priorityOf(b) - priorityOf(a) || b.div_score - a.div_score);
}

export function RefactorPlan({ functions, index, base, colorOf, selectedId, onSelect, limit = 10 }: Props) {
  const tipId = useId();
  const rows = useMemo(() => rankByPriority(functions).slice(0, limit), [functions, limit]);
  const maxPriority = rows.length ? priorityOf(rows[0]) : 0;

  if (!rows.length) return <p className="muted">No functions have a DIV score yet.</p>;

  return (
    <div className="plan">
      <div className="plan__formula">
        <span className="muted small">Ordered by impact per effort</span>
        <span className="info">
          <button type="button" className="info__btn" aria-describedby={tipId} aria-label="How priority is calculated">
            ?
          </button>
          <span role="tooltip" id={tipId} className="info__tip">
            <code>{PRIORITY_FORMULA}</code>
            <br />
            Impact is the function’s DIV. Effort is an <strong>estimate</strong> of change cost from complexity and
            length, not a measurement. Sizes: S &lt; 5, M &lt; 15, L ≥ 15 effort points.
          </span>
        </span>
      </div>

      <ol className="plan__list">
        {rows.map((n, i) => {
          const effort = effortOf(n);
          const band = effortBand(effort);
          const why = whyFragments(n, index, base);
          const p = priorityOf(n);
          const selected = n.id === selectedId;
          return (
            <motion.li
              key={n.id}
              initial={{ opacity: 0, y: 10 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.2 }}
              transition={{ duration: 0.35, delay: i * 0.04, ease: [0.22, 1, 0.36, 1] }}
            >
              <button
                type="button"
                className={`plan__row ${selected ? "is-selected" : ""}`}
                onClick={() => onSelect(n)}
                aria-pressed={selected}
              >
                <span className={`plan__rank mono ${i === 0 ? "is-first" : ""}`}>{i + 1}</span>
                <span className="plan__main">
                  <span className="plan__name">
                    {n.name}
                    <span className="plan__file">{n.file_path}</span>
                  </span>
                  <span className="plan__why">
                    <span className="plan__why-label">why</span> {why.length ? why.join(" + ") : "debt from its own metrics"}
                  </span>
                </span>
                <span className="plan__nums">
                  <span className="plan__score">
                    <span className="mono">{p.toFixed(1)}</span>
                    <span className="bar" aria-hidden="true">
                      <motion.span
                        className="bar__fill"
                        style={{ background: colorOf(n.div_score), originX: 0 }}
                        initial={{ scaleX: 0 }}
                        whileInView={{ scaleX: maxPriority > 0 ? Math.max(0.02, p / maxPriority) : 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.8, delay: 0.1 + i * 0.04, ease: [0.22, 1, 0.36, 1] }}
                      />
                    </span>
                  </span>
                  <span className="plan__meta mono">
                    DIV {formatDiv(n.div_score, 1)} · cc {n.cyclomatic_complexity} · {locOf(n)} LOC
                  </span>
                </span>
                <span
                  className={`effort effort--${band}`}
                  title={`Estimated effort ${effort.toFixed(1)} points (complexity ${n.cyclomatic_complexity} + ${locOf(n)} LOC ÷ 10)`}
                >
                  <span className="effort__tag">estimate</span>
                  {band} · {EFFORT_BAND_LABEL[band]}
                </span>
              </button>
            </motion.li>
          );
        })}
      </ol>
    </div>
  );
}
