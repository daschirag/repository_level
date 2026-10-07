import { motion } from "framer-motion";
import type { TopDebtNode } from "../api";
import { formatDiv } from "../lib/scales";
import { Skeleton } from "./ui";

type Props = {
  nodes: TopDebtNode[] | null;
  maxDiv: number;
  colorOf: (div: number) => string;
  selectedId: string | null;
  onSelect: (node: TopDebtNode) => void;
};

export function TopDebtList({ nodes, maxDiv, colorOf, selectedId, onSelect }: Props) {
  return (
    <div className="ranked">
      <div className="panel__head">
        <h2>Top Debt Nodes</h2>
        {nodes && <span className="pill">{nodes.length} ranked by DIV</span>}
      </div>

      {!nodes && (
        <ul className="ranked__list" aria-busy="true" aria-label="Loading top debt nodes">
          {Array.from({ length: 8 }, (_, i) => (
            <li key={i} className="ranked__skeleton">
              <Skeleton width={22} height={22} radius={6} />
              <div className="ranked__skeleton-body">
                <Skeleton width={`${70 - i * 4}%`} height={12} />
                <Skeleton width="100%" height={6} radius={3} />
              </div>
            </li>
          ))}
        </ul>
      )}

      {nodes && nodes.length === 0 && (
        <p className="muted ranked__empty">No functions have a DIV score yet.</p>
      )}

      {nodes && nodes.length > 0 && (
        <ol className="ranked__list">
          {nodes.map((n, i) => {
            const share = maxDiv > 0 ? n.div_score / maxDiv : 0;
            const selected = n.id === selectedId;
            return (
              <motion.li
                key={n.id}
                initial={{ opacity: 0, x: 12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.35, delay: 0.25 + i * 0.035, ease: [0.22, 1, 0.36, 1] }}
              >
                <button
                  type="button"
                  className={`ranked__row ${selected ? "is-selected" : ""}`}
                  onClick={() => onSelect(n)}
                  aria-pressed={selected}
                  aria-label={`Rank ${i + 1}: ${n.name} in ${n.file_path}, DIV ${formatDiv(n.div_score)}`}
                >
                  <span className="ranked__rank mono">{i + 1}</span>
                  <span className="ranked__body">
                    <span className="ranked__top">
                      <span className="ranked__name">{n.name}</span>
                      <span className="ranked__div mono" style={{ color: colorOf(n.div_score) }}>
                        {formatDiv(n.div_score)}
                      </span>
                    </span>
                    <span className="ranked__file">{n.file_path}</span>
                    <span className="bar" aria-hidden="true">
                      <motion.span
                        className="bar__fill"
                        style={{ background: colorOf(n.div_score), originX: 0 }}
                        initial={{ scaleX: 0 }}
                        animate={{ scaleX: Math.max(share, 0.015) }}
                        transition={{ duration: 0.8, delay: 0.35 + i * 0.035, ease: [0.22, 1, 0.36, 1] }}
                      />
                    </span>
                  </span>
                </button>
              </motion.li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
