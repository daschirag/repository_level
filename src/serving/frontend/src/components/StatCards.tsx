import { motion } from "framer-motion";
import { useCountUp } from "../lib/hooks";
import { Skeleton } from "./ui";

export type Stats = {
  nodeCount: number;
  edgeCount: number;
  functionCount: number;
  callCount: number;
  importCount: number;
  maxDiv: number;
  maxDivName: string | null;
  avgDiv: number;
};

type CardProps = {
  label: string;
  value: number;
  decimals?: number;
  hint: string;
  accent?: boolean;
  index: number;
};

function StatCard({ label, value, decimals = 0, hint, accent, index }: CardProps) {
  const shown = useCountUp(value);
  return (
    <motion.div
      className={`card stat ${accent ? "stat--accent" : ""}`}
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: 0.06 * index, ease: [0.22, 1, 0.36, 1] }}
    >
      <span className="stat__label">{label}</span>
      <span className="stat__value" aria-label={`${label}: ${value.toFixed(decimals)}`}>
        {shown.toLocaleString(undefined, {
          minimumFractionDigits: decimals,
          maximumFractionDigits: decimals,
        })}
      </span>
      <span className="stat__hint" title={hint}>
        {hint}
      </span>
    </motion.div>
  );
}

export function StatCards({ stats }: { stats: Stats | null }) {
  if (!stats) {
    return (
      <section className="stats" aria-label="Loading summary" aria-busy="true">
        {[0, 1, 2, 3].map((i) => (
          <div className="card stat" key={i}>
            <Skeleton width={90} height={12} />
            <Skeleton width={110} height={30} radius={8} />
            <Skeleton width={140} height={12} />
          </div>
        ))}
      </section>
    );
  }
  return (
    <section className="stats" aria-label="Summary">
      <StatCard
        index={0}
        label="Total nodes"
        value={stats.nodeCount}
        hint={`${stats.functionCount.toLocaleString()} functions`}
      />
      <StatCard
        index={1}
        label="Edges"
        value={stats.edgeCount}
        hint={`${stats.callCount.toLocaleString()} calls · ${stats.importCount.toLocaleString()} imports`}
      />
      <StatCard
        index={2}
        label="Highest DIV"
        value={stats.maxDiv}
        decimals={2}
        hint={stats.maxDivName ?? "—"}
        accent
      />
      <StatCard
        index={3}
        label="Average DIV"
        value={stats.avgDiv}
        decimals={3}
        hint={`across ${stats.functionCount.toLocaleString()} functions`}
      />
    </section>
  );
}
