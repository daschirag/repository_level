import * as d3 from "d3";

/** Cool blue -> amber -> red. Mirrors --div-low / --div-mid / --div-high in styles.css. */
export const DIV_STOPS = ["#38bdf8", "#fbbf24", "#f43f5e"] as const;

const interpolate = d3.interpolateRgbBasis([...DIV_STOPS]);

/** Colour for a DIV score relative to the repo maximum (sqrt spreads the low end). */
export function makeDivColor(maxDiv: number): (div: number) => string {
  const scale = d3.scaleSequentialSqrt(interpolate).domain([0, maxDiv > 0 ? maxDiv : 1]);
  return (div) => scale(Math.max(0, div || 0));
}

/** Node radius by DIV. */
export function makeDivRadius(maxDiv: number): (div: number) => number {
  const scale = d3
    .scaleSqrt()
    .domain([0, maxDiv > 0 ? maxDiv : 1])
    .range([5, 22]);
  return (div) => scale(Math.max(0, div || 0));
}

export type Severity = "critical" | "high" | "medium" | "low";

/** Bucket a DIV score by its share of the repo maximum. */
export function severityOf(div: number, maxDiv: number): Severity {
  if (maxDiv <= 0 || div <= 0) return "low";
  const r = div / maxDiv;
  if (r >= 0.75) return "critical";
  if (r >= 0.45) return "high";
  if (r >= 0.2) return "medium";
  return "low";
}

export function formatDiv(div: number | null | undefined, digits = 2): string {
  return Number(div ?? 0).toFixed(digits);
}
