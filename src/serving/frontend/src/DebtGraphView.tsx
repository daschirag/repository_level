import { useEffect, useMemo, useRef } from "react";
import * as d3 from "d3";
import type { DebtEdge, DebtNode } from "./api";

type GraphProps = {
  nodes: DebtNode[];
  edges: DebtEdge[];
  selectedId: string | null;
  onSelect: (node: DebtNode) => void;
};

type SimNode = DebtNode & d3.SimulationNodeDatum;
type SimLink = d3.SimulationLinkDatum<SimNode> & { type: string };

export function DebtGraphView({ nodes, edges, selectedId, onSelect }: GraphProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  const maxDiv = useMemo(
    () => Math.max(1, ...nodes.map((n) => n.div_score || 0)),
    [nodes],
  );

  useEffect(() => {
    const svgEl = svgRef.current;
    const wrap = wrapRef.current;
    if (!svgEl || !wrap) return;

    const width = wrap.clientWidth || 800;
    const height = wrap.clientHeight || 560;

    const color = d3
      .scaleSequential(d3.interpolateRdYlGn)
      .domain([maxDiv, 0]); // high debt = red, low = green

    const size = d3
      .scaleSqrt()
      .domain([0, maxDiv])
      .range([4, 22]);

    const simNodes: SimNode[] = nodes.map((n) => ({ ...n }));
    const idSet = new Set(simNodes.map((n) => n.id));
    const simLinks: SimLink[] = edges
      .filter((e) => idSet.has(e.source) && idSet.has(e.target))
      .map((e) => ({
        source: e.source,
        target: e.target,
        type: e.type,
      }));

    const svg = d3.select(svgEl);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const g = svg.append("g");

    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 4])
      .on("zoom", (event) => {
        g.attr("transform", event.transform.toString());
      });
    svg.call(zoom);

    const link = g
      .append("g")
      .attr("stroke", "#9aa3ad")
      .attr("stroke-opacity", 0.45)
      .selectAll("line")
      .data(simLinks)
      .join("line")
      .attr("stroke-width", (d) => (d.type === "CALLS" ? 1.2 : 0.6))
      .attr("stroke-dasharray", (d) => (d.type === "IMPORTS" ? "3,3" : null));

    const node = g
      .append("g")
      .selectAll("circle")
      .data(simNodes)
      .join("circle")
      .attr("r", (d) => size(d.div_score || 0))
      .attr("fill", (d) =>
        d.node_kind === "Function" ? color(d.div_score || 0) : "#94a3b8",
      )
      .attr("stroke", (d) => (d.id === selectedId ? "#0f172a" : "#fff"))
      .attr("stroke-width", (d) => (d.id === selectedId ? 2.5 : 1))
      .style("cursor", "pointer")
      .on("click", (_event, d) => onSelect(d));

    node.append("title").text(
      (d) => `${d.name}\n${d.file_path}\nDIV=${(d.div_score || 0).toFixed(2)}`,
    );

    const simulation = d3
      .forceSimulation(simNodes)
      .force(
        "link",
        d3
          .forceLink<SimNode, SimLink>(simLinks)
          .id((d) => d.id)
          .distance(40)
          .strength(0.4),
      )
      .force("charge", d3.forceManyBody().strength(-28))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide<SimNode>().radius((d) => size(d.div_score || 0) + 2));

    simulation.on("tick", () => {
      link
        .attr("x1", (d) => (d.source as SimNode).x ?? 0)
        .attr("y1", (d) => (d.source as SimNode).y ?? 0)
        .attr("x2", (d) => (d.target as SimNode).x ?? 0)
        .attr("y2", (d) => (d.target as SimNode).y ?? 0);
      node.attr("cx", (d) => d.x ?? 0).attr("cy", (d) => d.y ?? 0);
    });

    return () => {
      simulation.stop();
    };
  }, [nodes, edges, maxDiv, selectedId, onSelect]);

  return (
    <div className="graph-wrap" ref={wrapRef}>
      <svg ref={svgRef} className="graph-svg" />
    </div>
  );
}
