import { useEffect, useId, useMemo, useRef, useState } from "react";
import * as d3 from "d3";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import type { DebtEdge, DebtNode } from "../api";
import type { BlastRadius } from "../lib/analysis";
import { DIV_STOPS, formatDiv } from "../lib/scales";

type SimNode = DebtNode & d3.SimulationNodeDatum & { r: number };
type SimLink = d3.SimulationLinkDatum<SimNode> & { type: string };

/**
 * Reveal phase driven by the story stepper:
 * 0 = empty, 1 = nodes appear (neutral), 2 = call edges draw in,
 * 3 = colours flood in by DIV (the finished heat-map).
 */
export type GraphPhase = 0 | 1 | 2 | 3;

type Props = {
  /** Function nodes only. */
  nodes: DebtNode[];
  /** CALLS edges between function ids. */
  edges: DebtEdge[];
  maxDiv: number;
  colorOf: (div: number) => string;
  radiusOf: (div: number) => number;
  selectedId: string | null;
  onSelect: (node: DebtNode) => void;
  phase: GraphPhase;
  /** Bumped to rebuild and replay the reveal from phase 0. */
  replayKey: number;
  /** Transitive callers of the selected node (highlighted as a ripple). */
  blast: BlastRadius | null;
  /** Node to ring during the "Prioritise" story step. */
  spotlightId: string | null;
};

type Tooltip = { node: DebtNode; x: number; y: number };

const LIMIT_OPTIONS = [50, 100, 250, 0] as const; // 0 = all
const ARROW_PAD = 3;
const NEUTRAL_FILL = "#5b6880";

function truncate(text: string, max: number) {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

export function GraphView({
  nodes,
  edges,
  maxDiv,
  colorOf,
  radiusOf,
  selectedId,
  onSelect,
  phase,
  replayKey,
  blast,
  spotlightId,
}: Props) {
  const reduceMotion = useReducedMotion() ?? false;
  const uid = useId().replace(/:/g, "");
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const [limit, setLimit] = useState<number>(100);
  const [query, setQuery] = useState("");
  const [activeMatch, setActiveMatch] = useState(0);
  const [tooltip, setTooltip] = useState<Tooltip | null>(null);

  // D3 handles shared between the build effect and the emphasis/zoom effects.
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const fitRef = useRef<() => void>(() => {});
  const advanceRef = useRef<(target: GraphPhase) => void>(() => {});
  const nodeSelRef = useRef<d3.Selection<SVGGElement, SimNode, SVGGElement, unknown> | null>(null);
  const linkSelRef = useRef<d3.Selection<SVGPathElement, SimLink, SVGGElement, unknown> | null>(null);
  const neighborsRef = useRef<Map<string, Set<string>>>(new Map());
  const simNodesRef = useRef<Map<string, SimNode>>(new Map());
  const hoverIdRef = useRef<string | null>(null);
  const matchesRef = useRef<Set<string> | null>(null);
  const selectedIdRef = useRef<string | null>(selectedId);
  const blastRef = useRef<BlastRadius | null>(blast);
  const spotlightRef = useRef<string | null>(spotlightId);
  const phaseRef = useRef<GraphPhase>(phase);
  const edgesShownRef = useRef(false);
  const selfSelectRef = useRef(false);
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;

  // Highest-DIV first, capped at ``limit``; edges restricted to the kept set.
  const visible = useMemo(() => {
    const ranked = [...nodes].sort((a, b) => (b.div_score || 0) - (a.div_score || 0));
    const kept = limit > 0 ? ranked.slice(0, limit) : ranked;
    const ids = new Set(kept.map((n) => n.id));
    const keptEdges = edges.filter(
      (e) => e.source !== e.target && ids.has(String(e.source)) && ids.has(String(e.target)),
    );
    return { nodes: kept, edges: keptEdges };
  }, [nodes, edges, limit]);

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    return visible.nodes
      .filter((n) => n.name.toLowerCase().includes(q) || n.file_path.toLowerCase().includes(q))
      .slice(0, 8);
  }, [query, visible.nodes]);

  /** Emphasis precedence: hover > search > blast radius; plus selection/spotlight rings. */
  const applyEmphasis = useRef(() => {});
  applyEmphasis.current = () => {
    const nodeSel = nodeSelRef.current;
    const linkSel = linkSelRef.current;
    if (!nodeSel || !linkSel) return;
    const hover = hoverIdRef.current;
    const found = matchesRef.current;
    const radius = !hover && !found ? blastRef.current : null;
    const inBlast = (id: string) => !!radius && (id === radius.rootId || radius.depth.has(id));
    const keep = hover ? new Set([hover, ...(neighborsRef.current.get(hover) ?? [])]) : found;

    nodeSel
      .classed("is-dim", (d) => (keep ? !keep.has(d.id) : radius ? !inBlast(d.id) : false))
      .classed("is-match", (d) => !hover && !!found?.has(d.id))
      .classed("is-selected", (d) => d.id === selectedIdRef.current)
      .classed("is-spotlight", (d) => d.id === spotlightRef.current)
      .classed("is-blast", (d) => !!radius && radius.depth.has(d.id))
      .style("--ripple-delay", (d) => (radius?.depth.has(d.id) ? `${(radius.depth.get(d.id)! - 1) * 0.18}s` : null));

    const showArrows = edgesShownRef.current;
    linkSel.each(function (d) {
      const s = (d.source as SimNode).id;
      const t = (d.target as SimNode).id;
      const out = !!hover && s === hover;
      const inc = !!hover && t === hover;
      const blastLink = !!radius && inBlast(s) && inBlast(t);
      const dim = hover ? !out && !inc : found ? true : radius ? !blastLink : false;
      d3.select(this)
        .classed("is-out", out)
        .classed("is-in", inc)
        .classed("is-blast", blastLink)
        .classed("is-dim", dim)
        .attr(
          "marker-end",
          showArrows ? `url(#${uid}-arrow${out ? "-out" : inc ? "-in" : blastLink ? "-blast" : ""})` : null,
        );
    });

    // "N affected" badge on the blast root.
    nodeSel.selectAll(".blast-badge").remove();
    if (radius && phaseRef.current >= 3) {
      nodeSel
        .filter((d) => d.id === radius.rootId)
        .each(function (d) {
          const badge = d3
            .select(this)
            .append("g")
            .attr("class", "blast-badge")
            .attr("transform", `translate(0,${-(d.r + 20)})`);
          const rect = badge.append("rect").attr("rx", 9).attr("height", 18).attr("y", -9);
          const text = badge
            .append("text")
            .attr("dy", "0.35em")
            .text(`${radius.functions} affected`);
          const w = (text.node() as SVGTextElement).getComputedTextLength() + 16;
          rect.attr("width", w).attr("x", -w / 2);
          text.attr("text-anchor", "middle");
        });
    }
  };

  // ---- Build the force graph whenever the visible data changes (or on replay). ----
  useEffect(() => {
    const svgEl = svgRef.current;
    const container = containerRef.current;
    if (!svgEl || !container) return;

    let width = container.clientWidth || 800;
    let height = container.clientHeight || 560;
    const svg = d3.select(svgEl);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`);
    edgesShownRef.current = false;

    // Defs: glow filter + arrowheads (default / outgoing / incoming / blast).
    const defs = svg.append("defs");
    const glow = defs
      .append("filter")
      .attr("id", `${uid}-glow`)
      .attr("x", "-100%")
      .attr("y", "-100%")
      .attr("width", "300%")
      .attr("height", "300%");
    glow.append("feGaussianBlur").attr("stdDeviation", 6).attr("result", "blur");
    const merge = glow.append("feMerge");
    merge.append("feMergeNode").attr("in", "blur");
    merge.append("feMergeNode").attr("in", "SourceGraphic");
    for (const [suffix, cls] of [
      ["", "arrow"],
      ["-out", "arrow arrow--out"],
      ["-in", "arrow arrow--in"],
      ["-blast", "arrow arrow--blast"],
    ] as const) {
      defs
        .append("marker")
        .attr("id", `${uid}-arrow${suffix}`)
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 10)
        .attr("refY", 0)
        .attr("markerUnits", "userSpaceOnUse")
        .attr("markerWidth", 9)
        .attr("markerHeight", 9)
        .attr("orient", "auto")
        .append("path")
        .attr("class", cls)
        .attr("d", "M0,-4L10,0L0,4Z");
    }

    const root = svg.append("g");
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.15, 5])
      .on("zoom", (event) => {
        root.attr("transform", event.transform.toString());
        setTooltip(null);
      });
    svg.call(zoom).on("dblclick.zoom", null);
    zoomRef.current = zoom;

    // Dedupe ids (overloaded names in one file share an id server-side).
    const byId = new Map<string, SimNode>();
    visible.nodes.forEach((n, i) => {
      if (byId.has(n.id)) return;
      const angle = i * 2.39996;
      const radius = 9 * Math.sqrt(i + 0.5);
      byId.set(n.id, {
        ...n,
        r: radiusOf(n.div_score),
        x: width / 2 + radius * Math.cos(angle),
        y: height / 2 + radius * Math.sin(angle),
      });
    });
    const simNodes = [...byId.values()];
    simNodesRef.current = byId;
    const simLinks: SimLink[] = visible.edges.map((e) => ({
      source: String(e.source),
      target: String(e.target),
      type: e.type,
    }));

    const neighbors = new Map<string, Set<string>>();
    for (const l of simLinks) {
      const s = String(l.source);
      const t = String(l.target);
      if (!neighbors.has(s)) neighbors.set(s, new Set());
      if (!neighbors.has(t)) neighbors.set(t, new Set());
      neighbors.get(s)!.add(t);
      neighbors.get(t)!.add(s);
    }
    neighborsRef.current = neighbors;

    const ranked = [...simNodes].sort((a, b) => b.div_score - a.div_score);
    const hot = new Set(
      ranked
        .filter((n) => n.div_score > 0)
        .slice(0, Math.max(3, Math.ceil(simNodes.length * 0.03)))
        .map((n) => n.id),
    );
    const labelled =
      simNodes.length <= 40 ? new Set(simNodes.map((n) => n.id)) : new Set(ranked.slice(0, 15).map((n) => n.id));

    const link = root
      .append("g")
      .attr("class", "links")
      .selectAll<SVGPathElement, SimLink>("path")
      .data(simLinks)
      .join("path")
      .attr("class", "link")
      .style("opacity", 0);

    const node = root
      .append("g")
      .attr("class", "nodes")
      .selectAll<SVGGElement, SimNode>("g")
      .data(simNodes, (d) => d.id)
      .join("g")
      .attr("class", (d) => `node${hot.has(d.id) ? " is-hot" : ""}`)
      .attr("tabindex", 0)
      .attr("role", "button")
      .attr(
        "aria-label",
        (d) => `${d.name} in ${d.file_path}, DIV ${formatDiv(d.div_score)}. Press Enter for details.`,
      )
      .style("opacity", 0);

    node
      .append("circle")
      .attr("class", "node__ripple")
      .attr("r", (d) => d.r + 3);
    node
      .append("circle")
      .attr("class", "node__halo")
      .attr("r", (d) => d.r + 5);
    const circle = node
      .append("circle")
      .attr("class", "node__body")
      .attr("r", 0)
      .attr("fill", NEUTRAL_FILL);
    node
      .filter((d) => labelled.has(d.id))
      .append("text")
      .attr("class", "node__label")
      .attr("dy", (d) => d.r + 13)
      .text((d) => truncate(d.name, 24));

    nodeSelRef.current = node;
    linkSelRef.current = link;

    // ---- Phase-driven reveal ----
    let applied: GraphPhase = 0;
    const showEdgesNow = () => {
      edgesShownRef.current = true;
      applyEmphasis.current();
    };
    const advanceTo = (target: GraphPhase) => {
      if (target <= applied) return;
      if (reduceMotion) {
        if (target >= 1) {
          node.style("opacity", 1);
          circle.attr("r", (d) => d.r);
        }
        if (target >= 2) {
          link.style("opacity", 1);
          showEdgesNow();
        }
        if (target >= 3) {
          circle
            .attr("fill", (d) => colorOf(d.div_score))
            .attr("filter", (d) => (hot.has(d.id) ? `url(#${uid}-glow)` : null));
        }
        applied = target;
        applyEmphasis.current();
        return;
      }

      let offset = 0;
      if (target >= 1 && applied < 1) {
        const stagger = Math.min(12, 700 / Math.max(1, simNodes.length));
        node
          .transition("appear")
          .delay((_, i) => i * stagger)
          .duration(450)
          .style("opacity", 1);
        circle
          .transition("grow")
          .delay((_, i) => i * stagger)
          .duration(600)
          .ease(d3.easeBackOut.overshoot(1.6))
          .attr("r", (d) => d.r);
        offset += 700;
      }
      if (target >= 2 && applied < 2) {
        link
          .style("opacity", 1)
          .attr("pathLength", 1)
          .attr("stroke-dasharray", 1)
          .attr("stroke-dashoffset", 1)
          .transition("draw")
          .delay((_, i) => offset + Math.min(i * 6, 500))
          .duration(800)
          .ease(d3.easeCubicOut)
          .attr("stroke-dashoffset", 0)
          .on("end", function () {
            d3.select(this).attr("stroke-dasharray", null).attr("pathLength", null);
          });
        // Arrowheads once the lines have (mostly) drawn.
        window.setTimeout(showEdgesNow, offset + 700);
        offset += 900;
      }
      if (target >= 3 && applied < 3) {
        // Flood: hottest nodes light up first.
        circle
          .transition("flood")
          .delay((d) => offset + (1 - (maxDiv > 0 ? d.div_score / maxDiv : 0)) * 900)
          .duration(550)
          .attr("fill", (d) => colorOf(d.div_score))
          .on("end", function (d) {
            if (hot.has(d.id)) d3.select(this).attr("filter", `url(#${uid}-glow)`);
          });
        window.setTimeout(() => applyEmphasis.current(), offset + 1500);
      }
      applied = target;
    };
    advanceRef.current = advanceTo;

    const ticked = () => {
      link.attr("d", (d) => {
        const s = d.source as SimNode;
        const t = d.target as SimNode;
        const dx = (t.x ?? 0) - (s.x ?? 0);
        const dy = (t.y ?? 0) - (s.y ?? 0);
        const dist = Math.hypot(dx, dy) || 1;
        const ux = dx / dist;
        const uy = dy / dist;
        const sx = (s.x ?? 0) + ux * s.r;
        const sy = (s.y ?? 0) + uy * s.r;
        const tx = (t.x ?? 0) - ux * (t.r + ARROW_PAD);
        const ty = (t.y ?? 0) - uy * (t.r + ARROW_PAD);
        return `M${sx},${sy}L${tx},${ty}`;
      });
      node.attr("transform", (d) => `translate(${d.x ?? 0},${d.y ?? 0})`);
    };

    const fit = (animate: boolean) => {
      if (!simNodes.length) return;
      const xs = simNodes.map((n) => n.x ?? 0);
      const ys = simNodes.map((n) => n.y ?? 0);
      const pad = 72; // room for labels around the outermost nodes
      const x0 = Math.min(...xs) - pad;
      const x1 = Math.max(...xs) + pad;
      const y0 = Math.min(...ys) - pad;
      const y1 = Math.max(...ys) + pad;
      const scale = Math.min(1.6, 0.95 * Math.min(width / (x1 - x0), height / (y1 - y0)));
      const transform = d3.zoomIdentity
        .translate(width / 2, height / 2)
        .scale(scale)
        .translate(-(x0 + x1) / 2, -(y0 + y1) / 2);
      if (animate) svg.transition().duration(750).ease(d3.easeCubicInOut).call(zoom.transform, transform);
      else svg.call(zoom.transform, transform);
    };
    fitRef.current = () => fit(!reduceMotion);

    const sim = d3
      .forceSimulation<SimNode>(simNodes)
      .force(
        "link",
        d3
          .forceLink<SimNode, SimLink>(simLinks)
          .id((d) => d.id)
          .distance((l) => 46 + (l.source as SimNode).r + (l.target as SimNode).r)
          .strength(0.5),
      )
      .force("charge", d3.forceManyBody<SimNode>().strength((d) => -70 - d.r * 9))
      .force("collide", d3.forceCollide<SimNode>().radius((d) => d.r + 6))
      .force("x", d3.forceX<SimNode>(width / 2).strength(0.06))
      .force("y", d3.forceY<SimNode>(height / 2).strength(0.06))
      .alphaDecay(0.028);

    let fitted = false;
    if (reduceMotion) {
      sim.stop();
      for (let i = 0; i < 300; i += 1) sim.tick();
      ticked();
      fit(false);
      fitted = true;
      sim.on("tick", ticked);
    } else {
      sim.on("tick", () => {
        ticked();
        if (!fitted && sim.alpha() < 0.12) {
          fitted = true;
          fit(true);
        }
      });
    }

    const showTooltipFor = (d: SimNode, el: Element) => {
      const box = el.getBoundingClientRect();
      const host = container.getBoundingClientRect();
      setTooltip({ node: d, x: box.left + box.width / 2 - host.left, y: box.top - host.top });
    };

    node
      .on("mouseenter", function (_event, d) {
        hoverIdRef.current = d.id;
        applyEmphasis.current();
        showTooltipFor(d, this);
      })
      .on("mouseleave", () => {
        hoverIdRef.current = null;
        applyEmphasis.current();
        setTooltip(null);
      })
      .on("focus", function (_event, d) {
        hoverIdRef.current = d.id;
        applyEmphasis.current();
        showTooltipFor(d, this);
      })
      .on("blur", () => {
        hoverIdRef.current = null;
        applyEmphasis.current();
        setTooltip(null);
      })
      .on("click", (_event, d) => {
        selfSelectRef.current = true;
        onSelectRef.current(d);
      })
      .on("keydown", (event: KeyboardEvent, d) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selfSelectRef.current = true;
          onSelectRef.current(d);
        }
      });

    node.call(
      d3
        .drag<SVGGElement, SimNode>()
        .on("start", (event, d) => {
          if (!event.active) sim.alphaTarget(0.25).restart();
          d.fx = d.x;
          d.fy = d.y;
          setTooltip(null);
        })
        .on("drag", (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
          if (reduceMotion) ticked();
        })
        .on("end", (event, d) => {
          if (!event.active) sim.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        }),
    );

    advanceTo(phaseRef.current);
    applyEmphasis.current();

    const resize = new ResizeObserver(() => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      if (!w || !h || (w === width && h === height)) return;
      width = w;
      height = h;
      svg.attr("viewBox", `0 0 ${width} ${height}`);
      sim.force("x", d3.forceX<SimNode>(width / 2).strength(0.06));
      sim.force("y", d3.forceY<SimNode>(height / 2).strength(0.06));
      // Re-fit once the layout re-settles in the new size (e.g. drawer opened).
      if (reduceMotion) {
        for (let i = 0; i < 120; i += 1) sim.tick();
        ticked();
        fit(false);
      } else {
        fitted = false;
        sim.alpha(0.3).restart();
      }
    });
    resize.observe(container);

    return () => {
      resize.disconnect();
      sim.stop();
      svg.interrupt();
      node.interrupt("appear");
      circle.interrupt("grow").interrupt("flood");
      link.interrupt("draw");
      nodeSelRef.current = null;
      linkSelRef.current = null;
    };
  }, [visible, colorOf, radiusOf, reduceMotion, uid, maxDiv, replayKey]);

  // Story phase changes.
  useEffect(() => {
    phaseRef.current = phase;
    advanceRef.current(phase);
  }, [phase]);

  // Blast radius / spotlight changes.
  useEffect(() => {
    blastRef.current = blast;
    spotlightRef.current = spotlightId;
    applyEmphasis.current();
  }, [blast, spotlightId]);

  // Selection ring; pan to nodes selected from outside the graph (list, search, story).
  useEffect(() => {
    selectedIdRef.current = selectedId;
    applyEmphasis.current();
    const fromGraph = selfSelectRef.current;
    selfSelectRef.current = false;
    if (fromGraph || !selectedId || !svgRef.current || !zoomRef.current) return;
    const target = simNodesRef.current.get(selectedId);
    if (!target || target.x == null || target.y == null) return;
    const svg = d3.select(svgRef.current);
    const tx = zoomRef.current;
    if (reduceMotion) svg.call(tx.translateTo, target.x, target.y);
    else svg.transition().duration(650).ease(d3.easeCubicInOut).call(tx.translateTo, target.x, target.y);
  }, [selectedId, reduceMotion]);

  // Search emphasis.
  useEffect(() => {
    matchesRef.current = query.trim() ? new Set(matches.map((m) => m.id)) : null;
    setActiveMatch(0);
    applyEmphasis.current();
  }, [matches, query]);

  const zoomBy = (factor: number) => {
    if (!svgRef.current || !zoomRef.current) return;
    const svg = d3.select(svgRef.current);
    if (reduceMotion) svg.call(zoomRef.current.scaleBy, factor);
    else svg.transition().duration(300).call(zoomRef.current.scaleBy, factor);
  };

  const pickMatch = (n: DebtNode) => {
    setQuery("");
    onSelect(n);
  };

  const listId = `${uid}-matches`;
  const hiddenCount = nodes.length - visible.nodes.length;
  const blastHidden = blast ? [...blast.depth.keys()].filter((id) => !simNodesRef.current.has(id)).length : 0;

  return (
    <div className="graph">
      <div className="graph__toolbar">
        <div className="search">
          <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true" className="search__icon">
            <path
              fill="currentColor"
              d="M8.5 3a5.5 5.5 0 0 1 4.38 8.82l3.65 3.65a.75.75 0 1 1-1.06 1.06l-3.65-3.65A5.5 5.5 0 1 1 8.5 3Zm0 1.5a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z"
            />
          </svg>
          <input
            type="search"
            placeholder="Find a function…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "ArrowDown") {
                e.preventDefault();
                setActiveMatch((i) => Math.min(i + 1, Math.max(0, matches.length - 1)));
              } else if (e.key === "ArrowUp") {
                e.preventDefault();
                setActiveMatch((i) => Math.max(i - 1, 0));
              } else if (e.key === "Enter" && matches[activeMatch]) {
                e.preventDefault();
                pickMatch(matches[activeMatch]);
              } else if (e.key === "Escape" && query) {
                e.stopPropagation();
                setQuery("");
              }
            }}
            role="combobox"
            aria-expanded={matches.length > 0}
            aria-controls={listId}
            aria-activedescendant={matches[activeMatch] ? `${listId}-${activeMatch}` : undefined}
            aria-label="Search functions by name or file"
          />
          <AnimatePresence>
            {query.trim() && (
              <motion.ul
                id={listId}
                role="listbox"
                className="search__results card"
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: 0.15 }}
              >
                {matches.length === 0 && <li className="search__none">No matching functions in view</li>}
                {matches.map((m, i) => (
                  <li
                    key={m.id}
                    id={`${listId}-${i}`}
                    role="option"
                    aria-selected={i === activeMatch}
                    className={i === activeMatch ? "is-active" : ""}
                    onMouseDown={(e) => {
                      e.preventDefault();
                      pickMatch(m);
                    }}
                    onMouseEnter={() => setActiveMatch(i)}
                  >
                    <span className="swatch" style={{ background: colorOf(m.div_score) }} />
                    <span className="search__name">{m.name}</span>
                    <span className="search__file">{m.file_path}</span>
                    <span className="search__div mono">{formatDiv(m.div_score)}</span>
                  </li>
                ))}
              </motion.ul>
            )}
          </AnimatePresence>
        </div>

        <div className="graph__controls">
          {nodes.length > LIMIT_OPTIONS[0] && (
            <label className="select">
              <span>Show</span>
              <select value={limit} onChange={(e) => setLimit(Number(e.target.value))}>
                {LIMIT_OPTIONS.filter((o) => o === 0 || o < nodes.length).map((o) => (
                  <option key={o} value={o}>
                    {o === 0 ? `All ${nodes.length}` : `Top ${o}`}
                  </option>
                ))}
              </select>
            </label>
          )}
          <span className="graph__count">
            {visible.nodes.length} functions · {visible.edges.length} calls
            {hiddenCount > 0 && <span className="muted"> ({hiddenCount} hidden)</span>}
          </span>
          <div className="btn-group" role="group" aria-label="Zoom">
            <button type="button" className="icon-btn" onClick={() => zoomBy(1.3)} aria-label="Zoom in">
              +
            </button>
            <button type="button" className="icon-btn" onClick={() => zoomBy(1 / 1.3)} aria-label="Zoom out">
              −
            </button>
            <button type="button" className="icon-btn" onClick={() => fitRef.current()} aria-label="Fit graph to view">
              <svg viewBox="0 0 20 20" width="14" height="14" aria-hidden="true">
                <path
                  d="M3 7V3h4M13 3h4v4M17 13v4h-4M7 17H3v-4"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                />
              </svg>
            </button>
          </div>
        </div>
      </div>

      <div className="graph__canvas" ref={containerRef}>
        <svg ref={svgRef} className="graph__svg" role="group" aria-label="Call graph heat-map" />
        {blast && blastHidden > 0 && (
          <p className="graph__note">
            {blastHidden} of {blast.functions} affected functions are outside the current view — choose “All” to see them.
          </p>
        )}
        <AnimatePresence>
          {tooltip && (
            <motion.div
              key={tooltip.node.id}
              className="tooltip"
              style={{ left: tooltip.x, top: tooltip.y }}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.12 }}
              role="tooltip"
            >
              <div className="tooltip__head">
                <span className="swatch" style={{ background: colorOf(tooltip.node.div_score) }} />
                <strong>{tooltip.node.name}</strong>
              </div>
              <div className="tooltip__file">
                {tooltip.node.file_path}
                {tooltip.node.start_line ? `:${tooltip.node.start_line}` : ""}
              </div>
              <dl className="tooltip__grid">
                <dt>DIV</dt>
                <dd className="mono">{formatDiv(tooltip.node.div_score, 3)}</dd>
                <dt>Complexity</dt>
                <dd className="mono">{tooltip.node.cyclomatic_complexity}</dd>
                <dt>Churn</dt>
                <dd className="mono">{tooltip.node.change_frequency} commits to file</dd>
              </dl>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="legend" aria-label="Legend">
        <div className="legend__scale">
          <span className="legend__title">DIV score</span>
          <div
            className="legend__bar"
            style={{ background: `linear-gradient(90deg, ${DIV_STOPS.join(", ")})` }}
            aria-hidden="true"
          />
          <div className="legend__ticks mono">
            <span>0</span>
            <span>{formatDiv(maxDiv / 4)}</span>
            <span>{formatDiv(maxDiv)}</span>
          </div>
        </div>
        <ul className="legend__notes">
          <li>
            <span className="legend__sizes" aria-hidden="true">
              <i />
              <i />
              <i />
            </span>
            Size ∝ DIV
          </li>
          <li>
            <span className="legend__glow" aria-hidden="true" />
            Glow = hotspot
          </li>
          <li>
            <span className="legend__arrow" aria-hidden="true" />
            Caller → callee
          </li>
          <li>
            <span className="legend__line" style={{ background: "var(--callee)" }} aria-hidden="true" />
            Callees on hover
          </li>
          <li>
            <span className="legend__line" style={{ background: "var(--caller)" }} aria-hidden="true" />
            Callers on hover
          </li>
          <li>
            <span className="legend__ripple" aria-hidden="true" />
            Blast radius of selection
          </li>
        </ul>
        {maxDiv === 0 && (
          <p className="legend__warn">
            Every DIV score is 0 — churn needs git history. Re-ingest a repository with commits.
          </p>
        )}
      </div>
    </div>
  );
}
