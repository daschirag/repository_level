import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { AnimatePresence, MotionConfig, motion, useReducedMotion } from "framer-motion";
import {
  fetchAnalysis,
  fetchDebtGraph,
  fetchHealth,
  fetchTopDebt,
  type AnalysisResponse,
  type CallNeighbor,
  type DebtGraphResponse,
  type HealthResponse,
  type TopDebtNode,
} from "./api";
import { ComparisonPanel } from "./components/ComparisonPanel";
import { GraphView, type GraphPhase } from "./components/GraphView";
import { Header } from "./components/Header";
import { NodeDrawer } from "./components/NodeDrawer";
import { PipelineStepper, STORY_STEPS, type StepId, type StoryStep } from "./components/PipelineStepper";
import { RefactorPlan, rankByPriority } from "./components/RefactorPlan";
import { StatCards, type Stats } from "./components/StatCards";
import { TopDebtList } from "./components/TopDebtList";
import { EmptyState, ErrorBanner, Skeleton } from "./components/ui";
import { baselines, blastRadius, buildIndex, rankByDiv } from "./lib/analysis";
import { formatDiv, makeDivColor, makeDivRadius } from "./lib/scales";
import { useRefactorJobs } from "./lib/useRefactorJobs";

const TOP_K = 15;
const INTRO_STEP_MS = 1900;
const PRESENT_STEP_MS = 7000;

type LoadState =
  | { status: "loading" }
  | { status: "ready"; graph: DebtGraphResponse; top: TopDebtNode[]; analysis: AnalysisResponse }
  | { status: "error"; message: string };

function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

/** Story step index -> graph reveal phase. */
function phaseFor(index: number | null): GraphPhase {
  if (index == null) return 3;
  return index === 0 ? 1 : index === 1 ? 2 : 3;
}

function Section({
  id,
  title,
  kicker,
  aside,
  children,
}: {
  id: string;
  title: string;
  kicker: string;
  aside?: ReactNode;
  children: ReactNode;
}) {
  return (
    <motion.section
      id={id}
      className="card panel story-section"
      aria-labelledby={`${id}-title`}
      initial={{ opacity: 0, y: 28 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.12 }}
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      tabIndex={-1}
    >
      <div className="panel__head">
        <div>
          <span className="kicker">{kicker}</span>
          <h2 id={`${id}-title`}>{title}</h2>
        </div>
        {aside}
      </div>
      {children}
    </motion.section>
  );
}

export default function App() {
  const reduceMotion = useReducedMotion() ?? false;
  const [load, setLoad] = useState<LoadState>({ status: "loading" });
  const [health, setHealth] = useState<HealthResponse | null | "error">(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const { jobs, start } = useRefactorJobs();

  // Story / presentation state. The intro is armed from the first render (unless the
  // user prefers reduced motion) so the graph mounts at phase 1, not fully built.
  const [playing, setPlaying] = useState(() => !prefersReducedMotion());
  const [storyIndex, setStoryIndex] = useState<number | null>(() => (prefersReducedMotion() ? null : 0));
  const [presenting, setPresenting] = useState(false);
  const [replayKey, setReplayKey] = useState(0);
  const [inViewId, setInViewId] = useState<StepId | null>(null);

  // ---- Data ----
  useEffect(() => {
    let cancelled = false;
    setLoad((prev) => (prev.status === "ready" ? prev : { status: "loading" }));
    Promise.all([fetchDebtGraph(), fetchTopDebt(TOP_K), fetchAnalysis()])
      .then(([graph, top, analysis]) => {
        if (!cancelled) setLoad({ status: "ready", graph, top: top.nodes, analysis });
      })
      .catch((err) => {
        if (cancelled) return;
        const raw = err instanceof Error ? err.message : String(err);
        const message = /Failed to fetch|NetworkError|ECONNREFUSED|^50[234]/.test(raw)
          ? "The API at /api is unreachable. Start it with `uvicorn src.serving.api.main:app --port 8000` and make sure Neo4j is running."
          : raw;
        setLoad({ status: "error", message });
      });
    return () => {
      cancelled = true;
    };
  }, [attempt]);

  // Health: poll quickly while the model loads, slowly afterwards.
  const llmSettled = health !== null && health !== "error" && !!health.llm && ["ready", "failed"].includes(health.llm.status);
  useEffect(() => {
    let cancelled = false;
    const tick = () =>
      fetchHealth()
        .then((h) => !cancelled && setHealth(h))
        .catch(() => !cancelled && setHealth("error"));
    void tick();
    const id = setInterval(tick, llmSettled ? 30000 : 4000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [attempt, llmSettled]);

  const retry = useCallback(() => {
    setLoad({ status: "loading" });
    setAttempt((a) => a + 1);
  }, []);

  const ready = load.status === "ready" ? load : null;
  const graph = ready?.graph ?? null;
  const analysis = ready?.analysis ?? null;

  const functions = useMemo(
    () => (graph ? graph.nodes.filter((n) => (n.node_kind ?? "Function") === "Function") : []),
    [graph],
  );
  const callEdges = useMemo(() => (graph ? graph.edges.filter((e) => e.type === "CALLS") : []), [graph]);
  const index = useMemo(() => buildIndex(functions, callEdges), [functions, callEdges]);
  const base = useMemo(() => baselines(functions), [functions]);
  const divRank = useMemo(() => new Map(rankByDiv(functions).map((n, i) => [n.id, i + 1])), [functions]);
  const topPriority = useMemo(() => rankByPriority(functions)[0] ?? null, [functions]);

  const stats = useMemo<Stats | null>(() => {
    if (!analysis) return null;
    const c = analysis.counts;
    const top = rankByDiv(functions)[0];
    return {
      functions: c.functions,
      files: c.files,
      calls: c.calls,
      imports: c.imports,
      connected: c.connected,
      maxDiv: c.max_div ?? 0,
      maxDivName: top ? top.name : null,
      avgDiv: c.avg_div ?? 0,
      scored: c.scored,
    };
  }, [analysis, functions]);

  const maxDiv = stats?.maxDiv ?? 0;
  const colorOf = useMemo(() => makeDivColor(maxDiv), [maxDiv]);
  const radiusOf = useMemo(() => makeDivRadius(maxDiv), [maxDiv]);

  const selected = selectedId ? index.byId.get(selectedId) ?? null : null;
  const blast = useMemo(() => (selectedId && index.byId.has(selectedId) ? blastRadius(index, selectedId) : null), [index, selectedId]);
  const close = useCallback(() => setSelectedId(null), []);
  const selectNode = useCallback((n: { id: string }) => setSelectedId(n.id), []);
  const navigate = useCallback(
    (n: CallNeighbor) => {
      const id = `${n.file_path}::${n.name}`;
      if (index.byId.has(id)) setSelectedId(id);
    },
    [index],
  );

  // ---- Story controller ----
  const scrollToSection = useCallback(
    (sectionId: string) => {
      const el = document.getElementById(sectionId);
      if (!el) return;
      el.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
      el.focus({ preventScroll: true });
    },
    [reduceMotion],
  );

  // Reduced motion detected after mount: never auto-play the intro.
  useEffect(() => {
    if (reduceMotion && playing && !presenting) {
      setPlaying(false);
      setStoryIndex(null);
    }
  }, [reduceMotion, playing, presenting]);

  // Story clock (waits for data so the reveal is visible).
  useEffect(() => {
    if (!ready || !playing || storyIndex == null) return;
    const dwell = presenting ? PRESENT_STEP_MS : INTRO_STEP_MS;
    const id = window.setTimeout(() => {
      const next = storyIndex + 1;
      if (next < STORY_STEPS.length) {
        setStoryIndex(next);
      } else if (presenting) {
        // Loop the presentation from the top.
        setSelectedId(null);
        setReplayKey((k) => k + 1);
        setStoryIndex(0);
      } else {
        setPlaying(false);
        setStoryIndex(null);
      }
    }, dwell);
    return () => window.clearTimeout(id);
  }, [ready, playing, storyIndex, presenting]);

  // In presentation mode each step scrolls to its section; Refactor opens the #1 plan.
  useEffect(() => {
    if (!presenting || storyIndex == null) return;
    const step = STORY_STEPS[storyIndex];
    scrollToSection(step.section);
    if (step.id === "refactor" && topPriority) setSelectedId(topPriority.id);
  }, [presenting, storyIndex, scrollToSection, topPriority]);

  const skip = useCallback(() => {
    setPlaying(false);
    setStoryIndex(null);
  }, []);

  const replay = useCallback(() => {
    setSelectedId(null);
    setReplayKey((k) => k + 1);
    scrollToSection("sec-overview");
    if (reduceMotion) return;
    setStoryIndex(0);
    setPlaying(true);
  }, [reduceMotion, scrollToSection]);

  const togglePresenting = useCallback(() => {
    setPresenting((on) => {
      const next = !on;
      setSelectedId(null);
      if (next) {
        setReplayKey((k) => k + 1);
        setStoryIndex(0);
        setPlaying(true);
      } else {
        setPlaying(false);
        setStoryIndex(null);
      }
      return next;
    });
  }, []);

  const onStepClick = useCallback(
    (step: StoryStep) => {
      if (playing && !presenting) skip();
      scrollToSection(step.section);
    },
    [playing, presenting, skip, scrollToSection],
  );

  // Esc: close the drawer first, then leave presentation mode.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (selectedId) close();
      else if (presenting) togglePresenting();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedId, presenting, close, togglePresenting]);

  // Scroll-spy for the stepper when the story is idle.
  useEffect(() => {
    if (!ready) return;
    const sectionToStep = new Map<string, StepId>();
    for (const s of STORY_STEPS) if (!sectionToStep.has(s.section)) sectionToStep.set(s.section, s.id);
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) setInViewId(sectionToStep.get(entry.target.id) ?? null);
        }
      },
      { rootMargin: "-35% 0px -55% 0px" },
    );
    for (const id of sectionToStep.keys()) {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    }
    return () => observer.disconnect();
  }, [ready]);

  const llm = health && health !== "error" ? health.llm ?? null : null;
  const qdrantOnline = health && health !== "error" ? health.qdrant : health === "error" ? false : null;
  const isEmpty = load.status === "ready" && functions.length === 0;
  const phase: GraphPhase = ready ? (playing ? phaseFor(storyIndex) : 3) : 0;
  const spotlightId = playing && storyIndex === 3 && topPriority ? topPriority.id : null;
  const proposalsDone = Object.values(jobs).filter((j) => j.status === "completed" && j.proposal);
  const aiValid = proposalsDone.filter((j) => j.proposal?.source === "ai").length;

  return (
    <MotionConfig reducedMotion="user">
      <div className={`app ${selected ? "has-drawer" : ""} ${presenting ? "is-presenting" : ""}`}>
        <Header
          health={health}
          meta={analysis ? analysis.meta : undefined}
          presenting={presenting}
          onTogglePresenting={togglePresenting}
        />

        <div className="stepper-wrap">
          <PipelineStepper
            analysis={analysis}
            llm={llm}
            topPriorityName={topPriority?.name ?? null}
            activeIndex={playing ? storyIndex : null}
            inViewId={inViewId}
            playing={playing}
            presenting={presenting}
            onStepClick={onStepClick}
            onSkip={presenting ? togglePresenting : skip}
            onReplay={replay}
          />
        </div>

        <main className="content">
          <AnimatePresence>
            {load.status === "error" && (
              <ErrorBanner title="Couldn’t load debt data" message={load.message} onRetry={retry} />
            )}
          </AnimatePresence>

          <section id="sec-overview" className="overview" aria-label="Overview" tabIndex={-1}>
            <StatCards stats={stats} />
          </section>

          {isEmpty && (
            <div className="card panel">
              <EmptyState title="No functions in the graph yet">
                <p>Load the Flask demo (clones history, parses, scores and embeds in ~30s):</p>
                <pre className="code">
                  <code>python scripts/load_demo.py</code>
                </pre>
                <button type="button" className="btn btn--ghost" onClick={retry}>
                  Reload
                </button>
              </EmptyState>
            </div>
          )}

          {!isEmpty && (
            <>
              <Section
                id="sec-graph"
                kicker="Steps 1–3 · Ingest → Call graph → DIV"
                title="Call-graph heat-map"
                aside={<span className="muted small hide-presenting">Hover to trace · click for blast radius · drag · zoom</span>}
              >
                <div className="workspace">
                  <div className="workspace__graph">
                    {!ready && (
                      <div className="graph-skeleton" aria-busy={load.status === "loading"}>
                        <Skeleton height="100%" radius={14} />
                        {load.status === "error" && <span className="graph-skeleton__msg">No data — backend unreachable</span>}
                      </div>
                    )}
                    {ready && (
                      <GraphView
                        nodes={functions}
                        edges={callEdges}
                        maxDiv={maxDiv}
                        colorOf={colorOf}
                        radiusOf={radiusOf}
                        selectedId={selectedId}
                        onSelect={selectNode}
                        phase={phase}
                        replayKey={replayKey}
                        blast={phase === 3 ? blast : null}
                        spotlightId={spotlightId}
                      />
                    )}
                  </div>
                  <aside className="workspace__list hide-presenting" aria-label="Top debt nodes">
                    <TopDebtList
                      nodes={ready ? ready.top : load.status === "error" ? [] : null}
                      maxDiv={maxDiv}
                      colorOf={colorOf}
                      selectedId={selectedId}
                      onSelect={selectNode}
                    />
                  </aside>
                </div>
              </Section>

              <Section
                id="sec-compare"
                kicker="Step 3 · Why DIV"
                title="Flat metrics vs causal impact"
              >
                {ready ? (
                  <ComparisonPanel functions={functions} index={index} colorOf={colorOf} onSelect={selectNode} />
                ) : (
                  <Skeleton height={420} radius={12} />
                )}
              </Section>

              <Section id="sec-plan" kicker="Step 4 · Prioritise" title="Refactor order">
                {ready ? (
                  <RefactorPlan
                    functions={functions}
                    index={index}
                    base={base}
                    colorOf={colorOf}
                    selectedId={selectedId}
                    onSelect={selectNode}
                  />
                ) : (
                  <Skeleton height={420} radius={12} />
                )}
              </Section>

              <Section id="sec-refactor" kicker="Step 5 · Refactor" title="Remediation">
                <div className="remedy">
                  <div className="remedy__copy">
                    <p>
                      Each function gets <strong>rule-based</strong> suggestions instantly from its graph metrics. An{" "}
                      <strong>AI proposal</strong> runs in the background on the local model; output is accepted only if
                      it contains real code or a concrete change to that function, otherwise the agent retries once
                      with a stricter prompt and then falls back to the rules — never to unvalidated text.
                    </p>
                    <dl className="remedy__facts">
                      <div>
                        <dt>Model</dt>
                        <dd>
                          {llm
                            ? `${llm.model ?? "—"} · ${llm.status}${llm.load_s != null ? ` (loaded in ${llm.load_s}s)` : ""}`
                            : "—"}
                        </dd>
                      </div>
                      <div>
                        <dt>This session</dt>
                        <dd>
                          {proposalsDone.length === 0
                            ? "no proposals generated yet"
                            : `${proposalsDone.length} proposal${proposalsDone.length === 1 ? "" : "s"} · ${aiValid} passed validation · ${proposalsDone.length - aiValid} fell back to rules`}
                        </dd>
                      </div>
                    </dl>
                  </div>
                  {topPriority && (
                    <button type="button" className="remedy__cta card" onClick={() => setSelectedId(topPriority.id)}>
                      <span className="kicker">Start with #1</span>
                      <strong>{topPriority.name}</strong>
                      <span className="muted small">
                        {topPriority.file_path} · DIV {formatDiv(topPriority.div_score, 1)}
                      </span>
                      <span className="remedy__go">Open remediation →</span>
                    </button>
                  )}
                </div>
              </Section>
            </>
          )}
        </main>

        <AnimatePresence>
          {selected && blast && (
            <NodeDrawer
              node={selected}
              rank={divRank.get(selected.id) ?? 0}
              total={functions.length}
              maxDiv={maxDiv}
              index={index}
              base={base}
              blast={blast}
              colorOf={colorOf}
              qdrantOnline={qdrantOnline}
              llm={llm}
              proposal={jobs[selected.id]}
              onGenerate={(force) => start(selected, force)}
              onNavigate={navigate}
              onClose={close}
            />
          )}
        </AnimatePresence>
      </div>
    </MotionConfig>
  );
}

