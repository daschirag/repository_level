import { motion } from "framer-motion";
import type { AnalysisResponse, LlmState } from "../api";
import { formatDiv } from "../lib/scales";
import { Skeleton } from "./ui";

export type StepId = "ingest" | "graph" | "scoring" | "prioritise" | "refactor";

export type StoryStep = {
  id: StepId;
  title: string;
  /** Section the step scrolls to. */
  section: string;
};

export const STORY_STEPS: StoryStep[] = [
  { id: "ingest", title: "Ingest", section: "sec-overview" },
  { id: "graph", title: "Call Graph", section: "sec-graph" },
  { id: "scoring", title: "Debt Scoring (DIV)", section: "sec-compare" },
  { id: "prioritise", title: "Prioritise", section: "sec-plan" },
  { id: "refactor", title: "Refactor", section: "sec-refactor" },
];

type Props = {
  analysis: AnalysisResponse | null;
  llm: LlmState | null;
  topPriorityName: string | null;
  /** Index of the step the story is on; null when idle (all complete). */
  activeIndex: number | null;
  /** Step whose section is currently in view (idle highlight). */
  inViewId: StepId | null;
  playing: boolean;
  presenting: boolean;
  onStepClick: (step: StoryStep) => void;
  onSkip: () => void;
  onReplay: () => void;
};

const fmt = (n: number | null | undefined) => (n == null ? "—" : n.toLocaleString());
const secs = (s: number | null | undefined) => (s == null ? null : `${s < 10 ? s.toFixed(1) : Math.round(s)}s`);

function stepLines(
  id: StepId,
  a: AnalysisResponse,
  llm: LlmState | null,
  topPriorityName: string | null,
): { line: string; sub: string } {
  const c = a.counts;
  const m = a.meta;
  switch (id) {
    case "ingest":
      return {
        line: `Parsed ${fmt(c.functions)} functions across ${fmt(c.files)} files`,
        sub: [`${fmt(c.classes)} classes`, `${fmt(c.loc)} lines`, secs(m?.t_parse_s)].filter(Boolean).join(" · "),
      };
    case "graph":
      return {
        line: `Built ${fmt(c.calls)} call edges`,
        sub: [
          `${fmt(c.connected)} functions connected`,
          `${fmt(c.imports)} imports`,
          m ? `${fmt(m.call_sites)} call sites scanned` : null,
        ]
          .filter(Boolean)
          .join(" · "),
      };
    case "scoring":
      return {
        line: "Propagated debt along dependency chains",
        sub: [
          m ? `${fmt(m.commits_in_history)} commits of history` : null,
          `${fmt(c.scored)} functions scored`,
          c.max_div != null ? `max DIV ${formatDiv(c.max_div, 1)}` : null,
        ]
          .filter(Boolean)
          .join(" · "),
      };
    case "prioritise":
      return {
        line: "Ranked by impact per unit of effort",
        sub: topPriorityName ? `#1 to fix: ${topPriorityName}` : "no scored functions",
      };
    case "refactor":
      return {
        line: "Validated AI proposals, rule-based fallback",
        sub:
          llm == null
            ? "model status unknown"
            : llm.status === "ready"
              ? `${llm.model ?? "model"} loaded${llm.load_s != null ? ` in ${secs(llm.load_s)}` : ""}`
              : llm.status === "loading"
                ? "model loading…"
                : llm.status === "failed"
                  ? "model failed to load — rule-based only"
                  : "model not loaded",
      };
  }
}

export function PipelineStepper({
  analysis,
  llm,
  topPriorityName,
  activeIndex,
  inViewId,
  playing,
  presenting,
  onStepClick,
  onSkip,
  onReplay,
}: Props) {
  return (
    <nav className="stepper card" aria-label="Analysis pipeline">
      <ol className="stepper__list">
        {STORY_STEPS.map((step, i) => {
          const state =
            activeIndex == null ? "done" : i < activeIndex ? "done" : i === activeIndex ? "active" : "pending";
          const current = activeIndex == null ? inViewId === step.id : i === activeIndex;
          const lines = analysis ? stepLines(step.id, analysis, llm, topPriorityName) : null;
          return (
            <li key={step.id} className={`step step--${state} ${current ? "is-current" : ""}`}>
              <button
                type="button"
                className="step__btn"
                onClick={() => onStepClick(step)}
                aria-current={current ? "step" : undefined}
              >
                <span className="step__index" aria-hidden="true">
                  {state === "done" ? (
                    <svg viewBox="0 0 16 16" width="12" height="12">
                      <path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                    </svg>
                  ) : (
                    i + 1
                  )}
                </span>
                <span className="step__text">
                  <span className="step__title">{step.title}</span>
                  {lines ? (
                    <>
                      <span className="step__line">{lines.line}</span>
                      <span className="step__sub">{lines.sub}</span>
                    </>
                  ) : (
                    <>
                      <Skeleton width="85%" height={11} />
                      <Skeleton width="60%" height={10} />
                    </>
                  )}
                </span>
              </button>
              {i < STORY_STEPS.length - 1 && (
                <span className="step__connector" aria-hidden="true">
                  <motion.span
                    className="step__connector-fill"
                    initial={false}
                    animate={{ scaleX: state === "done" ? 1 : 0 }}
                    transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
                  />
                </span>
              )}
            </li>
          );
        })}
      </ol>
      <div className="stepper__actions">
        {playing ? (
          <button type="button" className="btn btn--ghost" onClick={onSkip}>
            {presenting ? "Stop presentation" : "Skip intro"}
          </button>
        ) : (
          <button type="button" className="btn btn--ghost" onClick={onReplay}>
            <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true">
              <path d="M4 3v10l9-5z" fill="currentColor" />
            </svg>
            Replay story
          </button>
        )}
      </div>
    </nav>
  );
}
