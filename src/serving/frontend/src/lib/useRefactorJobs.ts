import { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchRefactorJob,
  startRefactorJob,
  type JobStage,
  type RefactorJobStatus,
  type RefactorProposal,
} from "../api";

export type ProposalState = {
  jobId: string | null;
  status: RefactorJobStatus | "starting";
  stage: JobStage | null;
  stagesSeen: JobStage[];
  /** Epoch ms when this request started (for the elapsed timer). */
  startedAt: number;
  /** Server-side elapsed seconds once finished. */
  durationS: number | null;
  cached: boolean;
  proposal: RefactorProposal | null;
  error: string | null;
};

const POLL_MS = 1500;

/**
 * Tracks one refactor job per graph node id, polling the backend until each
 * finishes. Lives above the drawer so a running job survives closing it; the
 * backend also caches completed proposals per function.
 */
export function useRefactorJobs() {
  const [jobs, setJobs] = useState<Record<string, ProposalState>>({});
  const timers = useRef<Record<string, ReturnType<typeof setInterval>>>({});

  const patch = useCallback((id: string, update: Partial<ProposalState>) => {
    setJobs((prev) => (prev[id] ? { ...prev, [id]: { ...prev[id], ...update } } : prev));
  }, []);

  const stopPolling = useCallback((id: string) => {
    clearInterval(timers.current[id]);
    delete timers.current[id];
  }, []);

  const poll = useCallback(
    (nodeId: string, jobId: string) => {
      const tick = async () => {
        try {
          const job = await fetchRefactorJob(jobId);
          const durationS =
            job.started_at && job.finished_at
              ? (Date.parse(job.finished_at) - Date.parse(job.started_at)) / 1000
              : null;
          if (job.status === "completed") {
            stopPolling(nodeId);
            const proposal = job.result?.proposals?.[0] ?? null;
            patch(nodeId, {
              status: "completed",
              stage: "done",
              stagesSeen: job.stages_seen,
              durationS,
              proposal,
              error: proposal ? null : "The agent returned no proposal for this function.",
            });
          } else if (job.status === "failed") {
            stopPolling(nodeId);
            patch(nodeId, { status: "failed", durationS, error: job.error ?? "Refactor job failed." });
          } else {
            patch(nodeId, { status: job.status, stage: job.stage, stagesSeen: job.stages_seen });
          }
        } catch (err) {
          stopPolling(nodeId);
          patch(nodeId, { status: "failed", error: err instanceof Error ? err.message : String(err) });
        }
      };
      void tick();
      timers.current[nodeId] = setInterval(tick, POLL_MS);
    },
    [patch, stopPolling],
  );

  const start = useCallback(
    async (node: { id: string; file_path: string; name: string }, force = false) => {
      stopPolling(node.id);
      setJobs((prev) => ({
        ...prev,
        [node.id]: {
          jobId: null,
          status: "starting",
          stage: null,
          stagesSeen: [],
          startedAt: Date.now(),
          durationS: null,
          cached: false,
          proposal: null,
          error: null,
        },
      }));
      try {
        const { job_id, status, cached } = await startRefactorJob(
          { file_path: node.file_path, name: node.name },
          force,
        );
        patch(node.id, { jobId: job_id, status, cached });
        poll(node.id, job_id);
      } catch (err) {
        patch(node.id, { status: "failed", error: err instanceof Error ? err.message : String(err) });
      }
    },
    [patch, poll, stopPolling],
  );

  useEffect(() => {
    const active = timers.current;
    return () => Object.values(active).forEach(clearInterval);
  }, []);

  return { jobs, start };
}
