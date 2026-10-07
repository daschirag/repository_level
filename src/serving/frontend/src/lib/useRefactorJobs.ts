import { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchRefactorJob,
  startRefactorJob,
  type RefactorJobStatus,
  type RefactorProposal,
} from "../api";

export type ProposalState = {
  jobId: string | null;
  status: RefactorJobStatus | "starting";
  startedAt: number;
  proposal: RefactorProposal | null;
  error: string | null;
};

const POLL_MS = 2500;

/**
 * Tracks one refactor job per graph node id, polling the backend until each
 * finishes. Lives above the drawer so a running job survives closing it.
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

  const start = useCallback(
    async (node: { id: string; file_path: string; name: string }) => {
      stopPolling(node.id);
      setJobs((prev) => ({
        ...prev,
        [node.id]: {
          jobId: null,
          status: "starting",
          startedAt: Date.now(),
          proposal: null,
          error: null,
        },
      }));
      try {
        const { job_id, status } = await startRefactorJob({
          file_path: node.file_path,
          name: node.name,
        });
        patch(node.id, { jobId: job_id, status });
        timers.current[node.id] = setInterval(async () => {
          try {
            const job = await fetchRefactorJob(job_id);
            if (job.status === "completed") {
              stopPolling(node.id);
              const proposal = job.result?.proposals?.[0] ?? null;
              patch(node.id, {
                status: "completed",
                proposal,
                error: proposal ? null : "The agent returned no proposal for this function.",
              });
            } else if (job.status === "failed") {
              stopPolling(node.id);
              patch(node.id, { status: "failed", error: job.error ?? "Refactor job failed." });
            } else {
              patch(node.id, { status: job.status });
            }
          } catch (err) {
            stopPolling(node.id);
            patch(node.id, {
              status: "failed",
              error: err instanceof Error ? err.message : String(err),
            });
          }
        }, POLL_MS);
      } catch (err) {
        patch(node.id, {
          status: "failed",
          error: err instanceof Error ? err.message : String(err),
        });
      }
    },
    [patch, stopPolling],
  );

  useEffect(() => {
    const active = timers.current;
    return () => Object.values(active).forEach(clearInterval);
  }, []);

  return { jobs, start };
}
