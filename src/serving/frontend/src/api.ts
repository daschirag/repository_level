export type DebtNode = {
  id: string;
  file_path: string;
  name: string;
  div_score: number;
  cyclomatic_complexity: number;
  change_frequency: number;
  start_line?: number | null;
  end_line?: number | null;
  node_kind?: string;
};

export type DebtEdge = {
  source: string;
  target: string;
  type: string;
};

export type DebtGraphResponse = {
  nodes: DebtNode[];
  edges: DebtEdge[];
  node_count: number;
  edge_count: number;
};

/** Row from ``/api/top-debt`` (no line/kind info, but carries the graph ``id``). */
export type TopDebtNode = Pick<
  DebtNode,
  "id" | "file_path" | "name" | "div_score" | "cyclomatic_complexity" | "change_frequency"
>;

export type HealthResponse = {
  status: string;
  neo4j: boolean;
  qdrant: boolean;
  jobs_tracked: number;
};

/** 1-hop CALLS neighbour returned by the context endpoint. */
export type CallNeighbor = {
  file_path: string;
  name: string;
  div_score?: number | null;
};

/** Qdrant semantic hit (docstring / comment similarity). */
export type SemanticNeighbor = {
  file_path: string | null;
  name: string | null;
  node_type?: string | null;
  start_line?: number | null;
  end_line?: number | null;
  div_score?: number | null;
  text?: string | null;
  score: number;
};

export type FunctionContext = {
  file_path: string;
  name: string;
  metrics?: {
    div_score?: number | null;
    cyclomatic_complexity?: number | null;
    change_frequency?: number | null;
    start_line?: number | null;
    end_line?: number | null;
  };
  callers?: CallNeighbor[];
  callees?: CallNeighbor[];
  semantic_neighbors?: SemanticNeighbor[];
};

export type RefactorProposal = {
  file_path: string;
  name: string;
  div_score: number;
  refactor_type: string | null;
  proposal_text: string | null;
  rationale: string | null;
  raw_llm_output?: string | null;
};

export type RefactorJobStatus = "queued" | "running" | "completed" | "failed";

export type RefactorJob = {
  job_id: string;
  status: RefactorJobStatus;
  top_k: number;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  result: { top_k: number; proposals: RefactorProposal[] } | null;
  error: string | null;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export function fetchHealth(): Promise<HealthResponse> {
  return getJson("/api/health");
}

export function fetchDebtGraph(): Promise<DebtGraphResponse> {
  return getJson("/api/debt-graph");
}

export function fetchTopDebt(k = 10): Promise<{ k: number; nodes: TopDebtNode[] }> {
  return getJson(`/api/top-debt?k=${k}`);
}

export function fetchFunctionContext(
  filePath: string,
  name: string,
): Promise<FunctionContext> {
  const qs = new URLSearchParams({ file_path: filePath, name });
  return getJson(`/api/function/context?${qs.toString()}`);
}

/** Enqueue a refactor job for one function (``file_path`` + ``name``). */
export async function startRefactorJob(target: {
  file_path: string;
  name: string;
}): Promise<{ job_id: string; status: RefactorJobStatus }> {
  const res = await fetch(`${API_BASE}/api/refactor-proposals`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ top_k: 1, ...target }),
  });
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
  }
  return res.json();
}

export function fetchRefactorJob(jobId: string): Promise<RefactorJob> {
  return getJson(`/api/refactor-proposals/${jobId}`);
}
