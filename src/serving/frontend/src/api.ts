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

export type FunctionContext = {
  file_path: string;
  name: string;
  metrics?: Record<string, unknown>;
  callers?: Array<Record<string, unknown>>;
  callees?: Array<Record<string, unknown>>;
  semantic_neighbors?: Array<Record<string, unknown>>;
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

export function fetchDebtGraph(): Promise<DebtGraphResponse> {
  return getJson("/api/debt-graph");
}

export function fetchTopDebt(k = 10): Promise<{ k: number; nodes: DebtNode[] }> {
  return getJson(`/api/top-debt?k=${k}`);
}

export function fetchFunctionContext(
  filePath: string,
  name: string,
): Promise<FunctionContext> {
  const qs = new URLSearchParams({ file_path: filePath, name });
  return getJson(`/api/function/context?${qs.toString()}`);
}

export async function startRefactorJob(
  topK: number,
): Promise<{ job_id: string; status: string }> {
  const res = await fetch(`${API_BASE}/api/refactor-proposals`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ top_k: topK }),
  });
  if (!res.ok) {
    throw new Error(await res.text());
  }
  return res.json();
}

export function fetchRefactorJob(jobId: string): Promise<Record<string, unknown>> {
  return getJson(`/api/refactor-proposals/${jobId}`);
}
