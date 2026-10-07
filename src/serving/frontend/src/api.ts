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

export type LlmState = {
  status: "not_loaded" | "loading" | "ready" | "failed";
  model: string | null;
  error: string | null;
  load_s: number | null;
};

export type HealthResponse = {
  status: string;
  neo4j: boolean;
  qdrant: boolean;
  llm?: LlmState;
  jobs_tracked: number;
};

/** Written by scripts/load_demo.py as (:Analysis {id:'latest'}). */
export type AnalysisMeta = {
  repo_name: string;
  repo_url: string;
  commit: string;
  commit_date: string;
  commit_subject: string;
  source_path: string;
  scope: string;
  fixture_verified: boolean;
  commits_in_history: number;
  files_with_churn: number;
  call_sites: number;
  unresolved_calls: number;
  rag_embedded: number;
  rag_considered: number;
  analysed_at: string;
  duration_s: number;
  t_history_s?: number;
  t_parse_s?: number;
  t_graph_s?: number;
  t_scoring_s?: number;
  t_embeddings_s?: number;
};

export type AnalysisCounts = {
  files: number;
  functions: number;
  scored: number;
  max_div: number | null;
  avg_div: number | null;
  loc: number;
  classes: number;
  modules: number;
  calls: number;
  imports: number;
  connected: number;
};

export type AnalysisResponse = {
  meta: AnalysisMeta | null;
  counts: AnalysisCounts;
};

export type RuleSuggestion = {
  rule: string;
  title: string;
  detail: string;
};

export type ImpactMetrics = {
  cyclomatic_complexity: number;
  loc: number;
  change_frequency: number;
  direct_callers: number;
  direct_callees: number;
  transitive_callers: number;
  affected_files: number;
  churn_threshold: number;
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
  impact?: ImpactMetrics | null;
  rule_based?: RuleSuggestion[];
};

export type ProposalAttempt = {
  attempt: number;
  prompt_style: string;
  valid: boolean;
  reasons: string[];
};

export type RefactorProposal = {
  file_path: string;
  name: string;
  div_score: number;
  /** "ai" only when model output passed validation. */
  source: "ai" | "rule_based";
  model: string;
  valid: boolean;
  attempts: ProposalAttempt[];
  refactor_type: string | null;
  proposal_text: string | null;
  rationale: string | null;
  raw_llm_output?: string | null;
  source_error?: string | null;
  rule_based: RuleSuggestion[];
};

export type RefactorJobStatus = "queued" | "running" | "completed" | "failed";

export type JobStage =
  | "queued"
  | "waiting_for_model"
  | "load_span"
  | "gather_context"
  | "gather_source"
  | "generate_1"
  | "generate_2"
  | "rules"
  | "done";

export type RefactorJob = {
  job_id: string;
  status: RefactorJobStatus;
  top_k: number;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  result: { top_k: number; proposals: RefactorProposal[] } | null;
  error: string | null;
  stage: JobStage;
  stages_seen: JobStage[];
  stage_order: JobStage[];
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

export function fetchAnalysis(): Promise<AnalysisResponse> {
  return getJson("/api/analysis");
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

/**
 * Enqueue a refactor job for one function (``file_path`` + ``name``). The API
 * returns the cached job (``cached: true``) unless ``force`` is set.
 */
export async function startRefactorJob(
  target: { file_path: string; name: string },
  force = false,
): Promise<{ job_id: string; status: RefactorJobStatus; cached: boolean }> {
  const res = await fetch(`${API_BASE}/api/refactor-proposals`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ top_k: 1, force, ...target }),
  });
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
  }
  return res.json();
}

export function fetchRefactorJob(jobId: string): Promise<RefactorJob> {
  return getJson(`/api/refactor-proposals/${jobId}`);
}
