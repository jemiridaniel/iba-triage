// Mirrors backend/app/schemas.py (the subset the UI uses).

export type TriageLevel = "refer_now" | "refer_24h" | "treat_monitor";

export interface DangerSignHit {
  code: string;
  label: string;
  source: "rule" | "llm_intake" | "llm_reason";
  evidence?: string | null;
}

export interface FollowUpQuestion {
  id: string;
  text: string;
}

export interface FollowUpAnswer {
  id: string;
  answer: string;
}

export interface OutbreakSignal {
  disease: string;
  state: string;
  status: string;
  report_date?: string | null;
  url?: string | null;
  basis: "live" | "baseline";
  in_season?: boolean | null;
  citation?: string | null;
}

export interface OutbreakContext {
  status: "ok" | "unavailable";
  source: "tavily" | "mock" | "none";
  message: string;
  checked_at?: string | null;
  signals: OutbreakSignal[];
  baseline: OutbreakSignal[];
}

export interface Evidence {
  status: "verified" | "unsupported" | "patient" | "rule";
  chunk_id?: string | null;
  quote?: string | null;
  score?: number | null;
}

export interface Reason {
  text: string;
  evidence: Evidence;
}

export interface DifferentialItem {
  condition: string;
  likelihood: "high" | "moderate" | "low";
  reasons: Reason[];
  check_next: string[];
  citations: string[];
  source: "llm" | "rule";
}

export interface ActionItem {
  text: string;
  citations: string[];
  source: "llm" | "rule";
  details: Reason[];
  evidence?: Evidence | null;
}

export interface Citation {
  kind: "guideline" | "outbreak";
  ref: string;
  title?: string | null;
  doc_id?: string | null;
  section?: string | null;
  page?: number | null;
  page_end?: number | null;
  url?: string | null;
  source_date?: string | null;
  excerpt?: string | null;
}

export interface DoseRecommendation {
  drug: string;
  regimen: string;
  weight_band: string;
  verified: boolean;
  citation: Citation;
}

export interface TraceStep {
  step: string;
  kind: "llm" | "rules" | "retrieval" | "tool";
  model?: string | null;
  reasoning?: boolean | null;
  calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  reasoning_tokens?: number | null;
  latency_ms: number;
  model_latency_ms?: number | null;
  cost_usd?: number | null;
  cached: boolean;
  note?: string | null;
}

export interface DecisionTrace {
  steps: TraceStep[];
  total_latency_ms: number;
  total_tokens: number;
  total_cost_usd: number;
}

export interface TriageResult {
  status: "complete" | "needs_info" | "incomplete";
  triage_level?: TriageLevel | null;
  triage_label?: string | null;
  triage_rationale?: string | null;
  questions: FollowUpQuestion[];
  danger_signs: DangerSignHit[];
  lassa_suspected: boolean;
  outbreak?: OutbreakContext | null;
  differential: DifferentialItem[];
  actions: ActionItem[];
  doses: DoseRecommendation[];
  citations: Citation[];
  summary?: string | null;
  referral_note?: string | null;
  warnings: string[];
  disclaimer: string;
  decision_trace: DecisionTrace;
}

export interface TriageRequest {
  text: string;
  state?: string | null;
  answers?: FollowUpAnswer[];
  skip_questions?: boolean;
}

/** Partial result assembled from stream events as they arrive. */
export interface LiveRun {
  done: string[]; // node names finished so far
  questions?: FollowUpQuestion[];
  floor?: TriageLevel | null;
  floorLabel?: string | null;
  dangerSigns?: DangerSignHit[];
  outbreak?: OutbreakContext;
  post?: Omit<TriageResult, "questions" | "disclaimer" | "decision_trace" | "outbreak">;
  summary?: string | null;
  referralNote?: string | null;
  trace: TraceStep[];
  final?: TriageResult;
  error?: { fatal: boolean; message: string };
}

export interface Meta {
  models: Record<string, string>;
  reasoning: Record<string, boolean>;
  sources: { doc_id: string; title: string; edition?: string; url?: string; licence?: string; licence_note?: string }[];
  index: { chunks: number; model?: string | null };
  live_outbreak_search?: string | null;
  dose_table_verified: boolean;
  endemicity_verified: boolean;
}
