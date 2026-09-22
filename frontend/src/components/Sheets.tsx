import type { ReactNode } from "react";
import type { Citation, LiveRun, TraceStep } from "../types";
import { citationLabel } from "./ResultCard";

export function BottomSheet({ title, open, onClose, children }: { title: string; open: boolean; onClose: () => void; children: ReactNode }) {
  if (!open) return null;
  return (
    <div className="fade-in fixed inset-0 z-30 flex items-end bg-black/40" onClick={onClose} role="dialog" aria-modal="true" aria-label={title}>
      <div className="slide-up mx-auto max-h-[85vh] w-full max-w-xl overflow-y-auto rounded-t-2xl bg-white p-4 pb-[calc(1rem+env(safe-area-inset-bottom))]"
           onClick={(e) => e.stopPropagation()}>
        <div className="mx-auto mb-2 h-1.5 w-10 rounded-full bg-slate-300" />
        <div className="mb-3 flex items-center justify-between gap-2">
          <h2 className="text-lg font-bold leading-tight">{title}</h2>
          <button onClick={onClose} className="min-h-11 min-w-11 rounded-lg text-2xl" aria-label="Close">×</button>
        </div>
        {children}
      </div>
    </div>
  );
}

export function CitationSheet({ citation, quote, onClose }: { citation: Citation | null; quote?: string | null; onClose: () => void }) {
  return (
    <BottomSheet title={citation ? citationLabel(citation) : ""} open={Boolean(citation)} onClose={onClose}>
      {citation && (
        <div className="space-y-3 text-[15px] leading-relaxed">
          {citation.kind === "guideline" ? (
            <>
              <p className="text-sm text-slate-600">
                {citation.title}
                {citation.section && <> · <em>{citation.section}</em></>}
              </p>
              {quote ? (
                <>
                  <p className="text-xs font-bold uppercase tracking-wide text-teal-800">✓ Quote verified in this passage</p>
                  <blockquote className="border-l-4 border-teal-600 bg-teal-50 p-3 text-slate-800">“{quote}”</blockquote>
                </>
              ) : (
                citation.excerpt && (
                  <blockquote className="border-l-4 border-slate-300 bg-slate-50 p-3 text-slate-800">{citation.excerpt}</blockquote>
                )
              )}
              <p className="text-xs text-slate-500">From the indexed guideline passage Ibà retrieved. Check the full document for context.</p>
            </>
          ) : (
            <>
              <p>{citation.title}</p>
              {citation.source_date && <p className="text-sm text-slate-600">Report date: {citation.source_date}</p>}
              {citation.url && (
                <a href={citation.url} target="_blank" rel="noreferrer"
                   className="inline-flex min-h-11 items-center rounded-lg bg-sky-700 px-4 font-semibold text-white">
                  Open source ↗
                </a>
              )}
            </>
          )}
        </div>
      )}
    </BottomSheet>
  );
}

const STEP_NAMES: Record<string, string> = {
  intake: "Understand the case",
  rules_pre: "Danger-sign rules",
  embed_queries: "Prepare guideline search (in parallel)",
  retrieve: "Find guideline passages",
  outbreak: "Outbreak check",
  reason: "Clinical reasoning",
  rules_post: "Safety rules",
  compose: "Write summary & note",
};

function shortModel(model?: string | null): string {
  if (!model) return "";
  return model.split("/").pop()!.replace(/-\d+b-a\d+b$/i, "");
}

function stepMs(s: TraceStep): number {
  // Replayed (cached) calls report their original model time.
  return s.cached && s.model_latency_ms ? s.model_latency_ms : s.latency_ms;
}

export function TraceSheet({ steps, wallMs, open, onClose }: { steps: TraceStep[]; wallMs?: number | null; open: boolean; onClose: () => void }) {
  const sum = steps.reduce((a, s) => a + stepMs(s), 0) || 1;
  const total = wallMs ?? sum; // some steps run in parallel
  const cost = steps.reduce((a, s) => a + (s.cost_usd ?? 0), 0);
  return (
    <BottomSheet title="How Ibà decided" open={open} onClose={onClose}>
      <ol className="space-y-2">
        {steps.map((s) => {
          const ms = stepMs(s);
          const colour = s.kind === "rules" ? "bg-red-500" : s.reasoning ? "bg-violet-500" : s.kind === "retrieval" ? "bg-sky-500" : "bg-teal-500";
          return (
            <li key={s.step} className="rounded-lg border border-slate-200 p-2 text-sm">
              <div className="flex items-baseline justify-between gap-2">
                <span className="font-semibold">{STEP_NAMES[s.step] ?? s.step}</span>
                <span className="tabular-nums text-slate-500">{(ms / 1000).toFixed(1)} s</span>
              </div>
              <div className="mt-1 h-2 overflow-hidden rounded-full bg-slate-100">
                <div className={`h-full ${colour} grow-x`} style={{ width: `${Math.max(2, (ms / sum) * 100)}%` }} />
              </div>
              <div className="mt-1 flex flex-wrap gap-x-3 text-xs text-slate-600">
                <span>{s.model ? shortModel(s.model) : s.kind === "rules" ? "deterministic rules (not AI)" : s.kind}</span>
                {s.reasoning !== null && s.reasoning !== undefined && (
                  <span className={s.reasoning ? "font-semibold text-violet-700" : ""}>reasoning {s.reasoning ? "on" : "off"}</span>
                )}
                {s.calls > 0 && (
                  <span className="tabular-nums">
                    {s.prompt_tokens}→{s.completion_tokens} tok{s.reasoning_tokens ? ` (${s.reasoning_tokens} thinking)` : ""}
                  </span>
                )}
                {s.cost_usd != null && <span className="tabular-nums">${s.cost_usd.toFixed(5)}</span>}
                {s.cached && <span>replayed from cache</span>}
              </div>
              {s.note && <p className="mt-1 text-xs text-slate-500">{s.note}</p>}
            </li>
          );
        })}
      </ol>
      <p className="mt-3 text-sm font-semibold tabular-nums">Total {(total / 1000).toFixed(1)} s · ${cost.toFixed(4)}</p>
      <div className="mt-1 flex flex-wrap gap-3 text-xs text-slate-600">
        <span><i className="inline-block h-2 w-3 rounded bg-red-500" /> rules</span>
        <span><i className="inline-block h-2 w-3 rounded bg-teal-500" /> AI, reasoning off</span>
        <span><i className="inline-block h-2 w-3 rounded bg-violet-500" /> AI, reasoning on</span>
        <span><i className="inline-block h-2 w-3 rounded bg-sky-500" /> guideline search</span>
      </div>
      <p className="mt-2 text-xs text-slate-500">
        Models run on Nebius Token Factory (NVIDIA Nemotron). Danger signs, the triage floor, doses and
        citation checks are deterministic rules, not AI.
      </p>
    </BottomSheet>
  );
}

const TIMELINE = ["intake", "rules_pre", "outbreak", "retrieve", "reason", "rules_post", "compose"];
const TIMELINE_LABEL: Record<string, string> = {
  intake: "Reading the case",
  rules_pre: "Checking danger signs",
  retrieve: "Searching guidelines",
  outbreak: "Checking outbreaks",
  reason: "Reasoning (Nemotron)",
  rules_post: "Safety check",
  compose: "Writing referral note",
};

export function StepTimeline({ run }: { run: LiveRun }) {
  const times = new Map(run.trace.map((s) => [s.step, s.latency_ms]));
  const current = TIMELINE.find((s) => !run.done.includes(s));
  return (
    <ol className="fade-in grid grid-cols-1 gap-1 rounded-xl bg-white p-3 text-sm shadow-sm" aria-label="Progress">
      {TIMELINE.map((s) => {
        const done = run.done.includes(s);
        const active = s === current;
        return (
          <li key={s} className={`flex items-center gap-2 ${done ? "text-slate-700" : active ? "font-semibold text-teal-900" : "text-slate-400"}`}>
            <span className="flex h-5 w-5 items-center justify-center">
              {done ? <span className="text-teal-700">✓</span> : active ? <span className="spinner" /> : <span className="h-1.5 w-1.5 rounded-full bg-slate-300" />}
            </span>
            <span>{TIMELINE_LABEL[s]}</span>
            {done && times.has(s) && <span className="ml-auto tabular-nums text-xs text-slate-400">{((times.get(s) ?? 0) / 1000).toFixed(1)} s</span>}
          </li>
        );
      })}
    </ol>
  );
}
