import type { TraceStep } from "../types";

const STEP_NAMES: Record<string, string> = {
  intake: "Understand the case",
  rules_pre: "Danger-sign rules",
  retrieve: "Find guideline passages",
  outbreak: "Outbreak check",
  reason: "Clinical reasoning",
  rules_post: "Safety rules",
  compose: "Write summary & note",
};

function shortModel(model?: string | null): string {
  if (!model) return "—";
  return model.split("/").pop()!.replace(/-\d+b-a\d+b$/i, "");
}

export function TraceDrawer({ steps, open, onClose }: { steps: TraceStep[]; open: boolean; onClose: () => void }) {
  if (!open) return null;
  const totalMs = steps.reduce((a, s) => a + s.latency_ms, 0);
  const totalCost = steps.reduce((a, s) => a + (s.cost_usd ?? 0), 0);
  return (
    <div className="fixed inset-0 z-30 flex items-end bg-black/40" onClick={onClose} role="dialog" aria-modal="true" aria-label="How Iba decided">
      <div className="max-h-[85vh] w-full overflow-y-auto rounded-t-2xl bg-white p-4 pb-[calc(1rem+env(safe-area-inset-bottom))]"
           onClick={(e) => e.stopPropagation()}>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-lg font-bold">How Iba decided</h2>
          <button onClick={onClose} className="min-h-11 min-w-11 rounded-lg text-2xl" aria-label="Close">×</button>
        </div>
        <ol className="space-y-2">
          {steps.map((s) => (
            <li key={s.step} className="rounded-lg border border-slate-200 p-2 text-sm">
              <div className="flex items-baseline justify-between gap-2">
                <span className="font-semibold">{STEP_NAMES[s.step] ?? s.step}</span>
                <span className="tabular-nums text-slate-500">{(s.latency_ms / 1000).toFixed(1)} s</span>
              </div>
              <div className="mt-0.5 flex flex-wrap gap-x-3 text-xs text-slate-600">
                <span>{s.model ? shortModel(s.model) : s.kind === "rules" ? "deterministic rules" : s.kind}</span>
                {s.reasoning !== null && s.reasoning !== undefined && (
                  <span className={s.reasoning ? "font-semibold text-violet-700" : ""}>reasoning {s.reasoning ? "on" : "off"}</span>
                )}
                {s.calls > 0 && (
                  <span className="tabular-nums">
                    {s.prompt_tokens}→{s.completion_tokens} tok{s.reasoning_tokens ? ` (${s.reasoning_tokens} thinking)` : ""}
                  </span>
                )}
                {s.cost_usd != null && <span className="tabular-nums">${s.cost_usd.toFixed(5)}</span>}
                {s.cached && <span>cached</span>}
              </div>
              {s.note && <p className="mt-1 text-xs text-slate-500">{s.note}</p>}
            </li>
          ))}
        </ol>
        <p className="mt-3 text-sm font-semibold tabular-nums">
          Total {(totalMs / 1000).toFixed(1)} s · ${totalCost.toFixed(4)}
        </p>
        <p className="mt-1 text-xs text-slate-500">
          Models run on Nebius Token Factory (NVIDIA Nemotron). Danger signs, the triage floor, doses and
          citation checks are deterministic rules, not AI.
        </p>
      </div>
    </div>
  );
}
