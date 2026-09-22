import { useCallback, useEffect, useRef, useState } from "react";
import { streamTriage } from "./api";
import { About } from "./components/About";
import { Questions } from "./components/Questions";
import { ResultCard } from "./components/ResultCard";
import { CitationSheet, StepTimeline, TraceSheet } from "./components/Sheets";
import { NIGERIAN_STATES, loadState, saveState } from "./states";
import type { Citation, FollowUpAnswer, LiveRun, TriageRequest } from "./types";

// Tap to add common findings to the description (English and Pidgin).
const QUICK_ADD = [
  "RDT positive", "RDT negative", "fever 3 days", "vomit everything", "e dey convulse",
  "e no fit drink", "sleep too much", "yellow eyes", "bleeding", "took coartem, no better",
  "sore throat", "neck stiff", "watery stool",
];

const DEMOS = [
  {
    label: "Child with danger signs",
    state: "Kano",
    text: "Pikin 2 years, hot body 3 days, e dey convulse, e no fit drink, RDT positive",
  },
  {
    label: "Adult, Lassa suspicion (Ondo)",
    state: "Ondo",
    text: "Adult man 35 years, fever 5 days, RDT negative, took coartem for 3 days but no improvement, headache and sore throat",
  },
  {
    label: "Uncomplicated malaria",
    state: "Lagos",
    text: "Adult woman 28 years, 62 kg, fever 2 days, RDT positive, eating and drinking well",
  },
];

const STEPS = ["intake", "rules_pre", "outbreak", "retrieve", "reason", "rules_post", "compose"];

function usePath(): [string, (p: string) => void] {
  const [path, setPath] = useState(window.location.pathname);
  useEffect(() => {
    const onPop = () => setPath(window.location.pathname);
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);
  const go = (p: string) => {
    window.history.pushState(null, "", p);
    setPath(p);
    window.scrollTo(0, 0);
  };
  return [path, go];
}

interface CaseState {
  text: string;
  state: string;
  demo: boolean;
}

export default function App() {
  const [path, go] = usePath();
  const [region, setRegion] = useState(loadState);
  const [draft, setDraft] = useState("");
  const [current, setCurrent] = useState<CaseState | null>(null);
  const [run, setRun] = useState<LiveRun | null>(null);
  const [busy, setBusy] = useState(false);
  const [traceOpen, setTraceOpen] = useState(false);
  const [citation, setCitation] = useState<{ cite: Citation; quote?: string | null } | null>(null);
  const abort = useRef<AbortController | null>(null);

  const pickRegion = (value: string) => {
    setRegion(value);
    saveState(value);
  };

  const submit = useCallback(async (c: CaseState, extra: Partial<TriageRequest> = {}) => {
    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;
    setCurrent(c);
    setBusy(true);
    setRun({ done: [], trace: [] });
    const update = (fn: (r: LiveRun) => LiveRun) => setRun((r) => (r ? fn(r) : r));
    try {
      await streamTriage(
        { text: c.text, state: c.state || null, ...extra },
        (event, data) => {
          update((r) => {
            const next: LiveRun = { ...r, done: STEPS.includes(event) ? [...r.done, event] : r.done };
            if (data?.trace) next.trace = [...r.trace, data.trace];
            switch (event) {
              case "intake": next.questions = data.questions; break;
              case "rules_pre": next.floor = data.floor; next.floorLabel = data.floor_label; next.dangerSigns = data.danger_signs; break;
              case "outbreak": next.outbreak = data.outbreak; break;
              case "rules_post": next.post = data; break;
              case "compose": next.summary = data.summary; next.referralNote = data.referral_note; break;
              case "final":
                next.final = data;
                next.trace = data.decision_trace?.steps ?? next.trace;
                if (data.status !== "needs_info") {
                  next.post = next.post ?? data;
                  next.floor = data.triage_level;
                  next.floorLabel = data.triage_label;
                  next.dangerSigns = data.danger_signs;
                  next.summary = next.summary ?? data.summary;
                  next.referralNote = next.referralNote ?? data.referral_note;
                }
                break;
              case "error": next.error = data; break;
            }
            return next;
          });
        },
        controller.signal,
      );
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        update((r) => ({ ...r, error: { fatal: true, message: navigator.onLine ? (e as Error).message : "No internet connection. Iba needs the network to assess a case." } }));
      }
    } finally {
      setBusy(false);
    }
  }, []);

  const startCase = (text: string, state: string, demo = false) => {
    if (text.trim().length < 3) return;
    if (demo) pickRegion(state); // keep the header in step with the case being shown
    setDraft("");
    submit({ text: text.trim(), state, demo });
  };

  const quickAdd = (phrase: string) =>
    setDraft((d) => (d.trim() ? `${d.trim().replace(/[,.]$/, "")}, ${phrase}` : phrase));

  const needsInfo = run?.final?.status === "needs_info";

  return (
    <div className="mx-auto flex min-h-dvh max-w-xl flex-col bg-slate-100 text-slate-900">
      <header className="sticky top-0 z-20 flex items-center gap-2 bg-teal-800 px-3 pb-2 pt-[calc(0.5rem+env(safe-area-inset-top))] text-white shadow">
        <button onClick={() => go("/")} className="min-h-11 text-2xl font-extrabold tracking-tight" aria-label="Iba home">
          Iba
        </button>
        <label className="ml-auto flex items-center gap-1 text-sm">
          <span className="sr-only sm:not-sr-only">State</span>
          <select
            value={region}
            onChange={(e) => pickRegion(e.target.value)}
            className="min-h-11 max-w-[11rem] rounded-lg bg-white px-2 text-[15px] text-slate-900"
            aria-label="Patient's state"
          >
            <option value="">Select state…</option>
            {NIGERIAN_STATES.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <button onClick={() => go(path === "/about" ? "/" : "/about")} className="min-h-11 rounded-lg px-2 text-sm font-semibold underline">
          {path === "/about" ? "Back" : "About"}
        </button>
      </header>

      <main className="flex-1 space-y-3 px-3 pb-56 pt-3">
        {path === "/about" ? (
          <About />
        ) : (
          <>
            {!current && (
              <section className="space-y-3">
                <div className="rounded-xl bg-white p-4 shadow-sm">
                  <h1 className="text-lg font-bold">Describe a febrile patient</h1>
                  <p className="text-[15px] text-slate-600">
                    English or Pidgin. Age, days of fever, RDT result and any worrying signs. No names.
                  </p>
                  {!region && <p className="mt-2 text-sm font-semibold text-amber-800">Pick the patient's state above so Iba can check local outbreaks.</p>}
                </div>
                <div>
                  <p className="mb-2 text-sm font-bold uppercase tracking-wide text-slate-500">Try a demo case</p>
                  <div className="grid gap-2">
                    {DEMOS.map((d) => (
                      <button key={d.label} onClick={() => startCase(d.text, d.state, true)}
                              className="min-h-14 rounded-xl border border-teal-300 bg-white p-3 text-left shadow-sm active:bg-teal-50">
                        <span className="block font-semibold text-teal-900">{d.label}</span>
                        <span className="block text-sm text-slate-600">{d.state} · “{d.text.slice(0, 60)}…”</span>
                      </button>
                    ))}
                  </div>
                </div>
              </section>
            )}

            {current && (
              <>
                <div className="ml-8 rounded-2xl rounded-br-sm bg-teal-700 p-3 text-white shadow-sm">
                  <p className="text-[15px] leading-snug">{current.text}</p>
                  <p className="mt-1 text-xs opacity-80">{current.state || "No state selected"}{current.demo && " · demo"}</p>
                </div>

                {run && busy && !needsInfo && <StepTimeline run={run} />}

                {run?.error && (
                  <div className={`rounded-xl p-3 text-sm font-medium ${run.error.fatal ? "bg-red-100 text-red-900" : "bg-amber-100 text-amber-900"}`}>
                    {run.error.message}
                    {run.error.fatal && " If the patient is unwell, refer now."}
                  </div>
                )}

                {needsInfo && run?.final && (
                  <Questions
                    questions={run.final.questions}
                    disabled={busy}
                    onSubmit={(answers: FollowUpAnswer[]) => submit(current, { answers })}
                    onSkip={() => submit(current, { skip_questions: true })}
                  />
                )}

                {run && !needsInfo && (run.done.length > 0 || run.final) && !run.error?.fatal && (
                  <ResultCard run={run} state={current.state} onOpenTrace={() => setTraceOpen(true)} onOpenCitation={(cite, quote) => setCitation({ cite, quote })} />
                )}

                {!busy && (
                  <button onClick={() => { setCurrent(null); setRun(null); }}
                          className="min-h-11 w-full rounded-lg border border-slate-300 bg-white font-semibold">
                    New case
                  </button>
                )}
              </>
            )}
          </>
        )}
      </main>

      <div className="fixed inset-x-0 bottom-0 z-10 mx-auto max-w-xl">
        {path !== "/about" && (
          <div className="border-t border-slate-200 bg-white">
            {!busy && (
              <div className="no-scrollbar flex gap-1.5 overflow-x-auto px-2 pt-2" aria-label="Quick add">
                {QUICK_ADD.map((p) => (
                  <button key={p} type="button" onClick={() => quickAdd(p)}
                          className="min-h-9 shrink-0 rounded-full border border-teal-300 bg-white px-3 text-sm text-teal-900 active:scale-95 transition">
                    + {p}
                  </button>
                ))}
              </div>
            )}
            <form onSubmit={(e) => { e.preventDefault(); startCase(draft, region); }} className="flex gap-2 p-2">
              <textarea
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); startCase(draft, region); } }}
                rows={2}
                maxLength={4000}
                placeholder="e.g. Pikin 3 years, hot body 4 days, vomit everything, RDT negative"
                className="min-h-12 flex-1 resize-none rounded-lg border border-slate-300 px-3 py-2 text-[15px]"
                aria-label="Describe the patient"
              />
              <button type="submit" disabled={busy || draft.trim().length < 3}
                      className="min-h-12 rounded-lg bg-teal-800 px-4 font-bold text-white disabled:opacity-40">
                {busy ? <span className="spinner inline-block" /> : "Send"}
              </button>
            </form>
          </div>
        )}
        <footer className="bg-slate-900 px-3 pb-[calc(0.5rem+env(safe-area-inset-bottom))] pt-2 text-center text-xs font-semibold text-white">
          Decision support only — a health worker decides.
        </footer>
      </div>

      <TraceSheet steps={run?.trace ?? []} open={traceOpen} onClose={() => setTraceOpen(false)} />
      <CitationSheet citation={citation?.cite ?? null} quote={citation?.quote} onClose={() => setCitation(null)} />
    </div>
  );
}
