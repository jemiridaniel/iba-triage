import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type {
  ActionItem,
  Citation,
  Evidence,
  DangerSignHit,
  DifferentialItem,
  DoseRecommendation,
  GuidelineNote,
  LiveRun,
  OutbreakContext,
  OutbreakSignal,
  PatientCase,
  TriageLevel,
} from "../types";

const BANNER: Record<TriageLevel, { bg: string; icon: string }> = {
  refer_now: { bg: "bg-red-700", icon: "🔴" },
  refer_24h: { bg: "bg-amber-600", icon: "🟠" },
  treat_monitor: { bg: "bg-green-700", icon: "🟢" },
};

export const DOC_SHORT: Record<string, string> = {
  "ncdc-lassa": "NCDC Lassa guideline",
  "ncdc-vhf-ipc": "NCDC VHF IPC guideline",
  "ncdc-lassa-advisory-2026": "NCDC Lassa advisory 2026",
  "ncdc-cholera": "NCDC cholera guideline",
  "ncdc-csm": "NCDC CSM guide",
  "ncdc-csm-quickref": "NCDC CSM quick reference",
  "ncdc-case-definitions": "NCDC case definitions",
  "who-malaria": "WHO malaria guidelines",
  "who-imci": "WHO IMCI chart booklet",
  "nmep-malaria": "NMEP malaria guideline",
};

export function citationLabel(c: Citation): string {
  if (c.kind === "outbreak") return c.title ?? "Outbreak report";
  const doc = (c.doc_id && DOC_SHORT[c.doc_id]) || c.title || c.ref;
  const pages = c.page ? (c.page_end && c.page_end !== c.page ? ` pp.${c.page}–${c.page_end}` : ` p.${c.page}`) : "";
  return `${doc}${pages}`;
}

type OpenCitation = (c: Citation, quote?: string | null) => void;

const AI_SUGGESTION = "AI suggestion — no guideline source";

/** One claim with its grounding: verified -> chip showing the quote; unsupported -> grey. */
function Claim({ text, evidence, index, onOpen }: { text: string; evidence?: Evidence | null; index: Map<string, Citation>; onOpen: OpenCitation }) {
  if (evidence?.status === "unsupported") {
    return (
      <span className="text-slate-500">
        {text} <span className="ml-1 rounded bg-slate-100 px-1.5 py-0.5 text-[11px] font-medium text-slate-500">{AI_SUGGESTION}</span>
      </span>
    );
  }
  const cite = evidence?.status === "verified" && evidence.chunk_id ? index.get(evidence.chunk_id) : undefined;
  return (
    <span>
      {text}
      {cite && (
        <button type="button" onClick={() => onOpen(cite, evidence?.quote)}
                className="ml-1.5 rounded-md bg-slate-100 px-1.5 py-0.5 text-xs font-medium text-slate-700 ring-1 ring-slate-200 active:scale-95">
          📖 {citationLabel(cite)}
        </button>
      )}
    </span>
  );
}

function Cites({ refs, index, onOpen }: { refs: string[]; index: Map<string, Citation>; onOpen: OpenCitation }) {
  const items = refs.map((r) => index.get(r)).filter((c): c is Citation => Boolean(c));
  if (!items.length) return null;
  return (
    <div className="mt-1.5 flex flex-wrap gap-1.5">
      {items.map((c) => (
        <button
          key={c.ref}
          type="button"
          onClick={() => onOpen(c)}
          className={`min-h-8 rounded-md px-2 py-1 text-left text-xs font-medium ${
            c.kind === "outbreak" ? "bg-sky-50 text-sky-900 ring-1 ring-sky-200" : "bg-slate-100 text-slate-700 ring-1 ring-slate-200"
          } active:scale-95 transition`}
        >
          📖 {citationLabel(c)}
        </button>
      ))}
    </div>
  );
}

function Section({ title, children, delay = 0 }: { title: string; children: React.ReactNode; delay?: number }) {
  return (
    <section className="fade-in border-t border-slate-200 px-4 py-3" style={{ animationDelay: `${delay}ms` }}>
      <h3 className="mb-2 text-sm font-bold uppercase tracking-wide text-slate-500">{title}</h3>
      {children}
    </section>
  );
}

function Pending({ label }: { label: string }) {
  return (
    <div className="space-y-2" aria-label={label}>
      <p className="text-sm text-slate-400">{label}…</p>
      <div className="h-3 w-3/4 animate-pulse rounded bg-slate-200" />
      <div className="h-3 w-1/2 animate-pulse rounded bg-slate-200" />
    </div>
  );
}

function useElapsed(running: boolean): number {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (!running) return;
    const start = Date.now();
    const id = setInterval(() => setSeconds(Math.floor((Date.now() - start) / 1000)), 500);
    return () => clearInterval(id);
  }, [running]);
  return seconds;
}

function Banner({ run }: { run: LiveRun }) {
  const level = run.post?.triage_level ?? run.floor ?? null;
  const running = !run.final && !run.error;
  const elapsed = useElapsed(running);
  const buzzed = useRef(false);
  useEffect(() => {
    if (level === "refer_now" && !buzzed.current) {
      buzzed.current = true;
      navigator.vibrate?.([120, 80, 120]);
    }
  }, [level]);

  if (!level) {
    return (
      <div className="bg-slate-600 px-4 py-4 text-white">
        <p className="text-xl font-bold">Assessing… <span className="tabular-nums text-base font-medium opacity-80">{elapsed}s</span></p>
        <p className="text-sm opacity-90">Danger signs are checked first.</p>
      </div>
    );
  }
  const provisional = !run.post;
  const label = run.post?.triage_label ?? run.floorLabel ?? level;
  return (
    <div className={`${BANNER[level].bg} pop-in px-4 py-4 text-white`}>
      <p className="text-2xl font-extrabold leading-tight">
        {BANNER[level].icon} {label}
        {run.post?.status === "incomplete" && <span className="ml-2 text-base font-semibold">(assessment incomplete)</span>}
      </p>
      {provisional ? (
        <>
          {run.floorReasons && run.floorReasons.length > 0 && (
            <p className="mt-1 text-sm font-semibold">RULE: {run.floorReasons.join("; ")}</p>
          )}
          <p className="mt-1 text-sm font-medium">
            Safety rules already require at least this level. Finishing assessment… <span className="tabular-nums">{elapsed}s</span>
          </p>
        </>
      ) : (
        run.post?.triage_rationale && <p className="mt-1 text-sm leading-snug opacity-95">{run.post.triage_rationale}</p>
      )}
    </div>
  );
}

function DangerSigns({ signs }: { signs: DangerSignHit[] }) {
  if (!signs.length) return <p className="text-sm text-slate-600">✓ No danger signs detected.</p>;
  return (
    <ul className="space-y-2">
      {signs.map((s) => (
        <li key={s.code} className="flex flex-wrap items-center gap-2">
          <span className="font-semibold text-red-800">⚠ {s.label}</span>
          {s.source === "rule" ? (
            <span className="rounded-full bg-red-100 px-2 py-0.5 text-xs font-bold text-red-800">RULE-TRIGGERED</span>
          ) : (
            <span className="rounded-full bg-violet-100 px-2 py-0.5 text-xs font-bold text-violet-800">AI-FLAGGED</span>
          )}
          {s.evidence && <span className="text-xs text-slate-500">“{s.evidence}”</span>}
        </li>
      ))}
    </ul>
  );
}

function when(iso?: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

const normState = (s: string) => s.trim().toLowerCase().replace(/\s+state$/, "").replace(/[^a-z0-9]+/g, "-");
const signalKey = (s: OutbreakSignal) => `${s.disease}-${s.state}-${s.url}`;

/** One live signal. A report we can't date, or one over 60 days old, says so plainly:
 *  it still informs the differential but never raises the triage level on its own. */
function Signal({ s, muted = false }: { s: OutbreakSignal; muted?: boolean }) {
  const stale = s.recency === "older" || s.recency === "unknown";
  return (
    <p className={`rounded-md p-2 ${muted || stale ? "bg-slate-50 text-slate-700 ring-1 ring-slate-200" : "bg-sky-50"}`}>
      <strong>{s.disease}</strong>: {s.status} in {s.state}
      {s.recency === "unknown" ? (
        <> · <span className="font-semibold text-amber-800">date unknown</span></>
      ) : s.recency === "older" ? (
        <> · <span className="font-semibold text-amber-800">older report</span> {s.report_date}</>
      ) : (
        s.report_date && <> · report {s.report_date}</>
      )}
      {s.url && (
        <> · <a className="font-medium text-sky-800 underline" href={s.url} target="_blank" rel="noreferrer">source ↗</a></>
      )}
    </p>
  );
}

/** One baseline (static endemicity) entry: which NCDC report named it, and how stale that
 *  report is. A report over ~6 months old is labelled "older surveillance" -- descriptive
 *  only, unlike live signals this never blocks escalation (tier does that). */
function BaselineEntry({ s, index, onOpen }: { s: OutbreakSignal; index: Map<string, Citation>; onOpen: OpenCitation }) {
  const source = [s.report, s.epi_week, s.page ? `p.${s.page}` : null].filter(Boolean).join(", ");
  return (
    <div className="mt-1">
      <strong>{s.disease}</strong>: {s.state}
      {s.in_season && " — now in its usual peak season"}
      {s.confirmed === false && (
        <span className="ml-1 font-semibold text-amber-800">(not independently confirmed)</span>
      )}
      {source && (
        <span className="text-slate-500">
          {" "}
          · {source}
          {s.recency === "older" && <span className="font-semibold text-amber-800"> · older surveillance</span>}
        </span>
      )}
      {s.citation && <Cites refs={[s.citation]} index={index} onOpen={onOpen} />}
    </div>
  );
}

function Outbreak({ ctx, index, onOpen, state }: { ctx: OutbreakContext; index: Map<string, Citation>; onOpen: OpenCitation; state: string }) {
  const live = ctx.status === "ok";
  // NCDC sitreps are national, so a search for one state returns outbreaks in others. Only
  // the patient's own state can raise their triage level, so the two are shown apart.
  const target = state ? normState(state) : "";
  const here = ctx.signals.filter((s) => target && normState(s.state) === target);
  const elsewhere = ctx.signals.filter((s) => !here.includes(s));
  return (
    <div className="space-y-2 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        <span className={`rounded-full px-2 py-0.5 text-xs font-bold ${live ? "bg-sky-100 text-sky-800" : "bg-slate-200 text-slate-600"}`}>
          {live ? "● LIVE" : "○ LIVE: OFF"}
        </span>
        {ctx.source === "mock" && (
          <span className="rounded-full bg-fuchsia-100 px-2 py-0.5 text-xs font-bold text-fuchsia-800">MOCK TEST DATA</span>
        )}
        <span className={live ? "text-slate-600" : "text-slate-600"}>
          {live ? `Last checked ${when(ctx.checked_at)}` : ctx.message.replace(" Using baseline endemicity only.", "")}
        </span>
      </div>
      {live && !ctx.signals.length && <p className="text-slate-600">No current outbreak reports found for {state}.</p>}
      {here.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs font-bold uppercase tracking-wide text-sky-900">In {state}</p>
          {here.map((s) => <Signal key={signalKey(s)} s={s} />)}
        </div>
      )}
      {elsewhere.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs font-bold uppercase tracking-wide text-slate-500">
            Elsewhere in Nigeria · context only
          </p>
          <p className="text-xs text-slate-500">These do not change this patient's triage level.</p>
          {elsewhere.map((s) => <Signal key={signalKey(s)} s={s} muted />)}
        </div>
      )}
      {ctx.baseline.length > 0 ? (
        <div className="rounded-md bg-slate-50 p-2 ring-1 ring-slate-200">
          {!live && <span className="text-xs text-slate-500">used because live data is off</span>}
          {ctx.baseline.filter((s) => s.tier === "high_burden").length > 0 && (
            <div>
              <span className="rounded-full bg-slate-200 px-2 py-0.5 text-xs font-bold text-slate-700">BASELINE · high burden</span>
              {ctx.baseline
                .filter((s) => s.tier === "high_burden")
                .map((s) => <BaselineEntry key={`b-${s.disease}-${s.state}`} s={s} index={index} onOpen={onOpen} />)}
            </div>
          )}
          {ctx.baseline.filter((s) => s.tier === "reported").length > 0 && (
            <div className="mt-2">
              <span className="rounded-full bg-slate-200 px-2 py-0.5 text-xs font-bold text-slate-700">BASELINE · reported (context only)</span>
              {ctx.baseline
                .filter((s) => s.tier === "reported")
                .map((s) => <BaselineEntry key={`b-${s.disease}-${s.state}`} s={s} index={index} onOpen={onOpen} />)}
            </div>
          )}
        </div>
      ) : (
        <p className="text-xs text-slate-500">No endemic diseases listed for {state || "this state"} in Ibà's baseline.</p>
      )}
    </div>
  );
}

const LIKELIHOOD: Record<string, string> = {
  high: "bg-red-100 text-red-800",
  moderate: "bg-amber-100 text-amber-800",
  low: "bg-slate-100 text-slate-700",
};

function DifferentialRow({ d, index, onOpen, open: initiallyOpen }: { d: DifferentialItem; index: Map<string, Citation>; onOpen: OpenCitation; open: boolean }) {
  const [open, setOpen] = useState(initiallyOpen);
  const hasDetail = d.reasons.length > 0 || d.check_next.length > 0;
  return (
    <li className="rounded-lg ring-1 ring-slate-200">
      <button type="button" onClick={() => setOpen((o) => !o)} disabled={!hasDetail}
              className="flex min-h-11 w-full flex-wrap items-center gap-2 px-3 py-2 text-left" aria-expanded={open}>
        <span className="font-semibold">{d.condition}</span>
        <span className={`rounded-full px-2 py-0.5 text-xs font-bold ${LIKELIHOOD[d.likelihood]}`}>{d.likelihood}</span>
        {d.source === "rule" && <span className="rounded-full bg-red-100 px-2 py-0.5 text-xs font-bold text-red-800">RULE</span>}
        {hasDetail && <span className="ml-auto text-slate-400">{open ? "▴" : "▾"}</span>}
      </button>
      {open && (
        <div className="px-3 pb-2">
          {d.reasons.length > 0 && (
            <ul className="ml-4 list-disc space-y-1 text-sm text-slate-700">
              {d.reasons.map((r) => (
                <li key={r.text}><Claim text={r.text} evidence={r.evidence} index={index} onOpen={onOpen} /></li>
              ))}
            </ul>
          )}
          {d.check_next.length > 0 && <p className="mt-1 text-sm text-slate-600"><strong>Check next:</strong> {d.check_next.join("; ")}</p>}
          {d.source === "rule" && <Cites refs={d.citations} index={index} onOpen={onOpen} />}
        </div>
      )}
    </li>
  );
}

function isUrgentRule(a: ActionItem): boolean {
  return a.source === "rule" && /^(refer now|suspected lassa)/i.test(a.text);
}

function Actions({ items, index, onOpen }: { items: ActionItem[]; index: Map<string, Citation>; onOpen: OpenCitation }) {
  const [done, setDone] = useState<Set<number>>(new Set());
  const toggle = (i: number) => setDone((d) => { const n = new Set(d); n.has(i) ? n.delete(i) : n.add(i); return n; });
  return (
    <ol className="space-y-2">
      {items.map((a, i) => {
        const unsupported = a.evidence?.status === "unsupported";
        const style = isUrgentRule(a)
          ? "border-l-4 border-red-600 bg-red-50"
          : a.source === "rule" ? "border-l-4 border-teal-600 bg-teal-50"
          : unsupported ? "bg-slate-50 ring-1 ring-slate-200" : "bg-white ring-1 ring-slate-200";
        return (
          <li key={i} className={`rounded-md p-2 ${style}`}>
            <label className="flex cursor-pointer gap-2">
              <input type="checkbox" checked={done.has(i)} onChange={() => toggle(i)}
                     className="mt-1 h-5 w-5 shrink-0 accent-teal-700" aria-label="Mark as done" />
              <span className={`text-[15px] leading-snug ${done.has(i) ? "text-slate-400 line-through" : ""}`}>
                {a.source === "rule" && <span className="mr-1 rounded bg-white/70 px-1 text-[10px] font-bold uppercase tracking-wide text-slate-600 ring-1 ring-slate-300">Safety rule</span>}
                {a.source === "rule" ? a.text : <Claim text={a.text} evidence={a.evidence} index={index} onOpen={onOpen} />}
              </span>
            </label>
            {a.details.length > 0 && (
              <ul className="ml-11 mt-1 list-disc space-y-1 text-sm text-slate-600">
                {a.details.map((d) => (
                  <li key={d.text}><Claim text={d.text} evidence={d.evidence} index={index} onOpen={onOpen} /></li>
                ))}
              </ul>
            )}
            {a.source === "rule" && <div className="ml-7"><Cites refs={a.citations} index={index} onOpen={onOpen} /></div>}
          </li>
        );
      })}
    </ol>
  );
}

function DoseCard({ d, onOpen }: { d: DoseRecommendation; onOpen: OpenCitation }) {
  return (
    <div className="rounded-md border border-amber-300 bg-amber-50 p-2 text-sm">
      {!d.verified && (
        <p className="mb-1 font-bold text-amber-900">⚠ UNVERIFIED dose table: check against the printed national guideline before use.</p>
      )}
      <p><strong>{d.drug}</strong> · {d.weight_band}</p>
      <p>{d.regimen}</p>
      <Cites refs={[d.citation.ref]} index={new Map([[d.citation.ref, d.citation]])} onOpen={onOpen} />
    </div>
  );
}

function Doses({ doses, notes, onOpen }: { doses: DoseRecommendation[]; notes: GuidelineNote[]; onOpen: OpenCitation }) {
  const primary = doses.find((d) => d.is_default) ?? doses[0];
  const alternatives = doses.filter((d) => d !== primary);
  return (
    <div className="space-y-2">
      {primary && <DoseCard d={primary} onOpen={onOpen} />}
      {alternatives.length > 0 && (
        <details className="text-sm">
          <summary className="cursor-pointer font-semibold text-amber-900">
            Alternative tablet strength{alternatives.length > 1 ? "s" : ""} ({alternatives.map((d) => d.strength).join(", ")})
          </summary>
          <div className="mt-2 space-y-2">
            {alternatives.map((d) => <DoseCard key={d.drug} d={d} onOpen={onOpen} />)}
          </div>
        </details>
      )}
      {notes.length > 0 && (
        <div className="space-y-1 text-xs text-slate-600">
          {notes.map((n) => (
            <p key={n.text}>
              {n.text}{" "}
              <Cites refs={[n.citation.ref]} index={new Map([[n.citation.ref, n.citation]])} onOpen={onOpen} />
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

/** One labelled block of the printed form. Bordered, not shaded -- monochrome PHC printers
 *  need borders and weight to carry hierarchy, not colour. Never split across a page. */
function PrintBlock({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="print-block mb-1 border border-black">
      <h3 className="border-b border-black px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider">
        {title}
      </h3>
      <div className="px-2 py-1 text-[11.5px] leading-tight">{children}</div>
    </section>
  );
}

function PrintRow({ label, value }: { label: string; value: string }) {
  return (
    <p>
      <span className="font-semibold">{label}:</span> {value}
    </p>
  );
}

/** A labelled ruled line for the health worker to write on by hand. */
function FillLine({ label, tall = false }: { label: string; tall?: boolean }) {
  return (
    <div className="flex items-end gap-2">
      <span className="whitespace-nowrap text-[11px] font-semibold">{label}:</span>
      <span className={`flex-1 border-b border-black ${tall ? "h-8" : "h-4"}`} />
    </div>
  );
}

/** Tavily sometimes finds more than one live result for the same disease/state -- e.g. one
 *  dated, one not. Merge those into a single line rather than showing both: keep the dated
 *  one (or, if several are dated, the first). Baseline entries are never touched here. */
function dedupeLiveSignals(signals: OutbreakSignal[]): OutbreakSignal[] {
  const byKey = new Map<string, OutbreakSignal>();
  for (const s of signals) {
    const key = `${s.disease}|${s.state}`;
    const existing = byKey.get(key);
    if (!existing || (existing.recency === "unknown" && s.recency !== "unknown")) {
      byKey.set(key, s);
    }
  }
  return [...byKey.values()];
}

function outbreakSourceLabel(s: OutbreakSignal): string {
  if (s.basis === "live") {
    if (s.recency === "unknown") return "live report, date unknown";
    if (s.recency === "older") return `live report, ${s.report_date} (older report)`;
    return `live report, ${s.report_date}`;
  }
  const parts = [s.report, s.epi_week, s.page ? `p.${s.page}` : null].filter(Boolean);
  const source = parts.length ? parts.join(", ") : "Ibà baseline";
  return s.recency === "older" ? `${source} (older surveillance)` : source;
}

/** The printed referral form: A4, monochrome, one page where possible. Built from the
 *  structured result (not the plain-text note) so every section can be properly labelled. */
function PrintableNote({ run, state }: { run: LiveRun; state: string }) {
  const post = run.post;
  const final = run.final;
  const c: PatientCase | null | undefined = final?.case;
  const level = final?.triage_label ?? post?.triage_level ?? "";
  const differential = post?.differential ?? [];
  const dangerSigns = post?.danger_signs ?? run.dangerSigns ?? [];
  const outbreakSignals = [
    ...dedupeLiveSignals(run.outbreak?.signals ?? []),
    ...(run.outbreak?.baseline ?? []),
  ];
  const disclaimer = final?.disclaimer ?? "Decision support only, not a diagnosis. A qualified health worker must confirm.";
  const dateStr = new Date().toLocaleDateString(undefined, { day: "2-digit", month: "short", year: "numeric" });

  const subtitle = post?.lassa_suspected
    ? "SUSPECTED LASSA FEVER"
    : differential[0]
      ? `SUSPECTED ${differential[0].condition.toUpperCase()}`
      : null;

  const dangerCodes = new Set(dangerSigns.map((d) => d.code));
  const humanize = (s: string) => s.replace(/_/g, " ").replace(/^./, (ch) => ch.toUpperCase());
  const findings: string[] = [];
  if (c?.fever_days != null) findings.push(`Fever for ${c.fever_days} day(s)`);
  if (c?.temperature_c != null) findings.push(`Temperature ${c.temperature_c}°C`);
  if (c?.rdt_result) findings.push(`Malaria RDT: ${c.rdt_result.replace("_", " ")}`);
  // Danger signs are listed on their own line below; don't repeat the same code here.
  for (const s of c?.symptoms ?? []) if (!dangerCodes.has(s)) findings.push(humanize(s));

  const printRoot = typeof document !== "undefined" ? document.getElementById("print-root") : null;
  if (!printRoot) return null;

  return createPortal(
    <div id="print-note">
      <header className="print-block mb-1.5 flex items-baseline justify-between border-b-2 border-black pb-0.5">
        <p className="text-base font-extrabold tracking-wide">IBÀ TRIAGE NOTE</p>
        <p className="text-xs font-semibold">{dateStr}</p>
      </header>

      <div className="print-block mb-1.5 border-4 border-black p-1.5 text-center">
        <p className="text-xl font-extrabold uppercase leading-tight tracking-wide">
          {level}
          {subtitle && <><br />{subtitle}</>}
        </p>
      </div>

      <PrintBlock title="Patient">
        <div className="grid grid-cols-2 gap-x-4">
          <PrintRow label="Age" value={c?.age_years != null ? `${c.age_years} years` : "—"} />
          <PrintRow label="Sex" value={c?.sex ?? "—"} />
          <PrintRow label="State / LGA" value={[c?.state ?? state, c?.lga].filter(Boolean).join(" / ") || "—"} />
          {c?.weight_kg != null && <PrintRow label="Weight" value={`${c.weight_kg} kg`} />}
          {c?.pregnant ? <PrintRow label="Pregnant" value="Yes" /> : null}
        </div>
      </PrintBlock>

      <PrintBlock title="Presenting findings">
        {findings.length || dangerSigns.length ? (
          <ul className="list-disc pl-4">
            {findings.map((f) => <li key={f}>{f}</li>)}
            {dangerSigns.map((d) => <li key={d.code} className="font-bold">Danger sign: {d.label}</li>)}
          </ul>
        ) : (
          <p>None recorded.</p>
        )}
      </PrintBlock>

      <PrintBlock title="Triage decision">
        <p className="font-bold uppercase">{level}</p>
        {post?.triage_rationale && <p className="mt-0.5">{post.triage_rationale}</p>}
      </PrintBlock>

      {differential.length > 0 && (
        <PrintBlock title="Suspected condition">
          <ol className="list-decimal space-y-0.5 pl-4">
            {differential.map((d) => (
              <li key={d.condition}>{d.condition} <span className="italic">({d.likelihood})</span></li>
            ))}
          </ol>
        </PrintBlock>
      )}

      <PrintBlock title="Advised actions">
        <ol className="list-decimal space-y-0.5 pl-4">
          {(post?.actions ?? []).map((a, i) => <li key={i}>{a.text}</li>)}
          {(post?.doses ?? []).map((d) => (
            <li key={d.drug}>
              {d.drug}: {d.regimen}{!d.verified && " (UNVERIFIED table)"}
            </li>
          ))}
          {(post?.dosing_notes ?? []).map((n) => <li key={n.text}>{n.text}</li>)}
        </ol>
      </PrintBlock>

      <PrintBlock title="Outbreak context">
        {outbreakSignals.length ? (
          <ul className="list-disc space-y-0.5 pl-4">
            {outbreakSignals.map((s, i) => (
              <li key={`${s.disease}-${s.state}-${i}`}>
                {s.disease} — {s.state} ({outbreakSourceLabel(s)})
              </li>
            ))}
          </ul>
        ) : (
          <p>{run.outbreak?.message ?? "No outbreak data checked."}</p>
        )}
      </PrintBlock>

      <section className="print-block mt-1.5 border-2 border-black p-1.5">
        <h3 className="mb-1.5 border-b border-black pb-0.5 text-[11px] font-bold uppercase tracking-wider">
          To be completed by health worker
        </h3>
        <div className="space-y-1.5">
          <FillLine label="Referred to (facility)" />
          <FillLine label="Referred by (name and cadre)" />
          <div className="grid grid-cols-3 gap-4">
            <FillLine label="Date" />
            <FillLine label="Time" />
            <FillLine label="Phone" />
          </div>
          <FillLine label="Notes" tall />
        </div>
      </section>

      <footer className="print-block mt-1.5 border-t border-black pt-1 text-[9px] leading-tight">
        <p>{disclaimer}</p>
        <p className="mt-0.5 italic">Generated by Ibà — decision support only. Not a diagnosis.</p>
      </footer>
    </div>,
    printRoot,
  );
}

function ReferralNote({ note, run, state }: { note: string; run: LiveRun; state: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(note);
    } catch {
      const el = document.createElement("textarea");
      el.value = note;
      document.body.appendChild(el);
      el.select();
      document.execCommand("copy");
      el.remove();
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  const canShare = typeof navigator.share === "function";
  return (
    <div>
      <pre className="whitespace-pre-wrap rounded-md bg-slate-50 p-3 font-sans text-sm leading-relaxed ring-1 ring-slate-200">{note}</pre>
      <PrintableNote run={run} state={state} />
      <div className="mt-2 grid grid-cols-2 gap-2">
        <button onClick={copy} className="min-h-11 rounded-lg border border-slate-300 bg-white px-3 font-semibold active:scale-95 transition">
          {copied ? "Copied ✓" : "Copy"}
        </button>
        <a href={`https://wa.me/?text=${encodeURIComponent(note)}`} target="_blank" rel="noreferrer"
           className="flex min-h-11 items-center justify-center rounded-lg bg-green-600 px-3 font-semibold text-white active:scale-95 transition">
          WhatsApp
        </a>
        {canShare && (
          <button onClick={() => navigator.share({ title: "Ibà triage note", text: note }).catch(() => {})}
                  className="min-h-11 rounded-lg border border-slate-300 bg-white px-3 font-semibold active:scale-95 transition">
            Share…
          </button>
        )}
        <button onClick={() => window.print()}
                className={`min-h-11 rounded-lg border border-slate-300 bg-white px-3 font-semibold active:scale-95 transition ${canShare ? "" : "col-span-2"}`}>
          Print
        </button>
      </div>
    </div>
  );
}

export function ResultCard({ run, state, onOpenTrace, onOpenCitation }: {
  run: LiveRun; state: string; onOpenTrace: () => void; onOpenCitation: OpenCitation;
}) {
  const post = run.post;
  const index = new Map((post?.citations ?? []).map((c) => [c.ref, c] as const));
  const signs = post?.danger_signs ?? run.dangerSigns;
  const warnings = run.final?.warnings ?? post?.warnings ?? [];
  const running = !run.final && !run.error;

  return (
    <article className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
      <Banner run={run} />
      <Section title="Danger signs">{signs ? <DangerSigns signs={signs} /> : <Pending label="Checking" />}</Section>
      <Section title="Outbreak context">
        {run.outbreak ? <Outbreak ctx={run.outbreak} index={index} onOpen={onOpenCitation} state={state} /> : <Pending label="Checking outbreak reports" />}
      </Section>
      {post ? (
        <>
          {post.differential.length > 0 && (
            <Section title="Consider (differential)">
              <ol className="space-y-2">
                {post.differential.map((d, i) => (
                  <DifferentialRow key={d.condition} d={d} index={index} onOpen={onOpenCitation} open={i === 0} />
                ))}
              </ol>
            </Section>
          )}
          <Section title="Actions" delay={80}><Actions items={post.actions} index={index} onOpen={onOpenCitation} /></Section>
          {post.doses.length > 0 && <Section title="Dose (from table only)" delay={120}><Doses doses={post.doses} notes={post.dosing_notes} onOpen={onOpenCitation} /></Section>}
        </>
      ) : (
        running && <Section title="Assessment"><Pending label="Reasoning over guidelines and outbreak data" /></Section>
      )}
      {(run.summary || run.referralNote) && (
        <Section title="Summary & referral note">
          {run.summary && <p className="mb-2 text-[15px] leading-snug">{run.summary}</p>}
          {run.referralNote && <ReferralNote note={run.referralNote} run={run} state={state} />}
        </Section>
      )}
      {post && !run.referralNote && running && <Section title="Referral note"><Pending label="Writing note" /></Section>}
      {warnings.length > 0 && (
        <Section title="Notes">
          <ul className="space-y-1 text-sm text-amber-900">{warnings.map((w) => <li key={w}>! {w}</li>)}</ul>
        </Section>
      )}
      {run.trace.length > 0 && (
        <button onClick={onOpenTrace}
                className="flex w-full items-center justify-between border-t border-slate-200 px-4 py-3 text-left text-sm font-semibold text-teal-800">
          <span>How Ibà decided</span>
          <span className="text-xs font-normal text-slate-500">
            {run.trace.length} steps · ${run.trace.reduce((a, s) => a + (s.cost_usd ?? 0), 0).toFixed(4)} ›
          </span>
        </button>
      )}
    </article>
  );
}
