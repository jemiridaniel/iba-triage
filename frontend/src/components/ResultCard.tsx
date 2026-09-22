import { useState } from "react";
import type {
  ActionItem,
  Citation,
  DangerSignHit,
  DifferentialItem,
  DoseRecommendation,
  LiveRun,
  OutbreakContext,
  TriageLevel,
} from "../types";

const BANNER: Record<TriageLevel, { bg: string; icon: string }> = {
  refer_now: { bg: "bg-red-700", icon: "🔴" },
  refer_24h: { bg: "bg-amber-600", icon: "🟠" },
  treat_monitor: { bg: "bg-green-700", icon: "🟢" },
};

const DOC_SHORT: Record<string, string> = {
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

function citationLabel(c: Citation): string {
  if (c.kind === "outbreak") return c.title ?? "Outbreak report";
  const doc = (c.doc_id && DOC_SHORT[c.doc_id]) || c.title || c.ref;
  const pages = c.page ? (c.page_end && c.page_end !== c.page ? ` pp.${c.page}–${c.page_end}` : ` p.${c.page}`) : "";
  return `${doc}${pages}`;
}

function Cites({ refs, index }: { refs: string[]; index: Map<string, Citation> }) {
  const items = refs.map((r) => index.get(r)).filter((c): c is Citation => Boolean(c));
  if (!items.length) return null;
  return (
    <div className="mt-1 flex flex-wrap gap-1">
      {items.map((c) =>
        c.url ? (
          <a key={c.ref} href={c.url} target="_blank" rel="noreferrer"
             className="rounded bg-sky-50 px-1.5 py-0.5 text-xs text-sky-800 underline">
            {citationLabel(c)} ↗
          </a>
        ) : (
          <span key={c.ref} title={c.section ?? undefined}
                className="rounded bg-slate-100 px-1.5 py-0.5 text-xs text-slate-700">
            {citationLabel(c)}
          </span>
        ),
      )}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="border-t border-slate-200 px-4 py-3">
      <h3 className="mb-2 text-sm font-bold uppercase tracking-wide text-slate-500">{title}</h3>
      {children}
    </section>
  );
}

function Pending({ label }: { label: string }) {
  return <p className="animate-pulse text-sm text-slate-400">{label}…</p>;
}

function Banner({ run }: { run: LiveRun }) {
  const level = run.post?.triage_level ?? run.floor ?? null;
  if (!level) {
    return (
      <div className="bg-slate-600 px-4 py-4 text-white">
        <p className="text-xl font-bold">Assessing…</p>
        <p className="text-sm opacity-90">Checking danger signs first.</p>
      </div>
    );
  }
  const provisional = !run.post;
  const label = run.post?.triage_label ?? run.floorLabel ?? level;
  return (
    <div className={`${BANNER[level].bg} px-4 py-4 text-white`}>
      <p className="text-2xl font-extrabold leading-tight">
        {BANNER[level].icon} {label}
        {run.post?.status === "incomplete" && <span className="ml-2 text-base font-semibold">(assessment incomplete)</span>}
      </p>
      {provisional ? (
        <p className="mt-1 text-sm font-medium">Safety rules already require at least this level. Finishing assessment…</p>
      ) : (
        run.post?.triage_rationale && <p className="mt-1 text-sm leading-snug opacity-95">{run.post.triage_rationale}</p>
      )}
    </div>
  );
}

function DangerSigns({ signs }: { signs: DangerSignHit[] }) {
  if (!signs.length) return <p className="text-sm text-slate-600">No danger signs detected.</p>;
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

function Outbreak({ ctx, index }: { ctx: OutbreakContext; index: Map<string, Citation> }) {
  return (
    <div className="space-y-2 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded-full bg-sky-100 px-2 py-0.5 text-xs font-bold text-sky-800">LIVE</span>
        {ctx.source === "mock" && (
          <span className="rounded-full bg-fuchsia-100 px-2 py-0.5 text-xs font-bold text-fuchsia-800">MOCK TEST DATA</span>
        )}
        {ctx.status === "ok" ? (
          <span className="text-slate-600">Last checked {when(ctx.checked_at)}</span>
        ) : (
          <span className="font-medium text-amber-800">{ctx.message}</span>
        )}
      </div>
      {ctx.status === "ok" && !ctx.signals.length && <p className="text-slate-600">No current outbreak reports found.</p>}
      {ctx.signals.map((s) => (
        <p key={`${s.disease}-${s.state}-${s.url}`}>
          <strong>{s.disease}</strong>: {s.status} in {s.state}
          {s.report_date && <> · report {s.report_date}</>}
          {s.url && (
            <> · <a className="text-sky-800 underline" href={s.url} target="_blank" rel="noreferrer">source ↗</a></>
          )}
        </p>
      ))}
      {ctx.baseline.length > 0 && (
        <div className="rounded-md bg-slate-50 p-2">
          <span className="rounded-full bg-slate-200 px-2 py-0.5 text-xs font-bold text-slate-700">BASELINE (provisional)</span>
          {ctx.baseline.map((s) => (
            <p key={`b-${s.disease}`} className="mt-1">
              <strong>{s.disease}</strong>: {s.state} is an endemic / high-burden state
              {s.in_season && " — now in usual peak season"}
              {s.citation && <Cites refs={[s.citation]} index={index} />}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

const LIKELIHOOD: Record<string, string> = {
  high: "bg-red-100 text-red-800",
  moderate: "bg-amber-100 text-amber-800",
  low: "bg-slate-100 text-slate-700",
};

function Differential({ items, index }: { items: DifferentialItem[]; index: Map<string, Citation> }) {
  return (
    <ol className="space-y-3">
      {items.map((d) => (
        <li key={d.condition}>
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-semibold">{d.condition}</span>
            <span className={`rounded-full px-2 py-0.5 text-xs font-bold ${LIKELIHOOD[d.likelihood]}`}>{d.likelihood}</span>
            {d.source === "rule" && <span className="rounded-full bg-red-100 px-2 py-0.5 text-xs font-bold text-red-800">RULE</span>}
          </div>
          {d.reasons.length > 0 && (
            <ul className="ml-4 list-disc text-sm text-slate-700">{d.reasons.map((r) => <li key={r}>{r}</li>)}</ul>
          )}
          {d.check_next.length > 0 && <p className="text-sm text-slate-600">Check next: {d.check_next.join("; ")}</p>}
          <Cites refs={d.citations} index={index} />
        </li>
      ))}
    </ol>
  );
}

function Actions({ items, index }: { items: ActionItem[]; index: Map<string, Citation> }) {
  return (
    <ol className="space-y-3">
      {items.map((a, i) => (
        <li key={i} className={a.source === "rule" ? "rounded-md border-l-4 border-red-600 bg-red-50 p-2" : ""}>
          <p className="text-[15px] leading-snug">{a.text}</p>
          {a.details.length > 0 && (
            <ul className="ml-4 mt-1 list-disc text-sm text-slate-600">{a.details.map((d) => <li key={d}>{d}</li>)}</ul>
          )}
          <Cites refs={a.citations} index={index} />
        </li>
      ))}
    </ol>
  );
}

function Doses({ doses, index }: { doses: DoseRecommendation[]; index: Map<string, Citation> }) {
  return (
    <div className="space-y-2">
      {doses.map((d) => (
        <div key={d.drug} className="rounded-md border border-amber-300 bg-amber-50 p-2 text-sm">
          {!d.verified && (
            <p className="mb-1 font-bold text-amber-900">⚠ UNVERIFIED dose table: check against the printed national guideline before use.</p>
          )}
          <p><strong>{d.drug}</strong> · {d.weight_band}</p>
          <p>{d.regimen}</p>
          <Cites refs={[d.citation.ref]} index={new Map([...index, [d.citation.ref, d.citation]])} />
        </div>
      ))}
    </div>
  );
}

function ReferralNote({ note }: { note: string }) {
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
  return (
    <div>
      <pre className="whitespace-pre-wrap rounded-md bg-slate-50 p-3 font-sans text-sm leading-relaxed">{note}</pre>
      <div className="mt-2 flex gap-2">
        <button onClick={copy} className="min-h-11 flex-1 rounded-lg border border-slate-300 bg-white px-3 font-semibold">
          {copied ? "Copied ✓" : "Copy"}
        </button>
        <a href={`https://wa.me/?text=${encodeURIComponent(note)}`} target="_blank" rel="noreferrer"
           className="flex min-h-11 flex-1 items-center justify-center rounded-lg bg-green-600 px-3 font-semibold text-white">
          Share to WhatsApp
        </a>
      </div>
    </div>
  );
}

export function ResultCard({ run, onOpenTrace }: { run: LiveRun; onOpenTrace: () => void }) {
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
        {run.outbreak ? <Outbreak ctx={run.outbreak} index={index} /> : <Pending label="Checking outbreak reports" />}
      </Section>
      {post ? (
        <>
          {post.differential.length > 0 && (
            <Section title="Consider (differential)"><Differential items={post.differential} index={index} /></Section>
          )}
          <Section title="Actions"><Actions items={post.actions} index={index} /></Section>
          {post.doses.length > 0 && <Section title="Dose (from table only)"><Doses doses={post.doses} index={index} /></Section>}
        </>
      ) : (
        running && <Section title="Assessment"><Pending label="Reasoning over guidelines and outbreak data" /></Section>
      )}
      {(run.summary || run.referralNote) && (
        <Section title="Summary & referral note">
          {run.summary && <p className="mb-2 text-[15px] leading-snug">{run.summary}</p>}
          {run.referralNote && <ReferralNote note={run.referralNote} />}
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
                className="w-full border-t border-slate-200 px-4 py-3 text-left text-sm font-semibold text-teal-800">
          How Iba decided ›
        </button>
      )}
    </article>
  );
}
