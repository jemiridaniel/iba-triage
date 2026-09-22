import { useEffect, useState } from "react";
import { getMeta } from "../api";
import type { Meta } from "../types";

export function About() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    getMeta().then(setMeta).catch((e) => setError(String(e.message ?? e)));
  }, []);

  return (
    <div className="space-y-5 text-[15px] leading-relaxed">
      <section className="rounded-xl border-2 border-red-300 bg-red-50 p-4">
        <h2 className="mb-1 text-lg font-bold text-red-900">Decision support, not diagnosis</h2>
        <p>
          Iba helps primary health care workers triage febrile patients. It does not diagnose. A qualified
          health worker makes every decision. When Iba is unsure or anything fails, it says{" "}
          <strong>Refer now</strong>. It never gives a silent green.
        </p>
      </section>

      <section>
        <h2 className="mb-1 text-lg font-bold">How it works</h2>
        <ul className="ml-5 list-disc space-y-1">
          <li><strong>Danger signs</strong> are found by fixed rules (English and Pidgin), not AI. Any danger sign means Refer now, and the AI cannot lower that.</li>
          <li><strong>Doses</strong> come only from a weight-band table, never from the AI.</li>
          <li><strong>Outbreak context</strong>: live reports from trusted sites (NCDC, WHO, ReliefWeb) when available, plus a baseline list of endemic states.</li>
          <li><strong>Every recommendation cites</strong> a guideline passage or outbreak report. Citations that cannot be checked are removed.</li>
          <li><strong>Privacy</strong>: no names are collected and case text is not stored.</li>
        </ul>
      </section>

      {error && <p className="text-amber-800">Could not load live details: {error}</p>}

      {meta && (
        <>
          <section>
            <h2 className="mb-1 text-lg font-bold">Models</h2>
            <p className="text-sm text-slate-600">NVIDIA Nemotron models on Nebius Token Factory.</p>
            <ul className="mt-1 space-y-0.5 text-sm">
              <li>Understanding the case &amp; outbreak reports: <code>{meta.models.fast || "—"}</code> (reasoning {meta.reasoning.intake ? "on" : "off"})</li>
              <li>Clinical reasoning: <code>{meta.models.reason || "—"}</code> (reasoning {meta.reasoning.reason ? "on" : "off"})</li>
              <li>Summary &amp; referral note: <code>{meta.models.mid || "—"}</code></li>
              <li>Guideline search embeddings: <code>{meta.models.embed || "—"}</code> ({meta.index.chunks} passages indexed)</li>
            </ul>
          </section>

          <section>
            <h2 className="mb-1 text-lg font-bold">Data status</h2>
            <ul className="space-y-1 text-sm">
              <li>{meta.dose_table_verified ? "✅ Dose table verified against the national guideline." : "⚠ Dose table is an UNVERIFIED stub pending confirmation against the national malaria guideline."}</li>
              <li>{meta.endemicity_verified ? "✅ Endemic-state baseline verified." : "⚠ Endemic-state baseline is provisional (awaiting confirmation against NCDC reports)."}</li>
              <li>{meta.live_outbreak_search === "tavily" ? "✅ Live outbreak search on." : meta.live_outbreak_search === "mock" ? "⚠ Outbreak search uses MOCK test data." : "⚠ Live outbreak search is off; baseline only."}</li>
            </ul>
          </section>

          <section>
            <h2 className="mb-1 text-lg font-bold">Sources &amp; licences</h2>
            <ul className="space-y-2 text-sm">
              {meta.sources.map((s) => (
                <li key={s.doc_id} className="rounded-lg border border-slate-200 p-2">
                  <p className="font-semibold">
                    {s.url ? <a className="text-sky-800 underline" href={s.url} target="_blank" rel="noreferrer">{s.title}</a> : s.title}
                  </p>
                  <p className="text-slate-600">{s.edition}</p>
                  <p className="text-slate-600">Licence: {s.licence}</p>
                </li>
              ))}
            </ul>
            <p className="mt-2 text-xs text-slate-500">
              WHO materials are used under CC BY-NC-SA 3.0 IGO (attribution, non-commercial). NCDC documents are
              Nigerian government publications used for retrieval and citation only.
            </p>
          </section>
        </>
      )}
    </div>
  );
}
