# Ibà — Fever-Triage Copilot · Build Spec

Nebius x NVIDIA Global AI Hackathon.
- Track: **Best Apps and Agents**, plus the **Best Use of Tavily** bonus.
- Owner: Daniel Jemiri (HeyPulse Technologies).
- Hard deadline: **Fri Oct 30, 2026, 10:00 PT (18:00 Lagos)**. Internal deadline: **Wed Oct 28**.

---

## 1. Problem & audience

**Who:** community health extension workers (CHEWs) and nurses at Nigerian primary health centres (PHCs). They are the first contact for most febrile patients, often with no doctor on site.

**The failure mode:** fever is presumed to be malaria. When the RDT is negative, or the patient has malaria *plus* something else, the worker has no fast way to consider:
- **Lassa fever** — seasonal peak roughly Dec–Apr, high case fatality, a nosocomial risk to the worker themselves;
- **cholera**, **CSM (meningitis)**, **typhoid**;
- **severe malaria danger signs** that require immediate referral.

**The missing piece:** outbreak status changes weekly and by state (NCDC situation reports). The same presentation should be handled differently in an LGA with an active Lassa outbreak. No static guideline app knows that.

**Our claim:** Ibà combines three things at the point of care:
- national guideline-grounded reasoning,
- deterministic safety rules,
- live, state-specific outbreak intelligence.

The result is a triage and referral decision the worker can act on, with sources.

> **Framing for judges:** decision support with a human in the loop and referral-first behaviour. Never "AI diagnosis".

---

## 2. User flow (the demo)

1. The worker opens the PWA and picks their **state/LGA** (remembered on the device).
2. They describe the patient in free text or Pidgin, e.g.:
   > "Pikin 3 years, hot body 4 days, vomit everything, RDT negative, e don dey sleep too much"
3. Ibà asks up to **2 targeted follow-up questions** if critical fields are missing (age, duration, RDT result, danger signs).
4. The result card shows:
   - **Triage:** 🔴 Refer now / 🟠 Refer within 24h / 🟢 Treat & monitor
   - **Danger signs detected**, each flagged as rule-triggered or LLM-found
   - **Outbreak context:** "Lassa fever: active in Ondo — NCDC sitrep week 41" with a link
   - **Differential:** 2–4 conditions with reasons, and what to check next
   - **Actions:** a guideline-cited step list; doses come from the lookup table only
   - **Referral note:** one tap to copy or share (WhatsApp share intent)
6. An expandable **"How Ibà decided"** panel shows the models used per step, the sources, and the timing. It's good for judges, and good for trust.

---

## 3. Architecture

```
PWA (React) ──► FastAPI on Nebius Serverless Endpoint
                    │
               LangGraph pipeline
                    │
 intake ─► rules_pre ─► retrieve ─► outbreak ─► reason ─► rules_post ─► compose
  │          │           │            │           │          │            │
 FAST      Python      EMBED        Tavily     REASON     Python        MID
 (Nano)    rules       + NumPy      (cached)   (Ultra)    rules        (Super)
```

| Node | Model / tool | Job |
|---|---|---|
| `intake` | `MODEL_FAST` (Nemotron 3 Nano) | Normalise Pidgin/English into a `PatientCase` JSON; list missing critical fields; produce follow-up questions |
| `rules_pre` | Deterministic Python | Detect WHO/NMEP danger signs; set a triage floor (can only raise the level, never lower it) |
| `retrieve` | `MODEL_EMBED` + NumPy cosine | Top-k guideline chunks for the symptoms and suspected conditions |
| `outbreak` | **Tavily** | Search trusted domains for current outbreaks in the patient's state; extract `OutbreakSignal[]` (disease, state, date, source) using `MODEL_FAST` |
| `reason` | `MODEL_REASON` (Nemotron 3 Ultra) | Differential, triage recommendation, actions, all with citations to chunk IDs and outbreak URLs; structured JSON |
| `rules_post` | Deterministic Python | Enforce the triage floor; strip any dose text; attach doses from `dosing.py`; validate that citations exist |
| `compose` | `MODEL_MID` (Nemotron 3 Super) | Plain-language summary + referral note (English or Pidgin, matching the input) |

**Why this routing matters (say it in the video):**
- Ultra is used once per case, for the only step that needs deep reasoning.
- Nano and Super handle the other steps.
- Show measured cost and latency for **routed vs. Ultra-for-everything** (§6).

### Model IDs
- Base URL: `https://api.tokenfactory.nebius.com/v1/`. This is OpenAI-compatible, so it works with the `openai` SDK.
- On day 1, run `GET /v1/models` and put the real IDs in `.env`.
- Known IDs so far include:
  - `nvidia/nemotron-3-super-120b-a12b`
  - a Nemotron 3 Nano 30B-A3B
- **Confirm the Ultra ID and whether it's live.**
- If an NVIDIA embedding model is served, use it for `MODEL_EMBED`. Otherwise use the best multilingual embedder in the catalog. Nemotron still satisfies the "NVIDIA model" requirement, but note the embedding choice in the README.
- **Fallback if Ultra isn't available:** use Super for `reason` and keep the same story, "reasoning model for one step, Nano for the rest".

---

## 4. Knowledge sources

Record every source in `data/sources.yaml` with its URL, edition/date, and licence note. Only commit the text if the licence allows it.

| Source | Use |
|---|---|
| Nigeria National Guidelines for Diagnosis & Treatment of Malaria (NMEP, latest edition) | Uncomplicated/severe malaria criteria, treatment, weight-band dosing table → `dosing.py` |
| WHO Guidelines for Malaria | Danger signs, severe malaria definitions |
| NCDC Lassa Fever case management / IPC guidance | Suspected-case definition, isolation and referral |
| NCDC cholera and CSM guidance | Case definitions, referral triggers |
| WHO IMCI chart booklet | Paediatric general danger signs |

- **Chunking:** 400–600 tokens, split on headings. Keep `doc_id`, `section`, and `page` for citations.
- **Index:** a NumPy array plus JSON metadata in `data/index/`. It's tiny, so there's no vector DB to deploy.

### Tavily outbreak tool (`tools/outbreak.py`)
- **Query:** `"{state} Nigeria outbreak {disease_candidates} {current month year}"`, plus one general `"NCDC situation report"` query.
- **Filters:** `include_domains=["ncdc.gov.ng", "who.int", "reliefweb.int", "afro.who.int"]`, `search_depth="advanced"`, restricted to recent results.
- **Extraction:** Nano extracts `OutbreakSignal{disease, state, status, report_date, url}`. Drop any signal without a URL or date.
- **Caching:** per `(state, date)` in production. That keeps Tavily calls to about 1 per state per day, which cuts cost and gives consistent demos.
- **In the UI,** show a "last checked" timestamp.
- **For the Tavily prize,** Tavily must *change the decision*. The demo needs a case where the outbreak context moves Lassa up the differential and triggers the isolation/referral advice.

---

## 5. Safety design (a differentiator, so show it)

- **Rule-based floor.** `danger_signs.py` covers the adult severe malaria criteria and the IMCI general danger signs:
  - convulsions, unable to drink/breastfeed, vomiting everything, lethargy/unconsciousness,
  - bleeding, jaundice, respiratory distress, severe pallor, dark urine, prostration.
  - A hit forces 🔴. The LLM cannot override it.
- **Lassa suspicion rule.** Fever for ≥3 days + no response to antimalarials, or bleeding, in a state with an active outbreak signal → 🔴 plus an isolation/IPC reminder.
- **No generated doses.** A regex plus a schema check strip any dose from LLM output. Doses come from the weight-band table only, with a citation.
- **Fail safe.** Any parse or LLM error, or a timeout, gives **"Refer — Ibà could not complete assessment"**. We never return a silent green.
- **Privacy.** No names are collected. Case text is not persisted in production. Only per-step metrics are logged.
- **Visible disclaimer** in the UI and the referral note.

---

## 6. Evaluation (a differentiator: most entries show no numbers)

**`eval/vignettes.jsonl`:** 60 synthetic cases. Write them from the guideline criteria, then have 1–2 clinicians review the gold labels.

| Category | n |
|---|---|
| Uncomplicated malaria (RDT+) | 10 |
| Severe malaria (adult + child) | 10 |
| Suspected Lassa (with and without outbreak context) | 10 |
| Cholera / acute watery diarrhoea | 6 |
| Suspected meningitis | 6 |
| Typhoid / other febrile | 8 |
| Paediatric IMCI danger signs | 6 |
| Pidgin-phrased variants (drawn from the above) | 4 |

**Gold labels per case:**
- `triage_level`
- `danger_signs[]`
- `must_refer`
- `top_differential[]`
- `outbreak_relevant`

**Metrics:**
- **Danger-sign recall** (target 100%; any miss is a bug)
- **Referral accuracy / under-triage rate** (under-triage is the key safety metric)
- **Differential hit rate:** is the gold condition in the top-3?
- **Citation validity:** does every cited chunk or URL exist and support the claim? Use an LLM-judge with Super on a sample, and spot-check by hand.
- **Outbreak lift:** on the Lassa cases, compare with Tavily vs. without
- **Latency p50/p95** and **cost per case**

**Configurations to compare:**
- `routed` (the default)
- `reason-only` (Ultra for everything)
- `fast-only` (Nano for everything)

This produces the headline chart, e.g. "Routed matches Ultra-only on safety at X% of the cost and Y× faster". Report whatever the real numbers are; don't invent targets.

**Running it:**
- Run as a Nebius Serverless Job if convenient (extra Nebius usage to mention).
- Cache aggressively.
- **Budget:** run the full 3-config eval **once** in week 4. Iterate on 10-case subsets before that.

---

## 7. Frontend

- A chat-style single screen with a state/LGA picker in the header.
- Structured result card (§2), colour-coded, readable on a low-end Android phone.
- "How Ibà decided" drawer: per-step model, latency, sources.
- Share button → WhatsApp intent with the referral note.
- An `/about` page covering the disclaimer, sources, and the models used.
- Demo mode: three preloaded example cases, so judges can try it in one tap without typing.

---

## 8. Repo & submission checklist

- [ ] Public GitHub repo under **your own account**, with the **Apache-2.0** licence visible in the About section
- [ ] README containing:
  - problem, demo GIF, and architecture diagram;
  - **a "How we use NVIDIA Nemotron + Nebius Token Factory" section** covering per-step models, routing rationale, and eval numbers;
  - Tavily usage;
  - setup and run steps, plus `.env.example`
- [ ] Live demo URL (Nebius Serverless Endpoint). If it's gated, put the credentials in the testing instructions.
- [ ] Demo video under 3 minutes, public on YouTube, **with audio that explicitly explains the Token Factory and Nemotron usage**. No copyrighted music.
- [ ] Devpost description: what, why, how, and the eval results
- [ ] Detailed feedback on Token Factory, Serverless, and Nemotron (this is a separate $100 prize, so be specific: DX friction, docs gaps, latency observations)
- [ ] Track selection: Best Apps and Agents

---

## 9. Video script (≤ 3:00)

| Time | Beat |
|---|---|
| 0:00–0:20 | The problem: a PHC in Ondo, fever = "malaria", a missed Lassa case. One stat, one sentence. |
| 0:20–1:20 | Live demo, case 1: a Pidgin description of a child with danger signs → instant 🔴, with the rule badge shown. Case 2: adult, RDT−, 5 days of fever, in Ondo → Tavily pulls the NCDC sitrep → Lassa rises in the differential → isolation plus referral note shared to WhatsApp. |
| 1:20–2:10 | Under the hood: the pipeline diagram. Nano → rules → retrieval → **Tavily** → **Nemotron Ultra on Token Factory** → rules → Super. Cover why the routing matters and that it's deployed on Nebius Serverless. |
| 2:10–2:45 | Eval results chart: safety metrics, and cost/latency for routed vs. Ultra-only. |
| 2:45–3:00 | Impact and next steps (pilot with PHCs, offline mode on Jetson for low connectivity). Close. |

---

## 10. Timeline

| Week | Dates | Deliverables |
|---|---|---|
| 1 | Sep 22–28 | Credits claimed; `/v1/models` checked and `.env` filled; repo + licence; `sources.yaml`; ingest + index; `danger_signs.py` and `dosing.py` with unit tests; 20 vignettes drafted |
| 2 | Sep 29–Oct 5 | Full LangGraph pipeline end-to-end in the CLI; Tavily outbreak tool with caching; router cost/latency accounting |
| 3 | Oct 6–12 | React PWA; Pidgin in/out; Docker image; **deployed on Nebius Serverless Endpoint**; demo-mode cases |
| 4 | Oct 13–19 | All 60 vignettes, clinician-reviewed; full eval run; fixes; charts |
| 5 | Oct 20–25 | README, architecture diagram, demo GIF, video recorded and edited |
| Buffer | Oct 26–28 | Polish, feedback write-up, **submit by Oct 28** |

---

## 11. Open questions (resolve in week 1)

1. Exact Ultra model ID and whether it's live on Token Factory; its pricing vs. Super.
2. Is there an NVIDIA embedding or reranking model in the catalog?
3. Nebius Serverless Endpoint: cold-start time, container size limits, whether it supports scale-to-zero.
4. The licence terms of each guideline document (can the text be committed, or only fetched at build time?).
5. Which clinicians will review the vignettes and appear briefly in the video (with consent)?
