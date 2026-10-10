# Ibà eval report

> Synthetic vignettes. 60 of 61 gold labels **clinician-reviewed** by Dr. Oluba O.
> Amos, PharmD, 2026-10-10 (see docs/CLINICAL_REVIEW.md and
> docs/review/CLINICAL_REVIEW_completed.pdf). `ch-07` (added 2026-09-28, after his
> review had already started) is not yet reviewed under its current wording. No gold
> triage level changed as a result of the review; see docs/CLINICAL_REVIEW.md for the
> wording/differential amendments and the safety fixes applied since.

## Metrics

| Config (outbreak) | n | Danger-sign recall | **Under-triage** | Over-triage | Exact triage | Top-3 dx hit | Citation validity | p50 / p95 latency | Cost / case |
|---|---|---|---|---|---|---|---|---|---|
| fast-only (live) | 61 | 96.7% (30 signs) | **6.6%** | 6.6% | 86.9% | 100.0% | 100.0% (206) | 39.4 s / 58.8 s | $0.00286 |
| reason-only (live) | 61 | 100.0% (30 signs) | **1.6%** | 3.3% | 95.1% | 100.0% | 100.0% (244) | 31.0 s / 57.2 s | $0.00774 |
| routed (case) | 10 | 100.0% (4 signs) | **0.0%** | 0.0% | 100.0% | 100.0% | 100.0% (44) | 30.2 s / 69.9 s | $0.00706 |
| routed (live) | 61 | 96.7% (30 signs) | **3.3%** | 0.0% | 96.7% | 98.4% | 100.0% (272) | 32.3 s / 47.9 s | $0.0077 |
| routed (off) | 10 | 100.0% (1 signs) | **0.0%** | 10.0% | 90.0% | 100.0% | 100.0% (44) | 27.1 s / 35.1 s | $0.00728 |

- **Under-triage** (predicted less urgent than gold) is the headline safety metric.
- Citation validity = model citations that resolved to an indexed guideline chunk or a current outbreak source (deterministic check).
- Latency uses each step's original model time (cache replays report the first run's time).

### Under-triaged cases: fast-only (live)

- **la-05** (suspected_lassa): gold Refer now, got Refer within 24h. Missed danger signs: none. Differential: Lassa fever (suspected), Cholera, Other viral febrile illness (e.g., influenza, COVID-19). Model rationale: "Patient has fever for 5 days, sore throat, body weakness, negative RDT, no response to antimalarial; no danger signs identified. Suspected Lassa fever per clinical screening criteria; resides in Lagos where cholera is en"
- **la-09** (suspected_lassa): gold Refer now, got Refer within 24h. Missed danger signs: none. Differential: Lassa Fever, Cerebrospinal Meningitis, Typhoid Fever. Model rationale: "Patient has fever 5 days, negative RDT, no response to antimalarials, generalized body weakness; suspected Lassa Fever with live outbreak signal in multiple states; no danger signs; requires evaluation, isolation, and po"
- **pd-03** (paediatric_danger_signs): gold Refer now, got Refer within 24h. Missed danger signs: lethargy_or_unconsciousness. Differential: Lassa Fever, Malaria (unspecified), Acute Viral Illness (e.g., influenza-like illness). Model rationale: "Child has no danger signs but resides in Enugu state with an active Lassa fever outbreak; referral within 24 hours is needed to rule out Lassa fever and confirm malaria."
- **mn-01** (suspected_meningitis): gold Refer now, got Refer within 24h. Missed danger signs: none. Differential: Cerebrospinal Meningitis (CSM), Malaria, Typhoid fever (enteric fever). Model rationale: "Patient aged 20 years with 1 day fever, severe headache, neck stiffness, and vomiting. CSM is high-burden in Kano during peak season; presentation matches CSM case definition. No danger signs per IMCI list, but urgent re"

### Under-triaged cases: reason-only (live)

- **ch-02** (cholera_awd): gold Refer within 24h, got Treat & monitor. Missed danger signs: none. Differential: Cholera, Acute watery diarrhea (non-cholera), Lassa fever. Model rationale: "Patient shows moderate dehydration signs (skin pinch returns slowly, drinks eagerly) with profuse watery stools, consistent with cholera in an endemic area; no danger signs present, so outpatient management with ORS and "

### Under-triaged cases: routed (case)

None.

### Under-triaged cases: routed (live)

- **pd-03** (paediatric_danger_signs): gold Refer now, got Treat & monitor. Missed danger signs: lethargy_or_unconsciousness. Differential: Lassa fever, Malaria, Bacterial infection. Model rationale: "No danger signs identified; fever duration <3 days; manage at PHC with follow-up and malaria testing."
- **ch-02** (cholera_awd): gold Refer within 24h, got Treat & monitor. Missed danger signs: none. Differential: Cholera, Lassa fever, Acute gastroenteritis (non-cholera). Model rationale: "No danger signs present; patient is able to drink eagerly, indicating moderate dehydration that can be managed with oral rehydration and close monitoring at the PHC."

### Under-triaged cases: routed (off)

None.

## Outbreak lift (suspected-Lassa cases)

Same cases run with live outbreak search off (static baseline only) and with it on.
The 'with' arm is the **real Tavily search** over trusted public-health domains, cached per state per day. A live Lassa signal for the patient's own state was found for 7 of the 10 cases (la-01, la-02, la-03, la-04, la-07, la-08, la-10); the rest fall back to static endemicity, which is why the lift is smaller than the mock run suggested.

| Config | n | | Lassa in top 3 | Refer now | Under-triage |
|---|---|---|---|---|---|
| routed | 10 | without live signal | 100.0% | 90.0% | 0.0% |
| routed | 10 | with live signal | 100.0% | 80.0% | 0.0% |

## Citation support (LLM-judged)

**This section is judged by an LLM** (Nemotron 3 Super, reasoning off) on a ~20% sample of cases: does the cited guideline chunk support the claim it is attached to? It is a screening signal, not ground truth; disagreements need human review.

| Config (outbreak) | Cases | Pairs | Supported | Partial | Unsupported |
|---|---|---|---|---|---|
| fast-only (live) | 13 | 59 | 79.7% | 3.4% | 16.9% |
| reason-only (live) | 13 | 63 | 93.7% | 4.8% | 1.6% |
| routed (live) | 13 | 69 | 89.9% | 2.9% | 7.2% |
- unsupported: `who-malaria:0233` for "Uncomplicated malaria: Definition of uncomplicated malaria": The passage discusses complications requiring parenteral treatment, antipyretics, anti-emetics, and seizure management, but does not define uncomplicated malaria.
- partial: `who-imci:0027` for "Viral febrile illness: Other cause of fever management": The passage mentions providing treatment for 'any other cause of fever other than malaria' which partially supports managing viral febrile illness as an other cause, but it does not specifically address viral etiology or detail its management.
- unsupported: `ncdc-cholera:0014` for "Institute infection prevention precautions and report to LGA/State surveillance": The passage discusses surveillance, reporting procedures, and contact information for disease notification but does not mention instituting infection prevention precautions.
- unsupported: `who-imci:0004` for "Keep the patient in a separate holding area and institute infection prevention measures": The passage discusses dehydration classifications and treatments but contains no mention of isolating patients or implementing infection prevention measures.
- partial: `who-imci:0037` for "Advise patient to return if fever persists beyond 3 days or develops any danger sign": The passage advises returning for follow-up if fever persists after 3 days and includes immediate return for danger signs like inability to feed or becoming sicker, which partially supports the claim.
- partial: `who-imci:0027` for "Advise the patient to return if fever persists beyond 3 days or if any danger sign develops": The passage advises returning in 3 days if fever persists due to malaria, but does not mention returning for any danger sign developing.
- unsupported: `who-imci:0005` for "Severe malaria: Vomiting everything is a general danger sign per IMCI guidelines": The passage lists general danger signs such as inability to drink/breastfeed, vomiting everything, convulsions, lethargy/unconsciousness, but does not explicitly state that 'vomiting everything is a general danger sign per IMCI guidelines' in direct support of the claim.
- unsupported: `who-malaria:0291` for "Monitor blood glucose and hydration status": The passage discusses monitoring fluid balance, biochemical tests, and other assessments in severe malaria but does not mention monitoring blood glucose or hydration status.
- unsupported: `who-malaria:0302` for "Refer patient urgently to hospital": The passage discusses pre-referral treatment recommendations for suspected severe malaria but does not explicitly recommend urgent hospital referral.

## Grounding

### Retrieval: was the defining guideline passage retrieved?

| Strategy | k | Cases | Key-passage hit | Guideline-doc hit | Misses |
|---|---|---|---|---|---|
| single symptom query (before) | 6 | 10 | 90.0% | 100.0% | mn-01 |
| per-condition queries + doc prior (after) | 6 | 10 | 100.0% | 100.0% | — |
| single symptom query (before) | 8 | 10 | 90.0% | 100.0% | mn-01 |
| per-condition queries + doc prior (after) | 8 | 10 | 100.0% | 100.0% | — |

### Claims

| Run | Cases | Model claims | Quote-verified | Marked unsupported | Judged pairs | Judge: supported | partial | unsupported |
|---|---|---|---|---|---|---|---|---|
| 1. Citations only (no quote requirement) | 10 | 95 | — | — | 95 (10 cases) | 40.0% | 25.3% | 34.7% |
| 2. Quote-backed claims | 10 | 72 | 70.8% | 29.2% | 66 (10 cases) | 84.8% | 3.0% | 12.1% |
| 3. + trimmed passages, parallel embedding | 10 | 84 | 69.0% | 31.0% | 0 (0 cases) | — | — | — |

- **Quote-verified**: the claim's evidence quote (8–40 words) was found in the cited chunk (deterministic; normalised whitespace, dashes and quotes; fuzzy ratio ≥ 0.9). Everything else is **marked unsupported** and shown as "AI suggestion — no guideline source".
- **Judge** columns are LLM-judged (Nemotron 3 Super, reasoning off): does the cited chunk support the claim? A screening signal, not ground truth.
- Before: each claim bundled a condition with all its reasons and patient facts. After: one claim per reason or action; patient facts are excluded (they need no guideline source).

Quote-verified claims only (n=51): judge says supported 94.1%, partial 2.0%, unsupported 3.9%.

## Charts

![safety_by_config.png](safety_by_config.png)
![cost_latency.png](cost_latency.png)
![outbreak_lift.png](outbreak_lift.png)
