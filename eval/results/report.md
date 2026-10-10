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
| fast-only (live) | 61 | 96.7% (30 signs) | **4.9%** | 27.9% | 67.2% | 98.4% | 100.0% (217) | 36.7 s / 60.1 s | $0.0027 |
| reason-only (live) | 61 | 100.0% (30 signs) | **3.3%** | 27.9% | 68.9% | 98.4% | 100.0% (304) | 34.7 s / 49.5 s | $0.00809 |
| routed (case) | 10 | 100.0% (4 signs) | **0.0%** | 0.0% | 100.0% | 100.0% | 100.0% (44) | 30.2 s / 69.9 s | $0.00706 |
| routed (live) | 61 | 96.7% (30 signs) | **3.3%** | 27.9% | 68.9% | 100.0% | 100.0% (286) | 34.5 s / 49.4 s | $0.00751 |
| routed (off) | 10 | 100.0% (1 signs) | **0.0%** | 10.0% | 90.0% | 100.0% | 100.0% (62) | 30.8 s / 50.1 s | $0.00755 |

- **Under-triage** (predicted less urgent than gold) is the headline safety metric.
- Citation validity = model citations that resolved to an indexed guideline chunk or a current outbreak source (deterministic check).
- Latency uses each step's original model time (cache replays report the first run's time).

### Under-triaged cases: fast-only (live)

- **pd-03** (paediatric_danger_signs): gold Refer now, got Refer within 24h. Missed danger signs: lethargy_or_unconsciousness. Differential: Lassa fever, Malaria (uncomplicated), Bacterial fever (e.g., typhoid). Model rationale: "Active Lassa fever outbreak in Enugu state; patient febrile for 2 days with no danger signs; requires isolation and urgent evaluation at a higher facility."
- **la-05** (suspected_lassa): gold Refer now, got Refer within 24h. Missed danger signs: none. Differential: Lassa fever, Cholera, Other viral febrile illness (e.g., influenza, COVID-19). Model rationale: "Rule floor is refer_24h per deterministic findings; no danger signs present. Baseline cholera endemic in Lagos peak season but no active outbreak; therefore patient should be referred for facility review within 24 hours."
- **la-09** (suspected_lassa): gold Refer now, got Refer within 24h. Missed danger signs: none. Differential: Lassa Fever, Typhoid Fever, Cerebrospinal Meningitis. Model rationale: "Patient is a 45-year-old with 5 days of fever, RDT-negative malaria, no response to antimalarials, and generalized body weakness. No danger signs are present. Lassa fever is suspected based on lack of antimalarial respon"

### Under-triaged cases: reason-only (live)

- **la-05** (suspected_lassa): gold Refer now, got Refer within 24h. Missed danger signs: none. Differential: Lassa fever, Acute bacterial pharyngitis/tonsillitis, Malaria (possible false-negative RDT). Model rationale: "Patient has fever for 5 days with sore throat and body weakness, negative malaria RDT, no response to antimalarial, no danger signs; Lagos is a cholera-endemic area and the presentation fits viral haemorrhagic fever scre"
- **la-09** (suspected_lassa): gold Refer now, got Refer within 24h. Missed danger signs: none. Differential: Lassa Fever, Cerebrospinal Meningitis, Typhoid Fever. Model rationale: "No danger signs identified; however, the presentation raises suspicion for Lassa fever (fever >48hrs, generalized weakness, no response to antimalarial) and baseline endemicity for cerebrospinal meningitis in Plateau war"

### Under-triaged cases: routed (case)

None.

### Under-triaged cases: routed (live)

- **pd-03** (paediatric_danger_signs): gold Refer now, got Refer within 24h. Missed danger signs: lethargy_or_unconsciousness. Differential: Malaria, Lassa fever, Possible bacterial infection. Model rationale: "No danger signs are present, but the child is 1 year old with 2 days of fever and nonspecific symptoms (quiet, not playing, sleepy). Given the active Lassa fever outbreak in Enugu, close monitoring and timely referral to"
- **la-09** (suspected_lassa): gold Refer now, got Refer within 24h. Missed danger signs: none. Differential: Lassa Fever, Cerebrospinal Meningitis, Typhoid Fever. Model rationale: "No danger signs identified; however, the presentation raises suspicion for Lassa fever (fever >48hrs, generalized weakness, no response to antimalarial) and baseline endemicity for cerebrospinal meningitis in Plateau war"

### Under-triaged cases: routed (off)

None.

## Outbreak lift (suspected-Lassa cases)

Same cases run with live outbreak search off (static baseline only) and with it on.
The 'with' arm is the **real Tavily search** over trusted public-health domains, cached per state per day. A live Lassa signal for the patient's own state was found for 7 of the 10 cases (la-01, la-02, la-03, la-04, la-07, la-08, la-10); the rest fall back to static endemicity, which is why the lift is smaller than the mock run suggested.

| Config | n | | Lassa in top 3 | Refer now | Under-triage |
|---|---|---|---|---|---|
| routed | 10 | without live signal | 100.0% | 90.0% | 0.0% |
| routed | 10 | with live signal | 100.0% | 80.0% | 10.0% |

## Citation support (LLM-judged)

**This section is judged by an LLM** (Nemotron 3 Super, reasoning off) on a ~20% sample of cases: does the cited guideline chunk support the claim it is attached to? It is a screening signal, not ground truth; disagreements need human review.

| Config (outbreak) | Cases | Pairs | Supported | Partial | Unsupported |
|---|---|---|---|---|---|
| fast-only (live) | 14 | 57 | 75.4% | 8.8% | 15.8% |
| reason-only (live) | 14 | 73 | 89.0% | 5.5% | 5.5% |
| routed (live) | 14 | 80 | 86.2% | 6.2% | 7.5% |
- unsupported: `who-malaria:0233` for "Uncomplicated malaria: A patient who presents with symptoms of malaria and a positive parasitological test (microscopy o": The passage discusses additional considerations for managing malaria cases, including oral tolerance, antipyretics, anti-emetics, and seizures, but does not define uncomplicated malaria based on symptoms, positive parasitological test, and absence of severe features.
- unsupported: `who-malaria:0290` for "Refer the patient urgently to a hospital with capacity for severe malaria treatment and Lassa fever isolation.": The passage details criteria for diagnosing severe malaria and its management but does not mention urgent referral to a hospital with capacity for severe malaria treatment or Lassa fever isolation.
- unsupported: `who-imci:0027` for "Perform a rapid diagnostic test (RDT) for malaria.": The passage discusses repeating a malaria test under specific conditions (fever persisting after 3 days), but it does not mention performing an initial rapid diagnostic test (RDT) for malaria.
- partial: `who-malaria:0303` for "Initiate intravenous fluids and monitor vital signs, urine output, and blood glucose every 4 hours": The passage mentions monitoring vital signs, urine output, and blood glucose every 4 hours as part of supportive care for severe malaria, which aligns with the claim, but does not mention initiating intravenous fluids.
- unsupported: `who-imci:0002` for "Refer the child urgently to hospital with infection prevention measures for suspected Lassa Fever": The passage describes urgent referral for general danger signs but does not mention Lassa Fever or infection prevention measures.
- unsupported: `who-imci:0027` for "Advise patient to return if fever persists beyond 3 days": The passage advises returning in 3 days only if malaria test is positive, not generally for persistent fever beyond 3 days.
- unsupported: `who-malaria:0302` for "Refer patient urgently to higher-level facility": The passage discusses pre-referral treatment recommendations for severe malaria but does not explicitly recommend urgent referral to a higher-level facility.
- unsupported: `who-imci:0027` for "Perform malaria rapid diagnostic test.": The passage mentions repeating a malaria test under specific conditions but does not support performing a malaria rapid diagnostic test as an initial or standalone action.
- unsupported: `who-imci:0027` for "Advise patient to return if fever persists beyond 3 days": The passage advises returning in 3 days only if malaria test is positive, not generally for persistent fever beyond 3 days.

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
