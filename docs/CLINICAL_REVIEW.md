# Clinical review checklist

Iba is decision support, not diagnosis. Every clinical default below was set by the build
team from the cited guideline, **and needs sign-off by a qualified clinician** before Iba is
used with real patients. Where the guideline doesn't settle a question, we chose the option
that refers more, because under-triage costs more than over-triage.

Status key: **Pending** = not yet reviewed.

## 1. Lassa fever suspicion tiers (Pending)

Code: `lassa_assessment` in [backend/app/rules/danger_signs.py](../backend/app/rules/danger_signs.py).
Source: NCDC *National Guideline for Lassa Fever Case Management* (2018), §1.1.2 Suspected case
(pp. 8–9) and §2.1.1 triage of a suspected case (p. 11).

| Situation | Iba default |
|---|---|
| Active live outbreak signal in the patient's state + case definition met, OR fever ≥ 3 days not responding to treatment, OR abnormal bleeding | **Refer now** + isolation/IPC advice |
| Endemic state (baseline list) + case definition met (fever 3–21 days + ≥ 1 of vomiting, diarrhoea, sore throat, myalgia, generalised weakness, abnormal bleeding, abdominal pain) | **Refer now** + isolation/IPC advice |
| Endemic state + partial features only (fever ≥ 3 days with no treatment response or possible contact, none of the listed symptoms) | **Refer within 24h** + isolation/IPC advice |
| Non-endemic state, no live signal | No rule; the reasoning model may still raise Lassa |

Rationale: NCDC treats any suspected case as isolate-and-notify, and a missed case can infect
the health worker.

Questions for the reviewer:
1. The case definition is broad. In an endemic state, most febrile patients with 3+ days of
   fever and vomiting or weakness will be told to refer now and isolate, **including
   RDT-positive malaria**. Is that the intended trade-off, or should an RDT-positive result
   without treatment failure drop to Refer within 24h?
2. An unmeasured temperature counts as meeting the fever criterion; a measured temperature
   below 38 °C excludes. Agree?
3. Is "Refer within 24h" right for the partial-features tier?

## 2. Danger-sign rules (Pending)

Code: [backend/app/rules/danger_signs.py](../backend/app/rules/danger_signs.py). Sources: WHO IMCI
Chart Booklet (2014) general danger signs; WHO Guidelines for Malaria, severe malaria features.

- Any detected sign → **Refer now**; the model cannot lower it.
- Beyond the spec's list we also trigger on: persistent/repeated vomiting, confusion,
  drowsiness, "floppy" infant.
- We do **not** trigger on "fast breathing" alone (IMCI classifies it as pneumonia, not a
  general danger sign); chest indrawing, grunting, labored breathing and stridor do trigger.
- Negation handling ("no convulsions") only suppresses the text rules, never the
  intake model's flags.

Question: are the additions and the fast-breathing exclusion right for PHC use?

## 3. Dose table (Pending: UNVERIFIED stub)

Code: [backend/app/rules/dosing.py](../backend/app/rules/dosing.py). Artemether-lumefantrine
weight bands (5–<15 kg: 1 tablet; 15–<25: 2; 25–<35: 3; ≥ 35: 4; 6 doses over 3 days). Shown
with an UNVERIFIED warning until checked against the NMEP 4th edition (2020) or, as an
interim, the WHO malaria guideline. Doses are only attached for RDT-positive, Treat & monitor
cases with a known weight; no pre-referral doses yet.

## 4. Endemic-state baseline (Pending: provisional lists)

Data: [data/endemicity.yaml](../data/endemicity.yaml). Lassa: Ondo, Edo, Bauchi, Taraba, Ebonyi
(peak December–April). CSM and cholera lists also provisional. To confirm against current
NCDC situation reports.

## 5. Treat & monitor advice (Pending)

Code: [backend/app/rules/followup.py](../backend/app/rules/followup.py).
- "Review in 3 days if the fever persists; return immediately with any danger sign" (WHO IMCI
  p. 8, fever). **IMCI is for children 2 months–5 years; we apply the same interval to adults.**
  Is that acceptable, or should adults get a different interval?
- RDT negative: "do not give antimalarials" (WHO test-before-treat) and "refer for further
  tests, e.g. typhoid, if fever persists; refer if fever every day > 7 days" (IMCI p. 8).

## 6. Eval vignettes (Pending)

[eval/vignettes.jsonl](../eval/vignettes.jsonl): 60 synthetic cases with gold labels and a
guideline rationale each, all `clinician_reviewed: false`. Please review the gold
`triage_level`, `danger_signs` and `top_differential` for each case; the full eval runs
only after review. Where a live outbreak signal should change the expected triage, a case
carries `gold_live` overrides (currently la-06: partial Lassa features → Refer within 24h on
baseline data, Refer now with a live signal).

## Sign-off

| Item | Reviewer | Date | Decision / changes |
|---|---|---|---|
| 1. Lassa tiers | | | |
| 2. Danger-sign rules | | | |
| 3. Dose table | | | |
| 4. Endemic-state baseline | | | |
| 5. Treat & monitor advice | | | |
| 6. Eval vignettes | | | |
