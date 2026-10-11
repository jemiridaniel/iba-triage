"""Endemicity baseline, NCDC Lassa case definition tiers, treatment-response question,
actionable Treat & monitor advice, advice merging. No network."""

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from backend.app.graph import nodes
from backend.app.graph.pipeline import run_triage
from backend.app.graph.state import TriageRequest, TriageState
from backend.app.rag.store import VectorStore
from backend.app.rules.danger_signs import (
    LASSA_IPC_REMINDER,
    lassa_assessment,
    lassa_case_definition,
)
from backend.app.rules.followup import (
    NO_ANTIMALARIAL_TEXT,
    REVIEW_TEXT,
    TEST_FIRST_TEXT,
    merge_advice,
    topics,
    treat_monitor_advice,
)
from backend.app.schemas import ActionItem, GuidelineChunk, OutbreakSignal, PatientCase, TriageLevel
from backend.app.tools.endemicity import Endemicity
from scripts.fetch_sources import is_pdf, pdf_links
from tests.pipeline_fakes import COMPOSE_EN, OUTBREAK_EXTRACTION, REPO, TODAY, make_deps

ENDEMICITY = Endemicity.load(REPO / "data" / "endemicity.yaml")

# --- NCDC 2018 suspected-case definition -------------------------------------


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        (
            {"fever_days": 5, "raw_text": "sore throat and body weakness"},
            ["sore throat", "generalised weakness"],
        ),
        ({"fever_days": 2, "raw_text": "sore throat"}, []),  # < 3 days
        ({"fever_days": 25, "raw_text": "sore throat"}, []),  # > 21 days
        ({"fever_days": 5, "temperature_c": 37.2, "raw_text": "vomiting"}, []),  # measured < 38
        ({"fever_days": 5, "temperature_c": 38.6, "raw_text": "vomiting"}, ["vomiting"]),
        ({"fever_days": 5, "raw_text": "headache only"}, []),  # no listed symptom
        ({"fever_days": 4, "symptoms": ["abdominal pain"]}, ["abdominal pain"]),
    ],
)
def test_lassa_case_definition(fields: dict, expected: list[str]) -> None:
    assert lassa_case_definition(PatientCase(**fields)) == expected


def baseline_ondo() -> list[OutbreakSignal]:
    return ENDEMICITY.for_state("Ondo", TODAY)


LIVE_ONDO = OutbreakSignal(
    disease="Lassa fever",
    state="Ondo",
    status="active",
    report_date=date(2026, 9, 18),
    url="https://ncdc.gov.ng/x",
    basis="live",
)
UNRESPONSIVE = "fever 5 days, sore throat, took coartem but no improvement"


def test_baseline_case_definition_met_refers_now() -> None:
    case = PatientCase(fever_days=5, state="Ondo", raw_text=UNRESPONSIVE)
    finding = lassa_assessment(case, baseline_ondo(), [])
    assert finding.basis == "baseline"
    assert finding.floor == TriageLevel.REFER_NOW
    assert any("NCDC suspected-case definition" in c for c in finding.criteria)


def test_baseline_partial_features_refer_within_24h() -> None:
    text = "fever 5 days, headache, took coartem but no improvement"  # no listed symptom
    finding = lassa_assessment(
        PatientCase(fever_days=5, state="Ondo", raw_text=text), baseline_ondo(), []
    )
    assert finding.floor == TriageLevel.REFER_24H
    assert any("Partial features" in c for c in finding.criteria)


def test_live_tier_refers_now() -> None:
    case = PatientCase(fever_days=5, state="Ondo", raw_text=UNRESPONSIVE)
    finding = lassa_assessment(case, [LIVE_ONDO, *baseline_ondo()], [])
    assert finding.basis == "live" and finding.floor == TriageLevel.REFER_NOW


def test_baseline_definition_alone_is_enough() -> None:
    # Treatment response unknown, but the case definition is met in an endemic state.
    case = PatientCase(fever_days=5, state="Ondo", raw_text="fever 5 days, sore throat")
    assert lassa_assessment(case, baseline_ondo(), []).floor == TriageLevel.REFER_NOW


def test_baseline_without_features_is_none() -> None:
    case = PatientCase(fever_days=5, state="Ondo", raw_text="fever 5 days, headache")
    assert lassa_assessment(case, baseline_ondo(), []) is None


def test_non_endemic_state_has_no_baseline_tier_but_gets_the_any_state_floor() -> None:
    # Lagos is not in the Lassa baseline at all (not even "reported"): the case-definition-plus
    # -failed-treatment any-state floor (added 2026-10-10, Dr. Amos's review) still applies, at
    # Refer within 24h -- the baseline/live tiers above this one are what's unavailable here.
    case = PatientCase(fever_days=5, state="Lagos", raw_text=UNRESPONSIVE)
    finding = lassa_assessment(case, ENDEMICITY.for_state("Lagos", TODAY), [])
    assert finding is not None
    assert finding.basis == "any_state"
    assert finding.floor == TriageLevel.REFER_24H
    assert finding.signal is None


def test_antibiotic_failure_counts() -> None:
    text = "fever 6 days, vomiting, took amoxicillin, still no improvement"
    case = PatientCase(fever_days=6, state="Ondo", raw_text=text)
    assert lassa_assessment(case, baseline_ondo(), []) is not None


# --- endemicity baseline ---------------------------------------------------------


def test_endemicity_yaml_is_now_verified() -> None:
    # Sourced against current NCDC situation reports (2026-09-28); no longer provisional.
    assert ENDEMICITY.entries and ENDEMICITY.all_verified


def test_baseline_for_state_and_season() -> None:
    (lassa,) = [
        s for s in ENDEMICITY.for_state("ondo state", date(2026, 1, 10)) if "Lassa" in s.disease
    ]
    assert lassa.basis == "baseline" and lassa.status == "endemic" and lassa.in_season is True
    (off,) = [s for s in ENDEMICITY.for_state("Ondo", date(2026, 7, 1)) if "Lassa" in s.disease]
    assert off.in_season is False
    # Every real Nigerian state is now on at least one of the three disease lists (clinical
    # review, 2026-10-10: the CSM belt alone is 19 northern states + the FCT, plus 7 southern
    # states reported) -- a clearly fictional place is the only reliable "no match" case left.
    assert ENDEMICITY.for_state("Nowhereland", TODAY) == []
    assert ENDEMICITY.for_state(None, TODAY) == []


def test_baseline_citation_resolves_through_anchor() -> None:
    seen = []

    def resolve(doc_id, phrase):
        seen.append((doc_id, phrase))
        return "ncdc-lassa:0003"

    (lassa,) = [s for s in ENDEMICITY.for_state("Edo", TODAY, resolve) if "Lassa" in s.disease]
    assert lassa.citation == "ncdc-lassa:0003"
    assert seen[0] == ("ncdc-lassa-advisory-2026", "particularly in endemic and high-burden states")


def test_outbreak_tool_always_adds_baseline(tmp_path: Path) -> None:
    deps, _ = make_deps(tmp_path, {}, search=None)
    deps.outbreak.endemicity = ENDEMICITY
    ctx = deps.outbreak.check("Ondo")
    assert ctx.status == "unavailable"
    assert ctx.message.endswith("Using baseline endemicity only.")
    assert any(s.basis == "baseline" and "Lassa" in s.disease for s in ctx.baseline)
    assert ctx.signals == []


def test_live_and_baseline_are_separate(tmp_path: Path) -> None:
    deps, _ = make_deps(tmp_path, {"outbreak extraction": OUTBREAK_EXTRACTION})
    deps.outbreak.endemicity = ENDEMICITY
    ctx = deps.outbreak.check("Ondo")
    assert ctx.status == "ok"
    assert all(s.basis == "live" for s in ctx.signals)
    assert all(s.basis == "baseline" for s in ctx.baseline)
    assert "baseline" not in ctx.message


# --- endemicity tiers (NCDC-sourced, 2026-09-28) -----------------------------------


def test_lassa_tiers() -> None:
    for state in ("Ondo", "Edo", "Bauchi", "Taraba", "Benue"):
        (sig,) = [s for s in ENDEMICITY.for_state(state, TODAY) if "Lassa" in s.disease]
        assert sig.tier == "high_burden" and sig.confirmed is True

    (kogi,) = [s for s in ENDEMICITY.for_state("Kogi", TODAY) if "Lassa" in s.disease]
    assert kogi.tier == "reported" and kogi.confirmed is True

    (ebonyi,) = [s for s in ENDEMICITY.for_state("Ebonyi", TODAY) if "Lassa" in s.disease]
    assert ebonyi.tier == "reported" and ebonyi.confirmed is False


def test_reported_tier_lassa_state_does_not_escalate() -> None:
    # Ebonyi and Kogi are "reported": meeting the case definition there must NOT hit the
    # baseline escalation branch (endemic_baseline() only matches tier="high_burden").
    case = PatientCase(
        fever_days=5, state="Kogi", raw_text="sore throat and body weakness", symptoms=[]
    )
    assert lassa_assessment(case, ENDEMICITY.for_state("Kogi", TODAY), []) is None
    case2 = PatientCase(
        fever_days=5, state="Ebonyi", raw_text="sore throat and body weakness", symptoms=[]
    )
    assert lassa_assessment(case2, ENDEMICITY.for_state("Ebonyi", TODAY), []) is None


def test_csm_tiers() -> None:
    belt = (
        "Sokoto", "Kebbi", "Zamfara", "Katsina", "Kano",
        "Jigawa", "Yobe", "Borno", "Bauchi", "Gombe",
    )
    for state in belt:
        (sig,) = [
            s for s in ENDEMICITY.for_state(state, TODAY) if "meningitis" in s.disease.lower()
        ]
        assert sig.tier == "high_burden"
    (bayelsa,) = [
        s for s in ENDEMICITY.for_state("Bayelsa", TODAY) if "meningitis" in s.disease.lower()
    ]
    assert bayelsa.tier == "reported"


def test_bayelsa_cholera_now_gets_a_baseline() -> None:
    # Bayelsa accounts for 442 of 882 cumulative suspected cases (50%), NCDC Cholera Situation
    # Report Epi Week 22 2024, p.6 -- the single largest state burden in the report.
    signals = [s for s in ENDEMICITY.for_state("Bayelsa", TODAY) if s.disease == "Cholera"]
    assert len(signals) == 1
    assert signals[0].tier == "high_burden"
    assert signals[0].report == "NCDC Cholera Situation Report (Monthly Epidemiological Report 5)"


def test_kano_cholera_cited_directly_no_caveat() -> None:
    # Clinical review follow-up, 2026-10-11: Dr. Amos supplied the actual 2024 NCDC Cholera
    # Situation Report, which records 13 cumulative cases for Kano (p.6) directly -- this
    # supersedes the previous round's press_2026_unconfirmed flag, which was only needed
    # before this primary document was in hand.
    (cholera,) = [s for s in ENDEMICITY.for_state("Kano", TODAY) if s.disease == "Cholera"]
    assert cholera.tier == "high_burden"
    assert cholera.evidence is None
    # Kano is still CSM high_burden too.
    csm = [s for s in ENDEMICITY.for_state("Kano", TODAY) if "meningitis" in s.disease.lower()]
    assert len(csm) == 1 and csm[0].tier == "high_burden"


def test_zamfara_cholera_restored_no_caveat() -> None:
    # Zamfara has 64 cumulative cases in the 2024 NCDC sitrep (p.6) -- cited directly, no
    # evidence caveat needed.
    (cholera,) = [s for s in ENDEMICITY.for_state("Zamfara", TODAY) if s.disease == "Cholera"]
    assert cholera.tier == "high_burden"
    assert cholera.evidence is None


def test_cholera_reported_tier_states() -> None:
    (adamawa,) = [s for s in ENDEMICITY.for_state("Adamawa", TODAY) if s.disease == "Cholera"]
    assert adamawa.tier == "reported" and adamawa.confirmed is True


def test_bauchi_cholera_confirmed_no_caveat() -> None:
    # Bauchi has 46 cumulative cases in the 2024 NCDC sitrep (p.6) -- cited directly.
    (cholera,) = [s for s in ENDEMICITY.for_state("Bauchi", TODAY) if s.disease == "Cholera"]
    assert cholera.tier == "high_burden"
    assert cholera.evidence is None


def test_borno_cholera_has_evidence_caveat() -> None:
    # Clinical review follow-up, 2026-10-11: the 2024 NCDC sitrep records only 1 cumulative
    # case for Borno (p.6) -- too thin to stand alone. Borno's high_burden tier instead rests
    # on the UNICEF 2026 Cholera Response Sitrep, so it now carries the evidence caveat that
    # Kano previously carried (and no longer needs).
    (cholera,) = [s for s in ENDEMICITY.for_state("Borno", TODAY) if s.disease == "Cholera"]
    assert cholera.tier == "high_burden"
    assert cholera.evidence == "press_2026_unconfirmed"
    assert cholera.evidence_note and "unicef" in cholera.evidence_note.lower()


def test_lagos_cholera_promoted_with_report_window_gap() -> None:
    # Clinical review follow-up, 2026-10-11: Lagos is added to high_burden (not in the
    # reviewer's original eight) because the 2024 NCDC sitrep's own count is thin (1 case,
    # p.6) but its reporting window ends 2 Jun 2024 and so cannot evidence Lagos going quiet
    # since -- the report separately records an active cholera EOC in Lagos (p.7).
    (cholera,) = [s for s in ENDEMICITY.for_state("Lagos", TODAY) if s.disease == "Cholera"]
    assert cholera.tier == "high_burden"
    assert cholera.evidence == "report_window_gap"
    assert cholera.evidence_note and "emergency operations centre" in cholera.evidence_note.lower()


def test_cholera_in_season_all_year() -> None:
    # The 2024 sitrep's own monthly table (Jan 294, Feb 291, Mar 210, Apr 60, May 27) shows an
    # actual Jan-Mar peak that matches neither our previous Jun-Oct window nor the reviewer's
    # suggested May-Oct -- peak_months is set to all 12 rather than hard-coding a wrong season.
    (cholera,) = [s for s in ENDEMICITY.for_state("Bayelsa", TODAY) if s.disease == "Cholera"]
    assert cholera.in_season is True


def test_baseline_carries_report_source_and_date() -> None:
    (lassa,) = [s for s in ENDEMICITY.for_state("Ondo", TODAY) if "Lassa" in s.disease]
    assert lassa.report == "NCDC Lassa Fever Situation Report"
    assert lassa.epi_week == "Week 36, 2026"
    assert lassa.page == 1
    assert lassa.report_date == date(2026, 9, 6)


def test_recent_lassa_report_is_not_older_surveillance() -> None:
    (lassa,) = [s for s in ENDEMICITY.for_state("Ondo", TODAY) if "Lassa" in s.disease]
    assert lassa.recency == "current"


def test_stale_csm_and_cholera_reports_are_older_surveillance() -> None:
    # Both reports are the newest NCDC has published, but from early-mid 2025 -- well over
    # 6 months before TODAY.
    (csm,) = [s for s in ENDEMICITY.for_state("Kano", TODAY) if "meningitis" in s.disease.lower()]
    assert csm.recency == "older"
    (cholera,) = [s for s in ENDEMICITY.for_state("Bayelsa", TODAY) if s.disease == "Cholera"]
    assert cholera.recency == "older"


def test_reason_prompt_shows_baseline_tiers_and_source() -> None:
    from backend.app.graph.prompts import reason_messages
    from backend.app.graph.state import RuleSnapshot
    from backend.app.schemas import OutbreakContext

    ctx = OutbreakContext(
        status="unavailable",
        source="none",
        message="x",
        baseline=ENDEMICITY.for_state("Kano", TODAY),
    )
    _, user = reason_messages(
        PatientCase(age_years=30, state="Kano"), RuleSnapshot(floor=None), [], ctx
    )
    text = user["content"]
    assert "HIGH-BURDEN" in text and "REPORTED" not in text  # Kano has no reported-tier entry
    assert "NCDC Cerebrospinal Meningitis Situation Report" in text
    assert "older surveillance" in text


def test_reason_prompt_labels_reported_tier_and_unconfirmed() -> None:
    from backend.app.graph.prompts import reason_messages
    from backend.app.graph.state import RuleSnapshot
    from backend.app.schemas import OutbreakContext

    ctx = OutbreakContext(
        status="unavailable",
        source="none",
        message="x",
        baseline=ENDEMICITY.for_state("Ebonyi", TODAY),
    )
    _, user = reason_messages(
        PatientCase(age_years=30, state="Ebonyi"), RuleSnapshot(floor=None), [], ctx
    )
    text = user["content"]
    assert "REPORTED" in text and "do NOT treat as grounds to escalate" in text
    assert "NOT independently confirmed" in text


# --- treatment-response question -------------------------------------------------


def test_asks_about_treatment_when_fever_3_days_and_unknown(tmp_path: Path) -> None:
    intake = {"age_years": 35, "fever_days": 5, "rdt_result": "negative", "language": "en"}
    deps, _ = make_deps(tmp_path, {"intake": intake})
    update = nodes.intake(
        TriageState(request=TriageRequest(text="Adult, fever 5 days, RDT negative")), deps
    )
    assert [q.id for q in update["questions"]] == ["treatment_response"]
    assert "antimalarials or antibiotics" in update["questions"][0].text


@pytest.mark.parametrize(
    "fields",
    [
        {"fever_days": 2},  # short fever
        {"fever_days": 5, "antimalarial_taken": True},
        {"fever_days": 5, "antibiotic_taken": False},
        {"fever_days": 5, "raw_text": "took coartem, no improvement"},
    ],
)
def test_no_treatment_question_when_known_or_short(fields: dict) -> None:
    case = PatientCase(age_years=30, rdt_result="negative", **fields)
    assert "treatment_response" not in nodes.missing_information(case)


# --- Treat & monitor advice --------------------------------------------------------


def test_green_rdt_negative_advice() -> None:
    # 2-day review interval (NMEP), not the WHO IMCI 3-day one -- Dr. Amos's review, 2026-10-10.
    texts = [t for t, _ in treat_monitor_advice(PatientCase(rdt_result="negative"))]
    assert texts[0] == REVIEW_TEXT and "2 days" in REVIEW_TEXT
    assert NO_ANTIMALARIAL_TEXT in texts
    assert any("typhoid" in t for t in texts)


def test_green_untested_says_test_first() -> None:
    texts = [t for t, _ in treat_monitor_advice(PatientCase(rdt_result=None))]
    assert texts == [REVIEW_TEXT, TEST_FIRST_TEXT]


def test_green_rdt_positive_gets_review_only() -> None:
    assert [t for t, _ in treat_monitor_advice(PatientCase(rdt_result="positive"))] == [REVIEW_TEXT]


# --- merging -------------------------------------------------------------------------


def test_topics() -> None:
    assert topics(LASSA_IPC_REMINDER) >= {"isolation", "notify", "referral"}
    assert topics("Give paracetamol and fluids") == set()


def test_merge_folds_duplicates_into_rule_action() -> None:
    actions = [
        ActionItem(text=LASSA_IPC_REMINDER, citations=["a"], source="rule"),
        ActionItem(text="Isolate the patient and wear gloves", citations=["b"]),
        ActionItem(text="Give paracetamol for fever", citations=["c"]),
        ActionItem(text="Start ribavirin", citations=[]),
    ]
    merged, n = merge_advice(actions)
    assert n == 1
    assert [a.text for a in merged] == [
        LASSA_IPC_REMINDER,
        "Give paracetamol for fever",
        "Start ribavirin",
    ]
    assert [d.text for d in merged[0].details] == ["Isolate the patient and wear gloves"]
    assert merged[0].details[0].evidence.status == "unsupported"  # no evidence given
    assert merged[0].citations == ["a", "b"]


def test_merge_without_rules_is_noop() -> None:
    actions = [ActionItem(text="Refer to hospital")]
    assert merge_advice(actions) == (actions, 0)


# --- store.find and fetcher helpers ---------------------------------------------------


def test_store_find_is_whitespace_insensitive() -> None:
    chunk = GuidelineChunk(
        id="d:0001",
        doc_id="d",
        title="T",
        section=None,
        page=1,
        page_end=1,
        text="Follow-up in 3 days\\nif fever persists",
        tokens=5,
    )
    chunk.text = "Follow-up in 3 days\nif fever persists"
    store = VectorStore(np.ones((1, 2)), [chunk])
    assert store.find("d", "follow-up in 3 days if FEVER persists") == chunk
    assert store.find("other", "follow-up") is None


def test_pdf_links_prefers_download_button() -> None:
    page = (
        "<a href=https://cdn.who.int/other.pdf>x</a>"
        '<a onclick="sendGaEvent(&#39;Download&#39;,&#39;https://iris.who.int/server/api/core/'
        'bitstreams/abc/content&#39;)">Download</a>'
        "<a href='https://evil.example/x.pdf'>y</a>"
    )
    links = pdf_links(page, "https://www.who.int/publications/x")
    assert links[0] == "https://iris.who.int/server/api/core/bitstreams/abc/content"
    assert "https://cdn.who.int/other.pdf" in links
    assert all("evil" not in link for link in links)


def test_is_pdf() -> None:
    assert is_pdf(b"%PDF-1.7\n...")
    assert not is_pdf(b"<!doctype html>")


# --- regression: live CLI run 2026-09-22 (adult, 5d fever, RDT-, Ondo -> green, no Lassa) ---

REGRESSION_TEXT = (
    "Adult man 35 years, fever 5 days, RDT negative, took coartem for 3 days but no "
    "improvement, headache and sore throat"
)
REGRESSION_INTAKE = {
    "age_years": 35,
    "sex": "male",
    "fever_days": 5,
    "rdt_result": "negative",
    "symptoms": ["fever", "headache", "sore throat"],
    "antimalarial_taken": True,
    "antimalarial_no_response": True,
    "language": "en",
}
# What the model actually did without outbreak data or an index: green, no Lassa.
GENERIC_REASON = {
    "triage_level": "treat_monitor",
    "triage_rationale": "Febrile illness, malaria excluded by RDT.",
    "differential": [
        {"condition": "Viral illness", "likelihood": "moderate"},
        {"condition": "Typhoid fever", "likelihood": "low"},
    ],
    "actions": [{"text": "Give paracetamol and advise fluids"}],
}


def run_regression(tmp_path: Path, *, live: bool):
    replies = {"intake": REGRESSION_INTAKE, "reason": GENERIC_REASON, "compose": COMPOSE_EN}
    if live:
        replies["outbreak extraction"] = OUTBREAK_EXTRACTION
    deps, _ = make_deps(tmp_path, replies, search="mock" if live else None)
    deps.outbreak.endemicity = ENDEMICITY
    return run_triage(TriageRequest(text=REGRESSION_TEXT, state="Ondo"), deps)


def test_regression_baseline_only_still_suspects_lassa(tmp_path: Path) -> None:
    r = run_regression(tmp_path, live=False)
    assert r.triage_level == TriageLevel.REFER_NOW  # no longer "Treat & monitor"
    lassa = r.differential[0]
    assert lassa.condition == "Lassa fever" and lassa.likelihood == "moderate"
    assert lassa.source == "rule"
    assert r.actions[0].text == LASSA_IPC_REMINDER
    assert r.outbreak.baseline and r.outbreak.signals == []
    assert any("Using baseline endemicity only" in w for w in r.warnings)
    # No longer provisional: endemicity.yaml is now sourced against current NCDC reports.
    assert not any("provisional" in w for w in r.warnings)


def test_regression_live_signal_raises_likelihood(tmp_path: Path) -> None:
    r = run_regression(tmp_path, live=True)
    assert r.triage_level == TriageLevel.REFER_NOW
    lassa = r.differential[0]
    assert lassa.condition == "Lassa fever" and lassa.likelihood == "high"  # baseline: moderate
    assert any("Active Lassa outbreak" in r.text for r in lassa.reasons)
    assert all(r.evidence.status == "rule" for r in lassa.reasons)


def test_green_rdt_negative_case_is_actionable(tmp_path: Path) -> None:
    intake = {
        "age_years": 30,
        "fever_days": 2,
        "rdt_result": "negative",
        "symptoms": ["cough"],
        "language": "en",
    }
    reason = {
        "triage_level": "treat_monitor",
        "triage_rationale": "Mild febrile illness.",
        "differential": [{"condition": "Viral upper respiratory infection", "likelihood": "high"}],
        "actions": [{"text": "Come back for review if not better"}],
    }
    deps, _ = make_deps(
        tmp_path, {"intake": intake, "reason": reason, "compose": COMPOSE_EN}, search=None
    )
    r = run_triage(
        TriageRequest(text="Adult 30, fever 2 days, RDT negative, cough", state="Lagos"), deps
    )
    assert r.triage_level == TriageLevel.TREAT_MONITOR
    texts = [a.text for a in r.actions]
    assert REVIEW_TEXT in texts and NO_ANTIMALARIAL_TEXT in texts
    assert any("typhoid" in t for t in texts)
    review = next(a for a in r.actions if a.text == REVIEW_TEXT)
    assert [d.text for d in review.details] == ["Come back for review if not better"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ncdc-lassa:0002", "ncdc-lassa:0002"),
        ("[ncdc-lassa:0002]", "ncdc-lassa:0002"),
        ("ncdc-lassa:0002 (p.8)", "ncdc-lassa:0002"),
        (" who-imci:0005, ", "who-imci:0005"),
        ("[https://ncdc.gov.ng/x/y]", "https://ncdc.gov.ng/x/y"),
        ("https://ncdc.gov.ng/x.", "https://ncdc.gov.ng/x"),
        ("made-up", "made-up"),
    ],
)
def test_citation_refs_are_normalised(raw: str, expected: str) -> None:
    assert nodes._normalise_ref(raw) == expected


def test_unicode_hyphens_are_matched() -> None:
    # Super writes U+2011 non-breaking hyphens ("follow‑up"); seen on the phone test.
    rule = ActionItem(text="Review in 3 days if the fever persists.", source="rule")
    model = ActionItem(text="Schedule follow‑up visit in 3 days")
    merged, n = merge_advice([rule, model])
    assert n == 1 and [d.text for d in merged[0].details] == ["Schedule follow‑up visit in 3 days"]


def test_unicode_dash_doses_are_stripped() -> None:
    from backend.app.rules.dosing import DOSE_PLACEHOLDER, strip_doses

    cleaned, n = strip_doses("give 10–20 mg/kg")
    assert n == 1 and DOSE_PLACEHOLDER in cleaned
