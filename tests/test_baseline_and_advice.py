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


def test_non_endemic_state_has_no_baseline_tier() -> None:
    case = PatientCase(fever_days=5, state="Lagos", raw_text=UNRESPONSIVE)
    assert lassa_assessment(case, ENDEMICITY.for_state("Lagos", TODAY), []) is None


def test_antibiotic_failure_counts() -> None:
    text = "fever 6 days, vomiting, took amoxicillin, still no improvement"
    case = PatientCase(fever_days=6, state="Ondo", raw_text=text)
    assert lassa_assessment(case, baseline_ondo(), []) is not None


# --- endemicity baseline ---------------------------------------------------------


def test_endemicity_yaml_is_marked_provisional() -> None:
    assert ENDEMICITY.entries and not ENDEMICITY.all_verified


def test_baseline_for_state_and_season() -> None:
    (lassa,) = [
        s for s in ENDEMICITY.for_state("ondo state", date(2026, 1, 10)) if "Lassa" in s.disease
    ]
    assert lassa.basis == "baseline" and lassa.status == "endemic" and lassa.in_season is True
    (off,) = [s for s in ENDEMICITY.for_state("Ondo", date(2026, 7, 1)) if "Lassa" in s.disease]
    assert off.in_season is False
    assert ENDEMICITY.for_state("Lagos", TODAY) == []
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
    texts = [t for t, _ in treat_monitor_advice(PatientCase(rdt_result="negative"))]
    assert texts[0] == REVIEW_TEXT and "3 days" in REVIEW_TEXT
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
    assert any("provisional" in w for w in r.warnings)


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
