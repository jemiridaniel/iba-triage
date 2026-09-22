"""End-to-end pipeline scenarios with a scripted Token Factory. No network."""

from pathlib import Path

import pytest

from backend.app.graph.pipeline import run_triage
from backend.app.graph.state import TriageRequest
from backend.app.llm.spend import SpendLimitError
from backend.app.rules.danger_signs import LASSA_IPC_REMINDER
from backend.app.rules.dosing import DOSE_PLACEHOLDER, UNVERIFIED_WARNING
from backend.app.schemas import DangerSignCode, FollowUpAnswer, TriageLevel
from tests.pipeline_fakes import (
    COMPOSE_EN,
    COMPOSE_PCM,
    LASSA_URL,
    OUTBREAK_EXTRACTION,
    make_deps,
)

# --- scenario 1: child with danger signs -> refer now, by rule ---------------

CHILD_TEXT = "Pikin 2 years, hot body 3 days, e dey convulse, e no fit drink, RDT positive"
CHILD_INTAKE = {
    # The intake model misses both danger signs; the text rules must catch them.
    "age_years": 2,
    "fever_days": 3,
    "rdt_result": "positive",
    "symptoms": ["fever"],
    "danger_signs": [],
    "language": "pcm",
}
CHILD_REASON = {
    # The reasoning model under-triages; the rule floor must win.
    "triage_level": "refer_24h",
    "triage_rationale": "Malaria, RDT positive.",
    "differential": [
        {
            "condition": "Severe malaria",
            "likelihood": "high",
            "reasons": ["RDT positive"],
            "citations": ["fake-malaria:0000"],
        }
    ],
    "actions": [{"text": "Refer to hospital", "citations": ["fake-imci:0007"]}],
}


def test_child_with_danger_signs_is_refer_now_by_rule(tmp_path: Path) -> None:
    deps, llm = make_deps(
        tmp_path,
        {"intake": CHILD_INTAKE, "reason": CHILD_REASON, "compose": COMPOSE_PCM},
        search=None,
    )
    r = run_triage(TriageRequest(text=CHILD_TEXT, state="Kano"), deps)

    assert r.status == "complete"
    assert r.triage_level == TriageLevel.REFER_NOW
    assert r.triage_label == "Refer now"
    assert {h.code for h in r.danger_signs} == {
        DangerSignCode.CONVULSIONS,
        DangerSignCode.UNABLE_TO_DRINK,
    }
    assert all(h.source == "rule" for h in r.danger_signs)
    assert r.triage_rationale.startswith("Raised from Refer within 24h to Refer now")
    assert r.actions[0].source == "rule" and r.actions[0].text.startswith("Refer now")
    assert "Pidgin" in llm.messages_for("compose")[0]["content"]
    assert "helper only" in r.referral_note  # Pidgin disclaimer
    assert any("outbreak data unavailable" in w.lower() for w in r.warnings)  # never silent


# --- scenario 2: adult, RDT negative, 5 days fever, Ondo + Lassa signal ------

ADULT_TEXT = (
    "Adult man 35 years, fever 5 days, RDT negative, took coartem for 3 days but no "
    "improvement, headache and sore throat"
)
ADULT_INTAKE = {
    "age_years": 35,
    "sex": "male",
    "fever_days": 5,
    "rdt_result": "negative",
    "symptoms": ["fever", "headache", "sore throat"],
    "antimalarial_taken": True,
    "antimalarial_no_response": True,
    "language": "en",
}
TYPHOID_QUOTE = "Look for other causes of fever such as typhoid, pneumonia, urinary infection"
ADULT_REASON_NO_LASSA = {
    # Model ignores the outbreak entirely: the Lassa rule must add it anyway.
    "triage_level": "refer_24h",
    "triage_rationale": "Persistent fever with negative RDT.",
    "differential": [
        {
            "condition": "Typhoid fever",
            "likelihood": "moderate",
            "reasons": [
                {"text": "Fever for 5 days", "basis": "patient"},
                {
                    "text": "RDT-negative fever needs another cause",
                    "chunk_id": "fake-malaria:0002",
                    "evidence_quote": TYPHOID_QUOTE,
                },
                {  # invented chunk
                    "text": "Typhoid is common in adults",
                    "chunk_id": "fake-typhoid:0042",
                    "evidence_quote": TYPHOID_QUOTE,
                },
            ],
        },
    ],
    "actions": [
        {
            "text": "Refer for further tests",
            "chunk_id": "https://invented.example",
            "evidence_quote": "anything at all",
        },
    ],
}


def test_adult_in_lassa_outbreak_state_gets_lassa_and_isolation(tmp_path: Path) -> None:
    deps, llm = make_deps(
        tmp_path,
        {
            "intake": ADULT_INTAKE,
            "outbreak extraction": OUTBREAK_EXTRACTION,
            "reason": ADULT_REASON_NO_LASSA,
            "compose": COMPOSE_EN,
        },
    )
    r = run_triage(TriageRequest(text=ADULT_TEXT, state="Ondo"), deps)

    assert r.triage_level == TriageLevel.REFER_NOW
    assert r.lassa_suspected
    lassa = r.differential[0]
    assert lassa.condition == "Lassa fever" and lassa.source == "rule"
    assert lassa.citations == [LASSA_URL]
    assert r.actions[0].text == LASSA_IPC_REMINDER
    assert "holding area" in LASSA_IPC_REMINDER and "infection prevention" in LASSA_IPC_REMINDER

    # Outbreak: invented-URL signal dropped; mock data flagged. The undated one is kept for
    # context but marked "unknown", which bars it from escalating (FEEDBACK T1).
    assert r.outbreak.status == "ok" and r.outbreak.source == "mock"
    assert {(s.disease, s.state) for s in r.outbreak.signals} == {
        ("Lassa fever", "Ondo"),
        ("Cholera", "Bauchi"),
        ("Meningitis", "Kebbi"),
    }
    by_disease = {s.disease: s for s in r.outbreak.signals}
    assert by_disease["Meningitis"].recency == "unknown"
    assert by_disease["Meningitis"].report_date is None
    assert by_disease["Lassa fever"].recency == "current"
    assert any("MOCK" in w for w in r.warnings)

    # Grounding: only the quote-verified reason is cited; the rest are kept but unsupported.
    typhoid = r.differential[1]
    assert [x.evidence.status for x in typhoid.reasons] == ["patient", "verified", "unsupported"]
    assert typhoid.citations == ["fake-malaria:0002"]
    assert typhoid.reasons[1].evidence.quote == TYPHOID_QUOTE
    # The model's "Refer for further tests" is merged into the rule's Lassa action as detail;
    # its invented citation is still dropped.
    assert len(r.actions) == 1
    assert [d.text for d in r.actions[0].details] == ["Refer for further tests"]
    assert r.actions[0].details[0].evidence.status == "unsupported"  # shown grey, not as sourced
    assert "https://invented.example" not in r.actions[0].citations
    assert (r.grounding.claims, r.grounding.verified, r.grounding.unsupported) == (3, 1, 2)
    assert any("no verified guideline source" in w for w in r.warnings)
    assert {c.ref for c in r.citations} == {LASSA_URL, "fake-malaria:0002"}

    # The reasoning prompt saw the outbreak signal and the retrieved Lassa passages.
    reason_user = llm.messages_for("reason")[1]["content"]
    assert LASSA_URL in reason_user
    assert "fake-lassa:" in reason_user


def test_lassa_scenario_keeps_model_lassa_item_without_duplicating(tmp_path: Path) -> None:
    reason = {
        **ADULT_REASON_NO_LASSA,
        "triage_level": "refer_now",
        "differential": [
            {
                "condition": "Lassa fever (suspected)",
                "likelihood": "high",
                "reasons": ["Active outbreak in Ondo"],
                "citations": [LASSA_URL],
            }
        ],
    }
    deps, _ = make_deps(
        tmp_path,
        {
            "intake": ADULT_INTAKE,
            "outbreak extraction": OUTBREAK_EXTRACTION,
            "reason": reason,
            "compose": COMPOSE_EN,
        },
    )
    r = run_triage(TriageRequest(text=ADULT_TEXT, state="Ondo"), deps)
    assert [d.condition for d in r.differential] == ["Lassa fever (suspected)"]
    assert r.differential[0].source == "llm"


def test_same_case_outside_outbreak_state_has_no_lassa_rule(tmp_path: Path) -> None:
    deps, _ = make_deps(
        tmp_path,
        {
            "intake": ADULT_INTAKE,
            "outbreak extraction": OUTBREAK_EXTRACTION,
            "reason": ADULT_REASON_NO_LASSA,
            "compose": COMPOSE_EN,
        },
    )
    r = run_triage(TriageRequest(text=ADULT_TEXT, state="Lagos"), deps)
    assert not r.lassa_suspected
    assert r.triage_level == TriageLevel.REFER_24H
    assert all("lassa" not in d.condition.lower() for d in r.differential)


# --- scenario 3: uncomplicated RDT-positive malaria -> treat & monitor -------

MALARIA_TEXT = "Adult woman 28 years, 62 kg, fever 2 days, RDT positive, eating and drinking well"
MALARIA_INTAKE = {
    "age_years": 28,
    "sex": "female",
    "weight_kg": 62,
    "fever_days": 2,
    "rdt_result": "positive",
    "symptoms": ["fever"],
    "language": "en",
}
MALARIA_REASON = {
    "triage_level": "treat_monitor",
    "triage_rationale": "Uncomplicated malaria, no danger signs.",
    "differential": [
        {
            "condition": "Uncomplicated malaria",
            "likelihood": "high",
            "reasons": ["RDT positive", "no danger signs"],
            "citations": ["fake-malaria:0001"],
        }
    ],
    # The model sneaks in a dose: it must be stripped and replaced by the table.
    "actions": [
        {
            "text": "Give artemether-lumefantrine 80/480 mg twice daily",
            "citations": ["fake-malaria:0001"],
        },
        {"text": "Return at once if danger signs appear", "citations": []},
    ],
}


def test_uncomplicated_malaria_is_treat_and_monitor_with_table_dose(tmp_path: Path) -> None:
    deps, _ = make_deps(
        tmp_path,
        {"intake": MALARIA_INTAKE, "reason": MALARIA_REASON, "compose": COMPOSE_EN},
        search=None,
    )
    r = run_triage(TriageRequest(text=MALARIA_TEXT, state="Lagos"), deps)

    assert r.status == "complete"
    assert r.triage_level == TriageLevel.TREAT_MONITOR
    assert r.danger_signs == []
    assert "80/480" not in r.actions[0].text and DOSE_PLACEHOLDER in r.actions[0].text
    (dose,) = r.doses
    assert dose.weight_band == "35 kg and above"
    assert dose.regimen.startswith("4 tablets per dose")
    assert dose.verified is False
    assert UNVERIFIED_WARNING in r.warnings
    assert "UNVERIFIED table" in r.referral_note
    assert any("Removed 1 dose mention" in w for w in r.warnings)


# --- follow-up questions and resume ------------------------------------------


def test_missing_fields_ask_questions_then_resume(tmp_path: Path) -> None:
    sparse = {"age_years": 30, "symptoms": ["fever"], "language": "en"}
    deps, llm = make_deps(
        tmp_path,
        {"intake": [sparse, MALARIA_INTAKE], "reason": MALARIA_REASON, "compose": COMPOSE_EN},
        search=None,
    )
    first = run_triage(TriageRequest(text="Woman with fever", state="Lagos"), deps)
    assert first.status == "needs_info"
    assert [q.id for q in first.questions] == ["fever_days", "rdt_result"]
    assert [s.step for s in first.decision_trace.steps] == ["intake", "rules_pre"]
    assert llm.tasks() == ["intake"]

    answers = [
        FollowUpAnswer(id="fever_days", answer="2 days"),
        FollowUpAnswer(id="rdt_result", answer="positive"),
    ]
    second = run_triage(
        TriageRequest(text="Woman with fever", state="Lagos", answers=answers), deps
    )
    assert second.status == "complete"
    assert "How many days has the fever lasted? 2 days" in llm.messages_for_last("intake")


def test_danger_signs_never_wait_for_questions(tmp_path: Path) -> None:
    sparse = {"symptoms": ["fever"], "language": "pcm"}
    deps, _ = make_deps(
        tmp_path, {"intake": sparse, "reason": CHILD_REASON, "compose": COMPOSE_PCM}, search=None
    )
    r = run_triage(TriageRequest(text="Pikin hot body, e dey convulse"), deps)
    assert r.status == "complete"
    assert r.triage_level == TriageLevel.REFER_NOW


def test_skip_questions_proceeds(tmp_path: Path) -> None:
    sparse = {"age_years": 30, "language": "en"}
    deps, _ = make_deps(
        tmp_path, {"intake": sparse, "reason": MALARIA_REASON, "compose": COMPOSE_EN}, search=None
    )
    r = run_triage(TriageRequest(text="Woman with fever", skip_questions=True), deps)
    assert r.status == "complete"
    assert r.doses == []  # rdt/weight unknown: no table dose


# --- fail safe ----------------------------------------------------------------


def test_reason_failure_fails_safe_to_refer(tmp_path: Path) -> None:
    deps, _ = make_deps(
        tmp_path,
        {"intake": MALARIA_INTAKE, "reason": ["not json", "still not"], "compose": COMPOSE_EN},
        search=None,
    )
    r = run_triage(TriageRequest(text=MALARIA_TEXT), deps)
    assert r.status == "incomplete"
    assert r.triage_level == TriageLevel.REFER_NOW
    assert r.triage_rationale == "Refer: Ibà could not complete the assessment."
    assert r.doses == []
    reason_step = next(s for s in r.decision_trace.steps if s.step == "reason")
    assert "LLMParseError" in reason_step.note


def test_intake_failure_skips_to_rules_and_refers(tmp_path: Path) -> None:
    deps, llm = make_deps(tmp_path, {"intake": ["bad", "bad"], "compose": COMPOSE_PCM}, search=None)
    r = run_triage(TriageRequest(text="Pikin hot body, e dey convulse"), deps)
    assert r.status == "incomplete"
    assert r.triage_level == TriageLevel.REFER_NOW
    assert DangerSignCode.CONVULSIONS in {h.code for h in r.danger_signs}
    assert "reason" not in llm.tasks()
    assert [s.step for s in r.decision_trace.steps] == [
        "intake",
        "rules_pre",
        "rules_post",
        "compose",
    ]


def test_compose_failure_uses_template(tmp_path: Path) -> None:
    deps, _ = make_deps(
        tmp_path,
        {"intake": MALARIA_INTAKE, "reason": MALARIA_REASON, "compose": ["x", "y"]},
        search=None,
    )
    r = run_triage(TriageRequest(text=MALARIA_TEXT), deps)
    assert r.status == "complete"
    assert r.referral_note.startswith("IBÀ TRIAGE NOTE")
    assert "\nTriage: Treat & monitor" in r.referral_note
    assert any("template" in w for w in r.warnings)


def test_spend_limit_propagates(tmp_path: Path) -> None:
    deps, _ = make_deps(tmp_path, {"intake": MALARIA_INTAKE}, search=None, max_spend_usd=0.0)
    with pytest.raises(SpendLimitError):
        run_triage(TriageRequest(text=MALARIA_TEXT), deps)


# --- decision trace ------------------------------------------------------------


def test_decision_trace_records_each_step(tmp_path: Path) -> None:
    deps, _ = make_deps(
        tmp_path,
        {
            "intake": ADULT_INTAKE,
            "outbreak extraction": OUTBREAK_EXTRACTION,
            "reason": ADULT_REASON_NO_LASSA,
            "compose": COMPOSE_EN,
        },
    )
    r = run_triage(TriageRequest(text=ADULT_TEXT, state="Ondo"), deps)
    steps = {s.step: s for s in r.decision_trace.steps}
    order = list(steps)
    assert order[:2] == ["intake", "rules_pre"]
    assert set(order[2:4]) == {"outbreak", "embed_queries"}  # run in parallel
    assert order[4:] == ["retrieve", "reason", "rules_post", "compose"]
    assert "embedded while checking outbreaks" in steps["embed_queries"].note
    assert (steps["intake"].model, steps["intake"].reasoning) == ("fast", False)
    assert (steps["outbreak"].model, steps["outbreak"].reasoning) == ("fast", False)
    assert (steps["reason"].model, steps["reason"].reasoning) == ("reason", True)
    assert (steps["compose"].model, steps["compose"].reasoning) == ("mid", False)
    assert steps["rules_pre"].kind == "rules" and steps["rules_pre"].model is None
    assert steps["reason"].prompt_tokens == 100 and steps["reason"].completion_tokens == 50
    # 100 in * $1/M + 50 out * $2/M = $0.0002 per call, four LLM calls
    assert r.decision_trace.total_cost_usd == pytest.approx(0.0008)
    # Only the invented URL is dropped now; the undated signal is kept as recency "unknown".
    assert "3 signals kept, 1 dropped" in steps["outbreak"].note
    assert "verified=1/3" in steps["rules_post"].note


def test_eval_config_reason_only_routes_every_step_to_reason_model(tmp_path: Path) -> None:
    deps, llm = make_deps(
        tmp_path,
        {"intake": MALARIA_INTAKE, "reason": MALARIA_REASON, "compose": COMPOSE_EN},
        search=None,
    )
    run_triage(TriageRequest(text=MALARIA_TEXT), deps, eval_config="reason-only")
    assert {kw["model"] for _, kw in llm.calls} == {"reason"}
