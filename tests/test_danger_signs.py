"""Danger-sign rules. Pure Python: no LLM, no network."""

from datetime import date

import pytest

from backend.app.rules.danger_signs import (
    LASSA_IPC_REMINDER,
    apply_floor,
    assess,
    detect_danger_signs,
    detect_from_text,
    no_antimalarial_response,
)
from backend.app.schemas import (
    DangerSignCode as D,
    DangerSignHit,
    OutbreakSignal,
    PatientCase,
    TriageLevel,
)


def codes(text: str) -> set[D]:
    return {h.code for h in detect_from_text(text)}


# --- positive detection (English + Pidgin) ----------------------------------

POSITIVES = [
    (D.CONVULSIONS, "child had convulsions last night"),
    (D.CONVULSIONS, "pikin dey convulse since morning"),
    (D.CONVULSIONS, "had a seizure at home"),
    (D.CONVULSIONS, "e get fits yesterday"),
    (D.CONVULSIONS, "body dey jerk"),
    (D.UNABLE_TO_DRINK, "baby is unable to breastfeed"),
    (D.UNABLE_TO_DRINK, "she cannot drink anything"),
    (D.UNABLE_TO_DRINK, "e no fit drink water"),
    (D.UNABLE_TO_DRINK, "pikin no gree suck breast"),
    (D.UNABLE_TO_DRINK, "not sucking since yesterday"),
    (D.UNABLE_TO_DRINK, "refuses to feed"),
    (D.VOMITING_EVERYTHING, "vomiting everything"),
    (D.VOMITING_EVERYTHING, "vomit everything"),
    (D.VOMITING_EVERYTHING, "can't keep anything down"),
    (D.VOMITING_EVERYTHING, "anything e chop e dey vomit"),
    (D.VOMITING_EVERYTHING, "everything he eats, he vomits"),
    (D.VOMITING_EVERYTHING, "persistent vomiting for 2 days"),
    (D.LETHARGY_UNCONSCIOUS, "very lethargic"),
    (D.LETHARGY_UNCONSCIOUS, "e don dey sleep too much"),
    (D.LETHARGY_UNCONSCIOUS, "difficult to wake"),
    (D.LETHARGY_UNCONSCIOUS, "patient is unconscious"),
    (D.LETHARGY_UNCONSCIOUS, "confused since morning"),
    (D.LETHARGY_UNCONSCIOUS, "not responding to voice"),
    (D.LETHARGY_UNCONSCIOUS, "baby is floppy"),
    (D.LETHARGY_UNCONSCIOUS, "altered level of consciousness"),
    (D.BLEEDING, "bleeding from the gums"),
    (D.BLEEDING, "blood dey comot for nose"),
    (D.BLEEDING, "nosebleed this morning"),
    (D.BLEEDING, "vomiting blood"),
    (D.BLEEDING, "blood from injection site"),
    (D.JAUNDICE, "yellow eyes"),
    (D.JAUNDICE, "eyes don turn yellow"),
    (D.JAUNDICE, "jaundiced"),
    (D.JAUNDICE, "her skin is yellow"),
    (D.RESPIRATORY_DISTRESS, "difficulty breathing"),
    (D.RESPIRATORY_DISTRESS, "short of breath"),
    (D.RESPIRATORY_DISTRESS, "chest indrawing present"),
    (D.RESPIRATORY_DISTRESS, "e no fit breathe well"),
    (D.RESPIRATORY_DISTRESS, "grunting"),
    (D.RESPIRATORY_DISTRESS, "deep breathing"),
    (D.SEVERE_PALLOR, "very pale"),
    (D.SEVERE_PALLOR, "severe pallor of palms"),
    (D.SEVERE_PALLOR, "no blood for body"),
    (D.SEVERE_PALLOR, "palm don white"),
    (D.SEVERE_PALLOR, "severe anaemia"),
    (D.DARK_URINE, "dark urine"),
    (D.DARK_URINE, "coca-cola coloured urine"),
    (D.DARK_URINE, "tea-coloured urine"),
    (D.DARK_URINE, "urine dey black"),
    (D.DARK_URINE, "the urine is very dark"),
    (D.DARK_URINE, "blackwater fever suspected"),
    (D.PROSTRATION, "unable to sit without support"),
    (D.PROSTRATION, "too weak to stand"),
    (D.PROSTRATION, "e no fit waka"),
    (D.PROSTRATION, "prostrated"),
]


@pytest.mark.parametrize(("code", "text"), POSITIVES)
def test_detects_sign(code: D, text: str) -> None:
    assert code in codes(text)


def test_every_sign_has_a_positive_case() -> None:
    assert {code for code, _ in POSITIVES} == set(D)


def test_case_insensitive() -> None:
    assert D.CONVULSIONS in codes("CONVULSIONS x2")


def test_evidence_is_the_matched_phrase() -> None:
    (hit,) = detect_from_text("fever 3 days, vomiting everything")
    assert hit.source == "rule"
    assert hit.evidence == "vomiting everything"


def test_spec_demo_pidgin_case() -> None:
    text = (
        "Pikin 3 years, hot body 4 days, vomit everything, RDT negative, e don dey sleep too much"
    )
    assert codes(text) == {D.VOMITING_EVERYTHING, D.LETHARGY_UNCONSCIOUS}


def test_multiple_signs() -> None:
    text = "adult, fever 5 days, yellow eyes, dark urine, convulsed once, bleeding from gums"
    assert codes(text) == {D.JAUNDICE, D.DARK_URINE, D.CONVULSIONS, D.BLEEDING}


# --- no false alarms --------------------------------------------------------

CLEAN = [
    "Adult male, fever 2 days, headache, body pain, RDT positive, eating and drinking well",
    "hot body 2 days, catarrh, cough, e dey play",
    "fever and chills, took paracetamol",
    "mild pallor",
    "he acts normal",
    "fever not responding to antimalarials",
    "unresponsive to treatment",
]


@pytest.mark.parametrize("text", CLEAN)
def test_no_signs_in_uncomplicated_text(text: str) -> None:
    assert codes(text) == set()


# --- negation ---------------------------------------------------------------

NEGATED = [
    "no convulsions",
    "no history of convulsions",
    "denies bleeding",
    "not vomiting everything",
    "without jaundice",
    "e no dey sleep too much",
    "never had fits",
]


@pytest.mark.parametrize("text", NEGATED)
def test_negated_sign_is_ignored(text: str) -> None:
    assert codes(text) == set()


def test_negation_does_not_cross_clause() -> None:
    assert codes("no cough, vomiting everything") == {D.VOMITING_EVERYTHING}
    assert codes("no cough and convulsing") == {D.CONVULSIONS}
    assert codes("no rash but yellow eyes") == {D.JAUNDICE}


def test_negation_form_that_is_the_sign_itself() -> None:
    # "no fit drink", "no dey drink" contain "no" but describe the danger sign.
    assert codes("e no fit drink") == {D.UNABLE_TO_DRINK}
    assert codes("pikin no dey drink") == {D.UNABLE_TO_DRINK}


def test_illness_idiom_is_not_negation() -> None:
    # "no well" = unwell in Pidgin; must not suppress the following sign.
    assert codes("pikin no well sleeping too much") == {D.LETHARGY_UNCONSCIOUS}
    assert codes("he is not better convulsing now") == {D.CONVULSIONS}


def test_negated_mention_does_not_hide_later_positive() -> None:
    assert codes("no convulsions yesterday; today convulsions started") == {D.CONVULSIONS}


# --- structured channel -----------------------------------------------------


def test_structured_flags_from_intake_are_included() -> None:
    case = PatientCase(danger_signs=[D.JAUNDICE], raw_text="fever 3 days")
    (hit,) = detect_danger_signs(case)
    assert hit.code == D.JAUNDICE
    assert hit.source == "llm_intake"


def test_text_rule_wins_over_structured_duplicate() -> None:
    case = PatientCase(danger_signs=[D.CONVULSIONS], raw_text="convulsed twice")
    (hit,) = detect_danger_signs(case)
    assert hit.source == "rule"


def test_text_channel_catches_what_intake_missed() -> None:
    case = PatientCase(danger_signs=[], raw_text="baby no fit suck, grunting")
    assert {h.code for h in detect_danger_signs(case)} == {
        D.UNABLE_TO_DRINK,
        D.RESPIRATORY_DISTRESS,
    }


def test_raw_text_is_never_serialised() -> None:
    case = PatientCase(raw_text="secret free text")
    assert "raw_text" not in case.model_dump()
    assert "secret" not in case.model_dump_json()
    assert "secret" not in repr(case)


# --- triage floor -----------------------------------------------------------


def test_no_signs_means_no_floor() -> None:
    result = assess(PatientCase(fever_days=2, raw_text="fever 2 days, RDT positive"))
    assert result.floor is None
    assert result.danger_signs == []


def test_any_sign_forces_refer_now() -> None:
    result = assess(PatientCase(raw_text="fever, convulsions"))
    assert result.floor == TriageLevel.REFER_NOW
    assert result.reasons == ["Danger sign: Convulsions"]


@pytest.mark.parametrize("llm_level", list(TriageLevel))
def test_llm_cannot_downgrade_floor(llm_level: TriageLevel) -> None:
    assert apply_floor(llm_level, TriageLevel.REFER_NOW) == TriageLevel.REFER_NOW


def test_llm_can_escalate_above_floor() -> None:
    assert apply_floor(TriageLevel.REFER_24H, None) == TriageLevel.REFER_24H
    assert apply_floor(TriageLevel.REFER_NOW, TriageLevel.REFER_24H) == TriageLevel.REFER_NOW


def test_missing_llm_level_fails_safe_to_refer() -> None:
    assert apply_floor(None, None) == TriageLevel.REFER_NOW


def test_extra_hits_from_reason_step_raise_floor() -> None:
    case = PatientCase(raw_text="fever 2 days")
    extra = [DangerSignHit(code=D.SEVERE_PALLOR, source="llm_reason")]
    result = assess(case, extra_hits=extra)
    assert result.floor == TriageLevel.REFER_NOW
    assert [h.source for h in result.danger_signs] == ["llm_reason"]


def test_extra_hits_do_not_duplicate_rule_hits() -> None:
    case = PatientCase(raw_text="convulsions")
    extra = [DangerSignHit(code=D.CONVULSIONS, source="llm_reason")]
    result = assess(case, extra_hits=extra)
    assert [(h.code, h.source) for h in result.danger_signs] == [(D.CONVULSIONS, "rule")]


# --- Lassa suspicion --------------------------------------------------------

ONDO_LASSA = OutbreakSignal(
    disease="Lassa fever",
    state="Ondo",
    status="active",
    report_date=date(2026, 10, 12),
    url="https://ncdc.gov.ng/diseases/sitreps",
)


def lassa_case(**overrides) -> PatientCase:
    fields = {
        "age_years": 30,
        "fever_days": 5,
        "rdt_result": "negative",
        "antimalarial_no_response": True,
        "state": "Ondo",
        "raw_text": "adult, fever 5 days, took coartem, no change",
    }
    return PatientCase(**{**fields, **overrides})


def test_lassa_rule_fires_in_outbreak_state() -> None:
    result = assess(lassa_case(), [ONDO_LASSA])
    assert result.lassa_suspected
    assert result.lassa_signal == ONDO_LASSA
    assert result.floor == TriageLevel.REFER_NOW
    assert LASSA_IPC_REMINDER in result.reminders
    assert result.danger_signs == []


def test_lassa_rule_matches_state_loosely() -> None:
    assert assess(lassa_case(state="ondo state"), [ONDO_LASSA]).lassa_suspected


def test_lassa_rule_needs_outbreak_signal() -> None:
    # No live/baseline signal at all: the any-state floor (fever >=3 days, failed treatment)
    # still fires on its own, at Refer within 24h -- not the live-signal Refer now tier.
    result = assess(lassa_case(), [])
    assert result.lassa_suspected
    assert result.lassa is not None
    assert result.lassa.basis == "any_state"
    assert result.lassa.signal is None
    assert result.floor == TriageLevel.REFER_24H


def test_lassa_rule_ignores_other_states() -> None:
    # The Ondo live signal must not escalate a Lagos case via the "live" tier; the any-state
    # floor still applies on its own (fever >=3 days, failed treatment -> Refer within 24h).
    result = assess(lassa_case(state="Lagos"), [ONDO_LASSA])
    assert result.lassa is not None
    assert result.lassa.basis == "any_state"
    assert result.floor == TriageLevel.REFER_24H


@pytest.mark.parametrize(
    "signal_overrides",
    [{"status": "over"}, {"url": None}, {"report_date": None}, {"disease": "Cholera"}],
)
def test_lassa_rule_needs_active_sourced_lassa_signal(signal_overrides: dict) -> None:
    # An invalid/wrong-disease signal must not count as a live Lassa signal; the any-state
    # floor (fever >=3 days, failed treatment) still fires on its own, at Refer within 24h.
    signal = ONDO_LASSA.model_copy(update=signal_overrides)
    result = assess(lassa_case(), [signal])
    assert result.lassa is not None
    assert result.lassa.basis == "any_state"
    assert result.floor == TriageLevel.REFER_24H


def test_lassa_any_state_floor_contact_plus_fever_is_refer_now() -> None:
    case = lassa_case(
        state="Lagos",
        antimalarial_no_response=None,
        raw_text="fever 1 day, contact with a confirmed Lassa case last week",
    )
    result = assess(case, [])
    assert result.lassa is not None
    assert result.lassa.basis == "any_state"
    assert result.floor == TriageLevel.REFER_NOW


def test_lassa_any_state_floor_pregnant_with_prolonged_fever_is_refer_now() -> None:
    case = lassa_case(
        state="Lagos",
        antimalarial_no_response=None,
        pregnant=True,
        raw_text="adult, fever 4 days",
    )
    result = assess(case, [])
    assert result.lassa is not None
    assert result.lassa.basis == "any_state"
    assert result.floor == TriageLevel.REFER_NOW


def test_lassa_rule_needs_three_days_of_fever() -> None:
    assert not assess(lassa_case(fever_days=2), [ONDO_LASSA]).lassa_suspected


def test_lassa_rule_needs_antimalarial_failure_or_bleeding() -> None:
    case = lassa_case(antimalarial_no_response=None, raw_text="adult, fever 5 days")
    assert not assess(case, [ONDO_LASSA]).lassa_suspected


def test_lassa_rule_fires_on_bleeding_regardless_of_duration() -> None:
    case = lassa_case(fever_days=1, antimalarial_no_response=None, raw_text="bleeding from gums")
    result = assess(case, [ONDO_LASSA])
    assert result.lassa_suspected
    assert {h.code for h in result.danger_signs} == {D.BLEEDING}


@pytest.mark.parametrize(
    "text",
    [
        "took coartem for 3 days, no improvement",
        "e don take malaria drug but e no better",
        "fever not responding to antimalarials",
        "completed ACT, fever persists",
    ],
)
def test_antimalarial_failure_from_text(text: str) -> None:
    assert no_antimalarial_response(PatientCase(raw_text=text))


def test_antimalarial_failure_absent() -> None:
    assert not no_antimalarial_response(
        PatientCase(raw_text="took coartem yesterday, feeling better")
    )


# --- sunken eyes + vomiting (2026-10-10, Dr. Amos's review) -----------------------

CHOLERA_HIGH_BURDEN = OutbreakSignal(
    disease="Cholera", state="Lagos", basis="baseline", tier="high_burden"
)


def test_sunken_eyes_alone_does_not_fire_the_dehydration_rule() -> None:
    # The exact reported live case, as literally worded: only one of the four severe-
    # dehydration signs (sunken eyes). The two-of-four rule correctly does not fire.
    case = PatientCase(
        raw_text="watery stool since yesterday, vomiting, no fever, sunken eyes", state="Lagos"
    )
    result = assess(case)
    assert result.dehydration is None


def test_sunken_eyes_plus_vomiting_fires_refer_24h_with_no_outbreak_context() -> None:
    case = PatientCase(
        raw_text="watery stool since yesterday, vomiting, no fever, sunken eyes", state="Lagos"
    )
    result = assess(case)
    assert result.sunken_eyes_vomiting is not None
    assert not result.sunken_eyes_vomiting.cholera_context
    assert not result.sunken_eyes_vomiting.vulnerable
    assert result.floor == TriageLevel.REFER_24H


def test_sunken_eyes_plus_vomiting_escalates_in_a_cholera_high_burden_state() -> None:
    case = PatientCase(
        raw_text="watery stool since yesterday, vomiting, no fever, sunken eyes", state="Lagos"
    )
    result = assess(case, [CHOLERA_HIGH_BURDEN])
    assert result.sunken_eyes_vomiting is not None
    assert result.sunken_eyes_vomiting.cholera_context
    assert result.floor == TriageLevel.REFER_NOW


@pytest.mark.parametrize(
    "overrides",
    [{"pregnant": True, "age_years": 28}, {"age_years": 3}, {"age_years": 65}],
)
def test_sunken_eyes_plus_vomiting_escalates_for_vulnerable_patients(overrides: dict) -> None:
    case = PatientCase(
        raw_text="watery stool since yesterday, vomiting, no fever, sunken eyes",
        state="Lagos",
        **overrides,
    )
    result = assess(case)
    assert result.sunken_eyes_vomiting is not None
    assert result.sunken_eyes_vomiting.vulnerable
    assert result.floor == TriageLevel.REFER_NOW


def test_sunken_eyes_without_vomiting_does_not_fire_the_new_rule() -> None:
    case = PatientCase(raw_text="watery stool since yesterday, no fever, sunken eyes")
    assert assess(case).sunken_eyes_vomiting is None


def test_dehydration_rule_still_wins_over_sunken_eyes_vomiting_when_both_apply() -> None:
    # Two-of-four already fires (sunken eyes + slow skin pinch): the sunken-eyes-plus-vomiting
    # rule should not also add a separate, redundant differential/floor.
    case = PatientCase(
        raw_text="vomiting, no fever, sunken eyes, skin pinch goes back slowly", state="Lagos"
    )
    result = assess(case)
    assert result.dehydration is not None
    assert result.sunken_eyes_vomiting is None
    assert result.floor == TriageLevel.REFER_NOW


# --- moderate dehydration (2026-10-10, Dr. Amos's review) -------------------------


def test_two_moderate_signs_fire_refer_24h() -> None:
    case = PatientCase(
        raw_text="many watery stools since yesterday, restless, thirsty and drinking eagerly"
    )
    result = assess(case)
    assert result.moderate_dehydration is not None
    assert result.floor == TriageLevel.REFER_24H


def test_one_moderate_sign_does_not_fire() -> None:
    case = PatientCase(raw_text="many watery stools since yesterday, restless")
    assert assess(case).moderate_dehydration is None


def test_moderate_dehydration_does_not_fire_when_severe_already_does() -> None:
    case = PatientCase(
        raw_text="restless, thirsty, sunken eyes, skin pinch goes back slowly, unable to drink"
    )
    result = assess(case)
    assert result.dehydration is not None
    assert result.moderate_dehydration is None
    assert result.floor == TriageLevel.REFER_NOW


# --- oliguria / no urine output (2026-10-10, Dr. Amos's review) -------------------


def test_no_urine_output_is_an_independent_refer_now_sign() -> None:
    case = PatientCase(raw_text="watery stool since yesterday, passing no urine")
    result = assess(case)
    assert result.oliguria is not None
    assert result.floor == TriageLevel.REFER_NOW


def test_no_urine_output_alone_no_longer_counts_toward_the_two_of_four() -> None:
    # Sunken eyes + no urine output used to satisfy the old five-sign count; "no urine output"
    # is now independent (see oliguria_assessment) and must not also feed the dehydration rule.
    case = PatientCase(raw_text="sunken eyes, passing no urine, no fever")
    result = assess(case)
    assert result.dehydration is None
    assert result.oliguria is not None
    assert result.floor == TriageLevel.REFER_NOW  # from oliguria alone, not from dehydration
