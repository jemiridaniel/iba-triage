"""Truncation recovery, early rule-based banner, plain user-facing errors. No network."""

from pathlib import Path

from backend.app.graph import nodes
from backend.app.graph.pipeline import run_triage
from backend.app.graph.state import TriageRequest
from backend.app.llm.router import THINKING_OFF_KWARGS
from backend.app.schemas import TriageLevel
from backend.app.tools.endemicity import Endemicity
from tests.pipeline_fakes import COMPOSE_EN, REPO, TRUNCATED, make_deps
from tests.test_baseline_and_advice import REGRESSION_INTAKE, REGRESSION_TEXT
from tests.test_pipeline_e2e import MALARIA_INTAKE, MALARIA_REASON, MALARIA_TEXT

ENDEMICITY = Endemicity.load(REPO / "data" / "endemicity.yaml")


def test_truncated_reason_falls_back_to_reasoning_off(tmp_path: Path) -> None:
    deps, llm = make_deps(
        tmp_path,
        {"intake": MALARIA_INTAKE, "reason": [TRUNCATED, MALARIA_REASON], "compose": COMPOSE_EN},
        search=None,
    )
    r = run_triage(TriageRequest(text=MALARIA_TEXT), deps)
    assert r.status == "complete"
    assert r.triage_level == TriageLevel.TREAT_MONITOR
    step = next(s for s in r.decision_trace.steps if s.step == "reason")
    assert step.reasoning is False and "fallback after truncation" in step.note
    assert step.calls == 2  # the truncated call is still counted (and billed)
    reason_calls = [kw for t, kw in llm.calls if t == "reason"]
    assert "extra_body" not in reason_calls[0]  # reasoning on
    assert reason_calls[0]["max_tokens"] == deps.settings.max_tokens_reasoning_on == 16384
    assert reason_calls[1]["extra_body"] == THINKING_OFF_KWARGS
    assert reason_calls[1]["max_tokens"] == deps.settings.max_tokens_fallback


def test_fallback_failure_fails_safe_with_plain_message(tmp_path: Path) -> None:
    deps, _ = make_deps(
        tmp_path,
        {
            "intake": MALARIA_INTAKE,
            "reason": [TRUNCATED, "nope", "still nope"],
            "compose": COMPOSE_EN,
        },
        search=None,
    )
    r = run_triage(TriageRequest(text=MALARIA_TEXT), deps)
    assert r.status == "incomplete" and r.triage_level == TriageLevel.REFER_NOW
    assert nodes.ASSESSMENT_UNAVAILABLE in r.warnings
    assert not any("Error" in w for w in r.warnings)  # no exception names for the user
    step = next(s for s in r.decision_trace.steps if s.step == "reason")
    assert "LLMTruncatedError" in step.note and "LLMParseError" in step.note  # detail in trace


def test_lassa_rule_sets_floor_in_rules_pre_from_baseline(tmp_path: Path) -> None:
    deps, _ = make_deps(tmp_path, {"intake": REGRESSION_INTAKE}, search=None)
    deps.outbreak.endemicity = ENDEMICITY
    from backend.app.graph.state import TriageState

    s = TriageState(request=TriageRequest(text=REGRESSION_TEXT, state="Ondo"))
    s = s.model_copy(update=nodes.intake(s, deps))
    pre = nodes.rules_pre(s, deps)["pre"]
    assert pre.floor == TriageLevel.REFER_NOW  # before any network call
    assert pre.lassa_signal is not None and pre.lassa_signal.basis == "baseline"
    assert any("Lassa" in r for r in pre.reasons)


def test_final_level_never_below_rules_pre_floor(tmp_path: Path) -> None:
    green = {**MALARIA_REASON, "differential": MALARIA_REASON["differential"]}
    deps, _ = make_deps(
        tmp_path, {"intake": REGRESSION_INTAKE, "reason": green, "compose": COMPOSE_EN}, search=None
    )
    deps.outbreak.endemicity = ENDEMICITY
    r = run_triage(TriageRequest(text=REGRESSION_TEXT, state="Ondo"), deps)
    assert r.triage_level == TriageLevel.REFER_NOW  # model said treat_monitor


def test_failsafe_result_hides_exception_text() -> None:
    from backend.app.graph.pipeline import failsafe_result

    r = failsafe_result(
        TriageRequest(text="Pikin hot body, e dey convulse"), RuntimeError("db down")
    )
    assert r.warnings == [nodes.ASSESSMENT_UNAVAILABLE]
    assert "RuntimeError: db down" in r.decision_trace.steps[0].note


def test_demo_cases_are_shared_and_expected_levels_set() -> None:
    import json

    demos = json.loads((REPO / "frontend" / "src" / "demos.json").read_text())
    assert [d["expected"] for d in demos] == ["refer_now", "refer_now", "treat_monitor"]
    assert {d["state"] for d in demos} == {"Kano", "Ondo", "Lagos"}
