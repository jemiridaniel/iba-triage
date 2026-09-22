"""Individual pipeline nodes. Scripted Token Factory; no network."""

from pathlib import Path

from backend.app.graph import nodes
from backend.app.graph.prompts import full_text, reason_messages
from backend.app.graph.state import ReasonOutput, RuleSnapshot, TriageRequest, TriageState
from backend.app.schemas import (
    DangerSignCode,
    FollowUpAnswer,
    OutbreakContext,
    PatientCase,
    TriageLevel,
)
from tests.pipeline_fakes import COMPOSE_EN, make_deps


def state_for(text: str, **kw) -> TriageState:
    return TriageState(request=TriageRequest(text=text, **kw))


# --- intake -------------------------------------------------------------------


def test_intake_asks_up_to_two_questions_in_input_language(tmp_path: Path) -> None:
    deps, _ = make_deps(tmp_path, {"intake": {"symptoms": ["fever"], "language": "pcm"}})
    update = nodes.intake(state_for("Pikin get hot body"), deps)
    assert [q.id for q in update["questions"]] == ["fever_days", "rdt_result"]
    assert update["questions"][0].text == "How many days the body don dey hot?"
    assert update["case"].missing_fields == ["age_years", "fever_days", "rdt_result"]


def test_intake_does_not_ask_again_after_answers(tmp_path: Path) -> None:
    deps, llm = make_deps(tmp_path, {"intake": {"language": "en"}})
    answers = [FollowUpAnswer(id="age_years", answer="30")]
    update = nodes.intake(state_for("fever", answers=answers), deps)
    assert update["questions"] == []
    assert "How old is the patient? 30" in llm.messages_for("intake")[1]["content"]


def test_intake_request_state_overrides_model(tmp_path: Path) -> None:
    deps, _ = make_deps(tmp_path, {"intake": {"state": "Lagos", "language": "en"}})
    update = nodes.intake(state_for("fever", state="Ondo", lga="Owo"), deps)
    assert (update["case"].state, update["case"].lga) == ("Ondo", "Owo")


def test_intake_prompt_uses_fast_model_reasoning_off(tmp_path: Path) -> None:
    deps, llm = make_deps(tmp_path, {"intake": {"language": "en"}})
    nodes.intake(state_for("fever"), deps)
    ((_, kw),) = llm.calls
    assert kw["model"] == "fast"
    assert kw["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert kw["max_tokens"] == deps.settings.max_tokens_reasoning_off


def test_intake_failure_guesses_language_and_records_error(tmp_path: Path) -> None:
    deps, _ = make_deps(tmp_path, {"intake": ["x", "y"]})
    update = nodes.intake(state_for("Pikin dey hot"), deps)
    assert update["case"].language == "pcm"
    assert "LLMParseError" in update["intake_error"]


# --- rules_pre / routing --------------------------------------------------------


def test_rules_pre_uses_worker_text_and_answers(tmp_path: Path) -> None:
    deps, _ = make_deps(tmp_path, {})
    s = state_for("fever", answers=[FollowUpAnswer(id="fever_days", answer="3, and he convulsed")])
    s.case = PatientCase()
    pre = nodes.rules_pre(s, deps)["pre"]
    assert pre.floor == TriageLevel.REFER_NOW
    assert pre.danger_signs[0].code == DangerSignCode.CONVULSIONS


def test_after_rules_pre_routing() -> None:
    s = state_for("fever")
    s.pre = RuleSnapshot(floor=None)
    assert nodes.after_rules_pre(s) == "outbreak"
    s.questions = [nodes.follow_up("age_years", "en")]
    assert nodes.after_rules_pre(s) == "needs_info"
    s.pre = RuleSnapshot(floor=TriageLevel.REFER_NOW)
    assert nodes.after_rules_pre(s) == "outbreak"
    s.intake_error = "boom"
    assert nodes.after_rules_pre(s) == "rules_post"


# --- retrieve -------------------------------------------------------------------


def test_retrieve_without_index_is_flagged(tmp_path: Path) -> None:
    deps, _ = make_deps(tmp_path, {}, store=False)
    s = state_for("fever")
    s.case = PatientCase()
    assert "not available" in nodes.retrieve(s, deps)["retrieval_note"]


def test_retrieve_query_pulls_lassa_passages_for_rdt_negative_fever(tmp_path: Path) -> None:
    deps, _ = make_deps(tmp_path, {})
    s = state_for("fever")
    s.case = PatientCase(age_years=35, fever_days=5, rdt_result="negative")
    s.pre = RuleSnapshot(floor=None)
    ids = [h.chunk.id for h in nodes.retrieve(s, deps)["hits"]]
    assert any(i.startswith("fake-lassa") for i in ids)
    assert len(ids) == deps.settings.retrieve_top_k


def test_build_query() -> None:
    q = nodes.build_query(PatientCase(age_years=2, fever_days=3, rdt_result="positive"), None)
    assert "child" in q and "malaria RDT positive" in q and "fever for 3 days" in q


# --- reason prompt ----------------------------------------------------------------


def test_reason_prompt_forbids_doses_and_lists_sources() -> None:
    ctx = OutbreakContext(status="unavailable", source="none", message="x")
    system, user = reason_messages(PatientCase(age_years=3), RuleSnapshot(floor=None), [], ctx)
    assert "NEVER write drug doses" in system["content"]
    assert "VERBATIM span of 8 to 40" in system["content"]
    assert "Live outbreak data unavailable" in user["content"]
    assert "raw_text" not in user["content"]


def test_reason_prompt_separates_this_state_from_elsewhere() -> None:
    """FEEDBACK T3: NCDC sitreps are national, so one search names several states."""
    from datetime import date

    from backend.app.schemas import OutbreakSignal

    def sig(state: str, **kw) -> OutbreakSignal:
        fields = {
            "disease": "Lassa fever",
            "state": state,
            "status": "active",
            "report_date": date(2026, 9, 18),
            "url": f"https://ncdc.gov.ng/{state}.pdf",
            **kw,
        }
        return OutbreakSignal(**fields)

    ctx = OutbreakContext(
        status="ok",
        source="tavily",
        message="x",
        signals=[sig("Ondo"), sig("Edo"), sig("Benue", recency="older")],
    )
    case = PatientCase(age_years=30, state="Ondo")
    _, user = reason_messages(case, RuleSnapshot(floor=None), [], ctx)
    text = user["content"]

    here, elsewhere = text.split("ELSEWHERE IN NIGERIA")
    assert "IN THIS PATIENT'S STATE (Ondo)" in here
    assert "Ondo" in here.split("IN THIS PATIENT'S STATE")[1]
    assert "Edo" not in here.split("IN THIS PATIENT'S STATE")[1]
    assert "Edo" in elsewhere and "Benue" in elsewhere
    assert "must NOT raise this patient's triage level" in elsewhere
    # T1: staleness is spelled out so the model can't read an old report as current.
    assert "older report, 2026-09-18 - do NOT treat as current" in elsewhere


def test_reason_prompt_flags_an_undated_signal() -> None:
    from backend.app.schemas import OutbreakSignal

    ctx = OutbreakContext(
        status="ok",
        source="tavily",
        message="x",
        signals=[
            OutbreakSignal(
                disease="Meningitis",
                state="Ondo",
                status="active",
                url="https://ncdc.gov.ng/x",
                recency="unknown",
            )
        ],
    )
    _, user = reason_messages(
        PatientCase(age_years=30, state="Ondo"), RuleSnapshot(floor=None), [], ctx
    )
    assert "report date unknown - do NOT treat as current" in user["content"]


def test_reason_prompt_says_none_when_nothing_is_reported_in_this_state() -> None:
    from datetime import date

    from backend.app.schemas import OutbreakSignal

    ctx = OutbreakContext(
        status="ok",
        source="tavily",
        message="x",
        signals=[
            OutbreakSignal(
                disease="Lassa fever",
                state="Ondo",
                status="active",
                report_date=date(2026, 9, 18),
                url="https://ncdc.gov.ng/x",
            )
        ],
    )
    _, user = reason_messages(
        PatientCase(age_years=30, state="Lagos"), RuleSnapshot(floor=None), [], ctx
    )
    assert "IN THIS PATIENT'S STATE (Lagos):\n- none reported" in user["content"]


def test_reason_output_caps_differential_and_normalises_likelihood() -> None:
    items = [{"condition": f"c{i}", "likelihood": "medium"} for i in range(6)]
    out = ReasonOutput(triage_level="refer_24h", triage_rationale="r", differential=items)
    assert len(out.differential) == 4
    assert out.differential[0].likelihood == "moderate"


# --- compose ----------------------------------------------------------------------


def test_compose_strips_doses_then_appends_table_doses_and_disclaimer(tmp_path: Path) -> None:
    from backend.app.graph.state import PostOutput
    from backend.app.rules.dosing import doses_for

    reply = {"summary": "Give 4 tablets now.", "referral_note": "Note: paracetamol 500 mg."}
    deps, _ = make_deps(tmp_path, {"compose": reply})
    case = PatientCase(rdt_result="positive", weight_kg=60, language="en")
    doses, _ = doses_for(case, TriageLevel.TREAT_MONITOR)
    s = state_for("fever")
    s.case = case
    s.post = PostOutput(
        status="complete", triage_level=TriageLevel.TREAT_MONITOR, triage_rationale="r", doses=doses
    )
    out = nodes.compose(s, deps)["compose"]
    assert "4 tablets now" not in out.summary
    assert "500 mg" not in out.referral_note
    assert "Dose from table (UNVERIFIED table): Artemether-lumefantrine" in out.referral_note
    assert out.referral_note.endswith("A qualified health worker must confirm.")


def test_full_text_includes_answers() -> None:
    req = TriageRequest(text="fever", answers=[FollowUpAnswer(id="rdt_result", answer="negative")])
    assert full_text(req).endswith(
        "What was the malaria RDT result: positive, negative, or not done? negative"
    )


def test_compose_reply_is_used(tmp_path: Path) -> None:
    from backend.app.graph.state import PostOutput

    deps, _ = make_deps(tmp_path, {"compose": COMPOSE_EN})
    s = state_for("fever")
    s.case = PatientCase()
    s.post = PostOutput(status="complete", triage_level=TriageLevel.REFER_24H, triage_rationale="r")
    out = nodes.compose(s, deps)
    assert out["compose"].summary == "Summary text."
    assert out["compose_error"] is None


def test_compose_prompt_forbids_claiming_actions_were_done() -> None:
    # Live run 2026-09-22: Super wrote "Actions taken: isolated, ... notified" for advice.
    from backend.app.graph.prompts import compose_messages
    from backend.app.graph.state import PostOutput

    post = PostOutput(status="complete", triage_level=TriageLevel.REFER_NOW, triage_rationale="r")
    system = compose_messages(PatientCase(), post)[0]["content"]
    assert "Never state that an action was done" in system


def test_referral_note_has_header_and_referral_blanks(tmp_path: Path) -> None:
    from backend.app.graph.state import PostOutput

    deps, _ = make_deps(tmp_path, {"compose": COMPOSE_EN})
    s = state_for("fever")
    s.case = PatientCase(state="Ondo")
    s.post = PostOutput(status="complete", triage_level=TriageLevel.REFER_NOW, triage_rationale="r")
    note = nodes.compose(s, deps)["compose"].referral_note
    assert note.startswith("IBÀ TRIAGE NOTE · ") and "REFER NOW · Ondo" in note.splitlines()[0]
    assert "Referred to: ____" in note


def test_retrieve_reuses_pre_embedded_queries(tmp_path: Path) -> None:
    calls = []
    deps, _ = make_deps(tmp_path, {})
    inner = deps.embed
    deps.embed = lambda texts: calls.append(list(texts)) or inner(texts)
    s = state_for("fever")
    s.case = PatientCase(age_years=35, fever_days=5, rdt_result="negative")
    s.pre = RuleSnapshot(floor=None)
    s = s.model_copy(update=nodes.embed_queries(s, deps))
    assert len(calls) == 1 and s.query_texts
    out = nodes.retrieve(s, deps)
    assert len(calls) == 1  # nothing new to embed: outbreak added no condition
    assert out["hits"] and "(0 embedded here)" in out["_note"]


def test_prompt_passages_cap_and_window() -> None:
    from backend.app.rag.queries import PROMPT_TOP, WINDOW_CHARS, best_window, select_passages
    from tests.pipeline_fakes import fake_store

    store = fake_store()
    hits = store.search([1.0] * 256, k=8)  # every chunk scores > 0
    chosen = select_passages(hits, ["lassa"])
    top = sorted(hits, key=lambda h: -h.score)[:PROMPT_TOP]
    assert chosen[:PROMPT_TOP] == top
    assert all(h.chunk.doc_id == "fake-lassa" for h in chosen[PROMPT_TOP:])

    text = (
        ("filler words here " * 150) + "isolate the patient with gloves " + ("more filler " * 150)
    )
    window = best_window(text, {"isolate", "gloves"})
    assert "isolate the patient with gloves" in window
    assert len(window) <= WINDOW_CHARS + 4 and window.startswith("… ")
