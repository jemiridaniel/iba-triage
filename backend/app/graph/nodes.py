"""Pipeline nodes. Each takes (state, deps) and returns a partial state update.

LLM nodes catch LLMError and record it; rules_post then fails safe to "Refer now".
SpendLimitError is never caught: a budget stop must not look like a triage result.
"""

from __future__ import annotations

import re
from typing import Any

from backend.app.graph.prompts import (
    compose_messages,
    follow_up,
    full_text,
    intake_messages,
    reason_messages,
)
from backend.app.graph.state import (
    ComposeOutput,
    Deps,
    PostOutput,
    ReasonOutput,
    RuleSnapshot,
    TriageState,
)
from backend.app.llm.client import LLMError
from backend.app.llm.router import call_json
from backend.app.rules.danger_signs import LASSA_IPC_REMINDER, apply_floor, assess
from backend.app.rules.dosing import doses_for, strip_doses
from backend.app.schemas import (
    TRIAGE_LABELS,
    ActionItem,
    Citation,
    DangerSignHit,
    DifferentialItem,
    PatientCase,
    TriageLevel,
)

CRITICAL_FIELDS = ("age_years", "fever_days", "rdt_result")
MAX_QUESTIONS = 2
INCOMPLETE_RATIONALE = "Refer: Iba could not complete the assessment."
_PIDGIN_MARKERS = re.compile(r"\b(pikin|dey|don|wetin|abeg|dem|wahala|na im|e no)\b", re.I)


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def case_with_text(state: TriageState) -> PatientCase:
    """The case with the worker's full text attached (raw_text is never serialised)."""
    base = state.case or PatientCase(state=state.request.state, lga=state.request.lga)
    return base.model_copy(update={"raw_text": full_text(state.request)})


# --- intake -----------------------------------------------------------------


def intake(state: TriageState, deps: Deps) -> dict[str, Any]:
    req = state.request
    text = full_text(req)
    try:
        case, _ = call_json(
            deps.client, "intake", intake_messages(req), PatientCase, state.eval_config
        )
    except LLMError as exc:
        language = "pcm" if _PIDGIN_MARKERS.search(text) else "en"
        fallback = PatientCase(state=req.state, lga=req.lga, language=language, raw_text=text)
        return {"case": fallback, "intake_error": _error(exc), "questions": []}

    missing = [f for f in CRITICAL_FIELDS if getattr(case, f) is None]
    case = case.model_copy(
        update={
            "state": req.state or case.state,
            "lga": req.lga or case.lga,
            "missing_fields": missing,
            "raw_text": text,
        }
    )
    ask = not (req.answers or req.skip_questions)
    questions = [follow_up(f, case.language) for f in missing[:MAX_QUESTIONS]] if ask else []
    return {"case": case, "questions": questions}


# --- rules_pre --------------------------------------------------------------


def rules_pre(state: TriageState, deps: Deps) -> dict[str, Any]:
    return {"pre": RuleSnapshot.of(assess(case_with_text(state), []))}


def after_rules_pre(state: TriageState) -> str:
    """Where to go next: stop for questions, skip to rules_post, or continue."""
    if state.intake_error:
        return "rules_post"
    if state.questions and state.pre is not None and state.pre.floor is None:
        return "needs_info"  # danger signs never wait for answers
    return "retrieve"


# --- retrieve ---------------------------------------------------------------


def build_query(case: PatientCase, pre: RuleSnapshot | None) -> str:
    parts = ["fever"]
    if case.fever_days is not None:
        parts.append(f"fever for {case.fever_days:g} days")
    if case.age_years is not None:
        parts.append("child" if case.age_years < 5 else "adult")
    if case.rdt_result:
        parts.append(f"malaria RDT {case.rdt_result.replace('_', ' ')}")
    parts += case.symptoms
    if pre is not None:
        parts += [h.label for h in pre.danger_signs]
    if pre is not None and pre.danger_signs:
        parts.append("danger signs urgent referral severe malaria")
    if case.rdt_result == "negative" or (case.fever_days or 0) >= 3:
        parts.append("RDT negative fever not responding to antimalarials Lassa fever typhoid")
    return ", ".join(parts)


def retrieve(state: TriageState, deps: Deps) -> dict[str, Any]:
    if deps.store is None or deps.embed is None or len(deps.store) == 0:
        return {"retrieval_note": "Guideline index not available; no guideline excerpts used."}
    query = build_query(case_with_text(state), state.pre)
    try:
        hits = deps.store.search_text(query, deps.embed, k=deps.settings.retrieve_top_k)
    except LLMError as exc:
        return {"retrieval_note": f"Guideline retrieval failed ({type(exc).__name__})."}
    return {"hits": hits}


# --- outbreak ---------------------------------------------------------------


def outbreak(state: TriageState, deps: Deps) -> dict[str, Any]:
    region = state.request.state or (state.case.state if state.case else None)
    deps.outbreak.eval_config = state.eval_config
    return {"outbreak": deps.outbreak.check(region), "_note": deps.outbreak.last_note}


# --- reason -----------------------------------------------------------------


def reason(state: TriageState, deps: Deps) -> dict[str, Any]:
    assert state.pre is not None
    messages = reason_messages(case_with_text(state), state.pre, state.hits, state.outbreak)
    try:
        out, _ = call_json(deps.client, "reason", messages, ReasonOutput, state.eval_config)
    except LLMError as exc:
        return {"reason_error": _error(exc)}
    return {"reason": out}


# --- rules_post -------------------------------------------------------------


def _strip(text: str, counter: list[int]) -> str:
    cleaned, n = strip_doses(text)
    counter[0] += n
    return cleaned


def rules_post(state: TriageState, deps: Deps) -> dict[str, Any]:
    case = case_with_text(state)
    ctx = state.outbreak
    signals = ctx.signals if ctx is not None and ctx.status == "ok" else []
    out = state.reason
    extra = (
        [DangerSignHit(code=c, source="llm_reason") for c in out.danger_signs_found] if out else []
    )
    rules = assess(case, signals, extra_hits=extra)
    warnings: list[str] = []

    # 1. Triage level: the rule floor always wins; any failure refers.
    if out is None:
        status = "incomplete"
        level = TriageLevel.REFER_NOW
        rationale = INCOMPLETE_RATIONALE
        differential: list[DifferentialItem] = []
        actions: list[ActionItem] = []
        warnings.append(f"Assessment incomplete ({state.intake_error or state.reason_error}).")
    else:
        status = "complete"
        level = apply_floor(out.triage_level, rules.floor)
        rationale = out.triage_rationale
        if level != out.triage_level:
            rationale = (
                f"Raised from {TRIAGE_LABELS[out.triage_level]} to {TRIAGE_LABELS[level]} by "
                f"safety rules ({'; '.join(rules.reasons)}). {rationale}"
            )
        differential = [DifferentialItem(**d.model_dump()) for d in out.differential]
        actions = [ActionItem(**a.model_dump()) for a in out.actions]

    # 2. Rule-driven content the model cannot omit.
    if rules.danger_signs:
        labels = ", ".join(h.label for h in rules.danger_signs)
        actions.insert(
            0, ActionItem(text=f"Refer now: danger signs present ({labels}).", source="rule")
        )
    if rules.lassa_signal is not None:
        sig = rules.lassa_signal
        if not any("lassa" in d.condition.lower() for d in differential):
            differential.insert(
                0,
                DifferentialItem(
                    condition="Lassa fever",
                    likelihood="high",
                    reasons=rules.reasons,
                    check_next=["Notify the LGA disease surveillance officer for testing"],
                    citations=[sig.url] if sig.url else [],
                    source="rule",
                ),
            )
        actions.insert(
            0,
            ActionItem(
                text=LASSA_IPC_REMINDER, citations=[sig.url] if sig.url else [], source="rule"
            ),
        )

    # 3. Strip any dose text the model produced.
    stripped = [0]
    rationale = _strip(rationale, stripped)
    for d in differential:
        d.reasons = [_strip(r, stripped) for r in d.reasons]
        d.check_next = [_strip(c, stripped) for c in d.check_next]
    for a in actions:
        a.text = _strip(a.text, stripped)
    if stripped[0]:
        warnings.append(
            f"Removed {stripped[0]} dose mention(s) from model output; use the dose table."
        )

    # 4. Keep only citations that resolve to a stored chunk or a current outbreak signal.
    resolved: dict[str, Citation] = {}
    dropped = 0

    def resolve(refs: list[str]) -> list[str]:
        nonlocal dropped
        kept = []
        for ref in refs:
            cite = _resolve_citation(ref, deps, signals)
            if cite is None:
                dropped += 1
                continue
            resolved.setdefault(ref, cite)
            kept.append(ref)
        return kept

    for d in differential:
        d.citations = resolve(d.citations)
    for a in actions:
        a.citations = resolve(a.citations)
    if dropped:
        warnings.append(
            f"Dropped {dropped} citation(s) that did not match a guideline passage "
            "or outbreak source."
        )

    # 5. Doses from the table only.
    doses, dose_notes = doses_for(case, level)
    warnings += dose_notes

    # 6. Never hide missing context.
    if ctx is None or ctx.status != "ok":
        warnings.append(ctx.message if ctx else "Outbreak data unavailable.")
    elif ctx.source == "mock":
        warnings.append("Outbreak data is MOCK test data, not live surveillance.")
    if state.retrieval_note:
        warnings.append(state.retrieval_note)

    post = PostOutput(
        status=status,
        triage_level=level,
        triage_rationale=rationale,
        danger_signs=rules.danger_signs,
        lassa_suspected=rules.lassa_suspected,
        differential=differential,
        actions=actions,
        doses=doses,
        citations=list(resolved.values()),
        warnings=warnings,
    )
    floor = rules.floor.value if rules.floor else None
    note = f"floor={floor}, dropped_citations={dropped}, stripped_doses={stripped[0]}"
    return {"post": post, "_note": note}


def _resolve_citation(ref: str, deps: Deps, signals: list) -> Citation | None:
    if deps.store is not None and (chunk := deps.store.get(ref)) is not None:
        return Citation(
            kind="guideline",
            ref=ref,
            title=chunk.title,
            doc_id=chunk.doc_id,
            section=chunk.section,
            page=chunk.page,
        )
    for sig in signals:
        if sig.url and ref == sig.url:
            return Citation(
                kind="outbreak",
                ref=ref,
                title=f"{sig.disease}: {sig.state} ({sig.status})",
                url=sig.url,
                source_date=sig.report_date,
            )
    return None


# --- compose ----------------------------------------------------------------

_DISCLAIMER = {
    "en": "Decision support only, not a diagnosis. A qualified health worker must confirm.",
    "pcm": "Iba na helper only, e no be diagnosis. Qualified health worker must confirm am.",
}


def fallback_compose(case: PatientCase, post: PostOutput) -> ComposeOutput:
    label = TRIAGE_LABELS[post.triage_level]
    signs = ", ".join(h.label for h in post.danger_signs) or "none detected"
    conditions = ", ".join(d.condition for d in post.differential) or "not assessed"
    summary = f"{label}. {post.triage_rationale}"
    note = "\n".join(
        [
            f"Triage: {label}",
            f"Age: {case.age_years if case.age_years is not None else 'unknown'} years; "
            f"fever {case.fever_days if case.fever_days is not None else '?'} days; "
            f"RDT {case.rdt_result or 'unknown'}",
            f"Danger signs: {signs}",
            f"Consider: {conditions}",
            *[f"- {a.text}" for a in post.actions],
        ]
    )
    return ComposeOutput(summary=summary, referral_note=note)


def compose(state: TriageState, deps: Deps) -> dict[str, Any]:
    assert state.post is not None
    case = case_with_text(state)
    post = state.post
    error = None
    try:
        out, _ = call_json(
            deps.client, "compose", compose_messages(case, post), ComposeOutput, state.eval_config
        )
    except LLMError as exc:
        out, error = fallback_compose(case, post), _error(exc)

    summary, _ = strip_doses(out.summary)
    note, _ = strip_doses(out.referral_note)
    lines = [note.rstrip()]
    for d in post.doses:
        flag = "" if d.verified else " (UNVERIFIED table)"
        lines.append(f"Dose from table{flag}: {d.drug}, {d.weight_band}: {d.regimen}.")
    lines.append(_DISCLAIMER["pcm" if case.language == "pcm" else "en"])
    return {
        "compose": ComposeOutput(summary=summary, referral_note="\n\n".join(lines)),
        "compose_error": error,
    }
