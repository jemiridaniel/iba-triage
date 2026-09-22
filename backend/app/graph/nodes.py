"""Pipeline nodes. Each takes (state, deps) and returns a partial state update.

LLM nodes catch LLMError and record it; rules_post then fails safe to "Refer now".
SpendLimitError is never caught: a budget stop must not look like a triage result.
"""

from __future__ import annotations

import re
from datetime import date
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
from backend.app.llm.client import LLMError, LLMTruncatedError
from backend.app.llm.router import THINKING_OFF_KWARGS, call_json, route
from backend.app.rag.queries import (
    plan_queries,
    prompt_passages,
    search_plans,
    suspected_conditions,
)
from backend.app.rules.danger_signs import (
    LASSA_CASE_DEF_ANCHOR,
    LASSA_IPC_REMINDER,
    LASSA_TRIAGE_ANCHOR,
    apply_floor,
    assess,
    no_treatment_response,
)
from backend.app.rules.dosing import doses_for, strip_doses
from backend.app.rules.followup import merge_advice, treat_monitor_advice
from backend.app.rules.grounding import patient_fact_supported, verify_quote
from backend.app.schemas import (
    TRIAGE_LABELS,
    ActionItem,
    Citation,
    DangerSignHit,
    DifferentialItem,
    Evidence,
    GroundingSummary,
    OutbreakContext,
    PatientCase,
    Reason,
    RetrievalQuery,
    TriageLevel,
    most_urgent,
)

CRITICAL_FIELDS = ("age_years", "fever_days", "rdt_result")
# Order in which missing information is asked about (at most MAX_QUESTIONS per round).
QUESTION_ORDER = ("fever_days", "rdt_result", "treatment_response", "age_years")
MAX_QUESTIONS = 2
INCOMPLETE_RATIONALE = "Refer: Ibà could not complete the assessment."
ASSESSMENT_UNAVAILABLE = "Detailed assessment unavailable — safety rules applied. Refer."
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

    case = case.model_copy(
        update={"state": req.state or case.state, "lga": req.lga or case.lga, "raw_text": text}
    )
    missing = missing_information(case)
    case = case.model_copy(update={"missing_fields": missing})
    ask = not (req.answers or req.skip_questions)
    ordered = [f for f in QUESTION_ORDER if f in missing]
    questions = [follow_up(f, case.language) for f in ordered[:MAX_QUESTIONS]] if ask else []
    return {"case": case, "questions": questions}


def missing_information(case: PatientCase) -> list[str]:
    """Critical fields, plus treatment response once fever has lasted 3+ days (NCDC Lassa
    suspected-case definition, index of suspicion (a))."""
    missing = [f for f in CRITICAL_FIELDS if getattr(case, f) is None]
    unknown_treatment = (
        case.antimalarial_taken is None
        and case.antibiotic_taken is None
        and not no_treatment_response(case)
    )
    if (case.fever_days or 0) >= 3 and unknown_treatment:
        missing.append("treatment_response")
    return missing


# --- rules_pre --------------------------------------------------------------


def rules_pre(state: TriageState, deps: Deps) -> dict[str, Any]:
    """Danger signs + the Lassa case definition against the static endemicity baseline.

    Needs no network, so a rule-based Refer now streams within ~1.5 s. Live outbreak signals
    can only escalate this later (rules_post), never lower it.
    """
    region = state.request.state or (state.case.state if state.case else None)
    baseline = deps.outbreak.endemicity.for_state(
        region, deps.outbreak.today(), deps.outbreak.resolve
    )
    return {"pre": RuleSnapshot.of(assess(case_with_text(state), baseline))}


def after_rules_pre(state: TriageState) -> str:
    """Where to go next: stop for questions, skip to rules_post, or continue."""
    if state.intake_error:
        return "rules_post"
    if state.questions and state.pre is not None and state.pre.floor is None:
        return "needs_info"  # danger signs never wait for answers
    return "outbreak"


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


def _conditions(state: TriageState, outbreak_ctx: OutbreakContext | None) -> list[str]:
    case = case_with_text(state)
    signals = outbreak_ctx.all_signals if outbreak_ctx else []
    lassa_rule = assess(case, signals).lassa is not None
    danger = bool(state.pre and state.pre.danger_signs)
    return suspected_conditions(case, danger, outbreak_ctx, lassa_rule)


def embed_queries(state: TriageState, deps: Deps) -> dict[str, Any]:
    """Plan and embed retrieval queries in parallel with the outbreak step.

    Uses the static endemicity baseline (known without any search); the retrieve step embeds
    only queries that live outbreak data adds afterwards.
    """
    if deps.store is None or deps.embed is None or len(deps.store) == 0:
        return {}
    region = state.request.state or (state.case.state if state.case else None)
    baseline = deps.outbreak.endemicity.for_state(
        region, deps.outbreak.today(), deps.outbreak.resolve
    )
    ctx = OutbreakContext(status="unavailable", source="none", message="", baseline=baseline)
    plans = plan_queries(case_with_text(state), _conditions(state, ctx))
    try:
        vectors = deps.embed([p.text for p in plans])
    except LLMError as exc:  # retrieve will try again
        return {"_note": f"pre-embedding failed ({type(exc).__name__}); retrieve will embed"}
    return {
        "query_texts": [p.text for p in plans],
        "query_vectors": vectors,
        "_note": f"{len(plans)} queries embedded while checking outbreaks",
    }


def retrieve(state: TriageState, deps: Deps) -> dict[str, Any]:
    """One query per suspected condition and purpose, with a doc prior (rag/queries.py).

    Joins the outbreak and embed_queries steps: reuses pre-computed query vectors and embeds
    only the queries that live outbreak context added.
    """
    if deps.store is None or deps.embed is None or len(deps.store) == 0:
        return {"retrieval_note": "Guideline index not available; no guideline excerpts used."}
    conditions = _conditions(state, state.outbreak)
    plans = plan_queries(case_with_text(state), conditions)
    known = dict(zip(state.query_texts, state.query_vectors, strict=False))
    missing = [p.text for p in plans if p.text not in known]
    try:
        if missing:
            known.update(zip(missing, deps.embed(missing), strict=True))
    except LLMError as exc:
        return {
            "retrieval_note": "Guideline passages are unavailable right now.",
            "_note": f"retrieval failed: {_error(exc)}",
        }
    hits, logs = search_plans(
        deps.store, plans, [known[p.text] for p in plans], k_total=deps.settings.retrieve_top_k
    )
    return {
        "hits": hits,
        "conditions": conditions,
        "retrieval": [RetrievalQuery(**vars(lg)) for lg in logs],
        "_note": (
            f"conditions={conditions}, queries={len(plans)} ({len(missing)} embedded here), "
            f"passages={len(hits)}"
        ),
    }


# --- outbreak ---------------------------------------------------------------


def outbreak(state: TriageState, deps: Deps) -> dict[str, Any]:
    region = state.request.state or (state.case.state if state.case else None)
    deps.outbreak.eval_config = state.eval_config
    return {"outbreak": deps.outbreak.check(region), "_note": deps.outbreak.last_note}


# --- reason -----------------------------------------------------------------


def reason(state: TriageState, deps: Deps) -> dict[str, Any]:
    assert state.pre is not None
    case = case_with_text(state)
    plans = plan_queries(case, state.conditions)
    passages = prompt_passages(state.hits, state.conditions, plans, case.symptoms)
    messages = reason_messages(case, state.pre, passages, state.outbreak)
    try:
        out, _ = call_json(deps.client, "reason", messages, ReasonOutput, state.eval_config)
    except LLMTruncatedError as exc:
        # Reasoning ran past max_tokens. Token Factory offers no reasoning-budget control
        # (scripts/probe_reasoning_budget.py), so retry once with reasoning off. Quote
        # verification still applies to the result.
        cfg = route("reason", deps.settings, state.eval_config)
        try:
            out, _ = deps.client.chat_json(
                messages,
                ReasonOutput,
                model=cfg.model,
                step="reason:fallback",
                max_tokens=deps.settings.max_tokens_fallback,
                extra_body=THINKING_OFF_KWARGS,
            )
        except LLMError as retry_exc:
            return {"reason_error": f"{_error(exc)}; fallback: {_error(retry_exc)}"}
        return {
            "reason": out,
            "_reasoning": False,
            "_note": f"reasoning: off (fallback after truncation: {exc})",
        }
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
    signals = ctx.all_signals if ctx is not None else []  # live (if any) + baseline
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
        # Technical detail stays in the decision trace, not in the user's notes.
        warnings.append(ASSESSMENT_UNAVAILABLE)
    else:
        status = "complete"
        level = apply_floor(
            out.triage_level, most_urgent(rules.floor, state.pre.floor if state.pre else None)
        )
        rationale = out.triage_rationale
        if level != out.triage_level:
            rationale = (
                f"Raised from {TRIAGE_LABELS[out.triage_level]} to {TRIAGE_LABELS[level]} by "
                f"safety rules ({'; '.join(rules.reasons)}). {rationale}"
            )
        differential = [
            DifferentialItem(
                condition=d.condition,
                likelihood=d.likelihood,
                check_next=d.check_next,
                reasons=[
                    Reason(
                        text=r.text,
                        evidence=_ground(r.basis, r.chunk_id, r.evidence_quote, deps, r.text, case),
                    )
                    for r in d.reasons
                ],
            )
            for d in out.differential
        ]
        for d in differential:
            d.citations = list(
                dict.fromkeys(
                    r.evidence.chunk_id
                    for r in d.reasons
                    if r.evidence.status == "verified" and r.evidence.chunk_id
                )
            )
        actions = []
        for a in out.actions:
            ev = _ground("guideline", a.chunk_id, a.evidence_quote, deps)
            cites = [ev.chunk_id] if ev.status == "verified" and ev.chunk_id else []
            actions.append(ActionItem(text=a.text, citations=cites, evidence=ev))

    # 2. Rule-driven content the model cannot omit.
    if rules.danger_signs:
        labels = ", ".join(h.label for h in rules.danger_signs)
        actions.insert(
            0,
            ActionItem(
                text=f"Refer now: danger signs present ({labels}).",
                source="rule",
                evidence=Evidence(status="rule"),
            ),
        )
    if rules.lassa is not None:
        finding = rules.lassa
        sig = finding.signal
        place = [sig.url] if sig.url else ([sig.citation] if sig.citation else [])
        case_def = _anchors(deps, [LASSA_CASE_DEF_ANCHOR])
        if not any("lassa" in d.condition.lower() for d in differential):
            differential.insert(
                0,
                DifferentialItem(
                    condition="Lassa fever",
                    likelihood="high" if finding.basis == "live" else "moderate",
                    reasons=[
                        Reason(text=c, evidence=Evidence(status="rule")) for c in finding.criteria
                    ],
                    check_next=[
                        "Measure temperature (no antipyretic in the last 24 h)",
                        "Notify the LGA disease surveillance officer for Lassa testing",
                    ],
                    citations=[*case_def, *place],
                    source="rule",
                ),
            )
        actions.insert(
            0,
            ActionItem(
                text=LASSA_IPC_REMINDER,
                citations=[*_anchors(deps, [LASSA_TRIAGE_ANCHOR]), *place],
                source="rule",
                evidence=Evidence(status="rule"),
            ),
        )
    if level == TriageLevel.TREAT_MONITOR and status == "complete":
        for text, anchors in treat_monitor_advice(case):
            actions.append(
                ActionItem(
                    text=text,
                    citations=_anchors(deps, anchors),
                    source="rule",
                    evidence=Evidence(status="rule"),
                )
            )

    # 3. Strip any dose text the model produced.
    stripped = [0]
    rationale = _strip(rationale, stripped)
    for d in differential:
        for r in d.reasons:
            r.text = _strip(r.text, stripped)
            if r.evidence.quote:  # verified before stripping; displayed without doses
                r.evidence.quote, _ = strip_doses(r.evidence.quote)
        d.check_next = [_strip(c, stripped) for c in d.check_next]
    for a in actions:
        a.text = _strip(a.text, stripped)
        if a.evidence and a.evidence.quote:
            a.evidence.quote, _ = strip_doses(a.evidence.quote)
    if stripped[0]:
        warnings.append(
            f"Removed {stripped[0]} dose mention(s) from model output; use the dose table."
        )

    # 3b. Count grounding over every model claim, then merge advice that says the same thing
    #     (rule wording wins; merged model detail keeps its own grounding status).
    grounding = _grounding_summary(differential, actions)
    actions, merged = merge_advice(actions)

    # 4. Keep only citations that resolve to a stored chunk or a current outbreak signal.
    resolved: dict[str, Citation] = {}
    dropped = 0
    dropped_refs: list[str] = []

    def resolve(refs: list[str]) -> list[str]:
        nonlocal dropped
        kept = []
        for ref in refs:
            cite = _resolve_citation(ref, deps, signals)
            if cite is None:
                dropped += 1
                dropped_refs.append(ref[:80])
                continue
            resolved.setdefault(cite.ref, cite)
            kept.append(cite.ref)
        return list(dict.fromkeys(kept))

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
    if ctx is not None and ctx.baseline and not deps.outbreak.endemicity.all_verified:
        warnings.append(
            "Baseline endemicity list is provisional (not yet verified against NCDC sources)."
        )
    if state.retrieval_note:
        warnings.append(state.retrieval_note)

    if grounding.unsupported:
        warnings.append(
            f"{grounding.unsupported} of {grounding.claims} AI suggestion(s) have no verified "
            "guideline source; they are shown in grey."
        )

    post = PostOutput(
        status=status,
        grounding=grounding,
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
    lassa = rules.lassa.basis if rules.lassa else None
    note = (
        f"floor={floor}, lassa={lassa}, merged_advice={merged}, "
        f"verified={grounding.verified}/{grounding.claims}, "
        f"dropped_citations={dropped}, stripped_doses={stripped[0]}"
    )
    if dropped_refs:  # citation IDs/URLs only, never patient text
        note += f", dropped={dropped_refs[:5]}"
    return {"post": post, "_note": note}


def _ground(
    basis: str,
    chunk_id: str | None,
    quote: str | None,
    deps: Deps,
    text: str = "",
    case: PatientCase | None = None,
) -> Evidence:
    """Verify a model claim's evidence.

    A "patient" label is accepted only if the claim's key terms are in the worker's input or
    the structured case; otherwise it is relabelled a clinical claim needing a verified quote.
    """
    if basis == "patient":
        if case is not None and patient_fact_supported(text, case):
            return Evidence(status="patient")
        basis = "guideline"  # relabelled: must now be quote-verified
    chunk = deps.store.get(_normalise_ref(chunk_id)) if (chunk_id and deps.store) else None
    if chunk is None:
        return Evidence(status="unsupported")
    check = verify_quote(quote, chunk.text)
    if not check.verified:  # keep the attempted chunk for diagnostics; no citation shown
        return Evidence(status="unsupported", chunk_id=chunk.id, score=check.score)
    return Evidence(
        status="verified", chunk_id=chunk.id, quote=(quote or "").strip(), score=check.score
    )


def _grounding_summary(
    differential: list[DifferentialItem], actions: list[ActionItem]
) -> GroundingSummary:
    evidence = [r.evidence for d in differential if d.source == "llm" for r in d.reasons]
    evidence += [a.evidence for a in actions if a.source == "llm" and a.evidence]
    verified = sum(e.status == "verified" for e in evidence)
    unsupported = sum(e.status == "unsupported" for e in evidence)
    return GroundingSummary(
        claims=verified + unsupported,
        verified=verified,
        unsupported=unsupported,
        patient_facts=sum(e.status == "patient" for e in evidence),
    )


def _excerpt(text: str, limit: int = 320) -> str:
    flat = " ".join(
        text.replace("[FAKE TEST PASSAGE - placeholder, not guideline text]", "").split()
    )
    return flat if len(flat) <= limit else flat[:limit].rsplit(" ", 1)[0] + " …"


def _anchors(deps: Deps, anchors: list[tuple[str, str]]) -> list[str]:
    """Chunk IDs for guideline anchor phrases that are present in the index."""
    if deps.store is None:
        return []
    refs = []
    for doc_id, phrase in anchors:
        chunk = deps.store.find(doc_id, phrase)
        if chunk is not None:
            refs.append(chunk.id)
    return refs


_CHUNK_ID = re.compile(r"[a-z0-9][a-z0-9-]*:\d{4}")
_URL = re.compile(r"https?://[^\s\])>,;\"']+")


def _normalise_ref(ref: str) -> str:
    """Tolerate formatting around a valid ID: "[ncdc-lassa:0002] (p.8)" -> "ncdc-lassa:0002"."""
    ref = ref.strip()
    if (url := _URL.search(ref)) is not None:
        return url.group(0).rstrip(".")
    if (chunk_id := _CHUNK_ID.search(ref)) is not None:
        return chunk_id.group(0)
    return ref


def _resolve_citation(ref: str, deps: Deps, signals: list) -> Citation | None:
    ref = _normalise_ref(ref)
    if deps.store is not None and (chunk := deps.store.get(ref)) is not None:
        return Citation(
            kind="guideline",
            ref=ref,
            title=chunk.title,
            doc_id=chunk.doc_id,
            section=chunk.section,
            page=chunk.page,
            page_end=chunk.page_end,
            excerpt=_excerpt(chunk.text),
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
    "pcm": "Ibà na helper only, e no be diagnosis. Qualified health worker must confirm am.",
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
    label = TRIAGE_LABELS[post.triage_level]
    header = f"IBÀ TRIAGE NOTE · {date.today():%d %b %Y} · {label.upper()}"
    if case.state:
        header += f" · {case.state}"
    lines = [header, note.rstrip()]
    for d in post.doses:
        flag = "" if d.verified else " (UNVERIFIED table)"
        lines.append(f"Dose from table{flag}: {d.drug}, {d.weight_band}: {d.regimen}.")
    if post.triage_level != TriageLevel.TREAT_MONITOR:
        lines.append("Referred to: ____________________   By: ____________________")
    lines.append(_DISCLAIMER["pcm" if case.language == "pcm" else "en"])
    return {
        "compose": ComposeOutput(summary=summary, referral_note="\n\n".join(lines)),
        "compose_error": error,
    }
