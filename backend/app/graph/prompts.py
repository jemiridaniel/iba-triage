"""Prompt builders for the LLM steps. Every system prompt starts with "Task: <step>"."""

import json

from backend.app.graph.state import (
    ComposeOutput,
    PostOutput,
    ReasonOutput,
    RuleSnapshot,
    TriageRequest,
)
from backend.app.rag.store import SearchHit
from backend.app.schemas import (
    TRIAGE_LABELS,
    FollowUpQuestion,
    OutbreakContext,
    PatientCase,
)

MAX_CHUNK_CHARS = 1200

# Critical fields and their follow-up questions (fixed wording, English / Nigerian Pidgin).
FOLLOW_UPS: dict[str, dict[str, str]] = {
    "age_years": {
        "en": "How old is the patient?",
        "pcm": "How old the patient be?",
    },
    "fever_days": {
        "en": "How many days has the fever lasted?",
        "pcm": "How many days the body don dey hot?",
    },
    "rdt_result": {
        "en": "What was the malaria RDT result: positive, negative, or not done?",
        "pcm": "Wetin malaria RDT show: positive, negative, or dem never do am?",
    },
}


def follow_up(field: str, language: str) -> FollowUpQuestion:
    return FollowUpQuestion(id=field, text=FOLLOW_UPS[field]["pcm" if language == "pcm" else "en"])


def _schema_for_prompt(model: type, drop: tuple[str, ...] = ()) -> str:
    schema = model.model_json_schema()
    for field in drop:
        schema.get("properties", {}).pop(field, None)
    return json.dumps(schema, separators=(",", ":"))


def full_text(req: TriageRequest) -> str:
    """The worker's words plus any follow-up answers (feeds intake and the text rules)."""
    if not req.answers:
        return req.text
    lines = [req.text, "", "Follow-up answers:"]
    for a in req.answers:
        question = FOLLOW_UPS.get(a.id, {}).get("en", a.id)
        lines.append(f"- {question} {a.answer}")
    return "\n".join(lines)


# --- intake -----------------------------------------------------------------

INTAKE_SYSTEM = """Task: intake.
Convert a primary health care worker's description of a febrile patient (English or \
Nigerian Pidgin) into ONE JSON object matching the schema below. Reply with the JSON only.
- Use null for anything not stated. Never guess.
- age_years: convert months to years (e.g. 18 months -> 1.5).
- fever_days: duration of fever in days ("hot body 4 days" -> 4).
- symptoms: short English phrases for what is described.
- danger_signs: only codes from the schema's enum, only when clearly described.
- antimalarial_no_response: true only if antimalarials were taken without improvement.
- language: "pcm" if the text is Nigerian Pidgin, else "en".
Schema:
"""


def intake_messages(req: TriageRequest) -> list[dict[str, str]]:
    schema = _schema_for_prompt(PatientCase, drop=("raw_text", "missing_fields", "state", "lga"))
    return [
        {"role": "system", "content": INTAKE_SYSTEM + schema},
        {"role": "user", "content": full_text(req)},
    ]


# --- reason -----------------------------------------------------------------

REASON_SYSTEM = """Task: reason.
You are clinical decision support for a primary health care worker (CHEW or nurse) at a \
primary health centre in Nigeria. You do not diagnose; a human clinician decides.
Given the case, the deterministic rule findings, guideline excerpts (each with an ID) and \
current outbreak signals (each with a URL), reply with ONE JSON object matching the schema.

Hard rules:
1. NEVER write drug doses, strengths, amounts or frequencies (no mg, ml, tablet counts, \
"twice daily"). Name the drug or treatment only; doses come from a separate verified table.
2. Cite ONLY the guideline IDs and outbreak URLs given below, copied exactly. Never invent \
a citation. Cite at least one source for each differential item and action where possible.
3. If the rules found danger signs, triage_level must be "refer_now". You may raise the \
triage level above the rule floor, never lower it.
4. Weigh active outbreaks in the patient's state. If an outbreak disease fits the \
presentation, include it in the differential and give its isolation/referral steps.
5. differential: 2-4 conditions, most likely first, each with reasons and what to check next.
6. actions: short, concrete steps the health worker can take now.
7. triage_level: "refer_now" (emergency), "refer_24h" (needs facility review within a day), \
or "treat_monitor" (manage at the PHC with review).

Schema:
"""


def reason_messages(
    case: PatientCase,
    pre: RuleSnapshot,
    hits: list[SearchHit],
    outbreak: OutbreakContext | None,
) -> list[dict[str, str]]:
    excerpts = [
        f"[{h.chunk.id}] {h.chunk.title}"
        + (f" | {h.chunk.section}" if h.chunk.section else "")
        + f" (p.{h.chunk.page})\n{h.chunk.text[:MAX_CHUNK_CHARS]}"
        for h in hits
    ]
    if outbreak is None or outbreak.status != "ok":
        outbreak_text = "Outbreak data unavailable. Do not assume any outbreak status."
    elif not outbreak.signals:
        outbreak_text = "Checked; no current outbreak signals found for this state."
    else:
        outbreak_text = "\n".join(
            f"- {s.disease} in {s.state}: {s.status}, reported {s.report_date} [{s.url}]"
            for s in outbreak.signals
        )
    rules = {
        "danger_signs": [h.label for h in pre.danger_signs],
        "triage_floor": pre.floor.value if pre.floor else None,
        "lassa_suspected_by_rule": pre.lassa_signal is not None,
    }
    user = (
        f"CASE:\n{case.model_dump_json(exclude_none=True)}\n\n"
        f"RULE FINDINGS:\n{json.dumps(rules)}\n\n"
        f"OUTBREAK SIGNALS:\n{outbreak_text}\n\n"
        "GUIDELINE EXCERPTS:\n" + ("\n\n".join(excerpts) if excerpts else "None available.")
    )
    return [
        {"role": "system", "content": REASON_SYSTEM + _schema_for_prompt(ReasonOutput)},
        {"role": "user", "content": user},
    ]


# --- compose ----------------------------------------------------------------

COMPOSE_SYSTEM = """Task: compose.
Write for a primary health care worker in Nigeria. Reply with ONE JSON object:
{"summary": str, "referral_note": str}
Write both fields in {language}.
- summary: 2-4 short, plain sentences: the triage decision, why, and the next steps.
- referral_note: a note for the receiving facility (or, if not referred, a short care note \
for the patient record): age, key findings, danger signs, triage decision, likely \
conditions, actions advised, and outbreak context if relevant.
- Never include patient names or identifiers.
- Actions are ADVISED steps. Never state that an action was done (e.g. "patient isolated", \
"officer notified") unless the health worker's description says it was done.
- NEVER write drug doses, amounts or frequencies. Doses are appended separately.
- Do not change or soften the triage decision."""

LANGUAGE_NAMES = {"en": "clear, simple English", "pcm": "Nigerian Pidgin"}


def compose_messages(case: PatientCase, post: PostOutput) -> list[dict[str, str]]:
    system = COMPOSE_SYSTEM.replace("{language}", LANGUAGE_NAMES.get(case.language, "English"))
    payload = {
        "triage_decision": TRIAGE_LABELS[post.triage_level],
        "rationale": post.triage_rationale,
        "case": case.model_dump(exclude_none=True, exclude={"missing_fields"}),
        "danger_signs": [h.label for h in post.danger_signs],
        "lassa_suspected": post.lassa_suspected,
        "differential": [
            {"condition": d.condition, "likelihood": d.likelihood, "reasons": d.reasons}
            for d in post.differential
        ],
        "actions": [a.text for a in post.actions],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
    ]


__all__ = [
    "ComposeOutput",
    "ReasonOutput",
    "compose_messages",
    "follow_up",
    "full_text",
    "intake_messages",
    "reason_messages",
]
