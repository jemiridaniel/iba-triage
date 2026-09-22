"""Deterministic danger-sign rules and the triage floor.

Runs twice in the pipeline: `rules_pre` (before retrieval/reasoning) and `rules_post`
(after the LLM). Any danger sign or a positive Lassa suspicion sets a floor of REFER_NOW,
and `apply_floor` guarantees the LLM can raise the triage level but never lower it.

Two detection channels, both of which raise the floor:
  * text rules  - regexes over the worker's own words (English + Nigerian Pidgin);
  * structured  - `PatientCase.danger_signs` flagged by the intake model.
The text channel exists so a danger sign is caught even if the intake model misses it.

Design bias: recall over precision. A missed sign is a safety bug; an unnecessary referral
is not. Negation handling is deliberately narrow (a negation cue within the three words
before a match, in the same clause) and only suppresses the text channel.

Criteria: WHO / NMEP severe malaria features and WHO IMCI general danger signs (SPEC §5).
TODO: attach guideline section citations once the index is built (week 1, ingest).
"""

import re
from dataclasses import dataclass, field

from backend.app.schemas import (
    DANGER_SIGN_LABELS,
    DangerSignCode,
    DangerSignHit,
    OutbreakSignal,
    PatientCase,
    TriageLevel,
    most_urgent,
)

D = DangerSignCode

# Shared fragments
_CANT = r"(?:unable|not\s+able|can(?:no|')?t|can\s+not|could\s*n[o']?t|no\s+fit)"
_DRUG_CTX = (
    r"(?:the\s+)?(?:treatment|therapy|drugs?|medic\w*|anti-?malari\w*|(?-i:ACTs?)|coartem|"
    r"artemether\w*|antibiotics?|malaria\s+(?:drugs?|medicine|treatment))"
)

_PATTERNS: dict[DangerSignCode, list[str]] = {
    D.CONVULSIONS: [
        r"\bconvuls\w*",
        r"\bseiz(?:ure|ures|ing|ed)\b",
        r"\bfits\b",
        r"\bfitting\b",
        r"\bhad\s+an?\s+fit\b",
        r"\bjerk(?:s|ing|ed)?\b",
    ],
    D.UNABLE_TO_DRINK: [
        rf"\b(?:{_CANT}|refus\w*|stopped|no\s+gree|no\s+dey|no\s+wan)\s+(?:to\s+)?"
        r"(?:drink\w*|breast\s*-?\s*feed\w*|suck\w*|feed\w*|take\s+(?:fluids?|water|breast))\b",
        r"\bnot\s+(?:drinking|sucking|breast\s*-?\s*feeding|feeding)\b",
    ],
    D.VOMITING_EVERYTHING: [
        r"\b(?:vomit\w*|throw\w*\s+up|throw\w*\s+out)\s+(?:out\s+)?(?:every\s*thing|all|any\s*thing)\b",
        rf"\b{_CANT}\s+(?:to\s+)?(?:keep|hold)\s+"
        r"(?:any\s*thing|food|water|fluids?|drugs?|medicine|drinks?)\s+down\b",
        r"\b(?:every\s*thing|any\s*thing)\s+(?:wey\s+)?(?:(?:he|she|e|pikin|the\s+child)\s+)?"
        r"(?:eat|eats|ate|chop|chops|drink|drinks|take|takes|swallow\w*)\s*,?\s+"
        r"(?:(?:he|she|e|it)\s+)?(?:dey\s+|don\s+)?(?:vomit\w*|throw\w*)",
        r"\b(?:persistent|repeated|incessant|intractable|uncontrollable)\s+vomit\w*",
    ],
    D.LETHARGY_UNCONSCIOUS: [
        r"\bletharg\w*",
        r"\bunconscious\w*",
        r"\b(?:un|not\s+)rous(?:e)?able\b",
        r"\b(?:difficult|hard|unable|can(?:no|')?t|can\s+not)\s+(?:to\s+)?(?:wake|rouse|arouse)\w*",
        r"\bsleep\w*\s+(?:too\s+much|anyhow|all\s+(?:the\s+)?(?:time|day))",
        r"\bdrows\w*",
        r"\bcoma\w*\b",
        rf"\bunresponsive\b(?!\s+to\s+{_DRUG_CTX})",
        rf"\bnot\s+respond(?:s|ing|ed)?\b(?!\s+to\s+{_DRUG_CTX})",
        rf"\bno\s+(?:dey\s+)?respond\b(?!\s+to\s+{_DRUG_CTX})",
        r"\bconfus\w*",
        r"\bfloppy\b",
        r"\bpass(?:ed|es)?\s+out\b",
        r"\b(?:impaired|altered|reduced|decreased|loss\s+of)\s+(?:level\s+of\s+)?"
        r"(?:consciousness|sensorium|mental)",
    ],
    D.BLEEDING: [
        r"\bbleed\w*",
        r"\bnose\s*bleed\w*",
        r"\bh(?:a)?emorrhag\w*",
        r"\bepistaxis\b",
        r"\bh(?:a)?ematemesis\b",
        r"\bblood\s+(?:from|for|in)\s+(?:the\s+|his\s+|her\s+|im\s+)?"
        r"(?:nose|mouth|gums?|eyes?|ears?|vagina|injection\s+site|cannula|drip\s+site|vomit\w*)",
        r"\b(?:vomit\w*|cough\w*|spit\w*)\s+(?:out\s+)?blood\b",
        r"\bblood\w*\s+(?:dey\s+|don\s+)?(?:comot|commot|come\s+out)",
        r"\bbloody\s+(?:vomit\w*|urine|sputum|spit)",
    ],
    D.JAUNDICE: [
        r"\bjaundic\w*",
        r"\bicter\w*",
        r"\byellow\w*\s+(?:eyes?|skin|body|sclera\w*|palms?)",
        r"\b(?:eyes?|skin|body|sclera\w*)\s+(?:(?:is|are|dey|don|has|have|turn\w*|become|became|"
        r"look\w*|go|getting)\s+){0,3}yellow\w*",
    ],
    D.RESPIRATORY_DISTRESS: [
        r"\b(?:difficult\w*|trouble|struggl\w*|labou?r\w*|hard)\s+(?:in\s+|with\s+|to\s+)?breath\w*",
        r"\bbreath\w*\s+(?:is\s+|dey\s+)?(?:difficult\w*|labou?red|hard)",
        r"\bshort(?:ness)?\s+(?:of\s+)?breath\w*",
        r"\bbreathless\w*",
        r"\bchest\s+(?:wall\s+)?in-?drawing\b",
        r"\bgrunt\w*",
        r"\bgasp\w*",
        r"\bnasal\s+flar\w*",
        r"\brespiratory\s+distress\b",
        r"\bdeep\s+(?:and\s+)?(?:rapid\s+|fast\s+)?breath\w*",
        rf"\b{_CANT}\s+(?:to\s+)?breath\w*",
        r"\bbreath\s+dey\s+cut\b",
        r"\bpanting\b",
        r"\bdey\s+pant\b",
        r"\bdyspn\w*",
        r"\bstridor\b",
    ],
    D.SEVERE_PALLOR: [
        r"\b(?:severe\w*|very|extreme\w*|marked\w*|profound\w*)\s+pal(?:e|lor|eness)\b",
        r"\bsevere\w*\s+an(?:a)?emi\w*",
        r"\bno\s+blood\s+(?:for|in)\s+(?:(?:his|her|the|im)\s+)?body\b",
        r"\b(?:palms?|eyes?|lips|tongue|body)\s+(?:(?:dey|don|is|are|look\w*|turn\w*)\s+){0,2}white\b",
    ],
    D.DARK_URINE: [
        r"\b(?:dark|black|brown|cola|coke|coca[- ]?cola|tea|red)[- ]?(?:colou?red\s+)?"
        r"(?:urine|piss|pee)\b",
        r"\b(?:urine|piss|pee)\s+(?:(?:is|dey|don|turn\w*|become|became|look\w*|like|go)\s+){0,3}"
        r"(?:very\s+)?(?:dark|black|brown|cola|coke|tea|red)\b",
        r"\bh(?:a)?emoglobinuri\w*",
        r"\bblack\s*water\b",
    ],
    D.PROSTRATION: [
        r"\bprostrat\w*",
        rf"\b(?:{_CANT}|too\s+weak)\s+(?:to\s+)?(?:sit|stand|walk|waka)\w*",
    ],
}

_RULES: dict[DangerSignCode, list[re.Pattern[str]]] = {
    code: [re.compile(p, re.IGNORECASE) for p in patterns] for code, patterns in _PATTERNS.items()
}

# --- negation ---------------------------------------------------------------

_CLAUSE_BREAK = re.compile(
    r"[.;,!?:\n()]|\b(?:but|and|however|though|although|then)\b", re.IGNORECASE
)
_NEGATION_CUES = frozenset(
    {"no", "not", "never", "without", "denies", "denied", "deny", "nor", "non"}
    | {"isn't", "isnt", "wasn't", "wasnt", "hasn't", "hasnt", "didn't", "didnt"}
    | {"doesn't", "doesnt"}
)
# "no well" / "not well" / "not better" describe illness, not absence of a sign.
_ILLNESS_IDIOM = re.compile(
    r"\b(?:no|not)\s+(?:(?:dey|feeling|feel|too|very)\s+)?"
    r"(?:well|fine|ok(?:ay)?|good|better|improv\w*)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[\w']+")


def _is_negated(text: str, start: int, window: int = 3) -> bool:
    clause_start = 0
    for m in _CLAUSE_BREAK.finditer(text, 0, start):
        clause_start = m.end()
    prefix = _ILLNESS_IDIOM.sub(" ", text[clause_start:start])
    words = [w.lower() for w in _WORD.findall(prefix)][-window:]
    return any(w in _NEGATION_CUES for w in words)


# --- detection --------------------------------------------------------------


def detect_from_text(text: str) -> list[DangerSignHit]:
    hits: list[DangerSignHit] = []
    for code, patterns in _RULES.items():
        match = next(
            (
                m
                for pattern in patterns
                for m in pattern.finditer(text)
                if not _is_negated(text, m.start())
            ),
            None,
        )
        if match is not None:
            hits.append(DangerSignHit(code=code, source="rule", evidence=match.group(0)))
    return hits


def detect_danger_signs(
    case: PatientCase | None = None, text: str | None = None
) -> list[DangerSignHit]:
    """Union of text-rule hits and intake-flagged signs, one hit per sign, rules first."""
    if text is None and case is not None:
        text = case.raw_text
    hits = {h.code: h for h in detect_from_text(text)} if text else {}
    if case is not None:
        for code in case.danger_signs:
            hits.setdefault(code, DangerSignHit(code=code, source="llm_intake"))
    return [hits[code] for code in DangerSignCode if code in hits]


# --- Lassa suspicion --------------------------------------------------------

_NO_RESPONSE_RE = re.compile(
    r"\b(?:anti-?malari\w*|coartem|artemether\w*|lumefantrine|(?-i:ACTs?)|amatem|lonart|"
    r"malaria\s+(?:drugs?|medicine|treatment|injection))\b"
    r"[^.;]{0,60}?"
    r"\b(?:no\s+(?:change|improvement|better|difference)|not\s+(?:improv\w*|better|respond\w*|"
    r"work\w*|help\w*)|no\s+(?:dey\s+)?(?:better|work|help)|never\s+better|still|persist\w*|"
    r"failed|no\s+response)",
    re.IGNORECASE,
)
_NOT_RESPONDING_TO_DRUGS_RE = re.compile(
    rf"\b(?:not|no|never)\s+(?:dey\s+)?respond\w*\s+to\s+{_DRUG_CTX}", re.IGNORECASE
)


def no_antimalarial_response(case: PatientCase) -> bool:
    if case.antimalarial_no_response:
        return True
    text = case.raw_text or ""
    return bool(_NO_RESPONSE_RE.search(text) or _NOT_RESPONDING_TO_DRUGS_RE.search(text))


def _norm_state(state: str) -> str:
    s = re.sub(r"\s+state$", "", state.strip().lower())
    return "fct" if s in {"abuja", "federal capital territory", "fct abuja"} else s


def active_outbreak(
    signals: list[OutbreakSignal], state: str | None, disease: str
) -> OutbreakSignal | None:
    """An active, sourced (URL + date) outbreak signal for `disease` in `state`."""
    if not state:
        return None
    target = _norm_state(state)
    for sig in signals:
        if (
            disease in sig.disease.lower()
            and _norm_state(sig.state) == target
            and sig.status == "active"
            and sig.url
            and sig.report_date
        ):
            return sig
    return None


def lassa_suspected(
    case: PatientCase, signals: list[OutbreakSignal], hits: list[DangerSignHit]
) -> OutbreakSignal | None:
    """SPEC §5: in a state with an active Lassa signal, fever >= 3 days with no response to
    antimalarials, or bleeding. Returns the triggering signal (for citation) or None."""
    signal = active_outbreak(signals, case.state, "lassa")
    if signal is None:
        return None
    bleeding = any(h.code == D.BLEEDING for h in hits)
    prolonged_unresponsive = (case.fever_days or 0) >= 3 and no_antimalarial_response(case)
    return signal if (bleeding or prolonged_unresponsive) else None


# --- floor ------------------------------------------------------------------

LASSA_IPC_REMINDER = (
    "Suspected Lassa fever: isolate the patient, use gloves and standard IPC precautions, "
    "avoid contact with body fluids, notify the LGA disease surveillance officer, "
    "and refer urgently."
)


@dataclass
class RuleAssessment:
    floor: TriageLevel | None
    danger_signs: list[DangerSignHit] = field(default_factory=list)
    lassa_signal: OutbreakSignal | None = None
    reasons: list[str] = field(default_factory=list)
    reminders: list[str] = field(default_factory=list)

    @property
    def lassa_suspected(self) -> bool:
        return self.lassa_signal is not None


def assess(
    case: PatientCase,
    outbreak_signals: list[OutbreakSignal] | None = None,
    extra_hits: list[DangerSignHit] | None = None,
) -> RuleAssessment:
    """Compute the rule-based triage floor.

    `extra_hits` carries signs found by later steps (e.g. the reason model in rules_post);
    they can only add to the floor.
    """
    hits = detect_danger_signs(case)
    seen = {h.code for h in hits}
    for h in extra_hits or []:
        if h.code not in seen:
            hits.append(h)
            seen.add(h.code)

    lassa = lassa_suspected(case, outbreak_signals or [], hits)
    reasons = [f"Danger sign: {DANGER_SIGN_LABELS[h.code]}" for h in hits]
    reminders: list[str] = []
    if lassa is not None:
        reasons.append(f"Suspected Lassa fever with active outbreak in {lassa.state}")
        reminders.append(LASSA_IPC_REMINDER)

    floor = TriageLevel.REFER_NOW if (hits or lassa) else None
    return RuleAssessment(
        floor=floor, danger_signs=hits, lassa_signal=lassa, reasons=reasons, reminders=reminders
    )


def apply_floor(level: TriageLevel | None, floor: TriageLevel | None) -> TriageLevel:
    """Final triage = the more urgent of the model's level and the rule floor.

    A missing level (model failed) fails safe to REFER_NOW.
    """
    if level is None:
        return TriageLevel.REFER_NOW
    return most_urgent(level, floor) or level
