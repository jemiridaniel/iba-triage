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

from backend.app.rules.text import ascii_punct
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
        # IMCI's own wording for this sign is "not able to drink OR drinking poorly" -- both
        # halves count as the same criterion (WHO IMCI Chart Booklet, Classify Dehydration).
        r"\bdrink\w*\s+poorly\b",
        r"\bpoor\w*\s+(?:oral\s+)?(?:intake|feeding)\b",
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
        # "deep" is the anchor word here -- "fast breathing" or "breathing fast" alone, with no
        # "deep" and no other sign, does NOT match this pattern (or any other one in this list).
        # This is intentional: IMCI classifies fast breathing alone (child 2-59 months) as
        # pneumonia, not a general danger sign (see docs/CLINICAL_REVIEW.md §2 and
        # docs/review/clinical-logic-review.docx §1). Verified against this exact pattern,
        # 2026-10-10 -- Dr. Amos's review asked whether the phrase list and the stated
        # exclusion policy were actually consistent; they are.
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
    text = ascii_punct(text)
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
#
# NCDC National Guideline for Lassa Fever Case Management (2018), §1.1.2 Suspected case:
#   fever for 3-21 days (measured >= 38 °C) with one or more of: vomiting, diarrhoea,
#   sore throat, myalgia, generalised body weakness, abnormal bleeding, abdominal pain.
#   Index of suspicion is raised by: (a) no response to standard antimalarial treatment and
#   treatment for other common causes of fever within 48-72 h; (b) contact with a probable or
#   confirmed case within 21 days; (c) travel to a high-risk/burden area; (d) contact with body
#   fluids of a patient who died of a febrile illness suggestive of Lassa.
# Anchor phrases below let rules_post cite the exact guideline chunk.

LASSA_CASE_DEF_ANCHOR = ("ncdc-lassa", "fever for 3-21 days with a measured temperature")
LASSA_TRIAGE_ANCHOR = ("ncdc-lassa", "Put patient in a holding area")

_LASSA_SYMPTOMS: dict[str, str] = {
    "vomiting": r"\bvomit\w*|\bthrow\w*\s+up",
    "diarrhoea": r"\bdiarr?h(?:o)?ea\w*|\bstool\w*\s+(?:dey\s+)?(?:run|watery)|\brunning\s+stomach",
    "sore throat": r"\bsore\s+throat|\bthroat\s+(?:pain|dey\s+pain)",
    "myalgia": r"\bmyalgia|\bmuscle\s+(?:pain|ache)|\bbody\s+(?:pain|ache)",
    "generalised weakness": r"\b(?:general\w*\s+)?(?:body\s+)?weak\w*",
    "abnormal bleeding": r"\bbleed\w*|\bh(?:a)?emorrhag\w*",
    "abdominal pain": r"\babdominal\s+pain|\bstomach\s+(?:pain|ache)|\bbelle\s+(?:pain|dey\s+pain)",
}
_LASSA_SYMPTOM_RES = {k: re.compile(v, re.IGNORECASE) for k, v in _LASSA_SYMPTOMS.items()}


def _has_vomiting(text: str) -> bool:
    """Generic vomiting mention (any severity), negation-aware. Reused by the sunken-eyes-plus-

    vomiting cholera rule below -- distinct from D.VOMITING_EVERYTHING, which requires the
    stricter "vomiting everything" phrasing and is already an independent danger sign.
    """
    text = ascii_punct(text or "")
    pattern = _LASSA_SYMPTOM_RES["vomiting"]
    return any(not _is_negated(text, m.start()) for m in pattern.finditer(text))


_TREATMENT_WORDS = (
    r"(?:anti-?malari\w*|coartem|artemether\w*|lumefantrine|(?-i:ACTs?)|amatem|lonart|"
    r"malaria\s+(?:drugs?|medicine|treatment|injection)|antibiotic\w*|amoxic\w*|amoxil|"
    r"ampiclox|cipro\w*|augmentin|septrin|metronidazole|flagyl|drugs?|medicine|treatment)"
)
_NO_RESPONSE_RE = re.compile(
    rf"\b{_TREATMENT_WORDS}\b"
    r"[^.;]{0,60}?"
    r"\b(?:no\s+(?:change|improvement|better|difference)|not\s+(?:improv\w*|better|respond\w*|"
    r"work\w*|help\w*)|no\s+(?:dey\s+)?(?:better|work|help)|never\s+better|still|persist\w*|"
    r"failed|no\s+response)",
    re.IGNORECASE,
)
_NOT_RESPONDING_TO_DRUGS_RE = re.compile(
    rf"\b(?:not|no|never)\s+(?:dey\s+)?respond\w*\s+to\s+{_DRUG_CTX}", re.IGNORECASE
)
_LASSA_CONTACT_RE = re.compile(
    r"\bcontact\s+with\s+(?:a\s+)?(?:known\s+|confirmed\s+|suspected\s+|probable\s+)?"
    r"(?:lassa|case|patient\s+with\s+lassa)|\b(?:rat|rodent)s?\b.{0,40}\b(?:food|house|droppings?|urine)",
    re.IGNORECASE,
)


def no_treatment_response(case: PatientCase) -> bool:
    """Criterion (a): fever did not respond to antimalarials and/or antibiotics."""
    if case.antimalarial_no_response or case.antibiotic_no_response:
        return True
    text = case.raw_text or ""
    return bool(_NO_RESPONSE_RE.search(text) or _NOT_RESPONDING_TO_DRUGS_RE.search(text))


no_antimalarial_response = no_treatment_response  # backward-compatible name


def lassa_case_definition(case: PatientCase) -> list[str]:
    """Matched §1.1.2 symptoms if the fever criterion is met; [] otherwise.

    Fever: 3-21 days. A measured temperature below 38 °C excludes; an unmeasured one does not
    (recall over precision; the result prompts measuring it).
    """
    days = case.fever_days
    if days is None or not 3 <= days <= 21:
        return []
    if case.temperature_c is not None and case.temperature_c < 38.0:
        return []
    text = ascii_punct(" ".join([case.raw_text or "", *case.symptoms]))
    return [name for name, rx in _LASSA_SYMPTOM_RES.items() if rx.search(text)]


def _norm_state(state: str) -> str:
    s = re.sub(r"\s+state$", "", state.strip().lower())
    return "fct" if s in {"abuja", "federal capital territory", "fct abuja"} else s


def active_outbreak(
    signals: list[OutbreakSignal], state: str | None, disease: str
) -> OutbreakSignal | None:
    """An active, sourced, CURRENT live outbreak signal for `disease` in the patient's state.

    Three guards, all deliberate:
    - `_norm_state(sig.state) == target`: NCDC publishes national sitreps, so a search for one
      state returns outbreaks in others (FEEDBACK T3). Only the patient's own state escalates.
    - `sig.recency == "current"`: a report older than 60 days, or one whose date could not be
      established, informs the differential but never escalates by itself (FEEDBACK T1).
    - `sig.url`: an unsourced signal is not actionable.
    """
    if not state:
        return None
    target = _norm_state(state)
    for sig in signals:
        if (
            sig.basis == "live"
            and disease in sig.disease.lower()
            and _norm_state(sig.state) == target
            and sig.status == "active"
            and sig.url
            and sig.report_date
            and sig.recency == "current"
        ):
            return sig
    return None


def endemic_baseline(
    signals: list[OutbreakSignal], state: str | None, disease: str
) -> OutbreakSignal | None:
    """A high-burden static baseline entry for `disease` in `state`.

    Only tier="high_burden" baseline entries can drive an escalation rule (this is what makes
    "high_burden escalates" true for Lassa's baseline branch below). tier="reported" entries
    are still returned to the reasoning prompt (they inform the differential), but never from
    here -- this function is only ever used to decide whether a rule may escalate.
    """
    if not state:
        return None
    target = _norm_state(state)
    for sig in signals:
        if (
            sig.basis == "baseline"
            and sig.tier == "high_burden"
            and disease in sig.disease.lower()
            and _norm_state(sig.state) == target
        ):
            return sig
    return None


@dataclass
class LassaFinding:
    basis: str  # "live" (active outbreak signal), "baseline" (endemic state), or "any_state"
    signal: OutbreakSignal | None  # None for "any_state" -- no local outbreak data backs it
    floor: TriageLevel
    criteria: list[str]  # human-readable reasons, for the differential and trace


def lassa_assessment(
    case: PatientCase, signals: list[OutbreakSignal], hits: list[DangerSignHit]
) -> LassaFinding | None:
    """Lassa suspicion from the NCDC 2018 case definition plus place.

    live signal in the state:    definition met, OR fever >= 3 days with no treatment
                                 response, OR abnormal bleeding                -> Refer now
    endemic state (baseline):    definition met                                -> Refer now
                                 partial: fever >= 3 days with no treatment
                                 response or contact, no listed symptom        -> Refer within 24h

    Policy (2026-09-22, pending clinician review; see docs/CLINICAL_REVIEW.md): NCDC treats
    any suspected case as isolate-and-notify, and a missed case can infect the health worker,
    so under-triage costs far more than over-triage.
    """
    symptoms = lassa_case_definition(case)
    no_response = no_treatment_response(case)
    bleeding = any(h.code == D.BLEEDING for h in hits)
    contact = bool(_LASSA_CONTACT_RE.search(case.raw_text or ""))

    criteria: list[str] = []
    if symptoms:
        criteria.append(
            f"Meets NCDC suspected-case definition: fever {case.fever_days:g} days with "
            + ", ".join(symptoms)
        )
    if no_response:
        criteria.append("Fever not responding to antimalarial/antibiotic treatment")
    if contact:
        criteria.append("Possible contact with a Lassa case or rodents")
    if bleeding:
        criteria.append("Abnormal bleeding")

    live = active_outbreak(signals, case.state, "lassa")
    if live is not None:
        prolonged_unresponsive = (case.fever_days or 0) >= 3 and no_response
        if symptoms or prolonged_unresponsive or bleeding:
            criteria.append(f"Active Lassa outbreak reported in {live.state} ({live.report_date})")
            return LassaFinding("live", live, TriageLevel.REFER_NOW, criteria)
        return None

    base = endemic_baseline(signals, case.state, "lassa")
    if base is not None:
        season = " (peak season)" if base.in_season else ""
        place = f"{base.state} is a Lassa-endemic state{season} (baseline, no live data)"
        if symptoms or bleeding:
            return LassaFinding("baseline", base, TriageLevel.REFER_NOW, [*criteria, place])
        if (case.fever_days or 0) >= 3 and (no_response or contact):
            criteria.append("Partial features only: no listed symptom of the NCDC case definition")
            return LassaFinding("baseline", base, TriageLevel.REFER_24H, [*criteria, place])
        return None

    return _lassa_any_state_floor(case, symptoms, no_response, contact, criteria)


def _lassa_any_state_floor(
    case: PatientCase,
    symptoms: list[str],
    no_response: bool,
    contact: bool,
    criteria: list[str],
) -> LassaFinding | None:
    """Minimum Lassa floor for a state with no live signal and no high-burden baseline match.

    Clinical review (Dr. Amos, 2026-10-10), applied close to his exact wording: "reported" or
    unlisted states previously got no escalation at all from this rule -- not even Refer within
    24h -- so a textbook Lassa case (full case definition met, failed treatment) in, say, Lagos
    or Kogi had no deterministic backstop if the AI's own reasoning also missed it. This does
    not depend on the baseline list: it applies in any state.

    - contact (with a case or rodents) plus any stated fever -> Refer now.
    - pregnant with fever >= 3 days -> Refer now.
    - full case definition met, with RDT negative or failed treatment -> AT LEAST Refer
      within 24h (not automatically Refer now: there is no local outbreak evidence raising the
      prior the way a live signal or a high-burden state would).
    - partial features only (fever >= 3 days, no listed symptom, but failed treatment or
      contact) -> Refer within 24h, mirroring the existing high-burden "partial" tier above.
    """
    fever_known = case.fever_days is not None
    pregnant_prolonged = bool(case.pregnant) and (case.fever_days or 0) >= 3
    rdt_negative = case.rdt_result == "negative"

    if contact and fever_known:
        criteria.append("No local Lassa outbreak data, but possible contact plus fever")
        return LassaFinding("any_state", None, TriageLevel.REFER_NOW, criteria)
    if pregnant_prolonged:
        criteria.append("No local Lassa outbreak data, but pregnant with fever >= 3 days")
        return LassaFinding("any_state", None, TriageLevel.REFER_NOW, criteria)
    if symptoms and (rdt_negative or no_response):
        criteria.append(
            "No local Lassa outbreak data, but case definition met with RDT-negative or "
            "failed treatment (any-state floor)"
        )
        return LassaFinding("any_state", None, TriageLevel.REFER_24H, criteria)
    if (case.fever_days or 0) >= 3 and (no_response or contact):
        criteria.append("Partial features only, no local Lassa outbreak data (any-state floor)")
        return LassaFinding("any_state", None, TriageLevel.REFER_24H, criteria)
    return None


def lassa_suspected(
    case: PatientCase, signals: list[OutbreakSignal], hits: list[DangerSignHit]
) -> OutbreakSignal | None:
    """Backward-compatible: the triggering signal if Lassa is suspected at any tier."""
    finding = lassa_assessment(case, signals, hits)
    return finding.signal if finding else None


# --- severe dehydration -------------------------------------------------------------
#
# WHO IMCI Chart Booklet, "Classify Dehydration": Severe Dehydration = two or more of
#   lethargic or unconscious; sunken eyes; not able to drink or drinking poorly; skin pinch
#   goes back very slowly (>=2 s)                              -> Plan C, refer urgently.
# NCDC Guidelines for Acute Watery Diarrhoea (Cholera) Outbreak, independently: "Severe
#   dehydration: two or more of the following signs: lethargy/unconsciousness; sunken eyes;
#   inability to drink or drinking poorly; loss of skin turgor" -- the same four signs.
# Two of the four are already standalone IMCI general danger signs here (D.UNABLE_TO_DRINK,
# D.LETHARGY_UNCONSCIOUS) and already trigger Refer now alone; this rule adds the two that
# are not otherwise detected (sunken eyes; slow skin pinch) and combines all four, so a
# patient with e.g. sunken eyes + slow skin pinch is caught even with no single sign that
# would trigger on its own.
#
# Fires regardless of fever: dehydration is a fluid-loss emergency, not a febrile-illness
# finding, and the fever-path rules must not gate it (see followup.py).
#
# Clinical review (Dr. Amos, 2026-10-10): "no urine output" is not named by either WHO IMCI
# or the NCDC cholera guideline as one of the four dehydration-classification signs, so
# counting it in the two-of-four total would mean citing a guideline for something it doesn't
# say. It is now its own independent Refer-now sign instead (see `oliguria_assessment` below),
# not part of this count.
#
# Same review: the two-of-four rule alone does not catch "sunken eyes + vomiting" with no
# second dehydration-classification sign (the original live miss) -- see
# `sunken_eyes_vomiting_assessment` below, and a "two or more moderate signs" floor -- see
# `moderate_dehydration_assessment` below. Both are additional, independent floors.

DEHYDRATION_IMCI_ANCHOR = ("who-imci", "Skin pinch goes back very slowly")
DEHYDRATION_NCDC_ANCHOR = ("ncdc-cholera", "Severe dehydration: two or more")

_DEHYDRATION_SIGN_LABELS: dict[str, str] = {
    "unable_to_drink": "not able to drink or drinking poorly",
    "lethargic": "lethargic or unconsciousness",
    "sunken_eyes": "sunken eyes",
    "skin_pinch_slow": "skin pinch goes back slowly",
    "no_urine": "no urine output",
}
_DEHYDRATION_ONLY_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "sunken_eyes": [
        re.compile(p, re.IGNORECASE)
        for p in (
            r"\bsunken\s+eyes?\b",
            r"\beyes?\s+(?:dey\s+|don\s+|is\s+|are\s+|look\w*\s+|appear\w*\s+){0,2}sunk(?:en)?\b",
        )
    ],
    "skin_pinch_slow": [
        re.compile(p, re.IGNORECASE)
        for p in (
            r"\bskin\s*(?:pinch|turgor)\w*[^.;]{0,20}?\bslow\w*",
            r"\bslow\w*[^.;]{0,20}?\bskin\s*(?:pinch|turgor)\w*",
            r"\bpoor\s+skin\s+turgor\b",
            r"\bloss\s+of\s+skin\s+turgor\b",
        )
    ],
    "no_urine": [
        re.compile(p, re.IGNORECASE)
        for p in (
            r"\bno\s+urine\b",
            r"\banuria\b",
            r"\bnot\s+(?:pass\w*|urinat\w*)[^.;]{0,20}?\burine\b",
            r"\bno\s+(?:pass\w*|urinat\w*)\s+(?:of\s+)?urine\b",
            r"\bnot\s+urinat\w*\b",
        )
    ],
}


def _dehydration_text_signs(text: str) -> set[str]:
    text = ascii_punct(text or "")
    found: set[str] = set()
    for key, patterns in _DEHYDRATION_ONLY_PATTERNS.items():
        for pattern in patterns:
            match = next(
                (m for m in pattern.finditer(text) if not _is_negated(text, m.start())), None
            )
            if match is not None:
                found.add(key)
                break
    return found


@dataclass
class DehydrationFinding:
    signs: list[str]  # human-readable labels, in a stable order, for the differential
    floor: TriageLevel = TriageLevel.REFER_NOW


def dehydration_assessment(
    case: PatientCase, hits: list[DangerSignHit]
) -> DehydrationFinding | None:
    """Two or more severe-dehydration signs -> Refer now, independent of fever status.

    "No urine output" is deliberately not counted here -- see `oliguria_assessment`.
    """
    codes = {h.code for h in hits}
    present: list[str] = []
    if D.UNABLE_TO_DRINK in codes:
        present.append(_DEHYDRATION_SIGN_LABELS["unable_to_drink"])
    if D.LETHARGY_UNCONSCIOUS in codes:
        present.append(_DEHYDRATION_SIGN_LABELS["lethargic"])
    text_signs = _dehydration_text_signs(case.raw_text)
    for key in ("sunken_eyes", "skin_pinch_slow"):
        if key in text_signs:
            present.append(_DEHYDRATION_SIGN_LABELS[key])
    if len(present) < 2:
        return None
    return DehydrationFinding(signs=present)


# --- oliguria (no urine output) -- independent sign, not part of the two-of-four count -----
#
# Clinical review (Dr. Amos, 2026-10-10): make this its own Refer-now sign rather than a fifth
# dehydration-count sign, since no indexed guideline names it as a dehydration criterion. The
# closest resolvable citation is WHO's severe-malaria "renal impairment" criterion (creatinine
# / urea thresholds) -- a related but not identical clinical concept (lab values, not urine
# volume), cited honestly as "related criterion", not as a direct match.

OLIGURIA_ANCHOR = ("who-malaria", "Renal impairment: Plasma or serum creatinine")


@dataclass
class OliguriaFinding:
    floor: TriageLevel = TriageLevel.REFER_NOW


def oliguria_assessment(case: PatientCase) -> OliguriaFinding | None:
    """No/very little urine output -> Refer now on its own, independent of fever status."""
    if "no_urine" in _dehydration_text_signs(case.raw_text):
        return OliguriaFinding()
    return None


# --- moderate dehydration -- two-or-more floor, Refer within 24h ---------------------------
#
# Clinical review (Dr. Amos, 2026-10-10): NCDC's own dehydration chart (Table 6.2) has a
# moderate tier -- two or more of restless/irritable, sunken eyes, dry mouth, drinks eagerly,
# skin pinch goes back slowly, rapid pulse -- which NCDC and the IDSR both say needs a
# treatment centre or unit (Plan B), not home ORS alone. This rule only covers the subset of
# those signs distinguishable by text from the severe tier's wording (restless/irritable,
# drinks eagerly/thirsty, dry mouth, plus sunken eyes shared with the severe tier); "skin pinch
# goes back slowly" vs "very slowly" and pulse rate are not reliably distinguishable from free
# text and are not detected here.

_MODERATE_SIGN_LABELS: dict[str, str] = {
    "restless": "restless or irritable",
    "drinks_eagerly": "drinking eagerly / very thirsty",
    "dry_mouth": "dry mouth",
    "sunken_eyes": "sunken eyes",
}
_MODERATE_ONLY_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "restless": [
        re.compile(p, re.IGNORECASE) for p in (r"\brestless\w*\b", r"\birritab\w*\b")
    ],
    "drinks_eagerly": [
        re.compile(p, re.IGNORECASE)
        for p in (
            r"\bdrink\w*\s+eager\w*\b",
            r"\bvery\s+thirsty\b",
            r"\bthirsty\b",
        )
    ],
    "dry_mouth": [
        re.compile(p, re.IGNORECASE)
        for p in (r"\bdry\s+mouth\b", r"\bdry\s+tongue\b", r"\bmouth\s+(?:and\s+tongue\s+)?dry\b")
    ],
}


def _moderate_text_signs(text: str) -> set[str]:
    text = ascii_punct(text or "")
    found: set[str] = set()
    for key, patterns in _MODERATE_ONLY_PATTERNS.items():
        for pattern in patterns:
            match = next(
                (m for m in pattern.finditer(text) if not _is_negated(text, m.start())), None
            )
            if match is not None:
                found.add(key)
                break
    return found


@dataclass
class ModerateDehydrationFinding:
    signs: list[str]
    floor: TriageLevel = TriageLevel.REFER_24H


def moderate_dehydration_assessment(case: PatientCase) -> ModerateDehydrationFinding | None:
    """Two or more moderate-dehydration signs -> Refer within 24h (NCDC Plan B needs a

    treatment centre or unit, not home ORS alone), independent of fever status.
    """
    present: list[str] = []
    moderate_signs = _moderate_text_signs(case.raw_text)
    if "sunken_eyes" in _dehydration_text_signs(case.raw_text):
        present.append(_MODERATE_SIGN_LABELS["sunken_eyes"])
    for key in ("restless", "drinks_eagerly", "dry_mouth"):
        if key in moderate_signs:
            present.append(_MODERATE_SIGN_LABELS[key])
    if len(present) < 2:
        return None
    return ModerateDehydrationFinding(signs=present)


# --- sunken eyes + vomiting (suspected cholera) ---------------------------------------------
#
# Clinical review (Dr. Amos, 2026-10-10): the exact live case that prompted the two-of-four
# rule above (watery stool, vomiting, no fever, sunken eyes) has only ONE of the four counted
# signs (sunken eyes; vomiting is not one of the four), so that rule alone does not fix it.
# NCDC's own guide (p.61): "If at any time... frequent, severe vomiting occurs... shift
# immediately to Treatment Plan C"; a PHC with no lab and no IV fluids cannot deliver Plan C,
# so this combination is referred rather than treated with oral ORS alone.
#
# Tier: Refer now if the patient is in a cholera-prone area (endemicity high_burden) or there
# is a live cholera signal in their state, OR the patient is in a vulnerable group (pregnant,
# a child under 5, or 60 or older -- "comorbid" is not checkable from free text and is not
# applied). Refer within 24h otherwise. This does not depend on the baseline list for its
# *minimum* tier -- it fires in any state -- only for whether it escalates further to Refer
# now on area grounds.

SUNKEN_EYES_VOMITING_ANCHOR = ("ncdc-cholera", "if the patient vomits, wait 10 minutes")
SUNKEN_EYES_VOMITING_ADVICE_ANCHOR = ("ncdc-cholera", "stool specimen")

_ELDERLY_AGE_YEARS = 60
_YOUNG_CHILD_AGE_YEARS = 5


@dataclass
class SunkenEyesVomitingFinding:
    floor: TriageLevel
    vulnerable: bool
    cholera_context: bool


def sunken_eyes_vomiting_assessment(
    case: PatientCase, signals: list[OutbreakSignal] | None = None
) -> SunkenEyesVomitingFinding | None:
    text = case.raw_text or ""
    if "sunken_eyes" not in _dehydration_text_signs(text) or not _has_vomiting(text):
        return None
    vulnerable = bool(case.pregnant) or (
        case.age_years is not None
        and (case.age_years < _YOUNG_CHILD_AGE_YEARS or case.age_years >= _ELDERLY_AGE_YEARS)
    )
    cholera_context = active_outbreak(signals or [], case.state, "cholera") is not None or (
        endemic_baseline(signals or [], case.state, "cholera") is not None
    )
    floor = TriageLevel.REFER_NOW if (vulnerable or cholera_context) else TriageLevel.REFER_24H
    return SunkenEyesVomitingFinding(
        floor=floor, vulnerable=vulnerable, cholera_context=cholera_context
    )


# --- floor ------------------------------------------------------------------

# Wording follows NCDC 2018 §2.1.1 (triage of a suspected case).
LASSA_IPC_REMINDER = (
    "Suspected Lassa fever: keep the patient in a separate holding area, use infection "
    "prevention measures (gloves; avoid contact with blood and body fluids), notify the LGA "
    "disease surveillance officer, and refer for Lassa testing."
)


@dataclass
class RuleAssessment:
    floor: TriageLevel | None
    danger_signs: list[DangerSignHit] = field(default_factory=list)
    lassa: LassaFinding | None = None
    dehydration: DehydrationFinding | None = None
    moderate_dehydration: ModerateDehydrationFinding | None = None
    oliguria: OliguriaFinding | None = None
    sunken_eyes_vomiting: SunkenEyesVomitingFinding | None = None
    reasons: list[str] = field(default_factory=list)
    reminders: list[str] = field(default_factory=list)

    @property
    def lassa_signal(self) -> OutbreakSignal | None:
        return self.lassa.signal if self.lassa else None

    @property
    def lassa_suspected(self) -> bool:
        return self.lassa is not None


def assess(
    case: PatientCase,
    outbreak_signals: list[OutbreakSignal] | None = None,
    extra_hits: list[DangerSignHit] | None = None,
) -> RuleAssessment:
    """Compute the rule-based triage floor.

    `outbreak_signals` may mix live signals and baseline endemicity entries.
    `extra_hits` carries signs found by later steps (e.g. the reason model in rules_post);
    they can only add to the floor.
    """
    hits = detect_danger_signs(case)
    seen = {h.code for h in hits}
    for h in extra_hits or []:
        if h.code not in seen:
            hits.append(h)
            seen.add(h.code)

    lassa = lassa_assessment(case, outbreak_signals or [], hits)
    dehydration = dehydration_assessment(case, hits)
    moderate_dehydration = None if dehydration else moderate_dehydration_assessment(case)
    oliguria = oliguria_assessment(case)
    sunken_eyes_vomiting = (
        None if dehydration else sunken_eyes_vomiting_assessment(case, outbreak_signals)
    )
    reasons = [f"Danger sign: {DANGER_SIGN_LABELS[h.code]}" for h in hits]
    reminders: list[str] = []
    if lassa is not None:
        if lassa.basis == "live":
            tier = "active outbreak"
        elif lassa.basis == "baseline":
            tier = "endemic state (baseline)"
        else:
            tier = "any-state floor, no local outbreak data"
        place = lassa.signal.state if lassa.signal else (case.state or "state not given")
        reasons.append(f"Suspected Lassa fever ({tier}: {place})")
        reminders.append(LASSA_IPC_REMINDER)
    if dehydration is not None:
        reasons.append(
            "Severe dehydration (WHO IMCI + NCDC cholera guideline): two or more of "
            + ", ".join(dehydration.signs)
        )
    if moderate_dehydration is not None:
        reasons.append(
            "Moderate dehydration (NCDC Table 6.2): two or more of "
            + ", ".join(moderate_dehydration.signs)
        )
    if oliguria is not None:
        reasons.append("No urine output (little or no urine passed)")
    if sunken_eyes_vomiting is not None:
        context = (
            "cholera-prone area/active outbreak"
            if sunken_eyes_vomiting.cholera_context
            else "vulnerable patient"
            if sunken_eyes_vomiting.vulnerable
            else "no outbreak context"
        )
        reasons.append(f"Sunken eyes with vomiting, suspected cholera ({context})")

    floor = most_urgent(
        TriageLevel.REFER_NOW if hits else None,
        lassa.floor if lassa else None,
        dehydration.floor if dehydration else None,
        moderate_dehydration.floor if moderate_dehydration else None,
        oliguria.floor if oliguria else None,
        sunken_eyes_vomiting.floor if sunken_eyes_vomiting else None,
    )
    return RuleAssessment(
        floor=floor,
        danger_signs=hits,
        lassa=lassa,
        dehydration=dehydration,
        moderate_dehydration=moderate_dehydration,
        oliguria=oliguria,
        sunken_eyes_vomiting=sunken_eyes_vomiting,
        reasons=reasons,
        reminders=reminders,
    )


def apply_floor(level: TriageLevel | None, floor: TriageLevel | None) -> TriageLevel:
    """Final triage = the more urgent of the model's level and the rule floor.

    A missing level (model failed) fails safe to REFER_NOW.
    """
    if level is None:
        return TriageLevel.REFER_NOW
    return most_urgent(level, floor) or level
