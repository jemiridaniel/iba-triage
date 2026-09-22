"""Deterministic advice: what a green ("Treat & monitor") result must always say, and
merging of rule and model advice that says the same thing.

Each rule action carries guideline anchors (doc_id, phrase) that rules_post resolves to the
exact indexed chunk for citation:
  WHO IMCI Chart Booklet (2014), fever: "Follow-up in 3 days if fever persists";
    "If fever is present every day for more than 7 days, refer for assessment";
    classification "Malaria test NEGATIVE".
  WHO Guidelines for Malaria, good practice statement: "All cases of suspected malaria should
    have a parasitological test (microscopy or RDT) to confirm the diagnosis."
"""

import re

from backend.app.rules.text import ascii_punct
from backend.app.schemas import ActionItem, Evidence, PatientCase, Reason

Anchor = tuple[str, str]

REVIEW_ANCHOR: Anchor = ("who-imci", "Follow-up in 3 days if fever persists")
PERSISTENT_FEVER_ANCHOR: Anchor = (
    "who-imci",
    "If fever is present every day for more than 7 days, refer for assessment",
)
NEGATIVE_TEST_ANCHOR: Anchor = ("who-imci", "Malaria test NEGATIVE")
TEST_FIRST_ANCHOR: Anchor = (
    "who-malaria",
    "All cases of suspected malaria should have a parasitological test",
)

REVIEW_TEXT = (
    "Review in 3 days if the fever persists. Return immediately if any danger sign appears: "
    "convulsions, unable to drink, vomiting everything, very sleepy or unconscious, bleeding, "
    "yellow eyes or dark urine."
)
NO_ANTIMALARIAL_TEXT = (
    "Do not give antimalarials: the malaria test is negative. Look for another cause of fever."
)
TEST_FIRST_TEXT = "Do a malaria RDT before giving any antimalarial (test before treating)."
NEXT_TEST_TEXT = (
    "If the fever persists, refer for further tests (for example typhoid testing or a full "
    "blood count) instead of treating presumptively. Refer for assessment if fever is present "
    "every day for more than 7 days."
)


def treat_monitor_advice(case: PatientCase) -> list[tuple[str, list[Anchor]]]:
    """Advice every green result must carry: review interval, test-before-treat, next test."""
    advice = [(REVIEW_TEXT, [REVIEW_ANCHOR])]
    if case.rdt_result == "negative":
        advice.append((NO_ANTIMALARIAL_TEXT, [NEGATIVE_TEST_ANCHOR, TEST_FIRST_ANCHOR]))
        advice.append((NEXT_TEST_TEXT, [PERSISTENT_FEVER_ANCHOR]))
    elif case.rdt_result in (None, "not_done"):
        advice.append((TEST_FIRST_TEXT, [TEST_FIRST_ANCHOR]))
    return advice


# --- merging ------------------------------------------------------------------

TOPICS: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in {
        "isolation": r"isolat|holding area|separate|infection prevention|\bipc\b|\bppe\b|glove"
        r"|barrier nursing|body fluids",
        "referral": r"\brefer|transfer|treatment cent|hospital",
        "notify": r"notif|surveillance|\bdsno\b|report to",
        "review": r"\breview|follow[- ]?up|return (?:immediately|if|at once)|come back",
        "no_antimalarial": r"(?:do not|don't|avoid|no)\b.{0,30}antimalari",
        "testing": r"\btest|\brdt\b|typhoid|blood count|laboratory",
    }.items()
}


def topics(text: str) -> set[str]:
    text = ascii_punct(text)
    return {name for name, rx in TOPICS.items() if rx.search(text)}


def merge_advice(actions: list[ActionItem]) -> tuple[list[ActionItem], int]:
    """Fold model actions whose every topic is already covered by rule actions into the
    best-matching rule action (rule wording wins; model text kept as a detail).

    Returns (actions, number merged). Model actions with a topic no rule covers, or with no
    recognised topic (e.g. "give paracetamol"), are kept as they are.
    """
    rules = [a for a in actions if a.source == "rule"]
    if not rules:
        return actions, 0
    rule_topics = [topics(a.text) for a in rules]
    covered = set().union(*rule_topics)
    kept: list[ActionItem] = []
    merged = 0
    for action in actions:
        if action.source == "rule":
            kept.append(action)
            continue
        found = topics(action.text)
        if not found or not found <= covered:
            kept.append(action)
            continue
        best = max(range(len(rules)), key=lambda i: len(found & rule_topics[i]))
        target = rules[best]
        target.details.append(
            Reason(text=action.text, evidence=action.evidence or Evidence(status="unsupported"))
        )
        target.citations = list(dict.fromkeys([*target.citations, *action.citations]))
        merged += 1
    return kept, merged
