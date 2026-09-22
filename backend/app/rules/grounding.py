"""Deterministic quote verification: is the model's evidence quote really in the chunk?

Every guideline claim from the reasoning model carries (chunk_id, evidence_quote). A claim is
"verified" only if the quote (8-40 words) appears in that chunk's text after normalising
whitespace, dashes, quotes and case, exactly or with a fuzzy ratio >= 0.9 over a sliding window
of the same length. Otherwise the claim is kept but marked "unsupported" and shown as an AI
suggestion without a guideline source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from backend.app.rules.text import ascii_punct

MIN_WORDS = 8
MAX_WORDS = 40
FUZZY_THRESHOLD = 0.9

# Curly/low quotes and guillemets -> ASCII quotes; soft hyphen (U+00AD) removed.
_QUOTES = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u00ab": '"',
        "\u00bb": '"',
        "\u00ad": None,
    }
)
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?)])")


def normalise(text: str) -> str:
    text = ascii_punct(text).translate(_QUOTES).lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return text.strip(" \"'")


@dataclass
class QuoteCheck:
    verified: bool
    score: float  # 1.0 exact; otherwise best fuzzy ratio
    reason: str  # "exact" | "fuzzy" | "too_short" | "too_long" | "no_match" | "missing"


def verify_quote(quote: str | None, chunk_text: str) -> QuoteCheck:
    if not quote or not quote.strip():
        return QuoteCheck(False, 0.0, "missing")
    q = normalise(quote)
    words = q.split()
    if len(words) < MIN_WORDS:
        return QuoteCheck(False, 0.0, "too_short")
    if len(words) > MAX_WORDS:
        return QuoteCheck(False, 0.0, "too_long")
    hay = normalise(chunk_text)
    if q in hay:
        return QuoteCheck(True, 1.0, "exact")
    hay_words = hay.split()
    best = 0.0
    n = len(words)
    for size in (n - 1, n, n + 1):  # tolerate one dropped or added word
        if size < 1:
            continue
        for i in range(0, max(1, len(hay_words) - size + 1)):
            window = " ".join(hay_words[i : i + size])
            matcher = SequenceMatcher(None, q, window, autojunk=False)
            if (
                matcher.real_quick_ratio() < FUZZY_THRESHOLD
                or matcher.quick_ratio() < FUZZY_THRESHOLD
            ):
                continue
            best = max(best, matcher.ratio())
            if best >= 0.999:
                break
    if best >= FUZZY_THRESHOLD:
        return QuoteCheck(True, round(best, 3), "fuzzy")
    return QuoteCheck(False, round(best, 3), "no_match")


# --- patient facts ------------------------------------------------------------
#
# A reason labelled basis="patient" needs no guideline quote, so the label must be earned:
# its key terms have to appear (fuzzily) in the worker's words or the structured case.
# Otherwise it is treated as a clinical claim that needs a verified quote.

PATIENT_TERM_SHARE = 0.8  # share of key terms that must be found
TERM_FUZZ = 0.85  # per-term similarity (catches plurals and small misspellings)

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "had",
        "he",
        "her",
        "his",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "there",
        "this",
        "to",
        "was",
        "were",
        "which",
        "who",
        "with",
        "without",
        "no",
        "not",
        "nor",
        "any",
        "patient",
        "patients",
        "reported",
        "reports",
        "reporting",
        "presents",
        "presented",
        "presenting",
        "describes",
        "described",
        "history",
        "of",
        "current",
        "currently",
        "now",
        "since",
        "also",
        "very",
        "about",
    ]
)
_NUMBER_WORDS = {
    w: str(i)
    for i, w in enumerate(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
        ]
    )
}
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def _tokens(text: str) -> list[str]:
    words = _TOKEN.findall(normalise(text))
    return [_NUMBER_WORDS.get(w, w) for w in words]


def key_terms(text: str) -> list[str]:
    return [t for t in _tokens(text) if t not in _STOPWORDS and (len(t) > 2 or t.isdigit())]


def case_vocabulary(case) -> set[str]:
    """Words a patient fact may legitimately use: the worker's text plus derived case words."""
    words: list[str] = [case.raw_text or "", *case.symptoms]
    if case.age_years is not None:
        words.append(f"{case.age_years:g} years old aged age")
        if case.age_years < 1:
            words.append("infant baby child")
        elif case.age_years < 5:
            words.append("child toddler under five")
        elif case.age_years < 18:
            words.append("child adolescent young")
        else:
            words.append("adult")
    if case.sex:
        words.append("male man" if case.sex == "male" else "female woman")
    if case.pregnant:
        words.append("pregnant pregnancy")
    if case.fever_days is not None:
        words.append(f"fever {case.fever_days:g} days duration febrile")
    if case.temperature_c is not None:
        words.append(f"temperature {case.temperature_c:g}")
    if case.rdt_result:
        words.append(f"malaria rdt test {case.rdt_result.replace('_', ' ')} result")
    if case.antimalarial_taken:
        words.append("took taken antimalarial antimalarials treatment")
    if case.antimalarial_no_response or case.antibiotic_no_response:
        words.append("no response improvement improve failed despite")
    if case.antibiotic_taken:
        words.append("took taken antibiotic antibiotics treatment")
    if case.state:
        words.append(f"{case.state} state resident residence lives")
    words += [code.value.replace("_", " ") for code in case.danger_signs]
    return set(_tokens(" ".join(words)))


def _term_present(term: str, vocab: set[str]) -> bool:
    if term in vocab:
        return True
    if term.isdigit():
        return False
    return any(
        SequenceMatcher(None, term, v).ratio() >= TERM_FUZZ
        for v in vocab
        if abs(len(v) - len(term)) <= 3
    )


def patient_fact_supported(text: str, case) -> bool:
    terms = key_terms(text)
    if not terms:
        return False
    vocab = case_vocabulary(case)
    found = sum(_term_present(t, vocab) for t in terms)
    return found / len(terms) >= PATIENT_TERM_SHARE
