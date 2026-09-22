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
