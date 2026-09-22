"""Text normalisation shared by the deterministic rules.

Models (Super in particular) and phone keyboards emit Unicode dashes and spaces, e.g.
"follow‑up" with U+2011 NON-BREAKING HYPHEN, or "10–20 mg" with an en dash. Rules match
ASCII, so normalise before matching. Only used for matching; displayed text is unchanged
except where a rule rewrites it (dose stripping).
"""

_DASHES = "‐‑‒–—―−﹣－"
_SPACES = "     "
_TABLE = str.maketrans({**dict.fromkeys(_DASHES, "-"), **dict.fromkeys(_SPACES, " ")})


def ascii_punct(text: str) -> str:
    return text.translate(_TABLE)
