"""Report dates for outbreak signals, taken from the text rather than from the model.

Why this exists: Tavily returns no `published_date` on any result we have seen (FEEDBACK T1),
so the only date available is whatever appears in the page text or URL. Asking the model for it
is not safe — on 2026-09-22 a month-old NCDC sitrep came back dated "today" for all five Lagos
signals, which would make a stale report read as current.

So we extract dates ourselves with explicit patterns, and treat the model's date as a last
resort that can only make a signal *older*, never newer. No explicit date anywhere means
`None`: unknown, which downstream must not treat as current.

Recognised, in text or in a URL slug:
    ISO                 2026-08-23
    epi week            "Epi Week 34: ..." / "epi-week-34" (+ a year) -> that week's Sunday
    week ending         "week ending 23 August 2026", "week ending 23/08/2026"
    long form           "23 August 2026", "23rd August 2026", "August 23, 2026"
"""

from __future__ import annotations

import re
from datetime import date

MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ],
        start=1,
    )
    for m in names
}
MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))

ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
# "23 August 2026", "23rd Aug 2026" - the year may be a few words later ("17th - 23rd Aug 2026")
DAY_MONTH_YEAR = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?[\s\-]+({MONTH_RE})[\s\-,]+(\d{{4}})\b", re.I
)
# "August 23, 2026"
MONTH_DAY_YEAR = re.compile(
    rf"\b({MONTH_RE})[\s\-]+(\d{{1,2}})(?:st|nd|rd|th)?[\s\-,]+(\d{{4}})\b", re.I
)
# "week ending 23/08/2026" (day first: these are Nigerian and WHO sources)
WEEK_ENDING_NUMERIC = re.compile(
    r"week[\s\-]+ending[\s\-:]*(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", re.I
)
EPI_WEEK = re.compile(r"(?:epi(?:demiological)?[\s\-]*week|week)[\s\-:]*(\d{1,2})\b", re.I)
YEAR = re.compile(r"\b(20\d{2})\b")

STALE_DAYS = 60  # older than this is an "older report": informs, never escalates on its own


def _slug_to_text(value: str) -> str:
    """URL slugs carry dates too: .../epi-week-34-17th-23rd-august-2026 -> spaced words."""
    return re.sub(r"[-_/+]+", " ", value)


def _safe(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def epi_week_end(year: int, week: int) -> date | None:
    """Last day (Sunday) of an ISO week, which is how NCDC sitreps are dated."""
    if not 1 <= week <= 53:
        return None
    try:
        return date.fromisocalendar(year, week, 7)
    except ValueError:
        return None


def _scan(text: str, today: date) -> list[date | None]:
    found: list[date | None] = []
    for y, m, d in ISO.findall(text):
        found.append(_safe(int(y), int(m), int(d)))
    for d, mon, y in DAY_MONTH_YEAR.findall(text):
        found.append(_safe(int(y), MONTHS[mon.lower()], int(d)))
    for mon, d, y in MONTH_DAY_YEAR.findall(text):
        found.append(_safe(int(y), MONTHS[mon.lower()], int(d)))
    for d, m, y in WEEK_ENDING_NUMERIC.findall(text):
        found.append(_safe(int(y), int(m), int(d)))

    # "Epi Week 34" needs a year from somewhere nearby; the document's own year will do.
    weeks = [int(w) for w in EPI_WEEK.findall(text)]
    years = [int(y) for y in YEAR.findall(text) if int(y) <= today.year]
    if weeks and years:
        found += [epi_week_end(max(years), week) for week in weeks]
    return found


def explicit_dates(text: str, today: date) -> list[date]:
    """Every date the text states outright, ignoring future ones. Order is not meaningful.

    Scanned twice: as given, and with URL punctuation turned into spaces. A slug hides
    "23rd-august-2026"; normalising it would in turn hide "2026-08-23", so we do both.
    """
    if not text:
        return []
    found = _scan(text, today)
    slug = _slug_to_text(text)
    if slug != text:
        found += _scan(slug, today)
    return [d for d in found if d is not None and d <= today]


def resolve_report_date(
    model_date: date | None, sources: list[str], today: date
) -> tuple[date | None, str]:
    """The date to trust for a signal, and where it came from.

    Explicit dates in the text or URL win. The model's date is used only when it is *older*
    than what the text says, so a model that answers "today" can never make a report current.
    Nothing explicit anywhere -> (None, "unknown").
    """
    found: list[date] = []
    for source in sources:
        found += explicit_dates(source or "", today)
    if not found:
        return None, "unknown"
    # The most recent explicit date is the report's own reporting date; an older model date
    # is still honoured, since being wrong-old only ever suppresses an escalation.
    best = max(found)
    if model_date is not None and model_date < best:
        return model_date, "model (older than the text)"
    return best, "text"


def recency(report_date: date | None, today: date) -> str:
    """'unknown' | 'older' | 'current'. Only 'current' may escalate triage on its own."""
    if report_date is None:
        return "unknown"
    return "older" if (today - report_date).days > STALE_DAYS else "current"
