"""Report-date extraction (FEEDBACK T1). Pure Python, no network."""

from datetime import date

import pytest

from backend.app.schemas import OutbreakSignal
from backend.app.tools.dates import (
    STALE_DAYS,
    epi_week_end,
    explicit_dates,
    recency,
    resolve_report_date,
)
from backend.app.tools.outbreak import ExtractedSignal, SearchResult, filter_signals

TODAY = date(2026, 9, 22)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Data as of 2026-08-23", date(2026, 8, 23)),
        ("published 23 August 2026", date(2026, 8, 23)),
        ("published 23rd August 2026", date(2026, 8, 23)),
        ("August 23, 2026", date(2026, 8, 23)),
        ("week ending 23/08/2026", date(2026, 8, 23)),
        ("Epi Week 34, 2026", date(2026, 8, 23)),  # ISO week 34 of 2026 ends Sunday 23 Aug
        (
            "NCDC Lassa Fever Situation Report Epi Week 34: 17th - 23rd August 2026",
            date(2026, 8, 23),
        ),
    ],
)
def test_explicit_patterns(text: str, expected: date) -> None:
    assert expected in explicit_dates(text, TODAY)


def test_dates_are_read_out_of_a_url_slug() -> None:
    url = (
        "https://reliefweb.int/report/nigeria/"
        "ncdc-lassa-fever-situation-report-epi-week-34-17th-23rd-august-2026"
    )
    assert date(2026, 8, 23) in explicit_dates(url, TODAY)


def test_future_dates_are_ignored() -> None:
    assert explicit_dates("planned for 2027-01-05 and 2026-12-01", TODAY) == []


def test_no_date_at_all() -> None:
    assert explicit_dates("Nigeria Centre for Disease Control and Prevention", TODAY) == []
    assert explicit_dates("", TODAY) == []


def test_epi_week_end_is_the_sunday() -> None:
    assert epi_week_end(2026, 34) == date(2026, 8, 23)
    assert epi_week_end(2026, 0) is None and epi_week_end(2026, 54) is None


# --- resolve_report_date ----------------------------------------------------


def test_the_real_lagos_failure_is_fixed() -> None:
    """2026-09-22 live run: the model dated a month-old sitrep "today" for all 5 signals.

    The result text and URL both say epi week 34 (17-23 August), so the text must win.
    """
    title = "NCDC Lassa Fever Situation Report Epi Week 34: 17th - 23rd August 2026 - Nigeria"
    url = (
        "https://reliefweb.int/report/nigeria/"
        "ncdc-lassa-fever-situation-report-epi-week-34-17th-23rd-august-2026"
    )
    when, basis = resolve_report_date(TODAY, [title, "", "", url], TODAY)
    assert when == date(2026, 8, 23) and basis == "text"
    assert recency(when, TODAY) == "current"  # 30 days old: still inside the 60-day window


def test_model_date_is_used_only_when_it_is_older() -> None:
    text = "Data as of 2026-09-20"
    assert resolve_report_date(date(2026, 9, 1), [text], TODAY)[0] == date(2026, 9, 1)
    # A newer model date can never override the text.
    assert resolve_report_date(TODAY, [text], TODAY)[0] == date(2026, 9, 20)


def test_no_explicit_date_anywhere_is_unknown() -> None:
    when, basis = resolve_report_date(TODAY, ["Nigeria Centre for Disease Control"], TODAY)
    assert when is None and basis == "unknown"


# --- staleness --------------------------------------------------------------


def test_recency_bands() -> None:
    assert recency(None, TODAY) == "unknown"
    assert recency(TODAY, TODAY) == "current"
    assert recency(date(2026, 7, 25), TODAY) == "current"  # 59 days
    assert recency(date(2026, 7, 23), TODAY) == "older"  # 61 days
    assert STALE_DAYS == 60


def test_an_older_or_undated_signal_cannot_escalate() -> None:
    from backend.app.rules.danger_signs import active_outbreak

    def signal(**kw) -> OutbreakSignal:
        fields = {
            "disease": "Lassa fever",
            "state": "Ondo",
            "status": "active",
            "url": "https://ncdc.gov.ng/x.pdf",
            "report_date": TODAY,
            **kw,
        }
        return OutbreakSignal(**fields)

    assert active_outbreak([signal()], "Ondo", "lassa") is not None
    assert active_outbreak([signal(recency="older")], "Ondo", "lassa") is None
    assert active_outbreak([signal(recency="unknown", report_date=None)], "Ondo", "lassa") is None
    # T3: an outbreak in another state never escalates this patient.
    assert active_outbreak([signal(state="Edo")], "Ondo", "lassa") is None


# --- end to end through the filter ------------------------------------------


def test_filter_dates_a_signal_from_the_result_text_not_the_model() -> None:
    result = SearchResult(
        title="NCDC Lassa Fever Situation Report Epi Week 34: 17th - 23rd August 2026",
        url="https://ncdc.gov.ng/sitrep.pdf",
        content="Confirmed cases reported in Ondo State.",
    )
    sig = ExtractedSignal(
        disease="Lassa fever",
        state="Ondo",
        status="active",
        report_date=TODAY.isoformat(),  # the model's wrong answer
        url=result.url,
    )
    (kept,), dropped = filter_signals([sig], [result], TODAY)
    assert dropped == []
    assert kept.report_date == date(2026, 8, 23) and kept.recency == "current"


def test_filter_marks_an_undated_signal_unknown_rather_than_dropping_it() -> None:
    result = SearchResult(
        title="Nigeria Centre for Disease Control and Prevention",
        url="https://ncdc.gov.ng/news/541/invitation-to-tender",
        content="No dates here.",
    )
    sig = ExtractedSignal(disease="Lassa fever", state="Ondo", status="active", url=result.url)
    (kept,), dropped = filter_signals([sig], [result], TODAY)
    assert dropped == []
    assert kept.report_date is None and kept.recency == "unknown"


def test_filter_marks_a_stale_signal_older() -> None:
    result = SearchResult(
        title="Lassa Fever Situation Report, data as of 2026-05-10",
        url="https://ncdc.gov.ng/sitrep-old.pdf",
        content="Ondo State.",
    )
    sig = ExtractedSignal(disease="Lassa fever", state="Ondo", status="active", url=result.url)
    (kept,), _ = filter_signals([sig], [result], TODAY)
    assert kept.report_date == date(2026, 5, 10) and kept.recency == "older"
