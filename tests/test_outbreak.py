"""Outbreak tool: domain restriction, per-state/day cache, extraction filters, degradation."""

from datetime import date
from pathlib import Path

import pytest

from backend.app.llm.spend import SpendLimitError
from backend.app.tools.outbreak import (
    TRUSTED_DOMAINS,
    ExtractedSignal,
    OutbreakTool,
    SearchResult,
    filter_signals,
)
from tests.pipeline_fakes import LASSA_URL, OUTBREAK_EXTRACTION, TODAY, make_deps


class RecordingSearch:
    source = "tavily"

    def __init__(self, results=None, error: Exception | None = None):
        self.results = (
            results
            if results is not None
            else [
                SearchResult(
                    title="Lassa",
                    url=LASSA_URL,
                    content="Lassa in Ondo",
                    published_date="2026-09-18",
                ),
                SearchResult(title="Bad", url="https://random-blog.example/lassa", content="x"),
            ]
        )
        self.error = error
        self.calls: list[tuple[str, tuple]] = []

    def search(self, query, *, include_domains):
        self.calls.append((query, tuple(include_domains)))
        if self.error:
            raise self.error
        return self.results


def tool_with(tmp_path: Path, search, replies=None, **overrides) -> tuple[OutbreakTool, object]:
    deps, llm = make_deps(
        tmp_path,
        replies or {"outbreak extraction": OUTBREAK_EXTRACTION},
        search=search,
        **overrides,
    )
    return deps.outbreak, llm


def test_no_state_is_unavailable(tmp_path: Path) -> None:
    tool, _ = tool_with(tmp_path, RecordingSearch())
    ctx = tool.check(None)
    assert ctx.status == "unavailable" and "No state" in ctx.message


def test_no_search_client_is_unavailable_not_silent(tmp_path: Path) -> None:
    tool, llm = tool_with(tmp_path, None)
    ctx = tool.check("Ondo")
    assert ctx.status == "unavailable"
    assert ctx.source == "none"
    assert "unavailable" in ctx.message
    assert llm.calls == []


def test_queries_and_domain_restriction(tmp_path: Path) -> None:
    search = RecordingSearch()
    tool, _ = tool_with(tmp_path, search)
    tool.check("Ondo")
    queries = [q for q, _ in search.calls]
    assert queries == [
        "Ondo Nigeria outbreak Lassa fever cholera meningitis September 2026",
        "NCDC situation report",
    ]
    assert all(domains == TRUSTED_DOMAINS for _, domains in search.calls)


def test_untrusted_results_never_reach_the_model(tmp_path: Path) -> None:
    tool, llm = tool_with(tmp_path, RecordingSearch())
    tool.check("Ondo")
    user = llm.messages_for("outbreak extraction")[1]["content"]
    assert LASSA_URL in user
    assert "random-blog" not in user


def test_extraction_runs_with_reasoning_off_on_fast_model(tmp_path: Path) -> None:
    tool, llm = tool_with(tmp_path, RecordingSearch())
    tool.check("Ondo")
    ((_, kwargs),) = llm.calls
    assert kwargs["model"] == "fast"
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_cache_per_state_and_day(tmp_path: Path) -> None:
    search = RecordingSearch()
    tool, llm = tool_with(tmp_path, search)
    first = tool.check("Ondo")
    second = tool.check("ondo state")  # same state, normalised
    assert second == first
    assert len(search.calls) == 2  # only the first check searched
    assert len(llm.calls) == 1
    assert "cache hit" in tool.last_note

    tool.check("Lagos")  # different state -> new search
    assert len(search.calls) == 4
    tool.today = lambda: date(2026, 9, 23)  # next day -> new search
    tool.check("Ondo")
    assert len(search.calls) == 6


def test_unavailable_results_are_not_cached(tmp_path: Path) -> None:
    search = RecordingSearch(error=ConnectionError("down"))
    tool, _ = tool_with(tmp_path, search)
    assert tool.check("Ondo").status == "unavailable"
    search.error = None
    assert tool.check("Ondo").status == "ok"


def test_search_failure_degrades(tmp_path: Path) -> None:
    tool, _ = tool_with(tmp_path, RecordingSearch(error=TimeoutError()))
    ctx = tool.check("Ondo")
    assert ctx.status == "unavailable"
    assert ctx.message.startswith("Live outbreak data unavailable right now.")
    assert "TimeoutError" not in ctx.message  # user sees plain text
    assert "TimeoutError" in tool.last_note  # detail goes to the decision trace


def test_extraction_failure_degrades(tmp_path: Path) -> None:
    tool, _ = tool_with(tmp_path, RecordingSearch(), {"outbreak extraction": ["bad", "bad"]})
    ctx = tool.check("Ondo")
    assert ctx.status == "unavailable"
    assert "LLMParseError" in tool.last_note and "LLMParseError" not in ctx.message


def test_no_results_is_ok_with_no_signals(tmp_path: Path) -> None:
    tool, llm = tool_with(tmp_path, RecordingSearch(results=[]))
    ctx = tool.check("Ondo")
    assert ctx.status == "ok" and ctx.signals == []
    assert llm.calls == []


def test_spend_limit_propagates(tmp_path: Path) -> None:
    tool, _ = tool_with(tmp_path, RecordingSearch(), max_spend_usd=0.0)
    with pytest.raises(SpendLimitError):
        tool.check("Ondo")


# --- filter -----------------------------------------------------------------

RESULTS = [SearchResult(title="t", url=LASSA_URL, content="c")]


@pytest.mark.parametrize(
    ("overrides", "kept"),
    [
        ({}, True),
        ({"url": None}, False),
        ({"report_date": None}, False),
        ({"report_date": "last week"}, False),
        ({"report_date": "2026-12-01"}, False),  # future
        ({"url": "https://ncdc.gov.ng/not-in-results"}, False),
        ({"report_date": "Week 37, published 2026-09-18"}, True),
    ],
)
def test_filter_signals(overrides: dict, kept: bool) -> None:
    sig = ExtractedSignal(
        **{
            "disease": "Lassa fever",
            "state": "Ondo",
            "status": "active",
            "report_date": "2026-09-18",
            "url": LASSA_URL,
            **overrides,
        }
    )
    signals, dropped = filter_signals([sig], RESULTS, TODAY)
    assert (len(signals) == 1) is kept
    assert dropped == (0 if kept else 1)


def test_untrusted_domain_dropped_even_if_in_results() -> None:
    url = "https://ncdc.gov.ng.evil.example/x"
    sig = ExtractedSignal(disease="Lassa", state="Ondo", report_date="2026-09-18", url=url)
    signals, _ = filter_signals([sig], [SearchResult(title="t", url=url, content="c")], TODAY)
    assert signals == []


def test_status_synonyms() -> None:
    assert ExtractedSignal(disease="x", state="y", status="Ongoing").status == "active"
    assert ExtractedSignal(disease="x", state="y", status="weird").status == "unknown"
