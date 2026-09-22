"""Live outbreak context for the patient's state (SPEC §4, Tavily).

Flow: search trusted domains -> fast model (reasoning off) extracts OutbreakSignal[] ->
deterministic filter -> cached per (source, state, date).

Safety filters on extracted signals: must have a URL and a parseable, non-future date; the
URL must be one of the search results (no invented links) and on a trusted domain.

Degrades to status="unavailable" (never silently) when there is no API key, the search
fails, or extraction fails. SpendLimitError is never caught.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, field_validator

from backend.app.llm.client import LLMClient, LLMError
from backend.app.llm.router import EvalConfig, call_json
from backend.app.schemas import OutbreakContext, OutbreakSignal
from backend.app.tools.endemicity import AnchorResolver, Endemicity

logger = logging.getLogger("iba.outbreak")

TRUSTED_DOMAINS = ("ncdc.gov.ng", "who.int", "reliefweb.int", "afro.who.int")
# Fixed (not per case) so one cached result per state per day serves every case.
DISEASE_CANDIDATES = "Lassa fever cholera meningitis"
MAX_RESULT_CHARS = 1500


@dataclass
class SearchResult:
    title: str
    url: str
    content: str
    published_date: str | None = None


class SearchClient(Protocol):
    source: Literal["tavily", "mock"]

    def search(self, query: str, *, include_domains: Sequence[str]) -> list[SearchResult]: ...


class TavilySearch:
    source: Literal["tavily", "mock"] = "tavily"

    def __init__(self, api_key: str, max_results: int = 5):
        from tavily import TavilyClient

        self._client = TavilyClient(api_key=api_key)
        self.max_results = max_results

    def search(self, query: str, *, include_domains: Sequence[str]) -> list[SearchResult]:
        resp = self._client.search(
            query=query,
            search_depth="advanced",
            time_range="month",
            include_domains=list(include_domains),
            include_domains_mode="restrict",
            max_results=self.max_results,
        )
        return [
            SearchResult(
                title=r.get("title") or "",
                url=r.get("url") or "",
                content=r.get("content") or "",
                published_date=r.get("published_date"),
            )
            for r in resp.get("results", [])
        ]


class FixtureSearch:
    """Replays canned search results from a JSON file (tests, demos before Tavily credits)."""

    source: Literal["tavily", "mock"] = "mock"

    def __init__(self, path: Path):
        self.path = path
        self.queries: list[str] = []

    def search(self, query: str, *, include_domains: Sequence[str]) -> list[SearchResult]:
        self.queries.append(query)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return [SearchResult(**r) for r in data.get("results", [])]


# --- extraction schema ------------------------------------------------------


class ExtractedSignal(BaseModel):
    disease: str
    state: str
    status: Literal["active", "declining", "over", "unknown"] = "unknown"
    report_date: str | None = None
    url: str | None = None

    @field_validator("status", mode="before")
    @classmethod
    def _norm_status(cls, v: Any) -> Any:
        v = str(v or "unknown").strip().lower()
        synonyms = {
            "ongoing": "active",
            "confirmed": "active",
            "outbreak": "active",
            "controlled": "declining",
            "ended": "over",
            "resolved": "over",
        }
        return synonyms.get(v, v if v in {"active", "declining", "over"} else "unknown")


class OutbreakExtraction(BaseModel):
    signals: list[ExtractedSignal] = []


EXTRACT_SYSTEM = """Task: outbreak extraction.
You read search results from public-health sources and extract current disease outbreak \
signals in Nigeria. Reply with ONE JSON object and nothing else:
{"signals": [{"disease": str, "state": str, "status": "active"|"declining"|"over"|"unknown", \
"report_date": "YYYY-MM-DD" or null, "url": str}]}
Rules:
- Only include outbreaks explicitly reported in the text. Never infer or guess.
- state: the Nigerian state named in the text (one signal per disease per state).
- status "active" only if the text reports ongoing or recent cases.
- report_date: the date of the report or data, as stated; else the result's published date.
- url: copy the URL of the result the signal came from, exactly. Never invent a URL.
- If nothing qualifies, return {"signals": []}."""


def _is_trusted(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in TRUSTED_DOMAINS)


def _parse_date(value: str | None, today: date) -> date | None:
    if not value:
        return None
    match = re.search(r"\d{4}-\d{2}-\d{2}", value)
    if not match:
        return None
    try:
        parsed = date.fromisoformat(match.group(0))
    except ValueError:
        return None
    return parsed if parsed <= today else None


def _norm_state(state: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", re.sub(r"\s+state$", "", state.strip().lower())).strip("-")


def drop_reason(sig: ExtractedSignal, allowed: set[str], today: date) -> str | None:
    """Why this extracted signal can't be trusted, or None if it can be."""
    if not sig.url:
        return "no URL"
    if sig.url not in allowed:
        return "URL not among the search results (invented or altered)"
    if not _is_trusted(sig.url):
        return "URL is not on a trusted domain"
    if _parse_date(sig.report_date, today) is None:
        if not sig.report_date:
            return "no report date"
        if not re.search(r"\d{4}-\d{2}-\d{2}", sig.report_date):
            return f"unparseable report date {sig.report_date!r}"
        return f"report date {sig.report_date!r} is in the future"
    return None


def filter_signals(
    extracted: list[ExtractedSignal], results: list[SearchResult], today: date
) -> tuple[list[OutbreakSignal], list[tuple[ExtractedSignal, str]]]:
    """Keep only sourced, dated signals whose URL came from the search.

    Returns (kept, dropped) where each dropped signal carries the reason it failed.
    """
    allowed = {r.url for r in results}
    kept: list[OutbreakSignal] = []
    dropped: list[tuple[ExtractedSignal, str]] = []
    for sig in extracted:
        reason = drop_reason(sig, allowed, today)
        if reason is not None:
            dropped.append((sig, reason))
            continue
        kept.append(
            OutbreakSignal(
                disease=sig.disease,
                state=sig.state,
                status=sig.status,
                report_date=_parse_date(sig.report_date, today),  # type: ignore[arg-type]
                url=sig.url,
            )
        )
    return kept, dropped


class OutbreakTool:
    def __init__(
        self,
        client: LLMClient,
        search: SearchClient | None,
        cache_dir: Path,
        *,
        eval_config: EvalConfig = "routed",
        today: Callable[[], date] = date.today,
        endemicity: Endemicity | None = None,
        resolve: AnchorResolver | None = None,
    ):
        self.client = client
        self.search = search
        self.endemicity = endemicity or Endemicity([])
        self.resolve = resolve
        self.cache_dir = cache_dir / "outbreak"
        self.eval_config = eval_config
        self.today = today
        self.last_note: str | None = None  # for the decision trace
        # Last live check, for the trace and scripts/outbreak_trace.py (not persisted).
        self.last_queries: list[str] = []
        self.last_results: list[SearchResult] = []
        self.last_drops: list[tuple[ExtractedSignal, str]] = []

    def check(self, state: str | None) -> OutbreakContext:
        """Live signals (if search works) plus the static baseline, always."""
        context = self._check_live(state)
        baseline = self.endemicity.for_state(state, self.today(), self.resolve)
        if context.status == "unavailable" and baseline:
            context.message = f"{context.message} Using baseline endemicity only."
        return context.model_copy(update={"baseline": baseline})

    def _check_live(self, state: str | None) -> OutbreakContext:
        if not state:
            return self._unavailable("No state selected, so outbreak data was not checked.")
        if self.search is None:
            return self._unavailable("Live outbreak data unavailable: search is not configured.")

        today = self.today()
        cache_file = self.cache_dir / f"{self.search.source}_{_norm_state(state)}_{today}.json"
        if cache_file.exists():
            self.last_note = f"cache hit ({cache_file.name})"
            return OutbreakContext.model_validate_json(cache_file.read_text(encoding="utf-8"))

        queries = [
            f"{state} Nigeria outbreak {DISEASE_CANDIDATES} {today:%B %Y}",
            "NCDC situation report",
        ]
        self.last_queries, self.last_results, self.last_drops = queries, [], []
        try:
            results = self._search(queries)
            self.last_results = results
        except Exception as exc:  # network, auth, quota: degrade, never crash triage
            logger.warning("outbreak search failed: %s", type(exc).__name__)
            return self._unavailable(
                "Live outbreak data unavailable right now.",
                detail=f"search failed: {type(exc).__name__}",
            )

        signals: list[OutbreakSignal] = []
        if results:
            try:
                extraction, _ = call_json(
                    self.client,
                    "outbreak",
                    self._messages(results, state),
                    OutbreakExtraction,
                    self.eval_config,
                )
            except LLMError as exc:
                logger.warning("outbreak extraction failed: %s", type(exc).__name__)
                return self._unavailable(
                    "Live outbreak data unavailable right now.",
                    detail=f"extraction failed: {type(exc).__name__}",
                )
            signals, self.last_drops = filter_signals(extraction.signals, results, today)

        label = "MOCK test data" if self.search.source == "mock" else "trusted sources"
        context = OutbreakContext(
            status="ok",
            source=self.search.source,
            message=f"Checked {len(results)} results from {label} for {state}.",
            checked_at=datetime.now(UTC),
            signals=signals,
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(context.model_dump_json(), encoding="utf-8")
        self.last_note = (
            f"{self.search.source}: {len(queries)} searches, {len(results)} results, "
            f"{len(signals)} signals kept, {len(self.last_drops)} dropped"
        )
        return context

    def _search(self, queries: list[str]) -> list[SearchResult]:
        seen: dict[str, SearchResult] = {}
        for q in queries:
            for r in self.search.search(q, include_domains=TRUSTED_DOMAINS):  # type: ignore[union-attr]
                if r.url and _is_trusted(r.url) and r.url not in seen:
                    seen[r.url] = r
        return list(seen.values())

    def _messages(self, results: list[SearchResult], state: str) -> list[dict[str, str]]:
        blocks = [
            f"URL: {r.url}\nTitle: {r.title}\nPublished: {r.published_date or 'unknown'}\n"
            f"{r.content[:MAX_RESULT_CHARS]}"
            for r in results
        ]
        user = (
            f"Patient's state: {state}. Today: {self.today():%Y-%m-%d}.\n\n"
            "Search results:\n\n" + "\n\n---\n\n".join(blocks)
        )
        return [{"role": "system", "content": EXTRACT_SYSTEM}, {"role": "user", "content": user}]

    def _unavailable(self, message: str, detail: str | None = None) -> OutbreakContext:
        self.last_note = f"{message} ({detail})" if detail else message
        source = self.search.source if self.search is not None else "none"
        return OutbreakContext(status="unavailable", source=source, message=message)
