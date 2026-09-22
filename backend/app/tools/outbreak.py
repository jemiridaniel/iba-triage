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


def filter_signals(
    extracted: list[ExtractedSignal], results: list[SearchResult], today: date
) -> tuple[list[OutbreakSignal], int]:
    """Keep only sourced, dated signals whose URL came from the search. Returns (kept, dropped)."""
    allowed = {r.url for r in results}
    kept: list[OutbreakSignal] = []
    for sig in extracted:
        when = _parse_date(sig.report_date, today)
        if not sig.url or sig.url not in allowed or not _is_trusted(sig.url) or when is None:
            continue
        kept.append(
            OutbreakSignal(
                disease=sig.disease,
                state=sig.state,
                status=sig.status,
                report_date=when,
                url=sig.url,
            )
        )
    return kept, len(extracted) - len(kept)


class OutbreakTool:
    def __init__(
        self,
        client: LLMClient,
        search: SearchClient | None,
        cache_dir: Path,
        *,
        eval_config: EvalConfig = "routed",
        today: Callable[[], date] = date.today,
    ):
        self.client = client
        self.search = search
        self.cache_dir = cache_dir / "outbreak"
        self.eval_config = eval_config
        self.today = today
        self.last_note: str | None = None  # for the decision trace

    def check(self, state: str | None) -> OutbreakContext:
        if not state:
            return self._unavailable("No state selected, so outbreak data was not checked.")
        if self.search is None:
            return self._unavailable("Outbreak data unavailable: live search is not configured.")

        today = self.today()
        cache_file = self.cache_dir / f"{self.search.source}_{_norm_state(state)}_{today}.json"
        if cache_file.exists():
            self.last_note = f"cache hit ({cache_file.name})"
            return OutbreakContext.model_validate_json(cache_file.read_text(encoding="utf-8"))

        queries = [
            f"{state} Nigeria outbreak {DISEASE_CANDIDATES} {today:%B %Y}",
            "NCDC situation report",
        ]
        try:
            results = self._search(queries)
        except Exception as exc:  # network, auth, quota: degrade, never crash triage
            logger.warning("outbreak search failed: %s", type(exc).__name__)
            return self._unavailable(
                f"Outbreak data unavailable: search failed ({type(exc).__name__})."
            )

        signals: list[OutbreakSignal] = []
        dropped = 0
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
                    "Outbreak data unavailable: could not read search results."
                )
            signals, dropped = filter_signals(extraction.signals, results, today)

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
            f"{len(signals)} signals kept, {dropped} dropped"
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

    def _unavailable(self, message: str) -> OutbreakContext:
        self.last_note = message
        source = self.search.source if self.search is not None else "none"
        return OutbreakContext(status="unavailable", source=source, message=message)
