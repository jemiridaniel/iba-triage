"""Trace one live Tavily outbreak check end to end, and verify the safety filters.

    uv run python -m scripts.outbreak_trace                 # Ondo, full triage
    uv run python -m scripts.outbreak_trace --state Lagos --no-triage
    uv run python -m scripts.outbreak_trace --fresh         # ignore today's cache

Prints the exact queries sent, every result Tavily returned (title, URL, published date),
which results the domain filter kept, and every extracted signal with the reason it was kept
or dropped. Then checks, against the real API:

  * domain restriction  - every URL returned is on a TRUSTED_DOMAINS host
  * date parsing        - each kept signal's report_date parsed to a real, non-future date
  * cache               - a second identical check makes zero Tavily calls
  * cache key           - a different state writes its own file and does search

Spends Tavily credits (2 searches per uncached state) and a little Token Factory credit
(one extraction call on MODEL_FAST, plus the triage run unless --no-triage).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

from backend.app.config import get_settings
from backend.app.graph.pipeline import build_deps, run_triage
from backend.app.graph.state import TriageRequest
from backend.app.tools.outbreak import (
    TRUSTED_DOMAINS,
    OutbreakTool,
    SearchResult,
    TavilySearch,
    _is_trusted,
    _norm_state,
)

DEMOS = Path("frontend/src/demos.json")


class CountingSearch:
    """Wraps the real Tavily client: counts calls and keeps every raw result."""

    source = "tavily"

    def __init__(self, inner: TavilySearch):
        self.inner = inner
        self.calls: list[str] = []
        self.raw: list[tuple[str, list[SearchResult]]] = []

    def search(self, query: str, *, include_domains: Sequence[str]) -> list[SearchResult]:
        self.calls.append(query)
        results = self.inner.search(query, include_domains=include_domains)
        self.raw.append((query, results))
        return results


def demo_case(state: str) -> dict:
    cases = json.loads(DEMOS.read_text(encoding="utf-8"))
    return next((c for c in cases if c["state"] == state), {"state": state, "text": ""})


def show_search(counter: CountingSearch, tool: OutbreakTool) -> None:
    kept_urls = {r.url for r in tool.last_results}
    print(f"\nQUERIES SENT ({len(counter.calls)}, include_domains={list(TRUSTED_DOMAINS)}):")
    for q in counter.calls:
        print(f"  - {q!r}")

    total = sum(len(rs) for _, rs in counter.raw)
    print(f"\nRESULTS RETURNED BY TAVILY ({total} across {len(counter.raw)} queries):")
    seen: set[str] = set()
    for query, results in counter.raw:
        print(f"  query {query!r} -> {len(results)} results")
        for r in results:
            host = urlparse(r.url).hostname or "?"
            if not _is_trusted(r.url):
                verdict = "DROPPED: host not on a trusted domain"
            elif r.url in seen:
                verdict = "dropped: duplicate of an earlier query's result"
            elif r.url in kept_urls:
                verdict = "kept"
            else:
                verdict = "dropped"
            seen.add(r.url)
            print(f"    [{verdict}] {host}")
            print(f"      title:     {r.title[:110]}")
            print(f"      url:       {r.url}")
            print(f"      published: {r.published_date or '(none)'}")
    print(f"\n  -> {len(tool.last_results)} unique trusted results sent to the extractor")


def show_signals(tool: OutbreakTool, signals: list) -> None:
    print(f"\nSIGNALS KEPT ({len(signals)}):")
    for s in signals or []:
        print(f"  + {s.disease} / {s.state}: status={s.status}, report_date={s.report_date}")
        print(f"      {s.url}")
    if not signals:
        print("  (none)")
    print(f"\nSIGNALS DROPPED ({len(tool.last_drops)}):")
    for sig, reason in tool.last_drops:
        print(f"  - {sig.disease} / {sig.state} (status={sig.status}): {reason}")
        print(f"      url={sig.url} report_date={sig.report_date!r}")
    if not tool.last_drops:
        print("  (none)")


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace a live Tavily outbreak check.")
    parser.add_argument("--state", default="Ondo")
    parser.add_argument("--other-state", default="Lagos", help="second state, to test the key")
    parser.add_argument("--no-triage", action="store_true", help="outbreak step only")
    parser.add_argument("--fresh", action="store_true", help="delete today's cache file first")
    args = parser.parse_args()

    settings = get_settings()
    key = settings.tavily_api_key.get_secret_value() if settings.tavily_api_key else ""
    if not key:
        print("TAVILY_API_KEY is not set.")
        return 2
    if settings.outbreak_mock_file is not None:
        print(f"OUTBREAK_MOCK_FILE={settings.outbreak_mock_file} would shadow Tavily. Unset it.")
        return 2

    counter = CountingSearch(TavilySearch(key))
    deps = build_deps(settings, search=counter)
    tool = deps.outbreak
    today = date.today()
    cache = tool.cache_dir / f"tavily_{_norm_state(args.state)}_{today}.json"
    if args.fresh:
        cache.unlink(missing_ok=True)
    if cache.exists():
        print(f"NOTE: {cache} already exists; pass --fresh for a live search.")

    case = demo_case(args.state)
    print("=" * 78)
    print(f"LIVE OUTBREAK TRACE  state={args.state}  today={today}")
    print(f"case: {case['text'][:100] or '(outbreak step only)'}")
    print("=" * 78)

    if args.no_triage or not case["text"]:
        context = tool.check(args.state)
        result = None
    else:
        request = TriageRequest(text=case["text"], state=args.state, skip_questions=True)
        result = run_triage(request, deps, "routed")
        context = result.outbreak

    show_search(counter, tool)
    print(f"\nEXTRACTOR: {tool.last_note}")
    show_signals(tool, context.signals)
    print(f"\nCONTEXT: status={context.status} source={context.source}")
    print(f"  message: {context.message}")
    print(f"  baseline ({len(context.baseline)} static endemicity entries):")
    for b in context.baseline:
        print(f"    . {b.disease} / {b.state}: status={b.status}, in_season={b.in_season}")

    print("\n" + "=" * 78)
    print("VERIFICATION AGAINST THE REAL API")
    print("=" * 78)
    ok = True
    all_urls = [r.url for _, rs in counter.raw for r in rs]
    hosts = sorted({urlparse(u).hostname or "?" for u in all_urls})
    ok &= check(
        "domain restriction: every URL Tavily returned is on a trusted domain",
        all(_is_trusted(u) for u in all_urls),
        f"{len(all_urls)} URLs from {hosts}",
    )
    ok &= check(
        "date parsing: every kept signal has a real, non-future date",
        all(s.report_date <= today for s in context.signals),
        ", ".join(f"{s.disease}={s.report_date}" for s in context.signals) or "no signals",
    )
    ok &= check("cache file written", cache.exists(), str(cache))

    before = len(counter.calls)
    again = tool.check(args.state)
    ok &= check(
        "second identical check makes zero Tavily calls",
        len(counter.calls) == before,
        f"{len(counter.calls) - before} new calls; note={tool.last_note}",
    )
    ok &= check(
        "cached context is identical",
        [s.url for s in again.signals] == [s.url for s in context.signals],
    )

    other_cache = tool.cache_dir / f"tavily_{_norm_state(args.other_state)}_{today}.json"
    other_cache.unlink(missing_ok=True)
    before = len(counter.calls)
    tool.check(args.other_state)
    ok &= check(
        f"a different state ({args.other_state}) is a separate cache key and does search",
        len(counter.calls) > before and other_cache.exists(),
        f"{len(counter.calls) - before} new calls -> {other_cache.name}",
    )
    before = len(counter.calls)
    tool.check(args.other_state)
    ok &= check(f"{args.other_state} is then cached too", len(counter.calls) == before)

    if result is not None:
        print(f"\nTRIAGE: level={result.triage_level} status={result.status}")
        print(f"  lassa_suspected={result.lassa_suspected}")
        outbreak_cites = [c.url for d in result.differential for c in d.citations if c.url]
        print(f"  outbreak-sourced citations: {outbreak_cites or '(none)'}")
        print(f"  cost: ${result.decision_trace.total_cost_usd:.4f}")

    print(f"\nTavily searches used this run: {len(counter.calls)}")
    print("RESULT:", "all checks passed" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
