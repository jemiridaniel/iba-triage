"""Run the triage pipeline over the synthetic vignettes (SPEC §6).

    uv run python -m eval.run_eval --config routed --limit 10 --stratified
    uv run python -m eval.run_eval --config routed --category suspected_lassa --outbreak off
    uv run python -m eval.run_eval --config routed --category suspected_lassa --outbreak mock
    uv run python -m eval.run_eval --config routed --limit 10 --stratified --judge

--outbreak:
    case  per-vignette setting (mock live signal only where the vignette says "mock_live")
    off   no live outbreak search for any case (static baseline only)
    mock  a mock live Lassa signal for every case's state (measures Tavily lift without Tavily)

Results are appended to eval/results/<config>__<outbreak>.jsonl, one line per case; re-running
skips cases already in the file (resumable). LLM calls go through the disk cache, so repeated
runs of the same case and config are free. Spend is capped by MAX_SPEND_USD, and the script
asks before starting if the estimated cost exceeds $0.50.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.app.config import get_settings
from backend.app.graph.pipeline import build_deps, run_triage
from backend.app.graph.state import Deps, TriageRequest
from backend.app.llm.router import THINKING_OFF_KWARGS, EvalConfig
from backend.app.llm.spend import SpendLimitError
from backend.app.schemas import TriageResult
from backend.app.tools.outbreak import SearchResult

VIGNETTES = Path("eval/vignettes.jsonl")
RESULTS = Path("eval/results")
MOCK_TEMPLATE = Path("data/mock/outbreak_lassa_template.json")
CONFIRM_ABOVE_USD = 0.50

# Rough per-case token use by step (prompt, completion), from live runs on 2026-09-22.
STEP_TOKENS = {
    "intake": (900, 150),
    "outbreak": (700, 120),
    "reason": (3300, 2500),
    "compose": (650, 250),
}
STEP_MODEL = {
    "routed": {
        "intake": "model_fast",
        "outbreak": "model_fast",
        "reason": "model_reason",
        "compose": "model_mid",
    },
    "reason-only": dict.fromkeys(STEP_TOKENS, "model_reason"),
    "fast-only": dict.fromkeys(STEP_TOKENS, "model_fast"),
}


def effective_gold(case: dict, live_signal: bool) -> dict:
    """Gold labels for the mode the case ran in: `gold_live` overrides apply with a live signal."""
    return {**case["gold"], **(case.get("gold_live", {}) if live_signal else {})}


def load_vignettes(path: Path = VIGNETTES) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def select(
    cases: list[dict],
    category: str | None,
    ids: list[str] | None,
    limit: int | None,
    stratified: bool,
) -> list[dict]:
    if category:
        cases = [c for c in cases if c["category"] == category]
    if ids:
        wanted = set(ids)
        cases = [c for c in cases if c["id"] in wanted]
    if stratified:  # round-robin across categories so a small run covers the mix
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for c in cases:
            by_cat[c["category"]].append(c)
        queues = list(by_cat.values())
        mixed: list[dict] = []
        while any(queues):
            for q in queues:
                if q:
                    mixed.append(q.pop(0))
        cases = mixed
    return cases[:limit] if limit else cases


class TemplateSearch:
    """Mock live search: one Lassa sitrep for the case's state (from a template)."""

    source = "mock"

    def __init__(self, template: Path, state: str):
        slug = re.sub(r"[^a-z0-9]+", "-", state.lower()).strip("-")
        raw = template.read_text(encoding="utf-8").replace("{state}", state).replace("{slug}", slug)
        self.results = [SearchResult(**r) for r in json.loads(raw)["results"]]

    def search(self, query: str, *, include_domains) -> list[SearchResult]:
        return self.results


def outbreak_search(mode: str, case: dict, template: Path):
    live = mode == "mock" or (mode == "case" and case.get("outbreak", "").startswith("mock_live"))
    return TemplateSearch(template, case["state"]) if (live and case.get("state")) else None


def estimate_cost(settings, config: str, n: int) -> float:
    per_case = 0.0
    for step, (pt, ct) in STEP_TOKENS.items():
        model = getattr(settings, STEP_MODEL[config][step])
        price = settings.model_prices.get(model)
        if price:
            per_case += (pt * price.input + ct * price.output) / 1_000_000
    return per_case * n


_DROPPED = re.compile(r"dropped_citations=(\d+)")


def summarise(
    case: dict, result: TriageResult, config: str, mode: str, wall_ms: float, store=None
) -> dict:
    steps = result.decision_trace.steps
    rules_post = next((s for s in steps if s.step == "rules_post"), None)
    dropped = (
        int(m.group(1))
        if rules_post and rules_post.note and (m := _DROPPED.search(rules_post.note))
        else 0
    )
    kept = sum(len(d.citations) for d in result.differential if d.source == "llm") + sum(
        len(a.citations) for a in result.actions if a.source == "llm"
    )
    model_ms = sum(
        (s.model_latency_ms if s.cached and s.model_latency_ms else s.latency_ms) for s in steps
    )
    # One claim per guideline-basis reason / model action, with its grounding status.
    claims = [
        {
            "claim": f"{d.condition}: {r.text}",
            "refs": [r.evidence.chunk_id] if r.evidence.chunk_id else [],
            "status": r.evidence.status,
        }
        for d in result.differential
        if d.source == "llm"
        for r in d.reasons
        if r.evidence.status in ("verified", "unsupported")
    ]
    # Model actions, including those merged into rule actions as details.
    action_claims = [(a.text, a.evidence) for a in result.actions if a.source == "llm"]
    action_claims += [
        (d.text, d.evidence) for a in result.actions if a.source == "rule" for d in a.details
    ]
    claims += [
        {
            "claim": text,
            "refs": [ev.chunk_id] if ev and ev.chunk_id else [],
            "status": ev.status if ev else "unsupported",
        }
        for text, ev in action_claims
    ]

    def title(chunk_id: str) -> str:
        chunk = store.get(chunk_id) if store else None
        return f"{chunk_id} | {chunk.section} (p.{chunk.page})" if chunk else chunk_id

    live = mode == "mock" or (mode == "case" and case.get("outbreak", "").startswith("mock_live"))
    return {
        "id": case["id"],
        "category": case["category"],
        "config": config,
        "outbreak_mode": mode,
        "live_signal": live,
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "gold": effective_gold(case, live),
        "status": result.status,
        "triage_level": result.triage_level.value if result.triage_level else None,
        "triage_rationale": result.triage_rationale,
        "danger_signs": [{"code": h.code.value, "source": h.source} for h in result.danger_signs],
        "differential": [d.condition for d in result.differential],
        "lassa_suspected": result.lassa_suspected,
        "citations_kept": kept,
        "citations_dropped": dropped,
        "claims": claims,
        "grounding": result.grounding.model_dump(),
        "retrieval": [
            {
                "condition": q.condition,
                "purpose": q.purpose,
                "query": q.query,
                "top": [title(h) for h in q.hits[:5]],
            }
            for q in result.retrieval
        ],
        "passages": sorted({h for q in result.retrieval for h in q.hits}),
        "warnings": result.warnings,
        "latency_ms": round(model_ms, 1),
        "wall_ms": round(wall_ms, 1),
        "reason_fallback": any("fallback after truncation" in (s.note or "") for s in steps),
        "reason_step": next(
            (
                {
                    "model": s.model,
                    "reasoning": s.reasoning,
                    "calls": s.calls,
                    "prompt_tokens": s.prompt_tokens,
                    "completion_tokens": s.completion_tokens,
                    "reasoning_tokens": s.reasoning_tokens,
                    "latency_ms": s.model_latency_ms if s.cached else s.latency_ms,
                    "cost_usd": s.cost_usd,
                }
                for s in steps
                if s.step == "reason"
            ),
            None,
        ),
        "cost_usd": result.decision_trace.total_cost_usd,
        "tokens": result.decision_trace.total_tokens,
        "cached_steps": sum(1 for s in steps if s.cached),
        "error": None,
    }


def run_case(deps: Deps, case: dict, config: EvalConfig, mode: str, template: Path) -> dict:
    deps.outbreak.search = outbreak_search(mode, case, template)
    request = TriageRequest(text=case["text"], state=case.get("state"), skip_questions=True)
    start = time.perf_counter()
    result = run_triage(request, deps, config)
    return summarise(case, result, config, mode, (time.perf_counter() - start) * 1000, deps.store)


# --- LLM judge: does the cited chunk support the claim? --------------------------

JUDGE_SYSTEM = """Task: citation judge.
You check whether a guideline passage supports a clinical claim made by a triage assistant.
Reply with ONE JSON object: {"verdict": "supported" | "partial" | "unsupported", "reason": str}
- supported: the passage clearly states or directly implies the claim.
- partial: the passage is relevant but supports only part of the claim.
- unsupported: the passage does not support the claim.
Claims may mix patient facts (age, days of fever, test results, state) with clinical content
(criteria, classification, recommended action). Patient facts are given; do NOT require the
passage to state them. Judge only whether the passage supports the clinical content.
Judge only against the passage text. Keep the reason to one sentence."""


def judge_sample(records: list[dict], fraction: float = 0.2) -> list[dict]:
    if fraction >= 1.0:
        return [r for r in records if r.get("claims")]
    """Deterministic ~20% sample of cases (by id hash) that have claims."""
    with_claims = [r for r in records if r.get("claims")]
    ranked = sorted(with_claims, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
    return ranked[: max(1, math.ceil(len(with_claims) * fraction))] if with_claims else []


def run_judge(
    deps: Deps,
    records: list[dict],
    out: Path,
    fraction: float = 0.2,
    max_pairs_per_case: int | None = 6,
) -> list[dict]:
    """Judge claim/chunk pairs on a sample of cases (all with fraction=1.0, uncapped).

    One failed judgement is recorded as judge_error, never fatal. Claims with no chunk at all
    (no evidence given) can't be judged against text; the report counts them separately.
    """
    from pydantic import BaseModel

    from backend.app.llm.client import LLMError

    class Verdict(BaseModel):
        verdict: str
        reason: str = ""

    rows = []
    for rec in judge_sample(records, fraction):
        pairs = [(c["claim"], ref, c.get("status")) for c in rec["claims"] for ref in c["refs"]]
        for claim, ref, status in pairs[:max_pairs_per_case]:
            chunk = deps.store.get(ref) if deps.store else None
            if chunk is None:  # outbreak URLs aren't judged against text
                continue
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": f"CLAIM:\n{claim}\n\nPASSAGE [{ref}]:\n{chunk.text[:3400]}",
                },
            ]
            try:
                verdict, _ = deps.client.chat_json(
                    messages,
                    Verdict,
                    model=deps.settings.model_mid,
                    step="judge",
                    max_tokens=300,
                    extra_body=THINKING_OFF_KWARGS,
                )
                result = {"verdict": verdict.verdict.lower(), "reason": verdict.reason}
            except LLMError as exc:
                result = {"verdict": "judge_error", "reason": type(exc).__name__}
            row = {
                "id": rec["id"],
                "config": rec["config"],
                "outbreak_mode": rec["outbreak_mode"],
                "claim": claim,
                "ref": ref,
                "grounding_status": status,
                **result,
            }
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Ibà eval over synthetic vignettes.")
    parser.add_argument(
        "--config", choices=["routed", "reason-only", "fast-only"], default="routed"
    )
    parser.add_argument("--outbreak", choices=["case", "off", "mock"], default="case")
    parser.add_argument("--mock-template", type=Path, default=MOCK_TEMPLATE)
    parser.add_argument("--category")
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stratified", action="store_true", help="round-robin across categories")
    parser.add_argument(
        "--judge", action="store_true", help="LLM-judge citation support on ~20%% of cases"
    )
    parser.add_argument(
        "--judge-only", action="store_true", help="judge existing results; run no cases"
    )
    parser.add_argument(
        "--judge-fraction", type=float, default=0.2, help="share of cases to judge (1.0 = all)"
    )
    parser.add_argument("--yes", action="store_true", help="don't ask before an expensive run")
    parser.add_argument(
        "--no-cache", action="store_true", help="live calls only (for honest latency numbers)"
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    settings = get_settings().model_copy(update={"cache_enabled": not args.no_cache})
    out = args.out or RESULTS / f"{args.config}__{args.outbreak}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.judge_only:
        existing = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        judge_out = out.with_suffix(".judge.jsonl")
        judge_out.unlink(missing_ok=True)  # re-judge from scratch (cached calls are free)
        ok = [r for r in existing if not r.get("error")]
        cap = None if args.judge_fraction >= 1.0 else 6
        judged = run_judge(build_deps(settings), ok, judge_out, args.judge_fraction, cap)
        print(f"Judged {len(judged)} claim-citation pairs -> {judge_out}")
        return 0
    done = (
        {json.loads(line)["id"] for line in out.read_text().splitlines() if line.strip()}
        if out.exists()
        else set()
    )
    cases = [
        c
        for c in select(load_vignettes(), args.category, args.ids, args.limit, args.stratified)
        if c["id"] not in done
    ]

    estimate = estimate_cost(settings, args.config, len(cases))
    print(
        f"{len(cases)} cases to run ({len(done)} already in {out}); config={args.config}, "
        f"outbreak={args.outbreak}; estimated cost ≤ ${estimate:.3f} before cache hits"
    )
    if estimate > CONFIRM_ABOVE_USD and not args.yes:
        if not sys.stdin.isatty():
            print("Estimated cost is over $0.50; re-run with --yes to confirm.")
            return 2
        if input("Estimated cost is over $0.50. Continue? [y/N] ").strip().lower() != "y":
            return 1

    deps = build_deps(settings)
    records: list[dict] = []
    for i, case in enumerate(cases, 1):
        try:
            rec = run_case(deps, case, args.config, args.outbreak, args.mock_template)
        except SpendLimitError as exc:
            print(f"Stopped by spend guard: {exc}")
            break
        except Exception as exc:  # record and continue; never silently drop a case
            rec = {
                "id": case["id"],
                "category": case["category"],
                "config": args.config,
                "outbreak_mode": args.outbreak,
                "gold": case["gold"],
                "error": f"{type(exc).__name__}: {exc}",
            }
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        records.append(rec)
        print(
            f"[{i}/{len(cases)}] {case['id']:<8} gold={case['gold']['triage_level']:<13} "
            f"pred={rec.get('triage_level')!s:<13} ${rec.get('cost_usd') or 0:.4f} "
            f"{(rec.get('latency_ms') or 0) / 1000:.1f}s"
            + (f"  ERROR {rec['error']}" if rec.get("error") else "")
        )

    if args.judge and records:
        cap = None if args.judge_fraction >= 1.0 else 6
        ok = [r for r in records if not r.get("error")]
        judged = run_judge(deps, ok, out.with_suffix(".judge.jsonl"), args.judge_fraction, cap)
        print(f"Judged {len(judged)} claim-citation pairs -> {out.with_suffix('.judge.jsonl')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
