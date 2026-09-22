"""Run the triage pipeline from the terminal.

    uv run python -m backend.app.cli "Pikin 3 years, hot body 4 days ..." --state Ondo
    uv run python -m backend.app.cli "..." --answer fever_days=5 --answer rdt_result=negative
    uv run python -m backend.app.cli "..." --state Ondo \
        --mock-outbreak data/mock/outbreak_ondo_lassa.json --index data/index_fake
    uv run python -m backend.app.cli "..." --json

Spends credits on live runs (guarded by MAX_SPEND_USD).
"""

import argparse
import json
import sys
from pathlib import Path

from backend.app.config import get_settings
from backend.app.graph.pipeline import build_deps, run_triage
from backend.app.graph.state import TriageRequest
from backend.app.llm.spend import SpendLimitError
from backend.app.schemas import FollowUpAnswer, TriageResult

COLOURS = {"refer_now": "\033[31m", "refer_24h": "\033[33m", "treat_monitor": "\033[32m"}


def print_result(r: TriageResult, colour: bool) -> None:
    def c(text: str, code: str) -> str:
        return f"{code}{text}\033[0m" if colour else text

    if r.status == "needs_info":
        print("Iba needs more information. Re-run with --answer <id>=<answer>:")
        for q in r.questions:
            print(f"  {q.id}: {q.text}")
        if r.danger_signs:
            print("Danger signs so far: " + ", ".join(h.label for h in r.danger_signs))
    else:
        level = r.triage_level.value if r.triage_level else ""
        print(c(f"== {r.triage_label} ==", COLOURS.get(level, "")) + f"   ({r.status})")
        print(r.triage_rationale or "")
        if r.danger_signs:
            print("\nDanger signs:")
            for h in r.danger_signs:
                print(f"  - {h.label} [{h.source}]" + (f' "{h.evidence}"' if h.evidence else ""))
        if r.outbreak:
            print(f"\nOutbreak ({r.outbreak.source}, {r.outbreak.status}): {r.outbreak.message}")
            for s in r.outbreak.signals:
                print(f"  - {s.disease} in {s.state}: {s.status}, {s.report_date} {s.url}")
        print("\nDifferential:")
        for d in r.differential:
            cites = f"  [{', '.join(d.citations)}]" if d.citations else ""
            print(f"  - {d.condition} ({d.likelihood}, {d.source}){cites}")
            for reason in d.reasons:
                print(f"      because: {reason}")
        print("\nActions:")
        for a in r.actions:
            print(f"  - {a.text}" + (f"  [{', '.join(a.citations)}]" if a.citations else ""))
        for d in r.doses:
            flag = "verified" if d.verified else "UNVERIFIED"
            print(f"  - DOSE ({flag}): {d.drug}, {d.weight_band}: {d.regimen}")
        if r.summary:
            print(f"\nSummary: {r.summary}")
        if r.referral_note:
            print(f"\nReferral note:\n{r.referral_note}")
        if r.warnings:
            print("\nWarnings:")
            for w in r.warnings:
                print(f"  ! {w}")

    print("\nDecision trace:")
    header = ("step", "model", "reason", "in", "out", "think", "ms", "cost $")
    print("  {:<11} {:<40} {:<6} {:>6} {:>6} {:>6} {:>7} {:>10}".format(*header))
    for s in r.decision_trace.steps:
        think = "" if s.reasoning is None else ("on" if s.reasoning else "off")
        cost = f"{s.cost_usd:.6f}" if s.cost_usd is not None else "-"
        rtok = "-" if s.reasoning_tokens is None else s.reasoning_tokens
        print(
            f"  {s.step:<11} {(s.model or s.kind):<40} {think:<6} {s.prompt_tokens:>6} "
            f"{s.completion_tokens:>6} {rtok:>6} "
            f"{s.latency_ms:>7.0f} {cost:>10}"
            + (f"  {'(cached)' if s.cached else ''}{s.note or ''}"[:90])
        )
    t = r.decision_trace
    print(f"  total: {t.total_tokens} tokens, {t.total_latency_ms:.0f} ms, ${t.total_cost_usd:.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Iba fever triage (decision support, not diagnosis)."
    )
    parser.add_argument("text", help="case description (English or Pidgin); synthetic cases only")
    parser.add_argument("--state", help="Nigerian state, e.g. Ondo")
    parser.add_argument("--lga")
    parser.add_argument("--answer", action="append", default=[], metavar="ID=ANSWER")
    parser.add_argument("--skip-questions", action="store_true")
    parser.add_argument("--index", type=Path, help="index dir (default: INDEX_DIR)")
    parser.add_argument("--mock-outbreak", type=Path, help="replay canned search results")
    parser.add_argument(
        "--config", choices=["routed", "reason-only", "fast-only"], default="routed"
    )
    parser.add_argument("--json", action="store_true", help="print the TriageResult JSON")
    parser.add_argument(
        "--save-raw", type=Path, help="save raw API responses here (debug; synthetic cases only)"
    )
    args = parser.parse_args()

    answers = []
    for item in args.answer:
        key, sep, value = item.partition("=")
        if not sep:
            parser.error(f"--answer must be ID=ANSWER, got {item!r}")
        answers.append(FollowUpAnswer(id=key.strip(), answer=value.strip()))

    request = TriageRequest(
        text=args.text,
        state=args.state,
        lga=args.lga,
        answers=answers,
        skip_questions=args.skip_questions,
    )
    deps = build_deps(get_settings(), index_dir=args.index, mock_outbreak=args.mock_outbreak)
    raws: list[dict] = []
    if args.save_raw:
        deps.client.on_response = lambda r: raws.append(r.model_dump())
    try:
        result = run_triage(request, deps, args.config)
    except SpendLimitError as exc:
        print(f"Stopped by spend guard: {exc}", file=sys.stderr)
        return 2
    finally:
        if args.save_raw and raws:
            args.save_raw.parent.mkdir(parents=True, exist_ok=True)
            args.save_raw.write_text(json.dumps(raws, indent=1, default=str), encoding="utf-8")

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print_result(result, colour=sys.stdout.isatty())
    return 0


if __name__ == "__main__":
    sys.exit(main())
