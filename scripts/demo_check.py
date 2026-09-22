"""Demo reliability check: run the 3 demo cases through /triage/stream on a running server.

    docker compose up -d
    uv run python -m scripts.demo_check                  # once
    uv run python -m scripts.demo_check --repeat 3       # before recording / submission
    uv run python -m scripts.demo_check --url http://192.168.100.29:8000

Asserts, for every run: no error event, status is not "incomplete", and the final triage level
matches the case's expected level (frontend/src/demos.json, shared with the UI's demo
buttons). Also reports when the first triage banner arrived, reasoning fallbacks after
truncation, latency and cost. Exits 1 on any failure. Spends credits (~$0.007 per case).
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

DEMOS = Path(__file__).resolve().parents[1] / "frontend" / "src" / "demos.json"


def stream(client: httpx.Client, url: str, case: dict) -> dict:
    t0 = time.perf_counter()
    first_banner = None
    events: list[tuple[str, dict]] = []
    with client.stream(
        "POST", f"{url}/triage/stream", json={"text": case["text"], "state": case["state"]}
    ) as r:
        r.raise_for_status()
        buf = ""
        for chunk in r.iter_text():
            buf += chunk
            while "\n\n" in buf:
                block, buf = buf.split("\n\n", 1)
                name = next(line[7:] for line in block.splitlines() if line.startswith("event: "))
                data = json.loads(
                    next(line[6:] for line in block.splitlines() if line.startswith("data: "))
                )
                events.append((name, data))
                if first_banner is None and (
                    (name == "rules_pre" and data.get("floor")) or name == "rules_post"
                ):
                    first_banner = time.perf_counter() - t0
    elapsed = time.perf_counter() - t0
    final = next((d for n, d in events if n == "final"), None)
    errors = [d for n, d in events if n == "error"]
    reason = next(
        (
            s
            for s in (final or {}).get("decision_trace", {}).get("steps", [])
            if s["step"] == "reason"
        ),
        {},
    )
    note = reason.get("note") or ""
    return {
        "label": case["label"],
        "expected": case["expected"],
        "status": final.get("status") if final else None,
        "level": final.get("triage_level") if final else None,
        "errors": errors,
        "first_banner_s": round(first_banner, 1) if first_banner else None,
        "elapsed_s": round(elapsed, 1),
        "cost": (final or {}).get("decision_trace", {}).get("total_cost_usd"),
        "fallback": "fallback after truncation" in note,
        "truncated": "LLMTruncatedError" in note or "truncated" in note,
        "grounding": (final or {}).get("grounding"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the demo cases against a running Ibà server."
    )
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()

    cases = json.loads(DEMOS.read_text(encoding="utf-8"))
    results, failures = [], []
    with httpx.Client(timeout=args.timeout) as client:
        client.get(f"{args.url}/health").raise_for_status()
        for run in range(1, args.repeat + 1):
            for case in cases:
                try:
                    r = stream(client, args.url, case)
                except Exception as exc:  # network / server errors are failures too
                    r = {
                        "label": case["label"],
                        "expected": case["expected"],
                        "status": None,
                        "level": None,
                        "errors": [{"message": f"{type(exc).__name__}: {exc}"}],
                        "first_banner_s": None,
                        "elapsed_s": None,
                        "cost": None,
                        "fallback": False,
                        "truncated": False,
                        "grounding": None,
                    }
                problems = []
                if r["errors"]:
                    problems.append(f"error event: {r['errors'][0].get('message')}")
                if r["status"] != "complete":
                    problems.append(f"status {r['status']}")
                if r["level"] != r["expected"]:
                    problems.append(f"level {r['level']} != expected {r['expected']}")
                r["run"], r["ok"] = run, not problems
                results.append(r)
                if problems:
                    failures.append(f"run {run} {case['label']}: {'; '.join(problems)}")
                g = r["grounding"] or {}
                print(
                    f"run {run}  {case['label']:<32} {'PASS' if r['ok'] else 'FAIL'}  "
                    f"level={r['level']!s:<13} banner@{r['first_banner_s']}s  "
                    f"total {r['elapsed_s']}s  "
                    f"${r['cost'] or 0:.4f}  fallback={r['fallback']}  "
                    f"verified {g.get('verified', '-')}/{g.get('claims', '-')}"
                )

    ok = [r for r in results if r["elapsed_s"] is not None]
    if ok:
        times = sorted(r["elapsed_s"] for r in ok)
        print(
            f"\n{len(results)} runs, {len(failures)} failures, "
            f"fallbacks {sum(r['fallback'] for r in results)}; "
            f"latency p50 {statistics.median(times):.1f}s max {times[-1]:.1f}s; "
            f"cost ${sum(r['cost'] or 0 for r in ok):.4f}"
        )
    for f in failures:
        print("FAIL:", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
