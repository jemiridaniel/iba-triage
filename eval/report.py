"""Metrics, markdown report and charts from eval results (SPEC §6).

    uv run python -m eval.report                      # all eval/results/*.jsonl
    uv run python -m eval.report --runs eval/results/routed__case.jsonl

Writes eval/results/report.md and three PNGs:
  safety_by_config.png   danger-sign recall, under-/over-triage, top-3 differential hit
  cost_latency.png       cost per case vs p95 latency per config
  outbreak_lift.png      Lassa cases with vs without a (mock) live outbreak signal
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

RESULTS = Path("eval/results")
BEFORE = RESULTS / "before"
RANK = {"treat_monitor": 0, "refer_24h": 1, "refer_now": 2}
LABEL = {
    "treat_monitor": "Treat & monitor",
    "refer_24h": "Refer within 24h",
    "refer_now": "Refer now",
}


def load(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        rows += [
            json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    return rows


def pct(n: float, d: float) -> float | None:
    return round(100 * n / d, 1) if d else None


def fmt(v: Any, suffix: str = "") -> str:
    return "—" if v is None else f"{v}{suffix}"


def hit_top3(rec: dict) -> bool:
    top = [c.lower() for c in rec.get("differential", [])[:3]]
    return any(term.lower() in cond for term in rec["gold"]["top_differential"] for cond in top)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def metrics(rows: list[dict]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("error")]
    gold_signs = sum(len(r["gold"]["danger_signs"]) for r in ok)
    found_signs = sum(
        len(set(r["gold"]["danger_signs"]) & {d["code"] for d in r["danger_signs"]}) for r in ok
    )
    under = [r for r in ok if RANK[r["triage_level"]] < RANK[r["gold"]["triage_level"]]]
    over = [r for r in ok if RANK[r["triage_level"]] > RANK[r["gold"]["triage_level"]]]
    missed_refer = [
        r for r in ok if r["gold"]["must_refer"] and r["triage_level"] == "treat_monitor"
    ]
    kept = sum(r.get("citations_kept", 0) for r in ok)
    dropped = sum(r.get("citations_dropped", 0) for r in ok)
    lat = [r["latency_ms"] / 1000 for r in ok]
    costs = [r["cost_usd"] for r in ok]
    return {
        "n": len(ok),
        "errors": len(rows) - len(ok),
        "danger_recall": pct(found_signs, gold_signs),
        "danger_signs_gold": gold_signs,
        "under_triage": pct(len(under), len(ok)),
        "under_cases": under,
        "missed_refer": len(missed_refer),
        "over_triage": pct(len(over), len(ok)),
        "exact": pct(sum(r["triage_level"] == r["gold"]["triage_level"] for r in ok), len(ok)),
        "top3_hit": pct(sum(hit_top3(r) for r in ok), len(ok)),
        "citation_validity": pct(kept, kept + dropped),
        "citations": kept + dropped,
        "p50": round(percentile(lat, 0.5), 1) if lat else None,
        "p95": round(percentile(lat, 0.95), 1) if lat else None,
        "cost_per_case": round(statistics.mean(costs), 5) if costs else None,
        "cost_total": round(sum(costs), 4),
    }


def lassa_lift(rows: list[dict]) -> dict[str, dict] | None:
    """Same Lassa cases run with outbreak=off vs outbreak=mock, per config."""
    out = {}
    for config in {r["config"] for r in rows}:
        off = {
            r["id"]: r
            for r in rows
            if r["config"] == config
            and r["outbreak_mode"] == "off"
            and r["category"] == "suspected_lassa"
            and not r.get("error")
        }
        mock = {
            r["id"]: r
            for r in rows
            if r["config"] == config
            and r["outbreak_mode"] == "mock"
            and r["category"] == "suspected_lassa"
            and not r.get("error")
        }
        ids = sorted(set(off) & set(mock))
        if not ids:
            continue

        def stats(group: dict[str, dict], ids: list[str]) -> dict[str, float | None]:
            recs = [group[i] for i in ids]
            return {
                "lassa_top3": pct(
                    sum(any("lassa" in c.lower() for c in r["differential"][:3]) for r in recs),
                    len(recs),
                ),
                "refer_now": pct(sum(r["triage_level"] == "refer_now" for r in recs), len(recs)),
                "under_triage": pct(
                    sum(RANK[r["triage_level"]] < RANK[r["gold"]["triage_level"]] for r in recs),
                    len(recs),
                ),
            }

        out[config] = {
            "n": len(ids),
            "ids": ids,
            "without": stats(off, ids),
            "with": stats(mock, ids),
        }
    return out or None


def judge_summary(paths: list[Path]) -> dict[tuple, dict] | None:
    rows = load([p for p in paths if p.exists()])
    if not rows:
        return None
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by[(r["config"], r["outbreak_mode"])].append(r)
    return {
        k: {
            "pairs": len(v),
            "cases": len({r["id"] for r in v}),
            "supported": pct(sum(r["verdict"] == "supported" for r in v), len(v)),
            "partial": pct(sum(r["verdict"] == "partial" for r in v), len(v)),
            "unsupported": pct(sum(r["verdict"] == "unsupported" for r in v), len(v)),
            "examples": [r for r in v if r["verdict"] != "supported"][:3],
        }
        for k, v in by.items()
    }


def charts(groups: dict[tuple, dict], lift: dict | None, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    names = [f"{c}\n({m})" for c, m in groups]
    series = [
        ("Danger-sign recall", "danger_recall", "#b91c1c"),
        ("Under-triage", "under_triage", "#f59e0b"),
        ("Over-triage", "over_triage", "#64748b"),
        ("Top-3 differential hit", "top3_hit", "#0f766e"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.8 / len(series)
    for i, (label, key, colour) in enumerate(series):
        xs = [j + i * width for j in range(len(groups))]
        ys = [g[key] or 0 for g in groups.values()]
        bars = ax.bar(xs, ys, width, label=label, color=colour)
        ax.bar_label(bars, fmt="%.0f", fontsize=8)
    ax.set_xticks([j + width * (len(series) - 1) / 2 for j in range(len(groups))], names)
    ax.set_ylabel("%")
    ax.set_ylim(0, 110)
    ax.set_title("Safety metrics by configuration (synthetic vignettes, not clinician-reviewed)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "safety_by_config.png", dpi=150)
    plt.close(fig)
    written.append("safety_by_config.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    for (config, mode), g in groups.items():
        if g["cost_per_case"] is None:
            continue
        ax.scatter(g["cost_per_case"] * 1000, g["p95"], s=120)
        ax.annotate(
            f"{config} ({mode})",
            (g["cost_per_case"] * 1000, g["p95"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    ax.set_xlabel("Cost per case (USD × 1000)")
    ax.set_ylabel("p95 latency (s)")
    ax.set_title("Cost vs latency per configuration")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "cost_latency.png", dpi=150)
    plt.close(fig)
    written.append("cost_latency.png")

    if lift:
        fig, ax = plt.subplots(figsize=(7, 4))
        keys = [
            ("Lassa in top 3", "lassa_top3"),
            ("Refer now", "refer_now"),
            ("Under-triage", "under_triage"),
        ]
        config = next(iter(lift))
        without = [lift[config]["without"][k] or 0 for _, k in keys]
        with_ = [lift[config]["with"][k] or 0 for _, k in keys]
        xs = range(len(keys))
        b1 = ax.bar(
            [x - 0.2 for x in xs],
            without,
            0.4,
            label="Baseline only (no live data)",
            color="#94a3b8",
        )
        b2 = ax.bar(
            [x + 0.2 for x in xs],
            with_,
            0.4,
            label="With live outbreak signal (mock)",
            color="#0369a1",
        )
        ax.bar_label(b1, fmt="%.0f", fontsize=8)
        ax.bar_label(b2, fmt="%.0f", fontsize=8)
        ax.set_xticks(list(xs), [k for k, _ in keys])
        ax.set_ylim(0, 110)
        ax.set_ylabel("% of Lassa cases")
        ax.set_title(f"Outbreak lift on suspected-Lassa cases ({config}, n={lift[config]['n']})")
        ax.legend(fontsize=8, loc="center right")
        fig.tight_layout()
        fig.savefig(out / "outbreak_lift.png", dpi=150)
        plt.close(fig)
        written.append("outbreak_lift.png")
    return written


def regold(rows: list[dict]) -> list[dict]:
    """Score every row against the current vignette gold for the mode it ran in."""
    from eval.run_eval import effective_gold, load_vignettes

    cases = {c["id"]: c for c in load_vignettes()}
    for r in rows:
        if r["id"] in cases:
            r["gold"] = effective_gold(cases[r["id"]], bool(r.get("live_signal")))
    return rows


def grounding_stats(rows: list[dict], judge_rows: list[dict]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("error")]
    has_det = any("grounding" in r for r in ok)
    if has_det:
        claims = sum(r["grounding"]["claims"] for r in ok)
        verified = sum(r["grounding"]["verified"] for r in ok)
        marked = sum(r["grounding"]["unsupported"] for r in ok)
    else:  # before: one claim per cited ref, no deterministic check
        claims = sum(len(c["refs"]) for r in ok for c in r.get("claims", []))
        verified = marked = None
    judged = [j for j in judge_rows if j["verdict"] != "judge_error"]
    return {
        "cases": len(ok),
        "claims": claims,
        "verified_pct": pct(verified, claims) if verified is not None else None,
        "marked_pct": pct(marked, claims) if marked is not None else None,
        "judged": len(judged),
        "judge_cases": len({j["id"] for j in judge_rows}),
        "supported": pct(sum(j["verdict"] == "supported" for j in judged), len(judged)),
        "partial": pct(sum(j["verdict"] == "partial" for j in judged), len(judged)),
        "unsupported": pct(sum(j["verdict"] == "unsupported" for j in judged), len(judged)),
        "judge_errors": len(judge_rows) - len(judged),
        "verified_judged": [j for j in judged if j.get("grounding_status") == "verified"],
    }


def retrieval_summary(results_dir: Path) -> list[tuple[str, int, dict]]:
    out = []
    for k in (6, 8):
        for name in ("single", "multi"):
            p = results_dir / f"retrieval_{name}_k{k}.jsonl"
            if p.exists():
                rows = load([p])
                out.append(
                    (
                        name,
                        k,
                        {
                            "n": len(rows),
                            "key_hit": pct(sum(r["key_hit"] for r in rows), len(rows)),
                            "doc_hit": pct(sum(r["doc_hit"] for r in rows), len(rows)),
                            "misses": [r["id"] for r in rows if not r["key_hit"]],
                        },
                    )
                )
    return out


def render_grounding(stages: list[tuple[str, dict]], retrieval: list) -> list[str]:
    lines = ["## Grounding", ""]
    if retrieval:
        lines += [
            "### Retrieval: was the defining guideline passage retrieved?",
            "",
            "| Strategy | k | Cases | Key-passage hit | Guideline-doc hit | Misses |",
            "|---|---|---|---|---|---|",
        ]
        for name, k, s in retrieval:
            label = (
                "single symptom query (before)"
                if name == "single"
                else "per-condition queries + doc prior (after)"
            )
            lines.append(
                f"| {label} | {k} | {s['n']} | {fmt(s['key_hit'], '%')} | {fmt(s['doc_hit'], '%')} | {', '.join(s['misses']) or '—'} |"
            )
        lines.append("")
    lines += [
        "### Claims",
        "",
        "| Run | Cases | Model claims | Quote-verified | Marked unsupported | Judged pairs | Judge: supported | partial | unsupported |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, g in stages:
        if g:
            lines.append(
                f"| {label} | {g['cases']} | {g['claims']} | {fmt(g['verified_pct'], '%')} | {fmt(g['marked_pct'], '%')} | "
                f"{g['judged']} ({g['judge_cases']} cases) | {fmt(g['supported'], '%')} | {fmt(g['partial'], '%')} | {fmt(g['unsupported'], '%')} |"
            )
    lines += [
        "",
        "- **Quote-verified**: the claim's evidence quote (8–40 words) was found in the cited chunk "
        "(deterministic; normalised whitespace, dashes and quotes; fuzzy ratio ≥ 0.9). Everything else is "
        '**marked unsupported** and shown as "AI suggestion — no guideline source".',
        "- **Judge** columns are LLM-judged (Nemotron 3 Super, reasoning off): does the cited chunk support "
        "the claim? A screening signal, not ground truth.",
        "- Before: each claim bundled a condition with all its reasons and patient facts. After: one claim "
        "per reason or action; patient facts are excluded (they need no guideline source).",
        "",
    ]
    judged_stage = next((g for _, g in reversed(stages) if g and g["verified_judged"]), None)
    if judged_stage:
        v = judged_stage["verified_judged"]
        lines.append(
            f"Quote-verified claims only (n={len(v)}): judge says supported "
            f"{pct(sum(j['verdict'] == 'supported' for j in v), len(v))}%, partial "
            f"{pct(sum(j['verdict'] == 'partial' for j in v), len(v))}%, unsupported "
            f"{pct(sum(j['verdict'] == 'unsupported' for j in v), len(v))}%."
        )
        lines.append("")
    return lines


def render(
    groups: dict[tuple, dict],
    lift: dict | None,
    judge: dict | None,
    images: list[str],
    grounding: list[str] | None = None,
) -> str:
    lines = [
        "# Ibà eval report",
        "",
        "> Synthetic vignettes; gold labels **not yet clinician-reviewed**. Treat these numbers as",
        "> a development check, not a clinical validation.",
        "",
        "## Metrics",
        "",
        "| Config (outbreak) | n | Danger-sign recall | **Under-triage** | Over-triage | Exact triage | Top-3 dx hit | Citation validity | p50 / p95 latency | Cost / case |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (config, mode), g in groups.items():
        lines.append(
            f"| {config} ({mode}) | {g['n']} | {fmt(g['danger_recall'], '%')} ({g['danger_signs_gold']} signs) | "
            f"**{fmt(g['under_triage'], '%')}** | {fmt(g['over_triage'], '%')} | {fmt(g['exact'], '%')} | "
            f"{fmt(g['top3_hit'], '%')} | {fmt(g['citation_validity'], '%')} ({g['citations']}) | "
            f"{fmt(g['p50'], ' s')} / {fmt(g['p95'], ' s')} | ${fmt(g['cost_per_case'])} |"
        )
    lines += [
        "",
        "- **Under-triage** (predicted less urgent than gold) is the headline safety metric.",
        "- Citation validity = model citations that resolved to an indexed guideline chunk or a current "
        "outbreak source (deterministic check).",
        "- Latency uses each step's original model time (cache replays report the first run's time).",
        "",
    ]
    for (config, mode), g in groups.items():
        lines.append(f"### Under-triaged cases: {config} ({mode})")
        lines.append("")
        if not g["under_cases"]:
            lines += ["None.", ""]
            continue
        for r in g["under_cases"]:
            gold_signs = set(r["gold"]["danger_signs"])
            found = {d["code"] for d in r["danger_signs"]}
            missed = sorted(gold_signs - found)
            lines.append(
                f"- **{r['id']}** ({r['category']}): gold {LABEL[r['gold']['triage_level']]}, got "
                f"{LABEL[r['triage_level']]}. Missed danger signs: {', '.join(missed) or 'none'}. "
                f"Differential: {', '.join(r['differential'][:3]) or '—'}. "
                f'Model rationale: "{(r.get("triage_rationale") or "")[:220]}"'
            )
        lines.append("")
    if lift:
        lines += [
            "## Outbreak lift (suspected-Lassa cases)",
            "",
            "Same cases run with live outbreak search off (static baseline only) and with a mock live "
            "Lassa signal for the case's state. Mock data stands in for Tavily until credits arrive.",
            "",
            "| Config | n | | Lassa in top 3 | Refer now | Under-triage |",
            "|---|---|---|---|---|---|",
        ]
        for config, v in lift.items():
            for label, key in (("without live signal", "without"), ("with live signal", "with")):
                s = v[key]
                lines.append(
                    f"| {config} | {v['n']} | {label} | {fmt(s['lassa_top3'], '%')} | "
                    f"{fmt(s['refer_now'], '%')} | {fmt(s['under_triage'], '%')} |"
                )
        lines.append("")
    if judge:
        lines += [
            "## Citation support (LLM-judged)",
            "",
            "**This section is judged by an LLM** (Nemotron 3 Super, reasoning off) on a ~20% sample "
            "of cases: does the cited guideline chunk support the claim it is attached to? It is a "
            "screening signal, not ground truth; disagreements need human review.",
            "",
            "| Config (outbreak) | Cases | Pairs | Supported | Partial | Unsupported |",
            "|---|---|---|---|---|---|",
        ]
        for (config, mode), j in judge.items():
            lines.append(
                f"| {config} ({mode}) | {j['cases']} | {j['pairs']} | {fmt(j['supported'], '%')} | "
                f"{fmt(j['partial'], '%')} | {fmt(j['unsupported'], '%')} |"
            )
        for j in judge.values():
            for e in j["examples"]:
                lines.append(
                    f'- {e["verdict"]}: `{e["ref"]}` for "{e["claim"][:120]}": {e["reason"]}'
                )
        lines.append("")
    if grounding:
        lines += grounding
    if images:
        lines += ["## Charts", ""] + [f"![{i}]({i})" for i in images] + [""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the eval report.")
    parser.add_argument("--runs", nargs="*", type=Path)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()

    paths = args.runs or sorted(
        p
        for p in RESULTS.glob("*.jsonl")
        if not p.name.endswith(".judge.jsonl") and not p.name.startswith("retrieval_")
    )
    rows = regold(load(paths))
    if not rows:
        print("No results found.")
        return 1
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["config"], r["outbreak_mode"])].append(r)
    groups = {k: metrics(v) for k, v in sorted(grouped.items())}
    lift = lassa_lift(rows)
    judge = judge_summary([p.with_suffix(".judge.jsonl") for p in paths])
    args.out.mkdir(parents=True, exist_ok=True)
    images = charts(groups, lift, args.out)

    def judge_rows(p: Path) -> list[dict]:
        j = p.with_suffix(".judge.jsonl")
        return load([j]) if j.exists() else []

    stage_files = [
        ("1. Citations only (no quote requirement)", BEFORE / "routed__case.jsonl"),
        ("2. Quote-backed claims", RESULTS / "grounding" / "routed__case.jsonl"),
        ("3. + trimmed passages, parallel embedding", RESULTS / "routed__case.jsonl"),
    ]
    stages = [
        (label, grounding_stats(regold(load([p])), judge_rows(p)))
        for label, p in stage_files
        if p.exists()
    ]
    grounding = render_grounding(stages, retrieval_summary(RESULTS))
    report = render(groups, lift, judge, images, grounding)
    (args.out / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
