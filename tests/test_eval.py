"""Eval harness: vignette file, selection, mock search, metrics, report. No network."""

import json
from collections import Counter
from pathlib import Path

from eval.report import hit_top3, lassa_lift, metrics, render
from eval.run_eval import TemplateSearch, judge_sample, load_vignettes, outbreak_search, select
from tests.pipeline_fakes import REPO

CASES = load_vignettes(REPO / "eval" / "vignettes.jsonl")
CODES = {
    "convulsions",
    "unable_to_drink_or_breastfeed",
    "vomiting_everything",
    "lethargy_or_unconsciousness",
    "bleeding",
    "jaundice",
    "respiratory_distress",
    "severe_pallor",
    "dark_urine",
    "prostration",
}


def test_vignettes_match_spec_mix() -> None:
    assert len(CASES) == 60 and len({c["id"] for c in CASES}) == 60
    assert Counter(c["category"] for c in CASES) == {
        "uncomplicated_malaria": 10,
        "severe_malaria": 10,
        "suspected_lassa": 10,
        "cholera_awd": 6,
        "suspected_meningitis": 6,
        "other_febrile": 8,
        "paediatric_danger_signs": 6,
        "pidgin_variant": 4,
    }


def test_every_vignette_has_gold_and_rationale() -> None:
    for c in CASES:
        g = c["gold"]
        assert g["triage_level"] in {"treat_monitor", "refer_24h", "refer_now"}
        assert set(g["danger_signs"]) <= CODES
        assert g["must_refer"] == (g["triage_level"] != "treat_monitor")
        assert g["top_differential"] and isinstance(g["outbreak_relevant"], bool)
        assert c["rationale"] and c["clinician_reviewed"] is False and c["synthetic"] is True
        if g["danger_signs"]:
            assert g["triage_level"] == "refer_now"


def test_tricky_cases_present() -> None:
    tricky = " ".join(c.get("tricky", "") for c in CASES)
    assert "RDT-positive malaria with Lassa features" in tricky
    assert "single subtle danger sign" in tricky
    assert "cholera without fever" in tricky
    assert sum("Pidgin" in c.get("tricky", "") for c in CASES) >= 4


def test_stratified_selection_covers_categories() -> None:
    picked = select(CASES, None, None, 8, stratified=True)
    assert len({c["category"] for c in picked}) == 8


def test_mock_search_is_per_state(tmp_path: Path) -> None:
    template = REPO / "data" / "mock" / "outbreak_lassa_template.json"
    (result,) = TemplateSearch(template, "Cross River").search("q", include_domains=())
    assert "Cross River State" in result.content and "MOCK-lassa-cross-river" in result.url
    case = {"state": "Ondo", "outbreak": "baseline_only"}
    assert outbreak_search("case", case, template) is None
    assert outbreak_search("mock", case, template) is not None
    assert outbreak_search("off", {**case, "outbreak": "mock_live"}, template) is None


def rec(id, gold, pred, *, cat="x", signs=(), found=(), diff=(), mode="case", kept=2, dropped=0):
    return {
        "id": id,
        "category": cat,
        "config": "routed",
        "outbreak_mode": mode,
        "gold": {
            "triage_level": gold,
            "danger_signs": list(signs),
            "must_refer": gold != "treat_monitor",
            "top_differential": ["lassa"],
            "outbreak_relevant": True,
        },
        "triage_level": pred,
        "danger_signs": [{"code": c, "source": "rule"} for c in found],
        "differential": list(diff),
        "citations_kept": kept,
        "citations_dropped": dropped,
        "latency_ms": 10000,
        "cost_usd": 0.004,
        "triage_rationale": "r",
    }


def test_metrics() -> None:
    rows = [
        rec(
            "a",
            "refer_now",
            "refer_now",
            signs=["bleeding"],
            found=["bleeding"],
            diff=["Lassa fever"],
        ),
        rec("b", "refer_now", "refer_24h", signs=["convulsions"], diff=["Malaria"], dropped=2),
        rec("c", "treat_monitor", "refer_24h", diff=["x", "y", "Lassa"]),
        rec("d", "refer_24h", "treat_monitor"),
    ]
    m = metrics(rows)
    assert m["danger_recall"] == 50.0
    assert m["under_triage"] == 50.0 and [r["id"] for r in m["under_cases"]] == ["b", "d"]
    assert m["over_triage"] == 25.0
    assert m["missed_refer"] == 1
    assert m["top3_hit"] == 50.0
    assert m["citation_validity"] == 80.0
    assert m["cost_per_case"] == 0.004
    assert "**b**" in render({("routed", "case"): m}, None, None, [])


def test_top3_matching_is_substring_case_insensitive() -> None:
    assert hit_top3(rec("a", "refer_now", "refer_now", diff=["Suspected LASSA fever"]))


def test_lassa_lift() -> None:
    rows = [
        rec("l1", "refer_now", "refer_24h", cat="suspected_lassa", mode="off", diff=["Typhoid"]),
        rec(
            "l1", "refer_now", "refer_now", cat="suspected_lassa", mode="mock", diff=["Lassa fever"]
        ),
    ]
    lift = lassa_lift(rows)["routed"]
    assert lift["n"] == 1
    assert lift["without"]["lassa_top3"] == 0 and lift["with"]["lassa_top3"] == 100.0
    assert lift["without"]["under_triage"] == 100.0 and lift["with"]["under_triage"] == 0


def test_judge_sample_is_deterministic_20_percent() -> None:
    records = [{"id": f"c{i}", "claims": [{"claim": "x", "refs": ["r"]}]} for i in range(10)]
    first = judge_sample(records)
    assert len(first) == 2 and first == judge_sample(list(reversed(records)))
    assert judge_sample([{"id": "z", "claims": []}]) == []


def test_vignette_file_is_valid_jsonl() -> None:
    for line in (REPO / "eval" / "vignettes.jsonl").read_text().splitlines():
        json.loads(line)


def test_gold_depends_on_outbreak_mode() -> None:
    from eval.run_eval import effective_gold

    la06 = next(c for c in CASES if c["id"] == "la-06")
    assert effective_gold(la06, live_signal=False)["triage_level"] == "refer_24h"
    assert effective_gold(la06, live_signal=True)["triage_level"] == "refer_now"
    um01 = next(c for c in CASES if c["id"] == "um-01")
    assert effective_gold(um01, True) == um01["gold"]  # no override -> same gold


def test_report_regolds_rows_by_mode() -> None:
    from eval.report import regold

    base = rec("la-06", "refer_24h", "refer_now", cat="suspected_lassa", mode="mock")
    base["live_signal"] = True
    (row,) = regold([base])
    assert row["gold"]["triage_level"] == "refer_now"  # no longer counted as over-triage


def test_grounding_stats_before_and_after() -> None:
    from eval.report import grounding_stats

    after = [
        {
            **rec("a", "refer_now", "refer_now"),
            "grounding": {"claims": 4, "verified": 3, "unsupported": 1, "patient_facts": 2},
        }
    ]
    judge = [
        {"id": "a", "verdict": "supported", "grounding_status": "verified"},
        {"id": "a", "verdict": "unsupported", "grounding_status": "unsupported"},
        {"id": "a", "verdict": "judge_error", "grounding_status": "verified"},
    ]
    g = grounding_stats(after, judge)
    assert (g["claims"], g["verified_pct"], g["marked_pct"]) == (4, 75.0, 25.0)
    assert (g["judged"], g["supported"], g["judge_errors"]) == (2, 50.0, 1)
    before = [
        {**rec("b", "refer_now", "refer_now"), "claims": [{"claim": "x", "refs": ["r1", "r2"]}]}
    ]
    assert grounding_stats(before, [])["claims"] == 2
    assert grounding_stats(before, [])["verified_pct"] is None
