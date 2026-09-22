"""Retrieval diagnostic: did we retrieve the passage that defines the relevant criteria?

    uv run python -m eval.retrieval_eval                      # the 10 dry-run cases
    uv run python -m eval.retrieval_eval --ids la-01 ch-01

For each case: intake (disk cache) and rules, then retrieval with both strategies:
  single  one symptom query (the original retrieve step)
  multi   one query per suspected condition and purpose, with a doc prior (rag/queries.py)

Metrics per strategy, at the same k:
  key-passage hit  a retrieved chunk is one of the category's key passages (anchor phrases)
  doc hit          a retrieved chunk comes from one of the category's expected guidelines
Writes eval/results/retrieval_<strategy>.jsonl with the queries and top-k chunks per case.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.app.config import get_settings
from backend.app.graph import nodes
from backend.app.graph.pipeline import build_deps
from backend.app.graph.state import TriageRequest, TriageState
from backend.app.rag.queries import plan_queries, retrieve, suspected_conditions
from eval.run_eval import MOCK_TEMPLATE, RESULTS, load_vignettes, outbreak_search

# Anchor phrases for the passages that define each category's criteria (verified to resolve).
KEY_PASSAGES: dict[str, list[tuple[str, str]]] = {
    "suspected_lassa": [
        ("ncdc-lassa", "fever for 3-21 days with a measured temperature"),
        ("ncdc-lassa", "Put patient in a holding area"),
    ],
    "severe_malaria": [
        ("who-malaria", "Prostration: Generalized weakness"),
        ("who-imci", "VERY SEVERE FEBRILE DISEASE"),
    ],
    "uncomplicated_malaria": [
        ("who-malaria", "no features of severe malaria is defined as having uncomplicated malaria"),
        ("who-imci", "Follow-up in 3 days if fever persists"),
    ],
    "cholera_awd": [
        ("ncdc-cholera", "ASSESSMENT FOR LEVEL OF DEHYDRATION"),
        ("who-imci", "SEVERE DEHYDRATION"),
    ],
    "suspected_meningitis": [("who-imci", "Look or feel for stiff neck")],
    "other_febrile": [("who-imci", "If fever is present every day for more than 7 days")],
    "paediatric_danger_signs": [
        ("who-imci", "VERY SEVERE FEBRILE DISEASE"),
        ("who-imci", "Not able to drink or breastfeed"),
    ],
}
EXPECTED_DOCS: dict[str, set[str]] = {
    "suspected_lassa": {"ncdc-lassa", "ncdc-vhf-ipc", "ncdc-lassa-advisory-2026"},
    "severe_malaria": {"who-malaria", "who-imci"},
    "uncomplicated_malaria": {"who-malaria", "who-imci"},
    "cholera_awd": {"ncdc-cholera", "who-imci"},
    "suspected_meningitis": {"who-imci", "ncdc-csm", "ncdc-csm-quickref"},
    "other_febrile": {"who-imci"},
    "paediatric_danger_signs": {"who-imci"},
}
# Pidgin variants are drawn from other categories.
PIDGIN_BASE = {
    "pc-01": "uncomplicated_malaria",
    "pc-02": "severe_malaria",
    "pc-03": "suspected_lassa",
    "pc-04": "cholera_awd",
}


def base_category(case: dict) -> str:
    return PIDGIN_BASE.get(case["id"], case["category"])


def evaluate(ids: list[str], k: int, out_dir: Path) -> dict[str, dict]:
    settings = get_settings().model_copy(update={"cache_enabled": True})
    deps = build_deps(settings)
    store = deps.store
    assert store is not None and deps.embed is not None, "build the index first"
    cases = [c for c in load_vignettes() if c["id"] in set(ids)]
    key_ids = {
        cat: {store.find(*a).id for a in anchors if store.find(*a)}
        for cat, anchors in KEY_PASSAGES.items()
    }

    results: dict[str, list[dict]] = {"single": [], "multi": []}
    for case in cases:
        state = TriageState(
            request=TriageRequest(text=case["text"], state=case.get("state"), skip_questions=True)
        )
        state = state.model_copy(update=nodes.intake(state, deps))
        state = state.model_copy(update=nodes.rules_pre(state, deps))
        deps.outbreak.search = outbreak_search("case", case, MOCK_TEMPLATE)
        outbreak = deps.outbreak.check(case.get("state"))
        cat = base_category(case)
        full = nodes.case_with_text(state)

        single_q = nodes.build_query(full, state.pre)
        single_hits = store.search_text(single_q, deps.embed, k=k)
        conditions = suspected_conditions(
            full, bool(state.pre.danger_signs), outbreak, lassa_rule=False
        )
        multi_hits, logs = retrieve(store, deps.embed, plan_queries(full, conditions), k_total=k)

        for name, hits, queries in (
            ("single", single_hits, [{"query": single_q}]),
            ("multi", multi_hits, [vars(lg) for lg in logs]),
        ):
            chunk_ids = [h.chunk.id for h in hits]
            results[name].append(
                {
                    "id": case["id"],
                    "category": cat,
                    "k": k,
                    "queries": queries,
                    "conditions": conditions if name == "multi" else None,
                    "top": [
                        f"{h.chunk.id} | {h.chunk.section} (p.{h.chunk.page}) {h.score:.3f}"
                        for h in hits
                    ],
                    "key_hit": bool(set(chunk_ids) & key_ids.get(cat, set())),
                    "doc_hit": any(h.chunk.doc_id in EXPECTED_DOCS.get(cat, set()) for h in hits),
                    "docs": sorted({h.chunk.doc_id for h in hits}),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, rows in results.items():
        with (out_dir / f"retrieval_{name}_k{k}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n = len(rows)
        summary[name] = {
            "n": n,
            "key_hit": round(100 * sum(r["key_hit"] for r in rows) / n, 1) if n else None,
            "doc_hit": round(100 * sum(r["doc_hit"] for r in rows) / n, 1) if n else None,
            "misses": [r["id"] for r in rows if not r["key_hit"]],
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare retrieval strategies on eval cases.")
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--k", type=int, nargs="*", default=[6, 8])
    args = parser.parse_args()
    ids = args.ids
    if not ids:
        before = RESULTS / "before" / "routed__case.jsonl"
        src = before if before.exists() else RESULTS / "routed__case.jsonl"
        ids = [json.loads(line)["id"] for line in src.read_text().splitlines() if line.strip()]
    for k in args.k:
        for name, s in evaluate(ids, k, RESULTS).items():
            print(
                f"k={k} {name:<7} key-passage hit {s['key_hit']}%  doc hit {s['doc_hit']}%  "
                f"(n={s['n']}) misses: {', '.join(s['misses']) or 'none'}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
