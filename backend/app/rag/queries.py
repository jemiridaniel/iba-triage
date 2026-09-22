"""Retrieval planning: one query per suspected condition and purpose, with a doc prior.

The first version embedded a single symptom query per case, so the 494-page WHO malaria
guideline (70% of the index) dominated every result and chunks rarely matched the claims the
reasoning model made. Here we:
  1. derive suspected conditions deterministically (rules, RDT, text cues, outbreak context);
  2. build a query per condition and purpose (classification, treatment, referral, IPC);
  3. boost the guideline that owns each condition (small additive cosine prior);
  4. embed every query in one API call and merge the results.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from backend.app.rag.store import SearchHit, VectorStore
from backend.app.rules.text import ascii_punct
from backend.app.schemas import OutbreakContext, PatientCase

Embedder = Callable[[list[str]], list[list[float]]]

DOC_PRIOR = 0.06  # additive cosine boost for the condition's own guideline


@dataclass(frozen=True)
class Condition:
    key: str
    name: str
    docs: tuple[str, ...]  # preferred doc_ids (boosted)
    purposes: dict[str, str]  # purpose -> query text


CONDITIONS: dict[str, Condition] = {
    c.key: c
    for c in [
        Condition(
            "severe_malaria",
            "severe malaria",
            ("who-malaria", "who-imci"),
            {
                "classification": "severe malaria clinical features: impaired consciousness, prostration, convulsions, jaundice, bleeding",
                "referral": "refer urgently, pre-referral treatment for severe malaria",
            },
        ),
        Condition(
            "uncomplicated_malaria",
            "uncomplicated malaria",
            ("who-malaria", "who-imci"),
            {
                "classification": "malaria confirmed by parasitological test with no features of severe malaria is uncomplicated malaria",
                "treatment": "treat uncomplicated falciparum malaria with artemisinin-based combination therapy",
                "follow_up": "follow-up in 3 days if fever persists, return immediately if danger signs",
            },
        ),
        Condition(
            "fever_no_malaria",
            "fever with negative malaria test",
            ("who-imci", "who-malaria"),
            {
                "classification": "malaria test negative, look for other cause of fever",
                "follow_up": "fever present every day for more than 7 days refer for assessment",
            },
        ),
        Condition(
            "danger_signs_child",
            "general danger signs in a child",
            ("who-imci",),
            {
                "classification": "general danger signs: not able to drink or breastfeed, vomits everything, convulsions, lethargic or unconscious",
                "referral": "very severe febrile disease: give first dose and refer urgently to hospital",
            },
        ),
        Condition(
            "lassa",
            "Lassa fever",
            ("ncdc-lassa", "ncdc-vhf-ipc", "ncdc-lassa-advisory-2026"),
            {
                "classification": "Lassa fever suspected case definition: fever 3-21 days not responding to antimalarials",
                "referral": "suspected Lassa fever case triage: holding area, alert, notify, refer to treatment centre",
                "ipc": "infection prevention and control for viral haemorrhagic fever: isolation, gloves, body fluids",
            },
        ),
        Condition(
            "cholera",
            "cholera / acute watery diarrhoea",
            ("ncdc-cholera", "who-imci"),
            {
                "classification": "assessment for level of dehydration: severe, some, no dehydration",
                "treatment": "oral rehydration solution, plan A plan B plan C for dehydration",
                "referral": "suspected cholera case definition and referral",
            },
        ),
        Condition(
            "meningitis",
            "meningitis",
            ("ncdc-csm", "ncdc-csm-quickref", "who-imci"),
            {
                "classification": "fever with stiff neck, very severe febrile disease, suspected meningitis",
                "referral": "suspected meningitis refer urgently, first dose of antibiotic",
            },
        ),
        Condition(
            "pneumonia",
            "pneumonia",
            ("who-imci",),
            {
                "classification": "cough or difficult breathing, chest indrawing, fast breathing, pneumonia classification",
            },
        ),
    ]
}

_CUES: dict[str, re.Pattern[str]] = {
    "cholera": re.compile(r"diarr|watery|stool|rice water|dehydrat", re.I),
    "meningitis": re.compile(r"neck|fontanel|stiff|purpur|rash that does not fade", re.I),
    "pneumonia": re.compile(r"cough|breath|chest indrawing|grunt", re.I),
}


@dataclass
class QueryPlan:
    condition: str
    purpose: str
    text: str
    boost: dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalLog:
    condition: str
    purpose: str
    query: str
    hits: list[str]  # chunk IDs, best first


def suspected_conditions(
    case: PatientCase, danger_signs: bool, outbreak: OutbreakContext | None, lassa_rule: bool
) -> list[str]:
    text = ascii_punct(" ".join([case.raw_text or "", *case.symptoms]))
    out: list[str] = []
    child = case.age_years is not None and case.age_years < 5
    if danger_signs:
        if child:
            out.append("danger_signs_child")
        if case.rdt_result == "positive" or not child:
            out.append("severe_malaria")
    if case.rdt_result == "positive" and not danger_signs:
        out.append("uncomplicated_malaria")
    if case.rdt_result == "negative":
        out.append("fever_no_malaria")
    signals = outbreak.all_signals if outbreak else []
    lassa_place = any("lassa" in s.disease.lower() for s in signals)
    if lassa_rule or (lassa_place and (case.fever_days or 0) >= 3):
        out.append("lassa")
    for key, rx in _CUES.items():
        if rx.search(text):
            out.append(key)
    if not out:
        out.append("uncomplicated_malaria" if case.rdt_result != "negative" else "fever_no_malaria")
    return list(dict.fromkeys(out))


def plan_queries(case: PatientCase, conditions: list[str]) -> list[QueryPlan]:
    who = "child" if (case.age_years is not None and case.age_years < 5) else "patient"
    plans = []
    for key in conditions:
        cond = CONDITIONS[key]
        boost = dict.fromkeys(cond.docs, DOC_PRIOR)
        for purpose, text in cond.purposes.items():
            plans.append(QueryPlan(key, purpose, f"{text} ({who})", boost))
    return plans


def retrieve(
    store: VectorStore, embed: Embedder, plans: list[QueryPlan], k_total: int, k_per_query: int = 2
) -> tuple[list[SearchHit], list[RetrievalLog]]:
    """Embed all queries in one call; take the best k_per_query per query, then fill to k_total."""
    if not plans:
        return [], []
    vectors = embed([p.text for p in plans])
    per_query = [
        store.search(v, k=max(k_total, 4), boost=p.boost)
        for p, v in zip(plans, vectors, strict=True)
    ]
    logs = [
        RetrievalLog(p.condition, p.purpose, p.text, [h.chunk.id for h in hits[:k_total]])
        for p, hits in zip(plans, per_query, strict=True)
    ]
    chosen: dict[str, SearchHit] = {}
    for rank in range(k_per_query):  # round-robin: every query's best hit before any second-best
        for hits in per_query:
            if rank < len(hits) and len(chosen) < k_total:
                chosen.setdefault(hits[rank].chunk.id, hits[rank])
    pool = sorted((h for hits in per_query for h in hits), key=lambda h: -h.score)
    for h in pool:
        if len(chosen) >= k_total:
            break
        chosen.setdefault(h.chunk.id, h)
    return list(chosen.values()), logs
