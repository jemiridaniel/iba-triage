"""Static endemicity baseline (data/endemicity.yaml).

Always included in the outbreak step so triage is never blind when live search is down.
Entries are OutbreakSignal(basis="baseline", status="endemic"); they never satisfy the live
outbreak rule (which needs a dated, sourced, active live signal). Within baseline, only
tier="high_burden" entries feed a deterministic escalation rule (see
backend/app/rules/danger_signs.py::endemic_baseline) -- tier="reported" entries inform the
differential only, and never raise the triage level on their own.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import yaml

from backend.app.schemas import OutbreakSignal

DEFAULT_ENDEMICITY = Path("data/endemicity.yaml")

# A baseline report labelled "older surveillance" once it's more than ~6 months old. This is
# purely descriptive (UI + reasoning prompt); unlike live signals, it never gates escalation
# -- tier does that.
OLDER_SURVEILLANCE_DAYS = 183

# (doc_id, phrase) -> chunk ID, or None if that guideline isn't indexed.
AnchorResolver = Callable[[str, str], str | None]


def _norm(state: str) -> str:
    s = re.sub(r"\s+state$", "", state.strip().lower())
    return "fct" if s in {"abuja", "federal capital territory", "fct abuja"} else s


def _recency(report_date: date | None, today: date) -> Literal["current", "older", "unknown"]:
    if report_date is None:
        return "unknown"
    return "older" if (today - report_date).days > OLDER_SURVEILLANCE_DAYS else "current"


@dataclass(frozen=True)
class EndemicState:
    name: str
    confirmed: bool = True  # False = carried over from an earlier list; this report doesn't
    # confirm or rule it out (see data/endemicity.yaml's Lassa/Ebonyi entry for why).


def _parse_states(raw: object) -> tuple[EndemicState, ...]:
    out = []
    for item in raw or ():
        if isinstance(item, str):
            out.append(EndemicState(item))
        else:
            out.append(EndemicState(item["state"], confirmed=bool(item.get("confirmed", True))))
    return tuple(out)


@dataclass(frozen=True)
class EndemicEntry:
    key: str
    disease: str
    verified: bool
    report: str
    epi_week: str
    report_date: date | None
    page: int | None
    source: str
    peak_months: tuple[int, ...]
    anchor: tuple[str, str] | None
    high_burden: tuple[EndemicState, ...]
    reported: tuple[EndemicState, ...]

    @property
    def all_states(self) -> list[tuple[EndemicState, Literal["high_burden", "reported"]]]:
        return [(s, "high_burden") for s in self.high_burden] + [
            (s, "reported") for s in self.reported
        ]


class Endemicity:
    def __init__(self, entries: list[EndemicEntry]):
        self.entries = entries

    @classmethod
    def load(cls, path: Path = DEFAULT_ENDEMICITY) -> Endemicity:
        if not path.exists():
            return cls([])
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        entries = []
        for key, e in (data.get("diseases") or {}).items():
            anchor = e.get("anchor")
            rd = e.get("report_date")
            entries.append(
                EndemicEntry(
                    key=key,
                    disease=e["disease"],
                    verified=bool(e.get("verified")),
                    report=str(e.get("report") or ""),
                    epi_week=str(e.get("epi_week") or ""),
                    report_date=date.fromisoformat(str(rd)) if rd else None,
                    page=e.get("page"),
                    source=str(e.get("source") or "").strip(),
                    peak_months=tuple(e.get("peak_months") or ()),
                    anchor=(anchor[0], anchor[1]) if anchor else None,
                    high_burden=_parse_states(e.get("high_burden")),
                    reported=_parse_states(e.get("reported")),
                )
            )
        return cls(entries)

    @property
    def all_verified(self) -> bool:
        return all(e.verified for e in self.entries)

    def for_state(
        self, state: str | None, today: date, resolve: AnchorResolver | None = None
    ) -> list[OutbreakSignal]:
        if not state:
            return []
        target = _norm(state)
        out = []
        for e in self.entries:
            for est, tier in e.all_states:
                if _norm(est.name) != target:
                    continue
                citation = resolve(*e.anchor) if (e.anchor and resolve) else None
                out.append(
                    OutbreakSignal(
                        disease=e.disease,
                        state=est.name,
                        status="endemic",
                        basis="baseline",
                        in_season=today.month in e.peak_months if e.peak_months else None,
                        citation=citation,
                        tier=tier,
                        confirmed=est.confirmed,
                        report=e.report or None,
                        epi_week=e.epi_week or None,
                        page=e.page,
                        report_date=e.report_date,
                        recency=_recency(e.report_date, today),
                    )
                )
        return out
