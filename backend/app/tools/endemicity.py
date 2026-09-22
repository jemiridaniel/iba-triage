"""Static endemicity baseline (data/endemicity.yaml).

Always included in the outbreak step so triage is never blind when live search is down.
Entries are OutbreakSignal(basis="baseline", status="endemic"); they never satisfy the live
outbreak rule (which needs a dated, sourced, active live signal).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from backend.app.schemas import OutbreakSignal

DEFAULT_ENDEMICITY = Path("data/endemicity.yaml")

# (doc_id, phrase) -> chunk ID, or None if that guideline isn't indexed.
AnchorResolver = Callable[[str, str], str | None]


def _norm(state: str) -> str:
    s = re.sub(r"\s+state$", "", state.strip().lower())
    return "fct" if s in {"abuja", "federal capital territory", "fct abuja"} else s


@dataclass(frozen=True)
class EndemicEntry:
    key: str
    disease: str
    states: tuple[str, ...]
    peak_months: tuple[int, ...]
    verified: bool
    source: str
    anchor: tuple[str, str] | None


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
            entries.append(
                EndemicEntry(
                    key=key,
                    disease=e["disease"],
                    states=tuple(e.get("states") or ()),
                    peak_months=tuple(e.get("peak_months") or ()),
                    verified=bool(e.get("verified")),
                    source=str(e.get("source") or "").strip(),
                    anchor=(anchor[0], anchor[1]) if anchor else None,
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
            match = next((s for s in e.states if _norm(s) == target), None)
            if match is None:
                continue
            citation = resolve(*e.anchor) if (e.anchor and resolve) else None
            out.append(
                OutbreakSignal(
                    disease=e.disease,
                    state=match,
                    status="endemic",
                    basis="baseline",
                    in_season=today.month in e.peak_months if e.peak_months else None,
                    citation=citation,
                )
            )
        return out
