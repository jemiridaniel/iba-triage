"""Weight-band dose lookup and dose stripping. The ONLY source of doses in Ibà.

The LLM never generates doses (CLAUDE.md rule 3). `strip_doses` removes dose-like text from
model output; `doses_for` attaches doses from the table below, with a citation.

Source: NMEP *National Guidelines for Diagnosis and Treatment of Malaria* (4th Edition, May
2020), §4.2.1 Artemether-Lumefantrine, Table 4-4 "Dosage Regimens for Artemether-Lumefantrine
Treatment" (p.10 of the document; PDF page 24), transcribed exactly, NA cells included.

Schedule wording: NMEP's own text is "twice daily x 3days" / "twice daily x 3 days" (the
table's own cells have inconsistent spacing around "3 days" -- a PDF-extraction artefact, not
a meaningful difference) and, in the paragraph right below the table, "the 6 doses must be
taken by the patient". **NMEP does not give an hour-by-hour breakdown (no "0 h and 8 h" text
anywhere in the document)** -- that level of detail is a WHO convention, not stated here, so
it is never attributed to this citation.

Verified against the NMEP 4th edition (May 2020) Table 4-4, p.10 by DJ, 2026-09-28: all four
weight bands and their boundaries, every tablet count across the three strengths (including
the NA cells), matched the printed table exactly.
"""

import re
from dataclasses import dataclass

from backend.app.rules.text import ascii_punct
from backend.app.schemas import (
    Citation,
    DoseRecommendation,
    GuidelineNote,
    PatientCase,
    TriageLevel,
)

TABLE_VERIFIED = True  # Verified against NMEP 4th ed. (May 2020) Table 4-4, p.10 by DJ, 2026-09-28
UNVERIFIED_WARNING = (
    "Dose table is an UNVERIFIED stub pending confirmation against the national malaria "
    "guideline. Check every dose against the printed guideline before use."
)

_NMEP_TITLE = (
    "National Guidelines for Diagnosis and Treatment of Malaria (NMEP, 4th Edition, May 2020)"
)
_TABLE_CITATION = Citation(
    kind="guideline",
    ref="nmep-malaria:table-4-4",
    title=_NMEP_TITLE,
    doc_id="nmep-malaria",
    section="Table 4-4: Dosage Regimens for Artemether-Lumefantrine Treatment",
    page=10,
)
_ABSORPTION_CITATION = Citation(
    kind="guideline",
    ref="nmep-malaria:al-absorption-note",
    title=_NMEP_TITLE,
    doc_id="nmep-malaria",
    section="Note following Tables 4.2-4.4 (absorption)",
    page=10,
)


@dataclass(frozen=True)
class WeightBand:
    min_kg: float  # inclusive
    max_kg: float | None  # exclusive; None = no upper bound

    @property
    def label(self) -> str:
        if self.max_kg is None:
            return f"{self.min_kg:g} kg and above"
        return f"{self.min_kg:g} to under {self.max_kg:g} kg"


AL_BANDS = (
    WeightBand(5, 15),
    WeightBand(15, 25),
    WeightBand(25, 35),
    WeightBand(35, None),
)

# Table 4-4, transcribed exactly: tablets per dose, indexed by AL_BANDS position, per
# strength. None = "NA" in the table -- this strength has no dose for that weight band.
# There is deliberately no fallback: an NA cell must never borrow another strength's count.
AL_TABLETS_PER_DOSE: dict[str, tuple[int | None, int | None, int | None, int | None]] = {
    "20/120": (1, 2, 3, 4),
    "40/240": (None, 1, None, 2),
    "80/480": (None, None, None, 1),
}
AL_STRENGTH_ORDER = ("20/120", "40/240", "80/480")
DEFAULT_STRENGTH = "20/120"

# NMEP's own wording (see module docstring); no 0h/8h breakdown is stated in the source.
AL_SCHEDULE = "twice daily for 3 days (6 doses)"
AL_SPLITTING_NOTE = "Tablet splitting is not recommended (Table 4-4 caption)."
AL_ABSORPTION_NOTE = (
    "Absorption is enhanced by fatty meals: give after a meal, or with a tablespoon of milk."
)


def al_band_index(weight_kg: float) -> int | None:
    for i, band in enumerate(AL_BANDS):
        if weight_kg >= band.min_kg and (band.max_kg is None or weight_kg < band.max_kg):
            return i
    return None


def guideline_notes() -> list[GuidelineNote]:
    """The two cited pieces of guideline advice that go with an AL dose, not a dose itself."""
    return [
        GuidelineNote(text=AL_SPLITTING_NOTE, citation=_TABLE_CITATION),
        GuidelineNote(text=AL_ABSORPTION_NOTE, citation=_ABSORPTION_CITATION),
    ]


def doses_for(case: PatientCase, level: TriageLevel) -> tuple[list[DoseRecommendation], list[str]]:
    """Doses for confirmed uncomplicated malaria managed at the PHC; notes explain omissions.

    Returns every strength that has a dose for the patient's weight band (20/120 first, the
    UI's default), plus a plain-text note for every strength that doesn't (never a fallback
    to a different strength's tablet count).

    Referred patients get no table dose here: pre-referral treatment is not in the table yet.
    """
    if level != TriageLevel.TREAT_MONITOR or case.rdt_result != "positive":
        return [], []
    if case.weight_kg is None:
        return [], ["Weigh the patient: the antimalarial dose depends on the weight band."]
    idx = al_band_index(case.weight_kg)
    if idx is None:
        return [], [
            f"Weight {case.weight_kg:g} kg is below the dose table (under 5 kg): "
            "consult or refer; no table dose."
        ]
    band = AL_BANDS[idx]

    doses: list[DoseRecommendation] = []
    notes: list[str] = [] if TABLE_VERIFIED else [UNVERIFIED_WARNING]
    for strength in AL_STRENGTH_ORDER:
        tablets = AL_TABLETS_PER_DOSE[strength][idx]
        if tablets is None:
            notes.append(
                f"Artemether-lumefantrine {strength} mg tablets: not suitable for "
                f"{band.label} (Table 4-4, NMEP 4th ed. 2020, p.10) — use a strength "
                "that has a dose for this weight band."
            )
            continue
        tab_word = f"{tablets} tablet{'s' if tablets > 1 else ''}"
        doses.append(
            DoseRecommendation(
                drug=f"Artemether-lumefantrine {strength} mg tablets",
                regimen=f"{tab_word} per dose, {AL_SCHEDULE}",
                weight_band=band.label,
                strength=strength,
                is_default=(strength == DEFAULT_STRENGTH),
                verified=TABLE_VERIFIED,
                citation=_TABLE_CITATION,
            )
        )
    return doses, notes


# --- stripping --------------------------------------------------------------

DOSE_PLACEHOLDER = "[dose: see Ibà dose table]"
_NUM = r"\d+(?:[.,]\d+)?"
_DOSE_RE = re.compile(
    rf"(?<![\w.])(?:{_NUM}\s*/\s*)?{_NUM}(?:\s*(?:-|–|to)\s*{_NUM})?\s*"
    r"(?:mg|mcg|µg|g|ml|mls|iu|units?|tabs?|tablets?|caps?|capsules?|sachets?|drops?|puffs?)"
    r"(?:\s*/\s*kg)?\b",
    re.IGNORECASE,
)


def strip_doses(text: str) -> tuple[str, int]:
    """Replace dose amounts ("20/120 mg", "4 tablets", "10 mg/kg") with a placeholder."""
    return _DOSE_RE.subn(DOSE_PLACEHOLDER, ascii_punct(text))
