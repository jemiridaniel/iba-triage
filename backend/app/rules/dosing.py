"""Weight-band dose lookup and dose stripping. The ONLY source of doses in Iba.

The LLM never generates doses (CLAUDE.md rule 3). `strip_doses` removes dose-like text from
model output; `doses_for` attaches doses from the table below, with a citation.

!!! UNVERIFIED STUB !!!
The table mirrors the widely published artemether-lumefantrine weight bands but has NOT yet
been checked against the NMEP national guideline PDF (edition, section and page TBD). Every
dose is returned with verified=False and the UI shows a warning until TABLE_VERIFIED is set
to True after a human check against the source.
"""

import re
from dataclasses import dataclass

from backend.app.schemas import Citation, DoseRecommendation, PatientCase, TriageLevel

TABLE_VERIFIED = False
UNVERIFIED_WARNING = (
    "Dose table is an UNVERIFIED stub pending confirmation against the national malaria "
    "guideline. Check every dose against the printed guideline before use."
)

_SOURCE = Citation(
    kind="guideline",
    ref="nmep-malaria:dosing-table",
    title="National Guidelines for the Diagnosis and Treatment of Malaria (NMEP)",
    doc_id="nmep-malaria",
    section="TODO: artemether-lumefantrine weight-band table (confirm section/page)",
)


@dataclass(frozen=True)
class WeightBand:
    min_kg: float  # inclusive
    max_kg: float | None  # exclusive; None = no upper bound
    tablets_per_dose: int

    @property
    def label(self) -> str:
        if self.max_kg is None:
            return f"{self.min_kg:g} kg and above"
        return f"{self.min_kg:g} to under {self.max_kg:g} kg"


AL_DRUG = "Artemether-lumefantrine 20/120 mg tablets"
AL_SCHEDULE = "at 0 h and 8 h on day 1, then twice daily on days 2 and 3 (6 doses)"
AL_BANDS = (
    WeightBand(5, 15, 1),
    WeightBand(15, 25, 2),
    WeightBand(25, 35, 3),
    WeightBand(35, None, 4),
)


def al_band(weight_kg: float) -> WeightBand | None:
    for band in AL_BANDS:
        if weight_kg >= band.min_kg and (band.max_kg is None or weight_kg < band.max_kg):
            return band
    return None


def doses_for(case: PatientCase, level: TriageLevel) -> tuple[list[DoseRecommendation], list[str]]:
    """Doses for confirmed uncomplicated malaria managed at the PHC; notes explain omissions.

    Referred patients get no table dose here: pre-referral treatment is not in the table yet.
    """
    if level != TriageLevel.TREAT_MONITOR or case.rdt_result != "positive":
        return [], []
    if case.weight_kg is None:
        return [], ["Weigh the patient: the antimalarial dose depends on the weight band."]
    band = al_band(case.weight_kg)
    if band is None:
        return [], [
            f"Weight {case.weight_kg:g} kg is below the dose table (under 5 kg): "
            "consult or refer; no table dose."
        ]
    tablets = f"{band.tablets_per_dose} tablet{'s' if band.tablets_per_dose > 1 else ''}"
    dose = DoseRecommendation(
        drug=AL_DRUG,
        regimen=f"{tablets} per dose, {AL_SCHEDULE}",
        weight_band=band.label,
        verified=TABLE_VERIFIED,
        citation=_SOURCE,
    )
    return [dose], ([] if TABLE_VERIFIED else [UNVERIFIED_WARNING])


# --- stripping --------------------------------------------------------------

DOSE_PLACEHOLDER = "[dose: see Iba dose table]"
_NUM = r"\d+(?:[.,]\d+)?"
_DOSE_RE = re.compile(
    rf"(?<![\w.])(?:{_NUM}\s*/\s*)?{_NUM}(?:\s*(?:-|–|to)\s*{_NUM})?\s*"
    r"(?:mg|mcg|µg|g|ml|mls|iu|units?|tabs?|tablets?|caps?|capsules?|sachets?|drops?|puffs?)"
    r"(?:\s*/\s*kg)?\b",
    re.IGNORECASE,
)


def strip_doses(text: str) -> tuple[str, int]:
    """Replace dose amounts ("20/120 mg", "4 tablets", "10 mg/kg") with a placeholder."""
    return _DOSE_RE.subn(DOSE_PLACEHOLDER, text)
