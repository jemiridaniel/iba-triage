"""Shared data models. Grows as pipeline nodes land (TriageResult etc. in week 2)."""

from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class TriageLevel(StrEnum):
    TREAT_MONITOR = "treat_monitor"
    REFER_24H = "refer_24h"
    REFER_NOW = "refer_now"

    @property
    def rank(self) -> int:
        return _TRIAGE_RANK[self]


_TRIAGE_RANK = {
    TriageLevel.TREAT_MONITOR: 0,
    TriageLevel.REFER_24H: 1,
    TriageLevel.REFER_NOW: 2,
}


def most_urgent(*levels: TriageLevel | None) -> TriageLevel | None:
    present = [lvl for lvl in levels if lvl is not None]
    return max(present, key=lambda lvl: lvl.rank) if present else None


class DangerSignCode(StrEnum):
    """Adult severe malaria criteria + IMCI general danger signs (SPEC §5)."""

    CONVULSIONS = "convulsions"
    UNABLE_TO_DRINK = "unable_to_drink_or_breastfeed"
    VOMITING_EVERYTHING = "vomiting_everything"
    LETHARGY_UNCONSCIOUS = "lethargy_or_unconsciousness"
    BLEEDING = "bleeding"
    JAUNDICE = "jaundice"
    RESPIRATORY_DISTRESS = "respiratory_distress"
    SEVERE_PALLOR = "severe_pallor"
    DARK_URINE = "dark_urine"
    PROSTRATION = "prostration"


DANGER_SIGN_LABELS: dict[DangerSignCode, str] = {
    DangerSignCode.CONVULSIONS: "Convulsions",
    DangerSignCode.UNABLE_TO_DRINK: "Unable to drink or breastfeed",
    DangerSignCode.VOMITING_EVERYTHING: "Vomiting everything",
    DangerSignCode.LETHARGY_UNCONSCIOUS: "Lethargy or unconsciousness",
    DangerSignCode.BLEEDING: "Abnormal bleeding",
    DangerSignCode.JAUNDICE: "Jaundice",
    DangerSignCode.RESPIRATORY_DISTRESS: "Respiratory distress",
    DangerSignCode.SEVERE_PALLOR: "Severe pallor",
    DangerSignCode.DARK_URINE: "Dark (cola-coloured) urine",
    DangerSignCode.PROSTRATION: "Prostration (unable to sit/stand)",
}


class DangerSignHit(BaseModel):
    code: DangerSignCode
    # "rule": matched by deterministic text rules; "llm_*": flagged by a model step.
    source: Literal["rule", "llm_intake", "llm_reason"]
    evidence: str | None = None

    @property
    def label(self) -> str:
        return DANGER_SIGN_LABELS[self.code]


class PatientCase(BaseModel):
    """Normalised case produced by the intake step. No names or identifiers, ever."""

    age_years: float | None = Field(default=None, ge=0, le=120)
    sex: Literal["male", "female"] | None = None
    weight_kg: float | None = Field(default=None, gt=0, le=250)
    pregnant: bool | None = None
    fever_days: float | None = Field(default=None, ge=0)
    temperature_c: float | None = Field(default=None, ge=30, le=45)
    rdt_result: Literal["positive", "negative", "not_done"] | None = None
    symptoms: list[str] = []
    danger_signs: list[DangerSignCode] = []  # flagged by the intake model
    antimalarial_taken: bool | None = None
    antimalarial_no_response: bool | None = None
    state: str | None = None
    lga: str | None = None
    language: Literal["en", "pcm"] = "en"
    missing_fields: list[str] = []
    # Original free text. Excluded from serialisation so it is never logged or persisted.
    raw_text: str | None = Field(default=None, exclude=True, repr=False)


class OutbreakSignal(BaseModel):
    disease: str
    state: str
    status: Literal["active", "declining", "over", "unknown"] = "unknown"
    report_date: date | None = None
    url: str | None = None


class GuidelineChunk(BaseModel):
    """One retrievable passage of a guideline document, with what's needed to cite it."""

    id: str  # "<doc_id>:<nnnn>", stable for a given document + chunker
    doc_id: str
    title: str
    section: str | None  # heading the chunk starts under
    sections: list[str] = []  # every heading the chunk spans
    page: int  # 1-based
    page_end: int
    text: str
    tokens: int  # approximate


class Citation(BaseModel):
    kind: Literal["guideline", "outbreak"]
    doc_id: str | None = None
    section: str | None = None
    page: int | None = None
    url: str | None = None
    source_date: date | None = None
