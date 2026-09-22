"""Shared data models. Grows as pipeline nodes land (TriageResult etc. in week 2)."""

from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, computed_field


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


TRIAGE_LABELS: dict[TriageLevel, str] = {
    TriageLevel.REFER_NOW: "Refer now",
    TriageLevel.REFER_24H: "Refer within 24h",
    TriageLevel.TREAT_MONITOR: "Treat & monitor",
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

    @computed_field
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
    ref: str  # chunk ID or outbreak URL, as cited by the model
    title: str | None = None
    doc_id: str | None = None
    section: str | None = None
    page: int | None = None
    url: str | None = None
    source_date: date | None = None


# --- pipeline outputs -------------------------------------------------------


class FollowUpQuestion(BaseModel):
    id: str  # the PatientCase field it fills, e.g. "fever_days"
    text: str


class FollowUpAnswer(BaseModel):
    id: str
    answer: str


class DifferentialItem(BaseModel):
    condition: str
    likelihood: Literal["high", "moderate", "low"]
    reasons: list[str] = []
    check_next: list[str] = []
    citations: list[str] = []
    source: Literal["llm", "rule"] = "llm"


class ActionItem(BaseModel):
    text: str
    citations: list[str] = []
    source: Literal["llm", "rule"] = "llm"


class DoseRecommendation(BaseModel):
    drug: str
    regimen: str
    weight_band: str
    verified: bool  # False until checked against the guideline PDF
    citation: Citation


class OutbreakContext(BaseModel):
    status: Literal["ok", "unavailable"]
    source: Literal["tavily", "mock", "none"]
    message: str
    checked_at: datetime | None = None
    signals: list[OutbreakSignal] = []


class TraceStep(BaseModel):
    step: str
    kind: Literal["llm", "rules", "retrieval", "tool"]
    model: str | None = None
    reasoning: bool | None = None
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int | None = None
    latency_ms: float = 0.0
    cost_usd: float | None = None
    cached: bool = False
    note: str | None = None


class DecisionTrace(BaseModel):
    steps: list[TraceStep] = []

    @computed_field
    @property
    def total_latency_ms(self) -> float:
        return round(sum(s.latency_ms for s in self.steps), 1)

    @computed_field
    @property
    def total_tokens(self) -> int:
        return sum(s.prompt_tokens + s.completion_tokens for s in self.steps)

    @computed_field
    @property
    def total_cost_usd(self) -> float:
        return round(sum(s.cost_usd or 0.0 for s in self.steps), 6)


DISCLAIMER = (
    "Decision support only, not a diagnosis. A qualified health worker must confirm every "
    "decision. Doses come from the guideline table only."
)


class TriageResult(BaseModel):
    # complete: full assessment; needs_info: answer `questions` and resubmit;
    # incomplete: Iba could not finish and failed safe to "Refer now".
    status: Literal["complete", "needs_info", "incomplete"]
    triage_level: TriageLevel | None = None
    triage_label: str | None = None
    triage_rationale: str | None = None
    questions: list[FollowUpQuestion] = []
    case: PatientCase | None = None
    danger_signs: list[DangerSignHit] = []
    lassa_suspected: bool = False
    outbreak: OutbreakContext | None = None
    differential: list[DifferentialItem] = []
    actions: list[ActionItem] = []
    doses: list[DoseRecommendation] = []
    citations: list[Citation] = []
    summary: str | None = None
    referral_note: str | None = None
    warnings: list[str] = []
    disclaimer: str = DISCLAIMER
    decision_trace: DecisionTrace = DecisionTrace()
