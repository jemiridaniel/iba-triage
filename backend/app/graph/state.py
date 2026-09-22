"""Typed pipeline state, request, model-output schemas and dependencies."""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator

from backend.app.config import Settings
from backend.app.llm.client import LLMClient
from backend.app.llm.router import EvalConfig
from backend.app.rag.store import SearchHit, VectorStore
from backend.app.rules.danger_signs import RuleAssessment
from backend.app.schemas import (
    ActionItem,
    Citation,
    DangerSignCode,
    DangerSignHit,
    DifferentialItem,
    DoseRecommendation,
    FollowUpAnswer,
    FollowUpQuestion,
    GroundingSummary,
    OutbreakContext,
    OutbreakSignal,
    PatientCase,
    RetrievalQuery,
    TraceStep,
    TriageLevel,
)
from backend.app.tools.outbreak import OutbreakTool

Embedder = Callable[[list[str]], list[list[float]]]


class TriageRequest(BaseModel):
    text: str = Field(min_length=3, max_length=4000)
    state: str | None = None
    lga: str | None = None
    # Resume: resend the same text with answers to the questions from a needs_info result.
    answers: list[FollowUpAnswer] = []
    # Proceed without answering follow-up questions (assessment uses what is known).
    skip_questions: bool = False


# --- model output schemas ---------------------------------------------------

_LIKELIHOOD = {"medium": "moderate", "possible": "moderate", "likely": "high", "unlikely": "low"}


class ReasonDraft(BaseModel):
    text: str
    # "patient": restates the worker's description (no source needed);
    # "guideline": clinical knowledge that must be backed by chunk_id + evidence_quote.
    basis: Literal["patient", "guideline"] = "guideline"
    chunk_id: str | None = None
    evidence_quote: str | None = None


class DifferentialDraft(BaseModel):
    condition: str
    likelihood: Literal["high", "moderate", "low"]
    reasons: list[ReasonDraft] = []
    check_next: list[str] = []

    @field_validator("likelihood", mode="before")
    @classmethod
    def _norm_likelihood(cls, v: Any) -> Any:
        v = str(v).strip().lower()
        return _LIKELIHOOD.get(v, v)

    @field_validator("reasons", mode="before")
    @classmethod
    def _plain_strings_are_unsourced(cls, v: Any) -> Any:
        # A bare string has no evidence: keep it, it will be marked unsupported.
        return [{"text": r} if isinstance(r, str) else r for r in (v or [])]


class ActionDraft(BaseModel):
    text: str
    chunk_id: str | None = None
    evidence_quote: str | None = None


class ReasonOutput(BaseModel):
    triage_level: TriageLevel
    triage_rationale: str
    danger_signs_found: list[DangerSignCode] = []
    differential: list[DifferentialDraft] = Field(min_length=1)
    actions: list[ActionDraft] = []

    @field_validator("differential")
    @classmethod
    def _cap_differential(cls, v: list[DifferentialDraft]) -> list[DifferentialDraft]:
        return v[:4]


class ComposeOutput(BaseModel):
    summary: str
    referral_note: str


# --- rule snapshots & post-rule output --------------------------------------


class RuleSnapshot(BaseModel):
    floor: TriageLevel | None
    danger_signs: list[DangerSignHit] = []
    lassa_signal: OutbreakSignal | None = None
    reasons: list[str] = []
    reminders: list[str] = []

    @classmethod
    def of(cls, a: RuleAssessment) -> RuleSnapshot:
        return cls(
            floor=a.floor,
            danger_signs=a.danger_signs,
            lassa_signal=a.lassa_signal,
            reasons=a.reasons,
            reminders=a.reminders,
        )


class PostOutput(BaseModel):
    status: Literal["complete", "incomplete"]
    triage_level: TriageLevel
    triage_rationale: str
    danger_signs: list[DangerSignHit] = []
    lassa_suspected: bool = False
    differential: list[DifferentialItem] = []
    actions: list[ActionItem] = []
    doses: list[DoseRecommendation] = []
    citations: list[Citation] = []
    grounding: GroundingSummary = GroundingSummary()
    warnings: list[str] = []


class TriageState(BaseModel):
    request: TriageRequest
    eval_config: EvalConfig = "routed"
    case: PatientCase | None = None
    intake_error: str | None = None
    questions: list[FollowUpQuestion] = []
    pre: RuleSnapshot | None = None
    hits: list[SearchHit] = []
    retrieval: list[RetrievalQuery] = []
    retrieval_note: str | None = None  # set when retrieval is unavailable
    outbreak: OutbreakContext | None = None
    reason: ReasonOutput | None = None
    reason_error: str | None = None
    post: PostOutput | None = None
    compose: ComposeOutput | None = None
    compose_error: str | None = None
    trace: Annotated[list[TraceStep], operator.add] = []


@dataclass
class Deps:
    settings: Settings
    client: LLMClient
    outbreak: OutbreakTool
    store: VectorStore | None = None
    embed: Embedder | None = None
