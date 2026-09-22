"""FastAPI app. The PWA is served as static files from here in week 3."""

import logging
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from backend.app.config import get_settings
from backend.app.graph.pipeline import build_deps, run_triage
from backend.app.graph.state import Deps, TriageRequest
from backend.app.llm.spend import SpendLimitError
from backend.app.schemas import TriageResult

settings = get_settings()
logging.basicConfig(level=settings.log_level)

app = FastAPI(title="Iba", description="Fever-triage decision support. Not a diagnosis.")


@lru_cache
def get_deps() -> Deps:
    return build_deps(settings)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "env": settings.app_env}


@app.post("/triage", response_model=TriageResult)
def triage(request: TriageRequest) -> TriageResult:
    """Triage a febrile patient.

    If the result has status "needs_info", show `questions`, then POST the same `text`
    again with `answers` (or `skip_questions: true`) to resume. No session is stored:
    the case text is never persisted server-side.
    """
    try:
        return run_triage(request, get_deps())
    except SpendLimitError as exc:
        # A budget stop is an operator problem, never a triage answer.
        logging.getLogger("iba.api").error("spend limit: %s", exc)
        raise HTTPException(
            status_code=503, detail="Iba is temporarily unavailable (spending limit reached)."
        ) from exc
