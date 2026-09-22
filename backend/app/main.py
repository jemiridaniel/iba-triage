"""FastAPI app. The PWA is served as static files from here in week 3."""

import json
import logging
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.app.config import get_settings
from backend.app.graph.pipeline import build_deps, failsafe_result, run_triage, stream_triage
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
    except Exception as exc:  # never a silent green: fail safe to Refer now
        logging.getLogger("iba.api").exception("pipeline error")
        return failsafe_result(request, exc)


@app.get("/meta")
def meta() -> dict:
    """Sources, licences, models and data status for the About page. No secrets."""
    from backend.app.rag.ingest import load_sources
    from backend.app.rules.dosing import TABLE_VERIFIED
    from backend.app.tools.endemicity import Endemicity

    deps = get_deps()
    cfg = deps.settings
    return {
        "models": {
            "fast": cfg.model_fast,
            "mid": cfg.model_mid,
            "reason": cfg.model_reason,
            "embed": cfg.model_embed,
        },
        "reasoning": {
            step: getattr(cfg, f"reasoning_{step}")
            for step in ("intake", "outbreak", "reason", "compose")
        },
        "sources": [
            s.model_dump(include={"doc_id", "title", "edition", "url", "licence", "licence_note"})
            for s in load_sources()
        ],
        "index": {
            "chunks": len(deps.store) if deps.store else 0,
            "model": deps.store.model if deps.store else None,
        },
        "live_outbreak_search": deps.outbreak.search.source if deps.outbreak.search else None,
        "dose_table_verified": TABLE_VERIFIED,
        "endemicity_verified": Endemicity.load(cfg.endemicity_file).all_verified,
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@app.post("/triage/stream")
def triage_stream(request: TriageRequest) -> StreamingResponse:
    """Server-sent events: one event per pipeline node (intake, rules_pre, retrieve,
    outbreak, reason, rules_post, compose), then "final" with the full TriageResult.
    Errors arrive as "error" events; the fail-safe still applies."""
    deps = get_deps()
    events = (_sse(event, data) for event, data in stream_triage(request, deps))
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- built frontend (production) -----------------------------------------------

_DIST = Path(settings.frontend_dist)
if (_DIST / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        """Serve built files, falling back to index.html for client-side routes (/about)."""
        candidate = (_DIST / path).resolve()
        if path and candidate.is_file() and _DIST.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(_DIST / "index.html")
