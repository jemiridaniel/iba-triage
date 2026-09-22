"""FastAPI entry point. The triage endpoint and static frontend are added in weeks 2-3."""

import logging

from fastapi import FastAPI

from backend.app.config import get_settings

settings = get_settings()
logging.basicConfig(level=settings.log_level)

app = FastAPI(title="Iba", description="Fever-triage decision support. Not a diagnosis.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "env": settings.app_env}
