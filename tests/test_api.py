"""POST /triage, including the follow-up question / resume flow. No network."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.app.main as main
from tests.pipeline_fakes import COMPOSE_EN, make_deps
from tests.test_pipeline_e2e import MALARIA_INTAKE, MALARIA_REASON


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def install(replies, **kw):
        deps, llm = make_deps(tmp_path, replies, search=None, **kw)
        monkeypatch.setattr(main, "get_deps", lambda: deps)
        return TestClient(main.app), llm

    return install


def test_triage_returns_result_and_trace(api) -> None:
    client, _ = api({"intake": MALARIA_INTAKE, "reason": MALARIA_REASON, "compose": COMPOSE_EN})
    resp = client.post(
        "/triage",
        json={"text": "Adult woman 28 years, 62 kg, fever 2 days, RDT positive", "state": "Lagos"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "complete"
    assert body["triage_level"] == "treat_monitor"
    steps = body["decision_trace"]["steps"]
    assert [s["step"] for s in steps][0] == "intake"
    assert {"model", "reasoning", "prompt_tokens", "latency_ms", "cost_usd"} <= set(steps[0])
    assert "total_cost_usd" in body["decision_trace"]
    assert "raw_text" not in body["case"]


def test_needs_info_then_resume(api) -> None:
    sparse = {"age_years": 28, "language": "en"}
    client, _ = api(
        {"intake": [sparse, MALARIA_INTAKE], "reason": MALARIA_REASON, "compose": COMPOSE_EN}
    )
    first = client.post("/triage", json={"text": "Woman with fever"}).json()
    assert first["status"] == "needs_info"
    answers = [
        {"id": q["id"], "answer": "2" if q["id"] == "fever_days" else "positive"}
        for q in first["questions"]
    ]
    second = client.post("/triage", json={"text": "Woman with fever", "answers": answers}).json()
    assert second["status"] == "complete"


def test_spend_limit_is_503_not_a_triage(api) -> None:
    client, _ = api({"intake": MALARIA_INTAKE}, max_spend_usd=0.0)
    resp = client.post("/triage", json={"text": "Adult woman with fever"})
    assert resp.status_code == 503
    assert "spending limit" in resp.json()["detail"]


def test_validation(api) -> None:
    client, _ = api({})
    assert client.post("/triage", json={"text": ""}).status_code == 422
