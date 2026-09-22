"""POST /triage/stream (server-sent events). No network."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.app.main as main
from backend.app.graph import pipeline
from tests.pipeline_fakes import COMPOSE_EN, OUTBREAK_EXTRACTION, make_deps
from tests.test_pipeline_e2e import ADULT_INTAKE, ADULT_REASON_NO_LASSA, ADULT_TEXT


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


@pytest.fixture
def client_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def make(replies, **kw):
        deps, _ = make_deps(tmp_path, replies, **kw)
        monkeypatch.setattr(main, "get_deps", lambda: deps)
        return TestClient(main.app)

    return make


REPLIES = {
    "intake": ADULT_INTAKE,
    "outbreak extraction": OUTBREAK_EXTRACTION,
    "reason": ADULT_REASON_NO_LASSA,
    "compose": COMPOSE_EN,
}


def test_stream_emits_event_per_node_then_final(client_for) -> None:
    client = client_for(REPLIES)
    resp = client.post("/triage/stream", json={"text": ADULT_TEXT, "state": "Ondo"})
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(resp.text)
    assert [e for e, _ in events] == [
        "intake",
        "rules_pre",
        "retrieve",
        "outbreak",
        "reason",
        "rules_post",
        "compose",
        "final",
    ]
    data = dict(events)
    assert data["rules_pre"]["floor"] is None
    assert data["outbreak"]["outbreak"]["signals"][0]["disease"] == "Lassa fever"
    assert data["rules_post"]["triage_level"] == "refer_now"  # card before compose finishes
    assert data["final"]["status"] == "complete"
    assert data["final"]["decision_trace"]["steps"][0]["step"] == "intake"
    assert all(d["trace"]["step"] == e for e, d in events[:-1])


def test_stream_rules_pre_arrives_with_danger_signs(client_for) -> None:
    intake = {"age_years": 2, "fever_days": 2, "rdt_result": "positive", "language": "pcm"}
    reason = {**ADULT_REASON_NO_LASSA, "triage_level": "refer_now"}
    client = client_for({"intake": intake, "reason": reason, "compose": COMPOSE_EN}, search=None)
    events = dict(
        parse_sse(
            client.post(
                "/triage/stream", json={"text": "Pikin 2 years, hot body 2 days, e dey convulse"}
            ).text
        )
    )
    assert events["rules_pre"]["floor"] == "refer_now"
    assert events["rules_pre"]["danger_signs"][0]["source"] == "rule"


def test_stream_needs_info_stops_after_rules_pre(client_for) -> None:
    client = client_for({"intake": {"language": "en"}}, search=None)
    events = parse_sse(client.post("/triage/stream", json={"text": "Woman with fever"}).text)
    assert [e for e, _ in events] == ["intake", "rules_pre", "final"]
    assert events[0][1]["questions"][0]["id"] == "fever_days"
    assert events[-1][1]["status"] == "needs_info"


def test_stream_spend_limit_is_fatal_error_without_result(client_for) -> None:
    client = client_for(REPLIES, max_spend_usd=0.0)
    events = parse_sse(client.post("/triage/stream", json={"text": ADULT_TEXT}).text)
    assert events == [("error", {"fatal": True, "message": events[0][1]["message"]})]


def test_stream_unexpected_error_fails_safe(client_for, monkeypatch) -> None:
    client = client_for(REPLIES)

    def boom(*a, **k):
        raise RuntimeError("bug")

    monkeypatch.setattr(pipeline.nodes, "retrieve", boom)
    monkeypatch.setitem(pipeline.STEPS, "retrieve", (boom, "retrieval", None))
    events = parse_sse(
        client.post(
            "/triage/stream", json={"text": "Pikin hot body, e dey convulse", "state": "Ondo"}
        ).text
    )
    names = [e for e, _ in events]
    assert names[-2:] == ["error", "final"]
    final = events[-1][1]
    assert final["status"] == "incomplete" and final["triage_level"] == "refer_now"
    assert final["danger_signs"][0]["code"] == "convulsions"


def test_meta_lists_sources_and_licences(client_for) -> None:
    client = client_for({})
    body = client.get("/meta").json()
    assert body["models"]["reason"] == "reason"
    licences = {s["doc_id"]: s["licence"] for s in body["sources"]}
    assert licences["who-malaria"] == "CC BY-NC-SA 3.0 IGO"
    assert body["dose_table_verified"] is False
    assert "api_key" not in json.dumps(body).lower()
