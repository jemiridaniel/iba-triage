"""Regression tests replaying real Token Factory response shapes (synthetic sample only).

Fixture: tests/fixtures/nemotron_responses.json, captured by scripts/smoke_fast.py.
Responses are rebuilt as real openai SDK objects, so attribute access matches production.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

from backend.app.config import Settings
from backend.app.llm.client import LLMClient, LLMTruncatedError
from backend.app.schemas import DangerSignCode, PatientCase

FIXTURE = Path(__file__).parent / "fixtures" / "nemotron_responses.json"
CASES = json.loads(FIXTURE.read_text())["cases"]
MSGS = [{"role": "user", "content": "synthetic sample"}]


class ReplayCompletions:
    def __init__(self, names: list[str]):
        self.queue = [ChatCompletion.model_validate(CASES[n]["response"]) for n in names]
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.queue.pop(0)


def replay_client(tmp_path: Path, *names: str) -> LLMClient:
    models = {c["model"] for c in CASES.values()}
    settings = Settings(
        _env_file=None,
        cache_dir=tmp_path,
        cache_enabled=True,
        max_spend_usd=1.0,
        model_prices={m: {"input": 0.06, "output": 0.24} for m in models},
    )
    fake = SimpleNamespace(chat=SimpleNamespace(completions=ReplayCompletions(list(names))))
    return LLMClient(settings, client=fake)


def run(tmp_path: Path, name: str):
    client = replay_client(tmp_path, name)
    return client, client.chat_json(MSGS, PatientCase, model=CASES[name]["model"], step="intake")


# --- truncated replies are never parsed -------------------------------------


def test_lightning_reasoning_spill_is_rejected_not_parsed(tmp_path: Path) -> None:
    # Content is raw reasoning containing a draft JSON object that WOULD parse. It must not.
    name = "lightning_reasoning_default_truncated"
    content = CASES[name]["response"]["choices"][0]["message"]["content"]
    assert "thinking process" in content and '"age_years": 3' in content
    client = replay_client(tmp_path, name)
    with pytest.raises(LLMTruncatedError, match="truncated"):
        client.chat_json(MSGS, PatientCase, model=CASES[name]["model"], step="intake")
    assert len(client.client.chat.completions.calls) == 1  # no pointless retry
    assert not list((tmp_path / "llm").rglob("*.json"))  # not cached


def test_lightning_truncated_answer_is_rejected(tmp_path: Path) -> None:
    name = "lightning_reasoning_stripped_answer_truncated"
    client = replay_client(tmp_path, name)
    with pytest.raises(LLMTruncatedError) as exc:
        client.chat_json(MSGS, PatientCase, model=CASES[name]["model"], step="intake")
    assert "1478 reasoning" in str(exc.value)


def test_truncation_is_still_charged(tmp_path: Path) -> None:
    client = replay_client(tmp_path, "lightning_reasoning_default_truncated")
    with pytest.raises(LLMTruncatedError):
        client.chat_json(MSGS, PatientCase, model="nvidia/Nemotron-3_5-Lightning", step="s")
    # 746 prompt + 1500 completion tokens at $0.06/$0.24 per 1M
    assert client.ledger.total() == pytest.approx(0.00040476)


# --- clean replies parse -----------------------------------------------------


def test_lightning_thinking_off_parses(tmp_path: Path) -> None:
    _, (case, records) = run(tmp_path, "lightning_thinking_off")
    assert case.age_years == 3
    assert case.fever_days == 4
    assert case.rdt_result == "negative"
    assert case.language == "pcm"
    assert set(case.danger_signs) == {
        DangerSignCode.VOMITING_EVERYTHING,
        DangerSignCode.LETHARGY_UNCONSCIOUS,
    }
    assert records[0].reasoning_tokens == 0
    assert records[0].completion_tokens == 125


def test_nano_reasoning_field_is_ignored_and_content_parses(tmp_path: Path) -> None:
    name = "nano_reasoning_default"
    message = CASES[name]["response"]["choices"][0]["message"]
    assert message["reasoning"] and message["content"].lstrip().startswith("{")
    _, (case, records) = run(tmp_path, name)
    assert case.language == "pcm"
    assert DangerSignCode.LETHARGY_UNCONSCIOUS in case.danger_signs
    assert records[0].reasoning_tokens is None  # Nano doesn't report it
    assert records[0].completion_tokens == 892  # but reasoning is billed as completion


def test_nano_thinking_off_parses(tmp_path: Path) -> None:
    _, (case, records) = run(tmp_path, "nano_thinking_off")
    assert case.age_years == 3
    assert records[0].completion_tokens == 128


def test_cached_replay_keeps_finish_reason_and_reasoning_tokens(tmp_path: Path) -> None:
    client = replay_client(tmp_path, "lightning_thinking_off")
    model = CASES["lightning_thinking_off"]["model"]
    client.chat(MSGS, model=model, step="s")
    hit = client.chat(MSGS, model=model, step="s")
    assert hit.record.cached is True
    assert hit.finish_reason == "stop"
    assert hit.record.reasoning_tokens == 0


def test_extra_body_is_sent_and_part_of_cache_key(tmp_path: Path) -> None:
    client = replay_client(tmp_path, "lightning_thinking_off", "lightning_thinking_off")
    model = CASES["lightning_thinking_off"]["model"]
    off = {"chat_template_kwargs": {"enable_thinking": False}}
    client.chat(MSGS, model=model, step="s", extra_body=off)
    client.chat(MSGS, model=model, step="s")  # different key -> live call
    calls = client.client.chat.completions.calls
    assert calls[0]["extra_body"] == off
    assert "extra_body" not in calls[1]
