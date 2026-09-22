"""LLM client: caching, JSON retry, cost accounting. Token Factory is faked; no network."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from backend.app.config import Settings
from backend.app.llm.client import (
    LLMClient,
    LLMError,
    LLMParseError,
    extract_json,
    summarize,
    track_usage,
)

MODEL = "test/fast-model"


class Answer(BaseModel):
    level: str
    score: int


class FakeCompletions:
    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.replies.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=500),
        )


class FakeEmbeddings:
    def __init__(self):
        self.calls = 0

    def create(self, model, input):
        self.calls += 1
        return SimpleNamespace(
            data=[SimpleNamespace(index=i, embedding=[float(i), 1.0]) for i in range(len(input))],
            usage=SimpleNamespace(prompt_tokens=10),
        )


def make_client(tmp_path: Path, replies: list[str], *, cache: bool = True) -> LLMClient:
    settings = Settings(
        _env_file=None,
        cache_dir=tmp_path,
        cache_enabled=cache,
        model_prices={MODEL: {"input": 1.0, "output": 2.0}},
    )
    fake = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions(replies)),
        embeddings=FakeEmbeddings(),
    )
    return LLMClient(settings, client=fake)


MSGS = [{"role": "user", "content": "hi"}]


def test_chat_records_tokens_and_cost(tmp_path: Path) -> None:
    client = make_client(tmp_path, ["hello"])
    result = client.chat(MSGS, model=MODEL, step="intake")
    assert result.content == "hello"
    assert result.record.prompt_tokens == 1000
    assert result.record.completion_tokens == 500
    # 1000 * $1/M + 500 * $2/M
    assert result.record.cost_usd == pytest.approx(0.002)
    assert result.record.cached is False


def test_cost_is_none_without_price(tmp_path: Path) -> None:
    client = make_client(tmp_path, ["x"])
    assert client.chat(MSGS, model="other/model", step="s").record.cost_usd is None


def test_second_identical_call_hits_disk_cache(tmp_path: Path) -> None:
    client = make_client(tmp_path, ["hello"])
    client.chat(MSGS, model=MODEL, step="intake")
    again = client.chat(MSGS, model=MODEL, step="intake")
    assert again.content == "hello"
    assert again.record.cached is True
    assert len(client.client.chat.completions.calls) == 1


def test_cache_survives_new_client_instance(tmp_path: Path) -> None:
    make_client(tmp_path, ["hello"]).chat(MSGS, model=MODEL, step="s")
    fresh = make_client(tmp_path, [])  # would raise IndexError if it hit the API
    assert fresh.chat(MSGS, model=MODEL, step="s").content == "hello"


def test_cache_does_not_store_prompt_text(tmp_path: Path) -> None:
    client = make_client(tmp_path, ["ok"])
    client.chat([{"role": "user", "content": "PATIENT-TEXT-MARKER"}], model=MODEL, step="s")
    blobs = "".join(p.read_text() for p in tmp_path.rglob("*.json"))
    assert "ok" in blobs
    assert "PATIENT-TEXT-MARKER" not in blobs


def test_cache_disabled(tmp_path: Path) -> None:
    client = make_client(tmp_path, ["a", "b"], cache=False)
    assert client.chat(MSGS, model=MODEL, step="s").content == "a"
    assert client.chat(MSGS, model=MODEL, step="s").content == "b"
    assert not list(tmp_path.rglob("*.json"))


def test_different_model_is_a_different_cache_key(tmp_path: Path) -> None:
    client = make_client(tmp_path, ["a", "b"])
    client.chat(MSGS, model=MODEL, step="s")
    assert client.chat(MSGS, model="other/model", step="s").content == "b"


def test_empty_model_id_is_rejected(tmp_path: Path) -> None:
    client = make_client(tmp_path, ["x"])
    with pytest.raises(LLMError, match="model ID is empty"):
        client.chat(MSGS, model="", step="intake")


def test_missing_api_key_raises(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, cache_dir=tmp_path, nebius_api_key=None)
    with pytest.raises(LLMError, match="NEBIUS_API_KEY"):
        LLMClient(settings).chat(MSGS, model=MODEL, step="s")


def test_chat_json_parses_and_requests_json_mode(tmp_path: Path) -> None:
    client = make_client(tmp_path, ['{"level": "refer_now", "score": 3}'])
    answer, records = client.chat_json(MSGS, Answer, model=MODEL, step="reason")
    assert answer == Answer(level="refer_now", score=3)
    assert len(records) == 1
    call = client.client.chat.completions.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["temperature"] == 0.0


def test_chat_json_retries_once_with_error_feedback(tmp_path: Path) -> None:
    client = make_client(tmp_path, ['{"level": "refer_now"}', '{"level": "refer_now", "score": 1}'])
    answer, records = client.chat_json(MSGS, Answer, model=MODEL, step="reason")
    assert answer.score == 1
    assert [r.step for r in records] == ["reason", "reason:retry"]
    retry_msgs = client.client.chat.completions.calls[1]["messages"]
    assert retry_msgs[-2] == {"role": "assistant", "content": '{"level": "refer_now"}'}
    assert "score" in retry_msgs[-1]["content"]


def test_chat_json_raises_after_second_failure_and_evicts_cache(tmp_path: Path) -> None:
    client = make_client(tmp_path, ["not json", "still not json"])
    with pytest.raises(LLMParseError):
        client.chat_json(MSGS, Answer, model=MODEL, step="reason")
    assert not list(tmp_path.rglob("*.json"))


def test_extract_json_handles_think_blocks_and_fences() -> None:
    raw = '<think>reasoning here</think>\n```json\n{"level": "x", "score": 1}\n```'
    assert extract_json(raw) == '{"level": "x", "score": 1}'
    assert extract_json('Sure! {"a": 1} Done.') == '{"a": 1}'


def test_track_usage_collects_records_and_summarizes(tmp_path: Path) -> None:
    client = make_client(tmp_path, ["a"])
    with track_usage() as records:
        client.chat(MSGS, model=MODEL, step="intake")
        client.chat(MSGS, model=MODEL, step="intake")  # cache hit
    summary = summarize(records)
    assert summary["calls"] == 2
    assert summary["cached_calls"] == 1
    assert summary["prompt_tokens"] == 2000
    assert summary["est_cost_usd"] == pytest.approx(0.004)
    assert summary["spent_usd"] == pytest.approx(0.002)


def test_records_outside_tracking_context_are_not_collected(tmp_path: Path) -> None:
    client = make_client(tmp_path, ["a", "b"])
    client.chat(MSGS, model=MODEL, step="s")
    with track_usage() as records:
        pass
    assert records == []


def test_embed_is_cached(tmp_path: Path) -> None:
    client = make_client(tmp_path, [])
    vectors = client.embed(["a", "b"], model=MODEL)
    assert vectors == [[0.0, 1.0], [1.0, 1.0]]
    assert client.embed(["a", "b"], model=MODEL) == vectors
    assert client.client.embeddings.calls == 1
