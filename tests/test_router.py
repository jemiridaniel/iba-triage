"""Per-step routing and reasoning control. No network."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.app.config import Settings
from backend.app.llm.client import LLMClient, LLMError
from backend.app.llm.router import THINKING_OFF_KWARGS, call_json, route
from tests.test_llm_client import Answer, FakeCompletions


def settings(**overrides) -> Settings:
    base = {
        "_env_file": None,
        "model_fast": "fast",
        "model_mid": "mid",
        "model_reason": "reason",
    }
    return Settings(**{**base, **overrides})


def test_default_routing_table() -> None:
    s = settings()
    got = {step: route(step, s) for step in ("intake", "outbreak", "reason", "compose")}
    assert {k: (v.model, v.reasoning) for k, v in got.items()} == {
        "intake": ("fast", False),
        "outbreak": ("fast", False),
        "reason": ("reason", True),
        "compose": ("mid", False),
    }


def test_reasoning_off_sends_thinking_kwargs_on_sends_nothing() -> None:
    s = settings()
    assert route("intake", s).extra_body == THINKING_OFF_KWARGS
    assert route("reason", s).extra_body is None


def test_budgets_follow_reasoning() -> None:
    s = settings(max_tokens_reasoning_off=900, max_tokens_reasoning_on=7000)
    assert route("intake", s).max_tokens == 900
    assert route("reason", s).max_tokens == 7000


def test_reasoning_is_configurable_per_step() -> None:
    s = settings(reasoning_intake=True, reasoning_reason=False)
    assert route("intake", s).reasoning is True
    assert route("reason", s).reasoning is False
    assert route("reason", s).extra_body == THINKING_OFF_KWARGS


def test_reasoning_flags_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REASONING_COMPOSE", "true")
    assert route("compose", settings()).reasoning is True


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ("routed", ["fast", "fast", "reason", "mid"]),
        ("reason-only", ["reason"] * 4),
        ("fast-only", ["fast"] * 4),
    ],
)
def test_eval_configs_swap_models_but_keep_reasoning(config: str, expected: list[str]) -> None:
    s = settings()
    steps = ["intake", "outbreak", "reason", "compose"]
    cfgs = [route(step, s, config) for step in steps]
    assert [c.model for c in cfgs] == expected
    assert [c.reasoning for c in cfgs] == [False, False, True, False]


def test_missing_model_raises() -> None:
    with pytest.raises(LLMError, match="MODEL_MID"):
        route("compose", settings(model_mid=""))


def test_unknown_step_raises() -> None:
    with pytest.raises(ValueError):
        route("diagnose", settings())  # type: ignore[arg-type]


def test_call_json_uses_route(tmp_path: Path) -> None:
    s = settings(cache_dir=tmp_path, model_prices={"fast": {"input": 1.0, "output": 1.0}})
    completions = FakeCompletions(['{"level": "x", "score": 1}'])
    client = LLMClient(s, client=SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    answer, records = call_json(client, "intake", [{"role": "user", "content": "hi"}], Answer)
    assert answer.score == 1
    (call,) = completions.calls
    assert call["model"] == "fast"
    assert call["extra_body"] == THINKING_OFF_KWARGS
    assert call["max_tokens"] == s.max_tokens_reasoning_off
    assert records[0].step == "intake"
