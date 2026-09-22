"""Spend guard. Token Factory is faked; no network."""

import json
from pathlib import Path

import pytest

from backend.app.config import ModelPrice
from backend.app.llm.client import LLMClient, LLMError
from backend.app.llm.spend import (
    SpendLedger,
    SpendLimitError,
    estimate_prompt_tokens,
    worst_case_cost,
)
from tests.test_llm_client import MODEL, MSGS, make_client


def with_cap(client: LLMClient, cap: float) -> LLMClient:
    client.ledger.cap_usd = cap
    return client


def test_live_calls_accumulate_actual_cost_in_ledger(tmp_path: Path) -> None:
    client = make_client(tmp_path, ["a", "b"])
    client.chat(MSGS, model=MODEL, step="s")
    client.chat([{"role": "user", "content": "different"}], model=MODEL, step="s")
    data = json.loads((tmp_path / "spend.json").read_text())
    # Each fake call: 1000 in * $1/M + 500 out * $2/M = $0.002
    assert data["total_usd"] == pytest.approx(0.004)
    assert data["calls"] == 2
    assert data["by_model"] == {MODEL: pytest.approx(0.004)}


def test_ledger_persists_across_client_instances(tmp_path: Path) -> None:
    make_client(tmp_path, ["a"]).chat(MSGS, model=MODEL, step="s")
    assert make_client(tmp_path, []).ledger.total() == pytest.approx(0.002)


def test_call_refused_when_worst_case_exceeds_cap(tmp_path: Path) -> None:
    client = with_cap(make_client(tmp_path, ["a"]), 0.001)
    # worst case >= 1000 max_tokens * $2/M = $0.002 > $0.001
    with pytest.raises(SpendLimitError, match="MAX_SPEND_USD"):
        client.chat(MSGS, model=MODEL, step="reason", max_tokens=1000)
    assert client.client.chat.completions.calls == []  # never reached the API
    assert client.ledger.total() == 0.0


def test_worst_case_uses_max_tokens_not_actual_usage(tmp_path: Path) -> None:
    # Actual cost would be $0.002, but worst case with max_tokens=5000 is ~$0.01.
    client = with_cap(make_client(tmp_path, ["a"]), 0.005)
    with pytest.raises(SpendLimitError):
        client.chat(MSGS, model=MODEL, step="s", max_tokens=5000)
    # A small max_tokens fits.
    client.chat(MSGS, model=MODEL, step="s", max_tokens=100)


def test_refused_once_running_total_nears_cap(tmp_path: Path) -> None:
    client = with_cap(make_client(tmp_path, ["a", "b"]), 0.0035)
    client.chat(MSGS, model=MODEL, step="s", max_tokens=500)  # spends $0.002
    with pytest.raises(SpendLimitError):
        client.chat([{"role": "user", "content": "x"}], model=MODEL, step="s", max_tokens=1000)


def test_cache_hits_are_free_and_never_blocked(tmp_path: Path) -> None:
    client = make_client(tmp_path, ["a"])
    client.chat(MSGS, model=MODEL, step="s", max_tokens=500)
    with_cap(client, 0.0)  # budget now exhausted
    hit = client.chat(MSGS, model=MODEL, step="s", max_tokens=500)
    assert hit.record.cached is True
    assert client.ledger.total() == pytest.approx(0.002)  # unchanged


def test_unpriced_model_is_refused(tmp_path: Path) -> None:
    client = make_client(tmp_path, ["a"])
    with pytest.raises(SpendLimitError, match="no price"):
        client.chat(MSGS, model="unpriced/model", step="s")
    assert client.client.chat.completions.calls == []


def test_spend_limit_is_not_an_llm_error() -> None:
    # The pipeline's fail-safe catches LLMError; a budget stop must not look like "refer".
    assert not issubclass(SpendLimitError, LLMError)


def test_embed_is_guarded_and_charged(tmp_path: Path) -> None:
    client = make_client(tmp_path, [])
    client.embed(["a", "b"], model=MODEL)
    assert client.ledger.total() == pytest.approx(10 / 1_000_000)
    with_cap(client, 0.0)
    with pytest.raises(SpendLimitError):
        client.embed(["new text"], model=MODEL)


def test_failed_parse_retry_is_charged_for_both_calls(tmp_path: Path) -> None:
    client = make_client(tmp_path, ['{"bad": 1}', '{"level": "x", "score": 1}'])
    from tests.test_llm_client import Answer

    client.chat_json(MSGS, Answer, model=MODEL, step="s")
    assert client.ledger.total() == pytest.approx(0.004)


# --- ledger unit tests ------------------------------------------------------


def test_reset_clears_and_returns_previous(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.json", cap_usd=1.0)
    ledger.add("m", 0.25)
    assert ledger.reset()["total_usd"] == pytest.approx(0.25)
    assert ledger.total() == 0.0


def test_corrupt_ledger_refuses_rather_than_resetting(tmp_path: Path) -> None:
    path = tmp_path / "spend.json"
    path.write_text("{not json")
    ledger = SpendLedger(path, cap_usd=1.0)
    with pytest.raises(SpendLimitError, match="unreadable"):
        ledger.check("m", "s", 0.0)
    ledger.reset()  # the escape hatch still works
    assert ledger.total() == 0.0


def test_worst_case_cost_math() -> None:
    price = ModelPrice(input=0.06, output=0.24)
    assert worst_case_cost(price, 1_000_000, 1_000_000) == pytest.approx(0.30)
    assert worst_case_cost(price, 0, 400) == pytest.approx(0.000096)


def test_prompt_estimate_is_pessimistic() -> None:
    # ~4 chars/token is typical; the estimate assumes 3, plus per-message overhead.
    text = "a" * 3000
    assert estimate_prompt_tokens([{"role": "user", "content": text}]) >= 1000
