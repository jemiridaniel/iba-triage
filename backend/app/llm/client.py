"""Nebius Token Factory client: disk cache, structured JSON, token/latency/cost accounting,
and a hard spend cap.

All LLM traffic goes through `LLMClient`. It never logs prompt or response text, only
per-call metrics (step, model, tokens, latency, cost).

Usage:
    client = LLMClient()
    with track_usage() as records:
        case, _ = client.chat_json(messages, PatientCase, model=s.model_fast, step="intake")
    summarize(records)  # -> totals for the "How Iba decided" panel
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from backend.app.config import ModelPrice, Settings, get_settings
from backend.app.llm.spend import (
    SpendLedger,
    SpendLimitError,
    estimate_prompt_tokens,
    estimate_tokens,
    worst_case_cost,
)

logger = logging.getLogger("iba.llm")


class LLMError(RuntimeError):
    """Token Factory call failed. Callers must fail safe (escalate to refer)."""


class LLMParseError(LLMError):
    """Model output failed schema validation after one corrective retry."""


class LLMTruncatedError(LLMParseError):
    """Reply hit max_tokens (finish_reason="length").

    Never parsed: when a Nemotron reply is cut off mid-reasoning, Token Factory returns the
    raw reasoning in `content`, often including draft JSON that would parse but is not the
    answer. Not retried: the same budget truncates again. Fix the step's config instead.
    """


@dataclass
class CallRecord:
    step: str
    model: str
    kind: str  # "chat" | "embed"
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float | None  # None when no price is configured for the model
    cached: bool  # True: served from disk; tokens/latency/cost are from the original call
    reasoning_tokens: int | None = None  # part of completion_tokens, when reported


@dataclass
class ChatResult:
    content: str
    record: CallRecord
    cache_key: str
    finish_reason: str | None = None


# --- usage tracking ---------------------------------------------------------

_usage: ContextVar[list[CallRecord] | None] = ContextVar("iba_llm_usage", default=None)


@contextmanager
def track_usage() -> Iterator[list[CallRecord]]:
    """Collect every CallRecord made in this context (e.g. one triage request)."""
    records: list[CallRecord] = []
    token = _usage.set(records)
    try:
        yield records
    finally:
        _usage.reset(token)


def summarize(records: list[CallRecord]) -> dict[str, Any]:
    costs = [r.cost_usd for r in records if r.cost_usd is not None]
    return {
        "calls": len(records),
        "cached_calls": sum(r.cached for r in records),
        "prompt_tokens": sum(r.prompt_tokens for r in records),
        "completion_tokens": sum(r.completion_tokens for r in records),
        "latency_ms": round(sum(r.latency_ms for r in records), 1),
        "est_cost_usd": round(sum(costs), 6) if costs else None,
        # What this run actually spent (cache hits are free).
        "spent_usd": round(sum(r.cost_usd or 0.0 for r in records if not r.cached), 6),
        "steps": [asdict(r) for r in records],
    }


# --- disk cache -------------------------------------------------------------


class DiskCache:
    """One JSON file per key. Stores responses and metrics only, not requests."""

    def __init__(self, root: Path):
        self.root = root

    @staticmethod
    def key(payload: dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            logger.warning("cache entry unreadable, ignoring: %s", path.name)
            return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


# --- JSON extraction --------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> str:
    """Strip reasoning blocks and code fences; return the outermost JSON object."""
    cleaned = _THINK_RE.sub("", text).strip()
    cleaned = _FENCE_RE.sub("", cleaned).strip()
    if not cleaned.startswith("{"):
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return cleaned


# --- client -----------------------------------------------------------------


class LLMClient:
    def __init__(
        self,
        settings: Settings | None = None,
        client: OpenAI | None = None,
        *,
        on_response: Callable[[Any], None] | None = None,
    ):
        """`on_response` receives each raw live SDK response (debugging/smoke tests only)."""
        self.settings = settings or get_settings()
        self._client = client
        self.cache = DiskCache(self.settings.cache_dir / "llm") if self.settings.use_cache else None
        self.ledger = SpendLedger(self.settings.spend_file, self.settings.max_spend_usd)
        self.on_response = on_response

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            key = self.settings.nebius_api_key
            if key is None or not key.get_secret_value():
                raise LLMError("NEBIUS_API_KEY is not set")
            self._client = OpenAI(
                api_key=key.get_secret_value(),
                base_url=self.settings.nebius_base_url,
                timeout=self.settings.llm_timeout_s,
                max_retries=self.settings.llm_max_retries,
            )
        return self._client

    # -- public API --

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        step: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResult:
        """`json_mode` sends response_format=json_object. Off by default: the Nemotron models
        on Token Factory don't advertise json_mode, so we rely on prompt + validation.
        `extra_body` carries provider-specific fields (e.g. chat_template_kwargs)."""
        _require_model(model, step)
        max_tokens = max_tokens or self.settings.llm_default_max_tokens
        params: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        if extra_body:
            params["extra_body"] = extra_body
        key = DiskCache.key({"kind": "chat", "model": model, "messages": messages, **params})

        if (hit := self._cache_get(key)) is not None:
            record = self._record(step, model, "chat", hit, cached=True)
            return ChatResult(hit["content"], record, key, hit.get("finish_reason"))

        self._guard(model, step, estimate_prompt_tokens(messages), max_tokens)
        start = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(model=model, messages=messages, **params)
        except openai.OpenAIError as exc:
            raise LLMError(f"{step}: Token Factory call failed ({_describe(exc)})") from exc
        latency_ms = (time.perf_counter() - start) * 1000
        if self.on_response is not None:
            self.on_response(resp)

        usage = resp.usage
        choice = resp.choices[0]
        entry = {
            "content": choice.message.content or "",
            "finish_reason": choice.finish_reason,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "reasoning_tokens": _reasoning_tokens(usage),
            "latency_ms": latency_ms,
        }
        self._cache_set(key, entry)
        record = self._record(step, model, "chat", entry, cached=False)
        self._charge(record)
        return ChatResult(entry["content"], record, key, entry["finish_reason"])

    def chat_json[T: BaseModel](
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        model: str,
        step: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[T, list[CallRecord]]:
        """Call the model and validate its JSON against `schema`.

        On a parse failure, retry once with the validation error fed back. If that also
        fails, raise LLMParseError; the pipeline must then fail safe to "refer".
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "step": step,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "json_mode": json_mode,
            "extra_body": extra_body,
        }
        first = self.chat(messages, **kwargs)
        self._reject_truncated(first, step)
        try:
            return schema.model_validate_json(extract_json(first.content)), [first.record]
        except ValidationError as exc:
            error = _short_error(exc)

        logger.info(
            "llm step=%s schema=%s parse failed (finish_reason=%s), retrying once",
            step,
            schema.__name__,
            first.finish_reason,
        )
        retry_messages = [
            *messages,
            {"role": "assistant", "content": first.content},
            {
                "role": "user",
                "content": (
                    "Your reply did not match the required JSON schema.\n"
                    f"Validation error:\n{error}\n"
                    "Reply again with only the corrected JSON object."
                ),
            },
        ]
        second = self.chat(retry_messages, **{**kwargs, "step": f"{step}:retry"})
        self._reject_truncated(second, step)
        try:
            return schema.model_validate_json(extract_json(second.content)), [
                first.record,
                second.record,
            ]
        except ValidationError as exc:
            # Don't let a bad answer stick in the dev cache.
            if self.cache is not None:
                self.cache.delete(first.cache_key)
                self.cache.delete(second.cache_key)
            raise LLMParseError(
                f"{step}: output failed {schema.__name__} validation after retry"
            ) from exc

    def embed(self, texts: list[str], *, model: str, step: str = "embed") -> list[list[float]]:
        _require_model(model, step)
        key = DiskCache.key({"kind": "embed", "model": model, "input": texts})
        if (hit := self._cache_get(key)) is not None:
            self._record(step, model, "embed", hit, cached=True)
            return hit["vectors"]

        self._guard(model, step, sum(estimate_tokens(t) for t in texts), 0)
        start = time.perf_counter()
        try:
            resp = self.client.embeddings.create(model=model, input=texts)
        except openai.OpenAIError as exc:
            raise LLMError(f"{step}: Token Factory call failed ({_describe(exc)})") from exc
        latency_ms = (time.perf_counter() - start) * 1000

        entry = {
            "vectors": [d.embedding for d in sorted(resp.data, key=lambda d: d.index)],
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": 0,
            "latency_ms": latency_ms,
        }
        self._cache_set(key, entry)
        record = self._record(step, model, "embed", entry, cached=False)
        self._charge(record)
        return entry["vectors"]

    # -- internals --

    def cost_usd(self, model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
        price: ModelPrice | None = self.settings.model_prices.get(model)
        if price is None:
            return None
        return (prompt_tokens * price.input + completion_tokens * price.output) / 1_000_000

    def _reject_truncated(self, result: ChatResult, step: str) -> None:
        if result.finish_reason != "length":
            return
        if self.cache is not None:
            self.cache.delete(result.cache_key)
        raise LLMTruncatedError(
            f"{step}: reply truncated at max_tokens ({result.record.completion_tokens} "
            f"completion tokens, {result.record.reasoning_tokens} reasoning); raise max_tokens "
            "or turn reasoning off for this step"
        )

    def _guard(self, model: str, step: str, prompt_tokens: int, max_completion: int) -> None:
        price = self.settings.model_prices.get(model)
        if price is None:
            raise SpendLimitError(
                f"{step}: no price for {model} in MODEL_PRICES, so its cost can't be capped. "
                "Run `uv run python -m scripts.list_models --write-prices` and copy the "
                "MODEL_PRICES line into .env."
            )
        self.ledger.check(model, step, worst_case_cost(price, prompt_tokens, max_completion))

    def _charge(self, record: CallRecord) -> None:
        # _guard guarantees a price exists for every live call.
        self.ledger.add(record.model, record.cost_usd or 0.0)

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        return self.cache.get(key) if self.cache is not None else None

    def _cache_set(self, key: str, entry: dict[str, Any]) -> None:
        if self.cache is not None:
            self.cache.set(key, entry)

    def _record(
        self, step: str, model: str, kind: str, entry: dict[str, Any], *, cached: bool
    ) -> CallRecord:
        pt, ct = int(entry["prompt_tokens"]), int(entry["completion_tokens"])
        record = CallRecord(
            step=step,
            model=model,
            kind=kind,
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_ms=round(float(entry["latency_ms"]), 1),
            cost_usd=self.cost_usd(model, pt, ct),
            cached=cached,
            reasoning_tokens=entry.get("reasoning_tokens"),
        )
        logger.info(
            "llm step=%s model=%s kind=%s in=%d out=%d latency_ms=%.0f cost_usd=%s cached=%s",
            record.step,
            record.model,
            record.kind,
            record.prompt_tokens,
            record.completion_tokens,
            record.latency_ms,
            "n/a" if record.cost_usd is None else f"{record.cost_usd:.6f}",
            record.cached,
        )
        if (records := _usage.get()) is not None:
            records.append(record)
        return record


def _reasoning_tokens(usage: Any) -> int | None:
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    value = getattr(details, "reasoning_tokens", None) if details else None
    return int(value) if value is not None else None


def _describe(exc: openai.OpenAIError) -> str:
    status = getattr(exc, "status_code", None)
    return f"{type(exc).__name__} {status}" if status else type(exc).__name__


def _require_model(model: str, step: str) -> None:
    if not model:
        raise LLMError(f"{step}: model ID is empty; set MODEL_* in .env")


def _short_error(exc: ValidationError, limit: int = 1500) -> str:
    # Pydantic's default message embeds input values; keep only locations and reasons.
    lines = [f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}" for e in exc.errors()]
    return "\n".join(lines)[:limit]
