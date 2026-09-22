"""Hard cap on cumulative live LLM spend, persisted in a small JSON ledger.

Before each live call the client asks the ledger whether the call's *worst-case* cost
(estimated prompt tokens + full max_tokens at configured prices) still fits under
MAX_SPEND_USD; after the call it records the *actual* cost from the usage block.
Cache hits never touch the ledger.

Single-process guard: a thread lock plus atomic file writes. Good enough for dev and the
eval batch; not a distributed quota.
"""

from __future__ import annotations

import json
import math
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.app.config import ModelPrice

# Rough upper-side token estimate for text we haven't tokenised (~3 chars/token is
# pessimistic for English and Pidgin; real tokenisers average closer to 4).
CHARS_PER_TOKEN = 3
MESSAGE_OVERHEAD_TOKENS = 16


class SpendLimitError(RuntimeError):
    """A live call was refused because it could push spend past MAX_SPEND_USD.

    Deliberately NOT an LLMError: the pipeline's fail-safe catches LLMError and returns
    "refer", which would silently turn a budget stop into fake triage results.
    """


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def estimate_prompt_tokens(messages: list[dict[str, str]]) -> int:
    return sum(estimate_tokens(m.get("content") or "") + MESSAGE_OVERHEAD_TOKENS for m in messages)


def worst_case_cost(price: ModelPrice, prompt_tokens: int, max_completion_tokens: int) -> float:
    return (prompt_tokens * price.input + max_completion_tokens * price.output) / 1_000_000


class SpendLedger:
    def __init__(self, path: Path, cap_usd: float):
        self.path = path
        self.cap_usd = cap_usd
        self._lock = threading.Lock()

    def read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"total_usd": 0.0, "calls": 0, "by_model": {}, "updated": None}
        except (OSError, json.JSONDecodeError) as exc:
            # A corrupt ledger must not silently reset the budget to zero.
            raise SpendLimitError(
                f"spend ledger {self.path} is unreadable; fix or reset it"
            ) from exc
        data.setdefault("by_model", {})
        data.setdefault("calls", 0)
        return data

    def total(self) -> float:
        return float(self.read()["total_usd"])

    def check(self, model: str, step: str, estimate_usd: float) -> None:
        with self._lock:
            spent = self.total()
            if spent + estimate_usd > self.cap_usd:
                raise SpendLimitError(
                    f"{step}: refused call to {model}. Worst-case cost ${estimate_usd:.4f} "
                    f"would take spend from ${spent:.4f} past the MAX_SPEND_USD cap of "
                    f"${self.cap_usd:.2f}. Check `uv run python -m scripts.spend`, then raise "
                    "MAX_SPEND_USD or reset the ledger if this spend is intended."
                )

    def add(self, model: str, cost_usd: float) -> None:
        with self._lock:
            data = self.read()
            data["total_usd"] = round(float(data["total_usd"]) + cost_usd, 8)
            data["calls"] = int(data["calls"]) + 1
            data["by_model"][model] = round(data["by_model"].get(model, 0.0) + cost_usd, 8)
            data["updated"] = datetime.now(UTC).isoformat(timespec="seconds")
            self._write(data)

    def reset(self) -> dict[str, Any]:
        with self._lock:
            try:
                previous = self.read()
            except SpendLimitError:
                previous = {"total_usd": None, "calls": None, "by_model": {}, "updated": None}
            self.path.unlink(missing_ok=True)
            return previous

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.path)
