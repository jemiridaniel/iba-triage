"""Shared fakes for pipeline tests: scripted Token Factory, fake index, mock outbreak search."""

import json
import re
from collections.abc import Callable
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from backend.app.config import Settings
from backend.app.graph.state import Deps
from backend.app.llm.client import LLMClient
from backend.app.rag.fake_embed import HASH_EMBED_MODEL, HashEmbedder
from backend.app.rag.store import VectorStore
from backend.app.schemas import GuidelineChunk
from backend.app.tools.outbreak import FixtureSearch, OutbreakTool
from scripts.build_fake_index import LABEL, PASSAGES

REPO = Path(__file__).resolve().parents[1]
MOCK_OUTBREAK = REPO / "data" / "mock" / "outbreak_ondo_lassa.json"
TODAY = date(2026, 9, 22)
LASSA_URL = "https://ncdc.gov.ng/diseases/sitreps/MOCK-lassa-2026-w37"

Reply = str | dict | Callable[[dict], str] | list
TRUNCATED = object()  # sentinel: reply cut off mid-reasoning (finish_reason="length")


class ScriptedCompletions:
    """Answers by the "Task: <name>." line that starts every system prompt.

    A reply may be a JSON string, a dict (dumped), a callable(kwargs) -> str, or a list
    consumed in order (one item per call). TRUNCATED stands for a reply cut off at max_tokens.
    """

    def __init__(self, replies: dict[str, Reply]):
        self.replies = dict(replies)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def create(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        task = re.match(r"Task: ([a-z ]+)\.", system).group(1)
        self.calls.append((task, kwargs))
        reply = self.replies[task]
        if isinstance(reply, list):
            reply = reply.pop(0)
        if callable(reply):
            reply = reply(kwargs)
        finish = "stop"
        if reply is TRUNCATED:
            reply, finish = 'Here\'s a thinking process: ... {"draft": true}', "length"
        content = json.dumps(reply) if isinstance(reply, dict) else reply
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)
            ],
            usage=SimpleNamespace(
                prompt_tokens=100, completion_tokens=50, completion_tokens_details=None
            ),
        )

    def tasks(self) -> list[str]:
        return [t for t, _ in self.calls]

    def messages_for(self, task: str) -> list[dict[str, str]]:
        return next(kw["messages"] for t, kw in self.calls if t == task)

    def messages_for_last(self, task: str) -> str:
        """User message of the most recent call for `task`."""
        return [kw["messages"] for t, kw in self.calls if t == task][-1][-1]["content"]


def make_settings(tmp_path: Path, **overrides) -> Settings:
    base = {
        "_env_file": None,
        "cache_dir": tmp_path,
        "cache_enabled": False,
        "max_spend_usd": 1.0,
        "model_fast": "fast",
        "model_mid": "mid",
        "model_reason": "reason",
        "model_prices": {m: {"input": 1.0, "output": 2.0} for m in ("fast", "mid", "reason")},
    }
    return Settings(**{**base, **overrides})


def fake_store() -> VectorStore:
    chunks = [
        GuidelineChunk(
            id=f"{doc_id}:{i:04d}",
            doc_id=doc_id,
            title=title,
            section=section,
            page=1,
            page_end=1,
            text=LABEL + text,
            tokens=len(text.split()),
        )
        for i, (doc_id, title, section, text) in enumerate(PASSAGES)
    ]
    vectors = np.array(HashEmbedder()([c.text for c in chunks]))
    return VectorStore(vectors, chunks, HASH_EMBED_MODEL)


def make_deps(
    tmp_path: Path,
    replies: dict[str, Reply],
    *,
    search: Any = "mock",
    store: bool = True,
    **settings_overrides,
) -> tuple[Deps, ScriptedCompletions]:
    settings = make_settings(tmp_path, **settings_overrides)
    completions = ScriptedCompletions(replies)
    client = LLMClient(
        settings, client=SimpleNamespace(chat=SimpleNamespace(completions=completions))
    )
    if search == "mock":
        search = FixtureSearch(MOCK_OUTBREAK)
    tool = OutbreakTool(client, search, settings.cache_dir, today=lambda: TODAY)
    deps = Deps(
        settings=settings,
        client=client,
        outbreak=tool,
        store=fake_store() if store else None,
        embed=HashEmbedder() if store else None,
    )
    return deps, completions


# --- canned model replies ---------------------------------------------------

COMPOSE_EN = {"summary": "Summary text.", "referral_note": "Referral note text."}
COMPOSE_PCM = {"summary": "Summary for Pidgin.", "referral_note": "Referral note for Pidgin."}

OUTBREAK_EXTRACTION = {
    "signals": [
        {
            "disease": "Lassa fever",
            "state": "Ondo",
            "status": "active",
            "report_date": "2026-09-18",
            "url": LASSA_URL,
        },
        {
            "disease": "Cholera",
            "state": "Bauchi",
            "status": "ongoing",
            "report_date": "2026-09-10",
            "url": "https://reliefweb.int/report/nigeria/MOCK-cholera-bauchi-2026",
        },
        # dropped: no date
        {
            "disease": "Meningitis",
            "state": "Kebbi",
            "status": "active",
            "report_date": None,
            "url": "https://www.who.int/MOCK-meningitis-note",
        },
        # dropped: URL not in the search results (invented)
        {
            "disease": "Lassa fever",
            "state": "Edo",
            "status": "active",
            "report_date": "2026-09-15",
            "url": "https://ncdc.gov.ng/invented-page",
        },
    ]
}
