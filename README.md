# Iba: fever-triage copilot

Decision support for primary health care workers in Nigeria (CHEWs, PHC nurses). A worker
describes a febrile patient in English or Nigerian Pidgin; Iba returns a triage level, danger
signs, a referral recommendation with guideline citations, an outbreak-aware differential, and
a referral note. **Decision support, not diagnosis: a clinician always decides.**

Built for the Nebius x NVIDIA Global AI Hackathon. Full design: [docs/SPEC.md](docs/SPEC.md).

> Status: week-1 scaffold. Config, the Token Factory client, and the deterministic danger-sign
> rules are in place; the LangGraph pipeline, retrieval, Tavily outbreak tool and PWA come next.

## How we use NVIDIA Nemotron + Nebius Token Factory

All LLM calls go through Nebius Token Factory's OpenAI-compatible API using NVIDIA Nemotron
models, via one client ([backend/app/llm/client.py](backend/app/llm/client.py)) that:

- reads model IDs from env (`MODEL_FAST`, `MODEL_MID`, `MODEL_REASON`, `MODEL_EMBED`), never code;
- validates every response as structured JSON, retrying once with the error fed back;
- caches responses on disk in dev (`CACHE_DIR`) to save credits;
- logs per call: step, model, tokens, latency, estimated cost (never prompt text).

Per-step routing (Nano for intake, Ultra for reasoning, Super for the summary) and the eval
numbers will be documented here as they land.

## Safety

Danger signs are detected by deterministic rules
([backend/app/rules/danger_signs.py](backend/app/rules/danger_signs.py)) over the worker's own
words (English + Pidgin) and over the intake model's structured output. Any sign sets a
**Refer now** floor that the LLM can never lower. Drug doses never come from the LLM.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you).

```bash
uv sync
cp .env.example .env                      # add NEBIUS_API_KEY
uv run python -m scripts.list_models      # prints NVIDIA model IDs; copy them into .env
uv run pytest                             # offline; Token Factory is mocked
uv run uvicorn backend.app.main:app --reload
```

`uv run pytest -m live` runs opt-in tests against real APIs (spends credits).

## Licence

Apache-2.0. See [LICENSE](LICENSE).
