# Iba: fever-triage copilot

Decision support for primary health care workers in Nigeria (CHEWs, PHC nurses). A worker
describes a febrile patient in English or Nigerian Pidgin; Iba returns a triage level, danger
signs, a referral recommendation with guideline citations, an outbreak-aware differential, and
a referral note. **Decision support, not diagnosis: a clinician always decides.**

Built for the Nebius x NVIDIA Global AI Hackathon. Full design: [docs/SPEC.md](docs/SPEC.md).

> Status: week 1. Config, the Token Factory client (cache, cost logging, spend cap), the
> deterministic danger-sign rules, and the guideline ingest/retrieval code are in place; the
> LangGraph pipeline, Tavily outbreak tool and PWA come next.

## How we use NVIDIA Nemotron + Nebius Token Factory

All LLM calls go through Nebius Token Factory's OpenAI-compatible API using NVIDIA Nemotron
models, via one client ([backend/app/llm/client.py](backend/app/llm/client.py)) that:

- reads model IDs from env (`MODEL_FAST`, `MODEL_MID`, `MODEL_REASON`, `MODEL_EMBED`), never code;
- validates every response as structured JSON, retrying once with the error fed back;
- caches responses on disk in dev (`CACHE_DIR`) to save credits;
- logs per call: step, model, tokens, latency, estimated cost (never prompt text);
- enforces a hard spend cap (`MAX_SPEND_USD`): each live call's worst-case cost is checked
  against a ledger before it's sent.

Per-step routing (Nano for intake, Ultra for reasoning, Super for the summary) and the eval
numbers will be documented here as they land.

**Embeddings.** Token Factory currently serves no NVIDIA embedding model, so the guideline
index uses `Qwen/Qwen3-Embedding-8B` (the only embedding model in the catalogue). Every
generative step uses Nemotron.

## Safety

Danger signs are detected by deterministic rules
([backend/app/rules/danger_signs.py](backend/app/rules/danger_signs.py)) over the worker's own
words (English + Pidgin) and over the intake model's structured output. Any sign sets a
**Refer now** floor that the LLM can never lower. Drug doses never come from the LLM.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for you).

```bash
uv sync
cp .env.example .env                      # add NEBIUS_API_KEY, MODEL_* and MODEL_PRICES
uv run python -m scripts.list_models      # catalogue with prices (GET only, free)
uv run pytest                             # offline; Token Factory is mocked
uv run uvicorn backend.app.main:app --reload
```

Other scripts:

| Command | What it does | Spends credits? |
|---|---|---|
| `uv run python -m scripts.list_models --write-prices` | Writes a `MODEL_PRICES` example into `.env.example` | No |
| `uv run python -m scripts.spend [--reset]` | Shows (or resets) cumulative live spend vs. the cap | No |
| `uv run python -m scripts.smoke_nano` | One `MODEL_FAST` call parsing a synthetic case | ~$0.0002 |
| `uv run python -m backend.app.rag.ingest --dry-run` | Chunks PDFs in `data/raw/` | No |
| `uv run python -m backend.app.rag.ingest` | Chunks + embeds into `data/index/` | Yes (embeddings) |

Guideline PDFs are listed in [data/sources.yaml](data/sources.yaml) with licence notes and are
not committed; download them into `data/raw/`.

`uv run pytest -m live` runs opt-in tests against real APIs (spends credits).

## Licence

Apache-2.0. See [LICENSE](LICENSE).
