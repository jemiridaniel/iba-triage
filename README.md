# Iba: fever-triage copilot

Decision support for primary health care workers in Nigeria (CHEWs, PHC nurses). A worker
describes a febrile patient in English or Nigerian Pidgin; Iba returns a triage level, danger
signs, a referral recommendation with guideline citations, an outbreak-aware differential, and
a referral note. **Decision support, not diagnosis: a clinician always decides.**

Built for the Nebius x NVIDIA Global AI Hackathon. Full design: [docs/SPEC.md](docs/SPEC.md).

> Status: week 3. Installable PWA with streaming results, served by FastAPI from one Docker
> image; real guideline index (NCDC + WHO). Pending: live Tavily credits, the NMEP malaria
> guideline PDF, a verified dose table and a verified endemic-state list.

## Pipeline

```
intake -> rules_pre -> retrieve -> outbreak -> reason -> rules_post -> compose
Nemotron  Python     guideline   Tavily +    Nemotron  Python       Nemotron
fast      rules      index       fast model  reason    rules        mid
(think off)                      (think off) (think on)             (think off)
```

- **intake** normalises English/Pidgin text into a `PatientCase`; if age, fever duration or
  RDT result is missing (and there are no danger signs) it returns up to 2 follow-up
  questions. The client resubmits the same text with `answers` to resume; nothing is stored.
- **rules_pre / rules_post** detect danger signs deterministically and set a triage floor the
  model can never lower; rules_post also strips any model-written doses, attaches doses from
  the weight-band table (currently an UNVERIFIED stub), drops citations that don't resolve,
  and fails safe to "Refer now" if a model step failed.
- **outbreak** searches trusted domains (NCDC, WHO, ReliefWeb) for the patient's state,
  extracts dated, sourced signals, and caches per state per day. A static endemicity
  baseline ([data/endemicity.yaml](data/endemicity.yaml), e.g. Lassa hotspot states) is always
  included, so a failed search never leaves triage blind; the result says "using baseline
  endemicity only". Live and baseline are labelled separately everywhere.
- **Lassa suspicion** follows the NCDC suspected-case definition (National Guideline for Lassa
  Fever Case Management 2018, §1.1.2): an active live signal in the state means Refer now; an
  endemic state alone means Refer within 24h. Intake asks about prior antimalarial/antibiotic
  treatment when fever has lasted 3+ days.
- **Treat & monitor** always carries a review interval, "no antimalarials with a negative RDT",
  and what to test next, each cited to WHO guidance. Rule and model advice that say the same
  thing are merged (rule wording wins).
- **Quote-verified grounding**: every guideline claim from the reasoning model carries a chunk
  ID and a verbatim 8–40-word quote from that chunk; `rules_post` checks deterministically that
  the quote is really there. Claims that fail are kept but shown in grey as "AI suggestion —
  no guideline source". In the 10-case dev eval, 71% of claims were quote-verified and the rest
  honestly marked; an LLM judge rated 94% of verified claims as supported.
- **Retrieval** plans one query per suspected condition and purpose (classification,
  treatment, referral, IPC) with a small prior toward the owning guideline, embedded in one call.
- Every result carries a **decision trace**: model, reasoning on/off, tokens, latency and
  cost per step.

## How we use NVIDIA Nemotron + Nebius Token Factory

All LLM calls go through Nebius Token Factory's OpenAI-compatible API using NVIDIA Nemotron
models, via one client ([backend/app/llm/client.py](backend/app/llm/client.py)) that:

- reads model IDs from env (`MODEL_FAST`, `MODEL_MID`, `MODEL_REASON`, `MODEL_EMBED`), never code;
- validates every response as structured JSON, retrying once with the error fed back;
- caches responses on disk in dev (`CACHE_DIR`) to save credits;
- logs per call: step, model, tokens, latency, estimated cost (never prompt text);
- enforces a hard spend cap (`MAX_SPEND_USD`): each live call's worst-case cost is checked
  against a ledger before it's sent.

Per-step routing ([backend/app/llm/router.py](backend/app/llm/router.py)): the fast model
with reasoning **off** for intake and outbreak extraction, Nemotron 3 Ultra with reasoning
**on** for the single reasoning step, Nemotron 3 Super with reasoning off for the summary and
referral note. Each step's reasoning is configurable (`REASONING_<STEP>`).

Measured on Token Factory: reasoning is switched off with
`chat_template_kwargs={"enable_thinking": false}` (a `/no_think` system prompt is ignored).
Turning it off for intake cut completion tokens ~7x (892 → 128 on Nano) and latency ~3–5x.
A reply cut off mid-reasoning returns the raw reasoning in `content`, so the client rejects
any `finish_reason="length"` reply rather than parsing it. Eval numbers will follow.

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

### Frontend and Docker

```bash
cd frontend && corepack pnpm install && corepack pnpm dev   # dev UI on :5173, proxies to :8000
cd frontend && corepack pnpm build                          # then FastAPI serves frontend/dist
docker compose up --build                                   # production image on :8000
```

The image (91 MB compressed) runs as a non-root user with a read-only filesystem and a
`/health` check. Config comes from `.env` at runtime; the guideline index is mounted from
`data/index/` rather than baked in (NCDC documents have no stated licence).

### Guideline index and licensing

The index is **not** in the repo or the public image. NCDC documents carry no stated licence,
and WHO documents are CC BY-NC-SA 3.0 IGO (non-commercial, attribution). The repo ships
everything needed to rebuild it: [data/sources.yaml](data/sources.yaml) (sources, editions,
licence notes), `scripts/fetch_sources.py` (download) and `backend/app/rag/ingest.py`
(chunk + embed, about $0.007). For deployment the index is baked into a **private** image in
Nebius's container registry (or pulled from private object storage at startup), never
published. Scanned PDFs are OCR'd with `ocrmypdf` before ingest.

`POST /triage/stream` streams server-sent events after each pipeline step, so danger-sign
referrals appear in about 1.5 s; `POST /triage` returns the whole result at once.

Other scripts:

| Command | What it does | Spends credits? |
|---|---|---|
| `uv run python -m scripts.list_models --write-prices` | Writes a `MODEL_PRICES` example into `.env.example` | No |
| `uv run python -m scripts.spend [--reset]` | Shows (or resets) cumulative live spend vs. the cap | No |
| `uv run python -m backend.app.cli "case text" --state Ondo` | Runs the full pipeline and prints the result and decision trace | ~$0.003 per case |
| `uv run python -m scripts.fetch_sources` | Downloads confirmed guideline PDFs into `data/raw/` | No |
| `uv run python -m scripts.build_fake_index` | Builds a tiny labelled FAKE index in `data/index_fake/` for dev | No |
| `uv run python -m scripts.smoke_fast [--model ID] [--reasoning default\|kwargs-off\|no-think]` | One live call parsing a synthetic case; prints the raw response shape | ~$0.0001–0.0004 |
| `uv run python -m backend.app.rag.ingest --dry-run` | Chunks PDFs in `data/raw/` | No |
| `uv run python -m backend.app.rag.ingest` | Chunks + embeds into `data/index/` | Yes (embeddings) |

Guideline PDFs are listed in [data/sources.yaml](data/sources.yaml) with licence notes and are
not committed. `scripts.fetch_sources` downloads them; the NMEP malaria guideline must be added
by hand. WHO documents are CC BY-NC-SA 3.0 IGO (attribution, non-commercial).

Demo without Tavily credits or real guideline PDFs (synthetic data, clearly flagged in output):

```bash
uv run python -m scripts.build_fake_index
uv run python -m backend.app.cli "Adult man 35 years, fever 5 days, RDT negative, took coartem \
  for 3 days but no improvement" --state Ondo \
  --index data/index_fake --mock-outbreak data/mock/outbreak_ondo_lassa.json
```

### Evaluation

60 synthetic vignettes ([eval/vignettes.jsonl](eval/vignettes.jsonl)) following SPEC §6, each
with gold labels and a guideline rationale, **pending clinician review**
([docs/CLINICAL_REVIEW.md](docs/CLINICAL_REVIEW.md)). `eval/run_eval.py` runs the routed,
reason-only and fast-only configs (resumable, cached, confirms before spending over $0.50);
`--outbreak off|mock` measures outbreak lift without Tavily. `eval/report.py` writes
[eval/results/report.md](eval/results/report.md) and charts. Current results are a 10-case
development check only.

```bash
uv run python -m eval.run_eval --config routed --limit 10 --stratified --judge
uv run python -m eval.run_eval --config routed --category suspected_lassa --outbreak mock
uv run python -m eval.report
```

`uv run pytest -m live` runs opt-in tests against real APIs (spends credits).

## Licence

Apache-2.0. See [LICENSE](LICENSE).
