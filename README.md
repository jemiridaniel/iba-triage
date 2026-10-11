# Ibà: fever-triage copilot

Decision support for primary health care workers in Nigeria (CHEWs, PHC nurses). A worker
describes a febrile patient in English or Nigerian Pidgin; Ibà returns a triage level, danger
signs, a referral recommendation with guideline citations, an outbreak-aware differential, and
a referral note. **Decision support, not diagnosis: a clinician always decides.**

Built for the Nebius x NVIDIA Global AI Hackathon. Full design: [docs/SPEC.md](docs/SPEC.md).

> Status: week 5. Installable PWA with streaming results, served by FastAPI from one Docker
> image; real guideline index (NCDC + WHO); **live Tavily outbreak search**; **deployed at
> https://iba-triage.onrender.com**. Pending: the NMEP malaria guideline PDF, a verified dose
> table and a verified endemic-state list.

## Pipeline

```
intake -> rules_pre -> retrieve -> outbreak -> reason -> rules_post -> compose
Nemotron  Python     guideline   Tavily +    Nemotron  Python       Nemotron
fast      rules      index       fast model  reason    rules        mid
(think off)                      (think off) (think on)             (think off)
```

![Ibà request pipeline: intake (Nemotron 3.5 Lightning) -> rules_pre (deterministic) -> retrieve (Qwen3 embedding + NumPy index) and outbreak (Tavily search + Lightning extraction) running concurrently -> reason (Nemotron 3 Super, reasoning on) -> rules_post (deterministic) -> compose (Nemotron 3 Super). A "Refer now" banner streams after rules_pre at ~1s if a danger sign is found.](docs/gallery/07-architecture.png)

Model, reasoning state and typical latency per step (deterministic rule stages in slate, the
one tool-plus-model step in amber, everything else in teal) — see
[docs/gallery/07-architecture.html](docs/gallery/07-architecture.html) for the editable source.

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
(chunk + embed, about $0.007). Scanned PDFs are OCR'd with `ocrmypdf` before ingest.

At deploy time the index is kept out of the image and fetched from **private** storage instead:
a bind mount from a private volume on Nebius Compute (`docker-compose.yml`'s pattern), or —
where that isn't available — a private dataset repo, downloaded once at container boot by
`backend/app/rag/fetch_remote_index.py` (see **Deploy** below). Either way the guideline text
is never published or baked into a distributed image.

`POST /triage/stream` streams server-sent events after each pipeline step, so danger-sign
referrals appear in about 1.5 s; `POST /triage` returns the whole result at once.

Other scripts:

| Command | What it does | Spends credits? |
|---|---|---|
| `uv run python -m scripts.list_models --write-prices` | Writes a `MODEL_PRICES` example into `.env.example` | No |
| `uv run python -m scripts.spend [--reset]` | Shows (or resets) cumulative live spend vs. the cap | No |
| `uv run python -m scripts.demo_check --repeat 3` | Runs the 3 demo cases against a running server; fails on any incomplete or wrong-level result | ~$0.007 per case |
| `uv run python -m scripts.probe_reasoning_budget` | Tests whether Token Factory can cap Nemotron reasoning (it can't) | ~$0.02 |
| `uv run python -m backend.app.cli "case text" --state Ondo` | Runs the full pipeline and prints the result and decision trace | ~$0.003 per case |
| `uv run python -m scripts.fetch_sources` | Downloads confirmed guideline PDFs into `data/raw/` | No |
| `uv run python -m scripts.build_fake_index` | Builds a tiny labelled FAKE index in `data/index_fake/` for dev | No |
| `uv run python -m scripts.smoke_fast [--model ID] [--reasoning default\|kwargs-off\|no-think]` | One live call parsing a synthetic case; prints the raw response shape | ~$0.0001–0.0004 |
| `uv run python -m backend.app.rag.ingest --dry-run` | Chunks PDFs in `data/raw/` | No |
| `uv run python -m backend.app.rag.ingest` | Chunks + embeds into `data/index/` | Yes (embeddings) |

Guideline PDFs are listed in [data/sources.yaml](data/sources.yaml) with licence notes and are
not committed. `scripts.fetch_sources` downloads them; the NMEP malaria guideline must be added
by hand. WHO documents are CC BY-NC-SA 3.0 IGO (attribution, non-commercial).

With `TAVILY_API_KEY` set, the outbreak step searches `ncdc.gov.ng`, `who.int`, `reliefweb.int`
and `afro.who.int`, cached per state per day. To see exactly what a live check does — the
queries sent, every result, and why each extracted signal was kept or dropped:

```bash
uv run python -m scripts.outbreak_trace --state Ondo --fresh
```

Demo without Tavily credits or real guideline PDFs (synthetic data, clearly flagged in output):

```bash
uv run python -m scripts.build_fake_index
uv run python -m backend.app.cli "Adult man 35 years, fever 5 days, RDT negative, took coartem \
  for 3 days but no improvement" --state Ondo \
  --index data/index_fake --mock-outbreak data/mock/outbreak_ondo_lassa.json
```

### Deploy

**Live demo: https://iba-triage.onrender.com** (Render free tier, Frankfurt). **First load can
take ~40 s** if the free instance has gone to sleep (15 minutes with no traffic) — hit `/health`
a minute before a demo or a Devpost tester's first visit to warm it up.

The intended target is **Nebius AI Cloud Serverless Endpoints** (`docker-compose.yml`'s local
setup mirrors that: the index bind-mounted read-only, secrets from `.env`, never baked in).
That wasn't reachable during the hackathon: our tenant only had **AI Studio** (Token Factory)
provisioned, not the general Nebius AI Cloud product (Compute, Container Registry, IAM) needed
for Serverless Endpoints — see `docs/FEEDBACK.md` for the details. The hackathon rules only
require using Nebius Token Factory or Nebius AI Cloud **for inference**, which this app does
either way, so we deployed the container itself to Render instead, at $0/month.

**Reproducing this deploy:**

```bash
# 1. Guideline index -> a private dataset repo (read-only token at boot; see fetch_remote_index.py)
uvx --from huggingface_hub hf repos create <you>/iba-guideline-index --repo-type dataset --private
uvx --from huggingface_hub hf upload <you>/iba-guideline-index data/index . --repo-type dataset

# 2. Build for linux/amd64 (Render's arch) and push to GitHub Container Registry
gh auth refresh -h github.com -s write:packages,read:packages   # one-time, adds the scope
gh auth token | docker login ghcr.io -u <you> --password-stdin
docker buildx build --platform linux/amd64 -t ghcr.io/<you>/iba-triage:latest --push .

# 3. Render: a registry credential (GHCR is private) + the service itself, via API
#    (dashboard works too: New -> Web Service -> Existing Image -> paste the GHCR path)
curl -X POST https://api.render.com/v1/registrycredentials \
  -H "Authorization: Bearer $RENDER_TOKEN" -H "Content-Type: application/json" \
  -d '{"name":"ghcr","registry":"GITHUB","username":"<you>","authToken":"<gh token with read:packages>","ownerId":"<your Render owner id>"}'

curl -X POST https://api.render.com/v1/services \
  -H "Authorization: Bearer $RENDER_TOKEN" -H "Content-Type: application/json" \
  -d '{
    "type": "web_service", "name": "iba-triage", "ownerId": "<owner id>",
    "image": {"ownerId": "<owner id>", "imagePath": "ghcr.io/<you>/iba-triage:latest",
              "registryCredentialId": "<credential id from above>"},
    "serviceDetails": {"runtime": "image", "env": "image", "plan": "free",
                        "region": "frankfurt", "healthCheckPath": "/health"},
    "envVars": [
      {"key": "NEBIUS_API_KEY", "value": "..."}, {"key": "TAVILY_API_KEY", "value": "..."},
      {"key": "MODEL_FAST", "value": "..."}, {"key": "MODEL_MID", "value": "..."},
      {"key": "MODEL_REASON", "value": "..."}, {"key": "MODEL_EMBED", "value": "..."},
      {"key": "MODEL_PRICES", "value": "..."}, {"key": "MAX_SPEND_USD", "value": "0.80"},
      {"key": "APP_ENV", "value": "prod"},
      {"key": "HF_INDEX_REPO", "value": "<you>/iba-guideline-index"},
      {"key": "HF_TOKEN", "value": "<a READ-ONLY token>"}
    ]
  }'
```

Notes:
- `HF_TOKEN` here only needs **read** access to the one dataset — it downloads the index at
  boot and nothing else. Don't reuse a write token.
- Render doesn't support mounting a private external volume (unlike Nebius Compute or Hugging
  Face Spaces' `-v hf://datasets/...`), which is why the index is fetched at startup instead of
  mounted; `fetch_remote_index.py` no-ops when `HF_INDEX_REPO` is unset, so local dev and a
  future Nebius deploy are unaffected.
- **Free-tier caveats:** the instance has no persistent disk, so `.cache/spend.json` and the
  outbreak cache (`.cache/outbreak/`) reset on every restart, and it spins down after 15 minutes
  idle (cold start ≈ 40 s on wake — hit `/health` a minute before a demo to warm it up).
- We tried Hugging Face Spaces first: as of this hackathon, **Docker and Gradio Spaces require
  a paid PRO subscription** even on the free `cpu-basic` hardware — only fully static sites are
  free. Worth knowing if you're following an older tutorial that assumes otherwise.

#### Redeploying after a code change

**Render deploys from a pre-built container image tagged `latest` in GHCR — it does not watch
this GitHub repo.** Pushing a commit, or even clicking "Manual Deploy" in the Render dashboard,
does nothing by itself: a manual deploy just re-pulls whatever `latest` already points to in the
registry. The only way changes reach the live URL is rebuilding and pushing a new image, then
explicitly triggering a deploy. Exact steps, in order:

```bash
# 1. Build for linux/amd64 (Render's arch) and push BOTH an explicit version tag and `latest`.
#    Using the commit SHA as the version tag means the Render dashboard (Events tab, or the
#    "image" field on a deploy) always shows exactly which commit is actually running.
SHA=$(git rev-parse --short HEAD)
docker buildx build --platform linux/amd64 \
  -t ghcr.io/jemiridaniel/iba-triage:latest \
  -t ghcr.io/jemiridaniel/iba-triage:$SHA \
  --push .

# 2. Trigger the deploy (the dashboard's "Manual Deploy" button does the same thing)
curl -X POST https://api.render.com/v1/services/srv-dar4op0473hc739qte6g/deploys \
  -H "Authorization: Bearer $RENDER_TOKEN" -H "Content-Type: application/json" \
  -d '{"imageUrl":"ghcr.io/jemiridaniel/iba-triage:latest"}'

# 3. Poll until it's live (or watch the Events tab in the dashboard instead)
curl -s -H "Authorization: Bearer $RENDER_TOKEN" \
  https://api.render.com/v1/services/srv-dar4op0473hc739qte6g/deploys/<deploy id from step 2> \
  | grep -o '"status":"[a-z_]*"'

# 4. Confirm it's the build you expect (matches the SHA from step 1) and spot-check behaviour
curl -s https://iba-triage.onrender.com/health
curl -s https://iba-triage.onrender.com/meta | grep -o '"dose_table_verified":[a-z]*'
uv run python -m scripts.demo_check --url https://iba-triage.onrender.com --repeat 3
```

`RENDER_TOKEN` is a Render API key (Account Settings → API Keys); the GHCR push needs a GitHub
token with `write:packages` (`gh auth refresh -h github.com -s write:packages,read:packages`
once, then `gh auth token | docker login ghcr.io -u jemiridaniel --password-stdin`).

**Should this move to deploying straight from the GitHub repo instead?** For the time that's
left, no — the switch itself carries its own risk (disconnecting and reconnecting the service's
source, or standing up a new one, right before submission) for a benefit that only pays off over
many more deploys than are left in this project. If you do want it later: Render's GitHub-connected
services need their GitHub App installed with access to this repo (an interactive OAuth screen —
the reason this deploy used a registry image in the first place), a `Dockerfile`-based Web Service
pointing at this repo instead of an image, and the same env vars re-entered since they're
per-service, not portable between a from-image and a from-repo service. Once connected, a plain
`git push` to `main` deploys automatically and steps 1-2 above disappear — worth doing after
submission, not before.

### Evaluation

60 synthetic vignettes ([eval/vignettes.jsonl](eval/vignettes.jsonl)) following SPEC §6, each
with gold labels and a guideline rationale, **pending clinician review**
([docs/CLINICAL_REVIEW.md](docs/CLINICAL_REVIEW.md)). `eval/run_eval.py` runs the routed,
reason-only and fast-only configs (resumable, cached, confirms before spending over $0.50);
`--outbreak off|mock|live` measures outbreak lift (`live` uses the real Tavily search; `mock`
stands in for it offline). `eval/report.py` writes
[eval/results/report.md](eval/results/report.md) and charts. Current results are a 10-case
development check only.

```bash
uv run python -m eval.run_eval --config routed --limit 10 --stratified --judge
uv run python -m eval.run_eval --config routed --category suspected_lassa --outbreak mock
uv run python -m eval.report
```

`uv run pytest -m live` runs opt-in tests against real APIs (spends credits).

<!--
## Clinical review

Reviewed by Dr. Oluba O. Amos, PharmD ([@AmazingAmos](https://github.com/AmazingAmos)).
Credited for clinical review only, not a project collaborator or commit co-author (the
hackathon requires sole ownership by Daniel Jemiri). See docs/CLINICAL_REVIEW.md.

Uncomment this section once docs/CLINICAL_REVIEW.md's sign-off is dated and complete.
-->

## Clinical review findings — roadmap

Dr. Oluba O. Amos, PharmD, reviewed Ibà's clinical logic on 2026-10-10 (full review:
docs/review/CLINICAL_REVIEW_completed.pdf and docs/review/clinical-logic-review_completed.pdf,
as returned; our response is tracked in docs/CLINICAL_REVIEW.md). His safety-critical findings
were acted on immediately (see git history, 2026-10-10: the Lassa any-state floor and the
sunken-eyes-plus-vomiting cholera rule). The items below are real gaps he identified that are
**not yet implemented** — deferred, not rejected:

- **Pre-referral dosing.** NMEP (p.32, 33, 45) gives pre-referral doses for a patient being
  referred for severe malaria (IM/rectal artesunate, IM artemether, IM quinine) — not yet in
  `backend/app/rules/dosing.py`. Should be shown only to trained staff, always led with "refer
  now," never in place of referral.
- **Vomiting-triggered re-dosing.** NMEP: repeat the AL dose once if vomited within 30 minutes;
  a second vomit means switch to a parenteral antimalarial and refer. Not yet implemented.
- **AL dose on Refer within 24h.** Currently `doses_for()` only attaches a dose at Treat &
  monitor; an RDT-positive patient triaged Refer within 24h gets no dose shown, which can delay
  starting the ACT by a day.
- **New intake fields.** Pregnancy status (women 15–49), measured temperature, exposure history
  (contact/rodents/travel) in Lassa-risk areas, and diarrhoea detail (stool count, vomiting,
  urine, drinking) are not currently asked. He also recommends asking age first, since it
  decides which age-band rules apply, before fever duration/RDT/treatment response.
- **Afebrile Treat & monitor advice.** The 2026-09-28 fix correctly stops fever-specific advice
  from appearing on afebrile cases, but currently leaves no advice at all in its place. He
  recommends a dedicated non-fever block (NCDC Plan A: ORS volumes by age, return signs, zinc,
  cluster reporting to the DSNO).

See docs/CLINICAL_REVIEW.md for the full set of findings, including those already applied.

### Endemicity baseline update (2026-10-13)

Applied his CSM and cholera recommendations to `data/endemicity.yaml` after his final,
signed-off review resolved two earlier contradictions (whether `data/endemicity.yaml` had
actually been supplied to him, and whether the Lassa five includes Benue or Ebonyi — both
now settled: supplied, and Benue).

**CSM**: applied his belt definition in full — all 19 northern states plus the FCT as
high_burden (up from 10), the 7 southern belt states he named as reported, peak months
widened to October–June. Four states his own evidence table doesn't place either way
(Akwa-Ibom, Anambra, Ebonyi, Oyo) were kept in `reported` rather than dropped, since
`reported` never escalates a triage level on its own — there's no safety cost to keeping an
unaddressed state informational-only.

**Cholera — evidence bar.** His hotspot method uses four evidence tiers (hotspot, recurrent,
single-report, watch); our schema has two (`high_burden`, `reported`). We collapsed his four
into our two rather than adding new tiers, so a state's weight within "reported" isn't
visible from the tier alone — the full four-tier breakdown is preserved in
`data/endemicity.yaml`'s `source:` text for anyone who wants it.

His cholera hotspot recommendation (Bayelsa, Abia, Zamfara, Katsina, Bauchi, Niger, Kano,
Borno) was **applied as given, not overridden** — three of those eight (Bauchi, Kano, Borno)
only reach his 3-report threshold by counting a 2026 NCDC-DG/UNICEF/WHO press statement, and
his own caveat is that this layer should be *confirmed against primary documents*, not that
it's unusable. We independently checked: a genuine primary document (UNICEF Nigeria Cholera
Response Situation Report, Jul–Aug 2026) confirms Bauchi and Borno directly, so those two
carry no caveat. No primary document naming a state-wide 2026 Kano cholera burden was found
— only a press-covered, lab-confirmed outbreak at a single custodial centre, which is a real
but narrower fact than a state-level NCDC finding. Kano **keeps its high_burden tier** per
his recommendation; it also carries `evidence: press_2026_unconfirmed` plus a note naming
the source gap, shown in the UI ("2026 reports, primary source pending") and the reasoning
prompt. **No state loses high-burden status on the strength of our own uncertainty about its
newest evidence** — the flag is additive transparency, not a downgrade.

### Cholera: anchored on the 2024 NCDC sitrep (2026-10-11)

The paragraph above was written before we actually had Dr. Amos's source documents in hand.
He then sent two: the **NCDC Cholera Situation Report, Epi Week 22 2024** (primary — dated,
per-state case counts) and the **National Technical Guidelines for Integrated Disease
Surveillance and Response (IDSR), 3rd ed. 2019** (case definitions only, no per-state burden
list). The IDSR guideline is **not ingested** into the retrieval index — it adds no state-level
data our existing NCDC case-definition documents don't already cover — but it's a credible
post-hackathon source if IDSR's own case-definition wording is ever needed verbatim; noted here
rather than silently dropped.

We re-anchored the cholera baseline on the 2024 sitrep directly instead of the previous
multi-year/2026-statement method:

- **high_burden** is still his eight (Bayelsa, Abia, Zamfara, Katsina, Bauchi, Niger, Kano,
  Borno) plus **Lagos** (nine total). Seven are now cited directly from the sitrep's own
  cumulative Epi-week-1–22 case counts (Table 7, p.6): Bayelsa 442, Zamfara 64, Abia 51,
  Bauchi 46, Katsina 28, Kano 13, Niger 11 — no caveat needed. This **removes** the
  `press_2026_unconfirmed` flag we'd put on Kano last round (it only needed the 2026 layer
  before this document was in hand) and **adds** it to **Borno** instead — the sitrep records
  just 1 cumulative Borno case, too thin to stand alone, so Borno's tier still rests on the
  2026 UNICEF report, same as before, just now visibly flagged rather than silently equal to
  the other seven states' primary evidence. **Lagos** is new to the list: also just 1
  cumulative case in the sitrep, but the report's window ends 2 Jun 2024 and can't evidence
  Lagos going quiet after that date — the same report separately records a state-level
  cholera Emergency Operations Centre activated in Lagos (p.7). Lagos carries
  `evidence: report_window_gap` for this reason, kept high_burden on continuity grounds, not
  on this report's case count.
- **peak_months is now all 12.** The sitrep's own monthly table (p.1) shows cumulative cases
  of 294 (Jan), 291 (Feb), 210 (Mar), 60 (Apr), 27 (May) — a clear January–March peak that
  matches neither our previous June–October window nor Dr. Amos's suggested May–October.
  Rather than hard-code a season the primary data itself contradicts, every month is treated
  as in-season.
- **Methodology correction**: the very first version of this list (before any clinical review)
  was built from the sitrep's "Active" column. NCDC's own footnote defines "Active" as cases
  reported in the **last 3 weeks** — an outbreak-recency signal, not the multi-year endemicity
  this baseline is meant to capture. Recorded so the mistake isn't repeated.
- Nothing already in `reported` was deleted, including Benue, Enugu and Oyo, per instruction
  — Lagos is the only change, moved out of `reported` into `high_burden`.
- **Open finding, not yet actioned**: the sitrep (Table 7, p.6) also names Edo (1 case), FCT
  (1), Kwara (2) and Ogun (2) as having reported cholera cases in 2024. None of the four is
  currently in either tier. Flagged here rather than added unilaterally, since it changes
  what a health worker sees even though `reported` never escalates on its own.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
