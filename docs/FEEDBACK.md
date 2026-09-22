# Builder feedback: Nebius Token Factory, Nemotron, Serverless

Feedback from building **Ibà**, a fever-triage copilot for primary health care workers in
Nigeria (Nebius x NVIDIA Global AI Hackathon, Best Apps and Agents). Every item comes from
running code in this repo. Each has a reproduction, the impact on a real app, the workaround
we shipped, and a concrete suggestion. We add to this file as we go.

**Setup:** Token Factory base URL `https://api.tokenfactory.nebius.com/v1/`, `openai` Python SDK
3.17, Python 3.12, macOS. Models: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`,
`nvidia/Nemotron-3_5-Lightning`, `nvidia/nemotron-3-super-120b-a12b`,
`nvidia/Nemotron-3-Ultra-550b-a55b`. First measurements: 2026-09-22.

Test input (synthetic, used everywhere below): a structured-extraction prompt (JSON schema in
the system message, ~746 prompt tokens) plus the user text
`"Pikin 3 years, hot body 4 days, vomit everything, RDT negative, e don dey sleep too much"`.
Raw responses are saved in [tests/fixtures/nemotron_responses.json](../tests/fixtures/nemotron_responses.json).

## Summary

| # | Area | Issue | Severity for us |
|---|---|---|---|
| F1 | Nemotron / API | A reply truncated mid-reasoning returns raw reasoning in `content`, including draft JSON that parses | **High**: silent wrong output |
| F2 | Privacy | Nano responses echo the full prompt in `prompt_text` | **High** for health apps |
| F3 | Nemotron / docs | `/no_think` is ignored; only `chat_template_kwargs` turns reasoning off, and it's undocumented | Medium |
| F4 | Observability | Reasoning is invisible and inconsistently reported across models | Medium: cost is hard to see |
| F5 | Catalogue | No Nemotron model advertises `json_mode` / `structured_outputs` | Medium |
| F6 | Billing DX | New account without billing gets "You have exhausted your budget" | Low |
| F7 | Catalogue | Ultra and Super report `per_request_limits` of `1e10` | Low |
| F8 | Embeddings | Embedding latency is high and variable (0.4–7.3 s for one short query) | Medium: in the critical path |
| F9 | Catalogue | No NVIDIA embedding or reranking model | Medium for NVIDIA-only builds |
| F10 | Nemotron / API | No way to cap or budget reasoning tokens | **High**: truncation breaks structured output |
| F11 | Serving | Generation throughput varies 3x between identical calls | Medium: p95 latency unpredictable |
| F12 | Nemotron / API | Model rewrites a URL it was told to copy exactly (adds `www.`) | **High**: a citation is only as good as its URL |
| T1 | Tavily | `published_date` empty on every result, so recency depends on the LLM reading dates out of page text | **High**: recency is a safety property here (worked around) |
| T3 | Tavily | National sitreps yield signals for other states than the patient's | Low (rules filter by state; prompt and UI now group them) |

---

## F1. Truncated replies leak raw reasoning into `content`, including draft JSON

**Observed.** With reasoning at its default, Lightning spent all of `max_tokens=1500` reasoning
on a simple extraction and stopped with `finish_reason: "length"`. `message.content` held the
raw reasoning ("Here's a thinking process: 1. **Analyze User Input:** ..."), with no `<think>`
tags or other delimiter. `message.reasoning` and `message.reasoning_content` were present but
empty. The reasoning included a complete **draft** JSON object.

When reasoning *did* finish but the answer was cut off (1,478 reasoning tokens, 22 answer
tokens), the reasoning was stripped and `content` held only the partial answer. So what
`content` contains depends on *where* the cut happened, and nothing in the response says so.

**Why it matters.** A standard "find the JSON in the reply" parser (first `{` to last `}`)
**succeeded** on the truncated reply and returned the model's draft as if it were the answer.
Our client reported success. In a clinical triage app, that is a silent wrong result.

**Repro.**
```python
client.chat.completions.create(
    model="nvidia/Nemotron-3_5-Lightning",
    max_tokens=1500,
    temperature=0,
    messages=[
        {"role": "system", "content": "<extraction prompt with JSON schema>"},
        {"role": "user", "content": "Pikin 3 years, hot body 4 days, ..."},
    ],
)
# -> finish_reason="length", usage.completion_tokens=1500, reasoning_tokens=1500,
#    content="Here's a thinking process: ... { \"age_years\": 3, ... } ... Check schema..."
```

**Workaround shipped.** The client rejects every `finish_reason == "length"` reply before
parsing (`LLMTruncatedError`) and never caches it. Structured-extraction steps run with
reasoning off; the reasoning step gets an 8,192-token budget.

**Suggestions.**
1. Never put unterminated reasoning in `content`. Put it in `reasoning_content`, or drop it,
   the same as when reasoning completes.
2. Add a signal when the cut happened during reasoning (e.g. `finish_reason: "length"` plus
   `reasoning_truncated: true`).
3. Document how reasoning tokens interact with `max_tokens`, with a recommended minimum
   budget for reasoning models.

## F2. Nano echoes the full prompt in `prompt_text`

**Observed.** Every Nano response included a top-level `prompt_text` field containing the full
rendered prompt: system message, user text, and chat-template tokens. It also returned
`prompt_token_ids`. Lightning responses returned `prompt_token_ids` but no `prompt_text`.

**Why it matters.** Health apps send patient descriptions. Any code that logs or stores raw
API responses (a common debugging habit, and the default in many tracing tools) now
stores patient text too, even when the app itself carefully avoids logging prompts. It also
adds the whole prompt to every response payload.

**Workaround shipped.** Our client stores only `content`, `finish_reason` and usage from each
response. We never log raw responses, and the test fixtures contain only synthetic text.

**Suggestions.** Don't return `prompt_text` / `prompt_token_ids` by default (make them
opt-in debug fields), and make behaviour consistent across models.

## F3. `/no_think` is ignored; `chat_template_kwargs` works but isn't documented

**Observed.** Both models advertise `"reasoning"` in `supported_features`, but the catalogue
doesn't say how to control it.

| Method | Nano | Lightning | Super |
|---|---|---|---|
| `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` | ✅ off (892 → 128 completion tokens) | ✅ off (1500+ → 125, `reasoning_tokens: 0`) | ✅ off (`reasoning_tokens: 0`) |
| `/no_think` appended to the system prompt | ❌ still reasons (784 tokens) | ❌ still reasons, truncated | not tested |

**Impact.** With reasoning on, a trivial extraction took 5.4–5.7 s instead of 1.1–1.7 s and cost
3–5x more. Without knowing the switch, every call to a "fast" model pays for reasoning.

**Suggestions.** Document the reasoning control per model on the model page and in
`/v1/models` (e.g. `reasoning: {"default": "on", "toggle": "chat_template_kwargs.enable_thinking"}`).
A first-class request parameter (`reasoning: {"enabled": false}` or `reasoning_effort`) would
be even better.

## F4. Reasoning is invisible and inconsistently reported

**Observed.**
- **Lightning:** `usage.completion_tokens_details.reasoning_tokens` is reported, but the
  reasoning text is never returned (`reasoning` / `reasoning_content` always empty).
- **Nano:** the reasoning text is returned in `message.reasoning`, but `reasoning_tokens` is
  **not** reported, so you can only estimate it as completion tokens minus answer tokens.
- **Super:** reports `reasoning_tokens` (1,799 of 2,361 completion tokens in our run) and
  returns the reasoning text **twice**, identically, in both `message.reasoning` and
  `message.reasoning_content` (7,677 characters each). The most complete of the three, but
  the duplicate doubles the payload.

Three models from the same family give three different response shapes:

| | reasoning text | `reasoning_tokens` |
|---|---|---|
| Nano | `reasoning` | not reported |
| Lightning | not returned | reported |
| Super | `reasoning` + `reasoning_content` (duplicated) | reported |

**Impact.** Reasoning tokens are billed as completion tokens. Our cost dashboard (per-step
tokens and cost, needed to justify model routing) can't separate reasoning cost from answer
cost on Nano. On Lightning, reasoning can't be inspected to debug a bad answer.

**Suggestions.** Report `reasoning_tokens` for every reasoning model, and use one field
name for reasoning text (`reasoning_content`, as in other OpenAI-compatible APIs) across
models.

## F5. No Nemotron model advertises `json_mode` or `structured_outputs`

**Observed.** In `GET /v1/models?verbose=true`, all four Nemotron models list only
`["tools", "reasoning"]`. Other models in the catalogue list `json_mode` and
`structured_outputs`.

**Impact.** Agent pipelines depend on structured output. We send no `response_format` and rely
on prompt instructions, Pydantic validation and one corrective retry. That works (every
completed reply in our tests was clean JSON with no code fences), but every step pays for
validation and occasional retries, and it's the main reliability risk for Nemotron agents.

**Further evidence (2026-09-22).** In our eval's citation judge, Super with reasoning off
returned invalid JSON **twice in a row** for a two-field schema
(`{"verdict": ..., "reason": ...}`): the `reason` string was left unterminated, so the
corrective retry failed too. It was 1 failure in 11 judge calls, but without JSON mode there
is no way to make it impossible.

**Suggestions.** Enable `response_format` (JSON mode, ideally JSON Schema) for Nemotron models,
or document that it's accepted even though it isn't advertised.

## F6. Unbilled account gets "You have exhausted your budget"

**Observed.** Our first call, before billing was set up, returned
`402 Payment Required: You have exhausted your budget. Please add funds to continue using the API.`
`GET /v1/models` worked fine with the same key.

**Impact.** "Exhausted" suggests credits were used, which sent us looking for a leak in our own
code. That cost time during onboarding.

**Suggestion.** Use a distinct message for "no billing / no credits yet" with a link to the
billing page or credit-claim flow.

## F7. `per_request_limits` of `1e10` for Ultra and Super

**Observed.** Ultra and Super report `tokens_per_minute` and `requests_per_minute` of
`10000000000.0`. Nano reports 800k TPM / 100 RPM; Lightning 400k TPM / 600 RPM.

**Impact.** We use these values to size eval batch concurrency; `1e10` looks like a placeholder,
so we can't rely on it.

**Suggestion.** Publish the real limits, or `null` when there is no fixed limit.

## F8. Embedding latency is high and variable

**Observed.** `Qwen/Qwen3-Embedding-8B` via `POST /v1/embeddings`:

| Call | Input | Latency |
|---|---|---|
| Retrieval query (CLI) | 69 tokens | 5.5 s |
| Retrieval queries (3 demo cases, Docker) | ~70 tokens each | 1.1 s, 7.3 s, 0.4 s |
| Retrieval query (phone test, Docker) | 45 tokens | 7.2 s (34% of a 21.1 s case) |
| Ingest batches (32 chunks) | 10k–21k tokens | 3.3–13.3 s |

**Impact.** Retrieval sits in the critical path of every triage request. In one demo run, the
single query embedding was 31% of the case's end-to-end latency (7.3 s of 23.6 s), more
than the fast-model steps combined. The variance makes latency hard to promise to users.

**Workaround.** None yet. Options we're weighing: embed the query in parallel with the
outbreak step, or cache query embeddings.

**Suggestions.** Publish latency expectations for embedding models, and consider a
low-latency tier for short queries.

## F9. No NVIDIA embedding or reranking model in the catalogue

**Observed.** `GET /v1/models?verbose=true` lists one embedding model
(`Qwen/Qwen3-Embedding-8B`) and no rerankers. None are NVIDIA models.

**Impact.** For a hackathon that asks for NVIDIA models, retrieval is the one step we can't
run on NVIDIA. RAG agents normally need both an embedder and a reranker.

**Suggestion.** Serve an NVIDIA retrieval embedding model and reranker alongside Nemotron.

## F10. No reasoning budget or effort control

**Observed.** Our reasoning step hit `finish_reason: "length"` at 8,192 max tokens on a real
demo case (7,021 of them reasoning), so the whole assessment failed safe to "Refer — could not
complete". Looking for a cap, we sent the same prompt to Super with eight plausible options
(`scripts/probe_reasoning_budget.py`), asking for a 64-token budget:

| Option | Reasoning tokens |
|---|---|
| none (baseline) | 1,904 |
| `chat_template_kwargs: {reasoning_budget: 64}` | 1,902 |
| `chat_template_kwargs: {thinking_budget: 64}` | 3,401 |
| `chat_template_kwargs: {reasoning_effort: "low"}` | 649 (**wrong answer**) |
| `reasoning_effort: "low"` (top level) | 1,116 |
| `reasoning: {effort: "low"}` | 2,573 |
| `max_thinking_tokens: 64` | 1,904 |
| `thinking: {type: "enabled", budget_tokens: 64}` | 1,904 |

No option enforces a budget, and none is rejected either: unknown fields are silently
accepted, so there is no way to tell "not supported" from "ignored". The only option that
shortened reasoning also produced a wrong answer, and run-to-run variance (1,904 vs 2,573 vs
3,401 tokens for the same prompt at temperature 0) is large enough that a single sample proves
nothing.

**Impact.** With structured output, long reasoning eats the token budget and the answer is
truncated, which for us means a failed assessment. We now allow 16,384 tokens for that step
and retry with reasoning off if it still truncates; both cost money and latency.

**Suggestions.** Support a reasoning budget (`reasoning: {max_tokens: N}`) or a documented
effort level, and reject unknown request fields with a 400 instead of ignoring them.

## F11. Generation throughput varies about 3x between identical calls

**Observed.** Same model (Super), same pipeline step, similar token counts, minutes apart:

| Case | Completion tokens | Latency | Throughput |
|---|---|---|---|
| la-01 | 4,147 | 18.9 s | 219 tok/s |
| um-01 | 3,785 | 15.9 s | 238 tok/s |
| pd-01 | 6,907 | 91.6 s | 75 tok/s |
| sm-02 (eval) | ~4,000 | 83.4 s | ~48 tok/s |

Two of twelve calls took 4-5x longer than their token count predicts. Nothing in the response
distinguishes them (`finish_reason: "stop"`, no rate-limit headers, no 429).

**Impact.** Our p95 latency (69.9 s over 10 cases) is set by these outliers, not by our
pipeline. For a point-of-care app that is the number a user feels.

**Suggestions.** Publish expected throughput per model, and expose queue/wait time in the
response or headers so clients can tell a slow queue from a slow prompt.

---

## F12. Nemotron silently rewrites URLs it is told to copy exactly

**Where.** Outbreak extraction on `nvidia/Nemotron-3_5-Lightning`, reasoning off. The prompt
says: *"url: copy the URL of the result the signal came from, exactly. Never invent a URL."*

**Reproduction.** `uv run python -m scripts.outbreak_trace --fresh` (live Tavily, Ondo,
2026-09-22). Two Lassa signals were extracted from two different NCDC sitrep PDFs. One URL was
copied verbatim. The other came back as

```
given:    https://ncdc.gov.ng/themes/common/files/sitreps/b0fedda076a0b27d21d5a09678dd69a0.pdf
returned: https://www.ncdc.gov.ng/themes/common/files/sitreps/b0fedda076a0b27d21d5a09678dd69a0.pdf
```

a `www.` the source text never contained. The two source URLs were adjacent in the prompt and
differed only in their hash, so this is not a copy of some other result: the model normalised
the host.

**Impact.** A rewritten URL is indistinguishable from an invented one. Ours happened to still
resolve, but a health worker following a citation to a 404 — or to a different document —
would be worse than no citation. This is exactly the failure that mock data never shows: eight
weeks of fixture-based tests never produced an altered URL, and the first live call did.

**Workaround.** We never trust an extracted URL. `filter_signals` requires the URL to be
**byte-identical to one the search actually returned** (`backend/app/tools/outbreak.py`), so
the rewritten one was dropped with the reason *"URL not among the search results (invented or
altered)"*. 1 of 2 signals dropped on the first live call.

**Suggestions.** Document that verbatim copying of identifiers is not reliable even with
reasoning off, and that callers must validate. A `response_format` with a URL-typed enum
constrained to supplied values would remove the class of bug entirely.

---

## Tavily

Tavily was wired in on 2026-09-22 (`backend/app/tools/outbreak.py`). Two searches per state per
day, `search_depth="advanced"`, `time_range="month"`, restricted to `ncdc.gov.ng`, `who.int`,
`reliefweb.int`, `afro.who.int`.

### T1. `published_date` is empty on every result, so dates have to come from the model

**Reproduction.** `uv run python -m scripts.outbreak_trace --fresh`, saved at
[eval/results/tavily/trace_ondo.txt](../eval/results/tavily/trace_ondo.txt). **10 of 10 results
returned `published_date: None`**, across `ncdc.gov.ng`, `www.ncdc.gov.ng`, `iris.who.int`,
`reliefweb.int` and `www.afro.who.int`, with `time_range="month"` set. The ReliefWeb result even
carries the date in its own title ("Epi Week 34: 17th – 23rd August 2026") and in its slug.

**Impact.** This is the single biggest problem live search created for us. Recency is a safety
property here: a Lassa sitrep from last season must not read as current. With no date from the
API we have to ask the LLM to read the date out of the page text, and it is not reliable — for
the Lagos query, **all 5 extracted signals came back dated `2026-09-22` (today)** when the
underlying sitrep was epi week 34, a month old. Our filter only rejects unparseable or future
dates, so "today" always passes. We would rather trust a metadata field than a model.

**Suggestions.** Return `published_date` whenever the crawler has it (ReliefWeb exposes it in
the page, the URL and its own API). Failing that, a documented "date unknown" marker, so
callers can tell "no date" from "not extracted". A `min_published_date` filter would let us
enforce recency server-side rather than after the fact.

**What we shipped** (`backend/app/tools/dates.py`). We stopped asking the model for the date.
Explicit patterns are read out of the result's title, text and URL slug — ISO dates, "Epi Week
34" (resolved to that ISO week's Sunday), "week ending DD/MM/YYYY", and long-form dates — and
the model's answer is used only when it is *older* than what the text says, so a model
answering "today" can never make a report current. No explicit date anywhere means
`report_date: null`, which is shown as "date unknown" and cannot escalate triage. A signal
over `STALE_DAYS` (60) old is labelled "older report" and likewise cannot escalate on its own.

Re-running the same Lagos query afterwards: the five Lassa signals that had been dated "today"
came back **2026-09-06**, read from the sitrep text. Two WHO bulletin entries that the model
had also dated "today" resolved to **2024-06-23** and are now correctly marked `older` — a
two-year-old report that would previously have read as current.

### T2. `include_domains_mode="restrict"` was exact

**10 of 10 URLs** across both queries were on the allow-list, including subdomains we wanted
(`iris.who.int`, `www.afro.who.int`) and none we did not. Our own `_is_trusted` check, which
re-validates every host with a suffix match, dropped nothing. For a clinical tool where an
invented or low-quality source is a safety issue, a search API that respects a domain allow-list
exactly is the feature that made Tavily usable at all.

### T3. Relevance on a narrow topic is mixed, and a national report answers for every state

Of 8 unique trusted results for Ondo, 2 were plainly off-topic (an NCDC "invitation to tender"
page, a WHO infographics index at `?page=426`). Not harmful — the extractor ignored them — but
they cost prompt tokens.

More interesting: NCDC publishes **national** sitreps, so a search for Lagos returns a report
covering Ondo, Edo, Bauchi, Taraba and Benue. The extractor dutifully emits a signal per state.
Our deterministic rule matches the patient's own state before it escalates
(`active_outbreak()`), so a Lagos patient is never escalated on an Ondo outbreak.

**What we shipped.** The reasoning prompt now groups signals under "IN THIS PATIENT'S STATE"
and "ELSEWHERE IN NIGERIA (context only — these must NOT raise this patient's triage level or
be described as local)", and the UI shows the same two groups, with the second greyed and
marked "context only". The rule keeps the final say either way.

### T4. Latency and credits

| | Wall time |
|---|---|
| State-specific query, uncached (`search_depth="advanced"`) | 4.4 s |
| Shared `"NCDC situation report"` query, second state onward | 0.24 s |
| Both queries, one uncached state | 4.6 s |
| Outbreak step end to end (2 searches + extraction) | **5.4 s** |
| Outbreak step, cached (same state, same day) | **1 ms** |

The fixed second query appears to be cached Tavily-side, which is why it drops from seconds to
240 ms — useful, and worth documenting.

---

## What worked well

- **OpenAI compatibility:** the stock `openai` SDK worked with only `base_url` changed,
  including `extra_body` passthrough.
- **`/v1/models?verbose=true`** gives prices, context length, modality and features in one call.
  We generate our cost table and spend guard from it. Per-token prices as strings avoid
  float rounding.
- **Nemotron output quality:** with reasoning off, Lightning extracted every field of a
  Nigerian Pidgin case correctly in 1.1 s for $0.000075, and reasoning-off replies had no
  code fences or preamble.
- **Pricing:** Nano and Lightning at $0.06 / $0.24 per 1M tokens make a routed pipeline cheap
  enough for a public demo.

## Measurements log

**2026-09-22, full pipeline, one synthetic case** (adult, RDT negative, 5 days of fever, Ondo,
mock Lassa signal): Lightning for intake and outbreak extraction (reasoning off), Super for
reasoning (on) and the referral note (off).

| Step | Model | Reasoning | Prompt / completion (reasoning) tokens | Latency | Cost |
|---|---|---|---|---|---|
| intake | Lightning | off | 755 / 110 (0) | 1.3 s | $0.000072 |
| outbreak extraction | Lightning | off | 548 / 78 (0) | 0.7 s | $0.000052 |
| reason | Super | on | 1,522 / 2,361 (1,799) | 15.2 s | $0.002582 |
| compose | Super | off | 567 / 180 (0) | 1.7 s | $0.000332 |
| **total** | | | 6,121 tokens | 19.0 s | **$0.003037** |

Reasoning was 76% of Super's completion tokens and 80% of the case's latency. Every reply
finished with `finish_reason: "stop"` and was clean JSON on the first attempt (no retries).

**2026-09-22, same case with the real guideline index** (905 chunks from 6 NCDC/WHO PDFs,
Qwen3-Embedding-8B; live outbreak search off, static endemicity baseline on): Refer now,
Lassa fever top of the differential citing NCDC Lassa guideline §1.1.2 (pp. 8–9). Reasoning
step: 3,162 prompt / 2,451 completion tokens (1,821 reasoning), 11.5 s, $0.0032. Whole case
7,495 tokens, 20.1 s, $0.0036. Building the index: 29 embedding calls, ~500k tokens, $0.0065.

**2026-09-22, production Docker image, three demo cases** (Lightning + Super, streaming):

| Case | First event (rule check) | Final result | Total | Cost |
|---|---|---|---|---|
| Child with danger signs | 1.5 s (Refer now shown) | 15.0 s | 14.8 s of steps | $0.0033 |
| Adult, Lassa suspicion, Ondo | 0.7 s | 23.6 s | 23.5 s | $0.0039 |
| Uncomplicated malaria | 0.7 s | 11.4 s | 11.3 s | $0.0031 |

Super's reasoning step was 8.6–13.4 s of each case. Streaming the rule result first means the
health worker sees a danger-sign referral within ~1.5 s instead of ~15 s.

**2026-09-22, Ultra vs Super** (3 identical cases, reasoning on, uncached; routed pipeline):

| | Super | Ultra |
|---|---|---|
| Wall-clock per case | 22 s, 22 s, 101 s | 12 s, 15 s, 20 s (run 1); 15 s, 16 s (run 2) |
| Reason step tokens (prompt / completion / reasoning) | 4.2k / 4.1k / 3.2k | 4.2k / 4.6k / 3.3k |
| Cost per case | $0.005-0.008 | $0.013-0.027 |
| Failures | 0 of 13 runs today | 1 of 5: reason output failed schema validation twice (fail-safe "Refer") |

Ultra is faster per token and answered in fewer, longer reasoning steps; it also produced the
only structured-output failure of the day (see F5: no JSON mode).

**2026-09-22, first live Tavily runs** (Lightning for extraction, Super for reasoning):

| | Result |
|---|---|
| Demo cases, 3 runs each, live search | **9 / 9 pass**, 0 fallbacks, p50 27.4 s, max 45.1 s, $0.0575 |
| Outbreak lift, 10 suspected-Lassa cases, live vs off | Lassa in top 3 **80% → 100%**; Refer now **70% → 90%**; under-triage **20% → 10%** |
| Live Lassa signal found for the patient's own state | 8 of 10 cases (Ondo, Edo, Bauchi, Taraba, Plateau; not Lagos or Ebonyi) |
| Tavily searches used | 24 (2 per uncached state per day) |

The live signal fixed one under-triage (la-09, Plateau: treat-and-monitor → refer now) and
introduced one over-triage (la-10, Ondo: treat-and-monitor → refer now). The mock arm had
suggested a larger lift because it guaranteed a signal for every state; the real search finds
one for 8 of 10, which is the honest number.

**2026-09-22, after the T1 and T3 fixes** (same 10 cases, same day's live data):

| | Search off | Live, before fixes | Live, after fixes |
|---|---|---|---|
| Lassa in top 3 | 80% | 100% | 100% |
| Refer now | 70% | 90% | 90% |
| **Under-triage** | 20% | 10% | **0%** |

Both errors the live search had left went away: la-10 (Ondo) stopped over-triaging once stale
and out-of-state signals were labelled as such, and la-05 (Lagos) stopped under-triaging once
the prompt said plainly that the Lassa reports it could see were elsewhere in Nigeria. Demo
cases: 9/9 again, 0 fallbacks, p50 28.9 s, $0.0683.

## Still to evaluate

- Ultra: response shape, reasoning control, latency, cost.
- Serverless Endpoints: cold start, image size limits, scale to zero (week 3).
- Serverless Jobs for the eval batch (week 4).
- Whether 60 days is the right staleness threshold per disease (Lassa is seasonal; cholera is
  not), rather than one number for all of them.
- One transient failure worth watching: on 2026-09-22 the outbreak extraction for Kano failed
  twice in a row (degraded to "unavailable", nothing cached) and then succeeded minutes later
  on the same input. No error surfaced beyond the warning log.
