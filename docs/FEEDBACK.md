# Builder feedback: Nebius Token Factory, Nemotron, Serverless

Feedback from building **Iba**, a fever-triage copilot for primary health care workers in
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

## Still to evaluate

- Ultra: response shape, reasoning control, latency, cost.
- Serverless Endpoints: cold start, image size limits, scale to zero (week 3).
- Serverless Jobs for the eval batch (week 4).
- Tavily (Builder credits pending).
