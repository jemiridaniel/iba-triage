"""Live smoke test: one chat_json call to MODEL_FAST, parsing a synthetic case into PatientCase.

    uv run python -m scripts.smoke_nano

Spends real credits (a fraction of a cent; guarded by MAX_SPEND_USD). Prints the raw
response shape (reasoning field? <think> blocks? code fences?), the parsed PatientCase,
and tokens / latency / cost. The raw response is saved to CACHE_DIR/smoke/ for inspection.
The prompt here is a throwaway; the real intake prompt lands with the pipeline (week 2).
"""

import json
import sys
from typing import Any

from backend.app.config import get_settings
from backend.app.llm.client import LLMClient, LLMError, summarize, track_usage
from backend.app.llm.spend import SpendLimitError
from backend.app.schemas import PatientCase

# Synthetic vignette from SPEC §2. Not a real patient.
SAMPLE = "Pikin 3 years, hot body 4 days, vomit everything, RDT negative, e don dey sleep too much"


def intake_messages(text: str) -> list[dict[str, str]]:
    schema = PatientCase.model_json_schema()
    schema["properties"].pop("raw_text", None)
    return [
        {
            "role": "system",
            "content": (
                "You convert a health worker's description of a febrile patient (English or "
                "Nigerian Pidgin) into JSON. Reply with ONE JSON object matching this JSON "
                "schema and nothing else. Use null for unknown values; do not guess. "
                "List in missing_fields any of age_years, fever_days, rdt_result that are "
                "unknown. Set language to 'pcm' if the text is Pidgin.\n"
                f"Schema:\n{json.dumps(schema, separators=(',', ':'))}"
            ),
        },
        {"role": "user", "content": text},
    ]


def describe_shape(raw: dict[str, Any]) -> list[str]:
    choice = raw["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning_keys = [k for k in msg if "reason" in k.lower() or "think" in k.lower()]
    usage = raw.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return [
        f"message keys:        {sorted(k for k, v in msg.items() if v is not None)}",
        f"reasoning fields:    {reasoning_keys or 'none'}"
        + "".join(f"\n    {k}: {len(str(msg[k] or ''))} chars" for k in reasoning_keys),
        f"<think> in content:  {'<think>' in content.lower()}",
        f"code fence:          {'```' in content}",
        f"starts with '{{':      {content.lstrip().startswith('{')}",
        f"finish_reason:       {choice.get('finish_reason')}",
        f"reasoning_tokens:    {details.get('reasoning_tokens', 'not reported')}",
        f"content ({len(content)} chars):\n{content}",
    ]


def main() -> int:
    settings = get_settings()
    raws: list[dict[str, Any]] = []
    client = LLMClient(settings, on_response=lambda r: raws.append(r.model_dump()))

    print(f"Model: {settings.model_fast or '(MODEL_FAST not set)'}")
    print(f"Spend so far: ${client.ledger.total():.6f} / cap ${settings.max_spend_usd:.2f}\n")

    error: Exception | None = None
    case = None
    with track_usage() as records:
        try:
            case, _ = client.chat_json(
                intake_messages(SAMPLE),
                PatientCase,
                model=settings.model_fast,
                step="smoke:intake",
                max_tokens=400,
            )
        except (LLMError, SpendLimitError) as exc:
            error = exc

    for i, raw in enumerate(raws, 1):
        print(f"=== raw response {i} of {len(raws)} ===")
        print("\n".join(describe_shape(raw)))
        print()
    if raws:
        out = settings.cache_dir / "smoke" / "nano_raw.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(raws, indent=2, default=str), encoding="utf-8")
        print(f"(raw responses saved to {out})\n")
    elif records:
        print("(served from disk cache; no live call made)\n")

    if case is not None:
        print("=== parsed PatientCase ===")
        print(case.model_dump_json(indent=2))
    else:
        print(f"=== FAILED: {type(error).__name__}: {error}")
        if error is not None and error.__cause__ is not None:
            print(f"    cause: {error.__cause__}")

    s = summarize(records)
    print("\n=== usage ===")
    for r in s["steps"]:
        print(
            f"{r['step']:<20} in={r['prompt_tokens']} out={r['completion_tokens']} "
            f"latency={r['latency_ms']:.0f}ms cost=${r['cost_usd'] or 0:.6f} cached={r['cached']}"
        )
    print(f"total spent this run: ${s['spent_usd']:.6f}")
    print(f"ledger total:         ${client.ledger.total():.6f}")
    return 0 if case is not None else 1


if __name__ == "__main__":
    sys.exit(main())
