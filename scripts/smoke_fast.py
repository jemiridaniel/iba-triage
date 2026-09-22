"""Live smoke test: one chat_json call parsing a synthetic case into PatientCase.

    uv run python -m scripts.smoke_fast                              # MODEL_FAST, reasoning default
    uv run python -m scripts.smoke_fast --model <id> --reasoning kwargs-off
    uv run python -m scripts.smoke_fast --reasoning no-think

Reasoning modes:
    default     send nothing; the model reasons as it normally does
    kwargs-off  extra_body chat_template_kwargs={"enable_thinking": false}
    no-think    append "/no_think" to the system prompt

Spends real credits (fractions of a cent; guarded by MAX_SPEND_USD). Prints the raw response
shape, finish_reason, reasoning vs completion tokens, latency, cost and the parsed
PatientCase. Raw responses are saved to CACHE_DIR/smoke/<label>.json. Uses the disk cache
only if --use-cache is given, so each run is a real call.
"""

import argparse
import json
import sys
from typing import Any

from backend.app.config import get_settings
from backend.app.llm.client import LLMClient, LLMError, summarize, track_usage
from backend.app.llm.router import NO_THINK_DIRECTIVE, THINKING_OFF_KWARGS
from backend.app.llm.spend import SpendLimitError
from backend.app.schemas import PatientCase

# Synthetic vignette from SPEC §2. Not a real patient.
SAMPLE = "Pikin 3 years, hot body 4 days, vomit everything, RDT negative, e don dey sleep too much"


def intake_messages(text: str, *, no_think: bool = False) -> list[dict[str, str]]:
    schema = PatientCase.model_json_schema()
    schema["properties"].pop("raw_text", None)
    system = (
        "You convert a health worker's description of a febrile patient (English or "
        "Nigerian Pidgin) into JSON. Reply with ONE JSON object matching this JSON "
        "schema and nothing else. Use null for unknown values; do not guess. "
        "List in missing_fields any of age_years, fever_days, rdt_result that are "
        "unknown. Set language to 'pcm' if the text is Pidgin.\n"
        f"Schema:\n{json.dumps(schema, separators=(',', ':'))}"
    )
    if no_think:
        system = f"{system}\n{NO_THINK_DIRECTIVE}"
    return [{"role": "system", "content": system}, {"role": "user", "content": text}]


def describe_shape(raw: dict[str, Any]) -> list[str]:
    choice = raw["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning_keys = [k for k in msg if "reason" in k.lower() or "think" in k.lower()]
    usage = raw.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    lines = [
        f"message keys:        {sorted(k for k, v in msg.items() if v is not None)}",
        f"reasoning fields:    {reasoning_keys or 'none'}",
    ]
    for k in reasoning_keys:
        value = str(msg[k] or "")
        preview = value[:160].replace("\n", " ")
        lines.append(f"    {k}: {len(value)} chars" + (f"  '{preview}...'" if value else ""))
    lines += [
        f"<think> in content:  {'<think>' in content.lower()}",
        f"code fence:          {'```' in content}",
        f"starts with '{{':      {content.lstrip().startswith('{')}",
        f"finish_reason:       {choice.get('finish_reason')}",
        f"usage:               prompt={usage.get('prompt_tokens')} "
        f"completion={usage.get('completion_tokens')} "
        f"reasoning={details.get('reasoning_tokens', 'not reported')}",
        f"content ({len(content)} chars):\n{content}",
    ]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="One live chat_json call on a synthetic case.")
    parser.add_argument("--model", help="model ID (default: MODEL_FAST)")
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument(
        "--reasoning", choices=["default", "kwargs-off", "no-think"], default="default"
    )
    parser.add_argument("--label", help="name for the saved raw response file")
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    if not args.use_cache:
        settings = settings.model_copy(update={"cache_enabled": False})
    model = args.model or settings.model_fast
    label = args.label or f"{model.split('/')[-1]}_{args.reasoning}"

    raws: list[dict[str, Any]] = []
    client = LLMClient(settings, on_response=lambda r: raws.append(r.model_dump()))
    extra_body = THINKING_OFF_KWARGS if args.reasoning == "kwargs-off" else None
    messages = intake_messages(SAMPLE, no_think=args.reasoning == "no-think")

    print(
        f"Model: {model or '(MODEL_FAST not set)'}   reasoning: {args.reasoning}   "
        f"max_tokens: {args.max_tokens}"
    )
    print(f"Spend so far: ${client.ledger.total():.6f} / cap ${settings.max_spend_usd:.2f}\n")

    error: Exception | None = None
    case = None
    with track_usage() as records:
        try:
            case, _ = client.chat_json(
                messages,
                PatientCase,
                model=model,
                step="smoke:intake",
                max_tokens=args.max_tokens,
                extra_body=extra_body,
            )
        except (LLMError, SpendLimitError) as exc:
            error = exc

    for i, raw in enumerate(raws, 1):
        print(f"=== raw response {i} of {len(raws)} ===")
        print("\n".join(describe_shape(raw)))
        print()
    if raws:
        out = settings.cache_dir / "smoke" / f"{label}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": model, "reasoning": args.reasoning, "responses": raws}
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"(raw responses saved to {out})\n")

    if case is not None:
        print("=== parsed PatientCase: OK ===")
        print(case.model_dump_json(exclude_defaults=True))
    else:
        print(f"=== FAILED: {type(error).__name__}: {error}")
        if error is not None and error.__cause__ is not None:
            print(f"    cause: {error.__cause__}")

    s = summarize(records)
    print("\n=== usage ===")
    for r in s["steps"]:
        print(
            f"{r['step']:<20} in={r['prompt_tokens']} out={r['completion_tokens']} "
            f"reasoning={r['reasoning_tokens']} latency={r['latency_ms']:.0f}ms "
            f"cost=${r['cost_usd'] or 0:.6f}"
        )
    print(f"spent this run: ${s['spent_usd']:.6f}   ledger total: ${client.ledger.total():.6f}")
    return 0 if case is not None else 1


if __name__ == "__main__":
    sys.exit(main())
