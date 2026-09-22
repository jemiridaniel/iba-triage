"""Probe which request options (if any) cap Nemotron reasoning on Token Factory.

    uv run python -m scripts.probe_reasoning_budget [--model ID]

Sends the same small reasoning-heavy prompt with different options and prints reasoning
tokens, completion tokens and finish_reason for each. Costs ~$0.02 on Super. Results feed
docs/FEEDBACK.md.
"""

import argparse
import sys

from openai import OpenAI

from backend.app.config import get_settings
from backend.app.llm.spend import SpendLedger, worst_case_cost

PROMPT = [
    {
        "role": "user",
        "content": "A clinic sees patients in 7 rooms. Each hour, rooms 1-7 each see a number of "
        "patients equal to (hour * room) mod 11, for hours 1 to 12. Compute the total number of "
        "patients seen across all rooms and hours, showing no working. Reply with JSON "
        '{"total": int} only.',
    },
]
BUDGET = 64
OPTIONS: list[tuple[str, dict]] = [
    ("default (no option)", {}),
    (
        "chat_template_kwargs.reasoning_budget",
        {"chat_template_kwargs": {"reasoning_budget": BUDGET}},
    ),
    ("chat_template_kwargs.thinking_budget", {"chat_template_kwargs": {"thinking_budget": BUDGET}}),
    (
        "chat_template_kwargs.reasoning_effort=low",
        {"chat_template_kwargs": {"reasoning_effort": "low"}},
    ),
    ("reasoning_effort=low (top-level)", {"reasoning_effort": "low"}),
    ("reasoning={effort: low}", {"reasoning": {"effort": "low"}}),
    ("max_thinking_tokens", {"max_thinking_tokens": BUDGET}),
    ("thinking={budget_tokens}", {"thinking": {"type": "enabled", "budget_tokens": BUDGET}}),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="default: MODEL_REASON")
    parser.add_argument("--max-tokens", type=int, default=4000)
    args = parser.parse_args()
    s = get_settings()
    model = args.model or s.model_reason
    price = s.model_prices[model]
    ledger = SpendLedger(s.spend_file, s.max_spend_usd)
    client = OpenAI(api_key=s.nebius_api_key.get_secret_value(), base_url=s.nebius_base_url)
    print(f"model {model}, reasoning budget requested {BUDGET}, max_tokens {args.max_tokens}")
    for label, extra in OPTIONS:
        ledger.check(model, "probe", worst_case_cost(price, 200, args.max_tokens))
        try:
            r = client.chat.completions.create(
                model=model,
                messages=PROMPT,
                max_tokens=args.max_tokens,
                temperature=0,
                extra_body=extra,
            )
        except Exception as exc:  # report and continue
            print(f"  {label:<44} ERROR {type(exc).__name__}: {str(exc)[:120]}")
            continue
        u = r.usage
        ledger.add(
            model, (u.prompt_tokens * price.input + u.completion_tokens * price.output) / 1e6
        )
        details = getattr(u, "completion_tokens_details", None)
        rt = getattr(details, "reasoning_tokens", None) if details else None
        print(
            f"  {label:<44} reasoning={rt!s:>5} completion={u.completion_tokens:>5} "
            f"finish={r.choices[0].finish_reason} "
            f"answer={(r.choices[0].message.content or '')[:40]!r}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
