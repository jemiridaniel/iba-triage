"""Show or reset the live-spend ledger (CACHE_DIR/spend.json).

uv run python -m scripts.spend           # total, cap, remaining, per-model breakdown
uv run python -m scripts.spend --reset   # zero the ledger (prints the old totals first)
"""

import argparse
import sys

from backend.app.config import get_settings
from backend.app.llm.spend import SpendLedger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reset", action="store_true", help="zero the spend ledger")
    args = parser.parse_args()

    settings = get_settings()
    ledger = SpendLedger(settings.spend_file, settings.max_spend_usd)

    if args.reset:
        old = ledger.reset()
        print(f"Reset {ledger.path}. Previous total: ${old['total_usd'] or 0:.6f}")
        return 0

    data = ledger.read()
    total = float(data["total_usd"])
    print(f"Ledger:    {ledger.path}")
    print(f"Spent:     ${total:.6f} over {data['calls']} live calls")
    print(f"Cap:       ${ledger.cap_usd:.2f}  (MAX_SPEND_USD)")
    print(f"Remaining: ${max(ledger.cap_usd - total, 0):.6f}")
    for model, usd in sorted(data["by_model"].items(), key=lambda kv: -kv[1]):
        print(f"  {model:<48} ${usd:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
