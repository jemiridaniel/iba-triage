"""List NVIDIA model IDs served by Nebius Token Factory (GET /v1/models).

    uv run python -m scripts.list_models          # NVIDIA / Nemotron models only
    uv run python -m scripts.list_models --all    # everything in the catalog

Copy the IDs you want into MODEL_FAST / MODEL_MID / MODEL_REASON / MODEL_EMBED in .env.
This makes no inference calls and costs no credits.
"""

import argparse
import sys

from openai import OpenAI

from backend.app.config import get_settings


def is_nvidia(model_id: str, owned_by: str | None) -> bool:
    text = f"{model_id} {owned_by or ''}".lower()
    return "nvidia" in text or "nemotron" in text


def tag(model_id: str) -> str:
    mid = model_id.lower()
    if "embed" in mid:
        return "embedding"
    if "rerank" in mid:
        return "reranker"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="show every model, not just NVIDIA")
    args = parser.parse_args()

    settings = get_settings()
    if settings.nebius_api_key is None or not settings.nebius_api_key.get_secret_value():
        print("NEBIUS_API_KEY is not set. Copy .env.example to .env and add your key.")
        return 1

    client = OpenAI(
        api_key=settings.nebius_api_key.get_secret_value(),
        base_url=settings.nebius_base_url,
    )
    models = sorted(client.models.list(), key=lambda m: m.id)
    shown = [m for m in models if args.all or is_nvidia(m.id, getattr(m, "owned_by", None))]

    print(f"Token Factory at {settings.nebius_base_url}")
    for m in shown:
        extra = tag(m.id)
        print(f"  {m.id}" + (f"   [{extra}]" if extra else ""))
    label = "models" if args.all else "NVIDIA models"
    print(f"{len(shown)} {label} (of {len(models)} total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
