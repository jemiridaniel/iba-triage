"""Fetch the guideline index from a private Hugging Face dataset before the app starts.

Used only on hosts that can't mount a private volume directly (e.g. Render). Nebius Compute
and local dev keep bind-mounting data/index instead (see docker-compose.yml) and never need
this: it no-ops whenever HF_INDEX_REPO is unset or the index is already on disk.

    uv run python -m backend.app.rag.ingest --dry-run   # chunk only, no API calls
    HF_INDEX_REPO=<user>/iba-guideline-index HF_TOKEN=<read-only token> \
        uv run python -m backend.app.rag.fetch_remote_index

The token only needs read access: the dataset holds no secrets, just the chunked and
embedded guideline text, kept out of the image because the source PDFs have no stated
licence (see the Dockerfile). Fails loudly on any error other than "already have it" or
"not configured" — an index that silently fails to load would run triage with retrieval
disabled, and no config mistake should make that happen quietly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    repo = os.environ.get("HF_INDEX_REPO")
    index_dir = Path(os.environ.get("INDEX_DIR", "data/index"))
    if not repo:
        print("HF_INDEX_REPO not set; skipping remote index fetch (expecting a local mount).")
        return 0
    if (index_dir / "meta.json").exists():
        print(f"{index_dir}/meta.json already present; skipping remote index fetch.")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed but HF_INDEX_REPO is set.", file=sys.stderr)
        return 1

    token = os.environ.get("HF_TOKEN")
    print(f"Fetching guideline index from hf://datasets/{repo} -> {index_dir}")
    index_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo, repo_type="dataset", token=token, local_dir=str(index_dir)
        )
    except Exception as exc:  # noqa: BLE001 - any failure here must stop the boot, loudly
        print(f"ERROR: failed to fetch {repo}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if not (index_dir / "meta.json").exists():
        print(f"ERROR: {repo} did not contain meta.json after download.", file=sys.stderr)
        return 1
    print("Guideline index fetched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
