"""Build the guideline index: PDFs in data/raw/ -> heading-aware chunks -> embeddings.

    uv run python -m backend.app.rag.ingest --dry-run   # chunk only, no API calls
    uv run python -m backend.app.rag.ingest             # chunk + embed via MODEL_EMBED

Only documents listed in data/sources.yaml are ingested (that file is the licence register).
Output: data/index/vectors.npy (float32, L2-normalised) + data/index/meta.json (chunks with
doc_id / section / page for citations). data/index/ is gitignored.

Token counts are approximate (words * 4/3); no tokenizer is needed at this precision.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel
from pypdf import PdfReader

from backend.app.schemas import GuidelineChunk

logger = logging.getLogger("iba.ingest")

Embedder = Callable[[list[str]], list[list[float]]]

DEFAULT_SOURCES = Path("data/sources.yaml")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_INDEX_DIR = Path("data/index")
MIN_TOKENS = 400
MAX_TOKENS = 600


class Source(BaseModel):
    doc_id: str
    title: str
    file: str
    url: str | None = None
    url_confirmed: bool = False
    edition: str | None = None
    licence_note: str | None = None
    commit_text: bool = False


def load_sources(path: Path = DEFAULT_SOURCES) -> list[Source]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [Source.model_validate(s) for s in data.get("sources", [])]


# --- text extraction --------------------------------------------------------


def approx_tokens(text: str) -> int:
    return math.ceil(len(text.split()) * 4 / 3)


def extract_pages(pdf_path: Path) -> list[str]:
    reader = PdfReader(pdf_path)
    return [page.extract_text() or "" for page in reader.pages]


_PAGE_NUMBER = re.compile(r"^(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?$", re.IGNORECASE)
_NUMBERED_HEADING = re.compile(r"^(?:\d+(?:\.\d+)+\.?|\d+)\s+[A-Z(]")
_KEYWORD_HEADING = re.compile(r"^(?:chapter|section|annex|appendix|part)\s+[\w.]+", re.IGNORECASE)


def is_heading(line: str) -> bool:
    """Heuristic: numbered ("2.3 Severe malaria", "4 Referral"), keyword ("Annex 2"),
    or a short ALL-CAPS line. Numbered list items ("1. Give ...") are not headings."""
    s = line.strip()
    if not 3 <= len(s) <= 100 or s.endswith((".", ",", ";", ":")):
        return False
    if _KEYWORD_HEADING.match(s):
        return True
    words = s.split()
    if _NUMBERED_HEADING.match(s) and len(words) <= 14:
        return True
    letters = [c for c in s if c.isalpha()]
    return len(letters) >= 4 and s.upper() == s and len(words) <= 12


@dataclass
class Unit:
    text: str
    page: int  # 1-based
    heading: bool


def _clean_lines(pages: list[str]) -> list[list[str]]:
    per_page = []
    for text in pages:
        text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join hyphenated line breaks
        per_page.append([ln.strip() for ln in text.splitlines() if ln.strip()])
    # Drop running headers/footers: identical lines on more than half the pages.
    if len(per_page) >= 3:
        counts = Counter(ln for lines in per_page for ln in set(lines))
        repeated = {ln for ln, n in counts.items() if n > len(per_page) / 2}
        per_page = [[ln for ln in lines if ln not in repeated] for lines in per_page]
    return [[ln for ln in lines if not _PAGE_NUMBER.match(ln)] for lines in per_page]


def to_units(pages: list[str]) -> list[Unit]:
    return [
        Unit(text=ln, page=i, heading=is_heading(ln))
        for i, lines in enumerate(_clean_lines(pages), start=1)
        for ln in lines
    ]


# --- chunking ---------------------------------------------------------------


@dataclass
class _Piece:
    text: str
    page: int
    section: str | None
    tokens: int


def _split_long(text: str, max_tokens: int) -> list[str]:
    """Split one oversized line into <= max_tokens pieces: by sentence, then by words."""
    pieces: list[str] = []
    current: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        words = sentence.split()
        while approx_tokens(" ".join(words)) > max_tokens:
            cut = max(1, max_tokens * 3 // 4)
            if current:
                pieces.append(" ".join(current))
                current = []
            pieces.append(" ".join(words[:cut]))
            words = words[cut:]
        if approx_tokens(" ".join([*current, *words])) > max_tokens and current:
            pieces.append(" ".join(current))
            current = []
        current.extend(words)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_document(
    doc_id: str,
    title: str,
    pages: list[str],
    *,
    min_tokens: int = MIN_TOKENS,
    max_tokens: int = MAX_TOKENS,
) -> list[GuidelineChunk]:
    """Pack lines into chunks of <= max_tokens, preferring to break at headings.

    At a heading, the current chunk is closed if it has reached min_tokens; otherwise the
    small section is carried into the next chunk (so chunks can span several headings,
    recorded in `sections`). Only a document's final chunk, or one closed because the
    next line wouldn't fit, can fall below min_tokens.
    """
    chunks: list[GuidelineChunk] = []
    buf: list[_Piece] = []
    section: str | None = None

    def flush() -> None:
        if not buf:
            return
        sections = list(dict.fromkeys(p.section for p in buf if p.section))
        text = "\n".join(p.text for p in buf)
        chunks.append(
            GuidelineChunk(
                id=f"{doc_id}:{len(chunks):04d}",
                doc_id=doc_id,
                title=title,
                section=buf[0].section,
                sections=sections,
                page=buf[0].page,
                page_end=buf[-1].page,
                text=text,
                tokens=approx_tokens(text),
            )
        )
        buf.clear()

    for unit in to_units(pages):
        if unit.heading:
            if buf and sum(p.tokens for p in buf) >= min_tokens:
                flush()
            section = unit.text
        for text in _split_long(unit.text, max_tokens):
            piece = _Piece(text, unit.page, section, approx_tokens(text))
            if buf and sum(p.tokens for p in buf) + piece.tokens > max_tokens:
                flush()
            buf.append(piece)
    flush()
    return chunks


# --- embedding + index ------------------------------------------------------


def embedding_text(chunk: GuidelineChunk) -> str:
    header = chunk.title + (f" | {chunk.section}" if chunk.section else "")
    return f"{header}\n{chunk.text}"


def normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


def embed_chunks(chunks: list[GuidelineChunk], embed: Embedder, batch_size: int = 32) -> np.ndarray:
    texts = [embedding_text(c) for c in chunks]
    rows: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors = embed(batch)
        if len(vectors) != len(batch):
            raise ValueError(f"embedder returned {len(vectors)} vectors for {len(batch)} texts")
        rows.extend(vectors)
    return normalise(np.asarray(rows, dtype=np.float32))


def chunk_sources(sources: list[Source], raw_dir: Path) -> list[GuidelineChunk]:
    chunks: list[GuidelineChunk] = []
    listed = {s.file for s in sources}
    for stray in sorted(p.name for p in raw_dir.glob("*.pdf") if p.name not in listed):
        logger.warning("skipping %s: not listed in sources.yaml", stray)
    for source in sources:
        path = raw_dir / source.file
        if not path.exists():
            logger.warning("missing %s (%s); download it from %s", path, source.doc_id, source.url)
            continue
        doc_chunks = chunk_document(source.doc_id, source.title, extract_pages(path))
        logger.info("%s: %d chunks", source.doc_id, len(doc_chunks))
        chunks.extend(doc_chunks)
    return chunks


def write_index(
    out_dir: Path,
    vectors: np.ndarray,
    chunks: list[GuidelineChunk],
    sources: list[Source],
    model: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "vectors.npy", vectors)
    used = {c.doc_id for c in chunks}
    meta = {
        "model": model,
        "dim": int(vectors.shape[1]) if vectors.size else 0,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "sources": [
            s.model_dump(include={"doc_id", "title", "edition", "url", "url_confirmed"})
            for s in sources
            if s.doc_id in used
        ],
        "chunks": [c.model_dump() for c in chunks],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")


def build_index(
    sources: list[Source],
    raw_dir: Path,
    out_dir: Path,
    embed: Embedder,
    model: str,
    batch_size: int = 32,
) -> list[GuidelineChunk]:
    chunks = chunk_sources(sources, raw_dir)
    if not chunks:
        raise SystemExit(f"No listed PDFs found in {raw_dir}; nothing to index.")
    vectors = embed_chunks(chunks, embed, batch_size)
    write_index(out_dir, vectors, chunks, sources, model)
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the guideline vector index.")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true", help="chunk only; no embedding calls")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    sources = load_sources(args.sources)
    if args.dry_run:
        chunks = chunk_sources(sources, args.raw_dir)
        tokens = [c.tokens for c in chunks]
        print(f"{len(chunks)} chunks, ~{sum(tokens):,} tokens total")
        if tokens:
            median = sorted(tokens)[len(tokens) // 2]
            print(f"chunk tokens: min {min(tokens)}, median {median}, max {max(tokens)}")
        return 0

    from backend.app.config import get_settings
    from backend.app.llm.client import LLMClient, summarize, track_usage

    settings = get_settings()
    if not settings.model_embed:
        raise SystemExit("MODEL_EMBED is not set in .env")
    client = LLMClient(settings)
    with track_usage() as records:
        chunks = build_index(
            sources,
            args.raw_dir,
            args.out,
            lambda texts: client.embed(texts, model=settings.model_embed, step="ingest"),
            settings.model_embed,
            args.batch_size,
        )
    s = summarize(records)
    print(f"Indexed {len(chunks)} chunks -> {args.out}")
    print(
        f"Embedding calls: {s['calls']} ({s['cached_calls']} cached), spent ${s['spent_usd']:.6f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
