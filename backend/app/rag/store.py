"""Cosine top-k search over the NumPy guideline index built by `rag/ingest.py`."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from backend.app.schemas import GuidelineChunk

DEFAULT_INDEX_DIR = Path("data/index")


class SearchHit(BaseModel):
    chunk: GuidelineChunk
    score: float


def _normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


class VectorStore:
    def __init__(self, vectors: np.ndarray, chunks: list[GuidelineChunk], model: str | None = None):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(chunks):
            raise ValueError(
                f"index mismatch: vectors {vectors.shape} vs {len(chunks)} chunks; "
                "rebuild the index"
            )
        self.vectors = _normalise(vectors)
        self.chunks = chunks
        self.model = model
        self._by_id = {c.id: c for c in chunks}

    @classmethod
    def load(cls, index_dir: Path = DEFAULT_INDEX_DIR) -> VectorStore:
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        chunks = [GuidelineChunk.model_validate(c) for c in meta["chunks"]]
        return cls(np.load(index_dir / "vectors.npy"), chunks, meta.get("model"))

    def __len__(self) -> int:
        return len(self.chunks)

    def get(self, chunk_id: str) -> GuidelineChunk | None:
        """Look up a chunk by ID (used to validate model citations)."""
        return self._by_id.get(chunk_id)

    def find(self, doc_id: str, phrase: str) -> GuidelineChunk | None:
        """First chunk of `doc_id` containing `phrase` (case- and whitespace-insensitive).

        Lets deterministic rules cite the exact guideline passage they are based on.
        """
        needle = " ".join(phrase.lower().split())
        for chunk in self.chunks:
            if chunk.doc_id == doc_id and needle in " ".join(chunk.text.lower().split()):
                return chunk
        return None

    def search(
        self,
        query: Iterable[float],
        k: int = 5,
        doc_ids: set[str] | None = None,
        boost: dict[str, float] | None = None,
    ) -> list[SearchHit]:
        """Cosine top-k. `boost` adds a small per-doc_id prior to the score."""
        if not self.chunks or k <= 0:
            return []
        q = _normalise(np.asarray(list(query), dtype=np.float32))
        if q.shape != (self.vectors.shape[1],):
            raise ValueError(f"query dim {q.shape} != index dim {self.vectors.shape[1]}")
        scores = self.vectors @ q
        if boost:
            scores = scores + np.array(
                [boost.get(c.doc_id, 0.0) for c in self.chunks], dtype=np.float32
            )
        if doc_ids is not None:
            mask = np.array([c.doc_id in doc_ids for c in self.chunks])
            scores = np.where(mask, scores, -np.inf)
        k = min(k, len(self.chunks))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [
            SearchHit(chunk=self.chunks[i], score=float(scores[i]))
            for i in top
            if np.isfinite(scores[i])
        ]

    def search_text(
        self,
        query: str,
        embed: Callable[[list[str]], list[list[float]]],
        k: int = 5,
        doc_ids: set[str] | None = None,
    ) -> list[SearchHit]:
        (vector,) = embed([query])
        return self.search(vector, k=k, doc_ids=doc_ids)
