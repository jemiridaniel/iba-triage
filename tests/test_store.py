"""Vector store cosine search. No network."""

import numpy as np
import pytest

from backend.app.rag.store import VectorStore
from backend.app.schemas import GuidelineChunk
from tests.helpers import FakeEmbedder


def chunk(i: int, doc_id: str, text: str) -> GuidelineChunk:
    return GuidelineChunk(
        id=f"{doc_id}:{i:04d}",
        doc_id=doc_id,
        title=doc_id,
        section=None,
        page=1,
        page_end=1,
        text=text,
        tokens=len(text.split()),
    )


CHUNKS = [
    chunk(0, "nmep", "severe malaria convulsions prostration refer urgently"),
    chunk(1, "ncdc-lassa", "lassa fever bleeding isolation barrier nursing"),
    chunk(2, "ncdc-cholera", "cholera watery diarrhoea dehydration oral rehydration"),
    chunk(3, "who-imci", "child unable to drink vomits everything convulsions"),
]


@pytest.fixture
def store() -> VectorStore:
    embed = FakeEmbedder()
    return VectorStore(np.array(embed([c.text for c in CHUNKS])), CHUNKS, model="fake")


def test_top_hit_is_most_similar(store: VectorStore) -> None:
    hits = store.search_text("lassa bleeding isolation", FakeEmbedder(), k=2)
    assert hits[0].chunk.doc_id == "ncdc-lassa"
    assert hits[0].score > hits[1].score


def test_scores_are_cosine(store: VectorStore) -> None:
    query = FakeEmbedder()([CHUNKS[2].text])[0]
    (hit,) = store.search(query, k=1)
    assert hit.chunk.id == "ncdc-cholera:0002"
    assert hit.score == pytest.approx(1.0, abs=1e-6)


def test_results_sorted_descending(store: VectorStore) -> None:
    hits = store.search_text("convulsions", FakeEmbedder(), k=4)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert {h.chunk.doc_id for h in hits[:2]} == {"nmep", "who-imci"}


def test_k_larger_than_index(store: VectorStore) -> None:
    assert len(store.search_text("fever", FakeEmbedder(), k=50)) == len(CHUNKS)


def test_doc_filter(store: VectorStore) -> None:
    hits = store.search_text("convulsions", FakeEmbedder(), k=3, doc_ids={"who-imci"})
    assert [h.chunk.doc_id for h in hits] == ["who-imci"]


def test_zero_query_vector_does_not_crash(store: VectorStore) -> None:
    hits = store.search(np.zeros(FakeEmbedder().dim), k=2)
    assert all(h.score == 0.0 for h in hits)


def test_dim_mismatch_raises(store: VectorStore) -> None:
    with pytest.raises(ValueError, match="dim"):
        store.search([1.0, 2.0], k=1)


def test_vector_chunk_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="mismatch"):
        VectorStore(np.zeros((2, 4)), CHUNKS)


def test_get_by_id(store: VectorStore) -> None:
    assert store.get("nmep:0000") == CHUNKS[0]
    assert store.get("nope:9999") is None


def test_empty_store() -> None:
    assert VectorStore(np.zeros((0, 4)), []).search([1, 0, 0, 0]) == []
