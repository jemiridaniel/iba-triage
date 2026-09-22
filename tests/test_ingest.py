"""Ingest: PDF extraction, heading-aware chunking, embedding, index files. No network."""

import json
from pathlib import Path

import numpy as np
import pytest

from backend.app.rag.ingest import (
    Source,
    approx_tokens,
    build_index,
    chunk_document,
    embed_chunks,
    extract_pages,
    is_heading,
    load_sources,
)
from backend.app.rag.store import VectorStore
from tests.helpers import FakeEmbedder, make_pdf

REPO = Path(__file__).resolve().parents[1]


def filler(n_words: int, topic: str) -> str:
    return " ".join(f"{topic}{i % 7}" for i in range(n_words))


def lines_of(text: str, per_line: int = 12) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + per_line]) for i in range(0, len(words), per_line)]


@pytest.fixture
def guideline_pdf(tmp_path: Path) -> Path:
    page1 = [
        "NATIONAL MALARIA GUIDELINE",
        "1 Introduction",
        "This guideline covers diagnosis and treatment of malaria in Nigeria.",
        "2.1 Severe malaria",
        "Refer any patient with convulsions, prostration or inability to drink.",
        "Give pre-referral treatment and refer urgently.",
    ]
    page2 = [
        "2.2 Uncomplicated malaria",
        "Confirm with RDT before treatment. Treat RDT positive cases with ACT.",
        "3",  # page number, dropped
    ]
    return make_pdf(tmp_path / "doc.pdf", [page1, page2])


# --- sources ----------------------------------------------------------------


def test_repo_sources_yaml_register() -> None:
    sources = {s.doc_id: s for s in load_sources(REPO / "data" / "sources.yaml")}
    assert {
        "nmep-malaria",
        "who-malaria",
        "who-imci",
        "ncdc-lassa",
        "ncdc-vhf-ipc",
        "ncdc-lassa-advisory-2026",
        "ncdc-csm",
        "ncdc-csm-quickref",
        "ncdc-cholera",
        "ncdc-case-definitions",
    } == set(sources)
    for s in sources.values():
        assert s.edition and s.licence and s.licence_note
        assert s.commit_text is False
        assert s.url_confirmed is (s.doc_id != "nmep-malaria")
        if s.doc_id.startswith("who-"):
            assert s.licence == "CC BY-NC-SA 3.0 IGO"


# --- extraction + headings --------------------------------------------------


def test_extract_pages_reads_generated_pdf(guideline_pdf: Path) -> None:
    pages = extract_pages(guideline_pdf)
    assert len(pages) == 2
    assert "2.1 Severe malaria" in pages[0]
    assert "Confirm with RDT" in pages[1]


@pytest.mark.parametrize(
    "line",
    ["2.1 Severe malaria", "4 Referral", "3.2.1 Pre-referral treatment", "Annex 2", "DANGER SIGNS"],
)
def test_headings(line: str) -> None:
    assert is_heading(line)


@pytest.mark.parametrize(
    "line",
    [
        "1. Give ACT for three days.",
        "Refer any patient with convulsions.",
        "12",
        "The patient should be referred to hospital without delay and with a note",
        "OK",
    ],
)
def test_not_headings(line: str) -> None:
    assert not is_heading(line)


# --- chunking ---------------------------------------------------------------


def test_small_doc_keeps_sections_and_pages(guideline_pdf: Path) -> None:
    (chunk,) = chunk_document("nmep", "NMEP", extract_pages(guideline_pdf))
    assert chunk.id == "nmep:0000"
    assert chunk.page == 1
    assert chunk.page_end == 2
    assert chunk.sections == [
        "NATIONAL MALARIA GUIDELINE",
        "1 Introduction",
        "2.1 Severe malaria",
        "2.2 Uncomplicated malaria",
    ]
    assert "\n3\n" not in f"\n{chunk.text}\n"  # bare page number removed


def test_long_section_chunks_within_token_bounds() -> None:
    body = lines_of(filler(3000, "word"))  # ~4000 tokens, one section
    pages = ["\n".join(["2.1 Severe malaria", *body[:130]]), "\n".join(body[130:])]
    chunks = chunk_document("d", "T", pages)
    assert len(chunks) >= 6
    assert all(c.tokens <= 600 for c in chunks)
    assert all(c.tokens >= 400 for c in chunks[:-1])
    assert all(c.section == "2.1 Severe malaria" for c in chunks)
    assert chunks[0].page == 1
    assert chunks[-1].page_end == 2


def test_breaks_at_heading_once_min_reached() -> None:
    sec_a = lines_of(filler(330, "alpha"))  # ~440 tokens: >= min
    sec_b = lines_of(filler(150, "beta"))
    pages = ["\n".join(["1 Alpha", *sec_a, "2 Beta", *sec_b])]
    first, second = chunk_document("d", "T", pages)
    assert first.sections == ["1 Alpha"]
    assert second.section == "2 Beta"
    assert second.text.startswith("2 Beta")


def test_small_sections_are_merged_until_min() -> None:
    pages = ["\n".join(["1 One", filler(40, "a"), "2 Two", filler(40, "b"), "3 Three"])]
    (chunk,) = chunk_document("d", "T", pages)
    assert chunk.section == "1 One"
    assert chunk.sections == ["1 One", "2 Two", "3 Three"]


def test_oversized_single_line_is_split() -> None:
    chunks = chunk_document("d", "T", [filler(2000, "long")])
    assert all(c.tokens <= 600 for c in chunks)
    assert sum(len(c.text.split()) for c in chunks) == 2000


def test_running_header_is_dropped() -> None:
    pages = [f"MINISTRY OF HEALTH NIGERIA\nPage {i} content line" for i in range(1, 5)]
    chunks = chunk_document("d", "T", pages)
    assert all("MINISTRY OF HEALTH" not in c.text for c in chunks)


def test_approx_tokens() -> None:
    assert approx_tokens("one two three") == 4


# --- embedding + index ------------------------------------------------------


def test_embed_chunks_batches_and_normalises() -> None:
    chunks = chunk_document("d", "T", [filler(2500, "x")])
    embed = FakeEmbedder()
    vectors = embed_chunks(chunks, embed, batch_size=2)
    assert vectors.shape == (len(chunks), embed.dim)
    assert all(len(b) <= 2 for b in embed.calls)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_embed_chunks_rejects_wrong_vector_count() -> None:
    chunks = chunk_document("d", "T", ["hello world"])
    with pytest.raises(ValueError, match="vectors"):
        embed_chunks(chunks, lambda texts: [], batch_size=4)


def test_build_index_end_to_end(tmp_path: Path, guideline_pdf: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    guideline_pdf.rename(raw / "nmep.pdf")
    make_pdf(raw / "unlisted.pdf", [["ignored"]])
    sources = [
        Source(doc_id="nmep", title="NMEP", file="nmep.pdf", edition="test"),
        Source(doc_id="missing", title="Missing", file="missing.pdf"),
    ]
    out = tmp_path / "index"
    embed = FakeEmbedder()

    chunks = build_index(sources, raw, out, embed, model="fake-embed")

    assert {c.doc_id for c in chunks} == {"nmep"}
    meta = json.loads((out / "meta.json").read_text())
    assert meta["model"] == "fake-embed"
    assert meta["dim"] == embed.dim
    assert [s["doc_id"] for s in meta["sources"]] == ["nmep"]
    assert np.load(out / "vectors.npy").shape == (len(chunks), embed.dim)

    store = VectorStore.load(out)
    (hit,) = store.search_text("convulsions prostration refer", embed, k=1)
    assert hit.chunk.doc_id == "nmep"
    assert hit.chunk.page == 1


def test_build_index_with_no_pdfs_exits(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        build_index(
            [Source(doc_id="x", title="X", file="x.pdf")], tmp_path, tmp_path, FakeEmbedder(), "m"
        )


def test_table_of_contents_lines_are_dropped() -> None:
    pages = [
        "\n".join(
            [
                "1.1.2 Suspected case ........................................ 7",
                "3.4 Treatment …………… 19",
                "1.1.2 Suspected case",
                "Patient with fever for 3-21 days.",
            ]
        )
    ]
    (chunk,) = chunk_document("d", "T", pages)
    assert "......" not in chunk.text and "……" not in chunk.text
    assert chunk.sections == ["1.1.2 Suspected case"]
