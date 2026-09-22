"""Build a tiny FAKE guideline index for development (no PDFs, no API calls).

    uv run python -m scripts.build_fake_index            # -> data/index_fake/
    uv run python -m backend.app.cli "..." --index data/index_fake

The passages are short placeholders paraphrasing general knowledge so retrieval and
citations can be exercised end to end. They are NOT guideline text and must never be shown
as guidance: every doc_id starts with "fake-" and every passage is labelled.
Embeddings use the deterministic hash embedder (model "fake-hash-embed").
"""

import sys
from pathlib import Path

from backend.app.rag.fake_embed import HASH_EMBED_MODEL, HashEmbedder
from backend.app.rag.ingest import Source, embed_chunks, write_index
from backend.app.schemas import GuidelineChunk

OUT = Path("data/index_fake")
LABEL = "[FAKE TEST PASSAGE - placeholder, not guideline text] "

PASSAGES: list[tuple[str, str, str, str]] = [
    (
        "fake-malaria",
        "Fake malaria guideline",
        "Severe malaria danger signs",
        "Refer urgently any febrile patient with convulsions, inability to drink or breastfeed, "
        "vomiting everything, lethargy or unconsciousness, prostration, respiratory distress, "
        "jaundice, dark urine, abnormal bleeding or severe pallor. Give pre-referral treatment "
        "per national guideline and refer to a hospital.",
    ),
    (
        "fake-malaria",
        "Fake malaria guideline",
        "Uncomplicated malaria",
        "Confirm malaria with an RDT before treatment. Treat RDT-positive uncomplicated malaria "
        "with an artemisinin-based combination therapy according to the weight-band table. Give "
        "paracetamol for fever, advise fluids, and review in 3 days or earlier if worse.",
    ),
    (
        "fake-malaria",
        "Fake malaria guideline",
        "RDT-negative fever",
        "If the RDT is negative, do not give antimalarials. Look for other causes of fever such as "
        "typhoid, pneumonia, urinary infection or viral haemorrhagic fever, and refer if the "
        "fever persists beyond 3 days or danger signs develop.",
    ),
    (
        "fake-lassa",
        "Fake Lassa fever guideline",
        "Suspected case definition",
        "Suspect Lassa fever in a patient with fever for 3 days or more that does not respond to "
        "antimalarials or antibiotics, especially with sore throat, vomiting, bleeding or "
        "contact with a confirmed case, or residence in an area with active transmission.",
    ),
    (
        "fake-lassa",
        "Fake Lassa fever guideline",
        "Isolation and referral",
        "Isolate a suspected Lassa fever patient immediately. Use gloves and standard infection "
        "prevention precautions, avoid contact with blood and body fluids, notify the LGA "
        "disease surveillance and notification officer, and refer to a Lassa treatment centre.",
    ),
    (
        "fake-cholera",
        "Fake cholera guideline",
        "Case definition and dehydration",
        "Suspect cholera with acute watery diarrhoea, with or without vomiting, in an area with "
        "an outbreak. Assess dehydration; give oral rehydration and refer severe dehydration.",
    ),
    (
        "fake-csm",
        "Fake meningitis guideline",
        "Suspected meningitis",
        "Suspect meningitis with sudden fever and neck stiffness, altered consciousness, or a "
        "bulging fontanelle in infants. Refer urgently.",
    ),
    (
        "fake-imci",
        "Fake IMCI chart",
        "General danger signs in children",
        "A child who is unable to drink or breastfeed, vomits everything, has had convulsions, or "
        "is lethargic or unconscious has a general danger sign and needs urgent referral.",
    ),
]


def main() -> int:
    chunks = [
        GuidelineChunk(
            id=f"{doc_id}:{i:04d}",
            doc_id=doc_id,
            title=title,
            section=section,
            sections=[section],
            page=1,
            page_end=1,
            text=LABEL + text,
            tokens=len(text.split()),
        )
        for i, (doc_id, title, section, text) in enumerate(PASSAGES)
    ]
    sources = [
        Source(doc_id=d, title=t, file="(none)", edition="fake test index")
        for d, t in dict.fromkeys((p[0], p[1]) for p in PASSAGES)
    ]
    vectors = embed_chunks(chunks, HashEmbedder())
    write_index(OUT, vectors, chunks, sources, HASH_EMBED_MODEL)
    print(f"Wrote {len(chunks)} fake passages to {OUT} (model {HASH_EMBED_MODEL})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
