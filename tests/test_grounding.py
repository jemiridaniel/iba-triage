"""Quote verification. Pure Python."""

import pytest

from backend.app.rules.grounding import normalise, verify_quote

CHUNK = (
    "Suspect Lassa fever in a patient with fever for 3 days or more that does not respond to\n"
    "antimalarials or antibiotics, especially with sore throat, vomiting, bleeding or contact "
    "with a confirmed case. Isolate a suspected Lassa fever patient immediately. Use gloves and "
    "standard infection prevention precautions, avoid contact with blood and body fluids."
)


def test_exact_quote_across_line_breaks_and_case() -> None:
    q = "fever for 3 days or more that does not respond to antimalarials or antibiotics"
    check = verify_quote(q.upper(), CHUNK)
    assert check.verified and check.reason == "exact"


def test_unicode_dashes_quotes_and_spacing_are_normalised() -> None:
    q = "“Isolate a suspected Lassa‑fever patient immediately . Use gloves and standard”"
    assert verify_quote(q.replace("Lassa‑fever", "Lassa fever"), CHUNK).verified


def test_fuzzy_match_tolerates_small_differences() -> None:
    q = (
        "Use gloves and standard infection prevention precaution, "
        "avoid contact with blood and body fluid"
    )
    check = verify_quote(q, CHUNK)
    assert check.verified and check.reason == "fuzzy" and check.score >= 0.9


def test_paraphrase_is_rejected() -> None:
    q = (
        "Patients with Lassa must be kept apart and staff should wear "
        "protective equipment at all times"
    )
    check = verify_quote(q, CHUNK)
    assert not check.verified and check.reason == "no_match"


@pytest.mark.parametrize(
    ("quote", "reason"),
    [
        (None, "missing"),
        ("", "missing"),
        ("Isolate a suspected Lassa fever patient", "too_short"),  # 6 words
        (" ".join(["word"] * 41), "too_long"),
    ],
)
def test_length_and_missing(quote, reason) -> None:
    assert verify_quote(quote, CHUNK).reason == reason


def test_quote_from_another_chunk_fails() -> None:
    q = "All cases of suspected malaria should have a parasitological test to confirm the diagnosis"
    assert not verify_quote(q, CHUNK).verified


def test_normalise() -> None:
    assert normalise("  “A–b ,  C” ") == "a-b, c"
