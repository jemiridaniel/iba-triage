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


# --- patient-fact loophole ------------------------------------------------------

from backend.app.graph import nodes  # noqa: E402
from backend.app.rules.grounding import patient_fact_supported  # noqa: E402
from backend.app.schemas import PatientCase  # noqa: E402

CASE = PatientCase(
    age_years=35,
    sex="male",
    fever_days=5,
    rdt_result="negative",
    state="Ondo",
    symptoms=["fever", "headache", "sore throat"],
    antimalarial_taken=True,
    antimalarial_no_response=True,
    raw_text="Adult man 35 years, fever 5 days, RDT negative, took coartem for 3 days but no "
    "improvement, headache and sore throat",
)


@pytest.mark.parametrize(
    "text",
    [
        "Fever for 5 days",
        "Fever for five days",  # number words
        "RDT negative",
        "Negative malaria RDT",  # from structured case
        "Headache and sore throat",
        "Took coartem without improvement",
        "35-year-old adult man",
        "Lives in Ondo State",
    ],
)
def test_real_patient_facts_are_accepted(text: str) -> None:
    assert patient_fact_supported(text, CASE)


@pytest.mark.parametrize(
    "text",
    [
        # Adversarial: clinical inferences smuggled in under the "patient" label.
        "Presentation is consistent with Lassa fever given residence in an endemic area",
        "Likely viral haemorrhagic fever",
        "Requires isolation and ribavirin",
        "No danger signs, so outpatient management is appropriate",
        "Bleeding from the gums",  # not in this patient's input at all
        "",
    ],
)
def test_clinical_claims_labelled_patient_are_rejected(text: str) -> None:
    assert not patient_fact_supported(text, CASE)


def test_relabelled_claim_without_quote_is_unsupported(tmp_path) -> None:
    from tests.pipeline_fakes import make_deps

    deps, _ = make_deps(tmp_path, {})
    ev = nodes._ground("patient", None, None, deps, "Consistent with Lassa fever", CASE)
    assert ev.status == "unsupported"
    ev = nodes._ground("patient", None, None, deps, "Fever for 5 days", CASE)
    assert ev.status == "patient"


def test_relabelled_claim_with_valid_quote_is_verified(tmp_path) -> None:
    from tests.pipeline_fakes import make_deps

    deps, _ = make_deps(tmp_path, {})
    quote = "Look for other causes of fever such as typhoid, pneumonia, urinary infection"
    ev = nodes._ground("patient", "fake-malaria:0002", quote, deps, "Consider typhoid", CASE)
    assert ev.status == "verified"
