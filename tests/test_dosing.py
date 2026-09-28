"""Dose table lookup (Table 4-4, all three AL strengths) and dose stripping."""

import pytest

from backend.app.rules.dosing import (
    AL_ABSORPTION_NOTE,
    AL_BANDS,
    AL_SCHEDULE,
    AL_SPLITTING_NOTE,
    AL_TABLETS_PER_DOSE,
    DOSE_PLACEHOLDER,
    TABLE_VERIFIED,
    UNVERIFIED_WARNING,
    al_band_index,
    doses_for,
    guideline_notes,
    strip_doses,
)
from backend.app.schemas import PatientCase, TriageLevel

TREAT = TriageLevel.TREAT_MONITOR

# Table 4-4 (NMEP 4th ed. 2020, p.10), transcribed exactly, one row per band per strength.
# (weight_kg, strength, expected tablets-per-dose or None for the table's "NA").
TABLE_4_4_CELLS = [
    (10, "20/120", 1), (10, "40/240", None), (10, "80/480", None),
    (20, "20/120", 2), (20, "40/240", 1), (20, "80/480", None),
    (30, "20/120", 3), (30, "40/240", None), (30, "80/480", None),
    (60, "20/120", 4), (60, "40/240", 2), (60, "80/480", 1),
]


@pytest.mark.parametrize(
    ("kg", "expected_idx"),
    [(5, 0), (14.9, 0), (15, 1), (24.9, 1), (25, 2), (34.9, 2), (35, 3), (90, 3)],
)
def test_weight_band_boundaries(kg: float, expected_idx: int) -> None:
    assert al_band_index(kg) == expected_idx


def test_below_table() -> None:
    assert al_band_index(4.9) is None
    doses, notes = doses_for(PatientCase(rdt_result="positive", weight_kg=4), TREAT)
    assert doses == [] and "below the dose table" in notes[0]


# --- every cell of Table 4-4, all three strengths -----------------------------------


@pytest.mark.parametrize(("kg", "strength", "expected"), TABLE_4_4_CELLS)
def test_tablets_per_dose_matches_table_4_4_exactly(kg, strength, expected) -> None:
    idx = al_band_index(kg)
    assert AL_TABLETS_PER_DOSE[strength][idx] == expected


@pytest.mark.parametrize(("kg", "strength", "expected"), TABLE_4_4_CELLS)
def test_doses_for_per_band_per_strength(kg: float, strength: str, expected: int | None) -> None:
    doses, notes = doses_for(PatientCase(rdt_result="positive", weight_kg=kg), TREAT)
    by_strength = {d.strength: d for d in doses}
    if expected is None:
        # NA: no dose recommendation for this strength, and no other strength's count
        # is ever substituted -- just an explicit note naming the unsuitable strength.
        assert strength not in by_strength
        assert any(f"{strength} mg tablets: not suitable" in n for n in notes)
    else:
        dose = by_strength[strength]
        word = "tablet" if expected == 1 else "tablets"
        assert dose.regimen.startswith(f"{expected} {word} per dose")
        assert dose.regimen.endswith(AL_SCHEDULE)
        assert dose.drug == f"Artemether-lumefantrine {strength} mg tablets"
        assert dose.strength == strength
        assert dose.citation.doc_id == "nmep-malaria"
        assert dose.citation.page == 10
        assert dose.verified is TABLE_VERIFIED is True


def test_5_to_15kg_only_20_120_is_suitable() -> None:
    doses, notes = doses_for(PatientCase(rdt_result="positive", weight_kg=10), TREAT)
    assert [d.strength for d in doses] == ["20/120"]
    assert "40/240 mg tablets: not suitable" in "\n".join(notes)
    assert "80/480 mg tablets: not suitable" in "\n".join(notes)
    # Never borrows a tablet count from a strength that has no entry for this band.
    assert "2 tablets" not in " ".join(notes) and "1 tablet " not in " ".join(notes)


def test_15_to_25kg_has_two_strengths() -> None:
    doses, notes = doses_for(PatientCase(rdt_result="positive", weight_kg=20), TREAT)
    by_strength = {d.strength: d.regimen for d in doses}
    assert by_strength["20/120"].startswith("2 tablets per dose")
    assert by_strength["40/240"].startswith("1 tablet per dose")
    assert "80/480" not in by_strength
    assert "80/480 mg tablets: not suitable" in "\n".join(notes)


def test_35kg_and_above_has_all_three_strengths() -> None:
    doses, _ = doses_for(PatientCase(rdt_result="positive", weight_kg=60), TREAT)
    by_strength = {d.strength: d.regimen for d in doses}
    assert by_strength["20/120"].startswith("4 tablets per dose")
    assert by_strength["40/240"].startswith("2 tablets per dose")
    assert by_strength["80/480"].startswith("1 tablet per dose")


def test_only_20_120_is_flagged_default() -> None:
    doses, _ = doses_for(PatientCase(rdt_result="positive", weight_kg=60), TREAT)
    assert {d.strength for d in doses if d.is_default} == {"20/120"}
    assert {d.strength for d in doses if not d.is_default} == {"40/240", "80/480"}


def test_table_is_verified_no_warning() -> None:
    # Verified against NMEP 4th ed. (May 2020) Table 4-4, p.10 by DJ, 2026-09-28.
    doses, notes = doses_for(PatientCase(rdt_result="positive", weight_kg=10), TREAT)
    assert UNVERIFIED_WARNING not in notes
    assert all(d.verified for d in doses)


def test_no_weight_no_dose() -> None:
    doses, notes = doses_for(PatientCase(rdt_result="positive"), TREAT)
    assert doses == [] and "Weigh the patient" in notes[0]


@pytest.mark.parametrize(
    ("case", "level"),
    [
        (PatientCase(rdt_result="negative", weight_kg=60), TREAT),
        (PatientCase(rdt_result=None, weight_kg=60), TREAT),
        (PatientCase(rdt_result="positive", weight_kg=60), TriageLevel.REFER_NOW),
        (PatientCase(rdt_result="positive", weight_kg=60), TriageLevel.REFER_24H),
    ],
)
def test_no_table_dose_unless_treating_confirmed_malaria(case: PatientCase, level) -> None:
    assert doses_for(case, level) == ([], [])


# --- schedule text: NMEP's own wording only, never WHO's under this citation --------


def test_schedule_matches_nmep_wording_not_who() -> None:
    assert AL_SCHEDULE == "twice daily for 3 days (6 doses)"
    # NMEP states no hour-level timing anywhere in the document (checked against the PDF):
    # guard against silently reintroducing WHO's 0h/8h convention under the NMEP citation.
    for marker in ("0 h", "8 h", "0h", "8h"):
        assert marker not in AL_SCHEDULE


# --- guideline notes: tablet splitting + fatty-meal advice, both cited --------------


def test_guideline_notes_are_cited_to_nmep_page_10() -> None:
    notes = guideline_notes()
    assert {n.text for n in notes} == {AL_SPLITTING_NOTE, AL_ABSORPTION_NOTE}
    for n in notes:
        assert n.citation.doc_id == "nmep-malaria"
        assert n.citation.page == 10
    # Each note has its own distinct citation ref/section (splitting is the table's own
    # caption; the fatty-meal note is the paragraph right after the table).
    assert len({n.citation.ref for n in notes}) == 2


def test_al_bands_count_matches_table_4_4() -> None:
    assert len(AL_BANDS) == 4
    for strength, tablets in AL_TABLETS_PER_DOSE.items():
        assert len(tablets) == 4, strength


@pytest.mark.parametrize(
    "text",
    [
        "Give artemether-lumefantrine 80/480 mg twice daily",
        "artesunate 2.4 mg/kg IV",
        "4 tablets per dose",
        "paracetamol 500mg",
        "ORS 1 sachet after each stool",
        "give 10-20 ml/kg fluid",
    ],
)
def test_strip_doses(text: str) -> None:
    cleaned, n = strip_doses(text)
    assert n >= 1
    assert DOSE_PLACEHOLDER in cleaned


@pytest.mark.parametrize(
    "text",
    ["fever for 5 days", "child 3 years old", "temperature 39.5", "RDT positive", "Week 37"],
)
def test_strip_doses_leaves_non_doses(text: str) -> None:
    assert strip_doses(text) == (text, 0)
