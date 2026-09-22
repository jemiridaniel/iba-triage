"""Dose table lookup and dose stripping."""

import pytest

from backend.app.rules.dosing import (
    DOSE_PLACEHOLDER,
    TABLE_VERIFIED,
    UNVERIFIED_WARNING,
    al_band,
    doses_for,
    strip_doses,
)
from backend.app.schemas import PatientCase, TriageLevel

TREAT = TriageLevel.TREAT_MONITOR


@pytest.mark.parametrize(
    ("kg", "tablets"),
    [(5, 1), (14.9, 1), (15, 2), (24.9, 2), (25, 3), (34.9, 3), (35, 4), (90, 4)],
)
def test_weight_band_boundaries(kg: float, tablets: int) -> None:
    assert al_band(kg).tablets_per_dose == tablets


def test_below_table() -> None:
    assert al_band(4.9) is None
    doses, notes = doses_for(PatientCase(rdt_result="positive", weight_kg=4), TREAT)
    assert doses == [] and "below the dose table" in notes[0]


def test_dose_for_confirmed_uncomplicated_malaria() -> None:
    (dose,), notes = doses_for(PatientCase(rdt_result="positive", weight_kg=20), TREAT)
    assert dose.regimen.startswith("2 tablets per dose")
    assert dose.weight_band == "15 to under 25 kg"
    assert dose.citation.doc_id == "nmep-malaria"
    assert dose.verified is TABLE_VERIFIED is False
    assert notes == [UNVERIFIED_WARNING]


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
