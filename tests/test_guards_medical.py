"""is_personal_medical against hand-written cases and every test-set question."""

import json
from pathlib import Path

import pytest

from app.guards import is_personal_medical

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("text", [
    "my 4yo daughter has had a 38.8C fever since last night and weighs 16kg, how many mL should i give her?",
    "Can I take ibuprofen together with my blood pressure medicine (losartan)?",
    "a customer showed me a rash after cefalexin, should I tell them to switch to azithromycin instead?",
    "my 70-year-old mother takes warfarin, is aspirin ok?",
    "how much paracetamol should I give a 3 year old?",
])
def test_personal_medical_questions_are_flagged(text):
    assert is_personal_medical(text)


@pytest.mark.parametrize("text", [
    "which patients should NOT take ibuprofen according to our leaflet?",
    "max number of paracetamol 500mg tabs an adult can take in 24 hrs according to the leaflet?",
    "can we refund an unopened item with a receipt?",
    "what does the paracetamol leaflet say about alcohol?",
])
def test_general_questions_are_not_flagged(text):
    assert not is_personal_medical(text)


def test_only_the_personal_medical_test_set_questions_are_flagged():
    test_set = json.loads((REPO_ROOT / "eval" / "test_set.json").read_text(encoding="utf-8"))
    flagged = {q["id"] for q in test_set if is_personal_medical(q["question"])}
    assert flagged == {"Q19", "Q20", "Q21"}