"""Tests for app.rag.must_contain. Pure functions, no model."""

from app.rag.must_contain import check_must_contain, normalize


def test_all_present():
    r = check_must_contain("Record at 8:00 AM and 4:00 PM.", ["8:00 AM", "4:00 pm"])
    assert r.passed and r.missing == ()


def test_missing_items_are_listed():
    r = check_must_contain("Move it to the quarantine bin.", ["quarantine bin", "batch number"])
    assert not r.passed and r.missing == ("batch number",) and r.found == ("quarantine bin",)


def test_dash_and_degree_spacing_are_normalized():
    assert check_must_contain("Keep at 2 \u00b0C \u2013 8 \u00b0C.", ["2\u00b0C-8\u00b0C"]).passed


def test_spaced_dash_in_a_phrase_matches():
    assert check_must_contain("Tag it DO NOT DISPENSE \u2013 QUARANTINE.", ["DO NOT DISPENSE - QUARANTINE"]).passed


def test_the_q04_range_matches_in_its_own_style():
    assert check_must_contain("Keep between 2\u00b0C\u20138\u00b0C.", ["2\u00b0C\u20138\u00b0C"]).passed


def test_whitespace_is_collapsed():
    assert normalize("a  \n b") == "a b"


def test_no_requirements_passes():
    assert check_must_contain("anything", []).passed


def test_substring_of_a_longer_number_is_still_a_match():
    # known limit: "7 days" matches inside "17 days"
    assert check_must_contain("within 17 days", ["7 days"]).passed


def test_a_range_written_with_words_does_not_match():
    # known limit: "2°C and 8°C" is not the required "2°C–8°C"
    assert not check_must_contain("between 2\u00b0C and 8\u00b0C", ["2\u00b0C\u20138\u00b0C"]).passed