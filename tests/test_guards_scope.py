"""The off-topic input patterns."""

import pytest

from app.guards import check_input


@pytest.mark.parametrize("text, name", [
    ("can you write a python script to scrape Mercury Drug's website so we can compare their prices?", "write_code"),
    ("scrape the competitor's price list", "scraping"),
    ("how do I hack the POS terminal?", "hacking"),
    ("generate a bot that posts our promos", "write_code"),
])
def test_off_topic_requests_are_caught(text, name):
    assert check_input(text) == name


@pytest.mark.parametrize("text", [
    "write down the excursion in the temperature log, what else does the SOP say?",
    "which products are top sellers at Manila Main Branch?",
    "do we accept GCash or Maya QR payments when the main POS terminal is offline?",
])
def test_pharmacy_questions_are_not_off_topic(text):
    assert check_input(text) is None