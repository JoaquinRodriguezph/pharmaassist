"""Tests for app.rag.citations: pure functions, no model and no store."""

import json
from pathlib import Path

import pytest

from app.rag.citations import (
    CHUNK_ID,
    MALFORMED,
    NOT_IN_KB,
    NOT_RETRIEVED,
    NUMBERED,
    REFUSAL_SENTENCE,
    SECTION_ID_SHAPE,
    InvalidCitation,
    is_refusal,
    kb_section_ids,
    parse_citations,
    says_not_covered,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = REPO_ROOT / "docs" / "knowledge_base"

TEMP = "storage_sop.md#Temperature-Monitoring"
COLD = "storage_sop.md#Cold-Chain-And-Excursions"
FEFO = "storage_sop.md#Receiving-And-FEFO"
RETURNS = "returns_and_expiry_policy.md#Customer-Returns"
RETRIEVED = [TEMP, COLD]
KNOWN = {TEMP, COLD, FEFO, RETURNS}


def parse(text, retrieved=RETRIEVED, known=KNOWN):
    return parse_citations(text, retrieved, known)


# ---------------------------------------------------------------------------
# The section-id shape against the real knowledge base
# ---------------------------------------------------------------------------

def test_every_real_section_id_has_the_section_id_shape():
    ids = kb_section_ids(KB_DIR)

    assert len(ids) == 13  # 14 chunks, but Customer-Returns is split in two and has one section id
    for section_id in ids:
        assert SECTION_ID_SHAPE.fullmatch(section_id), section_id


def test_every_expected_source_in_the_test_set_is_a_real_section_id():
    test_set = json.loads((REPO_ROOT / "eval" / "test_set.json").read_text(encoding="utf-8"))
    ids = kb_section_ids(KB_DIR)

    for question in test_set:
        for source in question.get("expected_sources") or []:
            assert source in ids, (question["id"], source)


def test_the_readme_is_not_a_source():
    assert not any(section_id.lower().startswith("readme.md") for section_id in kb_section_ids(KB_DIR))


# ---------------------------------------------------------------------------
# Valid citations
# ---------------------------------------------------------------------------

def test_a_retrieved_section_id_in_brackets_is_valid():
    result = parse(f"Record it twice daily [{TEMP}].")

    assert result.valid == (TEMP,) and result.invalid == ()


def test_adjacent_brackets_are_two_citations_in_order_of_first_use():
    result = parse(f"Both apply [{COLD}][{TEMP}]. Again [{TEMP}].")

    assert result.valid == (COLD, TEMP)


@pytest.mark.parametrize("inside", [f"{TEMP}, {COLD}", f"{TEMP};{COLD}", f" {TEMP} ; {COLD} "])
def test_commas_and_semicolons_separate_citations_inside_one_bracket(inside):
    assert parse(f"Both [{inside}].").valid == (TEMP, COLD)


def test_a_citation_used_as_markdown_link_text_still_counts_and_the_url_is_ignored():
    result = parse(f"See [{TEMP}](https://example.com/storage_sop.md#Other).")

    assert result.valid == (TEMP,) and result.invalid == ()


def test_a_citation_inside_double_brackets_counts_once():
    assert parse(f"See [[{TEMP}]].").valid == (TEMP,)


# ---------------------------------------------------------------------------
# Invalid citations and their reasons
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "piece, reason",
    [
        pytest.param(FEFO, NOT_RETRIEVED, id="real-but-not-retrieved"),
        pytest.param("example.md#Example-Section", NOT_IN_KB, id="copied-example"),
        pytest.param("storage_sop.md#temperature-monitoring", NOT_IN_KB, id="wrong-case"),
        pytest.param(f"{TEMP}::1", CHUNK_ID, id="chunk-id"),
        pytest.param("1", NUMBERED, id="number"),
        pytest.param("storage_sop.md", MALFORMED, id="file-without-section"),
        pytest.param("storage_sop.md#Temperature Monitoring", MALFORMED, id="space-in-fragment"),
        pytest.param(f"see {TEMP}", MALFORMED, id="extra-words"),
    ],
)
def test_invalid_citations_get_a_reason(piece, reason):
    result = parse(f"Some claim [{piece}].")

    assert result.valid == ()
    assert result.invalid == (InvalidCitation(piece, reason),)


def test_valid_and_invalid_citations_are_kept_apart_and_deduplicated():
    result = parse(f"A [{TEMP}]. B [3]. C [{FEFO}]. D [3]. E [{TEMP}].")

    assert result.valid == (TEMP,)
    assert result.invalid == (InvalidCitation("3", NUMBERED), InvalidCitation(FEFO, NOT_RETRIEVED))


# ---------------------------------------------------------------------------
# What is not a citation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "No brackets at all.",
        "He said [sic] it was fine.",
        "Tick the box [ ] and the other [x].",
        "An empty pair [] here.",
        "A link [the SOP](https://example.com/storage_sop.md#Temperature-Monitoring).",
        f"Across lines [{TEMP}\n] is not one bracket.",
    ],
)
def test_text_that_is_not_a_citation_is_ignored(text):
    assert parse(text).valid == () and parse(text).invalid == ()


def test_the_text_is_not_changed():
    text = f"Claim [{FEFO}]."
    parse(text)

    assert text == f"Claim [{FEFO}]."


# ---------------------------------------------------------------------------
# The fixed sentence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        REFUSAL_SENTENCE,
        REFUSAL_SENTENCE.rstrip("."),
        "  the provided SOURCES do not answer   this question.\n",
        "The provided sources\ndo not answer this question.",
    ],
)
def test_the_fixed_sentence_alone_is_a_refusal(text):
    assert is_refusal(text) and says_not_covered(text)


def test_a_partial_answer_says_not_covered_but_is_not_a_refusal():
    text = f"Readings are taken at 8:00 AM and 4:00 PM [{TEMP}]. {REFUSAL_SENTENCE}"

    assert says_not_covered(text) and not is_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "The sources provided don't say.",
        "I don't know.",
        "The provided sources do not answer that question.",
        f'"{REFUSAL_SENTENCE}"',  # in quotes: not exactly the sentence, so not a pure refusal
    ],
)
def test_a_reworded_or_quoted_refusal_is_not_a_pure_refusal(text):
    assert not is_refusal(text)


def test_a_reworded_refusal_is_not_detected_at_all():
    assert not says_not_covered("The sources provided don't say.")


def test_the_quoted_sentence_still_counts_as_saying_not_covered():
    assert says_not_covered(f'"{REFUSAL_SENTENCE}"')