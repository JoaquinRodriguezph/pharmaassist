"""Tests for app.rag.prompting: grouping by section, part order, and the prompt text."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.rag.citations import REFUSAL_SENTENCE, SECTION_ID_SHAPE, kb_section_ids
from app.rag.prompting import (
    EXAMPLE_CITATION,
    EXAMPLE_PAIR,
    NO_SOURCES,
    SYSTEM_PROMPT,
    SourceBlock,
    build_user_message,
    format_block,
    group_sources,
)

KB_DIR = Path(__file__).resolve().parent.parent / "docs" / "knowledge_base"

RETURNS = "returns_and_expiry_policy.md#Customer-Returns"
EXPIRED = "returns_and_expiry_policy.md#Expired-Medications"
TEMP = "storage_sop.md#Temperature-Monitoring"


def chunk(section_id, part, text, section=None):
    return SimpleNamespace(
        id=f"{section_id}::{part}",
        text=text,
        section_id=section_id,
        section=section or section_id.split("#")[1].replace("-", " "),
    )


# ---------------------------------------------------------------------------
# Grouping and order
# ---------------------------------------------------------------------------

def test_parts_of_one_section_become_one_block_in_part_order():
    retrieved = [
        chunk(RETURNS, 2, "Part two."),  # rank 1
        chunk(TEMP, 1, "Temperature."),  # rank 2
        chunk(RETURNS, 1, "Part one."),  # rank 3
    ]

    blocks = group_sources(retrieved)

    assert [b.section_id for b in blocks] == [RETURNS, TEMP]
    assert blocks[0].chunk_ids == (f"{RETURNS}::1", f"{RETURNS}::2")
    assert blocks[0].text == "Part one.\n\nPart two."


def test_blocks_follow_the_rank_of_each_sections_best_chunk():
    retrieved = [chunk(TEMP, 1, "a"), chunk(EXPIRED, 1, "b"), chunk(RETURNS, 1, "c"), chunk(EXPIRED, 2, "d")]

    assert [b.section_id for b in group_sources(retrieved)] == [TEMP, EXPIRED, RETURNS]


def test_five_chunks_can_give_fewer_blocks():
    retrieved = [chunk(RETURNS, 1, "a"), chunk(RETURNS, 2, "b"), chunk(TEMP, 1, "c"), chunk(EXPIRED, 1, "d"),
                 chunk(EXPIRED, 2, "e")]

    assert len(group_sources(retrieved)) == 3


def test_only_retrieved_parts_are_shown_so_part_two_can_appear_alone():
    (block,) = group_sources([chunk(RETURNS, 2, "Only part two.")])

    assert block.chunk_ids == (f"{RETURNS}::2",) and block.text == "Only part two."


def test_a_chunk_retrieved_twice_is_used_once():
    (block,) = group_sources([chunk(TEMP, 1, "Same."), chunk(TEMP, 1, "Same.")])

    assert block.chunk_ids == (f"{TEMP}::1",) and block.text == "Same."


def test_part_numbers_sort_as_numbers_not_text():
    retrieved = [chunk(RETURNS, 10, "ten"), chunk(RETURNS, 2, "two")]

    assert group_sources(retrieved)[0].text == "two\n\nten"


def test_a_chunk_id_without_a_part_number_is_an_error():
    bad = SimpleNamespace(id=RETURNS, text="x", section_id=RETURNS, section="Customer Returns")

    with pytest.raises(ValueError, match="part number"):
        group_sources([bad])


def test_no_chunks_give_no_blocks():
    assert group_sources([]) == []


# ---------------------------------------------------------------------------
# The text the model sees
# ---------------------------------------------------------------------------

def test_a_block_starts_with_the_exact_citation_and_the_heading():
    block = SourceBlock(TEMP, "Temperature Monitoring", (f"{TEMP}::1",), "Check twice daily.")

    assert format_block(block) == (
        "Source [storage_sop.md#Temperature-Monitoring] (Temperature Monitoring)\n"
        "Check twice daily.\n"
        "---"
    )


def test_the_user_message_has_the_sources_then_the_question():
    blocks = group_sources([chunk(TEMP, 1, "Check twice daily.", "Temperature Monitoring"),
                            chunk(RETURNS, 1, "Seven days.", "Customer Returns")])

    message = build_user_message("  When do we check?  ", blocks)

    assert message == (
        "Sources:\n\n"
        "Source [storage_sop.md#Temperature-Monitoring] (Temperature Monitoring)\nCheck twice daily.\n---\n\n"
        "Source [returns_and_expiry_policy.md#Customer-Returns] (Customer Returns)\nSeven days.\n---\n\n"
        "Question: When do we check?"
    )


def test_no_sources_are_said_plainly():
    assert build_user_message("Anything?", []) == f"{NO_SOURCES}\n\nQuestion: Anything?"


def test_an_empty_question_is_an_error():
    with pytest.raises(ValueError, match="question"):
        build_user_message("   ", [])


def test_no_retrieval_score_reaches_the_prompt():
    scored = SimpleNamespace(id=f"{TEMP}::1", text="Check.", section_id=TEMP, section="Temperature Monitoring",
                             score=0.6123)

    assert "0.6123" not in build_user_message("Q?", group_sources([scored]))


# ---------------------------------------------------------------------------
# The system prompt
# ---------------------------------------------------------------------------

def test_the_system_prompt_contains_the_exact_fixed_sentence_and_the_rules():
    assert REFUSAL_SENTENCE in SYSTEM_PROMPT
    for rule in ("1. ", "2. ", "3. ", "4. ", "5. ", "6. "):
        assert f"\n{rule}" in SYSTEM_PROMPT


def test_the_example_citations_are_well_formed_but_cannot_be_real_sections():
    real = kb_section_ids(KB_DIR)
    examples = [EXAMPLE_CITATION[1:-1]] + EXAMPLE_PAIR[1:-1].split("][")

    for example in examples:
        assert example in SYSTEM_PROMPT
        assert SECTION_ID_SHAPE.fullmatch(example)  # the model sees the right shape
        assert example not in real  # a copied example is caught as "not in the knowledge base"


def test_no_real_section_id_appears_in_the_system_prompt():
    assert not [section_id for section_id in kb_section_ids(KB_DIR) if section_id in SYSTEM_PROMPT]