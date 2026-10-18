"""Tests for app.rag.chunking and scripts/chunk_stats.py, using a tiny temporary KB."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from app.rag.chunking import Chunk, chunk_kb, split_text, stray_preamble_lines
from app.rag.kb import is_kb_document, kb_documents, slugify_heading

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNK_STATS = REPO_ROOT / "scripts" / "chunk_stats.py"

BANNER = "> FICTIONAL document for software testing only. Not medical or pharmacy guidance."
SHORT_TEXT = "One short rule applies here."
LONG_BODY = (
    "Alpha rule applies to all branches every day. Beta rule applies after closing time.\n\n"
    "1. Gamma step comes first and takes a minute.\n"
    "2. Delta step comes second and takes two minutes.\n\n"
    "Epsilon note closes the section."
)


def write(kb, name, text):
    (kb / name).write_text(text, encoding="utf-8")


def policy_text(long_body=LONG_BODY):
    return (
        f"{BANNER}\n\n# Policy Title\n\nDocument ID: P-1.\n\n"
        f"## Short Section\n{SHORT_TEXT}\n\n"
        f"## Long Section\n{long_body}\n"
    )


def make_kb(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    write(kb, "policy.md", policy_text())
    write(kb, "README.md", "# KB README\n\n## Heading Rule\nREADME-ONLY-SENTENCE is documentation.\n")
    write(kb, "_draft.md", "# Draft\n\n## Draft Section\nDRAFT-ONLY-SENTENCE.\n")
    write(kb, ".hidden.md", "# Hidden\n\n## Hidden Section\nHIDDEN-ONLY-SENTENCE.\n")
    write(kb, "notes.txt", "## Not Markdown\nTXT-ONLY-SENTENCE.\n")
    return kb


def squash(text):
    """Remove all whitespace: for checking that no characters were lost or added."""
    return re.sub(r"\s+", "", text)


# ---------------------------------------------------------------------------
# Which files are documents (README is not indexed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, expected",
    [
        pytest.param("policy.md", True, id="plain-document"),
        pytest.param("Policy.MD", True, id="extension-case-ignored"),
        pytest.param("README.md", False, id="readme"),
        pytest.param("readme.md", False, id="readme-lowercase"),
        pytest.param("ReadMe.MD", False, id="readme-mixed-case"),
        pytest.param("_draft.md", False, id="underscore-prefix"),
        pytest.param(".hidden.md", False, id="dot-prefix"),
        pytest.param("notes.txt", False, id="not-markdown"),
    ],
)
def test_is_kb_document(name, expected):
    assert is_kb_document(Path(name)) is expected


def test_kb_documents_lists_only_documents_sorted(tmp_path):
    kb = make_kb(tmp_path)
    write(kb, "another.md", "# A\n\n## S\ntext\n")
    (kb / "sub").mkdir()
    write(kb / "sub", "nested.md", "# N\n\n## S\ntext\n")

    assert [p.name for p in kb_documents(kb)] == ["another.md", "policy.md"]


def test_readme_and_other_non_documents_are_not_indexed(tmp_path):
    kb = make_kb(tmp_path)

    chunks = chunk_kb(kb)

    assert {c.source for c in chunks} == {"policy.md"}
    all_text = " ".join(c.text for c in chunks) + " ".join(c.embed_text() for c in chunks)
    for marker in ("README-ONLY", "DRAFT-ONLY", "HIDDEN-ONLY", "TXT-ONLY"):
        assert marker not in all_text


# ---------------------------------------------------------------------------
# One chunk per short section, with the right metadata
# ---------------------------------------------------------------------------

def test_short_section_is_one_chunk_with_the_right_metadata(tmp_path):
    kb = make_kb(tmp_path)

    short = [c for c in chunk_kb(kb) if c.section == "Short Section"]

    assert short == [
        Chunk(
            id="policy.md#Short-Section::1",
            text=SHORT_TEXT,
            source="policy.md",
            section="Short Section",
            section_id="policy.md#Short-Section",
            doc_title="Policy Title",
            part=1,
            n_parts=1,
        )
    ]


def test_a_section_that_fits_is_not_split(tmp_path):
    kb = make_kb(tmp_path)

    long_chunks = [c for c in chunk_kb(kb, max_chars=len(LONG_BODY)) if c.section == "Long Section"]

    assert len(long_chunks) == 1 and long_chunks[0].text == LONG_BODY


def test_a_heading_slug_equals_its_section_id_fragment(tmp_path):
    kb = make_kb(tmp_path)
    write(kb, "cold.md", "# Cold\n\n## Cold Chain And Excursions\ntext\n\n## Receiving And FEFO\ntext\n")

    chunks = chunk_kb(kb)

    assert {c.section_id for c in chunks if c.source == "cold.md"} == {
        "cold.md#Cold-Chain-And-Excursions",
        "cold.md#Receiving-And-FEFO",
    }
    for chunk in chunks:
        assert chunk.section_id == f"{chunk.source}#{slugify_heading(chunk.section)}"
        assert chunk.id.startswith(chunk.section_id + "::")


# ---------------------------------------------------------------------------
# Text before the first ## heading: not indexed (explicit rule)
# ---------------------------------------------------------------------------

def test_preamble_is_not_indexed_but_the_title_is_kept_for_the_prefix(tmp_path):
    kb = make_kb(tmp_path)

    chunks = chunk_kb(kb, max_chars=40)

    for chunk in chunks:
        assert "FICTIONAL" not in chunk.text
        assert "Policy Title" not in chunk.text
        assert "Document ID" not in chunk.text
        assert chunk.doc_title == "Policy Title"


def test_title_falls_back_to_the_file_name_when_there_is_none(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    write(kb, "no_title.md", "## Only Section\ntext\n")

    assert chunk_kb(kb)[0].doc_title == "no_title"


@pytest.mark.parametrize(
    "markdown, expected",
    [
        pytest.param(policy_text(), [], id="banner-title-and-id-only"),
        pytest.param("# T\n\nAn intro paragraph with a fact.\n\n## S\nx\n", ["An intro paragraph with a fact."], id="intro-text"),
        pytest.param("Text with no heading at all\n", ["Text with no heading at all"], id="no-heading-at-all"),
    ],
)
def test_stray_preamble_lines_flags_text_that_would_be_lost(markdown, expected):
    assert stray_preamble_lines(markdown) == expected


def test_a_section_with_no_text_makes_no_chunk(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    write(kb, "d.md", "# D\n\n## Empty\n\n## Full\ntext\n")

    assert [c.section for c in chunk_kb(kb)] == ["Full"]


def test_two_headings_with_the_same_fragment_are_rejected(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    write(kb, "d.md", "# D\n\n## Returns\none\n\n## Returns!\ntwo\n")

    with pytest.raises(ValueError, match="Duplicate section id"):
        chunk_kb(kb)


# ---------------------------------------------------------------------------
# Splitting a long section
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_chars", [30, 45, 60, 90, 120])
@pytest.mark.parametrize("overlap", [0, 20])
def test_no_chunk_exceeds_max_chars(tmp_path, max_chars, overlap):
    kb = make_kb(tmp_path)

    chunks = chunk_kb(kb, max_chars=max_chars, overlap=overlap)

    assert chunks and all(len(c.text) <= max_chars for c in chunks)


@pytest.mark.parametrize("max_chars", [30, 45, 60, 90, 120])
def test_without_overlap_no_text_is_lost_or_repeated(tmp_path, max_chars):
    kb = make_kb(tmp_path)

    parts = [c.text for c in chunk_kb(kb, max_chars=max_chars) if c.section == "Long Section"]

    assert squash("".join(parts)) == squash(LONG_BODY)


LISTED_BODY = (
    "First paragraph is here.\n\n"
    "- Item one of the list.\n"
    "- Item two of the list.\n\n"
    "Last paragraph is here."
)


def test_splits_at_paragraph_and_list_item_boundaries_first():
    parts = split_text(LISTED_BODY, max_chars=30)

    assert parts == [
        "First paragraph is here.",
        "- Item one of the list.",
        "- Item two of the list.",
        "Last paragraph is here.",
    ]


def test_whole_units_are_packed_together_up_to_the_limit():
    parts = split_text(LISTED_BODY, max_chars=50)

    assert parts == [
        "First paragraph is here.\n\n- Item one of the list.",
        "- Item two of the list.\n\nLast paragraph is here.",
    ]


def test_a_part_keeps_whole_paragraphs_together_when_they_fit():
    body = "Aaa aaa aaa.\n\nBbb bbb bbb.\n\nCcc ccc ccc."

    assert split_text(body, max_chars=30) == ["Aaa aaa aaa.\n\nBbb bbb bbb.", "Ccc ccc ccc."]


def test_a_long_paragraph_splits_only_at_sentence_boundaries():
    sentences = [f"Sentence number {i} says something useful." for i in range(1, 7)]
    body = " ".join(sentences)

    parts = split_text(body, max_chars=100)

    assert len(parts) > 1 and all(len(p) <= 100 for p in parts)
    for part in parts:  # each part is whole sentences only
        assert part.startswith("Sentence number") and part.endswith("useful.")
    assert " ".join(parts) == body


def test_numbered_markers_stay_with_their_sentence_when_an_item_is_split():
    body = "1. Aaa bbb ccc ddd. Eee fff ggg hhh. Iii jjj kkk lll."

    parts = split_text(body, max_chars=30)

    assert parts[0].startswith("1. Aaa")
    assert all(p.strip(" .") not in {"1", ""} for p in parts)  # no part is only a marker
    assert squash("".join(parts)) == squash(body)


def test_one_oversized_sentence_is_cut_at_word_boundaries():
    words = [f"word{i:02d}" for i in range(1, 13)]  # 6 characters each
    body = " ".join(words) + "."

    parts = split_text(body, max_chars=20)

    assert all(len(p) <= 20 for p in parts) and len(parts) > 1
    assert " ".join(parts).split() == body.split()  # same words, same order, none broken
    assert all(set(p.split()) <= set(body.split()) for p in parts)


def test_one_oversized_word_is_cut_at_exactly_max_chars():
    parts = split_text("x" * 45, max_chars=20)

    assert [len(p) for p in parts] == [20, 20, 5]
    assert "".join(parts) == "x" * 45


def test_empty_text_gives_no_parts():
    assert split_text("   \n\n  ", max_chars=50) == []


@pytest.mark.parametrize("max_chars, overlap", [(0, 0), (-5, 0), (50, -1), (50, 50), (50, 80)])
def test_invalid_limits_are_rejected(max_chars, overlap):
    with pytest.raises(ValueError):
        split_text("some text", max_chars=max_chars, overlap=overlap)


# ---------------------------------------------------------------------------
# Overlap: whole sentences carried from the previous part
# ---------------------------------------------------------------------------

def test_overlap_carries_the_last_whole_sentence_that_fits_the_budget():
    body = "Aaa aaa aaa aaa. Bbb bbb bbb bbb. Ccc ccc ccc ccc. Ddd ddd ddd ddd. Eee eee eee eee."

    parts = split_text(body, max_chars=50, overlap=17)  # one 16-character sentence fits, two do not

    assert parts == [
        "Aaa aaa aaa aaa. Bbb bbb bbb bbb. Ccc ccc ccc ccc.",
        "Ccc ccc ccc ccc. Ddd ddd ddd ddd. Eee eee eee eee.",
    ]


def test_overlap_smaller_than_one_sentence_carries_nothing():
    body = "Aaa aaa aaa aaa. Bbb bbb bbb bbb. Ccc ccc ccc ccc. Ddd ddd ddd ddd."

    assert split_text(body, max_chars=35, overlap=10) == split_text(body, max_chars=35, overlap=0)


def test_overlap_never_carries_the_whole_previous_part():
    body = "Aaa aaa aaa aaa. Bbb bbb bbb bbb. Ccc ccc ccc ccc."

    parts = split_text(body, max_chars=20, overlap=19)  # budget would fit a whole part

    assert parts == ["Aaa aaa aaa aaa.", "Bbb bbb bbb bbb.", "Ccc ccc ccc ccc."]


def test_with_overlap_every_sentence_is_still_present_and_parts_stay_within_the_limit():
    sentences = [f"Sentence number {i} says something useful." for i in range(1, 9)]
    body = " ".join(sentences)

    parts = split_text(body, max_chars=100, overlap=45)

    assert all(len(p) <= 100 for p in parts)
    for sentence in sentences:
        assert any(sentence in p for p in parts)
    repeated = sum(1 for sentence in sentences for p in parts if sentence in p) - len(sentences)
    assert repeated > 0  # some sentences were carried into the next part


def test_overlap_does_not_cross_sections(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    write(kb, "d.md", "# D\n\n## One\nAaa aaa aaa aaa. Bbb bbb bbb bbb.\n\n## Two\nCcc ccc ccc ccc. Ddd ddd ddd ddd.\n")

    chunks = chunk_kb(kb, max_chars=20, overlap=19)

    two = [c.text for c in chunks if c.section == "Two"]
    assert two[0].startswith("Ccc") and "Bbb" not in " ".join(two)


def test_a_list_item_keeps_its_own_line_after_a_carried_sentence():
    body = "1. Aaa aaa aaa. Bbb bbb bbb.\n2. Ccc ccc ccc."

    parts = split_text(body, max_chars=30, overlap=15)

    assert parts[-1] == "Bbb bbb bbb.\n2. Ccc ccc ccc."


def test_carried_list_items_keep_their_own_lines():
    body = "1. Aaa aaa aaa.\n2. Bbb bbb bbb.\n3. Ccc ccc ccc.\n4. Ddd ddd ddd."

    parts = split_text(body, max_chars=48, overlap=34)

    assert parts == [
        "1. Aaa aaa aaa.\n2. Bbb bbb bbb.\n3. Ccc ccc ccc.",
        "2. Bbb bbb bbb.\n3. Ccc ccc ccc.\n4. Ddd ddd ddd.",
    ]


# ---------------------------------------------------------------------------
# Ids: unique, deterministic, stable under edits elsewhere
# ---------------------------------------------------------------------------

def test_ids_are_unique_and_identical_on_a_second_run(tmp_path):
    kb = make_kb(tmp_path)
    write(kb, "second.md", "# Second\n\n## Short Section\nAnother short rule.\n")

    first = chunk_kb(kb, max_chars=60)
    second = chunk_kb(kb, max_chars=60)

    ids = [c.id for c in first]
    assert len(set(ids)) == len(ids)
    assert ids == [c.id for c in second]
    assert "policy.md#Short-Section::1" in ids and "second.md#Short-Section::1" in ids


def test_parts_are_numbered_from_one_within_their_section(tmp_path):
    kb = make_kb(tmp_path)

    long_chunks = [c for c in chunk_kb(kb, max_chars=60) if c.section == "Long Section"]

    assert [c.part for c in long_chunks] == list(range(1, len(long_chunks) + 1))
    assert {c.n_parts for c in long_chunks} == {len(long_chunks)}
    assert [c.id for c in long_chunks] == [f"policy.md#Long-Section::{n}" for n in range(1, len(long_chunks) + 1)]


def test_editing_one_section_changes_only_that_sections_chunk_hashes(tmp_path):
    kb = make_kb(tmp_path)
    before = {c.id: c.text_hash for c in chunk_kb(kb, max_chars=60)}

    write(kb, "policy.md", policy_text(LONG_BODY.replace("Gamma", "Gamma-edited")))
    after = {c.id: c.text_hash for c in chunk_kb(kb, max_chars=60)}

    assert before["policy.md#Short-Section::1"] == after["policy.md#Short-Section::1"]
    changed = [i for i in before if i in after and before[i] != after[i]]
    assert changed and all("Long-Section" in i for i in changed)


def test_renaming_a_heading_changes_the_section_id(tmp_path):
    kb = make_kb(tmp_path)

    write(kb, "policy.md", policy_text().replace("## Short Section", "## Brief Section"))
    ids = {c.id for c in chunk_kb(kb)}

    assert "policy.md#Brief-Section::1" in ids and "policy.md#Short-Section::1" not in ids


# ---------------------------------------------------------------------------
# The context prefix and the text hash
# ---------------------------------------------------------------------------

def test_embed_text_with_context_puts_title_and_section_before_the_text(tmp_path):
    chunk = chunk_kb(make_kb(tmp_path))[0]

    assert chunk.embed_text(with_context=True) == (
        f"Document: {chunk.doc_title}\nSection: {chunk.section}\n\n{chunk.text}"
    )


def test_embed_text_without_context_is_just_the_text_and_text_is_never_prefixed(tmp_path):
    chunk = chunk_kb(make_kb(tmp_path))[0]

    assert chunk.embed_text(with_context=False) == chunk.text
    assert "Document:" not in chunk.text


def test_text_hash_is_stable_and_depends_on_the_text():
    a = Chunk("i", "same text", "f.md", "S", "f.md#S", "T", 1, 1)
    b = Chunk("other id", "same text", "g.md", "X", "g.md#X", "U", 2, 3)
    c = Chunk("i", "different text", "f.md", "S", "f.md#S", "T", 1, 1)

    assert a.text_hash == b.text_hash != c.text_hash
    assert len(a.text_hash) == 12


# ---------------------------------------------------------------------------
# scripts/chunk_stats.py
# ---------------------------------------------------------------------------

def run_stats(kb, test_set_path, *extra):
    return subprocess.run(
        [sys.executable, str(CHUNK_STATS), "--kb-dir", str(kb), "--test-set", str(test_set_path), *extra],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def write_test_set(tmp_path, source, facts):
    path = tmp_path / "test_set.json"
    path.write_text(
        json.dumps([{"id": "Q1", "expected_sources": [source], "must_contain": facts}]), encoding="utf-8"
    )
    return path


def test_stats_prints_counts_sizes_and_skipped_files(tmp_path):
    kb = make_kb(tmp_path)

    result = run_stats(kb, write_test_set(tmp_path, "policy.md#Short-Section", ["short rule"]), "--max-chars", "60")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Chunks: " in result.stdout and "min=" in result.stdout and "median=" in result.stdout
    assert "Section text size in characters" in result.stdout
    assert "README.md" in result.stdout.split("Not indexed")[1].splitlines()[0]
    assert "(a) every string found in a chunk of its section: PASS" in result.stdout


def test_stats_exits_1_when_a_string_is_found_in_no_chunk_of_its_section(tmp_path):
    kb = make_kb(tmp_path)

    result = run_stats(kb, write_test_set(tmp_path, "policy.md#Short-Section", ["no such fact"]))

    assert result.returncode == 1
    assert "MISSING ['no such fact']" in result.stdout
    assert "FAIL" in result.stdout


def test_stats_reports_but_does_not_fail_when_strings_are_spread_over_chunks(tmp_path):
    kb = make_kb(tmp_path)
    # "Alpha" and "Epsilon" are both in the Long Section, but 60 characters apart.
    test_set = write_test_set(tmp_path, "policy.md#Long-Section", ["Alpha", "Epsilon"])

    split = run_stats(kb, test_set, "--max-chars", "60")
    whole = run_stats(kb, test_set, "--max-chars", "800")

    assert split.returncode == 0 and whole.returncode == 0
    assert "NOT in one chunk" in split.stdout and "0 of 1" in split.stdout
    assert "in one chunk" in whole.stdout and "1 of 1" in whole.stdout