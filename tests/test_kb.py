"""Tests for the heading rule (app.rag.kb) and scripts/check_kb.py, using a tiny temporary KB."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.rag.kb import check_sources, expected_sources, section_headings, slugify_heading

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECK_KB = REPO_ROOT / "scripts" / "check_kb.py"


@pytest.mark.parametrize(
    "heading, expected",
    [
        pytest.param("Customer Returns", "Customer-Returns", id="two-words"),
        pytest.param("Cold Chain And Excursions", "Cold-Chain-And-Excursions", id="capital-And"),
        pytest.param("Receiving And FEFO", "Receiving-And-FEFO", id="acronym-kept-uppercase"),
        pytest.param("Contraindications", "Contraindications", id="single-word"),
        pytest.param("  Extra   Spaces  ", "Extra-Spaces", id="whitespace-collapsed-and-trimmed"),
        pytest.param("Q&A: Basics!", "QA-Basics", id="punctuation-dropped"),
        pytest.param("Pre- And Post-Care", "Pre-And-Post-Care", id="hyphen-runs-collapsed"),
        pytest.param("Already-Hyphenated", "Already-Hyphenated", id="hyphens-kept"),
        pytest.param("lowercase heading", "lowercase-heading", id="case-is-not-changed"),
    ],
)
def test_slugify_heading(heading, expected):
    assert slugify_heading(heading) == expected


def test_section_headings_finds_only_level_two():
    markdown = (
        "> banner\n\n# Title\n\n## First Section  \ntext with ## inside\n"
        "### Subsection\n## Second Section\n####### not a heading\n"
    )
    assert section_headings(markdown) == ["First Section", "Second Section"]


def test_expected_sources_dedupes_in_order_and_skips_empty():
    test_set = [
        {"id": "Q1", "expected_sources": ["a.md#One"]},
        {"id": "Q2", "expected_sources": []},
        {"id": "Q3", "expected_sources": None},
        {"id": "Q4"},
        {"id": "Q5", "expected_sources": ["b.md#Two", "a.md#One"]},
    ]
    assert expected_sources(test_set) == ["a.md#One", "b.md#Two"]


def make_kb(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "a.md").write_text("# A\n\n## Customer Returns\ntext\n\n## Cold Chain And Excursions\ntext\n", encoding="utf-8")
    return kb


def test_check_sources_all_present(tmp_path):
    kb = make_kb(tmp_path)
    assert check_sources(kb, ["a.md#Customer-Returns", "a.md#Cold-Chain-And-Excursions"]) == []


@pytest.mark.parametrize(
    "source, expected_fragment",
    [
        pytest.param("missing.md#Customer-Returns", "file not found", id="missing-file"),
        pytest.param("a.md#Refund-Rules", "section not found", id="missing-section"),
        pytest.param("a.md#customer-returns", "section not found", id="wrong-case-is-not-a-match"),
        pytest.param("a.md", "not in the form", id="no-fragment"),
        pytest.param("#Customer-Returns", "not in the form", id="no-file"),
    ],
)
def test_check_sources_reports_problems(tmp_path, source, expected_fragment):
    kb = make_kb(tmp_path)

    problems = check_sources(kb, [source])

    assert len(problems) == 1
    assert source in problems[0]
    assert expected_fragment in problems[0]


def test_missing_section_message_lists_what_the_file_has(tmp_path):
    kb = make_kb(tmp_path)

    (problem,) = check_sources(kb, ["a.md#Refund-Rules"])

    assert "Customer-Returns" in problem and "Cold-Chain-And-Excursions" in problem


def run_script(kb, test_set_path):
    return subprocess.run(
        [sys.executable, str(CHECK_KB), "--kb-dir", str(kb), "--test-set", str(test_set_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def write_test_set(tmp_path, sources):
    path = tmp_path / "test_set.json"
    path.write_text(json.dumps([{"id": "Q1", "expected_sources": sources}]), encoding="utf-8")
    return path


def test_script_exits_0_when_everything_is_present(tmp_path):
    kb = make_kb(tmp_path)

    result = run_script(kb, write_test_set(tmp_path, ["a.md#Customer-Returns"]))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 problem(s)" in result.stdout


def test_script_prints_each_missing_source_and_exits_1(tmp_path):
    kb = make_kb(tmp_path)
    sources = ["a.md#Customer-Returns", "a.md#Refund-Rules", "gone.md#Anything"]

    result = run_script(kb, write_test_set(tmp_path, sources))

    assert result.returncode == 1
    assert "MISSING: a.md#Refund-Rules" in result.stdout
    assert "MISSING: gone.md#Anything" in result.stdout
    assert "MISSING: a.md#Customer-Returns" not in result.stdout
    assert "2 problem(s)" in result.stdout