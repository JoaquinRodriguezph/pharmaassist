"""Tests for the heading rule (app.rag.kb) and scripts/check_kb.py, using a tiny temporary KB."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.rag.kb import (
    check_sources,
    expected_sources,
    required_strings,
    section_body,
    section_headings,
    slugify_heading,
)

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
    (kb / "a.md").write_text(
        "# A\n\n"
        "## Customer Returns\nItems within 30 days of expiry go to the Quarantine Bin.\n\n"
        "## Cold Chain And Excursions\nRange 2°C–8°C.\n\n"
        "## Other Section\nRecord the batch\nnumber here.\n",
        encoding="utf-8",
    )
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


def write_test_set_with_facts(tmp_path, source, facts):
    path = tmp_path / "test_set.json"
    path.write_text(
        json.dumps([{"id": "Q1", "expected_sources": [source], "must_contain": facts}]),
        encoding="utf-8",
    )
    return path


def test_script_reports_a_missing_fact_and_exits_1(tmp_path):
    kb = make_kb(tmp_path)
    # "30 days" is in the section; "batch number" only appears in the NEXT section.
    test_set = write_test_set_with_facts(tmp_path, "a.md#Customer-Returns", ["30 days", "batch number"])

    result = run_script(kb, test_set)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "MISSING: a.md#Customer-Returns" in result.stdout
    assert "batch number" in result.stdout
    assert "30 days" not in result.stdout


# ---------------------------------------------------------------------------
# Section text and required strings (must_contain)
# ---------------------------------------------------------------------------

def test_section_body_runs_to_the_next_level_two_heading_and_skips_the_heading_line():
    markdown = "## One\nalpha\n### Sub\nbeta\n## Two\ngamma\n"

    body = section_body(markdown, "One")

    assert "alpha" in body and "beta" in body  # subsection text belongs to the section
    assert "gamma" not in body  # the next section does not
    assert "One" not in body  # the heading line itself is left out


def test_section_body_of_the_last_section_runs_to_the_end():
    assert "gamma" in section_body("## One\nalpha\n## Two\ngamma\n", "Two")


def test_section_body_of_an_unknown_fragment_is_none():
    assert section_body("## One\nalpha\n", "Nope") is None


def test_check_sources_passes_when_all_facts_are_present(tmp_path):
    kb = make_kb(tmp_path)
    required = {
        "a.md#Customer-Returns": ["30 days", "quarantine bin"],
        "a.md#Cold-Chain-And-Excursions": ["2°C–8°C"],
    }

    assert check_sources(kb, list(required), required) == []


@pytest.mark.parametrize(
    "source, needed",
    [
        pytest.param("a.md#Customer-Returns", "30 DAYS", id="case-is-ignored"),
        pytest.param("a.md#Customer-Returns", "quarantine bin", id="case-is-ignored-other-direction"),
        pytest.param("a.md#Other-Section", "batch number", id="line-break-counts-as-a-space"),
    ],
)
def test_fact_matching_is_case_insensitive_and_whitespace_tolerant(tmp_path, source, needed):
    kb = make_kb(tmp_path)

    assert check_sources(kb, [source], {source: [needed]}) == []


def test_missing_fact_names_the_string_and_the_section(tmp_path):
    kb = make_kb(tmp_path)
    source = "a.md#Customer-Returns"

    problems = check_sources(kb, [source], {source: ["30 days", "supplier return"]})

    assert problems == [f"{source} (section text is missing 'supplier return')"]


def test_a_fact_that_only_appears_in_the_next_section_does_not_count(tmp_path):
    kb = make_kb(tmp_path)
    source = "a.md#Customer-Returns"

    problems = check_sources(kb, [source], {source: ["batch number"]})  # it is in Other Section

    assert len(problems) == 1 and "batch number" in problems[0]


def test_a_fact_that_only_appears_in_the_heading_does_not_count(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "s.md").write_text("## Receiving And FEFO\nShelve by expiry date.\n", encoding="utf-8")
    source = "s.md#Receiving-And-FEFO"

    problems = check_sources(kb, [source], {source: ["FEFO"]})

    assert len(problems) == 1 and "'FEFO'" in problems[0]


def test_missing_heading_is_reported_once_even_with_required_strings(tmp_path):
    kb = make_kb(tmp_path)
    source = "a.md#Refund-Rules"

    problems = check_sources(kb, [source], {source: ["x", "y"]})

    assert len(problems) == 1 and "section not found" in problems[0]


def test_required_strings_list_form_applies_to_every_source():
    test_set = [{"id": "Q1", "expected_sources": ["a.md#One", "b.md#Two"], "must_contain": ["x"]}]

    assert required_strings(test_set) == {"a.md#One": ["x"], "b.md#Two": ["x"]}


def test_required_strings_object_form_is_per_source():
    test_set = [
        {
            "id": "Q1",
            "expected_sources": ["a.md#One", "b.md#Two"],
            "must_contain": {"a.md#One": ["x", "y"], "b.md#Two": ["y"]},
        }
    ]

    assert required_strings(test_set) == {"a.md#One": ["x", "y"], "b.md#Two": ["y"]}


def test_required_strings_merges_questions_citing_the_same_source():
    test_set = [
        {"id": "Q1", "expected_sources": ["a.md#One"], "must_contain": ["x", "y"]},
        {"id": "Q2", "expected_sources": ["a.md#One"], "must_contain": ["y", "z"]},
        {"id": "Q3", "expected_sources": ["a.md#One"]},
    ]

    assert required_strings(test_set) == {"a.md#One": ["x", "y", "z"]}


@pytest.mark.parametrize(
    "must_contain",
    [
        pytest.param({"a.md#Typo": ["x"]}, id="object-key-not-in-expected-sources"),
        pytest.param("x", id="a-bare-string"),
        pytest.param({"a.md#One": "x"}, id="object-value-not-a-list"),
        pytest.param(["x", ""], id="empty-string"),
        pytest.param(["x", 3], id="non-string"),
    ],
)
def test_required_strings_rejects_malformed_must_contain(must_contain):
    test_set = [{"id": "Q1", "expected_sources": ["a.md#One"], "must_contain": must_contain}]

    with pytest.raises(ValueError, match="Q1"):
        required_strings(test_set)


def test_script_exits_0_when_headings_and_facts_are_all_present(tmp_path):
    kb = make_kb(tmp_path)
    test_set = write_test_set_with_facts(tmp_path, "a.md#Customer-Returns", ["30 days", "quarantine bin"])

    result = run_script(kb, test_set)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 required string(s): 0 problem(s)" in result.stdout


def test_script_reports_a_missing_heading_and_exits_1(tmp_path):
    kb = make_kb(tmp_path)
    test_set = write_test_set_with_facts(tmp_path, "a.md#Refund-Rules", ["30 days"])

    result = run_script(kb, test_set)

    assert result.returncode == 1
    assert "MISSING: a.md#Refund-Rules (section not found" in result.stdout


def test_script_rejects_a_malformed_must_contain_with_exit_1(tmp_path):
    kb = make_kb(tmp_path)
    test_set = tmp_path / "test_set.json"
    test_set.write_text(
        json.dumps([{"id": "Q7", "expected_sources": ["a.md#Customer-Returns"], "must_contain": "x"}]),
        encoding="utf-8",
    )

    result = run_script(kb, test_set)

    assert result.returncode == 1
    assert "ERROR" in result.stdout and "Q7" in result.stdout


def test_check_sources_rejects_citing_readme_or_other_non_documents(tmp_path):
    kb = make_kb(tmp_path)
    (kb / "README.md").write_text("# R\n\n## Heading Rule\ntext\n", encoding="utf-8")
    (kb / "_draft.md").write_text("# D\n\n## Draft Section\ntext\n", encoding="utf-8")

    problems = check_sources(kb, ["README.md#Heading-Rule", "_draft.md#Draft-Section", "a.md#Customer-Returns"])

    assert len(problems) == 2
    assert all("not a knowledge-base document" in p for p in problems)