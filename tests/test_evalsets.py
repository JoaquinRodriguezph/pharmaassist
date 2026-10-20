"""Tests for app.rag.evalsets and for the real variants file."""

import json
from pathlib import Path

import pytest

from app.rag.evalsets import (
    ORIGINAL_SCORE_ONLY_IDS,
    ROUTES,
    load_independent,
    load_variants,
    original_questions,
    split_answerable,
)
from app.rag.kb import check_sources, normalize_for_match, section_body

REPO_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = REPO_ROOT / "docs" / "knowledge_base"
TEST_SET = json.loads((REPO_ROOT / "eval" / "test_set.json").read_text(encoding="utf-8"))
RAG_IDS = {"Q01", "Q02", "Q03", "Q04", "Q05", "Q06", "Q07", "Q08", "Q17", "Q18", "Q19"}


def write_json(tmp_path, data, name="set.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def small_test_set():
    return [
        {"id": "Q1", "question": "first question?", "expected_sources": ["a.md#A"]},
        {"id": "Q09", "question": "unanswerable?", "expected_sources": []},
        {"id": "Q10", "question": "unanswerable too?", "expected_sources": []},
        {"id": "Q20", "question": "refuse?", "expected_sources": []},
        {"id": "Q30", "question": "another empty one", "expected_sources": []},
    ]


# ---------------------------------------------------------------------------
# The original questions
# ---------------------------------------------------------------------------

def test_original_questions_are_the_eleven_rag_questions_plus_three_score_only():
    answerable, score_only = split_answerable(original_questions(TEST_SET))

    assert {q.id for q in answerable} == RAG_IDS
    assert {q.id for q in score_only} == set(ORIGINAL_SCORE_ONLY_IDS)
    assert all(q.group == "original" for q in answerable + score_only)
    assert [q.expected for q in answerable if q.id == "Q18"] == [
        ("storage_sop.md#Receiving-And-FEFO", "storage_sop.md#Temperature-Monitoring")
    ]


def test_original_questions_leaves_out_other_unanswerable_questions():
    ids = {q.id for q in original_questions(small_test_set())}

    assert ids == {"Q1", "Q09", "Q10", "Q20"}  # Q30 is neither measured nor in the score-only list


def test_a_test_set_missing_a_score_only_question_is_an_error():
    with pytest.raises(ValueError, match="Q20"):
        original_questions(small_test_set()[:3])


# ---------------------------------------------------------------------------
# The real variants file
# ---------------------------------------------------------------------------

def test_the_real_variants_file_has_one_valid_variant_per_rag_question():
    variants = load_variants(REPO_ROOT / "eval" / "retrieval_variants.json", TEST_SET)

    assert len(variants) == 11
    assert {v.base_id for v in variants} == RAG_IDS
    assert len({v.base_id for v in variants}) == 11
    assert all(v.group == "variants" and v.style for v in variants)


def test_the_real_variants_file_says_who_wrote_the_variants_and_when():
    items = json.loads((REPO_ROOT / "eval" / "retrieval_variants.json").read_text(encoding="utf-8"))

    assert all(item["written_by"] == "me, after seeing the documents" for item in items)


def test_the_real_variants_use_a_spread_of_styles():
    styles = {v.style for v in load_variants(REPO_ROOT / "eval" / "retrieval_variants.json", TEST_SET)}

    assert {"keywords only", "typo-heavy", "different phrasing", "Taglish"} <= styles


# ---------------------------------------------------------------------------
# Checks on a variants file
# ---------------------------------------------------------------------------

def good_variant(**changes):
    item = {
        "id": "V1",
        "base_id": "Q1",
        "style": "keywords only",
        "question": "first keywords",
        "expected_sources": ["a.md#A"],
        "written_by": "me, after seeing the documents",
    }
    item.update(changes)
    return item


def test_a_good_variant_loads(tmp_path):
    (variant,) = load_variants(write_json(tmp_path, [good_variant()]), small_test_set())

    assert (variant.id, variant.base_id, variant.style, variant.expected) == ("V1", "Q1", "keywords only", ("a.md#A",))


@pytest.mark.parametrize(
    "change, message",
    [
        pytest.param({"base_id": "Q99"}, "not in the test set", id="unknown-base"),
        pytest.param({"expected_sources": ["a.md#Other"]}, "must equal", id="different-expected-sources"),
        pytest.param({"expected_sources": []}, "must equal", id="missing-expected-sources"),
        pytest.param({"question": "first question?"}, "identical", id="same-as-the-original"),
        pytest.param({"question": "  "}, "question", id="blank-question"),
        pytest.param({"style": ""}, "style", id="no-style"),
        pytest.param({"written_by": ""}, "written_by", id="no-writer"),
    ],
)
def test_a_bad_variant_is_rejected(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        load_variants(write_json(tmp_path, [good_variant(**change)]), small_test_set())


def test_duplicate_variant_ids_are_rejected(tmp_path):
    path = write_json(tmp_path, [good_variant(), good_variant(question="other keywords")])

    with pytest.raises(ValueError, match="duplicate"):
        load_variants(path, small_test_set())


# ---------------------------------------------------------------------------
# Checks on the independent file
# ---------------------------------------------------------------------------

def independent(**changes):
    item = {
        "id": "IND01",
        "question": "can i return opened cough syrup",
        "written_by": "a colleague",
        "expected_sources": ["a.md#A"],
        "expected_route": "rag",
        "expected_behavior": "Answers from section A.",
    }
    item.update(changes)
    return item


def unanswerable(**changes):
    return independent(expected_sources=[], expected_route="rag_unanswerable", **changes)


def tool_question(**changes):
    return independent(
        expected_sources=[], expected_route="tool", expected_tool="get_expiring_batches", **changes
    )


def test_labeled_independent_questions_load_including_unanswerable_ones(tmp_path):
    path = write_json(tmp_path, [independent(), unanswerable(id="IND02")])

    answerable, unanswered = split_answerable(load_independent(path))

    assert [q.id for q in answerable] == ["IND01"] and [q.id for q in unanswered] == ["IND02"]
    assert answerable[0].group == "independent"
    assert (answerable[0].route, unanswered[0].route) == ("rag", "rag_unanswerable")
    assert answerable[0].behavior == "Answers from section A."


def test_a_tool_question_and_the_optional_fields_load(tmp_path):
    item = tool_question(related_sections=["a.md#B"], must_contain=["90 days"])

    (question,) = load_independent(write_json(tmp_path, [item]))

    assert (question.route, question.tool, question.expected) == ("tool", "get_expiring_batches", ())
    assert question.related == ("a.md#B",) and question.must_contain == ("90 days",)


def test_an_unlabeled_independent_question_is_an_error_not_a_silent_skip(tmp_path):
    item = independent()
    del item["expected_sources"]

    with pytest.raises(ValueError, match="not labeled yet"):
        load_independent(write_json(tmp_path, [item]))


def test_a_question_with_no_route_is_not_labeled_yet(tmp_path):
    item = independent()
    del item["expected_route"]

    with pytest.raises(ValueError, match="not labeled yet.*expected_route"):
        load_independent(write_json(tmp_path, [item]))


@pytest.mark.parametrize(
    "change, message",
    [
        pytest.param({"expected_sources": "a.md#A"}, "list", id="not-a-list"),
        pytest.param({"question": ""}, "question", id="blank-question"),
        pytest.param({"written_by": ""}, "written_by", id="no-writer"),
        pytest.param({"expected_route": "search"}, "not one of", id="unknown-route"),
        pytest.param({"expected_behavior": " "}, "expected_behavior", id="no-behavior"),
        pytest.param({"expected_sources": [], "expected_route": "rag"}, "needs at least one expected source", id="rag-without-a-source"),
        pytest.param({"expected_route": "rag_unanswerable"}, "must have no expected sources", id="unanswerable-with-a-source"),
        pytest.param({"expected_route": "tool"}, "must have no expected sources", id="tool-with-a-source"),
        pytest.param({"expected_sources": [], "expected_route": "tool"}, "needs an 'expected_tool'", id="tool-without-a-tool"),
        pytest.param({"expected_sources": [], "expected_route": "both", "expected_tool": "get_restock"}, "needs at least one expected source", id="both-without-a-source"),
        pytest.param({"expected_tool": "get_restock"}, "must not have an 'expected_tool'", id="rag-with-a-tool"),
        pytest.param({"related_sections": "a.md#B"}, "related_sections", id="related-not-a-list"),
        pytest.param({"must_contain": [""]}, "must_contain", id="blank-required-string"),
    ],
)
def test_a_bad_independent_question_is_rejected(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        load_independent(write_json(tmp_path, [independent(**change)]))


def test_duplicate_independent_ids_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        load_independent(write_json(tmp_path, [independent(), independent(question="another")]))


def test_both_needs_a_source_and_a_tool(tmp_path):
    item = independent(expected_route="both", expected_tool="get_restock")

    (question,) = load_independent(write_json(tmp_path, [item]))

    assert (question.route, question.tool, question.expected) == ("both", "get_restock", ("a.md#A",))


def test_original_questions_carry_their_route_from_the_test_set():
    routes = {q.id: q.route for q in original_questions(TEST_SET)}

    assert routes["Q01"] == "rag" and routes["Q09"] == "rag_unanswerable" and routes["Q20"] == "refuse"
    assert set(routes.values()) <= set(ROUTES)


# ---------------------------------------------------------------------------
# The real independent file
# ---------------------------------------------------------------------------

def real_independent():
    return load_independent(REPO_ROOT / "eval" / "retrieval_independent.json")


def test_the_real_independent_file_is_labeled_and_mostly_asks_about_live_state():
    questions = real_independent()
    answerable, unanswerable_ = split_answerable(questions)

    assert len(questions) == 8
    assert {q.id for q in answerable} == {"IND04", "IND05"}
    assert len(unanswerable_) == 6
    sections = {s for q in answerable for s in q.expected}
    assert sections == {
        "returns_and_expiry_policy.md#Expired-Medications",
        "returns_and_expiry_policy.md#Customer-Returns",
    }


def test_the_real_independent_routes():
    routes = {q.id: q.route for q in real_independent()}

    assert routes == {
        "IND01": "rag_unanswerable",
        "IND02": "rag_unanswerable",
        "IND03": "tool",
        "IND04": "rag",
        "IND05": "rag",
        "IND06": "rag_unanswerable",
        "IND07": "rag_unanswerable",
        "IND08": "rag_unanswerable",
    }


def test_the_one_real_tool_question_names_the_tool_for_near_expiry_stock():
    (tool,) = [q for q in real_independent() if q.route == "tool"]

    assert (tool.id, tool.tool) == ("IND03", "get_expiring_batches")


def test_every_real_unanswerable_question_that_has_a_rule_to_state_says_it_cannot_see_live_data():
    for q in real_independent():
        if q.route == "rag_unanswerable":
            assert q.related and q.must_contain, q.id
            assert "cannot" in q.behavior and "States the" in q.behavior, q.id


def test_the_real_independent_labels_point_at_real_sections_and_strings():
    for q in real_independent():
        sections = list(q.expected) + list(q.related)
        assert check_sources(KB_DIR, sections) == [], q.id
        text = " ".join(
            section_body((KB_DIR / ref.split("#")[0]).read_text(encoding="utf-8"), ref.split("#")[1]) or ""
            for ref in sections
        )
        for needed in q.must_contain:
            assert normalize_for_match(needed) in normalize_for_match(text), (q.id, needed)