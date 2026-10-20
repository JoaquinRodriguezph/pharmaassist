"""Tests for app.rag.metrics, on hand-made rankings (no model, no Chroma)."""

import pytest

from app.rag.metrics import (
    all_hit_at_k,
    collapse_to_sections,
    duplicate_slots,
    first_hit_rank,
    hit_at_k,
    mean_reciprocal_rank,
    reciprocal_rank,
    summarize,
)

A, B, C, D, E = "a.md#A", "a.md#B", "b.md#C", "b.md#D", "c.md#E"


@pytest.mark.parametrize(
    "ranking, expected, rank",
    [
        pytest.param([A, B, C], {A}, 1, id="first-place"),
        pytest.param([B, A, C], {A}, 2, id="second-place"),
        pytest.param([B, C, D, E, A], {A}, 5, id="fifth-place"),
        pytest.param([B, C, D], {A}, None, id="not-retrieved"),
        pytest.param([], {A}, None, id="nothing-retrieved"),
        pytest.param([B, C, A, D], {A, D}, 3, id="first-of-two-expected-sections"),
    ],
)
def test_first_hit_rank(ranking, expected, rank):
    assert first_hit_rank(ranking, expected) == rank


def test_a_repeated_chunk_takes_a_slot_so_it_pushes_the_expected_section_down():
    ranking = [B, B, A]  # two chunks of B, then the expected section

    assert first_hit_rank(ranking, {A}) == 3
    assert first_hit_rank(collapse_to_sections(ranking), {A}) == 2


def test_the_rank_of_a_section_is_the_rank_of_its_first_chunk():
    assert first_hit_rank([A, A, B], {A}) == 1
    assert first_hit_rank([B, A, A], {A}) == 2


def test_collapse_to_sections_keeps_first_positions():
    assert collapse_to_sections([A, B, A, C, B]) == [A, B, C]
    assert collapse_to_sections([]) == []


@pytest.mark.parametrize("k, expected_result", [(1, False), (2, False), (3, True), (4, True), (10, True)])
def test_hit_at_k_flips_exactly_at_the_rank(k, expected_result):
    assert hit_at_k([B, C, A, D], {A}, k) is expected_result


def test_hit_at_k_is_false_when_nothing_expected_is_retrieved():
    assert hit_at_k([B, C], {A}, 5) is False


def test_all_hit_at_k_needs_every_expected_section():
    ranking = [A, B, D]
    expected = {A, D}

    assert hit_at_k(ranking, expected, 1) is True  # one of the two is enough for hit@k
    assert all_hit_at_k(ranking, expected, 1) is False
    assert all_hit_at_k(ranking, expected, 2) is False
    assert all_hit_at_k(ranking, expected, 3) is True


def test_all_hit_at_k_equals_hit_at_k_for_a_single_expected_section():
    for k in (1, 2, 3):
        assert all_hit_at_k([B, A, C], {A}, k) == hit_at_k([B, A, C], {A}, k)


@pytest.mark.parametrize(
    "ranking, expected_value",
    [
        pytest.param([A], 1.0, id="rank-1"),
        pytest.param([B, A], 0.5, id="rank-2"),
        pytest.param([B, C, D, A], 0.25, id="rank-4"),
        pytest.param([B, C], 0.0, id="no-hit"),
    ],
)
def test_reciprocal_rank(ranking, expected_value):
    assert reciprocal_rank(ranking, {A}) == pytest.approx(expected_value)


def test_mean_reciprocal_rank_averages_over_questions():
    cases = [([A], {A}), ([B, A], {A}), ([B, C, D, A], {A}), ([B], {A})]  # 1, 0.5, 0.25, 0

    assert mean_reciprocal_rank(cases) == pytest.approx((1 + 0.5 + 0.25 + 0) / 4)


def test_mean_reciprocal_rank_of_no_questions_is_an_error():
    with pytest.raises(ValueError, match="at least one"):
        mean_reciprocal_rank([])


@pytest.mark.parametrize(
    "ranking, k, expected_count",
    [
        pytest.param([A, B, C], 3, 0, id="no-repeats"),
        pytest.param([A, A, B, A, C], 5, 2, id="two-repeated-slots"),
        pytest.param([A, A, A, A, A], 5, 4, id="one-section-fills-the-top-5"),
        pytest.param([A, A, B, B], 2, 1, id="only-the-top-k-count"),
        pytest.param([A, A], 5, 1, id="fewer-results-than-k"),
        pytest.param([], 5, 0, id="no-results"),
    ],
)
def test_duplicate_slots(ranking, k, expected_count):
    assert duplicate_slots(ranking, k) == expected_count


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda: first_hit_rank([A], set()), id="first-hit-rank-empty-expected"),
        pytest.param(lambda: hit_at_k([A], set(), 3), id="hit-empty-expected"),
        pytest.param(lambda: all_hit_at_k([A], [], 3), id="all-hit-empty-expected"),
        pytest.param(lambda: reciprocal_rank([A], set()), id="rr-empty-expected"),
        pytest.param(lambda: hit_at_k([A], {A}, 0), id="hit-k-zero"),
        pytest.param(lambda: all_hit_at_k([A], {A}, -1), id="all-hit-k-negative"),
        pytest.param(lambda: duplicate_slots([A], 0), id="duplicates-k-zero"),
    ],
)
def test_undefined_inputs_raise_value_error(call):
    with pytest.raises(ValueError):
        call()


def test_summarize_counts_hits_and_averages_over_hand_made_rankings():
    cases = [
        ([A, B, C, D, E], {A}),  # rank 1
        ([B, C, A, D, E], {A}),  # rank 3
        ([B, C, D, E, "x"], {A}),  # not in the top 5
        ([B, B, A, D, E], {A}),  # rank 3 as chunks, rank 2 as sections
        ([A, D, B, C, E], {A, D}),  # both expected sections within the top 2
    ]

    result = summarize(cases)

    assert result["n"] == 5
    assert result["hits"] == {1: 2, 3: 4, 5: 4}
    assert result["all_hits"] == {1: 1, 3: 4, 5: 4}  # the two-section question cannot have both in the top 1
    assert result["mrr"] == pytest.approx((1 + 1 / 3 + 0 + 1 / 3 + 1) / 5)
    assert result["mrr_distinct"] == pytest.approx((1 + 1 / 3 + 0 + 1 / 2 + 1) / 5)
    assert result["repeat_slots"] == 1 and result["questions_with_repeats"] == 1


def test_summarize_separates_all_hit_from_hit_when_a_question_has_two_sections():
    result = summarize([([A, B, C], {A, D})], ks=(1, 3))

    assert result["hits"] == {1: 1, 3: 1}
    assert result["all_hits"] == {1: 0, 3: 0}


def test_summarize_of_no_questions_is_an_error():
    with pytest.raises(ValueError):
        summarize([])