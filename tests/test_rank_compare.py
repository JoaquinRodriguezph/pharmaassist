"""Tests for app.rag.rank_compare (the tie rule) and for the vector helpers in app.rag.faiss_index."""

from types import SimpleNamespace

import numpy as np
import pytest

from app.rag.faiss_index import normalized_max_abs_difference, vector_stats
from app.rag.rank_compare import SCORE_TOLERANCE, compare_results


def hits(*pairs):
    return [SimpleNamespace(id=i, score=s) for i, s in pairs]


# ---------------------------------------------------------------------------
# The tie rule
# ---------------------------------------------------------------------------

def test_identical_lists_with_scores_inside_tolerance():
    result = compare_results(hits(("a", 0.9), ("b", 0.8)), hits(("a", 0.9 + 4e-6), ("b", 0.8)))

    assert result.identical and result.first_difference is None and result.is_tie is None
    assert result.max_score_difference == pytest.approx(4e-6)


def test_a_swap_of_two_chunks_that_are_equal_in_both_backends_is_a_tie():
    chroma = hits(("a", 0.9), ("b", 0.700000), ("c", 0.700001))
    faiss = hits(("a", 0.9), ("c", 0.700001), ("b", 0.700000))

    result = compare_results(chroma, faiss)

    assert not result.identical and result.first_difference == 1 and result.is_tie is True
    assert result.reason.startswith("tie:")


def test_a_swap_of_two_chunks_whose_scores_are_far_apart_is_not_a_tie():
    chroma = hits(("a", 0.9), ("b", 0.8), ("c", 0.7))
    faiss = hits(("a", 0.9), ("c", 0.8), ("b", 0.7))

    result = compare_results(chroma, faiss)

    assert result.is_tie is False and result.reason.startswith("not a tie:")


def test_chunks_that_are_close_in_only_one_backend_are_not_a_tie():
    chroma = hits(("a", 0.9), ("b", 0.700000), ("c", 0.700001))
    faiss = hits(("a", 0.9), ("c", 0.71), ("b", 0.70))

    assert compare_results(chroma, faiss).is_tie is False


def test_a_chunk_missing_from_the_other_list_is_not_shown_to_be_a_tie():
    chroma = hits(("a", 0.9), ("b", 0.8))
    faiss = hits(("a", 0.9), ("z", 0.8))

    result = compare_results(chroma, faiss)

    assert result.is_tie is False and "missing" in result.reason


def test_lists_of_different_length_are_a_difference():
    result = compare_results(hits(("a", 0.9), ("b", 0.8)), hits(("a", 0.9)))

    assert not result.identical and result.is_tie is False and result.first_difference == 1


def test_the_tolerance_is_the_one_the_docs_state():
    assert SCORE_TOLERANCE == 1e-5


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def test_vector_stats_reports_count_dtype_shape_and_norm_range():
    vectors = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]], dtype=np.float32)

    stats = vector_stats(vectors)

    assert stats["count"] == 3 and stats["dtype"] == "float32" and stats["shape"] == (3, 2)
    assert stats["min_norm"] == pytest.approx(1.0) and stats["max_norm"] == pytest.approx(5.0)


def test_vector_stats_of_no_vectors_has_no_norms():
    stats = vector_stats(np.zeros((0, 4), dtype=np.float32))

    assert stats["count"] == 0 and stats["min_norm"] is None and stats["max_norm"] is None


def test_vector_stats_needs_a_2d_set():
    with pytest.raises(ValueError, match="2-D"):
        vector_stats(np.array([1.0, 2.0]))


def test_difference_after_normalizing_ignores_vector_length():
    a = np.array([[3.0, 4.0], [1.0, 0.0]])
    b = np.array([[0.3, 0.4], [10.0, 0.0]])  # same directions, other lengths

    difference, _ = normalized_max_abs_difference(a, b)

    assert difference == pytest.approx(0.0, abs=1e-12)


def test_difference_after_normalizing_finds_the_worst_row():
    a = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.001]])

    difference, row = normalized_max_abs_difference(a, b)

    assert row == 1 and difference == pytest.approx(1.0)


@pytest.mark.parametrize(
    "a, b, message",
    [
        pytest.param(np.ones((2, 3)), np.ones((3, 3)), "same shape", id="different-counts"),
        pytest.param(np.ones((0, 3)), np.ones((0, 3)), "non-empty", id="empty"),
        pytest.param(np.zeros((1, 3)), np.ones((1, 3)), "zero vector", id="zero-vector"),
    ],
)
def test_difference_after_normalizing_refuses_bad_input(a, b, message):
    with pytest.raises(ValueError, match=message):
        normalized_max_abs_difference(a, b)