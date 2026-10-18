"""Tests for app.rag.similarity, with hand-made vectors."""

import math

import pytest

from app.rag.similarity import cosine_similarity


def test_identical_vectors_give_1():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_orthogonal_vectors_give_0():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 1.0], [1.0, -1.0]) == pytest.approx(0.0)


def test_opposite_vectors_give_minus_1():
    assert cosine_similarity([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]) == pytest.approx(-1.0)


@pytest.mark.parametrize("scale", [0.001, 0.5, 2.0, 1000.0])
def test_a_positively_scaled_copy_gives_1(scale):
    a = [1.0, -2.0, 3.0, 0.5]

    assert cosine_similarity(a, [x * scale for x in a]) == pytest.approx(1.0)


def test_a_negatively_scaled_copy_gives_minus_1():
    a = [1.0, -2.0, 3.0]

    assert cosine_similarity(a, [x * -4.0 for x in a]) == pytest.approx(-1.0)


def test_a_45_degree_angle_gives_the_cosine_of_45_degrees():
    assert cosine_similarity([1.0, 0.0], [1.0, 1.0]) == pytest.approx(math.sqrt(2) / 2)


def test_the_order_of_the_arguments_does_not_matter():
    a, b = [1.0, 2.0, 3.0], [3.0, -1.0, 0.5]

    assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a))


def test_the_result_never_exceeds_1_despite_rounding():
    a = [0.69, 0.81, 0.18]  # the raw division gives 1.0000000000000002 for this pair
    b = [1.38, 1.62, 0.36]

    assert cosine_similarity(a, b) <= 1.0


def test_lists_tuples_and_numpy_arrays_are_accepted():
    numpy = pytest.importorskip("numpy")
    a = [1.0, 2.0, 3.0]

    assert cosine_similarity(a, tuple(a)) == pytest.approx(1.0)
    assert cosine_similarity(numpy.array(a), numpy.array(a)) == pytest.approx(1.0)
    assert cosine_similarity(numpy.array(a, dtype="float32"), a) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "a, b",
    [
        pytest.param([1.0, 2.0], [1.0, 2.0, 3.0], id="different-lengths"),
        pytest.param([], [], id="both-empty"),
        pytest.param([1.0], [], id="one-empty"),
        pytest.param([0.0, 0.0], [1.0, 2.0], id="first-is-a-zero-vector"),
        pytest.param([1.0, 2.0], [0.0, 0.0], id="second-is-a-zero-vector"),
        pytest.param([0.0, 0.0], [0.0, 0.0], id="both-zero-vectors"),
        pytest.param([float("nan"), 1.0], [1.0, 1.0], id="nan"),
        pytest.param([float("inf"), 1.0], [1.0, 1.0], id="infinity"),
    ],
)
def test_undefined_inputs_raise_value_error(a, b):
    with pytest.raises(ValueError):
        cosine_similarity(a, b)


def test_the_error_messages_say_what_is_wrong():
    with pytest.raises(ValueError, match="same length.*2 and 3"):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity([0.0], [1.0])
    with pytest.raises(ValueError, match="empty"):
        cosine_similarity([], [])