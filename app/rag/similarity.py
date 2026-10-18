"""Cosine similarity between two embedding vectors.

Decisions (the function never returns a made-up number):

- **Mismatched lengths** raise ValueError. Two vectors from different models,
  or a truncated one, are a bug, and silently comparing the common prefix
  would hide it.
- **Empty vectors** raise ValueError: there is nothing to compare.
- **A zero vector** (on either side) raises ValueError. Its direction is
  undefined, so the cosine is too; returning 0.0 would rank a broken embedding
  as "unrelated" instead of failing loudly.
- **NaN or infinite values** raise ValueError (checked on the result), because
  clamping would otherwise turn NaN into a plausible-looking score.
- The result is clamped to [-1.0, 1.0]: floating-point rounding can push the
  cosine of two parallel vectors to 1.0000000000000002.

Works on any sequence of numbers (lists, tuples, numpy arrays). It divides by
the vectors' lengths, so the inputs do not need to be normalized.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the cosine of the angle between `a` and `b`, from -1.0 to 1.0.

    1.0 means the same direction (a scaled copy counts), 0.0 means orthogonal,
    -1.0 means opposite. Raises ValueError as described in the module docstring.
    """
    if len(a) != len(b):
        raise ValueError(f"vectors must have the same length, got {len(a)} and {len(b)}")
    if len(a) == 0:
        raise ValueError("vectors must not be empty")

    dot = 0.0
    norm_a_squared = 0.0
    norm_b_squared = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a_squared += x * x
        norm_b_squared += y * y
    if norm_a_squared == 0.0 or norm_b_squared == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")

    value = dot / (math.sqrt(norm_a_squared) * math.sqrt(norm_b_squared))
    if not math.isfinite(value):
        raise ValueError("vectors contain NaN or infinite values")
    return max(-1.0, min(1.0, value))