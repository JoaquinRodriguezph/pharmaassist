"""Comparing two top-k result lists, with the tie rule written down as code.

This is the rule `docs/vector_store_comparison.md` states before the first run of
`scripts/compare_faiss.py`. Results are anything with `id` and `score` attributes.

**Identical:** the two lists have the same chunk ids in the same order.

**Tie:** if the lists differ, look at the first rank where the ids differ. Call Chroma's
chunk there X and FAISS's chunk there Y. The difference is a *tie* only if X and Y both
appear in both lists and, in each backend, their two scores are within
`SCORE_TOLERANCE` of each other. Anything else is *not a tie* and has to be explained
before any conclusion is written: a chunk missing from the other list counts as not a
tie, because the tie cannot be shown.

**Score agreement:** for chunks at the same rank in both lists, scores should differ by
at most `SCORE_TOLERANCE`.

`SCORE_TOLERANCE` is 1e-5, about float32 rounding on cosine scores; it is also the
tolerance of the store tests.
"""

from __future__ import annotations

from dataclasses import dataclass

SCORE_TOLERANCE = 1e-5


@dataclass(frozen=True)
class Comparison:
    identical: bool
    max_score_difference: float  # over ranks where both lists have the same chunk
    first_difference: int | None  # 0-based rank, or None if identical
    is_tie: bool | None  # None when identical
    reason: str  # why it is, or is not, called a tie


def compare_results(chroma, faiss, tolerance: float = SCORE_TOLERANCE) -> Comparison:
    ids_a, ids_b = [r.id for r in chroma], [r.id for r in faiss]
    worst = max((abs(a.score - b.score) for a, b in zip(chroma, faiss) if a.id == b.id), default=0.0)
    if ids_a == ids_b:
        return Comparison(True, worst, None, None, "identical")

    first = next((i for i, (a, b) in enumerate(zip(ids_a, ids_b)) if a != b), min(len(ids_a), len(ids_b)))
    if first >= len(ids_a) or first >= len(ids_b):
        return Comparison(False, worst, first, False, "the lists have different lengths")

    x, y = ids_a[first], ids_b[first]
    by_chroma = {r.id: r.score for r in chroma}
    by_faiss = {r.id: r.score for r in faiss}
    if not (x in by_faiss and y in by_chroma):
        missing = [name for name, found in ((x, x in by_faiss), (y, y in by_chroma)) if not found]
        return Comparison(False, worst, first, False,
                          f"not shown to be a tie: {missing} is missing from the other top-k list")
    gap_chroma = abs(by_chroma[x] - by_chroma[y])
    gap_faiss = abs(by_faiss[x] - by_faiss[y])
    if gap_chroma <= tolerance and gap_faiss <= tolerance:
        return Comparison(False, worst, first, True,
                          f"tie: {x} and {y} differ by {gap_chroma:.2e} (Chroma) and {gap_faiss:.2e} (FAISS)")
    return Comparison(False, worst, first, False,
                      f"not a tie: {x} and {y} differ by {gap_chroma:.2e} (Chroma) and {gap_faiss:.2e} (FAISS)")