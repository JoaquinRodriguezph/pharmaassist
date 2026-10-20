"""Retrieval metrics on rankings of section ids. Pure functions: no model, no Chroma.

A **ranking** is the list of section ids of the retrieved chunks, best first
(`Retrieved.section_id` for each result). The same section id can appear more
than once, because a long section is split into several chunks (for example
`returns_and_expiry_policy.md#Customer-Returns::1` and `::2` both have the
section id `returns_and_expiry_policy.md#Customer-Returns`). **Expected** is the
set of section ids that answer the question (`expected_sources` in the test set).

Ranks are 1-based and count **chunks as retrieved**: a repeated chunk of a
section takes a slot, because those are the slots a language model would
receive. The rank of a section is the rank of its first chunk. To count distinct
sections instead, pass the ranking through `collapse_to_sections` first.

- `first_hit_rank`: the rank of the first chunk whose section is expected, or
  None if there is none.
- `hit_at_k`: True if at least one of the top k chunks is from an expected section.
- `all_hit_at_k`: True if every expected section appears in the top k chunks. It
  differs from `hit_at_k` only for a question with more than one expected section.
- `reciprocal_rank`: 1 / `first_hit_rank`, or 0.0 if there is no hit.
- `mean_reciprocal_rank` (MRR): the mean of the reciprocal ranks over questions.
- `duplicate_slots`: how many of the top k slots are taken by a chunk whose
  section already appeared higher up. It says how much of the top k a repeated
  section uses.
- `summarize`: all of the above over a list of (ranking, expected) pairs, as counts.

A question with no expected section (one the knowledge base cannot answer) has
no hit to measure, so these functions raise ValueError for an empty `expected`;
report such questions by score instead.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence


def _check(expected: Collection[str], k: int | None = None) -> None:
    if not expected:
        raise ValueError("expected must not be empty: a question the KB cannot answer has no hit to measure")
    if k is not None and k < 1:
        raise ValueError("k must be at least 1")


def collapse_to_sections(ranking: Sequence[str]) -> list[str]:
    """The ranking with repeats removed, keeping each section's first position."""
    seen: set[str] = set()
    collapsed = []
    for section_id in ranking:
        if section_id not in seen:
            seen.add(section_id)
            collapsed.append(section_id)
    return collapsed


def first_hit_rank(ranking: Sequence[str], expected: Collection[str]) -> int | None:
    """1-based rank of the first chunk from an expected section, or None."""
    _check(expected)
    wanted = set(expected)
    for rank, section_id in enumerate(ranking, start=1):
        if section_id in wanted:
            return rank
    return None


def hit_at_k(ranking: Sequence[str], expected: Collection[str], k: int) -> bool:
    """True if at least one of the top k chunks is from an expected section."""
    _check(expected, k)
    rank = first_hit_rank(ranking, expected)
    return rank is not None and rank <= k


def all_hit_at_k(ranking: Sequence[str], expected: Collection[str], k: int) -> bool:
    """True if every expected section appears among the top k chunks."""
    _check(expected, k)
    return set(expected) <= set(ranking[:k])


def reciprocal_rank(ranking: Sequence[str], expected: Collection[str]) -> float:
    """1 / rank of the first expected chunk, or 0.0 if none was retrieved."""
    rank = first_hit_rank(ranking, expected)
    return 0.0 if rank is None else 1.0 / rank


def mean_reciprocal_rank(cases: Sequence[tuple[Sequence[str], Collection[str]]]) -> float:
    """MRR over (ranking, expected) pairs. Raises ValueError for no cases."""
    if not cases:
        raise ValueError("MRR needs at least one question")
    return sum(reciprocal_rank(ranking, expected) for ranking, expected in cases) / len(cases)


def duplicate_slots(ranking: Sequence[str], k: int) -> int:
    """How many of the top k slots hold a chunk from a section already seen higher up."""
    if k < 1:
        raise ValueError("k must be at least 1")
    top = ranking[:k]
    return len(top) - len(set(top))


def summarize(
    cases: Sequence[tuple[Sequence[str], Collection[str]]], ks: Sequence[int] = (1, 3, 5)
) -> dict:
    """Counts and means over (ranking, expected) pairs, for a table.

    Returns `n` (questions), `hits` and `all_hits` (k -> how many questions),
    `mrr` (chunk ranks) and `mrr_distinct` (repeats collapsed first),
    `repeat_slots` (total slots taken by repeated chunks within the top
    `max(ks)`) and `questions_with_repeats`.
    """
    if not cases:
        raise ValueError("summarize needs at least one question")
    top = max(ks)
    duplicates = [duplicate_slots(ranking, top) for ranking, _ in cases]
    return {
        "n": len(cases),
        "hits": {k: sum(hit_at_k(r, e, k) for r, e in cases) for k in ks},
        "all_hits": {k: sum(all_hit_at_k(r, e, k) for r, e in cases) for k in ks},
        "mrr": mean_reciprocal_rank(cases),
        "mrr_distinct": mean_reciprocal_rank([(collapse_to_sections(r), e) for r, e in cases]),
        "repeat_slots": sum(duplicates),
        "questions_with_repeats": sum(1 for d in duplicates if d),
    }