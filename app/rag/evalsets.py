"""Loading and checking the question sets used to evaluate retrieval.

- **original:** the RAG questions in `eval/test_set.json`. Those with
  `expected_sources` are measured by rank; Q09, Q10 and Q20 (no expected
  source) are reported by score only.
- **variants:** `eval/retrieval_variants.json`, reworded versions of the original
  questions. Each names its `base_id` and must have exactly the same expected
  sources as that question.
- **independent:** `eval/retrieval_independent.json`, questions written by someone
  who had not seen the documents, labeled afterwards. An empty `expected_sources`
  means the knowledge base cannot answer it. A question with no
  `expected_sources` key at all is not labeled yet and is an error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ORIGINAL_SCORE_ONLY_IDS = ("Q09", "Q10", "Q20")


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    expected: tuple[str, ...]  # empty: the knowledge base cannot answer it
    group: str  # "original", "variants" or "independent"
    style: str | None = None  # how a variant was reworded
    base_id: str | None = None  # the original question a variant rewords


def split_answerable(questions: list[EvalQuestion]) -> tuple[list[EvalQuestion], list[EvalQuestion]]:
    """(questions with an expected section, questions without one)."""
    return [q for q in questions if q.expected], [q for q in questions if not q.expected]


def original_questions(test_set: list[dict]) -> list[EvalQuestion]:
    """The questions with expected sources, plus Q09, Q10 and Q20, from the test set."""
    by_id = {q["id"]: q for q in test_set}
    missing = [i for i in ORIGINAL_SCORE_ONLY_IDS if i not in by_id]
    if missing:
        raise ValueError(f"the test set has no question(s) {missing}")
    chosen = [q for q in test_set if q.get("expected_sources") or q["id"] in ORIGINAL_SCORE_ONLY_IDS]
    return [
        EvalQuestion(q["id"], q["question"], tuple(q.get("expected_sources") or ()), "original")
        for q in chosen
    ]


def _load_list(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{Path(path).name}: expected a JSON list of questions")
    return data


def _require_text(item: dict, field: str, where: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: '{field}' must be a non-empty string")
    return value


def _check_unique(ids: list[str], name: str) -> None:
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"{name}: duplicate id(s) {duplicates}")


def load_variants(path: Path, test_set: list[dict]) -> list[EvalQuestion]:
    """Reworded questions. Each must name an existing `base_id` and repeat its expected sources exactly."""
    base = {q["id"]: q for q in test_set}
    questions = []
    for item in _load_list(path):
        where = f"{Path(path).name}: {item.get('id', '?')}"
        qid = _require_text(item, "id", where)
        base_id = _require_text(item, "base_id", where)
        text = _require_text(item, "question", where)
        style = _require_text(item, "style", where)
        _require_text(item, "written_by", where)
        if base_id not in base:
            raise ValueError(f"{where}: base_id {base_id!r} is not in the test set")
        expected = item.get("expected_sources")
        if not isinstance(expected, list) or sorted(expected) != sorted(base[base_id].get("expected_sources") or []):
            raise ValueError(f"{where}: expected_sources must equal those of {base_id}")
        if text.strip() == base[base_id]["question"].strip():
            raise ValueError(f"{where}: the variant is identical to {base_id}")
        questions.append(EvalQuestion(qid, text, tuple(expected), "variants", style, base_id))
    _check_unique([q.id for q in questions], Path(path).name)
    return questions


def load_independent(path: Path) -> list[EvalQuestion]:
    """Independently written questions. Every one must already be labeled with `expected_sources`."""
    questions = []
    for item in _load_list(path):
        where = f"{Path(path).name}: {item.get('id', '?')}"
        qid = _require_text(item, "id", where)
        text = _require_text(item, "question", where)
        _require_text(item, "written_by", where)
        if "expected_sources" not in item:
            raise ValueError(f"{where}: not labeled yet (no 'expected_sources'; use [] if the KB cannot answer it)")
        expected = item["expected_sources"]
        if not isinstance(expected, list) or not all(isinstance(s, str) and s for s in expected):
            raise ValueError(f"{where}: expected_sources must be a list of section ids")
        questions.append(EvalQuestion(qid, text, tuple(expected), "independent"))
    _check_unique([q.id for q in questions], Path(path).name)
    return questions