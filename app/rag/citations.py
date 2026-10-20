"""Reading citations and the fixed "not covered" sentence out of a model's answer.

Pure functions: no model, no store. The rules are in docs/answer_notes.md, sections 2 and 3.

- **Citations:** every `[...]` with no bracket or newline inside is a candidate. Inside one
  bracket, commas and semicolons separate several citations. Each piece is either a
  **valid** citation (a section id that was retrieved), an **invalid** one (with a
  reason), or not a citation at all (`[sic]`), which is ignored.
- **The fixed sentence:** `REFUSAL_SENTENCE`. `is_refusal` means the whole answer is that
  sentence; `says_not_covered` means it appears anywhere. A reworded refusal matches
  neither, on purpose, so it can be counted as not following the instruction.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from app.rag.kb import kb_documents, split_sections

REFUSAL_SENTENCE = "The provided sources do not answer this question."

# A section id: file name, '#', fragment. Fragments come from kb.slugify_heading, which
# keeps only letters, digits, '_' and '-'. A test checks every real id has this shape.
SECTION_ID_SHAPE = re.compile(r"[A-Za-z0-9_.-]+\.md#[A-Za-z0-9_-]+")
_CHUNK_ID_SHAPE = re.compile(r"[A-Za-z0-9_.-]+\.md#[A-Za-z0-9_-]+::\d+")
_NUMBER = re.compile(r"\d+")
_BRACKET = re.compile(r"\[([^\[\]\n]{1,200})\]")
_SEPARATORS = re.compile(r"[,;]")

NOT_RETRIEVED = "not retrieved"
NOT_IN_KB = "not in the knowledge base"
CHUNK_ID = "chunk id, not section id"
NUMBERED = "numbered citation"
MALFORMED = "malformed"


@dataclass(frozen=True)
class InvalidCitation:
    raw: str  # the piece as the model wrote it, spaces trimmed
    reason: str  # one of the reason constants above


@dataclass(frozen=True)
class Citations:
    valid: tuple[str, ...]  # retrieved section ids, deduplicated, in order of first use
    invalid: tuple[InvalidCitation, ...]  # deduplicated, in order of first use


def kb_section_ids(kb_dir: Path) -> set[str]:
    """Every `file.md#Fragment` in the knowledge base, using the same document and section rules as the chunker."""
    return {
        f"{path.name}#{section.fragment}"
        for path in kb_documents(kb_dir)
        for section in split_sections(path.read_text(encoding="utf-8"))
    }


def _classify(piece: str, retrieved: Collection[str], known: Collection[str]) -> str | None:
    """'valid', an invalid reason, or None if the piece is not a citation."""
    if SECTION_ID_SHAPE.fullmatch(piece):
        if piece in retrieved:
            return "valid"
        return NOT_RETRIEVED if piece in known else NOT_IN_KB
    if _CHUNK_ID_SHAPE.fullmatch(piece):
        return CHUNK_ID
    if _NUMBER.fullmatch(piece):
        return NUMBERED
    if ".md" in piece:
        return MALFORMED
    return None


def parse_citations(text: str, retrieved_ids: Collection[str], known_ids: Collection[str]) -> Citations:
    """Split the bracketed citations in `text` into valid and invalid ones.

    `retrieved_ids` are the section ids shown to the model; `known_ids` are all section
    ids in the knowledge base (`kb_section_ids`), used only to explain an invalid one.
    Matching is exact and case-sensitive. The text is not changed.
    """
    valid: dict[str, None] = {}
    invalid: dict[InvalidCitation, None] = {}
    for bracket in _BRACKET.finditer(text):
        for piece in _SEPARATORS.split(bracket.group(1)):
            piece = piece.strip()
            if not piece:
                continue
            kind = _classify(piece, retrieved_ids, known_ids)
            if kind == "valid":
                valid.setdefault(piece, None)
            elif kind is not None:
                invalid.setdefault(InvalidCitation(piece, kind), None)
    return Citations(tuple(valid), tuple(invalid))


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


_CORE = _normalize(REFUSAL_SENTENCE).removesuffix(".")


def is_refusal(text: str) -> bool:
    """True if the whole answer is the fixed sentence (case, spacing and the final period aside)."""
    return _normalize(text).removesuffix(".") == _CORE


def says_not_covered(text: str) -> bool:
    """True if the fixed sentence appears anywhere: a pure refusal or a partial answer."""
    return _CORE in _normalize(text)