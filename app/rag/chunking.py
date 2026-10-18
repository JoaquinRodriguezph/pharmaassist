"""Turn knowledge-base documents into chunks for retrieval.

Rules (see docs/chunking_notes.md for the reasoning):

- **What is chunked:** only knowledge-base documents (`kb.kb_documents`), so
  README.md is never indexed. Only the text under `##` headings is chunked.
  The *preamble* (banner, title, document ID) is not indexed; the title is kept
  only as `Chunk.doc_title`, for the optional context prefix in `embed_text`.
- **One chunk per section** when the section's text fits in `max_chars`.
- **Longer sections** are split at paragraph and list-item boundaries first,
  then at sentence boundaries. A sentence is cut only if it alone exceeds
  `max_chars`: then at word boundaries, and a single word longer than
  `max_chars` is cut at exactly `max_chars`. Every part is at most `max_chars`
  characters (the limit applies to `text`, not to the optional prefix).
- **Overlap** is a character budget for whole sentences: a part after the first
  starts with the last whole sentence(s) of the previous part, as many as fit
  in `overlap` characters, never the whole previous part. Carried sentences
  count toward the new part's `max_chars`. It applies only between parts of
  the same section. `overlap=0` means none.
- **IDs:** `<file>.md#<Fragment>::<n>`, n counting from 1 within the section.
  Two headings in one file with the same fragment raise ValueError.

The sentence splitter is a simple heuristic (a sentence ends at `.`, `!` or `?`
followed by whitespace and a capital letter, digit, quote, bracket or list
marker), so an abbreviation such as "e.g. Paracetamol" would be split.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from app.rag.kb import document_title, kb_documents, preamble, split_sections

DEFAULT_MAX_CHARS = 800
DEFAULT_OVERLAP = 0

_PARAGRAPH_BREAK = re.compile(r"(\n[ \t]*\n\s*)")
_LIST_ITEM = re.compile(r"^[ \t]*(?:[-*]|\d+\.)[ \t]+")
_NUMBER_MARKER = re.compile(r"(?m)^([ \t]*)(\d+)\.(?=[ \t])")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\u201c(\x00]|[-*][ \t])")


@dataclass(frozen=True)
class Chunk:
    id: str  # "<source>#<Fragment>::<n>": unique, and the same on every run for the same files
    text: str  # what is stored and shown to the model; never includes the context prefix
    source: str  # file name, e.g. "storage_sop.md"
    section: str  # readable heading, e.g. "Cold Chain And Excursions"
    section_id: str  # "<source>#<Fragment>": the citation form used by the test set
    doc_title: str  # the document's level-1 heading (only used for the context prefix)
    part: int  # 1-based position within the section
    n_parts: int  # how many chunks the section was split into

    def embed_text(self, with_context: bool = True) -> str:
        """The text to embed. With context, the document title and section heading come first."""
        if not with_context:
            return self.text
        return f"Document: {self.doc_title}\nSection: {self.section}\n\n{self.text}"

    @property
    def text_hash(self) -> str:
        """Short fingerprint of `text`, for noticing that a chunk's text changed."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Splitting one section's text
# ---------------------------------------------------------------------------

def _units(text: str) -> list[tuple[str, str]]:
    """Split into (separator before, unit): paragraphs, with each list item its own unit."""
    units: list[tuple[str, str]] = []
    separator = ""
    for k, piece in enumerate(_PARAGRAPH_BREAK.split(text)):
        if k % 2 == 1:  # the captured paragraph break
            separator = piece
            continue
        groups: list[list[str]] = []
        for line in piece.split("\n"):
            if not groups or _LIST_ITEM.match(line):
                groups.append([line])
            else:
                groups[-1].append(line)
        for j, lines in enumerate(groups):
            unit = "\n".join(lines).strip()
            if unit:
                units.append((separator if j == 0 else "\n", unit))
        separator = ""
    return units


def _sentences(text: str) -> list[str]:
    """Split into sentences, keeping a numbered list marker ("1.") with its sentence."""
    protected = _NUMBER_MARKER.sub(lambda m: f"{m.group(1)}{m.group(2)}\x00", text)
    parts = _SENTENCE_END.split(protected)
    return [p.replace("\x00", ".").strip() for p in parts if p.strip()]


def _split_words(sentence: str, max_chars: int) -> list[str]:
    """Cut a sentence that is too long at word boundaries; cut a too-long word at max_chars."""
    if len(sentence) <= max_chars:
        return [sentence]
    pieces: list[str] = []
    current = ""
    for word in sentence.split():
        while len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_chars:
            current += " " + word
        else:
            pieces.append(current)
            current = word
    if current:
        pieces.append(current)
    return pieces


def _fit(unit: str, max_chars: int) -> list[str]:
    """Return the unit whole if it fits, otherwise its sentences (cut further only if needed)."""
    if len(unit) <= max_chars:
        return [unit]
    pieces: list[str] = []
    for sentence in _sentences(unit):
        pieces.extend(_split_words(sentence, max_chars))
    return pieces


def _tail(part: str, overlap: int) -> list[str]:
    """The last whole sentences of `part` that fit in `overlap` characters; never the whole part."""
    if overlap <= 0:
        return []
    tail: list[str] = []
    total = 0
    for sentence in reversed(_sentences(part)[1:]):
        added = len(sentence) + (1 if tail else 0)
        if total + added > overlap:
            break
        tail.insert(0, sentence)
        total += added
    return tail


def _join(sentences: list[str]) -> str:
    """Join sentences with a space, but keep each list item on its own line."""
    joined = ""
    for sentence in sentences:
        if not joined:
            joined = sentence
        else:
            joined += ("\n" if _LIST_ITEM.match(sentence) else " ") + sentence
    return joined


def _start_part(tail: list[str], piece: str, max_chars: int) -> str:
    """Begin a new part with as much of the carried tail as still fits with `piece`.

    A list item keeps its own line, so the carried sentence ends its line first.
    """
    separator = "\n" if _LIST_ITEM.match(piece) else " "
    while tail:
        candidate = _join(tail) + separator + piece
        if len(candidate) <= max_chars:
            return candidate
        tail = tail[1:]
    return piece


def split_text(text: str, max_chars: int, overlap: int = 0) -> list[str]:
    """Split one section's text into parts of at most `max_chars` characters (see module docstring)."""
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be at least 0 and less than max_chars")
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    pieces: list[tuple[str, str]] = []
    for separator, unit in _units(text):
        for i, piece in enumerate(_fit(unit, max_chars)):
            pieces.append((separator if i == 0 else " ", piece))

    parts: list[str] = []
    current = ""
    for separator, piece in pieces:
        if not current:
            current = piece
        elif len(current) + len(separator) + len(piece) <= max_chars:
            current += separator + piece
        else:
            parts.append(current)
            current = _start_part(_tail(current, overlap), piece, max_chars)
    parts.append(current)
    return [part.strip() for part in parts]


# ---------------------------------------------------------------------------
# Chunking the knowledge base
# ---------------------------------------------------------------------------

def chunk_kb(
    kb_dir: Path, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP
) -> list[Chunk]:
    """Chunk every knowledge-base document in `kb_dir`, in file order, then section order.

    A section with no text produces no chunk. Raises ValueError if two headings
    in one file give the same fragment, because their chunk ids would collide.
    """
    chunks: list[Chunk] = []
    for path in kb_documents(kb_dir):
        markdown = path.read_text(encoding="utf-8")
        title = document_title(markdown) or path.stem
        headings_by_id: dict[str, str] = {}
        for section in split_sections(markdown):
            section_id = f"{path.name}#{section.fragment}"
            if section_id in headings_by_id:
                raise ValueError(
                    f"Duplicate section id {section_id!r}: the headings "
                    f"{headings_by_id[section_id]!r} and {section.heading!r} give the same fragment"
                )
            headings_by_id[section_id] = section.heading
            parts = split_text(section.body, max_chars, overlap)
            for n, text in enumerate(parts, start=1):
                chunks.append(
                    Chunk(
                        id=f"{section_id}::{n}",
                        text=text,
                        source=path.name,
                        section=section.heading,
                        section_id=section_id,
                        doc_title=title,
                        part=n,
                        n_parts=len(parts),
                    )
                )
    return chunks


def stray_preamble_lines(markdown: str) -> list[str]:
    """Preamble lines that are not blank, a blockquote (banner), the title or a "Document ID:" line.

    The preamble is never indexed, so text here would be silently lost; the
    stats script warns about it.
    """
    stray = []
    for line in preamble(markdown).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(">") or re.match(r"#[ \t]", stripped):
            continue
        if stripped.lower().startswith("document id:"):
            continue
        stray.append(stripped)
    return stray