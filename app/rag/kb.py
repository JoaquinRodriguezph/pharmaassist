"""Knowledge-base rules shared by the chunker and the citation checker.

- **Documents:** a knowledge-base document is a `*.md` file directly inside the
  KB folder whose name is not `README.md` (any case) and does not start with
  `_` or `.` (`is_kb_document`, `kb_documents`).
- **Sections:** a section is a level-2 Markdown heading (`## Customer Returns`)
  and the text under it up to the next `##` (`split_sections`). Text before the
  first `##` is the *preamble* (banner, title); it is not part of any section.
- **IDs:** a section's ID is `<file>.md#<slugify_heading(heading)>`, e.g.
  `returns_and_expiry_policy.md#Customer-Returns`.
- **Required strings:** a question's `must_contain` strings are checked against
  the section's text.

See docs/knowledge_base/README.md for the full rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Exactly two hashes then whitespace, so '#' (title) and '###' (subsection) never match.
_H2 = re.compile(r"^##[ \t]+(.+?)[ \t]*$", re.MULTILINE)
# Exactly one hash then whitespace: the document title.
_H1 = re.compile(r"^#[ \t]+(.+?)[ \t]*$", re.MULTILINE)

# File names (lower case) that live in the KB folder but are not documents.
NON_DOCUMENTS = {"readme.md"}


# ---------------------------------------------------------------------------
# Which files are documents
# ---------------------------------------------------------------------------

def is_kb_document(path: Path) -> bool:
    """True for a `.md` file that is not README.md and does not start with `_` or `.`."""
    name = Path(path).name
    return (
        name.lower().endswith(".md")
        and name.lower() not in NON_DOCUMENTS
        and not name.startswith(("_", "."))
    )


def kb_documents(kb_dir: Path) -> list[Path]:
    """The documents directly inside `kb_dir`, sorted by name (not recursive)."""
    return sorted(p for p in Path(kb_dir).iterdir() if p.is_file() and is_kb_document(p))


# ---------------------------------------------------------------------------
# Headings and sections
# ---------------------------------------------------------------------------

def slugify_heading(heading: str) -> str:
    """Turn heading text into its citation fragment, keeping capitalization.

    Punctuation other than hyphens is dropped, and every run of whitespace or
    hyphens becomes one hyphen: "Cold Chain And Excursions" -> "Cold-Chain-And-Excursions".
    """
    cleaned = re.sub(r"[^\w\s-]", "", heading)
    return re.sub(r"[\s-]+", "-", cleaned).strip("-")


@dataclass(frozen=True)
class Section:
    heading: str  # readable text, e.g. "Cold Chain And Excursions"
    fragment: str  # slugify_heading(heading), e.g. "Cold-Chain-And-Excursions"
    body: str  # text after the heading line, up to the next `##` (heading line not included)


def split_sections(markdown: str) -> list[Section]:
    """Split a document into its `##` sections, in order. The one splitter everything uses."""
    headings = list(_H2.finditer(markdown))
    sections = []
    for i, match in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(markdown)
        heading = match.group(1)
        sections.append(Section(heading, slugify_heading(heading), markdown[match.end() : end]))
    return sections


def preamble(markdown: str) -> str:
    """Text before the first `##` heading (banner, title, document ID). Never indexed."""
    first = _H2.search(markdown)
    return markdown if first is None else markdown[: first.start()]


def document_title(markdown: str) -> str | None:
    """The first level-1 heading in the preamble, or None."""
    match = _H1.search(preamble(markdown))
    return match.group(1) if match else None


def section_headings(markdown: str) -> list[str]:
    """Return the text of every level-2 heading, in order."""
    return [section.heading for section in split_sections(markdown)]


def section_body(markdown: str, fragment: str) -> str | None:
    """Return the text under the heading whose fragment matches, up to the next `##`.

    The heading line itself is left out, so a string that appears only in the
    heading (e.g. "FEFO" in "Receiving And FEFO") does not count as present.
    Returns None if no heading matches.
    """
    for section in split_sections(markdown):
        if section.fragment == fragment:
            return section.body
    return None


def normalize_for_match(text: str) -> str:
    """Case-insensitive plain text, with any run of whitespace (even a line break) as one space."""
    return " ".join(text.split()).casefold()


# ---------------------------------------------------------------------------
# The test set's expected sources and required strings
# ---------------------------------------------------------------------------

def expected_sources(test_set: list[dict]) -> list[str]:
    """Collect every `expected_sources` entry, de-duplicated, in first-seen order."""
    seen: dict[str, None] = {}
    for question in test_set:
        for source in question.get("expected_sources") or []:
            seen.setdefault(source, None)
    return list(seen)


def question_requirements(question: dict) -> dict[str, list[str]]:
    """Map each of one question's cited sources to the strings its section must contain.

    `must_contain` is either a list (every one of the question's
    `expected_sources` must contain all of them) or an object mapping a source
    to its own list (for questions citing several sections). Returns {} when
    the question has no `must_contain`. Raises ValueError for a malformed field,
    or an object key that is not one of the question's sources, because a typo
    there would otherwise silently skip a check.
    """
    must = question.get("must_contain")
    if not must:
        return {}
    qid = question.get("id", "?")
    sources = question.get("expected_sources") or []
    if isinstance(must, list):
        per_source = {source: must for source in sources}
    elif isinstance(must, dict):
        unknown = [key for key in must if key not in sources]
        if unknown:
            raise ValueError(f"{qid}: must_contain key(s) not in expected_sources: {unknown}")
        per_source = must
    else:
        raise ValueError(f"{qid}: must_contain must be a list or an object")
    for source, strings in per_source.items():
        if not isinstance(strings, list) or not all(isinstance(s, str) and s for s in strings):
            raise ValueError(f"{qid}: must_contain for {source} must be a list of non-empty strings")
    return {source: list(strings) for source, strings in per_source.items()}


def required_strings(test_set: list[dict]) -> dict[str, list[str]]:
    """Map each cited source to the strings its section text must contain, across all questions.

    Strings for the same source from different questions are merged, without duplicates.
    """
    required: dict[str, list[str]] = {}
    for question in test_set:
        for source, strings in question_requirements(question).items():
            bucket = required.setdefault(source, [])
            bucket.extend(s for s in strings if s not in bucket)
    return required


def check_sources(
    kb_dir: Path, sources: list[str], required: dict[str, list[str]] | None = None
) -> list[str]:
    """Return one problem string per missing file, section or required string; empty means all good.

    Heading matching is exact and case-sensitive: the heading's slug must equal
    the fragment after '#'. Required strings (from `required_strings`) are matched
    case-insensitively against the section's text. Only knowledge-base documents
    can be cited: README.md and files starting with `_` or `.` are not.
    """
    problems: list[str] = []
    for source in sources:
        filename, sep, fragment = source.partition("#")
        if not sep or not filename or not fragment:
            problems.append(f"{source} (not in the form file.md#Section)")
            continue
        path = Path(kb_dir) / filename
        if not path.is_file():
            problems.append(f"{source} (file not found)")
            continue
        if not is_kb_document(path):
            problems.append(f"{source} (not a knowledge-base document; README.md and files starting with '_' or '.' are excluded)")
            continue
        text = path.read_text(encoding="utf-8")
        sections = split_sections(text)
        slugs = [section.fragment for section in sections]
        if fragment not in slugs:
            found = ", ".join(slugs) if slugs else "none"
            problems.append(f"{source} (section not found; sections in file: {found})")
            continue
        body = normalize_for_match(section_body(text, fragment) or "")
        for needed in (required or {}).get(source, []):
            if normalize_for_match(needed) not in body:
                problems.append(f"{source} (section text is missing {needed!r})")
    return problems