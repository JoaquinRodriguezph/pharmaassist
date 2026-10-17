"""Knowledge-base section IDs: the heading rule shared by the chunker and the citation checker.

A section is a level-2 Markdown heading (`## Customer Returns`). Its ID is
`<file>.md#<slugify_heading(heading)>`, e.g. `returns_and_expiry_policy.md#Customer-Returns`.
A question's `must_contain` strings are checked against the section's text.
See docs/knowledge_base/README.md for the full rules.
"""

from __future__ import annotations

import re
from pathlib import Path

# Exactly two hashes then whitespace, so '#' (title) and '###' (subsection) never match.
_H2 = re.compile(r"^##[ \t]+(.+?)[ \t]*$", re.MULTILINE)


def slugify_heading(heading: str) -> str:
    """Turn heading text into its citation fragment, keeping capitalization.

    Punctuation other than hyphens is dropped, and every run of whitespace or
    hyphens becomes one hyphen: "Cold Chain And Excursions" -> "Cold-Chain-And-Excursions".
    """
    cleaned = re.sub(r"[^\w\s-]", "", heading)
    return re.sub(r"[\s-]+", "-", cleaned).strip("-")


def section_headings(markdown: str) -> list[str]:
    """Return the text of every level-2 heading, in order."""
    return _H2.findall(markdown)


def section_body(markdown: str, fragment: str) -> str | None:
    """Return the text under the heading whose fragment matches, up to the next `##`.

    The heading line itself is left out, so a string that appears only in the
    heading (e.g. "FEFO" in "Receiving And FEFO") does not count as present.
    Returns None if no heading matches.
    """
    headings = list(_H2.finditer(markdown))
    for i, heading in enumerate(headings):
        if slugify_heading(heading.group(1)) == fragment:
            end = headings[i + 1].start() if i + 1 < len(headings) else len(markdown)
            return markdown[heading.end() : end]
    return None


def _normalize(text: str) -> str:
    """Case-insensitive plain text, with any run of whitespace (even a line break) as one space."""
    return " ".join(text.split()).casefold()


def expected_sources(test_set: list[dict]) -> list[str]:
    """Collect every `expected_sources` entry, de-duplicated, in first-seen order."""
    seen: dict[str, None] = {}
    for question in test_set:
        for source in question.get("expected_sources") or []:
            seen.setdefault(source, None)
    return list(seen)


def required_strings(test_set: list[dict]) -> dict[str, list[str]]:
    """Map each cited source to the strings its section text must contain.

    A question's `must_contain` is either a list (every one of its
    `expected_sources` must contain all of them) or an object mapping a source
    to its own list (for questions citing several sections). Strings for the
    same source from different questions are merged. Raises ValueError for a
    malformed field, or an object key that is not one of the question's sources,
    because a typo there would otherwise silently skip a check.
    """
    required: dict[str, list[str]] = {}
    for question in test_set:
        must = question.get("must_contain")
        if not must:
            continue
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
            bucket = required.setdefault(source, [])
            bucket.extend(s for s in strings if s not in bucket)
    return required


def check_sources(
    kb_dir: Path, sources: list[str], required: dict[str, list[str]] | None = None
) -> list[str]:
    """Return one problem string per missing file, section or required string; empty means all good.

    Heading matching is exact and case-sensitive: the heading's slug must equal
    the fragment after '#'. Required strings (from `required_strings`) are matched
    case-insensitively against the section's text.
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
        text = path.read_text(encoding="utf-8")
        slugs = [slugify_heading(h) for h in section_headings(text)]
        if fragment not in slugs:
            found = ", ".join(slugs) if slugs else "none"
            problems.append(f"{source} (section not found; sections in file: {found})")
            continue
        body = _normalize(section_body(text, fragment) or "")
        for needed in (required or {}).get(source, []):
            if _normalize(needed) not in body:
                problems.append(f"{source} (section text is missing {needed!r})")
    return problems