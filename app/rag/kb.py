"""Knowledge-base section IDs: the heading rule shared by the chunker and the citation checker.

A section is a level-2 Markdown heading (`## Customer Returns`). Its ID is
`<file>.md#<slugify_heading(heading)>`, e.g. `returns_and_expiry_policy.md#Customer-Returns`.
See docs/knowledge_base/README.md for the full rule.
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


def expected_sources(test_set: list[dict]) -> list[str]:
    """Collect every `expected_sources` entry, de-duplicated, in first-seen order."""
    seen: dict[str, None] = {}
    for question in test_set:
        for source in question.get("expected_sources") or []:
            seen.setdefault(source, None)
    return list(seen)


def check_sources(kb_dir: Path, sources: list[str]) -> list[str]:
    """Return one problem string per source that cannot be found; empty means all good.

    Matching is exact and case-sensitive: the heading's slug must equal the
    fragment after '#'.
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
        slugs = [slugify_heading(h) for h in section_headings(path.read_text(encoding="utf-8"))]
        if fragment not in slugs:
            found = ", ".join(slugs) if slugs else "none"
            problems.append(f"{source} (section not found; sections in file: {found})")
    return problems