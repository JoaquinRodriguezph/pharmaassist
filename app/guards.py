"""Deterministic checks around the model.

check_input runs before any model call and catches prompt injection, requests to reveal the
instructions, requests to change data, and clearly off-topic work (writing code, scraping,
hacking). is_personal_medical flags questions about a specific person's dose, symptoms or
medicines. leaks_internals runs on the final answer. Patterns are kept narrow; tests check
them against every test-set question.
"""

from __future__ import annotations

import re

_FLAGS = re.IGNORECASE | re.DOTALL

_INPUT_PATTERNS: tuple[tuple[str, re.Pattern], ...] = tuple(
    (name, re.compile(pattern, _FLAGS))
    for name, pattern in (
        ("ignore_instructions",
         r"\bignore\s+(all\s+|any\s+)?(of\s+)?(your|the|my|these|previous|prior|above)\b.{0,30}\b(instructions|rules|prompt)\b"),
        ("disregard_rules", r"\bdisregard\b.{0,40}\b(instructions|rules|safety)\b"),
        ("system_override", r"\bsystem\s+(override|instruction|prompt)\b"),
        ("special_mode", r"\b(developer|maintenance|debug|god)\s+mode\b"),
        ("reveal_prompt",
         r"\b(print|show|reveal|repeat|output|display|dump)\b.{0,40}\b(system\s+prompt|your\s+(instructions|rules|prompt)|hidden\s+(instructions|rules))\b"),
        ("schema_change", r"\b(drop|truncate|alter)\b.{0,30}\b(table|database|schema)\b"),
        ("bulk_delete", r"\bdelete\s+(all|every|the)\b.{0,30}\b(rows|records|batches|stock|table|data)\b"),
        ("data_change", r"\b(set|change|update|make)\b.{0,60}\b(quantit(y|ies)|stock|prices?)\b.{0,20}\bto\s+\d"),
        ("write_code",
         r"\b(write|create|build|generate|make|code)\b.{0,40}\b(script|program|scraper|bot|macro|sql\s+query|regex|website)\b"),
        ("scraping", r"\bscrap(e|es|ed|ing|er|ers)\b"),
        ("hacking", r"\b(hack|exploit|crack)\b"),
    )
)

_MEDICAL_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(pattern, _FLAGS)
    for pattern in (
        r"\b(can|should|may|could)\s+(i|we|she|he|they|my\s+\w+)\s+(take|give|use|switch|combine|stop|double)\b",
        r"\bhow\s+(many|much)\b.{0,60}\b(should|can|do|to)\s+(i\s+|we\s+)?(give|take)\b",
        r"\bmy\s+(\d+\s*(yo|y/o|-?\s*years?[- ]old)\s+)?(son|daughter|baby|child|kid|wife|husband|mother|father|mom|dad|grandmother|grandfather)\b",
        r"\bshould\s+i\s+tell\s+(them|him|her)\s+to\s+(switch|stop|take|change)\b",
    )
)

_LEAK_MARKERS = ('"type": "function"', "You are PharmaAssist")


def check_input(text: str) -> str | None:
    """The name of the first injection, write or off-topic pattern the text matches, or None."""
    for name, pattern in _INPUT_PATTERNS:
        if pattern.search(text):
            return name
    return None


def is_personal_medical(text: str) -> bool:
    """True for a question about a specific person's dose, symptoms or medicines."""
    return any(pattern.search(text) for pattern in _MEDICAL_PATTERNS)


def leaks_internals(text: str, fragments: list[str]) -> bool:
    """True if the answer contains a leak marker or any of the given fragments
    (tool descriptions, sentences of the system prompt)."""
    if any(marker in text for marker in _LEAK_MARKERS):
        return True
    return any(fragment[:80] in text for fragment in fragments if fragment)