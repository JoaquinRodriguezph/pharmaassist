"""Deterministic checks around the model.

check_input runs before any model call and catches prompt injection, requests to reveal the
instructions, and requests to change data. leaks_internals runs on the final answer and
catches tool definitions or system-prompt text copied into it. Patterns are kept narrow; a
test checks that no test-set question outside the "refuse" route is caught.
"""

from __future__ import annotations

import re

_INPUT_PATTERNS: tuple[tuple[str, re.Pattern], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE | re.DOTALL))
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
    )
)

_LEAK_MARKERS = ('"type": "function"', "Rules, in order of priority", "You are PharmaAssist")


def check_input(text: str) -> str | None:
    """The name of the first pattern the text matches, or None."""
    for name, pattern in _INPUT_PATTERNS:
        if pattern.search(text):
            return name
    return None


def leaks_internals(text: str, tool_descriptions: list[str]) -> bool:
    """True if the answer contains system-prompt markers or a tool description."""
    if any(marker in text for marker in _LEAK_MARKERS):
        return True
    return any(description[:80] in text for description in tool_descriptions if description)