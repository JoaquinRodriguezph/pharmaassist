"""Check that an answer contains the strings a test-set question requires. Pure function."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_DASHES = re.compile(r"[\u2010-\u2015\u2212]")
_AROUND_DASH = re.compile(r"\s*-\s*")
_SPACE_BEFORE_DEGREE = re.compile(r"\s+(?=\u00b0)")


def normalize(text: str) -> str:
    """Casefold, unify dashes, drop spaces around dashes and before the degree sign, collapse whitespace."""
    text = _DASHES.sub("-", text.casefold())
    text = _AROUND_DASH.sub("-", text)
    text = _SPACE_BEFORE_DEGREE.sub("", text)
    return " ".join(text.split())


@dataclass(frozen=True)
class MustContainResult:
    found: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.missing


def check_must_contain(text: str, required: Iterable[str]) -> MustContainResult:
    """Each required string must appear in `text` after normalizing both."""
    haystack = normalize(text)
    found, missing = [], []
    for item in required:
        (found if normalize(item) in haystack else missing).append(item)
    return MustContainResult(tuple(found), tuple(missing))