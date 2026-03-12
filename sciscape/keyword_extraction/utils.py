"""Shared utilities for the keyword extraction package.

Contains functions and constants used across multiple submodules,
preventing circular imports and code duplication.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import MutableMapping

# Shared type aliases
TokenCounter = Counter[str]
YearCounter = Counter[int]
TermYearCounter = MutableMapping[str, YearCounter]
ClusterTermYearCounter = MutableMapping[int, TermYearCounter]

# Shared regex
_HTML_TAG_RE = re.compile(r"</?[^<>]+>")


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings.

    Includes early termination when the length difference alone
    exceeds a practical threshold (3).
    """
    if abs(len(a) - len(b)) > 3:
        return abs(len(a) - len(b))
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[len(b)]


def _normalize_text_basic(text: object) -> str:
    """Cheap normalisation shared across stages."""
    if not isinstance(text, str):
        return ""
    text = _HTML_TAG_RE.sub(" ", text)
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())
