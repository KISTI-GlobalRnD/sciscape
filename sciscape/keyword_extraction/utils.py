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


def _normalize_text_basic(text: object) -> str:
    """Cheap normalisation shared across stages."""
    if not isinstance(text, str):
        return ""
    text = _HTML_TAG_RE.sub(" ", text)
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())
