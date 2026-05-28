"""Shared utilities for the keyword extraction package.

Contains functions and constants used across multiple submodules,
preventing circular imports and code duplication.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from typing import MutableMapping

# Shared type aliases
TokenCounter = Counter[str]
YearCounter = Counter[int]
TermYearCounter = MutableMapping[str, YearCounter]
ClusterTermYearCounter = MutableMapping[int, TermYearCounter]

# Shared regex
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_SCRIPT_STYLE_RE = re.compile(
    r"<\s*(script|style)\b[^>]*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_HTML_TAG_RE = re.compile(r"</?[^<>]+>")
_HTML_ENTITY_RE = re.compile(r"&(?:[a-zA-Z][a-zA-Z0-9]+|#[0-9]+|#x[0-9a-fA-F]+);")
_ENCODED_TAG_RESIDUE_RE = re.compile(
    r"\b(?:lt\s+/?\s*(?:div|span|p|br|table|tr|td|body|html)(?:\s+\w+){0,6}\s+gt|"
    r"(?:div|span)\s+class\s+htmlview(?:\s+paragraph)?)\b",
    re.IGNORECASE,
)
_TEXT_METADATA_FRAGMENT_RE = re.compile(
    r"\b(?:get\s+access|journal\s+article|articles?\s+author|"
    r"works?\s+author|author\s+gsw\s+google|google\s+scholar|"
    r"view\s+article|download\s+pdf|article\s+info)\b",
    re.IGNORECASE,
)
_TEXT_VOLUME_FRAGMENT_RE = re.compile(
    r"\b(?:vol|volume|issue)\.?\s*\d+[a-z]?\b|\b\d+[a-z]?\s*(?:vol|volume|issue)\b",
    re.IGNORECASE,
)
_LATEX_PREAMBLE_RE = re.compile(
    r"\\?\b(?:usepackage|documentclass|newcommand|renewcommand|providecommand|"
    r"requirepackage|declaremathoperator|bibliographystyle|texorpdfstring|"
    r"maketitle)\b(?:\s*\[[^\]]*\])?(?:\s*\{[^{}]*\})*",
    re.IGNORECASE,
)
_LATEX_BEGIN_END_DOCUMENT_RE = re.compile(
    r"\\?\b(?:begin|end)\s*\{?\s*document\s*\}?",
    re.IGNORECASE,
)
_TERM_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HTML_RESIDUE_TOKENS = frozenset(
    {
        "htmlview",
        "href",
        "nbsp",
        "onclick",
        "script",
        "stylesheet",
        "javascript",
    }
)
_HTML_TAG_TOKENS = frozenset({"div", "span", "html", "body", "table", "tr", "td", "br", "p"})
_METADATA_EXACT_PHRASES = frozenset(
    {
        "get access",
        "journal article",
        "articles author",
        "article author",
        "works author",
        "work author",
        "author gsw google",
    }
)
_METADATA_ID_TOKENS = frozenset({"doi", "issn", "isbn", "pmid", "pmcid"})
_LATEX_PREAMBLE_TOKENS = frozenset(
    {
        "usepackage",
        "documentclass",
        "newcommand",
        "renewcommand",
        "providecommand",
        "requirepackage",
        "declaremathoperator",
        "bibliographystyle",
        "texorpdfstring",
        "maketitle",
    }
)


def _html_unescape_repeated(text: str, *, max_rounds: int = 3) -> str:
    """Decode HTML entities a few times to handle double-encoded inputs."""
    current = text
    for _ in range(max(1, int(max_rounds))):
        decoded = html.unescape(current)
        if decoded == current:
            break
        current = decoded
    return current


def _looks_like_metadata_artifact_term(term: object) -> bool:
    """Return True for HTML/publisher-page fragments, not topical phrases.

    The checks are deliberately shape-based.  They catch combinations that
    arise from encoded tags or scraped page metadata while avoiding broad
    domain words such as "class", "author", or "volume" by themselves.
    """
    if term is None:
        return False
    text = str(term).strip().lower()
    if not text:
        return False
    tokens = _TERM_TOKEN_RE.findall(text)
    if not tokens:
        return False
    token_set = set(tokens)
    phrase = " ".join(tokens)

    if phrase in _METADATA_EXACT_PHRASES:
        return True
    if token_set & _LATEX_PREAMBLE_TOKENS:
        return True
    if {"begin", "document"} <= token_set or {"end", "document"} <= token_set:
        return True
    if token_set & _HTML_RESIDUE_TOKENS:
        return True
    if {"lt", "gt"} & token_set and token_set & (_HTML_TAG_TOKENS | {"class"}):
        return True
    if "class" in token_set and token_set & {"htmlview", "paragraph"}:
        return True
    if {"div", "class"} <= token_set and token_set & {"htmlview", "paragraph"}:
        return True
    if token_set & _METADATA_ID_TOKENS:
        return True
    if "gsw" in token_set:
        return True
    if {"articles", "author"} <= token_set or {"article", "author"} <= token_set:
        return True
    if {"works", "author"} <= token_set or {"work", "author"} <= token_set:
        return True
    if "author" in token_set and token_set & {"google", "scholar"}:
        return True
    if "vol" in token_set:
        return True
    return False


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
    text = _html_unescape_repeated(text)
    text = text.replace("\xa0", " ")
    text = _HTML_COMMENT_RE.sub(" ", text)
    text = _HTML_SCRIPT_STYLE_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _HTML_ENTITY_RE.sub(" ", text)
    text = _ENCODED_TAG_RESIDUE_RE.sub(" ", text)
    text = _TEXT_METADATA_FRAGMENT_RE.sub(" ", text)
    text = _TEXT_VOLUME_FRAGMENT_RE.sub(" ", text)
    text = _LATEX_PREAMBLE_RE.sub(" ", text)
    text = _LATEX_BEGIN_END_DOCUMENT_RE.sub(" ", text)
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())
