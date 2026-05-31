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

METADATA_ARTIFACT_FILTER_VERSION = 7

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
    r"view\s+article|download\s+pdf|article\s+info|"
    r"invalid\s+email\s+address|required\s+invalid\s+characters|"
    r"pubmed\s+scopus|cross\s+ref|score\s+score\s+calculated|"
    r"problems\s+references|altmetric\s+attention\s+score|"
    r"abstractcitation\s+referencesmore\s+options|add\s+toview\s+inadd|"
    r"articles?\s+citing\s+article|citations?\s+number\s+articles?|"
    r"citing\s+article\s+calculated|given\s+article\s+information|"
    r"information\s+crossref\s+citation|metricsarticle\s+views\s+counter|"
    r"referenceadd\s+description|exportriscitationcitation|"
    r"text\s+article\s+downloads|text\s+referenceadd\s+description|"
    r"research\s+article\s+received|leading\s+days\s+citations?|"
    r"presence\s+given\s+article|institutions\s+individuals\s+metrics|"
    r"attention\s+research\s+article|article\s+information\s+score|"
    r"html\s+institutions\s+individuals|days\s+citations?\s+number|"
    r"number\s+articles?\s+citing|views\s+counter\s+compliant|"
    r"pdf\s+html\s+institutions|permissionsarticle|"
    r"scopus\s+google|preview\s+article\s+review|"
    r"level\s+evidence\s+articles?|individuals\s+metrics\s+regularly|"
    r"ueber\s+die|deutsch\s+med|klin\s+wchnschr)\b",
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
        "invalid email address",
        "required invalid characters",
        "pubmed scopus",
        "cross ref",
        "score score calculated",
        "problems references",
        "altmetric attention score",
        "abstractcitation referencesmore options",
        "add toview inadd",
        "articles citing article",
        "citations number articles",
        "citing article calculated",
        "given article information",
        "information crossref citation",
        "metricsarticle views counter",
        "referenceadd description",
        "referenceadd description exportriscitationcitation",
        "text article downloads",
        "text referenceadd description",
        "research article received",
        "leading days citations",
        "presence given article",
        "institutions individuals metrics",
        "attention research article",
        "article information score",
        "html institutions individuals",
        "days citations number",
        "number articles citing",
        "views counter compliant",
        "pdf html institutions",
        "2008 pdf html",
        "scopus google",
        "preview article review",
        "level evidence articles",
        "individuals metrics regularly",
        "et et et",
        "ma harvard",
        "ann dermat et",
        "jr biol chem",
        "ueber die",
        "deutsch med",
        "klin wchnschr",
        "gruyter",
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
    if {"invalid", "email", "address"} <= token_set:
        return True
    if {"required", "invalid", "characters"} <= token_set:
        return True
    if {"pubmed", "scopus"} <= token_set:
        return True
    if "google" in token_set and (
        token_set & {"scopus", "pubmed", "medline", "scholar"}
        or any(re.fullmatch(r"(?:19|20)\d{2}", tok) for tok in tokens)
    ):
        return True
    if any("crossref" in tok for tok in tokens):
        return True
    if any(tok in {"medlinegoogle", "isigoogle", "crossrefgoogle"} for tok in tokens):
        return True
    if {"cross", "ref"} <= token_set:
        return True
    if {"score", "calculated"} <= token_set:
        return True
    if {"problems", "references"} <= token_set:
        return True
    if {"altmetric", "attention", "score"} <= token_set:
        return True
    if token_set & {
        "abstractcitation",
        "referencesmore",
        "toview",
        "inadd",
        "gruyter",
        "referenceadd",
        "exportriscitationcitation",
        "citationcitation",
        "metricsarticle",
    }:
        return True
    if {"citing", "article"} <= token_set and token_set & {"articles", "calculated"}:
        return True
    if {"citation", "counts", "quantitative"} <= token_set:
        return True
    if any(re.fullmatch(r"citation\d{4}", tok) for tok in tokens):
        return True
    if {"citations", "number", "articles"} <= token_set:
        return True
    if {"days", "citations", "number"} <= token_set:
        return True
    if {"number", "articles", "citing"} <= token_set:
        return True
    if {"given", "article"} <= token_set:
        return True
    if {"attention", "research", "article"} <= token_set:
        return True
    if {"article", "information", "score"} <= token_set:
        return True
    if {"article", "information"} <= token_set and token_set & {"given", "citation", "crossref"}:
        return True
    if {"article", "downloads"} <= token_set or {"article", "views"} <= token_set:
        return True
    if {"views", "counter"} <= token_set and token_set & {"article", "metricsarticle"}:
        return True
    if {"views", "counter", "compliant"} <= token_set:
        return True
    if {"pdf", "html"} <= token_set and token_set & {"institutions", "article"}:
        return True
    if {"html", "institutions", "individuals"} <= token_set:
        return True
    if any(tok.startswith("permissionsarticle") for tok in tokens):
        return True
    if {"research", "article", "received"} <= token_set:
        return True
    if {"leading", "days"} <= token_set and token_set & {"citation", "citations"}:
        return True
    if {"institutions", "individuals", "metrics"} <= token_set:
        return True
    if {"individuals", "metrics", "regularly"} <= token_set:
        return True
    if {"preview", "article", "review"} <= token_set:
        return True
    if {"level", "evidence"} <= token_set and token_set & {"article", "articles"}:
        return True
    if {"ueber", "die"} <= token_set:
        return True
    if {"deutsch", "med"} <= token_set:
        return True
    if {"klin", "wchnschr"} <= token_set:
        return True
    if any(tok.endswith("journal") and len(tok) > len("journal") for tok in tokens):
        return True
    if "vol" in token_set:
        return True
    if len(tokens) > 1 and tokens[-1] == "abstract":
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
