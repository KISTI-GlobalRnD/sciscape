"""Domain-agnostic keyword quality annotation and reranking.

The helpers in this module avoid domain dictionaries.  They score terms by
how useful they are as cluster-facing labels: cluster concentration, phrase
specificity, redundancy with longer phrases, and artifact-like shape.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .normalization import _normalize_notation, _normalize_spelling, _phrase_singular
from .utils import _looks_like_metadata_artifact_term


METADATA_TERMS: frozenset[str] = frozenset(
    {
        "abstract",
        "article",
        "articles",
        "introduction",
        "background",
        "conclusion",
        "conclusions",
        "copyright",
        "cross ref",
        "crossref",
        "date",
        "description",
        "download",
        "downloads",
        "keyword",
        "keywords",
        "online",
        "paper",
        "pubmed scopus",
        "study",
    }
)

LOW_INFORMATION_TERMS: frozenset[str] = frozenset(
    {
        "approach",
        "analysis",
        "application",
        "applications",
        "data",
        "effect",
        "effects",
        "experiment",
        "experiments",
        "framework",
        "method",
        "methods",
        "model",
        "models",
        "performance",
        "research",
        "result",
        "results",
        "standard",
        "system",
        "systems",
        "table",
        "tables",
        "technique",
        "version",
        "work",
    }
)

COMMON_SHORT_WORDS: frozenset[str] = frozenset(
    {
        "acid",
        "area",
        "band",
        "bias",
        "body",
        "case",
        "cell",
        "code",
        "data",
        "dose",
        "drug",
        "film",
        "flow",
        "form",
        "fuel",
        "gene",
        "heat",
        "ion",
        "lead",
        "line",
        "load",
        "loss",
        "mass",
        "mean",
        "mode",
        "peak",
        "rate",
        "ring",
        "risk",
        "salt",
        "seed",
        "size",
        "soil",
        "spin",
        "term",
        "test",
        "time",
        "tin",
        "type",
        "user",
        "wave",
        "wind",
        "work",
        "year",
    }
)

FRAGMENT_INITIAL_TOKENS: frozenset[str] = frozenset(
    {
        "based",
        "like",
        "meets",
        "produced",
        "things",
    }
)
FRAGMENT_FINAL_TOKENS: frozenset[str] = frozenset(
    {
        "based",
        "black",
        "constructed",
        "isolated",
        "new",
        "nov",
        "novel",
        "stay",
    }
)
FRAGMENT_GENERIC_HEADS: frozenset[str] = frozenset(
    {
        "analysis",
        "application",
        "applications",
        "design",
        "effect",
        "effects",
        "method",
        "model",
        "properties",
        "property",
        "synthesis",
        "system",
    }
)
MISSING_STOPWORD_FRAGMENTS: frozenset[tuple[str, ...]] = frozenset(
    {
        ("quality", "life"),
    }
)
NUMERIC_EVENT_TERMS: frozenset[str] = frozenset({"covid", "mers", "pandemic", "sars"})
OXIDATION_STATE_TOKENS: frozenset[str] = frozenset({"ii", "iii", "iv", "vi"})
ROMAN_MATERIAL_CLASS_TOKENS: frozenset[str] = frozenset({"ii", "iii", "iv", "v", "vi"})
ROMAN_MATERIAL_CLASS_TERMS: frozenset[str] = frozenset(
    {
        "compound",
        "compounds",
        "heterostructure",
        "heterostructures",
        "material",
        "materials",
        "semiconductor",
        "semiconductors",
    }
)
OXIDATION_GAP_CONTEXT_TOKENS: frozenset[str] = frozenset(
    {
        "adsorption",
        "aqueous",
        "ion",
        "ions",
        "metal",
        "metals",
        "removal",
        "solution",
        "solutions",
        "sorption",
        "uptake",
        "wastewater",
        "water",
    }
)
BROAD_SEMANTIC_HEAD_PHRASES: frozenset[tuple[str, ...]] = frozenset(
    {
        ("high", "performance"),
    }
)
DOCUMENT_GENRE_REVIEW_PHRASES: frozenset[tuple[str, ...]] = frozenset(
    {
        ("book", "review"),
        ("book", "reviews"),
    }
)
TITLE_LIKE_STORY_PHRASES: frozenset[tuple[str, ...]] = frozenset(
    {
        ("tell", "stories"),
        ("tell", "story"),
    }
)

_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_ALNUM_RE = re.compile(r"(?=.*[a-z])(?=.*\d)[a-z0-9]+")
_SHORT_ALPHA_RE = re.compile(r"^[a-z]{2,8}$")
_DIMENSION_TOKENS: frozenset[str] = frozenset({"0d", "1d", "2d", "3d", "4d"})
_FORMULA_FRAGMENT_SYMBOLS: frozenset[str] = frozenset(
    {
        "ag",
        "as",
        "bi",
        "br",
        "cd",
        "co",
        "cs",
        "cu",
        "fe",
        "ga",
        "in",
        "mn",
        "ni",
        "pb",
        "sb",
        "se",
        "sn",
        "te",
        "ti",
        "zn",
    }
)
_ELEMENT_SYMBOLS: frozenset[str] = frozenset(
    {
        "h", "he", "li", "be", "b", "c", "n", "o", "f", "ne",
        "na", "mg", "al", "si", "p", "s", "cl", "ar", "k", "ca",
        "sc", "ti", "v", "cr", "mn", "fe", "co", "ni", "cu", "zn",
        "ga", "ge", "as", "se", "br", "kr", "rb", "sr", "y", "zr",
        "nb", "mo", "tc", "ru", "rh", "pd", "ag", "cd", "in", "sn",
        "sb", "te", "i", "xe", "cs", "ba", "la", "ce", "pr", "nd",
        "pm", "sm", "eu", "gd", "tb", "dy", "ho", "er", "tm", "yb",
        "lu", "hf", "ta", "w", "re", "os", "ir", "pt", "au", "hg",
        "tl", "pb", "bi", "po", "at", "rn", "fr", "ra", "ac", "th",
        "pa", "u", "np", "pu", "am", "cm", "bk", "cf", "es", "fm",
        "md", "no", "lr", "rf", "db", "sg", "bh", "hs", "mt", "ds",
        "rg", "cn", "nh", "fl", "mc", "lv", "ts", "og",
    }
)
_NO_DIGIT_MATERIAL_FORMULAS: frozenset[str] = frozenset(
    {
        "cds",
        "gan",
        "pbs",
        "sic",
        "sno",
        "tio",
        "zno",
    }
)


def _normalise_term(term: object) -> str:
    text = "" if term is None else str(term).strip().lower()
    text = text.replace("-", " ")
    return " ".join(text.split())


def _label_key(term: object) -> str:
    text = _normalise_term(term)
    text = _normalize_notation(text)
    text = _normalize_spelling(text)
    singular = _phrase_singular(text)
    return singular or text


def _tokens(term: str) -> list[str]:
    return [tok for tok in _SPLIT_RE.split(term.lower()) if tok]


def _short_form_base(compact: str) -> str:
    if len(compact) > 3 and compact.endswith("s"):
        return compact[:-1]
    return compact


def _has_compact_short_form_shape(compact: str, *, max_length: int) -> bool:
    base = _short_form_base(compact)
    if len(base) < 2 or len(base) > int(max_length):
        return False
    if "rna" in base or "dna" in base:
        return len(base) <= int(max_length)
    if not any(ch in base for ch in "aeiou"):
        return True
    if len(base) == 3 and base[-1] in {"i", "y"} and not any(ch in base[:-1] for ch in "aeiou"):
        return True
    if len(base) == 3 and base.endswith("q"):
        return True
    if len(base) <= 3 and len(set(base)) < len(base):
        return True
    return False


def _parse_element_formula(compact: str) -> tuple[bool, int, bool, bool]:
    if not compact or not compact.isalnum() or compact in COMMON_SHORT_WORDS:
        return False, 0, False, False

    best: tuple[int, bool, bool] | None = None

    def _walk(pos: int, n_elements: int, has_digit: bool, has_multiletter: bool) -> None:
        nonlocal best
        if pos == len(compact):
            candidate = (n_elements, has_digit, has_multiletter)
            if best is None or candidate[0] > best[0]:
                best = candidate
            return
        if pos > len(compact):
            return
        if not compact[pos].isalpha():
            return
        for width in (2, 1):
            if pos + width > len(compact):
                continue
            symbol = compact[pos:pos + width]
            if symbol not in _ELEMENT_SYMBOLS:
                continue
            next_pos = pos + width
            saw_digit = False
            while next_pos < len(compact) and compact[next_pos].isdigit():
                saw_digit = True
                next_pos += 1
            _walk(
                next_pos,
                n_elements + 1,
                has_digit or saw_digit,
                has_multiletter or width == 2,
            )

    _walk(0, 0, False, False)
    if best is None:
        return False, 0, False, False
    n_elements, has_digit, has_multiletter = best
    return True, n_elements, has_digit, has_multiletter


def _is_material_formula_like(term: str) -> bool:
    compact = "".join(_tokens(term))
    if (
        not compact
        or compact in COMMON_SHORT_WORDS
        or compact in LOW_INFORMATION_TERMS
        or compact in METADATA_TERMS
        or compact in _DIMENSION_TOKENS
        or not re.fullmatch(r"[a-z0-9]{2,16}", compact)
    ):
        return False
    parsed, n_elements, has_digit, has_multiletter = _parse_element_formula(compact)
    if not parsed or n_elements < 2:
        return False
    if has_digit:
        return True
    return compact in _NO_DIGIT_MATERIAL_FORMULAS and n_elements == 2 and has_multiletter


def _is_formula_like(term: str) -> bool:
    tokens = _tokens(term)
    if not tokens:
        return False
    if any(_ALNUM_RE.match(tok) and tok not in _DIMENSION_TOKENS for tok in tokens):
        return True
    short_tokens = sum(1 for tok in tokens if len(tok) <= 2)
    long_dense_tokens = sum(1 for tok in tokens if len(tok) >= 5 and re.search(r"[bcfhiknopsuvwyz]{4,}", tok))
    return short_tokens > 0 and long_dense_tokens > 0


def _is_supported_roman_material_class_phrase(tokens: Sequence[str]) -> bool:
    if len(tokens) < 3:
        return False
    has_roman_pair = any(
        left in ROMAN_MATERIAL_CLASS_TOKENS and right in ROMAN_MATERIAL_CLASS_TOKENS
        for left, right in zip(tokens, tokens[1:])
    )
    return has_roman_pair and bool(set(tokens) & ROMAN_MATERIAL_CLASS_TERMS)


def _is_compact_formula_fragment_token(token: str) -> bool:
    compact = _normalise_term(token).replace(" ", "")
    if (
        len(compact) < 3
        or len(compact) > 8
        or compact in COMMON_SHORT_WORDS
        or compact in LOW_INFORMATION_TERMS
        or compact in METADATA_TERMS
        or compact in _DIMENSION_TOKENS
        or not compact.isalpha()
        or _is_material_formula_like(compact)
    ):
        return False
    formula_symbol_count = sum(1 for symbol in _FORMULA_FRAGMENT_SYMBOLS if symbol in compact)
    if formula_symbol_count < 2:
        return False

    parsed, n_elements, has_digit, has_multiletter = _parse_element_formula(compact)
    if parsed and n_elements >= 3 and has_multiletter and not has_digit:
        return True

    # Domain formulas sometimes carry a short organic-cation prefix that the
    # pure element parser cannot segment, e.g. "fapbi" -> "fa" + "pbi".
    # Treat this only as representative-label risk; the keyword row remains.
    if len(compact) <= 6:
        for start in range(1, min(3, len(compact) - 2) + 1):
            suffix = compact[start:]
            parsed, n_elements, has_digit, has_multiletter = _parse_element_formula(suffix)
            if parsed and not has_digit and (n_elements >= 3 or (n_elements >= 2 and has_multiletter)):
                return True
    return False


def _is_unresolved_compact_short_form_fragment(token: str) -> bool:
    compact = _normalise_term(token).replace(" ", "")
    if (
        len(compact) < 3
        or len(compact) > 5
        or compact in COMMON_SHORT_WORDS
        or compact in LOW_INFORMATION_TERMS
        or compact in METADATA_TERMS
        or compact in _DIMENSION_TOKENS
        or not compact.isalpha()
    ):
        return False
    base = _short_form_base(compact)
    vowel_count = sum(1 for char in base if char in "aeiou")
    if not any(char in base for char in "aeiou"):
        return True
    if len(base) <= 4 and vowel_count <= 1 and base[-1] in {"p", "q", "x"}:
        return True
    return False


def _representative_artifact_flags(term: str, *, abbreviation_status: str) -> list[str]:
    if abbreviation_status in {"cluster_expanded", "corpus_expanded", "duplicate_expansion"}:
        return []
    tokens = _tokens(term)
    if not tokens:
        return []

    flags: list[str] = []
    formula_tokens = [tok for tok in tokens if _is_compact_formula_fragment_token(tok)]
    short_form_tokens = [
        tok
        for tok in tokens
        if _is_unresolved_compact_short_form_fragment(tok)
    ]
    has_dimension = any(tok in _DIMENSION_TOKENS for tok in tokens)
    compact_dimension_tokens = [
        tok
        for tok in tokens
        if tok not in _DIMENSION_TOKENS
        and 3 <= len(tok) <= 5
        and tok.endswith("s")
        and tok not in COMMON_SHORT_WORDS
        and tok not in LOW_INFORMATION_TERMS
        and tok not in METADATA_TERMS
    ]
    if has_dimension and (formula_tokens or short_form_tokens or compact_dimension_tokens):
        flags.append("dimension_fragment")
    if formula_tokens:
        if len(tokens) == 1:
            flags.append("compact_formula_fragment")
        else:
            flags.append("mixed_formula_fragment")
    if short_form_tokens:
        flags.append("unresolved_compact_short_form")
    return flags


def _representative_fragment_flags(term: str, longer_terms: Sequence[str]) -> list[str]:
    tokens = _tokens(term)
    if len(tokens) < 2:
        return []

    flags: list[str] = []
    token_tuple = tuple(tokens)
    first = tokens[0]
    last = tokens[-1]

    if token_tuple in MISSING_STOPWORD_FRAGMENTS:
        flags.append("missing_stopword_phrase_fragment")
    if first.isdigit() and len(first) <= 2:
        flags.append("numeric_leading_phrase_fragment")
    if first in FRAGMENT_INITIAL_TOKENS:
        flags.append("initial_modifier_phrase_fragment")
    if last in FRAGMENT_FINAL_TOKENS and (
        len(tokens) == 2
        or first in FRAGMENT_GENERIC_HEADS
        or first in LOW_INFORMATION_TERMS
        or first in METADATA_TERMS
        or (last == "isolated" and len(tokens) <= 3)
    ):
        flags.append("terminal_modifier_phrase_fragment")
    if any(left == "sp" and right == "nov" for left, right in zip(tokens, tokens[1:])):
        flags.append("taxonomic_marker_phrase_fragment")
    if "nov" in tokens and ("sp" in tokens or "gen" in tokens):
        flags.append("taxonomic_marker_phrase_fragment")
    if first == "nov" and len(tokens) <= 3:
        flags.append("taxonomic_marker_phrase_fragment")
    if token_tuple == ("things", "change", "stay"):
        flags.append("title_sentence_phrase_fragment")
    if _is_oxidation_state_gap_fragment(token_tuple, longer_terms):
        flags.append("oxidation_state_gap_fragment")
    if _is_broad_semantic_head_fragment(token_tuple, longer_terms):
        flags.append("broad_semantic_head_fragment")

    longer_token_sets = [tuple(_tokens(longer)) for longer in longer_terms if len(_tokens(longer)) > len(tokens)]
    is_prefix = any(longer[: len(tokens)] == token_tuple for longer in longer_token_sets)
    is_suffix = any(longer[-len(tokens) :] == token_tuple for longer in longer_token_sets)
    if (is_prefix or is_suffix) and (last in FRAGMENT_FINAL_TOKENS or first in FRAGMENT_INITIAL_TOKENS):
        flags.append("subphrase_fragment")

    return sorted(set(flags))


def _is_broad_semantic_head_fragment(
    tokens: Sequence[str],
    cluster_terms: Sequence[str],
) -> bool:
    """Detect broad phrase heads only when local derivative labels exist."""

    token_tuple = tuple(tokens)
    if token_tuple not in BROAD_SEMANTIC_HEAD_PHRASES:
        return False
    child_count = 0
    for term in cluster_terms:
        term_tokens = tuple(_tokens(term))
        if len(term_tokens) <= len(token_tuple):
            continue
        if term_tokens[: len(token_tuple)] == token_tuple:
            child_count += 1
            if child_count >= 2:
                return True
    return False


def _contains_element_oxidation_pair(tokens: Sequence[str]) -> bool:
    return any(
        left in _ELEMENT_SYMBOLS and right in OXIDATION_STATE_TOKENS
        for left, right in zip(tokens, tokens[1:])
    )


def _has_oxidation_gap_replacement(
    tokens: Sequence[str],
    cluster_phrase_tokens: Sequence[Sequence[str]],
) -> bool:
    trailing = set(tokens[1:])
    if not trailing:
        return False
    for other in cluster_phrase_tokens:
        other_tuple = tuple(other)
        if other_tuple == tuple(tokens) or len(other_tuple) < 2:
            continue
        if other_tuple[0] in OXIDATION_STATE_TOKENS:
            continue
        other_set = set(other_tuple)
        if _contains_element_oxidation_pair(other_tuple) and (trailing & other_set):
            return True
        if trailing.issubset(other_set) and (other_set & OXIDATION_GAP_CONTEXT_TOKENS):
            return True
        if (
            len(tokens) == 2
            and tokens[1] in OXIDATION_GAP_CONTEXT_TOKENS
            and tokens[1] in other_set
            and len(other_tuple) >= 2
        ):
            return True
    return False


def _is_oxidation_state_gap_fragment(
    tokens: Sequence[str],
    cluster_terms: Sequence[str],
) -> bool:
    """Detect stopword-compressed oxidation-state phrase fragments.

    The target shape is not chemistry notation itself.  It is a phrase that
    starts with an oxidation-state token because an entity and a preposition
    were lost before n-gram construction, e.g. "Pb(II) in aqueous solution" ->
    "ii aqueous".  We only mark the row when the same cluster also contains a
    cleaner phrase candidate that can carry the label instead.
    """

    if len(tokens) < 2 or tokens[0] not in OXIDATION_STATE_TOKENS:
        return False
    if len(tokens) >= 2 and tokens[1] in OXIDATION_STATE_TOKENS:
        return False
    starts_with_generic_context = tokens[1] in OXIDATION_GAP_CONTEXT_TOKENS
    starts_with_shifted_entity = (
        tokens[1] in _ELEMENT_SYMBOLS
        and any(tok in OXIDATION_STATE_TOKENS for tok in tokens[2:])
    )
    if not starts_with_generic_context and not starts_with_shifted_entity:
        return False

    cluster_phrase_tokens = []
    for term in cluster_terms:
        term_tokens = tuple(_tokens(term))
        if len(term_tokens) >= 2:
            cluster_phrase_tokens.append(term_tokens)
    return _has_oxidation_gap_replacement(tokens, cluster_phrase_tokens)


def _is_supported_numeric_event_phrase(tokens: Sequence[str]) -> bool:
    return any(tok.isdigit() for tok in tokens) and bool(set(tokens) & NUMERIC_EVENT_TERMS)


def _keyword_label_tier(
    *,
    term: str,
    display_label: str,
    flags: Sequence[str],
    representative_role: str,
    abbreviation_status: str,
) -> str:
    """Classify whether a keyword should be a primary label or support evidence."""
    flag_set = set(flags)
    label_tokens = _tokens(display_label)
    is_phrase_label = len(label_tokens) >= 2

    if representative_role == "review_artifact" or flag_set & {
        "artifact_formula",
        "artifact_like",
        "compact_formula_fragment",
        "dimension_fragment",
        "mixed_formula_fragment",
        "unresolved_compact_short_form",
    }:
        return "review_artifact"

    if representative_role == "review_fragment" or flag_set & {
        "initial_modifier_phrase_fragment",
        "broad_semantic_head_fragment",
        "missing_stopword_phrase_fragment",
        "numeric_leading_phrase_fragment",
        "oxidation_state_gap_fragment",
        "subphrase_fragment",
        "taxonomic_marker_phrase_fragment",
        "terminal_modifier_phrase_fragment",
        "title_sentence_phrase_fragment",
    }:
        return "review_fragment"

    if representative_role == "review_short_form" or abbreviation_status in {
        "ambiguous_expansion",
        "candidate_short_form",
        "low_support_expansion",
        "unlinked_short_form",
    }:
        return "review_short_form"

    if abbreviation_status in {"cluster_expanded", "corpus_expanded"} and is_phrase_label:
        return "primary_phrase"

    if "material_formula" in flag_set:
        return "support_formula"

    if is_phrase_label and representative_role in {
        "expanded_short_form",
        "expansion_phrase",
        "representative_phrase",
    }:
        return "primary_phrase"
    if is_phrase_label and "phrase" in flag_set:
        return "primary_phrase"

    if abbreviation_status in {"cluster_expanded", "corpus_expanded", "duplicate_expansion", "expanded"}:
        return "support_abbreviation"

    if len(_tokens(term)) <= 1:
        return "support_unigram"
    return "support_phrase"


_PHRASE_FRAGMENT_FLAGS: frozenset[str] = frozenset(
    {
        "initial_modifier_phrase_fragment",
        "missing_stopword_phrase_fragment",
        "numeric_leading_phrase_fragment",
        "oxidation_state_gap_fragment",
        "subphrase_fragment",
        "taxonomic_marker_phrase_fragment",
        "terminal_modifier_phrase_fragment",
        "title_sentence_phrase_fragment",
    }
)
_FORMULA_COMPACT_RISK_FLAGS: frozenset[str] = frozenset(
    {
        "artifact_formula",
        "compact_formula_fragment",
        "dimension_fragment",
        "mixed_formula_fragment",
        "unresolved_compact_short_form",
    }
)
_SHORT_FORM_RISK_FLAGS: frozenset[str] = frozenset(
    {
        "ambiguous_expansion",
        "candidate_short_form",
        "low_support_expansion",
        "unlinked_short_form",
    }
)
_SHAPE_BASIS_FLAGS: frozenset[str] = frozenset(
    {
        "acronym_like",
        "artifact_formula",
        "compact_formula_fragment",
        "dimension_fragment",
        "formula_like",
        "initial_modifier_phrase_fragment",
        "missing_stopword_phrase_fragment",
        "mixed_formula_fragment",
        "numeric_leading_phrase_fragment",
        "taxonomic_marker_phrase_fragment",
        "terminal_modifier_phrase_fragment",
        "title_sentence_phrase_fragment",
        "unresolved_compact_short_form",
    }
)
_CLUSTER_REPLACEMENT_BASIS_FLAGS: frozenset[str] = frozenset(
    {
        "broad_semantic_head_fragment",
        "duplicate_expansion",
        "duplicate_label",
        "oxidation_state_gap_fragment",
        "phrase_preferred",
        "subphrase_fragment",
    }
)
_NETWORK_BASIS_FLAGS: frozenset[str] = frozenset(
    {
        "anchor_unigram",
        "expansion_phrase",
        "generic_bridge",
        "linked_unigram",
        "representative_phrase",
        "unlinked_short_form",
    }
)


def _quality_risk_family(flags: Sequence[str], *, keyword_label_tier: str) -> str:
    flag_set = set(flags)
    families: list[str] = []
    if flag_set & {"artifact_like", "metadata_fragment"}:
        families.append("metadata_artifact")
    if flag_set & _PHRASE_FRAGMENT_FLAGS:
        families.append("phrase_fragment")
    if flag_set & _SHORT_FORM_RISK_FLAGS or keyword_label_tier == "review_short_form":
        families.append("short_form_unresolved")
    if flag_set & _FORMULA_COMPACT_RISK_FLAGS:
        families.append("formula_or_compact_symbol")
    if flag_set & {"broad_semantic_head_fragment", "low_information"}:
        families.append("broad_or_generic")
    if "too_global" in flag_set:
        families.append("scope_too_global")
    if flag_set & {"duplicate_expansion", "duplicate_label", "phrase_preferred"}:
        families.append("duplicate_or_shadowed")
    if not families:
        return "clean"
    return _flag_string(families)


def _quality_flag_basis(
    flags: Sequence[str],
    *,
    abbreviation_status: str,
    network_role: str,
) -> str:
    flag_set = set(flags)
    basis: list[str] = []
    if flag_set & {"artifact_like", "metadata_fragment"}:
        basis.append("metadata_pattern")
    if flag_set & {"low_information"}:
        basis.append("dictionary_or_stoplist")
    if "too_global" in flag_set:
        basis.append("corpus_distribution")
    if flag_set & _CLUSTER_REPLACEMENT_BASIS_FLAGS:
        basis.append("cluster_replacement")
    if abbreviation_status in {
        "ambiguous_expansion",
        "cluster_expanded",
        "corpus_expanded",
        "duplicate_expansion",
        "expanded",
        "low_support_expansion",
    }:
        basis.append("abbreviation_evidence")
    if flag_set & _NETWORK_BASIS_FLAGS or network_role not in {"", "neutral"}:
        basis.append("network_structure")
    if flag_set & _SHAPE_BASIS_FLAGS or abbreviation_status in {"candidate_short_form", "unlinked_short_form"}:
        basis.append("shape_only")
    if not basis:
        return "none"
    return _flag_string(basis)


def _quality_flag_confidence(
    flags: Sequence[str],
    *,
    risk_family: str,
    flag_basis: str,
    keyword_label_tier: str,
) -> str:
    flag_set = set(flags)
    if risk_family == "clean":
        return "none"
    if flag_set & {"artifact_like", "metadata_fragment"}:
        return "high"
    basis_set = set(flag_basis.split("|")) if flag_basis else set()
    if basis_set & {"abbreviation_evidence", "cluster_replacement"}:
        return "medium"
    if keyword_label_tier.startswith("review_") and not (basis_set - {"shape_only", "none"}):
        return "low"
    if "too_global" in flag_set:
        return "medium"
    return "low"


def _clean_view_action(
    flags: Sequence[str],
    *,
    keyword_label_tier: str,
    risk_family: str,
) -> str:
    flag_set = set(flags)
    if flag_set & {"artifact_like", "metadata_fragment"}:
        return "drop_from_candidates"
    if keyword_label_tier.startswith("review_"):
        return "hide_from_clean"
    if risk_family != "clean":
        return "keep_but_review"
    return "keep"


def _is_acronym_like(term: str, *, max_length: int, has_expansion: bool = False) -> bool:
    compact = term.replace(" ", "")
    if not bool(_SHORT_ALPHA_RE.match(compact)) or len(compact) > int(max_length):
        return False
    if has_expansion:
        return True
    return len(compact) <= 4 and _has_compact_short_form_shape(compact, max_length=max_length)


def _is_abbreviation_candidate(term: str, *, max_length: int) -> bool:
    compact = term.replace(" ", "")
    if not bool(_SHORT_ALPHA_RE.match(compact)) or len(compact) > int(max_length):
        return False
    if compact in COMMON_SHORT_WORDS or compact in LOW_INFORMATION_TERMS or compact in METADATA_TERMS:
        return False
    return _has_compact_short_form_shape(compact, max_length=max_length)


def _phrase_acronym_variants(phrase: str) -> set[str]:
    toks = _tokens(phrase)
    if len(toks) < 2:
        return set()
    initials = "".join(tok[0] for tok in toks if tok)
    variants = {initials}
    if toks[-1] in {"rna", "dna"}:
        variants.add("".join(tok[0] for tok in toks[:-1]) + toks[-1])
    for idx, tok in enumerate(toks):
        if tok in {"rna", "dna"} and idx > 0:
            variants.add("".join(t[0] for t in toks[:idx]) + tok)
    if toks[-1] in {"network", "networks"} and len(toks) >= 3:
        variants.add(initials)
    return {v for v in variants if 2 <= len(v) <= 8}


def _entropy(values: Sequence[float]) -> float:
    total = float(sum(v for v in values if v > 0))
    if total <= 0.0 or len(values) <= 1:
        return 0.0
    ent = 0.0
    for value in values:
        if value <= 0:
            continue
        p = float(value) / total
        ent -= p * math.log(p)
    denom = math.log(len(values))
    return 0.0 if denom <= 0.0 else min(1.0, ent / denom)


def _flag_string(flags: Iterable[str]) -> str:
    return "|".join(sorted(set(flags)))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(pd.to_numeric(value, errors="coerce"))
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return number


def _quality_adjustment(name: str, factor: float, reason: str, **evidence: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "name": str(name),
        "factor": round(float(factor), 6),
        "reason": str(reason),
    }
    for key, value in evidence.items():
        if value is None:
            continue
        if isinstance(value, float):
            entry[key] = round(float(value), 6)
        else:
            entry[key] = value
    return entry


def _decision_trace_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _keyword_scope(cluster_count: int, cluster_ratio: float, *, threshold: float) -> str:
    if cluster_count <= 1:
        return "cluster_specific"
    if cluster_ratio >= float(threshold):
        return "common"
    return "shared"


def _family_tokens(label: object) -> set[str]:
    return set(_tokens(_normalise_term(label)))


def _iter_family_evidence_terms(value: object) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        if value.strip():
            yield value
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and key.strip():
                yield key
            yield from _iter_family_evidence_terms(nested)
        return
    if isinstance(value, Iterable):
        for nested in value:
            yield from _iter_family_evidence_terms(nested)
        return
    if not pd.isna(value):
        yield str(value)


def _collect_family_evidence_terms(
    df: pd.DataFrame,
    *,
    term_col: str,
    display_labels: Sequence[str],
) -> list[list[str]]:
    evidence_columns = (
        "source_terms",
        "candidates",
        "norm_merged_from",
        "expanded_from",
    )
    rows: list[list[str]] = []
    for row_idx, row in enumerate(df.itertuples(index=False)):
        terms: set[str] = set()
        for value in (
            getattr(row, term_col, None),
            display_labels[row_idx] if row_idx < len(display_labels) else None,
        ):
            for term in _iter_family_evidence_terms(value):
                normalized = _normalise_term(term)
                if normalized:
                    terms.add(normalized)
        for column in evidence_columns:
            if column not in df.columns:
                continue
            value = getattr(row, column)
            for term in _iter_family_evidence_terms(value):
                normalized = _normalise_term(term)
                if normalized:
                    terms.add(normalized)
        rows.append(sorted(terms))
    return rows


def _compute_family_representative_support(
    *,
    cluster_ids: Sequence[Any],
    display_labels: Sequence[str],
    representative_scores: Sequence[float],
    doc_coverages: Sequence[float],
    family_terms: Sequence[Sequence[str]] | None = None,
    enabled: bool,
    weight: float,
    max_bonus: float,
    min_parent_tokens: int = 2,
) -> list[dict[str, Any]]:
    """Return per-row support from exact aliases and contained derivative phrases."""
    n_rows = len(display_labels)
    support = [
        {
            "child_count": 0,
            "member_count": 1,
            "avg_child_coverage": 0.0,
            "multiplier": 1.0,
            "children": [],
        }
        for _ in range(n_rows)
    ]
    if not enabled or n_rows == 0:
        return support
    if family_terms is None:
        family_terms = [[] for _ in range(n_rows)]

    rows_by_cluster: dict[Any, list[int]] = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        rows_by_cluster[cluster_id].append(idx)

    for row_indices in rows_by_cluster.values():
        label_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx in row_indices:
            label = _normalise_term(display_labels[idx])
            if label:
                label_to_indices[label].append(idx)

        canonical_by_label: dict[str, int] = {}
        evidence_by_label: dict[str, set[str]] = defaultdict(set)
        for label, indices in label_to_indices.items():
            canonical_by_label[label] = max(
                indices,
                key=lambda i: (float(representative_scores[i]), float(doc_coverages[i]), -i),
            )
            canonical_idx = canonical_by_label[label]
            support[canonical_idx]["member_count"] = len(indices)
            evidence_by_label[label].add(label)
            for idx in indices:
                if idx < len(family_terms):
                    evidence_by_label[label].update(str(term) for term in family_terms[idx] if str(term))

        labels = list(canonical_by_label)
        tokens_by_label = {label: _family_tokens(label) for label in labels}
        child_to_parent: dict[str, str] = {}
        for child_label in labels:
            child_tokens = tokens_by_label[child_label]
            if not child_tokens:
                continue
            candidates: list[tuple[float, int, str]] = []
            for parent_label in labels:
                if parent_label == child_label:
                    continue
                parent_tokens = tokens_by_label[parent_label]
                if len(parent_tokens) < int(min_parent_tokens):
                    continue
                if len(parent_tokens) >= len(child_tokens):
                    continue
                if parent_tokens < child_tokens:
                    parent_idx = canonical_by_label[parent_label]
                    candidates.append((float(representative_scores[parent_idx]), -len(parent_tokens), parent_label))
            if candidates:
                child_to_parent[child_label] = max(candidates)[2]

        for child_label, parent_label in child_to_parent.items():
            parent_idx = canonical_by_label[parent_label]
            child_idx = canonical_by_label[child_label]
            child_member_count = int(support[child_idx]["member_count"])
            support[parent_idx]["children"].append(
                {
                    "term": display_labels[child_idx],
                    "member_count": child_member_count,
                    "doc_coverage": round(float(doc_coverages[child_idx]), 6),
                }
            )
            support[parent_idx]["member_count"] += child_member_count

        label_set = set(labels)
        for parent_label in labels:
            parent_tokens = tokens_by_label[parent_label]
            if len(parent_tokens) < int(min_parent_tokens):
                continue
            parent_idx = canonical_by_label[parent_label]
            seen_children = {
                _normalise_term(child["term"])
                for child in support[parent_idx]["children"]
            }
            for evidence_terms in evidence_by_label.values():
                for evidence_label in evidence_terms:
                    if evidence_label == parent_label:
                        continue
                    if evidence_label in label_set:
                        continue
                    if evidence_label in seen_children:
                        continue
                    evidence_tokens = _family_tokens(evidence_label)
                    if len(evidence_tokens) <= len(parent_tokens):
                        continue
                    if parent_tokens < evidence_tokens:
                        support[parent_idx]["children"].append(
                            {
                                "term": evidence_label,
                                "member_count": 1,
                                "doc_coverage": 0.0,
                                "evidence_source": "candidate_terms",
                            }
                        )
                        support[parent_idx]["member_count"] += 1
                        seen_children.add(evidence_label)

        for idx in row_indices:
            info = support[idx]
            children = info["children"]
            child_count = len(children)
            info["child_count"] = child_count
            if child_count:
                info["avg_child_coverage"] = sum(float(child["doc_coverage"]) for child in children) / child_count
            member_extras = max(0, int(info["member_count"]) - 1)
            if child_count or member_extras:
                child_signal = min(1.0, child_count / 4.0)
                member_signal = min(1.0, member_extras / 8.0)
                bonus = min(float(max_bonus), float(weight) * (0.65 * child_signal + 0.35 * member_signal))
                info["multiplier"] = 1.0 + max(0.0, bonus)

    return support


def _representative_signal(
    term: str,
    *,
    n_tokens: int,
    flags: Sequence[str],
    keyword_scope: str,
    network_role: str,
    abbreviation_status: str,
) -> tuple[float, str, str]:
    """Score how suitable a keyword is for cluster-facing labels.

    ``quality_score`` is deliberately conservative because it also controls
    whether useful unigrams remain visible.  This display-only score can be
    stricter: unresolved abbreviations, common bridge terms, and shadowed
    unigrams are useful audit rows but weak representative labels.
    """

    flag_set = set(flags)
    multiplier = 1.0
    rep_flags: list[str] = []
    role = "candidate"

    if keyword_scope == "cluster_specific":
        multiplier *= 1.08
        rep_flags.append("cluster_specific")
    elif keyword_scope == "shared":
        multiplier *= 0.92
        rep_flags.append("shared")
        if n_tokens == 1:
            multiplier *= 0.52
            rep_flags.append("shared_unigram")
            role = "shared_unigram"
    elif keyword_scope == "common":
        multiplier *= 0.58
        rep_flags.append("common_term")
        role = "common_term"

    if n_tokens >= 2:
        multiplier *= 1.0 + min(0.22, 0.08 + 0.04 * min(3, n_tokens - 1))
        rep_flags.append("phrase_label")
        if keyword_scope == "cluster_specific":
            role = "representative_phrase"
    elif "phrase_preferred" in flag_set:
        rep_flags.append("shadowed_unigram")
        if network_role == "anchor_unigram":
            multiplier *= 0.72
            role = "anchor_unigram"
        elif network_role == "linked_unigram":
            multiplier *= 0.46
            role = "linked_unigram"
        else:
            multiplier *= 0.36
            role = "shadowed_unigram"
    elif term in COMMON_SHORT_WORDS or term in LOW_INFORMATION_TERMS or term in METADATA_TERMS:
        multiplier *= 0.70
        rep_flags.append("low_information_unigram")
        role = "low_information_unigram"

    if network_role == "expansion_phrase":
        multiplier *= 1.14
        rep_flags.append("expansion_phrase")
        role = "expansion_phrase"
    elif network_role == "representative_phrase":
        multiplier *= 1.10
        rep_flags.append("network_representative")
        if role == "candidate":
            role = "representative_phrase"
    elif network_role == "anchor_unigram":
        multiplier *= 1.05
        rep_flags.append("network_anchor")
        if role == "candidate":
            role = "anchor_unigram"
    elif network_role == "generic_bridge":
        multiplier *= 0.62
        rep_flags.append("generic_bridge")
        role = "generic_bridge"
    elif network_role == "unlinked_short_form":
        multiplier *= 0.64
        rep_flags.append("unlinked_short_form")
        role = "review_short_form"

    if abbreviation_status in {"cluster_expanded", "corpus_expanded"}:
        multiplier *= 1.08
        rep_flags.append("expanded_short_form")
        if role == "candidate":
            role = "expanded_short_form"
    elif abbreviation_status == "duplicate_expansion":
        multiplier *= 0.08
        rep_flags.append("duplicate_expansion")
        role = "duplicate_expansion"
    elif abbreviation_status in {
        "ambiguous_expansion",
        "candidate_short_form",
        "low_support_expansion",
        "unlinked_short_form",
    }:
        multiplier *= 0.52
        rep_flags.append("review_short_form")
        role = "review_short_form"

    if "artifact_like" in flag_set or "artifact_formula" in flag_set:
        multiplier *= 0.45
        rep_flags.append("artifact_demoted")
        role = "artifact_demoted"
    elif "material_formula" in flag_set:
        multiplier *= 1.04
        rep_flags.append("material_formula")
        if role == "candidate":
            role = "material_formula"

    representative_artifacts = flag_set & {
        "compact_formula_fragment",
        "dimension_fragment",
        "mixed_formula_fragment",
        "unresolved_compact_short_form",
    }
    if representative_artifacts:
        if "dimension_fragment" in representative_artifacts:
            factor = 0.18
        elif "mixed_formula_fragment" in representative_artifacts:
            factor = 0.22
        elif "unresolved_compact_short_form" in representative_artifacts:
            factor = 0.28
        else:
            factor = 0.30
        multiplier *= factor
        rep_flags.extend(sorted(representative_artifacts))
        rep_flags.append("representative_artifact")
        role = "review_artifact"

    representative_fragments = flag_set & {
        "initial_modifier_phrase_fragment",
        "broad_semantic_head_fragment",
        "missing_stopword_phrase_fragment",
        "numeric_leading_phrase_fragment",
        "oxidation_state_gap_fragment",
        "subphrase_fragment",
        "taxonomic_marker_phrase_fragment",
        "terminal_modifier_phrase_fragment",
        "title_sentence_phrase_fragment",
    }
    if representative_fragments:
        multiplier *= 0.28
        rep_flags.extend(sorted(representative_fragments))
        rep_flags.append("representative_fragment")
        role = "review_fragment"

    if role == "candidate" and n_tokens >= 2:
        role = "representative_phrase"

    return max(0.01, multiplier), role, _flag_string(rep_flags)


def _abbreviation_evidence(
    term: str,
    *,
    expansion: str | None,
    corpus_evidence: Mapping[str, Any] | None,
    network_role: str,
    network_flags: Sequence[str],
    acronym_max_length: int,
) -> tuple[str, str, float, str, int, int, float, str]:
    if corpus_evidence:
        status = str(corpus_evidence.get("status", "corpus_expanded"))
        target = str(corpus_evidence.get("long_form", ""))
        confidence = _safe_float(corpus_evidence.get("confidence", 0.0))
        support_docs = int(_safe_float(corpus_evidence.get("support_docs", 0.0)))
        cluster_support_docs = int(_safe_float(corpus_evidence.get("cluster_support_docs", 0.0)))
        top_support_ratio = _safe_float(corpus_evidence.get("top_support_ratio", 0.0))
        ambiguity_type = str(corpus_evidence.get("ambiguity_type", "none"))
        source = "cluster_parenthetical" if status == "cluster_expanded" else "corpus_parenthetical"
        return status, target, confidence, source, support_docs, cluster_support_docs, top_support_ratio, ambiguity_type
    if expansion:
        if "duplicate_label" in network_flags:
            return "duplicate_expansion", expansion, 1.0, "candidate_terms", 0, 0, 0.0, "none"
        return "expanded", expansion, 0.9, "candidate_terms", 0, 0, 0.0, "none"
    if network_role == "unlinked_short_form":
        return "unlinked_short_form", "", 0.35, "shape", 0, 0, 0.0, "none"
    if _is_abbreviation_candidate(term, max_length=acronym_max_length):
        return "candidate_short_form", "", 0.4, "shape", 0, 0, 0.0, "none"
    return "not_abbreviation", "", 0.0, "", 0, 0, 0.0, "none"


def _lookup_abbreviation_evidence(
    abbreviation_lookup: Mapping[str, Any] | None,
    *,
    cluster_id: Any,
    term: str,
) -> Mapping[str, Any] | None:
    if not abbreviation_lookup:
        return None
    short = term.replace(" ", "").lower()
    cluster_lookup = abbreviation_lookup.get("cluster", {})
    try:
        cluster_key = (int(cluster_id), short)
    except (TypeError, ValueError):
        cluster_key = (cluster_id, short)
    evidence = cluster_lookup.get(cluster_key)
    if evidence and evidence.get("usable", True):
        return evidence
    global_lookup = abbreviation_lookup.get("global", {})
    evidence = global_lookup.get(short)
    if evidence and (
        evidence.get("usable", False)
        or evidence.get("status") in {"ambiguous_expansion", "low_support_expansion"}
    ):
        return evidence
    return None


def _find_phrase_expansions(df: pd.DataFrame, *, term_col: str, cluster_col: str) -> dict[tuple[Any, str], str]:
    expansions: dict[tuple[Any, str], str] = {}
    if df.empty or term_col not in df.columns or cluster_col not in df.columns:
        return expansions
    for cluster_id, group in df.groupby(cluster_col, sort=False):
        phrases = [
            _normalise_term(term)
            for term in group[term_col].tolist()
            if len(_tokens(_normalise_term(term))) >= 2
        ]
        acronym_to_phrases: dict[str, list[str]] = defaultdict(list)
        for phrase in phrases:
            for acro in _phrase_acronym_variants(phrase):
                acronym_to_phrases[acro].append(phrase)
        for term in group[term_col].tolist():
            normalised = _normalise_term(term)
            if " " in normalised:
                continue
            matches = acronym_to_phrases.get(normalised)
            if not matches:
                continue
            matches = sorted(set(matches), key=lambda t: (-len(_tokens(t)), -len(t), t))
            expansions[(cluster_id, normalised)] = matches[0]
    return expansions


def _network_role_hints(
    df: pd.DataFrame,
    *,
    cluster_col: str,
    frequency_col: str,
    term_cluster_counts: Mapping[str, int],
    expansions: Mapping[tuple[Any, str], str],
    acronym_max_length: int,
) -> dict[tuple[Any, str], dict[str, Any]]:
    """Infer term roles from the local candidate-term graph.

    This is intentionally lightweight: nodes are candidate terms, while edges
    are implicit phrase-containment and acronym-expansion links.  The output is
    used as quality evidence, not as an unconditional merge instruction.
    """

    if df.empty or cluster_col not in df.columns or "normalized_term" not in df.columns:
        return {}

    role_hints: dict[tuple[Any, str], dict[str, Any]] = {}

    for cluster_id, group in df.groupby(cluster_col, sort=False):
        term_freq: dict[str, float] = defaultdict(float)
        for row in group.itertuples(index=False):
            term = str(getattr(row, "normalized_term"))
            if not term:
                continue
            if frequency_col in group.columns:
                term_freq[term] += _safe_float(getattr(row, frequency_col, 0.0))
            else:
                term_freq[term] += 1.0

        terms = list(term_freq.keys())
        term_set = set(terms)
        phrases = [term for term in terms if len(_tokens(term)) >= 2]
        token_to_phrases: dict[str, list[str]] = defaultdict(list)
        expansion_targets: dict[str, list[str]] = defaultdict(list)

        for phrase in phrases:
            phrase_tokens = _tokens(phrase)
            for token in set(phrase_tokens):
                if len(token) >= 2:
                    token_to_phrases[token].append(phrase)
            for acro in _phrase_acronym_variants(phrase):
                expansion_targets[phrase].append(acro)

        for term in terms:
            toks = _tokens(term)
            n_tokens = len(toks)
            key = (cluster_id, term)
            flags: list[str] = []
            role = "neutral"
            network_score = 0.5
            multiplier = 1.0

            expansion = expansions.get(key)
            if expansion:
                role = "alias_acronym"
                network_score = 0.72
                flags.extend(["alias_acronym", "expansion_linked"])
                multiplier *= 0.95
                if expansion in term_set:
                    flags.append("duplicate_label")
                    multiplier *= 0.45
            elif n_tokens == 1:
                phrase_neighbors = token_to_phrases.get(term, [])
                cluster_count = int(term_cluster_counts.get(term, 1))
                compact = term.replace(" ", "")
                short_unexpanded = (
                    bool(_SHORT_ALPHA_RE.match(compact))
                    and len(compact) <= int(acronym_max_length)
                    and compact not in COMMON_SHORT_WORDS
                    and _has_compact_short_form_shape(compact, max_length=acronym_max_length)
                )

                if phrase_neighbors and cluster_count == 1 and len(set(phrase_neighbors)) >= 2:
                    head_neighbor_ratio = sum(
                        1 for phrase in set(phrase_neighbors) if _tokens(phrase)[-1:] == [term]
                    ) / max(1, len(set(phrase_neighbors)))
                    low_information_unigram = (
                        term in COMMON_SHORT_WORDS
                        or term in LOW_INFORMATION_TERMS
                        or term in METADATA_TERMS
                    )
                    if low_information_unigram or head_neighbor_ratio >= 0.6:
                        role = "linked_unigram"
                        network_score = 0.58
                        flags.append("linked_unigram")
                        if head_neighbor_ratio >= 0.6:
                            flags.append("head_unigram")
                        if low_information_unigram:
                            flags.append("low_information_unigram")
                        multiplier *= 0.96
                    else:
                        role = "anchor_unigram"
                        network_score = min(1.0, 0.62 + 0.08 * len(set(phrase_neighbors)))
                        flags.extend(["anchor_unigram", "phrase_linked"])
                        multiplier *= 1.18
                elif phrase_neighbors and cluster_count == 1:
                    role = "linked_unigram"
                    network_score = 0.62
                    flags.append("linked_unigram")
                    multiplier *= 1.04
                elif phrase_neighbors:
                    role = "generic_bridge"
                    network_score = 0.28
                    flags.append("generic_bridge")
                    multiplier *= 0.85
                elif short_unexpanded:
                    role = "unlinked_short_form"
                    network_score = 0.25
                    flags.append("unlinked_short_form")
                    multiplier *= 0.75
            elif n_tokens >= 2:
                alias_count = sum(
                    1
                    for acro in expansion_targets.get(term, [])
                    if expansions.get((cluster_id, acro)) == term
                )
                has_specific_unigram = any(
                    token in term_set and int(term_cluster_counts.get(token, 1)) == 1
                    for token in toks
                )
                if alias_count > 0:
                    role = "expansion_phrase"
                    network_score = min(1.0, 0.82 + 0.04 * alias_count)
                    flags.append("expansion_phrase")
                    multiplier *= 1.12
                elif has_specific_unigram:
                    role = "representative_phrase"
                    network_score = 0.78
                    flags.append("representative_phrase")
                    multiplier *= 1.07

            role_hints[key] = {
                "role": role,
                "score": float(network_score),
                "flags": flags,
                "multiplier": float(multiplier),
            }

    return role_hints


def annotate_keyword_quality(
    df: pd.DataFrame,
    *,
    term_col: str = "term",
    cluster_col: str = "cluster_id",
    score_col: str = "score",
    frequency_col: str = "frequency",
    doc_coverage_col: str = "doc_coverage",
    rerank: bool = False,
    global_n_clusters: int | None = None,
    term_cluster_count_col: str | None = None,
    term_entropy_col: str | None = None,
    global_term_threshold: float = 0.5,
    global_term_penalty: float = 0.45,
    entropy_penalty: float = 0.35,
    phrase_preference_weight: float = 0.25,
    artifact_demotion_weight: float = 0.8,
    acronym_demotion_weight: float = 0.1,
    formula_demotion_weight: float = 0.25,
    single_token_shadow_penalty: float = 0.65,
    cluster_specific_bonus: float = 0.08,
    min_multiplier: float = 0.05,
    acronym_max_length: int = 6,
    network_roles_enabled: bool = True,
    abbreviation_lookup: Mapping[str, Any] | None = None,
    family_representative_enabled: bool = True,
    family_representative_weight: float = 0.08,
    family_representative_max_bonus: float = 0.15,
) -> pd.DataFrame:
    """Add quality audit columns and optionally rerank by quality score.

    The original ``term`` and ``score`` columns are preserved.  ``quality_score``
    is a display/ranking score that combines the base score with generic,
    domain-independent evidence.
    """

    if df is None or df.empty or term_col not in df.columns:
        return df

    out = df.copy()
    if "raw_term" not in out.columns:
        out["raw_term"] = out[term_col].astype(str)
    out["normalized_term"] = out[term_col].map(_normalise_term)

    cluster_ids = (
        out[cluster_col].dropna().unique().tolist()
        if cluster_col in out.columns
        else [0]
    )
    n_clusters = max(
        1,
        int(global_n_clusters)
        if global_n_clusters is not None and int(global_n_clusters) > 0
        else len(cluster_ids),
    )

    term_cluster_counts: Counter[str] = Counter()
    term_cluster_weights: dict[str, list[float]] = defaultdict(list)
    if cluster_col in out.columns:
        for term, group in out.groupby("normalized_term", sort=False):
            cluster_values = group[cluster_col].dropna().unique().tolist()
            term_cluster_counts[str(term)] = len(cluster_values)
            if frequency_col in group.columns:
                weights = pd.to_numeric(group[frequency_col], errors="coerce").fillna(0.0).tolist()
            elif doc_coverage_col in group.columns:
                weights = pd.to_numeric(group[doc_coverage_col], errors="coerce").fillna(0.0).tolist()
            else:
                weights = [1.0] * len(group)
            term_cluster_weights[str(term)] = [float(v) for v in weights]
    else:
        term_cluster_counts.update({str(t): 1 for t in out["normalized_term"].tolist()})
    if term_cluster_count_col and term_cluster_count_col in out.columns:
        for term, group in out.groupby("normalized_term", sort=False):
            values = pd.to_numeric(group[term_cluster_count_col], errors="coerce").dropna()
            if not values.empty:
                term_cluster_counts[str(term)] = max(1, int(values.max()))
    term_entropies: dict[str, float] = {}
    if term_entropy_col and term_entropy_col in out.columns:
        for term, group in out.groupby("normalized_term", sort=False):
            values = pd.to_numeric(group[term_entropy_col], errors="coerce").dropna()
            if not values.empty:
                term_entropies[str(term)] = max(0.0, min(1.0, float(values.max())))

    longer_terms_by_cluster: dict[Any, list[str]] = defaultdict(list)
    if cluster_col in out.columns:
        grouped = out.groupby(cluster_col, sort=False)
    else:
        grouped = [(0, out)]
    term_labels_by_cluster: dict[Any, dict[str, str]] = defaultdict(dict)
    term_label_scores_by_cluster: dict[Any, dict[str, float]] = defaultdict(dict)
    for cluster_id, group in grouped:
        cluster_terms = [str(term) for term in group["normalized_term"].tolist()]
        longer_terms_by_cluster[cluster_id] = [
            term
            for term in cluster_terms
            if len(_tokens(str(term))) >= 2
        ]
        if score_col in group.columns:
            group_scores = pd.to_numeric(group[score_col], errors="coerce").fillna(0.0).tolist()
        else:
            group_scores = [1.0] * len(group)
        for term, score in zip(cluster_terms, group_scores):
            if len(_tokens(term)) < 2:
                continue
            key = _label_key(term)
            current_score = term_label_scores_by_cluster[cluster_id].get(key, float("-inf"))
            if float(score) > current_score:
                term_labels_by_cluster[cluster_id][key] = term
                term_label_scores_by_cluster[cluster_id][key] = float(score)
    term_keys_by_cluster: dict[Any, set[str]] = {
        cluster_id: set(labels)
        for cluster_id, labels in term_labels_by_cluster.items()
    }

    expansions = _find_phrase_expansions(out, term_col="normalized_term", cluster_col=cluster_col) if cluster_col in out.columns else {}
    network_roles = (
        _network_role_hints(
            out,
            cluster_col=cluster_col,
            frequency_col=frequency_col,
            term_cluster_counts=term_cluster_counts,
            expansions=expansions,
            acronym_max_length=acronym_max_length,
        )
        if network_roles_enabled and cluster_col in out.columns
        else {}
    )

    quality_scores: list[float] = []
    quality_multipliers: list[float] = []
    quality_flags: list[str] = []
    display_labels: list[str] = []
    network_role_values: list[str] = []
    network_scores: list[float] = []
    network_flag_values: list[str] = []
    keyword_scopes: list[str] = []
    keyword_cluster_counts: list[int] = []
    keyword_cluster_ratios: list[float] = []
    abbreviation_statuses: list[str] = []
    abbreviation_targets: list[str] = []
    abbreviation_confidences: list[float] = []
    abbreviation_sources: list[str] = []
    abbreviation_support_docs_values: list[int] = []
    abbreviation_cluster_support_docs_values: list[int] = []
    abbreviation_top_support_ratios: list[float] = []
    abbreviation_ambiguity_types: list[str] = []
    representative_scores: list[float] = []
    representative_multipliers: list[float] = []
    representative_roles: list[str] = []
    representative_flag_values: list[str] = []
    keyword_label_tiers: list[str] = []
    quality_risk_families: list[str] = []
    quality_flag_bases: list[str] = []
    quality_flag_confidences: list[str] = []
    clean_view_actions: list[str] = []
    quality_decision_traces: list[str] = []
    row_cluster_ids: list[Any] = []
    doc_coverage_values: list[float] = []

    raw_scores = out[score_col] if score_col in out.columns else pd.Series(1.0, index=out.index)
    base_scores = pd.to_numeric(raw_scores, errors="coerce").fillna(0.0).reset_index(drop=True)

    for row_index, (_, row) in enumerate(out.iterrows()):
        term = str(row["normalized_term"])
        toks = _tokens(term)
        n_tokens = len(toks)
        cluster_id = row[cluster_col] if cluster_col in out.columns else 0
        row_cluster_ids.append(cluster_id)
        doc_coverage_values.append(_safe_float(row.get(doc_coverage_col, row.get(frequency_col, 0.0))))
        role_hint = network_roles.get(
            (cluster_id, term),
            {
                "role": "neutral",
                "score": 0.5,
                "flags": [],
                "multiplier": 1.0,
            },
        )
        flags: list[str] = []
        multiplier = 1.0
        adjustment_trace: list[dict[str, object]] = []

        cluster_count = int(term_cluster_counts.get(term, 1))
        cluster_ratio = cluster_count / n_clusters
        term_entropy = term_entropies.get(term, _entropy(term_cluster_weights.get(term, [1.0])))
        keyword_scopes.append(
            _keyword_scope(
                cluster_count,
                cluster_ratio,
                threshold=global_term_threshold,
            )
        )
        keyword_cluster_counts.append(cluster_count)
        keyword_cluster_ratios.append(float(cluster_ratio))

        if cluster_count == 1:
            flags.append("cluster_specific")
            factor = 1.0 + float(cluster_specific_bonus)
            multiplier *= factor
            adjustment_trace.append(
                _quality_adjustment(
                    "scope",
                    factor,
                    "cluster_specific",
                    cluster_count=cluster_count,
                    cluster_ratio=float(cluster_ratio),
                )
            )
        elif cluster_ratio >= float(global_term_threshold):
            flags.append("too_global")
            factor = 1.0 - float(global_term_penalty) * min(1.0, cluster_ratio)
            multiplier *= factor
            adjustment_trace.append(
                _quality_adjustment(
                    "scope",
                    factor,
                    "too_global",
                    cluster_count=cluster_count,
                    cluster_ratio=float(cluster_ratio),
                )
            )
            factor = 1.0 - float(entropy_penalty) * min(1.0, term_entropy)
            multiplier *= factor
            adjustment_trace.append(
                _quality_adjustment(
                    "cluster_entropy",
                    factor,
                    "distributed_across_clusters",
                    entropy=float(term_entropy),
                )
            )

        if _looks_like_metadata_artifact_term(term):
            flags.extend(["artifact_like", "metadata_fragment"])
            factor = 1.0 - float(artifact_demotion_weight)
            multiplier *= factor
            adjustment_trace.append(_quality_adjustment("term_shape", factor, "metadata_fragment"))
        elif term in METADATA_TERMS:
            flags.extend(["artifact_like", "low_information"])
            factor = 1.0 - float(artifact_demotion_weight)
            multiplier *= factor
            adjustment_trace.append(_quality_adjustment("term_shape", factor, "metadata_term"))
        elif term in LOW_INFORMATION_TERMS:
            flags.append("low_information")
            factor = 1.0 - max(0.0, float(artifact_demotion_weight) * 0.5)
            multiplier *= factor
            adjustment_trace.append(_quality_adjustment("term_shape", factor, "low_information_term"))

        expansion = expansions.get((cluster_id, term))
        corpus_abbreviation = _lookup_abbreviation_evidence(
            abbreviation_lookup,
            cluster_id=cluster_id,
            term=term,
        )
        material_formula_like = _is_material_formula_like(term)
        supported_roman_material_class = _is_supported_roman_material_class_phrase(toks)
        formula_like = material_formula_like or (
            _is_formula_like(term) and not _is_supported_numeric_event_phrase(toks)
            and not supported_roman_material_class
        )
        acronym_like = _is_acronym_like(
            term,
            max_length=acronym_max_length,
            has_expansion=bool(expansion or (corpus_abbreviation and corpus_abbreviation.get("long_form"))),
        )
        if formula_like:
            flags.append("formula_like")
            if material_formula_like:
                flags.append("material_formula")
                factor = 1.0 - min(0.1, float(formula_demotion_weight) * 0.2)
                multiplier *= factor
                adjustment_trace.append(_quality_adjustment("formula", factor, "material_formula"))
            else:
                flags.append("artifact_formula")
                factor = 1.0 - float(formula_demotion_weight)
                multiplier *= factor
                adjustment_trace.append(_quality_adjustment("formula", factor, "artifact_formula"))
        if acronym_like:
            flags.append("acronym_like")
            factor = 1.0 - float(acronym_demotion_weight)
            multiplier *= factor
            adjustment_trace.append(_quality_adjustment("short_form_shape", factor, "acronym_like"))

        network_role = str(role_hint.get("role", "neutral"))
        network_flags = [str(flag) for flag in role_hint.get("flags", []) if str(flag)]
        network_multiplier = float(role_hint.get("multiplier", 1.0))
        (
            abbreviation_status,
            abbreviation_target,
            abbreviation_confidence,
            abbreviation_source,
            abbreviation_support_docs,
            abbreviation_cluster_support_docs,
            abbreviation_top_support_ratio,
            abbreviation_ambiguity_type,
        ) = _abbreviation_evidence(
            term,
            expansion=expansion,
            corpus_evidence=corpus_abbreviation,
            network_role=network_role,
            network_flags=network_flags,
            acronym_max_length=acronym_max_length,
        )
        if (
            abbreviation_status in {"cluster_expanded", "corpus_expanded"}
            and abbreviation_target
            and (target_key := _label_key(abbreviation_target)) in term_keys_by_cluster.get(cluster_id, set())
            and target_key != _label_key(term)
        ):
            abbreviation_target = term_labels_by_cluster.get(cluster_id, {}).get(target_key, abbreviation_target)
            abbreviation_status = "duplicate_expansion"
        if abbreviation_status == "duplicate_expansion":
            flags.append("duplicate_label")
            multiplier *= 0.15
            adjustment_trace.append(
                _quality_adjustment(
                    "abbreviation",
                    0.15,
                    "duplicate_expansion",
                    target=abbreviation_target,
                    confidence=abbreviation_confidence,
                )
            )
        abbreviation_statuses.append(abbreviation_status)
        abbreviation_targets.append(abbreviation_target)
        abbreviation_confidences.append(abbreviation_confidence)
        abbreviation_sources.append(abbreviation_source)
        abbreviation_support_docs_values.append(abbreviation_support_docs)
        abbreviation_cluster_support_docs_values.append(abbreviation_cluster_support_docs)
        abbreviation_top_support_ratios.append(abbreviation_top_support_ratio)
        abbreviation_ambiguity_types.append(abbreviation_ambiguity_type)
        if network_roles_enabled:
            flags.extend(network_flags)
        if abbreviation_status in {
            "ambiguous_expansion",
            "candidate_short_form",
            "cluster_expanded",
            "corpus_expanded",
            "duplicate_expansion",
            "low_support_expansion",
            "unlinked_short_form",
        }:
            flags.append(abbreviation_status)
        flags.extend(_representative_artifact_flags(term, abbreviation_status=abbreviation_status))
        flags.extend(_representative_fragment_flags(term, longer_terms_by_cluster.get(cluster_id, [])))

        if n_tokens == 1:
            for longer in longer_terms_by_cluster.get(cluster_id, []):
                longer_tokens = set(_tokens(longer))
                if term in longer_tokens:
                    flags.append("phrase_preferred")
                    shadow_penalty = float(single_token_shadow_penalty)
                    if network_role == "anchor_unigram":
                        shadow_penalty *= 0.25
                    elif network_role == "linked_unigram":
                        shadow_penalty *= 0.55
                    factor = 1.0 - shadow_penalty
                    multiplier *= factor
                    adjustment_trace.append(
                        _quality_adjustment(
                            "phrase_shadow",
                            factor,
                            "longer_phrase_contains_unigram",
                            longer_phrase=longer,
                            network_role=network_role,
                        )
                    )
                    break
        elif n_tokens >= 2:
            flags.append("phrase")
            phrase_bonus = min(3, n_tokens - 1) / 3.0
            factor = 1.0 + float(phrase_preference_weight) * phrase_bonus
            multiplier *= factor
            adjustment_trace.append(
                _quality_adjustment(
                    "phrase_specificity",
                    factor,
                    "multi_token_phrase",
                    n_tokens=n_tokens,
                )
            )

        if not flags:
            flags.append("neutral")

        pre_clamp_multiplier = multiplier
        multiplier = max(float(min_multiplier), multiplier)
        if multiplier != pre_clamp_multiplier:
            adjustment_trace.append(
                _quality_adjustment(
                    "floor",
                    multiplier / pre_clamp_multiplier if pre_clamp_multiplier else multiplier,
                    "quality_min_multiplier",
                    min_multiplier=float(min_multiplier),
                )
            )
        if network_roles_enabled:
            network_factor = max(float(min_multiplier), network_multiplier)
            multiplier *= network_factor
            adjustment_trace.append(
                _quality_adjustment(
                    "network_role",
                    network_factor,
                    network_role,
                    network_score=float(role_hint.get("score", 0.5)),
                )
            )
            pre_clamp_multiplier = multiplier
            multiplier = max(float(min_multiplier), multiplier)
            if multiplier != pre_clamp_multiplier:
                adjustment_trace.append(
                    _quality_adjustment(
                        "floor",
                        multiplier / pre_clamp_multiplier if pre_clamp_multiplier else multiplier,
                        "quality_min_multiplier_after_network",
                        min_multiplier=float(min_multiplier),
                    )
                )
        base = float(base_scores.iloc[row_index])
        quality_score = base * multiplier
        representative_multiplier, representative_role, representative_flags = _representative_signal(
            term,
            n_tokens=n_tokens,
            flags=flags,
            keyword_scope=keyword_scopes[-1],
            network_role=network_role,
            abbreviation_status=abbreviation_status,
        )
        if abbreviation_status in {"cluster_expanded", "corpus_expanded", "duplicate_expansion"} and abbreviation_target:
            display_label = abbreviation_target
        elif expansion:
            display_label = expansion
        else:
            display_label = term
        keyword_label_tier = _keyword_label_tier(
            term=term,
            display_label=display_label,
            flags=flags,
            representative_role=representative_role,
            abbreviation_status=abbreviation_status,
        )
        quality_risk_family = _quality_risk_family(flags, keyword_label_tier=keyword_label_tier)
        quality_flag_basis = _quality_flag_basis(
            flags,
            abbreviation_status=abbreviation_status,
            network_role=network_role,
        )
        quality_flag_confidence = _quality_flag_confidence(
            flags,
            risk_family=quality_risk_family,
            flag_basis=quality_flag_basis,
            keyword_label_tier=keyword_label_tier,
        )
        clean_view_action = _clean_view_action(
            flags,
            keyword_label_tier=keyword_label_tier,
            risk_family=quality_risk_family,
        )
        representative_score = quality_score * representative_multiplier
        quality_scores.append(quality_score)
        quality_multipliers.append(multiplier)
        quality_flags.append(_flag_string(flags))
        quality_risk_families.append(quality_risk_family)
        quality_flag_bases.append(quality_flag_basis)
        quality_flag_confidences.append(quality_flag_confidence)
        clean_view_actions.append(clean_view_action)
        representative_scores.append(representative_score)
        representative_multipliers.append(representative_multiplier)
        representative_roles.append(representative_role)
        representative_flag_values.append(representative_flags)
        keyword_label_tiers.append(keyword_label_tier)
        network_role_values.append(network_role)
        network_scores.append(float(role_hint.get("score", 0.5)))
        network_flag_values.append(_flag_string(network_flags))
        display_labels.append(display_label)
        quality_decision_traces.append(
            _decision_trace_json(
                {
                    "term": term,
                    "display_label": display_label,
                    "base_score": round(base, 6),
                    "quality_score": round(float(quality_score), 6),
                    "quality_multiplier": round(float(multiplier), 6),
                    "quality_flags": sorted(set(flags)),
                    "quality_risk_family": quality_risk_family,
                    "quality_flag_basis": quality_flag_basis,
                    "quality_flag_confidence": quality_flag_confidence,
                    "clean_view_action": clean_view_action,
                    "quality_adjustments": adjustment_trace,
                    "keyword_scope": keyword_scopes[-1],
                    "keyword_cluster_count": cluster_count,
                    "keyword_cluster_ratio": round(float(cluster_ratio), 6),
                    "abbreviation_status": abbreviation_status,
                    "abbreviation_target": abbreviation_target,
                    "abbreviation_confidence": round(float(abbreviation_confidence), 6),
                    "network_role": network_role,
                    "network_score": round(float(role_hint.get("score", 0.5)), 6),
                    "representative_score": round(float(representative_score), 6),
                    "representative_multiplier": round(float(representative_multiplier), 6),
                    "representative_role": representative_role,
                    "representative_flags": representative_flags.split("|") if representative_flags else [],
                    "keyword_label_tier": keyword_label_tier,
                }
            )
        )

    family_support = _compute_family_representative_support(
        cluster_ids=row_cluster_ids,
        display_labels=display_labels,
        representative_scores=representative_scores,
        doc_coverages=doc_coverage_values,
        family_terms=_collect_family_evidence_terms(
            out,
            term_col=term_col,
            display_labels=display_labels,
        ),
        enabled=family_representative_enabled,
        weight=family_representative_weight,
        max_bonus=family_representative_max_bonus,
    )
    representative_family_child_counts: list[int] = []
    representative_family_member_counts: list[int] = []
    representative_family_avg_child_coverages: list[float] = []
    representative_family_multipliers: list[float] = []
    for idx, info in enumerate(family_support):
        family_multiplier = float(info["multiplier"])
        representative_scores[idx] *= family_multiplier
        representative_multipliers[idx] *= family_multiplier
        child_count = int(info["child_count"])
        member_count = int(info["member_count"])
        avg_child_coverage = float(info["avg_child_coverage"])
        representative_family_child_counts.append(child_count)
        representative_family_member_counts.append(member_count)
        representative_family_avg_child_coverages.append(avg_child_coverage)
        representative_family_multipliers.append(family_multiplier)

        trace = json.loads(quality_decision_traces[idx])
        trace["representative_score"] = round(float(representative_scores[idx]), 6)
        trace["representative_multiplier"] = round(float(representative_multipliers[idx]), 6)
        trace["representative_family_support"] = {
            "child_count": child_count,
            "member_count": member_count,
            "avg_child_coverage": round(avg_child_coverage, 6),
            "multiplier": round(family_multiplier, 6),
            "children": info["children"],
        }
        if family_multiplier > 1.0:
            trace.setdefault("quality_adjustments", []).append(
                _quality_adjustment(
                    "representative_family_support",
                    family_multiplier,
                    "parent_label_has_derivatives_or_aliases",
                    child_count=child_count,
                    member_count=member_count,
                    avg_child_coverage=avg_child_coverage,
                )
            )
        quality_decision_traces[idx] = _decision_trace_json(trace)

    out["display_label"] = display_labels
    out["quality_score"] = quality_scores
    out["quality_multiplier"] = quality_multipliers
    out["quality_flags"] = quality_flags
    out["quality_risk_family"] = quality_risk_families
    out["quality_flag_basis"] = quality_flag_bases
    out["quality_flag_confidence"] = quality_flag_confidences
    out["clean_view_action"] = clean_view_actions
    out["quality_decision_trace"] = quality_decision_traces
    out["representative_score"] = representative_scores
    out["representative_multiplier"] = representative_multipliers
    out["representative_role"] = representative_roles
    out["representative_flags"] = representative_flag_values
    out["keyword_label_tier"] = keyword_label_tiers
    out["representative_family_child_count"] = representative_family_child_counts
    out["representative_family_member_count"] = representative_family_member_counts
    out["representative_family_avg_child_coverage"] = representative_family_avg_child_coverages
    out["representative_family_multiplier"] = representative_family_multipliers
    out["keyword_scope"] = keyword_scopes
    out["keyword_cluster_count"] = keyword_cluster_counts
    out["keyword_cluster_ratio"] = keyword_cluster_ratios
    out["abbreviation_status"] = abbreviation_statuses
    out["abbreviation_target"] = abbreviation_targets
    out["abbreviation_confidence"] = abbreviation_confidences
    out["abbreviation_source"] = abbreviation_sources
    out["abbreviation_support_docs"] = abbreviation_support_docs_values
    out["abbreviation_cluster_support_docs"] = abbreviation_cluster_support_docs_values
    out["abbreviation_top_support_ratio"] = abbreviation_top_support_ratios
    out["abbreviation_ambiguity_type"] = abbreviation_ambiguity_types
    if network_roles_enabled:
        out["network_role"] = network_role_values
        out["network_score"] = network_scores
        out["network_flags"] = network_flag_values
    if cluster_col in out.columns:
        out["representative_rank"] = (
            out.groupby(cluster_col, sort=False)["representative_score"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
    else:
        out["representative_rank"] = (
            out["representative_score"]
            .rank(method="first", ascending=False)
            .astype(int)
        )

    if rerank and cluster_col in out.columns:
        out = (
            out.sort_values(
                [cluster_col, "quality_score", score_col],
                ascending=[True, False, False],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
    return out


def quality_flag_counts(df: pd.DataFrame, *, flag_col: str = "quality_flags") -> dict[str, int]:
    """Return counts for pipe-delimited quality flags."""

    counts: Counter[str] = Counter()
    if df is None or df.empty or flag_col not in df.columns:
        return {}
    for raw in df[flag_col].dropna().astype(str):
        for flag in raw.split("|"):
            flag = flag.strip()
            if flag:
                counts[flag] += 1
    return dict(counts)


def _residual_weak_label_reasons(row: Mapping[str, Any]) -> list[str]:
    term = _normalise_term(row.get("term", row.get("display_label", "")))
    tokens = tuple(_tokens(term))
    flags = {
        flag.strip()
        for flag in str(row.get("quality_flags", "")).split("|")
        if flag.strip()
    }
    tier = str(row.get("keyword_label_tier", ""))
    reasons: list[str] = []
    if tier.startswith("review_"):
        reasons.append(tier)
    if "broad_semantic_head_fragment" in flags or tokens in BROAD_SEMANTIC_HEAD_PHRASES:
        reasons.append("broad_semantic_head")
    if tokens in DOCUMENT_GENRE_REVIEW_PHRASES:
        reasons.append("document_genre_review_phrase")
    if tokens in TITLE_LIKE_STORY_PHRASES:
        reasons.append("title_like_story_phrase")
    return sorted(set(reasons))


def keyword_quality_residual_report(
    df: pd.DataFrame,
    *,
    top_rank: int = 50,
    max_rows: int = 200,
) -> dict[str, Any]:
    """Summarise weak final-label candidates without changing keyword rows."""

    if df is None or df.empty:
        return {
            "schema_version": "sciscape_keyword_quality_residual_report_v1",
            "top_rank": int(top_rank),
            "rows_scanned": 0,
            "flagged_rows": 0,
            "reason_counts": {},
            "rows": [],
        }

    rows: list[dict[str, Any]] = []
    rank_col = "rank" if "rank" in df.columns else None
    scan_df = df
    if rank_col:
        ranks = pd.to_numeric(df[rank_col], errors="coerce")
        scan_df = df[ranks <= int(top_rank)].copy()

    for row in scan_df.to_dict(orient="records"):
        reasons = _residual_weak_label_reasons(row)
        if not reasons:
            continue
        rows.append(
            {
                "cluster_id": row.get("cluster_id"),
                "rank": int(_safe_float(row.get("rank", 0.0))) if row.get("rank") is not None else None,
                "term": str(row.get("term", "")),
                "display_label": str(row.get("display_label", row.get("term", ""))),
                "reasons": reasons,
                "keyword_label_tier": str(row.get("keyword_label_tier", "")),
                "quality_flags": str(row.get("quality_flags", "")),
                "quality_risk_family": str(row.get("quality_risk_family", "")),
                "quality_flag_basis": str(row.get("quality_flag_basis", "")),
                "quality_flag_confidence": str(row.get("quality_flag_confidence", "")),
                "clean_view_action": str(row.get("clean_view_action", "")),
                "doc_coverage": int(_safe_float(row.get("doc_coverage", row.get("frequency", 0.0)))),
                "cluster_df": int(_safe_float(row.get("cluster_df", row.get("keyword_cluster_count", 0.0)))),
            }
        )

    rows.sort(key=lambda item: (item.get("rank") or 999_999, str(item.get("cluster_id")), item["term"]))
    if int(max_rows) > 0:
        rows = rows[: int(max_rows)]

    reason_counts: Counter[str] = Counter()
    for row in rows:
        reason_counts.update(row["reasons"])

    return {
        "schema_version": "sciscape_keyword_quality_residual_report_v1",
        "top_rank": int(top_rank),
        "rows_scanned": int(len(scan_df)),
        "flagged_rows": int(len(rows)),
        "reason_counts": dict(sorted(reason_counts.items())),
        "rows": rows,
    }


def write_keyword_quality_residual_report(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    top_rank: int = 50,
    max_rows: int = 200,
) -> dict[str, str]:
    """Write residual weak-label JSON and Markdown summaries."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report = keyword_quality_residual_report(df, top_rank=top_rank, max_rows=max_rows)
    json_path = output_path / "keyword_quality_residual_report.json"
    md_path = output_path / "keyword_quality_residual_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Keyword Quality Residual Report",
        "",
        f"- Top-rank window: {report['top_rank']}",
        f"- Rows scanned: {report['rows_scanned']}",
        f"- Flagged rows: {report['flagged_rows']}",
        "",
        "## Reason Counts",
        "",
    ]
    reason_counts = report.get("reason_counts", {})
    if reason_counts:
        for reason, count in reason_counts.items():
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| cluster_id | rank | term | reasons | tier | risk_family | basis | confidence | action | doc_coverage | cluster_df |",
            "| --- | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |",
        ]
    )

    def _md_cell(value: Any) -> str:
        return str(value).replace("|", "\\|")

    for row in report.get("rows", []):
        reasons = ", ".join(f"`{reason}`" for reason in row["reasons"])
        lines.append(
            f"| {row['cluster_id']} | {row['rank']} | {_md_cell(row['term'])} | {reasons} | "
            f"{_md_cell(row['keyword_label_tier'])} | {_md_cell(row.get('quality_risk_family', ''))} | "
            f"{_md_cell(row.get('quality_flag_basis', ''))} | {_md_cell(row.get('quality_flag_confidence', ''))} | "
            f"{_md_cell(row.get('clean_view_action', ''))} | {row['doc_coverage']} | {row['cluster_df']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


__all__ = [
    "METADATA_TERMS",
    "LOW_INFORMATION_TERMS",
    "annotate_keyword_quality",
    "keyword_quality_residual_report",
    "quality_flag_counts",
    "write_keyword_quality_residual_report",
]
