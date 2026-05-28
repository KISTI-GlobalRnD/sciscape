"""Corpus-level abbreviation evidence from parenthetical definitions.

The extractor is intentionally conservative.  It only keeps parenthetical
definitions whose initials match the short form, and it returns auditable
support counts rather than applying unconditional text rewrites.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Any, Iterable, Iterator, Mapping

import pandas as pd

from .normalization import _normalize_notation, _normalize_spelling, _phrase_singular
from .utils import _normalize_text_basic


_PAREN_RE = re.compile(r"\(([^()]{2,120})\)")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SHORT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9\-/]{1,11}s?$")
_STOP_INITIAL_TOKENS = frozenset(
    {"a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "of", "on", "or", "the", "to", "via", "with"}
)
_BAD_SHORT_FORMS = frozenset({"al", "eg", "eq", "et", "fig", "ie", "no", "pp", "ref", "sec", "vs"})


def _normalise_short_form(text: object) -> str:
    short = re.sub(r"[^A-Za-z0-9]", "", "" if text is None else str(text).strip())
    if len(short) > 3 and short.lower().endswith("s"):
        short = short[:-1]
    return short.lower()


def _tokens(text: object) -> list[str]:
    return _TOKEN_RE.findall("" if text is None else str(text))


def _is_short_form(text: object) -> bool:
    raw = "" if text is None else str(text).strip().strip(".;:,/\\[]{}")
    if " " in raw or not _SHORT_RE.match(raw):
        return False
    short = _normalise_short_form(raw)
    if len(short) < 2 or len(short) > 10 or short in _BAD_SHORT_FORMS:
        return False
    letters = [ch for ch in raw if ch.isalpha()]
    return bool(letters) and any(ch.isupper() for ch in letters)


def _clean_long_form(text: object) -> str:
    cleaned = _normalize_text_basic(text).strip(" ;:,.")
    cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split())


def _normalise_long_form(text: object) -> str:
    long_form = _clean_long_form(text).lower()
    long_form = _normalize_notation(long_form)
    long_form = _normalize_spelling(long_form)
    return " ".join(long_form.split())


def _long_form_key(text: object) -> str:
    long_form = _normalise_long_form(text)
    singular = _phrase_singular(long_form)
    return singular or long_form


def _valid_long_form(text: object) -> bool:
    words = [word for word in _tokens(text) if re.search(r"[A-Za-z]", word)]
    return 2 <= len(words) <= 12 and sum(len(word) for word in words) >= 6


def _initials(words: Iterable[str], *, drop_stopwords: bool) -> str:
    chars: list[str] = []
    for word in words:
        lower = word.lower()
        if drop_stopwords and lower in _STOP_INITIAL_TOKENS:
            continue
        if re.search(r"[A-Za-z]", word):
            chars.append(lower[0])
    return "".join(chars)


def _long_form_matches_short(short_form: object, long_form: object) -> bool:
    short = _normalise_short_form(short_form)
    words = _tokens(long_form)
    if not short or not _valid_long_form(long_form):
        return False
    return short in {
        _initials(words, drop_stopwords=False),
        _initials(words, drop_stopwords=True),
    }


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(_tokens(left.lower()))
    right_tokens = set(_tokens(right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _candidate_ambiguity_type(
    short_form: str,
    ranked_entries: list[tuple[tuple[str, str], int]],
) -> str:
    if len(ranked_entries) <= 1:
        return "none"
    top_key = ranked_entries[0][0][1]
    alternatives = [key[1] for key, _ in ranked_entries[1:]]
    if alternatives and all(_token_overlap(top_key, other) >= 0.8 for other in alternatives):
        return "variant"
    return "semantic"


def _suffix_candidates(prefix: str, *, max_words: int) -> Iterator[str]:
    words = _tokens(prefix)[-max_words:]
    for n_words in range(2, min(max_words, len(words)) + 1):
        yield _clean_long_form(" ".join(words[-n_words:]))


def _extract_pairs_from_text(text: str, *, max_long_form_words: int) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    if not text:
        return pairs

    text = _normalize_text_basic(text)
    for match in _PAREN_RE.finditer(text):
        inside = _clean_long_form(match.group(1))
        prefix = text[max(0, match.start() - 260):match.start()]
        suffix = text[match.end():match.end() + 180]

        if _is_short_form(inside):
            for long_form in _suffix_candidates(prefix, max_words=max_long_form_words):
                if _long_form_matches_short(inside, long_form):
                    pairs.append(
                        (
                            _normalise_short_form(inside),
                            _normalise_long_form(long_form),
                            "long_before_short_in_parens",
                        )
                    )
                    break

            after_words = _tokens(suffix)[:max_long_form_words]
            for n_words in range(2, min(max_long_form_words, len(after_words)) + 1):
                long_form = _clean_long_form(" ".join(after_words[:n_words]))
                if _long_form_matches_short(inside, long_form):
                    pairs.append(
                        (
                            _normalise_short_form(inside),
                            _normalise_long_form(long_form),
                            "short_in_parens_long_after",
                        )
                    )
                    break
        elif " " in inside and _valid_long_form(inside):
            previous = _tokens(prefix[-80:])[-1:] if prefix else []
            if previous and _is_short_form(previous[-1]) and _long_form_matches_short(previous[-1], inside):
                pairs.append(
                    (
                        _normalise_short_form(previous[-1]),
                        _normalise_long_form(inside),
                        "short_before_long_in_parens",
                    )
                )
    return pairs


def _iter_docs(docs: pd.DataFrame | Iterable[pd.DataFrame]) -> Iterator[pd.Series]:
    if isinstance(docs, pd.DataFrame):
        for _, row in docs.iterrows():
            yield row
        return
    for batch in docs:
        for _, row in batch.iterrows():
            yield row


def extract_parenthetical_abbreviations(
    docs: pd.DataFrame | Iterable[pd.DataFrame],
    *,
    uid_col: str = "uid",
    cluster_col: str | None = "cluster_id",
    title_col: str = "title",
    abstract_col: str = "abstract",
    max_long_form_words: int = 12,
) -> pd.DataFrame:
    """Extract abbreviation evidence from title/abstract parenthetical patterns."""

    pair_docs: dict[tuple[str, str], set[str]] = defaultdict(set)
    pair_occurrences: Counter[tuple[str, str]] = Counter()
    pair_patterns: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    pair_clusters: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    pair_long_forms: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    for row in _iter_docs(docs):
        uid = str(row.get(uid_col, ""))
        if not uid:
            continue
        title = row.get(title_col, "")
        abstract = row.get(abstract_col, "")
        text = f"{title or ''}. {abstract or ''}"
        seen_doc_pairs = set(_extract_pairs_from_text(text, max_long_form_words=max_long_form_words))
        if not seen_doc_pairs:
            continue

        cluster_id: int | None = None
        if cluster_col and cluster_col in row.index and pd.notna(row.get(cluster_col)):
            try:
                cluster_id = int(row.get(cluster_col))
            except (TypeError, ValueError):
                cluster_id = None

        for short_form, long_form, pattern_type in seen_doc_pairs:
            long_key = _long_form_key(long_form)
            key = (short_form, long_key)
            pair_docs[key].add(uid)
            pair_occurrences[key] += 1
            pair_patterns[key][pattern_type] += 1
            pair_long_forms[key][long_form] += 1
            if cluster_id is not None:
                pair_clusters[key][cluster_id] += 1

    if not pair_docs:
        return pd.DataFrame(
            columns=[
                "short_form",
                "long_form",
                "long_form_key",
                "support_docs",
                "support_occurrences",
                "cluster_supports",
                "pattern_types",
                "raw_long_forms",
                "candidate_rank",
                "short_form_candidate_count",
                "top_support_docs",
                "top_support_ratio",
                "is_ambiguous",
                "ambiguity_type",
                "confidence",
            ]
        )

    short_totals: dict[str, int] = defaultdict(int)
    short_candidates: dict[str, list[tuple[tuple[str, str], int]]] = defaultdict(list)
    for key, docs_for_pair in pair_docs.items():
        support = len(docs_for_pair)
        short_totals[key[0]] += support
        short_candidates[key[0]].append((key, support))

    ranks: dict[tuple[str, str], int] = {}
    top_supports: dict[str, int] = {}
    candidate_counts: dict[str, int] = {}
    ambiguity_types: dict[str, str] = {}
    for short_form, entries in short_candidates.items():
        ranked = sorted(entries, key=lambda item: (-item[1], item[0][1]))
        candidate_counts[short_form] = len(ranked)
        top_supports[short_form] = ranked[0][1]
        ambiguity_types[short_form] = _candidate_ambiguity_type(short_form, ranked)
        for rank, (key, _) in enumerate(ranked, start=1):
            ranks[key] = rank

    rows: list[dict[str, Any]] = []
    for key, docs_for_pair in pair_docs.items():
        short_form, long_key = key
        support_docs = len(docs_for_pair)
        raw_long_forms = pair_long_forms[key]
        display_long = raw_long_forms.most_common(1)[0][0] if raw_long_forms else long_key
        total_for_short = max(1, short_totals[short_form])
        top_support = top_supports[short_form]
        top_ratio = top_support / total_for_short
        confidence = min(1.0, 0.45 + 0.25 * min(1.0, math.log1p(support_docs) / math.log(10)) + 0.3 * top_ratio)
        rows.append(
            {
                "short_form": short_form,
                "long_form": display_long,
                "long_form_key": long_key,
                "support_docs": int(support_docs),
                "support_occurrences": int(pair_occurrences[key]),
                "cluster_supports": dict(sorted(pair_clusters[key].items())),
                "pattern_types": "|".join(sorted(pair_patterns[key])),
                "raw_long_forms": "|".join(
                    form for form, _ in sorted(raw_long_forms.items(), key=lambda item: (-item[1], item[0]))
                ),
                "candidate_rank": int(ranks[key]),
                "short_form_candidate_count": int(candidate_counts[short_form]),
                "top_support_docs": int(top_support),
                "top_support_ratio": float(top_ratio),
                "is_ambiguous": bool(candidate_counts[short_form] > 1),
                "ambiguity_type": ambiguity_types.get(short_form, "none"),
                "confidence": float(confidence),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["short_form", "candidate_rank", "support_docs", "long_form"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_abbreviation_lookup(
    evidence: pd.DataFrame | None,
    *,
    min_support_docs: int = 2,
    min_cluster_support_docs: int = 2,
    min_top_support_ratio: float = 0.75,
) -> dict[str, Any]:
    """Build global and cluster-specific lookup maps from evidence rows."""

    lookup: dict[str, Any] = {"global": {}, "cluster": {}}
    if evidence is None or evidence.empty:
        return lookup

    required = {"short_form", "long_form", "support_docs", "candidate_rank", "top_support_ratio"}
    if not required.issubset(evidence.columns):
        return lookup

    for row in evidence.itertuples(index=False):
        short = str(getattr(row, "short_form"))
        if int(getattr(row, "candidate_rank")) != 1:
            continue
        support = int(getattr(row, "support_docs"))
        ratio = float(getattr(row, "top_support_ratio"))
        ambiguous = bool(getattr(row, "is_ambiguous", False))
        ambiguity_type = str(getattr(row, "ambiguity_type", "semantic" if ambiguous else "none"))
        usable_global = support >= int(min_support_docs) and (
            not ambiguous or ratio >= float(min_top_support_ratio)
        )
        if usable_global:
            status = "corpus_expanded"
        elif support < int(min_support_docs):
            status = "low_support_expansion"
        else:
            status = "ambiguous_expansion"
        lookup["global"][short] = {
            "long_form": str(getattr(row, "long_form")),
            "support_docs": support,
            "cluster_support_docs": 0,
            "confidence": float(getattr(row, "confidence", 0.0)),
            "is_ambiguous": ambiguous,
            "ambiguity_type": ambiguity_type,
            "top_support_ratio": ratio,
            "status": status,
            "usable": usable_global,
        }

    for short_form, group in evidence.groupby("short_form", sort=False):
        cluster_best: dict[int, list[tuple[int, Any]]] = defaultdict(list)
        for row in group.itertuples(index=False):
            supports = getattr(row, "cluster_supports", {})
            if not isinstance(supports, Mapping):
                continue
            for cluster_id, count in supports.items():
                try:
                    cid = int(cluster_id)
                    n_docs = int(count)
                except (TypeError, ValueError):
                    continue
                if n_docs >= int(min_cluster_support_docs):
                    cluster_best[cid].append((n_docs, row))

        for cid, entries in cluster_best.items():
            ranked = sorted(entries, key=lambda item: (-item[0], str(getattr(item[1], "long_form"))))
            best_count, best_row = ranked[0]
            tie = len(ranked) > 1 and ranked[1][0] == best_count
            if tie:
                continue
            support = int(getattr(best_row, "support_docs"))
            ratio = float(getattr(best_row, "top_support_ratio"))
            lookup["cluster"][(cid, str(short_form))] = {
                "long_form": str(getattr(best_row, "long_form")),
                "support_docs": support,
                "cluster_support_docs": int(best_count),
                "confidence": max(0.0, min(1.0, float(getattr(best_row, "confidence", 0.0)) + 0.05)),
                "is_ambiguous": bool(getattr(best_row, "is_ambiguous", False)),
                "ambiguity_type": str(getattr(best_row, "ambiguity_type", "none")),
                "top_support_ratio": ratio,
                "status": "cluster_expanded",
                "usable": True,
            }

    return lookup


__all__ = [
    "build_abbreviation_lookup",
    "extract_parenthetical_abbreviations",
]
