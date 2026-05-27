"""Domain-agnostic keyword quality annotation and reranking.

The helpers in this module avoid domain dictionaries.  They score terms by
how useful they are as cluster-facing labels: cluster concentration, phrase
specificity, redundancy with longer phrases, and artifact-like shape.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


METADATA_TERMS: frozenset[str] = frozenset(
    {
        "abstract",
        "introduction",
        "background",
        "conclusion",
        "conclusions",
        "copyright",
        "keyword",
        "keywords",
        "paper",
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
        "system",
        "systems",
        "technique",
        "work",
    }
)

_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_ALNUM_RE = re.compile(r"(?=.*[a-z])(?=.*\d)[a-z0-9]+")
_SHORT_ALPHA_RE = re.compile(r"^[a-z]{2,8}$")


def _normalise_term(term: object) -> str:
    text = "" if term is None else str(term).strip().lower()
    text = text.replace("-", " ")
    return " ".join(text.split())


def _tokens(term: str) -> list[str]:
    return [tok for tok in _SPLIT_RE.split(term.lower()) if tok]


def _is_formula_like(term: str) -> bool:
    tokens = _tokens(term)
    if not tokens:
        return False
    if any(_ALNUM_RE.match(tok) for tok in tokens):
        return True
    short_tokens = sum(1 for tok in tokens if len(tok) <= 2)
    long_dense_tokens = sum(1 for tok in tokens if len(tok) >= 5 and re.search(r"[bcfhiknopsuvwyz]{4,}", tok))
    return short_tokens > 0 and long_dense_tokens > 0


def _is_acronym_like(term: str, *, max_length: int, has_expansion: bool = False) -> bool:
    compact = term.replace(" ", "")
    if not bool(_SHORT_ALPHA_RE.match(compact)) or len(compact) > int(max_length):
        return False
    if has_expansion:
        return True
    return len(compact) <= 4 and not any(ch in compact for ch in "aeou")


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


def annotate_keyword_quality(
    df: pd.DataFrame,
    *,
    term_col: str = "term",
    cluster_col: str = "cluster_id",
    score_col: str = "score",
    frequency_col: str = "frequency",
    doc_coverage_col: str = "doc_coverage",
    rerank: bool = False,
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
    n_clusters = max(1, len(cluster_ids))

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

    longer_terms_by_cluster: dict[Any, list[str]] = defaultdict(list)
    if cluster_col in out.columns:
        grouped = out.groupby(cluster_col, sort=False)
    else:
        grouped = [(0, out)]
    for cluster_id, group in grouped:
        longer_terms_by_cluster[cluster_id] = [
            str(term)
            for term in group["normalized_term"].tolist()
            if len(_tokens(str(term))) >= 2
        ]

    expansions = _find_phrase_expansions(out, term_col="normalized_term", cluster_col=cluster_col) if cluster_col in out.columns else {}

    quality_scores: list[float] = []
    quality_multipliers: list[float] = []
    quality_flags: list[str] = []
    display_labels: list[str] = []

    raw_scores = out[score_col] if score_col in out.columns else pd.Series(1.0, index=out.index)
    base_scores = pd.to_numeric(raw_scores, errors="coerce").fillna(0.0).reset_index(drop=True)

    for row_index, (_, row) in enumerate(out.iterrows()):
        term = str(row["normalized_term"])
        toks = _tokens(term)
        n_tokens = len(toks)
        flags: list[str] = []
        multiplier = 1.0

        cluster_count = int(term_cluster_counts.get(term, 1))
        cluster_ratio = cluster_count / n_clusters
        term_entropy = _entropy(term_cluster_weights.get(term, [1.0]))

        if cluster_count == 1:
            flags.append("cluster_specific")
            multiplier *= 1.0 + float(cluster_specific_bonus)
        elif cluster_ratio >= float(global_term_threshold):
            flags.append("too_global")
            multiplier *= 1.0 - float(global_term_penalty) * min(1.0, cluster_ratio)
            multiplier *= 1.0 - float(entropy_penalty) * min(1.0, term_entropy)

        if term in METADATA_TERMS:
            flags.extend(["artifact_like", "low_information"])
            multiplier *= 1.0 - float(artifact_demotion_weight)
        elif term in LOW_INFORMATION_TERMS:
            flags.append("low_information")
            multiplier *= 1.0 - max(0.0, float(artifact_demotion_weight) * 0.5)

        expansion = expansions.get((cluster_id, term))
        formula_like = _is_formula_like(term)
        acronym_like = _is_acronym_like(
            term,
            max_length=acronym_max_length,
            has_expansion=bool(expansion),
        )
        if formula_like:
            flags.append("formula_like")
            multiplier *= 1.0 - float(formula_demotion_weight)
        if acronym_like:
            flags.append("acronym_like")
            multiplier *= 1.0 - float(acronym_demotion_weight)

        cluster_id = row[cluster_col] if cluster_col in out.columns else 0
        if n_tokens == 1:
            for longer in longer_terms_by_cluster.get(cluster_id, []):
                longer_tokens = set(_tokens(longer))
                if term in longer_tokens:
                    flags.append("phrase_preferred")
                    multiplier *= 1.0 - float(single_token_shadow_penalty)
                    break
        elif n_tokens >= 2:
            flags.append("phrase")
            phrase_bonus = min(3, n_tokens - 1) / 3.0
            multiplier *= 1.0 + float(phrase_preference_weight) * phrase_bonus

        if not flags:
            flags.append("neutral")

        multiplier = max(float(min_multiplier), multiplier)
        base = float(base_scores.iloc[row_index])
        quality_scores.append(base * multiplier)
        quality_multipliers.append(multiplier)
        quality_flags.append(_flag_string(flags))

        if expansion:
            display_labels.append(expansion)
        else:
            display_labels.append(term)

    out["display_label"] = display_labels
    out["quality_score"] = quality_scores
    out["quality_multiplier"] = quality_multipliers
    out["quality_flags"] = quality_flags

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


__all__ = [
    "METADATA_TERMS",
    "LOW_INFORMATION_TERMS",
    "annotate_keyword_quality",
    "quality_flag_counts",
]
