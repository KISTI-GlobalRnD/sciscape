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

from .normalization import _normalize_notation, _normalize_spelling, _phrase_singular


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

_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_ALNUM_RE = re.compile(r"(?=.*[a-z])(?=.*\d)[a-z0-9]+")
_SHORT_ALPHA_RE = re.compile(r"^[a-z]{2,8}$")
_DIMENSION_TOKENS: frozenset[str] = frozenset({"0d", "1d", "2d", "3d", "4d"})


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
    if len(base) <= 3 and len(set(base)) < len(base):
        return True
    return False


def _is_formula_like(term: str) -> bool:
    tokens = _tokens(term)
    if not tokens:
        return False
    if any(_ALNUM_RE.match(tok) and tok not in _DIMENSION_TOKENS for tok in tokens):
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


def _keyword_scope(cluster_count: int, cluster_ratio: float, *, threshold: float) -> str:
    if cluster_count <= 1:
        return "cluster_specific"
    if cluster_ratio >= float(threshold):
        return "common"
    return "shared"


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

    raw_scores = out[score_col] if score_col in out.columns else pd.Series(1.0, index=out.index)
    base_scores = pd.to_numeric(raw_scores, errors="coerce").fillna(0.0).reset_index(drop=True)

    for row_index, (_, row) in enumerate(out.iterrows()):
        term = str(row["normalized_term"])
        toks = _tokens(term)
        n_tokens = len(toks)
        cluster_id = row[cluster_col] if cluster_col in out.columns else 0
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

        cluster_count = int(term_cluster_counts.get(term, 1))
        cluster_ratio = cluster_count / n_clusters
        term_entropy = _entropy(term_cluster_weights.get(term, [1.0]))
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
        corpus_abbreviation = _lookup_abbreviation_evidence(
            abbreviation_lookup,
            cluster_id=cluster_id,
            term=term,
        )
        formula_like = _is_formula_like(term)
        acronym_like = _is_acronym_like(
            term,
            max_length=acronym_max_length,
            has_expansion=bool(expansion or (corpus_abbreviation and corpus_abbreviation.get("long_form"))),
        )
        if formula_like:
            flags.append("formula_like")
            multiplier *= 1.0 - float(formula_demotion_weight)
        if acronym_like:
            flags.append("acronym_like")
            multiplier *= 1.0 - float(acronym_demotion_weight)

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
                    multiplier *= 1.0 - shadow_penalty
                    break
        elif n_tokens >= 2:
            flags.append("phrase")
            phrase_bonus = min(3, n_tokens - 1) / 3.0
            multiplier *= 1.0 + float(phrase_preference_weight) * phrase_bonus

        if not flags:
            flags.append("neutral")

        multiplier = max(float(min_multiplier), multiplier)
        if network_roles_enabled:
            multiplier *= max(float(min_multiplier), network_multiplier)
            multiplier = max(float(min_multiplier), multiplier)
        base = float(base_scores.iloc[row_index])
        quality_scores.append(base * multiplier)
        quality_multipliers.append(multiplier)
        quality_flags.append(_flag_string(flags))
        network_role_values.append(network_role)
        network_scores.append(float(role_hint.get("score", 0.5)))
        network_flag_values.append(_flag_string(network_flags))

        if abbreviation_status in {"cluster_expanded", "corpus_expanded", "duplicate_expansion"} and abbreviation_target:
            display_labels.append(abbreviation_target)
        elif expansion:
            display_labels.append(expansion)
        else:
            display_labels.append(term)

    out["display_label"] = display_labels
    out["quality_score"] = quality_scores
    out["quality_multiplier"] = quality_multipliers
    out["quality_flags"] = quality_flags
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
