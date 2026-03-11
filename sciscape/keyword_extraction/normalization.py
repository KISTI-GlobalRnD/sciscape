"""Post-top-K keyword-level normalization (Stage 5).

Operates on the scored keyword DataFrame after top-K selection.
Handles abbreviation expansion, notation normalization, and
frequency-based heuristic merging of near-duplicate terms.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Set

import pandas as pd


def _expand_abbreviations(
    term: str,
    builtin_aliases: Mapping[str, str],
) -> str:
    """Expand known abbreviations to their canonical forms."""
    lower = term.lower().strip()
    if lower in builtin_aliases:
        return builtin_aliases[lower]
    return term


def _normalize_notation(term: str) -> str:
    """Normalize Greek letters, units, and common notation variants."""
    replacements = {
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "δ": "delta",
        "ε": "epsilon",
        "ζ": "zeta",
        "η": "eta",
        "θ": "theta",
        "λ": "lambda",
        "μ": "mu",
        "π": "pi",
        "σ": "sigma",
        "τ": "tau",
        "φ": "phi",
        "ω": "omega",
    }
    result = term
    for greek, latin in replacements.items():
        if greek in result:
            # "γ-ray" -> "gamma-ray" -> "gamma ray"
            result = result.replace(greek, latin)
    # Normalize remaining hyphens to spaces
    result = result.replace("-", " ")
    # Collapse whitespace
    result = " ".join(result.split())
    return result


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _edit_distance(b, a)
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


def normalize_keywords(
    top_df: pd.DataFrame,
    builtin_aliases: Mapping[str, str],
    stopwords: Optional[Set[str]] = None,
    max_edit_distance: int = 2,
    min_frequency_ratio: float = 0.01,
) -> pd.DataFrame:
    """Post-top-K keyword normalization.

    Steps:
    1. Expand known abbreviations (builtin_aliases)
    2. Normalize notation (Greek letters, hyphens)
    3. Merge near-duplicates by edit distance within each cluster

    Returns a new DataFrame with normalized terms and merged frequencies.
    """
    if top_df.empty:
        return top_df

    result_rows = []

    for cluster_id, group in top_df.groupby("cluster_id"):
        terms = group["term"].tolist()
        scores = group["score"].tolist()
        freqs = group["frequency"].tolist()

        # Step 1+2: normalize each term
        normalized: Dict[int, str] = {}
        for i, term in enumerate(terms):
            t = _expand_abbreviations(term, builtin_aliases)
            t = _normalize_notation(t)
            if stopwords and t.lower() in stopwords:
                t = term  # revert if normalization produced a stopword
            normalized[i] = t.strip()

        # Step 3: merge near-duplicates (greedy, high-freq absorbs low-freq)
        # Sort by frequency descending so higher-freq terms are canonical
        order = sorted(range(len(terms)), key=lambda i: -freqs[i])
        merged_into: Dict[int, int] = {}  # source_idx -> target_idx

        for i in range(len(order)):
            idx_i = order[i]
            if idx_i in merged_into:
                continue
            term_i = normalized[idx_i]
            for j in range(i + 1, len(order)):
                idx_j = order[j]
                if idx_j in merged_into:
                    continue
                term_j = normalized[idx_j]
                if term_i == term_j:
                    # Exact match after normalization
                    merged_into[idx_j] = idx_i
                    continue
                if max_edit_distance > 0 and len(term_i) > 3 and len(term_j) > 3:
                    dist = _edit_distance(term_i.lower(), term_j.lower())
                    if dist <= max_edit_distance:
                        # Check frequency ratio to avoid merging distinct terms
                        if freqs[idx_j] <= min_frequency_ratio * freqs[idx_i]:
                            merged_into[idx_j] = idx_i

        # Build merged output
        canonical_freqs: Dict[int, int] = {}
        canonical_scores: Dict[int, float] = {}
        for i in range(len(terms)):
            target = merged_into.get(i, i)
            canonical_freqs[target] = canonical_freqs.get(target, 0) + freqs[i]
            if target not in canonical_scores or scores[i] > canonical_scores[target]:
                canonical_scores[target] = scores[i]

        # Preserve other columns
        extra_cols = [c for c in group.columns if c not in ("cluster_id", "term", "score", "frequency")]

        for i in range(len(terms)):
            if i in merged_into:
                continue
            row = {"cluster_id": cluster_id, "term": normalized[i]}
            row["score"] = canonical_scores[i]
            row["frequency"] = canonical_freqs[i]
            # Copy extra columns from the original row
            orig_row = group.iloc[i]
            for col in extra_cols:
                row[col] = orig_row[col]
            result_rows.append(row)

    if not result_rows:
        return top_df.iloc[:0]

    result = pd.DataFrame(result_rows)
    # Ensure column order matches input
    cols = [c for c in top_df.columns if c in result.columns]
    extra = [c for c in result.columns if c not in top_df.columns]
    return result[cols + extra].reset_index(drop=True)
