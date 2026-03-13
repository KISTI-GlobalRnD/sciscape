"""Post-top-K keyword-level normalization (Stage 5).

Operates on the scored keyword DataFrame after top-K selection.
Handles abbreviation expansion, notation normalization, and
frequency-based heuristic merging of near-duplicate terms.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Set, Tuple

import pandas as pd

from .utils import _edit_distance
from .vocab_merge import _simple_singular


def _phrase_singular(term: str) -> Optional[str]:
    """Attempt to singularize the *last* word of a multi-word term.

    Returns the singular form if the last word is a regular English plural,
    otherwise returns None.  Handles both unigrams and phrases:
      "point clouds"  -> "point cloud"
      "transformers"  -> "transformer"
      "series"        -> None  (not a regular plural)
    """
    words = term.split()
    if not words:
        return None
    last = words[-1]
    singular = _simple_singular(last)
    if singular is None:
        return None
    return " ".join(words[:-1] + [singular]) if len(words) > 1 else singular


def _expand_abbreviations(
    term: str,
    builtin_aliases: Mapping[str, str],
) -> str:
    """Expand known abbreviations to their canonical forms."""
    lower = term.lower().strip()
    if lower in builtin_aliases:
        return builtin_aliases[lower]
    return term


# British/American and common spelling variants — maps variant → canonical.
# Applied per-word so "protoplanetary disc" → "protoplanetary disk".
_SPELLING_VARIANTS: Dict[str, str] = {
    "disc": "disk",
    "discs": "disks",
    "colour": "color",
    "colours": "colors",
    "behaviour": "behavior",
    "behaviours": "behaviors",
    "modelling": "modeling",
    "analyse": "analyze",
    "optimise": "optimize",
    "characterise": "characterize",
    "utilise": "utilize",
    "recognise": "recognize",
    "minimise": "minimize",
    "maximise": "maximize",
    "catalyse": "catalyze",
    "synthesise": "synthesize",
    "fibre": "fiber",
    "fibres": "fibers",
    "centre": "center",
    "centres": "centers",
    "metre": "meter",
    "metres": "meters",
    "litre": "liter",
    "litres": "liters",
    "defence": "defense",
    "licence": "license",
    "aluminium": "aluminum",
    "sulphur": "sulfur",
    "vapour": "vapor",
    "vapours": "vapors",
    "tumour": "tumor",
    "tumours": "tumors",
    "grey": "gray",
}


def _normalize_spelling(term: str) -> str:
    """Normalize known British/American spelling variants per word."""
    words = term.split()
    normalized = [_SPELLING_VARIANTS.get(w.lower(), w) for w in words]
    return " ".join(normalized)


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


def _build_norm_blocks(
    terms: List[str],
    max_edit_distance: int,
    prefix_len: int = 3,
) -> Dict[str, List[int]]:
    """Group term indices into blocks for edit-distance comparison.

    Terms in different blocks cannot be within ``max_edit_distance`` of each
    other, so we skip those pairs entirely.  Blocking keys:
    - Multi-word terms: each word token (so "machine learning" is in both
      the "machine" and "learning" blocks)
    - Single-word terms: first ``prefix_len`` characters

    A term may appear in multiple blocks; the merge loop de-duplicates via
    the ``merged_into`` dict.
    """
    blocks: Dict[str, List[int]] = defaultdict(list)
    for idx, term in enumerate(terms):
        lower = term.lower()
        words = lower.split()
        if len(words) > 1:
            for w in words:
                blocks[w].append(idx)
        else:
            key = lower[:prefix_len] if len(lower) >= prefix_len else lower
            blocks[key].append(idx)
            # Short terms go into a shared block so they can be compared with each other
            if len(lower) < prefix_len:
                blocks["_short"].append(idx)
    return dict(blocks)


def normalize_keywords(
    top_df: pd.DataFrame,
    builtin_aliases: Mapping[str, str],
    stopwords: Optional[Set[str]] = None,
    max_edit_distance: int = 2,
    min_frequency_ratio: float = 0.01,
    plural_merge_enabled: bool = True,
) -> pd.DataFrame:
    """Post-top-K keyword normalization.

    Steps:
    1. Expand known abbreviations (builtin_aliases)
    2. Normalize notation (Greek letters, hyphens)
    2b. Merge plural forms into singular (phrase-level)
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
            t = _normalize_spelling(t)
            if stopwords and t.lower() in stopwords:
                t = term  # revert if normalization produced a stopword
            normalized[i] = t.strip()

        # Step 2b: plural merge — merge "Xs" into "X" when both present
        if plural_merge_enabled:
            norm_lower_to_idx: Dict[str, int] = {}
            for i in range(len(terms)):
                norm_lower_to_idx.setdefault(normalized[i].lower(), i)
            for i in range(len(terms)):
                singular = _phrase_singular(normalized[i])
                if singular is None:
                    continue
                target_idx = norm_lower_to_idx.get(singular.lower())
                if target_idx is not None and target_idx != i:
                    # Point the plural's normalized form to the singular
                    # (actual merging happens in step 3a exact-match)
                    normalized[i] = normalized[target_idx]

        # Step 3: merge near-duplicates (greedy, high-freq absorbs low-freq)
        # Sort by frequency descending so higher-freq terms are canonical
        order = sorted(range(len(terms)), key=lambda i: -freqs[i])
        merged_into: Dict[int, int] = {}  # source_idx -> target_idx

        # 3a: exact-match pass (always O(n) via dict lookup)
        norm_to_first: Dict[str, int] = {}
        for idx in order:
            key = normalized[idx].lower()
            if key in norm_to_first:
                merged_into[idx] = norm_to_first[key]
            else:
                norm_to_first[key] = idx

        # 3b: edit-distance pass with blocking to avoid O(n²)
        if max_edit_distance > 0:
            norm_list = [normalized[i] for i in range(len(terms))]
            blocks = _build_norm_blocks(norm_list, max_edit_distance)

            for block_indices in blocks.values():
                # Within each block, compare pairs (sorted by frequency desc)
                block_order = [i for i in order if i in set(block_indices)]
                for bi in range(len(block_order)):
                    idx_i = block_order[bi]
                    if idx_i in merged_into:
                        continue
                    term_i = normalized[idx_i]
                    if len(term_i) <= 3:
                        continue
                    for bj in range(bi + 1, len(block_order)):
                        idx_j = block_order[bj]
                        if idx_j in merged_into:
                            continue
                        term_j = normalized[idx_j]
                        if len(term_j) <= 3:
                            continue
                        # Length filter: edit distance can't be smaller than length diff
                        if abs(len(term_i) - len(term_j)) > max_edit_distance:
                            continue
                        dist = _edit_distance(term_i.lower(), term_j.lower())
                        if dist <= max_edit_distance:
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
