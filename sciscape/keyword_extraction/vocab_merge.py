"""Vocabulary-level merge: combine sparse matrix columns for equivalent terms.

Operates post-vectorizer (after CountVectorizer.fit_transform), merging columns
for plural/singular and hyphen/space variants. This is safer than text-level
preprocessing because it preserves the original token patterns and avoids
accidental merges of domain-specific terms (e.g., "AIDS" != "aid").

Usage:
    merge_map = build_merge_map(feature_names, config)
    X_merged, names_merged = apply_merge_map(X, feature_names, merge_map)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import sparse as sp

from .config import VocabMergeConfig


_INVARIANT_PLURALS = frozenset({
    "series", "species", "chassis", "diabetes", "rabies",
})

# Suffixes that look like they end in "s" but are NOT English plurals.
# -sis: analysis, synthesis, diagnosis, basis, crisis, thesis, hypothesis
# -us: status, focus, radius, census, corpus, apparatus, stimulus
# -is: axis, mantis, ibis, cannabis
# -os: chaos, ethos, cosmos
_NON_PLURAL_SUFFIXES = ("sis", "us", "is", "os")


def _simple_singular(word: str) -> Optional[str]:
    """Heuristic plural -> singular for English.

    Returns the singular form if the word looks like a regular English plural,
    otherwise returns None.  Deliberately conservative to avoid false positives
    on domain terms.
    """
    if len(word) <= 3:
        return None
    if word.lower() in _INVARIANT_PLURALS:
        return None
    if word.endswith("ies") and len(word) > 4:
        # "batteries" -> "battery"
        candidate = word[:-3] + "y"
        return candidate
    if word.endswith("ses") and len(word) > 4:
        # "analyses" -> "analysis" is irregular, skip
        # "processes" -> "process"
        candidate = word[:-2]
        return candidate
    if word.endswith("es") and len(word) > 3:
        if word[-3] in ("s", "x", "z", "h"):
            # "boxes" -> "box", "matches" -> "match"
            candidate = word[:-2]
            return candidate
    if word.endswith("s") and not word.endswith("ss"):
        # Skip Latin/Greek-origin words that aren't English plurals
        lower = word.lower()
        if any(lower.endswith(sfx) for sfx in _NON_PLURAL_SUFFIXES):
            return None
        candidate = word[:-1]
        return candidate
    return None


def build_merge_map(
    feature_names: np.ndarray,
    config: VocabMergeConfig,
    C: Optional[sp.spmatrix] = None,
) -> Dict[int, int]:
    """Analyze vocabulary and return col_idx -> col_idx merge mapping.

    For each mergeable pair, the higher-frequency form (by global count) is
    kept as the target. The returned dict maps source_idx -> target_idx.

    When *C* (the aggregated count matrix) is provided, pairs where the minor
    form's total count exceeds ``config.merge_frequency_ratio`` times the major
    form's count are skipped.  This prevents false merges such as
    "aids" (HIV/AIDS) -> "aid" (assistance).
    """
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    merge: Dict[int, int] = {}

    # Pre-compute per-column totals for frequency gating
    col_sums: Optional[np.ndarray] = None
    if C is not None:
        col_sums = np.asarray(C.sum(axis=0)).ravel()

    def _freq_ok(src_idx: int, tgt_idx: int) -> bool:
        """Return True if the merge passes the frequency-ratio gate."""
        if col_sums is None:
            return True
        src_freq = col_sums[src_idx]
        tgt_freq = col_sums[tgt_idx]
        major = max(src_freq, tgt_freq)
        minor = min(src_freq, tgt_freq)
        if major == 0:
            return True
        return (minor / major) <= config.merge_frequency_ratio

    if config.plural_to_singular:
        for name in feature_names:
            if " " in name:
                # Only handle unigrams for plural merge
                continue
            singular = _simple_singular(name)
            if singular is None:
                continue
            if singular not in name_to_idx:
                continue
            plural_idx = name_to_idx[name]
            singular_idx = name_to_idx[singular]
            if plural_idx == singular_idx:
                continue
            if plural_idx in merge or singular_idx in merge:
                continue
            if not _freq_ok(plural_idx, singular_idx):
                continue
            # Always merge plural -> singular (singular is the canonical form)
            merge[plural_idx] = singular_idx

    if config.hyphen_normalize:
        for name in feature_names:
            if "-" not in name:
                continue
            space_form = name.replace("-", " ")
            if space_form not in name_to_idx:
                continue
            hyphen_idx = name_to_idx[name]
            space_idx = name_to_idx[space_form]
            if hyphen_idx == space_idx:
                continue
            if hyphen_idx in merge or space_idx in merge:
                continue
            if not _freq_ok(hyphen_idx, space_idx):
                continue
            merge[hyphen_idx] = space_idx

    return merge


def apply_merge_map(
    X: sp.csr_matrix,
    feature_names: np.ndarray,
    merge_map: Dict[int, int],
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Merge sparse matrix columns and return updated matrix + names.

    Uses a sparse projection matrix ``X @ P`` instead of per-column LIL
    extraction, avoiding O(merges × rows) memory churn.
    """
    if not merge_map:
        return X, feature_names

    n_cols = X.shape[1]

    # Filter out-of-range entries
    valid_merges = {s: t for s, t in merge_map.items() if s < n_cols and t < n_cols}
    if not valid_merges:
        return X, feature_names

    # Build column mapping: source cols redirect to their target col
    drop_cols = set(valid_merges.keys())
    keep_cols = sorted(set(range(n_cols)) - drop_cols)
    n_new = len(keep_cols)

    col_remap = np.full(n_cols, -1, dtype=np.int32)
    for new_idx, old_idx in enumerate(keep_cols):
        col_remap[old_idx] = new_idx
    for src, tgt in valid_merges.items():
        col_remap[src] = col_remap[tgt]

    # Sparse projection matrix P (n_cols × n_new): X_merged = X @ P
    p_rows = np.arange(n_cols, dtype=np.int32)
    p_cols = col_remap
    valid = p_cols >= 0
    P = sp.csc_matrix(
        (np.ones(valid.sum(), dtype=np.float64), (p_rows[valid], p_cols[valid])),
        shape=(n_cols, n_new),
    )
    X_merged = (X @ P).tocsr()

    keep_arr = np.array(keep_cols)
    names_merged = feature_names[keep_arr]

    return X_merged, names_merged


__all__ = ["build_merge_map", "apply_merge_map"]
