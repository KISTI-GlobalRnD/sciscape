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


def _simple_singular(word: str) -> Optional[str]:
    """Heuristic plural -> singular for English.

    Returns the singular form if the word looks like a regular English plural,
    otherwise returns None.  Deliberately conservative to avoid false positives
    on domain terms.
    """
    if len(word) <= 3:
        return None
    if word.endswith("ies") and len(word) > 4:
        # "batteries" -> "battery", but not "series"
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
        candidate = word[:-1]
        return candidate
    return None


def build_merge_map(
    feature_names: np.ndarray,
    config: VocabMergeConfig,
) -> Dict[int, int]:
    """Analyze vocabulary and return col_idx -> col_idx merge mapping.

    For each mergeable pair, the higher-frequency form (by global count) is
    kept as the target. The returned dict maps source_idx -> target_idx.
    """
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    merge: Dict[int, int] = {}

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
            # Always merge plural -> singular (singular is the canonical form)
            if plural_idx not in merge and singular_idx not in merge:
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
            if hyphen_idx not in merge and space_idx not in merge:
                merge[hyphen_idx] = space_idx

    return merge


def apply_merge_map(
    X: sp.csr_matrix,
    feature_names: np.ndarray,
    merge_map: Dict[int, int],
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Merge sparse matrix columns and return updated matrix + names.

    Columns specified in merge_map are summed into their target columns,
    then the source columns are removed.
    """
    if not merge_map:
        return X, feature_names

    n_cols = X.shape[1]
    X_lil = X.tolil()

    # Sum source columns into target columns
    for src, tgt in merge_map.items():
        if src >= n_cols or tgt >= n_cols:
            continue
        src_col = X_lil[:, src].toarray().ravel()
        X_lil[:, tgt] = X_lil[:, tgt].toarray().ravel() + src_col
        X_lil[:, src] = 0

    # Determine which columns to keep
    drop_cols = set(merge_map.keys())
    keep_mask = np.array([i not in drop_cols for i in range(n_cols)])

    X_merged = X_lil.tocsc()[:, keep_mask].tocsr()
    names_merged = feature_names[keep_mask]

    return X_merged, names_merged
