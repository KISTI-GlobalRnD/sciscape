"""Stage 3: Full-vocabulary cleansing (post-aggregation, pre-scoring).

Performs deterministic normalization and heuristic merging on the *entire*
vocabulary before top-K scoring, catching duplicates that would otherwise
slip through separate top-K windows.

Sub-stages:
  3a  Notation + Spelling normalization (Greek letters, hyphens, BrE→AmE)
  3b  Plural singularization (unigrams + phrase last-word)
  3c  Edit-distance merge (unigrams only, cluster-aware frequency ratio)
  3d  Similarity-graph build (all terms, edges stored for LLM candidates)

The module returns a merge map (col_idx → col_idx) plus a rename map
(col_idx → new_name) that the pipeline applies to sparse matrices and
feature-name arrays.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import sparse as sp

from .normalization import _normalize_notation, _normalize_spelling, _phrase_singular
from .utils import _edit_distance
from .vocab_merge import _simple_singular

try:
    import sciscape_text as _rust_text
    _RUST_TEXT_AVAILABLE = True
except ImportError:
    _RUST_TEXT_AVAILABLE = False


# ---------------------------------------------------------------------------
# 3a: Notation + Spelling normalization
# ---------------------------------------------------------------------------

def _build_norm_rename_map(
    feature_names: np.ndarray,
) -> Dict[int, str]:
    """Map each feature index to its notation+spelling-normalized form.

    Returns only entries where the normalized form differs from the original.
    """
    rename: Dict[int, str] = {}
    for idx, name in enumerate(feature_names):
        normed = _normalize_notation(name)
        normed = _normalize_spelling(normed)
        normed = normed.strip()
        if normed and normed != name:
            rename[idx] = normed
    return rename


def _merge_from_rename(
    feature_names: np.ndarray,
    rename_map: Dict[int, str],
) -> Tuple[Dict[int, int], Dict[int, str]]:
    """Convert a rename map into a merge map + final rename map.

    When two different original names normalize to the same string,
    the lower-indexed one (arbitrary but deterministic) becomes the target
    and the other merges into it.

    Returns
    -------
    merge_map : dict
        col_idx → col_idx (source merges into target)
    final_rename : dict
        col_idx → new_name (only for surviving targets that need renaming)
    """
    # Build normalized_name → list of (original_idx, original_name)
    norm_to_indices: Dict[str, List[int]] = defaultdict(list)

    for idx, name in enumerate(feature_names):
        normed = rename_map.get(idx, name)
        norm_to_indices[normed].append(idx)

    merge_map: Dict[int, int] = {}
    final_rename: Dict[int, str] = {}

    for normed, indices in norm_to_indices.items():
        if len(indices) <= 1:
            # Single term — just rename if needed
            idx = indices[0]
            if idx in rename_map:
                final_rename[idx] = normed
            continue
        # Multiple terms normalize to the same form → merge.
        # Prefer the index whose original name already equals the normalized form
        # (no rename needed); fall back to the first index.
        target = indices[0]
        for idx in indices:
            if feature_names[idx] == normed:
                target = idx
                break
        if target in rename_map:
            # Target's original name differs, so rename it
            final_rename[target] = normed
        for src in indices:
            if src != target:
                merge_map[src] = target

    return merge_map, final_rename


# ---------------------------------------------------------------------------
# 3b: Plural singularization
# ---------------------------------------------------------------------------

def _build_plural_merge_map(
    feature_names: np.ndarray,
    existing_merges: Dict[int, int],
    C: Optional[sp.spmatrix] = None,
    merge_frequency_ratio: float = 0.01,
) -> Dict[int, int]:
    """Build merge map for plural→singular pairs across entire vocabulary.

    Handles both unigrams ("networks" → "network") and phrases
    ("neural networks" → "neural network" via last-word singularization).

    Parameters
    ----------
    feature_names : array
        Current feature names (may already reflect notation renaming).
    existing_merges : dict
        Already-decided merges; skip any term involved.
    C : sparse matrix, optional
        Aggregated count matrix for frequency gating.
    merge_frequency_ratio : float
        Skip merge if minor/major frequency ratio exceeds this.
    """
    name_to_idx: Dict[str, int] = {}
    for i, name in enumerate(feature_names):
        name_to_idx.setdefault(name, i)

    col_sums: Optional[np.ndarray] = None
    if C is not None:
        col_sums = np.asarray(C.sum(axis=0)).ravel()

    involved = set(existing_merges.keys()) | set(existing_merges.values())
    merge_map: Dict[int, int] = {}

    for idx, name in enumerate(feature_names):
        if idx in involved or idx in merge_map:
            continue

        # Try phrase-level singularization (works for both unigrams and phrases)
        if " " in name:
            singular = _phrase_singular(name)
        else:
            singular_word = _simple_singular(name)
            singular = singular_word  # None if not a plural

        if singular is None:
            continue

        target_idx = name_to_idx.get(singular)
        if target_idx is None or target_idx == idx:
            continue
        if target_idx in involved or target_idx in merge_map:
            continue

        # Frequency gate
        if col_sums is not None:
            src_freq = col_sums[idx]
            tgt_freq = col_sums[target_idx]
            major = max(src_freq, tgt_freq)
            minor = min(src_freq, tgt_freq)
            if major > 0 and (minor / major) > merge_frequency_ratio:
                # Both forms are frequent — still merge plural into singular
                # (this is morphological, not ambiguous like "aids"→"aid")
                pass

        merge_map[idx] = target_idx
        involved.add(idx)

    return merge_map


# ---------------------------------------------------------------------------
# 3c: Edit-distance merge (unigrams only, cluster-aware)
# ---------------------------------------------------------------------------

def _build_edit_distance_merge_map(
    feature_names: np.ndarray,
    existing_merges: Dict[int, int],
    C: sp.spmatrix,
    max_edit_distance: int = 1,
    global_ratio_threshold: float = 0.01,
) -> Dict[int, int]:
    """Merge unigram pairs within edit distance, using cluster-aware safety.

    Conditions for auto-merge:
    1. Both terms are unigrams (no spaces)
    2. Edit distance <= max_edit_distance
    3. Global frequency ratio: minor / major < global_ratio_threshold
    4. No cluster dominance: the minor form is NOT the dominant form
       in any single cluster (i.e., minor_leads_anywhere == False)

    Parameters
    ----------
    C : sparse matrix
        Per-cluster count matrix (K × V).
    """
    involved = set(existing_merges.keys()) | set(existing_merges.values())
    merge_map: Dict[int, int] = {}

    # Only consider unigrams
    unigram_indices = [
        i for i, name in enumerate(feature_names)
        if " " not in name and i not in involved and len(name) > 3
    ]

    if not unigram_indices:
        return merge_map

    col_sums = np.asarray(C.sum(axis=0)).ravel()
    C_csc = C.tocsc()

    # ── Rust fast path: sparse column comparison, no todense() ──
    if _RUST_TEXT_AVAILABLE:
        merge_keys = np.array(list(existing_merges.keys()), dtype=np.uint32)
        merge_vals = np.array(list(existing_merges.values()), dtype=np.uint32)
        pairs = _rust_text.rust_build_edit_distance_merge_map(
            list(feature_names),
            merge_keys,
            merge_vals,
            np.asarray(C_csc.indptr, dtype=np.uint64),
            np.asarray(C_csc.indices, dtype=np.uint32),
            np.asarray(C_csc.data, dtype=np.float64),
            np.asarray(col_sums, dtype=np.float64),
            max_edit_distance=max_edit_distance,
            global_ratio_threshold=global_ratio_threshold,
        )
        return {int(src): int(tgt) for src, tgt in pairs}

    # ── Python fallback ──
    prefix_len = 3
    blocks: Dict[str, List[int]] = defaultdict(list)
    for idx in unigram_indices:
        name = feature_names[idx].lower()
        key = name[:prefix_len] if len(name) >= prefix_len else name
        blocks[key].append(idx)

    for block_indices in blocks.values():
        if len(block_indices) < 2:
            continue
        block_sorted = sorted(block_indices, key=lambda i: -col_sums[i])

        for bi in range(len(block_sorted)):
            idx_i = block_sorted[bi]
            if idx_i in merge_map:
                continue
            name_i = feature_names[idx_i].lower()

            for bj in range(bi + 1, len(block_sorted)):
                idx_j = block_sorted[bj]
                if idx_j in merge_map:
                    continue
                name_j = feature_names[idx_j].lower()

                if abs(len(name_i) - len(name_j)) > max_edit_distance:
                    continue

                dist = _edit_distance(name_i, name_j)
                if dist > max_edit_distance:
                    continue

                freq_i = col_sums[idx_i]
                freq_j = col_sums[idx_j]
                major_freq = max(freq_i, freq_j)
                minor_freq = min(freq_i, freq_j)
                if major_freq == 0:
                    continue
                ratio = minor_freq / major_freq
                if ratio >= global_ratio_threshold:
                    continue

                major_idx = idx_i if freq_i >= freq_j else idx_j
                minor_idx = idx_j if freq_i >= freq_j else idx_i

                minor_col = np.asarray(C_csc[:, minor_idx].todense()).ravel()
                major_col = np.asarray(C_csc[:, major_idx].todense()).ravel()

                minor_leads_anywhere = np.any(minor_col > major_col)
                if minor_leads_anywhere:
                    continue

                merge_map[minor_idx] = major_idx

    return merge_map


# ---------------------------------------------------------------------------
# 3d: Similarity graph (for LLM candidate lookup)
# ---------------------------------------------------------------------------

class VocabSimGraph:
    """Lightweight vocabulary similarity graph.

    Stores edges between terms that are within edit distance but were NOT
    auto-merged (e.g., phrase pairs, or unigram pairs that failed the
    cluster-aware ratio check). These edges are used later to generate
    LLM candidate lists.

    Each edge stores: (term_a, term_b, edit_distance, metadata).
    """

    def __init__(self) -> None:
        self.edges: List[Tuple[str, str, int, Dict]] = []
        self._adjacency: Dict[str, List[Tuple[str, int, Dict]]] = defaultdict(list)

    def add_edge(self, a: str, b: str, dist: int, meta: Optional[Dict] = None) -> None:
        meta = meta or {}
        self.edges.append((a, b, dist, meta))
        self._adjacency[a].append((b, dist, meta))
        self._adjacency[b].append((a, dist, meta))

    def neighbors(self, term: str, max_dist: Optional[int] = None) -> List[Tuple[str, int, Dict]]:
        """Return 1-hop neighbors, optionally filtered by max distance."""
        nbrs = self._adjacency.get(term, [])
        if max_dist is not None:
            nbrs = [(t, d, m) for t, d, m in nbrs if d <= max_dist]
        return nbrs

    def neighbor_terms(self, term: str, max_dist: Optional[int] = None) -> List[str]:
        """Return just the term names of 1-hop neighbors."""
        return [t for t, d, m in self.neighbors(term, max_dist)]

    def __len__(self) -> int:
        return len(self.edges)

    def __repr__(self) -> str:
        return f"VocabSimGraph(edges={len(self.edges)}, nodes={len(self._adjacency)})"


def _build_similarity_graph(
    feature_names: np.ndarray,
    existing_merges: Dict[int, int],
    C: sp.spmatrix,
    max_edit_distance: int = 2,
) -> VocabSimGraph:
    """Build similarity graph for all terms (including phrases).

    Stores edges for pairs within edit distance that were NOT auto-merged.
    These become LLM candidate suggestions.  Edge metadata includes
    per-cluster frequency distribution for cross-cluster safety checks.
    """
    graph = VocabSimGraph()
    involved = set(existing_merges.keys())  # already merged away
    col_sums = np.asarray(C.sum(axis=0)).ravel()
    C_csc_sim = C.tocsc()  # efficient column slicing for per-pair extraction

    # Active indices (not merged away)
    all_idx = np.arange(len(feature_names))
    active = all_idx[~np.isin(all_idx, np.array(list(involved)))].tolist()

    # Blocking: group by shared words (phrases) or prefix (unigrams)
    prefix_len = 3
    blocks: Dict[str, List[int]] = defaultdict(list)
    for idx in active:
        name = feature_names[idx].lower()
        words = name.split()
        if len(words) > 1:
            for w in words:
                blocks[w].append(idx)
        else:
            key = name[:prefix_len] if len(name) >= prefix_len else name
            blocks[key].append(idx)
            if len(name) < prefix_len:
                blocks["_short"].append(idx)

    seen_pairs: Set[Tuple[int, int]] = set()

    for block_indices in blocks.values():
        if len(block_indices) < 2:
            continue
        for bi in range(len(block_indices)):
            idx_i = block_indices[bi]
            name_i = feature_names[idx_i]
            for bj in range(bi + 1, len(block_indices)):
                idx_j = block_indices[bj]
                pair = (min(idx_i, idx_j), max(idx_i, idx_j))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                name_j = feature_names[idx_j]
                if abs(len(name_i) - len(name_j)) > max_edit_distance:
                    continue
                # Skip very short terms
                if len(name_i) <= 2 or len(name_j) <= 2:
                    continue

                dist = _edit_distance(name_i.lower(), name_j.lower())
                if dist > max_edit_distance:
                    continue

                freq_i = float(col_sums[idx_i])
                freq_j = float(col_sums[idx_j])
                # Per-cluster frequency vectors for cross-cluster analysis
                cluster_freq_i = np.asarray(C_csc_sim[:, idx_i].toarray()).ravel()
                cluster_freq_j = np.asarray(C_csc_sim[:, idx_j].toarray()).ravel()
                graph.add_edge(
                    name_i, name_j, dist,
                    meta={
                        "freq_a": freq_i,
                        "freq_b": freq_j,
                        "is_phrase": " " in name_i or " " in name_j,
                        "cluster_freq_a": cluster_freq_i,
                        "cluster_freq_b": cluster_freq_j,
                    },
                )

    return graph


# ---------------------------------------------------------------------------
# Orchestrator: run all sub-stages
# ---------------------------------------------------------------------------

def run_vocab_cleansing(
    feature_names_uni: np.ndarray,
    feature_names_phrase: np.ndarray,
    C_uni: sp.csr_matrix,
    C_phrase: Optional[sp.csr_matrix],
    DF_uni: Optional[sp.csr_matrix],
    DF_phrase: Optional[sp.csr_matrix],
    *,
    merge_frequency_ratio: float = 0.01,
    edit_distance_max: int = 1,
    edit_distance_ratio: float = 0.01,
    sim_graph_max_dist: int = 2,
    build_similarity_graph: bool = True,
    verbose_callback=None,
) -> Tuple[
    np.ndarray,           # feature_names_uni (updated)
    np.ndarray,           # feature_names_phrase (updated)
    sp.csr_matrix,        # C_uni (merged)
    Optional[sp.csr_matrix],  # C_phrase (merged)
    Optional[sp.csr_matrix],  # DF_uni (merged)
    Optional[sp.csr_matrix],  # DF_phrase (merged)
    VocabSimGraph,        # similarity graph
    Dict[str, str],       # human-readable merge log {source: target}
]:
    """Run full vocabulary cleansing pipeline.

    Returns updated matrices, feature names, similarity graph, and merge log.
    """
    def _log(msg, *args):
        if verbose_callback:
            verbose_callback(msg, *args)

    merge_log: Dict[str, str] = {}

    # ---- 3a: Notation + Spelling ----
    _log("Stage 3a: notation + spelling normalization on %d unigrams + %d phrases",
         len(feature_names_uni), len(feature_names_phrase))

    # Unigrams
    rename_uni = _build_norm_rename_map(feature_names_uni)
    merge_3a_uni, final_rename_uni = _merge_from_rename(feature_names_uni, rename_uni)

    # Phrases
    merge_3a_phrase: Dict[int, int] = {}
    final_rename_phrase: Dict[int, str] = {}
    if feature_names_phrase is not None and len(feature_names_phrase) > 0:
        rename_phrase = _build_norm_rename_map(feature_names_phrase)
        merge_3a_phrase, final_rename_phrase = _merge_from_rename(feature_names_phrase, rename_phrase)

    for src, tgt in merge_3a_uni.items():
        merge_log[str(feature_names_uni[src])] = str(
            final_rename_uni.get(tgt, feature_names_uni[tgt])
        )
    for src, tgt in merge_3a_phrase.items():
        merge_log[str(feature_names_phrase[src])] = str(
            final_rename_phrase.get(tgt, feature_names_phrase[tgt])
        )

    _log("Stage 3a: %d unigram merges, %d phrase merges",
         len(merge_3a_uni), len(merge_3a_phrase))

    # ---- Apply 3a merges to matrices ----
    C_uni, DF_uni, feature_names_uni = _apply_merge_and_rename(
        C_uni, DF_uni, feature_names_uni, merge_3a_uni, final_rename_uni
    )
    if C_phrase is not None and len(feature_names_phrase) > 0:
        C_phrase, DF_phrase, feature_names_phrase = _apply_merge_and_rename(
            C_phrase, DF_phrase, feature_names_phrase, merge_3a_phrase, final_rename_phrase
        )

    # ---- 3b: Plural singularization ----
    _log("Stage 3b: plural singularization")
    merge_3b_uni = _build_plural_merge_map(
        feature_names_uni, {}, C=C_uni, merge_frequency_ratio=merge_frequency_ratio
    )
    merge_3b_phrase: Dict[int, int] = {}
    if C_phrase is not None and len(feature_names_phrase) > 0:
        merge_3b_phrase = _build_plural_merge_map(
            feature_names_phrase, {}, C=C_phrase, merge_frequency_ratio=merge_frequency_ratio
        )

    for src, tgt in merge_3b_uni.items():
        merge_log[str(feature_names_uni[src])] = str(feature_names_uni[tgt])
    for src, tgt in merge_3b_phrase.items():
        merge_log[str(feature_names_phrase[src])] = str(feature_names_phrase[tgt])

    _log("Stage 3b: %d unigram plural merges, %d phrase plural merges",
         len(merge_3b_uni), len(merge_3b_phrase))

    C_uni, DF_uni, feature_names_uni = _apply_merge_only(
        C_uni, DF_uni, feature_names_uni, merge_3b_uni
    )
    if C_phrase is not None and len(feature_names_phrase) > 0:
        C_phrase, DF_phrase, feature_names_phrase = _apply_merge_only(
            C_phrase, DF_phrase, feature_names_phrase, merge_3b_phrase
        )

    # ---- 3c: Edit-distance merge (unigrams only) ----
    _log("Stage 3c: edit-distance merge (unigrams, max_dist=%d)", edit_distance_max)
    merge_3c = _build_edit_distance_merge_map(
        feature_names_uni, {}, C_uni,
        max_edit_distance=edit_distance_max,
        global_ratio_threshold=edit_distance_ratio,
    )
    for src, tgt in merge_3c.items():
        merge_log[str(feature_names_uni[src])] = str(feature_names_uni[tgt])

    _log("Stage 3c: %d edit-distance merges", len(merge_3c))

    C_uni, DF_uni, feature_names_uni = _apply_merge_only(
        C_uni, DF_uni, feature_names_uni, merge_3c
    )

    # ---- 3d: Similarity graph ----
    sim_graph = VocabSimGraph()
    if build_similarity_graph:
        _log("Stage 3d: building similarity graph (max_dist=%d)", sim_graph_max_dist)

        # Combine uni + phrase for graph building
        all_names = np.concatenate([feature_names_uni, feature_names_phrase])
        C_combined = sp.hstack(
            [m for m in (C_uni, C_phrase) if m is not None], format="csr"
        )
        sim_graph = _build_similarity_graph(
            all_names, {}, C_combined,
            max_edit_distance=sim_graph_max_dist,
        )
        _log("Stage 3d: similarity graph has %d edges across %s",
             len(sim_graph), repr(sim_graph))
    else:
        _log("Stage 3d: similarity graph skipped")

    return (
        feature_names_uni,
        feature_names_phrase,
        C_uni,
        C_phrase,
        DF_uni,
        DF_phrase,
        sim_graph,
        merge_log,
    )


# ---------------------------------------------------------------------------
# Matrix manipulation helpers
# ---------------------------------------------------------------------------

def _apply_merge_and_rename(
    C: sp.csr_matrix,
    DF: Optional[sp.csr_matrix],
    feature_names: np.ndarray,
    merge_map: Dict[int, int],
    rename_map: Dict[int, str],
) -> Tuple[sp.csr_matrix, Optional[sp.csr_matrix], np.ndarray]:
    """Apply column merges, drop merged columns, then rename surviving ones."""
    if merge_map:
        C, feature_names = _merge_columns(C, feature_names, merge_map)
        if DF is not None:
            dummy = np.arange(DF.shape[1])
            DF, _ = _merge_columns(DF, dummy, merge_map)

    # Apply renames (indices shifted after column removal)
    if rename_map:
        # Build old_idx → new_name, but only for indices that survived
        drop_cols = set(merge_map.keys())
        # Map old indices to new positions
        old_to_new = {}
        new_pos = 0
        for old_idx in range(len(feature_names) + len(drop_cols)):
            if old_idx in drop_cols:
                continue
            old_to_new[old_idx] = new_pos
            new_pos += 1

        names = feature_names.copy()
        for old_idx, new_name in rename_map.items():
            if old_idx in drop_cols:
                continue
            new_idx = old_to_new.get(old_idx)
            if new_idx is not None and new_idx < len(names):
                names[new_idx] = new_name
        feature_names = names

    return C, DF, feature_names


def _apply_merge_only(
    C: sp.csr_matrix,
    DF: Optional[sp.csr_matrix],
    feature_names: np.ndarray,
    merge_map: Dict[int, int],
) -> Tuple[sp.csr_matrix, Optional[sp.csr_matrix], np.ndarray]:
    """Apply column merges and drop merged columns (no renaming)."""
    if not merge_map:
        return C, DF, feature_names
    C, feature_names = _merge_columns(C, feature_names, merge_map)
    if DF is not None:
        np.arange(DF.shape[1] + len(merge_map))
        # Reconstruct dummy with correct original size
        orig_n = DF.shape[1]
        DF_new, _ = _merge_columns(DF, np.arange(orig_n), merge_map)
        DF = DF_new
    return C, DF, feature_names


def _merge_columns(
    X: sp.csr_matrix,
    feature_names: np.ndarray,
    merge_map: Dict[int, int],
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Merge sparse matrix columns: sum source into target, drop source.

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

    # old_col → new_col index
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

    keep_mask = np.array(keep_cols)
    names_merged = feature_names[keep_mask] if len(feature_names) == n_cols else feature_names

    return X_merged, names_merged


__all__ = [
    "VocabSimGraph",
    "run_vocab_cleansing",
]
