"""Multi-layer term similarity network (Stage 7).

Builds similarity layers between terms using different signals:
- String layer: character-level similarity (edit distance, n-gram overlap)
- Token layer: word-level overlap, containment, abbreviation patterns
- Co-occurrence layer: normalized co-occurrence from document scans

Layers are combined with configurable weights, then merge groups are
detected for downstream canonicalization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    import pandas as pd

import numpy as np
from scipy import sparse as sp

from .utils import _edit_distance

logger = logging.getLogger(__name__)


@dataclass
class TermNetworkConfig:
    """Configuration for term similarity network construction."""

    enabled: bool = False
    layers: List[str] = field(default_factory=lambda: ["string", "token"])
    layer_weights: Dict[str, float] = field(
        default_factory=lambda: {"string": 1.0, "token": 0.8, "cooccurrence": 0.6}
    )
    merge_threshold: float = 0.5
    # String layer
    max_edit_distance: int = 2
    min_char_ngram_sim: float = 0.3
    char_ngram_n: int = 3
    # Token layer
    min_token_overlap: float = 0.5
    # Blocking
    blocking_strategy: str = "token"  # "token" | "prefix"
    max_block_size: int = 500
    prefix_length: int = 3
    # Merge group limits
    max_group_size: int = 5  # split oversized connected components

    def __post_init__(self) -> None:
        if self.max_block_size < 1:
            raise ValueError(f"max_block_size must be >= 1, got {self.max_block_size}")
        if self.max_group_size < 2:
            raise ValueError(f"max_group_size must be >= 2, got {self.max_group_size}")
        if not (0.0 <= self.merge_threshold <= 1.0):
            raise ValueError(
                f"merge_threshold must be in [0.0, 1.0], got {self.merge_threshold}"
            )
        valid_strategies = ("token", "prefix")
        if self.blocking_strategy not in valid_strategies:
            raise ValueError(
                f"blocking_strategy must be one of {valid_strategies}, got {self.blocking_strategy!r}"
            )


def _char_ngrams(s: str, n: int = 3) -> set:
    """Generate character n-grams from a string."""
    if len(s) < n:
        return {s}
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0



def _build_blocks(terms: Sequence[str], strategy: str, prefix_length: int = 3) -> Dict[str, List[int]]:
    """Group term indices into blocks for efficient pairwise comparison.

    Returns a dict mapping block_key -> list of term indices.
    Terms in different blocks are not compared (blocking assumption:
    similar terms share at least one block).

    Short terms (<= 5 chars) and multi-word terms are placed in a shared
    "_abbrev" block to enable abbreviation detection across boundaries.
    """
    blocks: Dict[str, List[int]] = {}
    abbrev_candidates: List[int] = []  # short terms that could be abbreviations
    multi_word: List[int] = []  # multi-word terms that could have abbreviations

    if strategy == "prefix":
        for i, term in enumerate(terms):
            key = term[:prefix_length].lower() if len(term) >= prefix_length else term.lower()
            blocks.setdefault(key, []).append(i)
            if len(term.replace(" ", "")) <= 5:
                abbrev_candidates.append(i)
            elif " " in term:
                multi_word.append(i)
    else:  # "token" blocking
        for i, term in enumerate(terms):
            tokens = term.lower().split()
            if not tokens:
                blocks.setdefault("", []).append(i)
                continue
            for token in tokens:
                blocks.setdefault(token, []).append(i)
            if len(term.replace(" ", "")) <= 5:
                abbrev_candidates.append(i)
            elif len(tokens) > 1:
                multi_word.append(i)

    # Create abbreviation block: pair short terms with multi-word terms
    if abbrev_candidates and multi_word:
        abbrev_block = abbrev_candidates + multi_word
        blocks["_abbrev"] = abbrev_block

    return blocks


class TermNetwork:
    """Multi-layer similarity network for keyword merge candidates."""

    def __init__(self, config: TermNetworkConfig) -> None:
        self.config = config

    def build_layer_string(self, terms: Sequence[str]) -> sp.csr_matrix:
        """Layer 1: character-level similarity (edit distance + char n-gram).

        Uses blocking to avoid O(n^2) comparisons.
        """
        n = len(terms)
        if n == 0:
            return sp.csr_matrix((0, 0), dtype=np.float32)

        cfg = self.config
        blocks = _build_blocks(terms, cfg.blocking_strategy, cfg.prefix_length)
        ngrams = [_char_ngrams(t.lower(), cfg.char_ngram_n) for t in terms]

        rows, cols, vals = [], [], []
        seen = set()

        for block_key, block_indices in blocks.items():
            if len(block_indices) > cfg.max_block_size:
                logger.warning(
                    "String layer: skipping block '%s' (%d terms > max_block_size=%d)",
                    block_key, len(block_indices), cfg.max_block_size,
                )
                continue
            for ii in range(len(block_indices)):
                for jj in range(ii + 1, len(block_indices)):
                    i, j = block_indices[ii], block_indices[jj]
                    pair = (min(i, j), max(i, j))
                    if pair in seen:
                        continue
                    seen.add(pair)

                    # Edit distance similarity
                    dist = _edit_distance(terms[i].lower(), terms[j].lower())
                    max_len = max(len(terms[i]), len(terms[j]))
                    if max_len == 0:
                        continue
                    ed_sim = 1.0 - (dist / max_len) if dist <= cfg.max_edit_distance else 0.0

                    # Char n-gram similarity
                    ng_sim = _jaccard(ngrams[i], ngrams[j])

                    # Combined
                    sim = max(ed_sim, ng_sim)
                    if sim >= cfg.min_char_ngram_sim:
                        rows.extend([i, j])
                        cols.extend([j, i])
                        vals.extend([sim, sim])

        if not rows:
            return sp.csr_matrix((n, n), dtype=np.float32)

        return sp.csr_matrix(
            (np.array(vals, dtype=np.float32), (np.array(rows), np.array(cols))),
            shape=(n, n),
        )

    def build_layer_token(self, terms: Sequence[str]) -> sp.csr_matrix:
        """Layer 2: word-level overlap, containment, abbreviation detection."""
        n = len(terms)
        if n == 0:
            return sp.csr_matrix((0, 0), dtype=np.float32)

        cfg = self.config
        token_sets = [set(t.lower().split()) for t in terms]
        blocks = _build_blocks(terms, cfg.blocking_strategy, cfg.prefix_length)

        rows, cols, vals = [], [], []
        seen = set()

        for block_key, block_indices in blocks.items():
            if len(block_indices) > cfg.max_block_size:
                logger.warning(
                    "Token layer: skipping block '%s' (%d terms > max_block_size=%d)",
                    block_key, len(block_indices), cfg.max_block_size,
                )
                continue
            for ii in range(len(block_indices)):
                for jj in range(ii + 1, len(block_indices)):
                    i, j = block_indices[ii], block_indices[jj]
                    pair = (min(i, j), max(i, j))
                    if pair in seen:
                        continue
                    seen.add(pair)

                    ts_i, ts_j = token_sets[i], token_sets[j]
                    if not ts_i or not ts_j:
                        continue

                    # Jaccard overlap
                    overlap = _jaccard(ts_i, ts_j)

                    # Containment: one term's tokens are subset of the other
                    containment = 0.0
                    if ts_i <= ts_j or ts_j <= ts_i:
                        containment = len(ts_i & ts_j) / min(len(ts_i), len(ts_j))

                    # Abbreviation check: one term could be initials of the other
                    # Use original term word order (not set order) for initials
                    abbrev_sim = 0.0
                    ti_lower = terms[i].lower()
                    tj_lower = terms[j].lower()
                    ti_words = ti_lower.split()
                    tj_words = tj_lower.split()
                    if len(ti_lower.replace(" ", "")) <= 5 and len(tj_words) > 1:
                        initials = "".join(w[0] for w in tj_words if w)
                        if ti_lower.replace(" ", "") == initials:
                            abbrev_sim = 0.9
                    if len(tj_lower.replace(" ", "")) <= 5 and len(ti_words) > 1:
                        initials = "".join(w[0] for w in ti_words if w)
                        if tj_lower.replace(" ", "") == initials:
                            abbrev_sim = 0.9

                    sim = max(overlap, containment, abbrev_sim)
                    if sim >= cfg.min_token_overlap:
                        rows.extend([i, j])
                        cols.extend([j, i])
                        vals.extend([sim, sim])

        if not rows:
            return sp.csr_matrix((n, n), dtype=np.float32)

        return sp.csr_matrix(
            (np.array(vals, dtype=np.float32), (np.array(rows), np.array(cols))),
            shape=(n, n),
        )

    def build_layer_cooccurrence(self, cooc_matrix: sp.csr_matrix) -> sp.csr_matrix:
        """Layer 3: normalized co-occurrence similarity.

        Converts raw co-occurrence counts to PMI-like similarity.
        """
        if cooc_matrix.shape[0] == 0:
            return cooc_matrix

        # Normalize: PMI-style. sim(i,j) = cooc(i,j) / sqrt(cooc(i,i_total) * cooc(j,j_total))
        row_sums = np.asarray(cooc_matrix.sum(axis=1)).ravel().astype(np.float64)
        row_sums[row_sums == 0] = 1.0

        # Compute normalized co-occurrence
        cooc_f = cooc_matrix.astype(np.float64)
        inv_sqrt = 1.0 / np.sqrt(row_sums)
        # D^{-1/2} * C * D^{-1/2}
        diag = sp.diags(inv_sqrt)
        normalized = diag @ cooc_f @ diag

        # Clip to [0, 1]
        normalized.data = np.clip(normalized.data, 0.0, 1.0)

        return normalized.astype(np.float32).tocsr()

    def combine_layers(
        self,
        layers: List[sp.csr_matrix],
        weights: List[float],
    ) -> sp.csr_matrix:
        """Weighted combination of similarity layers."""
        if not layers:
            return sp.csr_matrix((0, 0), dtype=np.float32)

        n = layers[0].shape[0]
        total_weight = sum(w for w in weights if w > 0)
        if total_weight == 0:
            return sp.csr_matrix((n, n), dtype=np.float32)

        combined = sp.csr_matrix((n, n), dtype=np.float32)
        for layer, weight in zip(layers, weights):
            if weight > 0 and layer.shape == (n, n):
                combined = combined + (weight / total_weight) * layer

        return combined.tocsr()

    def find_merge_groups(
        self,
        combined: sp.csr_matrix,
        terms: Sequence[str],
        threshold: Optional[float] = None,
    ) -> List[List[str]]:
        """Find connected components above threshold as merge groups.

        Groups larger than ``max_group_size`` are split by iteratively
        removing the weakest edge until all sub-components are within
        the size limit.
        """
        if combined.shape[0] == 0:
            return []

        thresh = threshold if threshold is not None else self.config.merge_threshold
        max_size = self.config.max_group_size

        # Threshold the similarity matrix (in-place on CSR)
        thresholded = combined.tocsr()
        thresholded.data[thresholded.data < thresh] = 0
        thresholded.eliminate_zeros()

        # Find connected components
        n_components, labels = sp.csgraph.connected_components(
            thresholded, directed=False, return_labels=True
        )

        groups_idx: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            groups_idx.setdefault(label, []).append(idx)

        result: List[List[str]] = []
        for member_indices in groups_idx.values():
            if len(member_indices) < 2:
                continue
            if len(member_indices) <= max_size:
                result.append([terms[i] for i in member_indices])
                continue

            # Split oversized group: extract subgraph, remove weakest edges
            sub_idx = np.array(member_indices)
            sub_csr = thresholded[np.ix_(sub_idx, sub_idx)].tocsr()
            max_iter = sub_csr.nnz // 2 + 1  # upper bound: remove all edges
            prev_comp_sizes = None
            for _split_iter in range(max_iter):
                n_sub, sub_labels = sp.csgraph.connected_components(
                    sub_csr, directed=False, return_labels=True
                )
                # Check if all components are within limit
                comp_sizes = np.bincount(sub_labels)
                if comp_sizes.max() <= max_size:
                    break
                comp_sizes_dict = {i: int(s) for i, s in enumerate(comp_sizes) if s > 0}
                if prev_comp_sizes is not None and comp_sizes_dict == prev_comp_sizes:
                    break
                prev_comp_sizes = comp_sizes_dict
                # Find weakest edge in any oversized component via CSR direct access
                min_val, min_i, min_j = float("inf"), -1, -1
                for i in range(sub_csr.shape[0]):
                    if comp_sizes[sub_labels[i]] <= max_size:
                        continue
                    s0, s1 = sub_csr.indptr[i], sub_csr.indptr[i + 1]
                    for pos in range(s0, s1):
                        j = sub_csr.indices[pos]
                        v = sub_csr.data[pos]
                        if j > i and 0 < v < min_val:
                            min_val, min_i, min_j = v, i, j
                if min_i < 0:
                    break
                # Convert to LIL only for the single edge removal, then back
                sub_lil = sub_csr.tolil()
                sub_lil[min_i, min_j] = 0
                sub_lil[min_j, min_i] = 0
                sub_csr = sub_lil.tocsr()
                sub_csr.eliminate_zeros()

            # Collect final sub-components
            n_sub, sub_labels = sp.csgraph.connected_components(
                sub_csr, directed=False, return_labels=True
            )
            sub_groups: Dict[int, List[str]] = {}
            for si, sl in enumerate(sub_labels):
                sub_groups.setdefault(sl, []).append(terms[sub_idx[si]])
            for sg in sub_groups.values():
                if len(sg) >= 2:
                    result.append(sg)

        return result

    def generate_candidate_sets(
        self,
        groups: List[List[str]],
        top_df: Optional["pd.DataFrame"] = None,
        combined: Optional[sp.csr_matrix] = None,
        terms_list: Optional[Sequence[str]] = None,
    ) -> List[Dict]:
        """Format merge groups for downstream processing.

        Returns a list of dicts with group info for each merge candidate set.
        When *combined* and *terms_list* are provided, includes per-pair
        similarity scores for auto-merge confidence gating (P3).
        """

        term_to_idx = {t: i for i, t in enumerate(terms_list)} if terms_list is not None else {}

        candidates = []
        for group_id, group_terms in enumerate(groups):
            entry: Dict = {
                "group_id": group_id,
                "terms": group_terms,
                "size": len(group_terms),
            }

            if top_df is not None and not top_df.empty:
                # Add frequency info for the canonical term selection
                freq_map = {}
                for term in group_terms:
                    matches = top_df[top_df["term"] == term]
                    if not matches.empty:
                        freq_map[term] = int(matches["frequency"].sum())
                    else:
                        freq_map[term] = 0
                entry["freq_map"] = freq_map
                # Highest-frequency term as suggested canonical
                if freq_map:
                    entry["suggested_canonical"] = max(freq_map, key=freq_map.get)

            # Per-pair similarities for P3 auto-merge
            if combined is not None and term_to_idx:
                pair_sims: Dict[Tuple[str, str], float] = {}
                for i, ta in enumerate(group_terms):
                    for tb in group_terms[i + 1:]:
                        ia = term_to_idx.get(ta)
                        ib = term_to_idx.get(tb)
                        if ia is not None and ib is not None:
                            pair_sims[(ta, tb)] = float(combined[ia, ib])
                entry["pair_similarities"] = pair_sims

            candidates.append(entry)

        return candidates


__all__ = ["TermNetwork", "TermNetworkConfig"]
