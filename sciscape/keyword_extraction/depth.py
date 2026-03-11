"""Conceptual depth estimation for keywords (Stage 9).

Estimates how broad or specific each keyword is within its cluster.
Uses multiple signals:
1. doc_coverage (inverted): terms appearing in many docs are broader
2. cross_cluster_count (inverted): terms in many clusters are broader
3. ngram_length: longer phrases tend to be more specific
4. co-occurrence asymmetry: P(A|B) >> P(B|A) implies B is broader than A

Depth levels: 0 (broadest) to n_levels-1 (most specific).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse as sp


@dataclass
class DepthConfig:
    """Configuration for conceptual depth estimation."""

    enabled: bool = False
    n_levels: int = 4  # 0, 1, 2, 3
    weight_doc_coverage: float = 0.3
    weight_cross_cluster: float = 0.3
    weight_ngram_length: float = 0.1
    weight_cooc_asymmetry: float = 0.3
    asymmetry_threshold: float = 0.3  # minimum asymmetry to count


def _compute_cross_cluster_counts(top_df: pd.DataFrame) -> pd.Series:
    """Count how many distinct clusters each term appears in."""
    return top_df.groupby("term")["cluster_id"].nunique()


def _compute_asymmetry_scores(
    top_df: pd.DataFrame,
    cooc_matrix: Optional[sp.csr_matrix],
    term_to_idx: dict,
    threshold: float,
) -> pd.Series:
    """Compute co-occurrence asymmetry score per term.

    For each term A, count how many other terms B satisfy P(A|B) >> P(B|A),
    meaning B is a "parent" (broader) concept. More parents → deeper term.
    """
    if cooc_matrix is None or cooc_matrix.nnz == 0:
        return pd.Series(0.0, index=top_df.index)

    terms = top_df["term"].tolist()
    row_sums = np.asarray(cooc_matrix.sum(axis=1)).ravel().astype(np.float64)
    row_sums[row_sums == 0] = 1.0

    scores = []
    for term in terms:
        idx = term_to_idx.get(term)
        if idx is None:
            scores.append(0.0)
            continue

        # Count parents: terms B where P(term|B) > P(B|term) + threshold
        parent_count = 0
        row = cooc_matrix.getrow(idx)
        for j_pos in range(row.nnz):
            j = row.indices[j_pos]
            cooc_val = float(row.data[j_pos])
            if cooc_val == 0:
                continue
            # P(term|j) = cooc(term,j) / sum_cooc(j)
            p_term_given_j = cooc_val / row_sums[j]
            # P(j|term) = cooc(term,j) / sum_cooc(term)
            p_j_given_term = cooc_val / row_sums[idx]
            asymmetry = p_term_given_j - p_j_given_term
            if asymmetry > threshold:
                parent_count += 1

        scores.append(float(parent_count))

    return pd.Series(scores, index=top_df.index)


def _normalize_signal(values: pd.Series) -> pd.Series:
    """Min-max normalize a signal to [0, 1]."""
    vmin = values.min()
    vmax = values.max()
    if vmax == vmin:
        return pd.Series(0.5, index=values.index)
    return (values - vmin) / (vmax - vmin)


def estimate_depth(
    top_df: pd.DataFrame,
    cooc_matrix: Optional[sp.csr_matrix] = None,
    selected_terms: Optional[list] = None,
    config: Optional[DepthConfig] = None,
) -> pd.DataFrame:
    """Add depth_level and depth_score columns to top_df.

    Parameters
    ----------
    top_df : pd.DataFrame
        Keyword DataFrame with at least cluster_id, term, frequency columns.
    cooc_matrix : sparse matrix, optional
        Term co-occurrence matrix (from cooccurrence.py).
    selected_terms : list, optional
        Term list matching cooc_matrix rows/cols.
    config : DepthConfig, optional
        Configuration. Uses defaults if not provided.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added depth_score and depth_level columns.
    """
    if config is None:
        config = DepthConfig()

    if top_df.empty:
        return top_df.assign(depth_score=[], depth_level=[])

    signals = []
    weights = []

    # Signal 1: doc_coverage (inverted — high coverage = broad = low depth)
    if "doc_coverage" in top_df.columns and config.weight_doc_coverage > 0:
        doc_cov = top_df["doc_coverage"].astype(float)
        # Invert: high coverage → low depth score
        signals.append(_normalize_signal(doc_cov.max() - doc_cov))
        weights.append(config.weight_doc_coverage)

    # Signal 2: cross-cluster count (inverted — many clusters = broad)
    if config.weight_cross_cluster > 0:
        cross_counts = _compute_cross_cluster_counts(top_df)
        per_row_cross = top_df["term"].map(cross_counts).fillna(1).astype(float)
        signals.append(_normalize_signal(per_row_cross.max() - per_row_cross))
        weights.append(config.weight_cross_cluster)

    # Signal 3: ngram length (longer = more specific = deeper)
    if config.weight_ngram_length > 0:
        ngram_len = top_df["term"].map(lambda t: len(str(t).split())).astype(float)
        signals.append(_normalize_signal(ngram_len))
        weights.append(config.weight_ngram_length)

    # Signal 4: co-occurrence asymmetry (more parents = deeper)
    if cooc_matrix is not None and selected_terms and config.weight_cooc_asymmetry > 0:
        term_to_idx = {t: i for i, t in enumerate(selected_terms)}
        asym = _compute_asymmetry_scores(top_df, cooc_matrix, term_to_idx, config.asymmetry_threshold)
        signals.append(_normalize_signal(asym))
        weights.append(config.weight_cooc_asymmetry)

    if not signals:
        return top_df.assign(depth_score=0.5, depth_level=0)

    # Weighted combination
    total_weight = sum(weights)
    depth_score = sum(s * w for s, w in zip(signals, weights)) / total_weight

    # Quantile-based level assignment
    n_levels = max(2, config.n_levels)
    quantiles = np.linspace(0, 1, n_levels + 1)[1:-1]
    thresholds = np.quantile(depth_score.values, quantiles) if len(depth_score) > 1 else []
    depth_level = np.digitize(depth_score.values, thresholds)

    return top_df.assign(
        depth_score=depth_score.values,
        depth_level=depth_level,
    )
