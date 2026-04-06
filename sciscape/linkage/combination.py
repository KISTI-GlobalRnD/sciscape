"""Combine multiple link-type edge sets into a single edge table.

Combination methods
-------------------
**Layer-agnostic** (each input pre-normalized to [0, 1]):
    SUM         Σ wᵢ
    MAX         max(wᵢ)
    MIN         min(wᵢ)  — only edges present in ALL layers survive
    NOISY_OR    1 - Π(1 - wᵢ)

**Consensus** (reward edges confirmed by multiple layers):
    CONSENSUS   (Σ wᵢ / n) × consensus_count

**Weighted**:
    WEIGHTED_SUM  Σ αᵢ · wᵢ  — requires a ``weights`` dict

**Multiplicative** (edges absent in any layer get weight 0):
    PRODUCT         Π wᵢ
    GEOMETRIC_MEAN  (Π wᵢ)^(1/n)
    HARMONIC_MEAN   n / Σ(1/wᵢ)  — only for edges in ALL layers

All methods pre-normalize each layer to [0, 1] by default.
Use ``pre_normalize=False`` to skip.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import numpy as np
import polars as pl
from scipy import sparse

from .config import CombineMethod

log = logging.getLogger(__name__)


# ── Sparse helpers ────────────────────────────────────────────────


def _df_to_sparse(
    df: pl.DataFrame,
    uid_map: pl.DataFrame,
    n: int,
) -> sparse.csr_matrix:
    """Convert edge DataFrame to symmetric sparse matrix (vectorized).

    Uses Polars join for UID → index mapping instead of Python dict lookup.
    """
    src = (
        df.select(pl.col("uid1").alias("uid"))
        .join(uid_map, on="uid", how="left")["_idx"]
        .to_numpy()
        .astype(np.int32)
    )
    dst = (
        df.select(pl.col("uid2").alias("uid"))
        .join(uid_map, on="uid", how="left")["_idx"]
        .to_numpy()
        .astype(np.int32)
    )
    w = df["rel_sum2"].to_numpy().astype(np.float32)

    # Symmetric: add both directions
    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    data = np.concatenate([w, w])

    return sparse.csr_matrix((data, (row, col)), shape=(n, n))


def _sparse_to_df(
    M: sparse.csr_matrix,
    categories: pl.Series,
) -> pl.DataFrame:
    """Upper triangle of sparse matrix → edge DataFrame (vectorized).

    Uses ``Series.gather()`` for index→UID mapping instead of Python
    list comprehension.
    """
    upper = sparse.triu(M, k=1).tocoo()
    mask = upper.data != 0
    row_idx = upper.row[mask]
    col_idx = upper.col[mask]

    return pl.DataFrame({
        "uid1": categories.gather(row_idx),
        "uid2": categories.gather(col_idx),
        "rel_sum2": upper.data[mask].astype(np.float64),
    })


def _normalize_01(M: sparse.csr_matrix) -> sparse.csr_matrix:
    """Scale weights to [0, 1] by dividing by max."""
    if M.nnz == 0:
        return M
    mx = M.data.max()
    if mx > 0:
        M = M.copy()
        M.data = M.data / mx
    return M


# ── Combination functions ─────────────────────────────────────────

def _combine_sum(
    matrices: Sequence[sparse.csr_matrix],
    **_kw,
) -> sparse.csr_matrix:
    # Each `+` creates a new matrix; no initial copy needed
    result = matrices[0]
    for M in matrices[1:]:
        result = result + M
    return result


def _combine_max(
    matrices: Sequence[sparse.csr_matrix],
    **_kw,
) -> sparse.csr_matrix:
    result = matrices[0]
    for M in matrices[1:]:
        result = result.maximum(M)
    return result


def _combine_min(
    matrices: Sequence[sparse.csr_matrix],
    **_kw,
) -> sparse.csr_matrix:
    """Element-wise minimum — only edges present in ALL layers survive."""
    result = matrices[0]
    for M in matrices[1:]:
        result = result.minimum(M)
    return result


def _combine_consensus(
    matrices: Sequence[sparse.csr_matrix],
    **_kw,
) -> sparse.csr_matrix:
    """Consensus: (Σ wᵢ / n) × consensus_count.

    Edges present in more layers get a multiplicative bonus equal to
    the number of layers that agree on the edge.
    """
    n_layers = len(matrices)
    # Sum of weights — each `+` creates fresh matrix, no copy needed
    weight_sum = matrices[0].astype(np.float64)
    for M in matrices[1:]:
        weight_sum = weight_sum + M

    # Consensus count: binary sum (reuse astype copy)
    consensus = matrices[0].astype(np.float64, copy=True)
    consensus.data[:] = 1.0
    for M in matrices[1:]:
        binary = M.astype(np.float64, copy=True)
        binary.data[:] = 1.0
        consensus = consensus + binary

    # (sum / n_layers) × count — mutate weight_sum directly (already fresh)
    weight_sum.data /= n_layers
    return weight_sum.multiply(consensus)


def _combine_noisy_or(
    matrices: Sequence[sparse.csr_matrix],
    **_kw,
) -> sparse.csr_matrix:
    """Noisy-OR: 1 - Π(1 - wᵢ).  Vectorized via log-space.

    For missing entries (w=0), log(1-0)=0, so they contribute nothing
    to the sum — exactly the correct identity for noisy-OR.
    """
    log_comp_sum = sparse.csr_matrix(matrices[0].shape, dtype=np.float64)
    for M in matrices:
        M_log = M.astype(np.float64, copy=True)
        M_log.data = np.log1p(-np.clip(M_log.data, 0, 1 - 1e-15))
        log_comp_sum = log_comp_sum + M_log

    # log_comp_sum is fresh from additions — mutate directly
    log_comp_sum.data = 1.0 - np.exp(log_comp_sum.data)
    return log_comp_sum


def _combine_weighted_sum(
    matrices: Sequence[sparse.csr_matrix],
    *,
    layer_weights: Sequence[float],
    **_kw,
) -> sparse.csr_matrix:
    """Weighted sum: Σ αᵢ · Mᵢ."""
    result = matrices[0] * layer_weights[0]
    for M, alpha in zip(matrices[1:], layer_weights[1:]):
        result = result + M * alpha
    return result


def _combine_product(
    matrices: Sequence[sparse.csr_matrix],
    **_kw,
) -> sparse.csr_matrix:
    """Element-wise product — edge absent in any layer → 0."""
    result = matrices[0]
    for M in matrices[1:]:
        result = result.multiply(M)
    return result


def _combine_geometric_mean(
    matrices: Sequence[sparse.csr_matrix],
    **_kw,
) -> sparse.csr_matrix:
    """Geometric mean: (Π wᵢ)^(1/n)."""
    product = _combine_product(matrices)
    n_layers = len(matrices)
    # multiply() returns fresh matrix — mutate directly
    product.data = np.power(product.data, 1.0 / n_layers)
    return product


def _combine_harmonic_mean(
    matrices: Sequence[sparse.csr_matrix],
    **_kw,
) -> sparse.csr_matrix:
    """Harmonic mean: n / Σ(1/wᵢ) — only edges in ALL layers.

    Vectorized: sum reciprocals and count layers per edge using sparse
    matrix addition, then filter by count == n_layers.
    """
    n_layers = len(matrices)

    recip_sum = sparse.csr_matrix(matrices[0].shape, dtype=np.float64)
    count = sparse.csr_matrix(matrices[0].shape, dtype=np.float64)

    for M in matrices:
        M_recip = M.astype(np.float64, copy=True)
        M_recip.data = 1.0 / M_recip.data
        recip_sum = recip_sum + M_recip

        M_bin = M.astype(np.float64, copy=True)
        M_bin.data[:] = 1.0
        count = count + M_bin

    recip_sum.sort_indices()
    count.sort_indices()

    # Mutate recip_sum directly (fresh from additions)
    all_present = count.data == n_layers
    recip_sum.data = np.where(all_present, n_layers / recip_sum.data, 0.0)
    recip_sum.eliminate_zeros()
    return recip_sum


# ── Dispatch ──────────────────────────────────────────────────────

_COMBINE_FN = {
    CombineMethod.SUM: _combine_sum,
    CombineMethod.MAX: _combine_max,
    CombineMethod.MIN: _combine_min,
    CombineMethod.CONSENSUS: _combine_consensus,
    CombineMethod.NOISY_OR: _combine_noisy_or,
    CombineMethod.WEIGHTED_SUM: _combine_weighted_sum,
    CombineMethod.PRODUCT: _combine_product,
    CombineMethod.GEOMETRIC_MEAN: _combine_geometric_mean,
    CombineMethod.HARMONIC_MEAN: _combine_harmonic_mean,
}


# ── Public API ────────────────────────────────────────────────────

def combine_edges(
    edge_sets: Dict[str, pl.DataFrame],
    method: CombineMethod = CombineMethod.SUM,
    *,
    weights: Optional[Dict[str, float]] = None,
    pre_normalize: bool = True,
) -> pl.DataFrame:
    """Combine multiple edge DataFrames into one.

    Parameters
    ----------
    edge_sets : dict[str, pl.DataFrame]
        Mapping from link-type name to edge DataFrame
        (columns: ``uid1``, ``uid2``, ``rel_sum2``).
    method : CombineMethod
        How to combine weights.
    weights : dict[str, float], optional
        Per-layer weights for ``WEIGHTED_SUM``.  Keys must match *edge_sets*.
        Ignored for other methods.
    pre_normalize : bool
        If True (default), each input is normalized to [0, 1] before combining.

    Returns
    -------
    pl.DataFrame
        Combined edge DataFrame (uid1, uid2, rel_sum2).

    Examples
    --------
    >>> combined = combine_edges(
    ...     {"bc": bc_edges, "cc": cc_edges},
    ...     method=CombineMethod.WEIGHTED_SUM,
    ...     weights={"bc": 0.7, "cc": 0.3},
    ... )
    """
    if not edge_sets:
        raise ValueError("edge_sets must be non-empty")
    if len(edge_sets) == 1:
        return next(iter(edge_sets.values()))

    names = list(edge_sets.keys())

    if method == CombineMethod.WEIGHTED_SUM:
        if weights is None:
            raise ValueError("WEIGHTED_SUM requires a `weights` dict")
        missing = set(names) - set(weights)
        if missing:
            raise ValueError(f"weights missing keys: {sorted(missing)}")

    # Build shared node universe — vectorized via Polars unique+sort+join
    uid_series = []
    for df in edge_sets.values():
        uid_series.append(df["uid1"])
        uid_series.append(df["uid2"])

    all_uids = pl.concat(uid_series).unique().sort()
    n = all_uids.len()

    uid_map = pl.DataFrame({
        "uid": all_uids,
        "_idx": np.arange(n, dtype=np.int32),
    })

    # Convert to sparse (optionally normalize) — vectorized joins
    matrices = []
    for name in names:
        df = edge_sets[name]
        M = _df_to_sparse(df, uid_map, n)
        if pre_normalize:
            M = _normalize_01(M)
        log.info("  %s: %d edges%s",
                 name, M.nnz // 2, " (normalized)" if pre_normalize else "")
        matrices.append(M)

    # Combine
    combine_fn = _COMBINE_FN[method]
    kwargs = {}
    if method == CombineMethod.WEIGHTED_SUM:
        kwargs["layer_weights"] = [weights[name] for name in names]
    combined = combine_fn(matrices, **kwargs)

    n_edges = sparse.triu(combined, k=1).nnz
    log.info("Combined [%s] via %s: %d edges", "+".join(names), method.value, n_edges)

    return _sparse_to_df(combined, all_uids)


def priority_fill_edges(
    edge_sets: Dict[str, pl.DataFrame],
    k: int,
    *,
    layer_priority: Optional[Dict[str, int]] = None,
    k_pool: int = 30,
    uid1_col: str = "uid1",
    uid2_col: str = "uid2",
    weight_col: str = "rel_sum2",
) -> pl.DataFrame:
    """Priority-based slot filling: consensus edges first, then single-layer.

    For each node, fill *k* neighbor slots using local (per-node) ranking:

    1. Edges present in 2+ layers (consensus) — sorted by rank sum (lower = better)
    2. Single-layer edges — sorted by local rank
    3. Tie-breaking — ``layer_priority`` (lower value = preferred)

    Output weight for each edge is its consensus count (number of agreeing
    layers), providing a natural weight gradient without global normalization.

    This is the discrete special case of :func:`combine_edges` with
    :attr:`CombineMethod.CONSENSUS`, suitable when the number of layers is
    small (2–4) and global weight normalization is undesirable.

    Parameters
    ----------
    edge_sets : dict[str, pl.DataFrame]
        Mapping from layer name to edge DataFrame.
    k : int
        Number of neighbor slots per node in the combined graph.
    layer_priority : dict[str, int], optional
        Tie-breaking priority (lower = preferred).  Defaults to alphabetical
        order of layer names.
    k_pool : int
        Candidates per node per layer to consider (default 30).
    uid1_col, uid2_col, weight_col : str
        Column names in the input DataFrames.

    Returns
    -------
    pl.DataFrame
        Combined edge DataFrame (uid1, uid2, rel_sum2) where rel_sum2
        is the consensus count.
    """
    if not edge_sets:
        raise ValueError("edge_sets must be non-empty")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    names = list(edge_sets.keys())
    if layer_priority is None:
        layer_priority = {name: i for i, name in enumerate(sorted(names))}

    # ── Step 1: Build per-node local rank for each layer (Polars) ──
    ranked_frames = []
    for name in names:
        df = edge_sets[name]
        fwd = df.select(
            pl.col(uid1_col).alias("node"),
            pl.col(uid2_col).alias("neighbor"),
            pl.col(weight_col).alias("w"),
        )
        rev = df.select(
            pl.col(uid2_col).alias("node"),
            pl.col(uid1_col).alias("neighbor"),
            pl.col(weight_col).alias("w"),
        )
        bidi = pl.concat([fwd, rev])

        ranked = (
            bidi.with_columns(
                pl.col("w")
                .rank("ordinal", descending=True)
                .over("node")
                .alias("rank")
            )
            .filter(pl.col("rank") <= k_pool)
            .with_columns(pl.lit(name).alias("layer"))
        )
        ranked_frames.append(ranked)

    all_ranked = pl.concat(ranked_frames)

    # ── Step 2: Count layers per (node, neighbor) pair ─────────────
    candidates = (
        all_ranked
        .group_by("node", "neighbor")
        .agg(
            pl.col("layer").n_unique().cast(pl.Int32).alias("n_layers"),
            pl.col("rank").sum().alias("rank_sum"),
            pl.col("layer").first().alias("first_layer"),
        )
    )

    # Add priority for the best (lowest-priority) layer
    prio_map = pl.DataFrame({
        "first_layer": list(layer_priority.keys()),
        "_prio": list(layer_priority.values()),
    })
    candidates = candidates.join(prio_map, on="first_layer", how="left").with_columns(
        pl.col("_prio").fill_null(99),
    )

    # ── Step 3: Per-node top-k selection ───────────────────────────
    # Sort: -n_layers (desc), rank_sum (asc), _prio (asc)
    selected = (
        candidates
        .with_columns(
            pl.col("n_layers").neg().alias("_neg_nl"),
        )
        .sort("_neg_nl", "rank_sum", "_prio")
        .with_columns(
            pl.col("_neg_nl")
            .cum_count()
            .over("node")
            .alias("_slot")
        )
        .filter(pl.col("_slot") <= k)
    )

    # ── Step 4: Deduplicate to undirected edges ────────────────────
    edges_df = (
        selected
        .with_columns(
            pl.min_horizontal("node", "neighbor").alias("lo"),
            pl.max_horizontal("node", "neighbor").alias("hi"),
        )
        .group_by("lo", "hi")
        .agg(pl.col("n_layers").max())
        .rename({"lo": "uid1", "hi": "uid2", "n_layers": "rel_sum2"})
        .with_columns(pl.col("rel_sum2").cast(pl.Float64))
    )

    log.info(
        "priority_fill [%s] k=%d: %d edges",
        "+".join(names), k, edges_df.height,
    )

    if edges_df.height == 0:
        return pl.DataFrame({"uid1": [], "uid2": [], "rel_sum2": []}).cast(
            {"uid1": pl.Utf8, "uid2": pl.Utf8, "rel_sum2": pl.Float64}
        )

    return edges_df


__all__ = ["combine_edges", "priority_fill_edges"]
