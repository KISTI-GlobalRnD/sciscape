"""Edge filtering and weight normalization for link-type edge tables.

Pipeline: build_dc/bc/cc → **filter** → **normalize** → combine
"""

from __future__ import annotations

import logging
from enum import Enum

import numpy as np
import polars as pl
from scipy import sparse
from scipy.sparse.csgraph import connected_components

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Weight normalization
# ═══════════════════════════════════════════════════════════════════

class WeightNorm(str, Enum):
    """Weight normalization methods applied before combining edge sets."""

    MAX = "max"           # w / max(w)  → [0, 1]
    MINMAX = "minmax"     # (w - min) / (max - min)  → [0, 1]
    RANK = "rank"         # rank / N  → (0, 1]
    ZSCORE = "zscore"     # (w - mean) / std  (not bounded)
    QUANTILE = "quantile" # empirical CDF  → [0, 1]


def normalize_weights(
    edges: pl.DataFrame,
    method: WeightNorm = WeightNorm.MAX,
    *,
    weight_col: str = "rel_sum2",
) -> pl.DataFrame:
    """Normalize edge weights in-place (returns new DataFrame).

    Parameters
    ----------
    edges : pl.DataFrame
        Edge table with a weight column.
    method : WeightNorm
        Normalization method.
    weight_col : str
        Name of the weight column.

    Returns
    -------
    pl.DataFrame
        Copy with normalized weights.
    """
    if edges.height == 0:
        return edges

    w = edges[weight_col]

    if method == WeightNorm.MAX:
        mx = w.max()
        if mx > 0:
            normed = w / mx
        else:
            normed = w
    elif method == WeightNorm.MINMAX:
        mn, mx = w.min(), w.max()
        span = mx - mn
        if span > 0:
            normed = (w - mn) / span
        else:
            normed = pl.lit(0.0).alias(weight_col).cast(pl.Float64)
            normed = pl.Series(weight_col, [0.0] * edges.height)
    elif method == WeightNorm.RANK:
        # rank / N → (0, 1]  (average ties)
        normed = w.rank("average") / edges.height
    elif method == WeightNorm.ZSCORE:
        mean = w.mean()
        std = w.std()
        if std and std > 0:
            normed = (w - mean) / std
        else:
            normed = w - mean
    elif method == WeightNorm.QUANTILE:
        # Empirical CDF: fraction of values ≤ w
        normed = w.rank("average") / edges.height
    else:
        raise ValueError(f"Unknown normalization: {method}")

    return edges.with_columns(normed.alias(weight_col))


# ═══════════════════════════════════════════════════════════════════
# Edge filters
# ═══════════════════════════════════════════════════════════════════

def filter_min_weight(
    edges: pl.DataFrame,
    min_weight: float,
    *,
    weight_col: str = "rel_sum2",
) -> pl.DataFrame:
    """Drop edges with weight below *min_weight*.

    Parameters
    ----------
    edges : pl.DataFrame
        Edge table.
    min_weight : float
        Minimum weight threshold (exclusive).
    """
    before = edges.height
    result = edges.filter(pl.col(weight_col) >= min_weight)
    log.info("filter_min_weight(%.6f): %d → %d edges", min_weight, before, result.height)
    return result


def filter_top_k(
    edges: pl.DataFrame,
    k: int,
    *,
    uid1_col: str = "uid1",
    uid2_col: str = "uid2",
    weight_col: str = "rel_sum2",
    mode: str = "symmetric",
) -> pl.DataFrame:
    """Keep only the top-*k* strongest edges per node.

    Parameters
    ----------
    edges : pl.DataFrame
        Edge table (undirected: each edge appears once).
    k : int
        Maximum number of neighbors per node.
    mode : str
        ``"symmetric"`` (default): keep edge if **either** endpoint has it in
        their top-k.  ``"mutual"``: keep only if **both** endpoints agree.

    Returns
    -------
    pl.DataFrame
        Filtered edge table.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if edges.height == 0:
        return edges

    before = edges.height

    # Expand to bidirectional for per-node ranking
    fwd = edges.select(
        pl.col(uid1_col).alias("node"),
        pl.col(uid2_col).alias("neighbor"),
        pl.col(weight_col).alias("w"),
    )
    rev = edges.select(
        pl.col(uid2_col).alias("node"),
        pl.col(uid1_col).alias("neighbor"),
        pl.col(weight_col).alias("w"),
    )
    bidi = pl.concat([fwd, rev])

    # Rank per node (descending weight)
    ranked = bidi.with_columns(
        pl.col("w")
        .rank("ordinal", descending=True)
        .over("node")
        .alias("rank")
    )

    # Keep edges where rank <= k
    top = ranked.filter(pl.col("rank") <= k)

    if mode == "symmetric":
        # Edge survives if either direction is in top-k
        keep_pairs = top.select(
            pl.min_horizontal("node", "neighbor").alias("a"),
            pl.max_horizontal("node", "neighbor").alias("b"),
        ).unique()
    elif mode == "mutual":
        # Edge survives only if both directions are in top-k
        fwd_top = top.select(
            pl.col("node").alias("a"),
            pl.col("neighbor").alias("b"),
        )
        rev_top = top.select(
            pl.col("neighbor").alias("a"),
            pl.col("node").alias("b"),
        )
        # Normalize pair order for join
        fwd_norm = fwd_top.with_columns(
            pl.min_horizontal("a", "b").alias("lo"),
            pl.max_horizontal("a", "b").alias("hi"),
        ).select("lo", "hi").unique()
        rev_norm = rev_top.with_columns(
            pl.min_horizontal("a", "b").alias("lo"),
            pl.max_horizontal("a", "b").alias("hi"),
        ).select("lo", "hi").unique()
        keep_pairs = fwd_norm.join(rev_norm, on=["lo", "hi"]).rename(
            {"lo": "a", "hi": "b"}
        )
    else:
        raise ValueError(f"mode must be 'symmetric' or 'mutual', got {mode!r}")

    # Join back to original edges
    edges_normed = edges.with_columns(
        pl.min_horizontal(uid1_col, uid2_col).alias("_a"),
        pl.max_horizontal(uid1_col, uid2_col).alias("_b"),
    )
    result = (
        edges_normed
        .join(keep_pairs, left_on=["_a", "_b"], right_on=["a", "b"], how="semi")
        .drop("_a", "_b")
    )

    log.info("filter_top_k(k=%d, mode=%s): %d → %d edges", k, mode, before, result.height)
    return result


def filter_giant_component(
    edges: pl.DataFrame,
    *,
    uid1_col: str = "uid1",
    uid2_col: str = "uid2",
) -> pl.DataFrame:
    """Keep only edges within the giant connected component.

    Useful after aggressive filtering (top-k, min_weight) that may fragment
    the graph.  Uses scipy sparse GCC detection (no igraph needed).
    """
    if edges.height == 0:
        return edges

    # ── Vectorized UID → int mapping via Polars join ──────────────
    all_uids = pl.concat([edges[uid1_col], edges[uid2_col]]).unique().sort()
    n = all_uids.len()
    uid_map = pl.DataFrame({
        "uid": all_uids,
        "_idx": np.arange(n, dtype=np.int32),
    })

    src = (
        edges.select(pl.col(uid1_col).alias("uid"))
        .join(uid_map, on="uid", how="left")["_idx"]
        .to_numpy()
    )
    dst = (
        edges.select(pl.col(uid2_col).alias("uid"))
        .join(uid_map, on="uid", how="left")["_idx"]
        .to_numpy()
    )

    # ── Build symmetric sparse adjacency ──────────────────────────
    ones = np.ones(len(src), dtype=np.float32)
    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    data = np.concatenate([ones, ones])
    adj = sparse.csr_matrix((data, (row, col)), shape=(n, n))

    # ── GCC via scipy (avoids igraph construction entirely) ───────
    n_comps, labels = connected_components(adj, directed=False)

    if n_comps == 1:
        log.info("filter_giant_component: graph already connected (%d edges)", edges.height)
        return edges

    # Find largest component label
    comp_sizes = np.bincount(labels)
    gcc_label = int(comp_sizes.argmax())
    gcc_mask = labels == gcc_label

    # Map back: gather UIDs in GCC, filter edges via is_in
    gcc_uids = all_uids.gather(np.nonzero(gcc_mask)[0])

    before = edges.height
    result = edges.filter(
        pl.col(uid1_col).is_in(gcc_uids) & pl.col(uid2_col).is_in(gcc_uids)
    )
    log.info("filter_giant_component: %d → %d edges (%d nodes)",
             before, result.height, int(gcc_mask.sum()))
    return result


__all__ = [
    "WeightNorm",
    "filter_giant_component",
    "filter_min_weight",
    "filter_top_k",
    "normalize_weights",
]
