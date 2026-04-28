"""Multi-layer edge combination and GCC filtering.

Combines multiple edge types (DC, BC, CC, Emb) into a single weighted
edge table, then filters to the giant connected component.

Combination strategies:
  - "union": keep all edges, sum weights across layers
  - "rank": 1/rank normalization per layer, then sum
  - "max": keep max weight per pair across layers
  - "vote": binary vote (edge present in k layers → weight = k)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import polars as pl

from .filters import compute_adaptive_k, filter_giant_component, filter_top_k

log = logging.getLogger(__name__)


def combine_edge_layers(
    layers: Dict[str, pl.DataFrame],
    *,
    strategy: str = "rank",
    weights: Dict[str, float] | None = None,
    gcc: bool = True,
    top_k: int | str = "auto",
    uid1_col: str = "uid1",
    uid2_col: str = "uid2",
    weight_col: str = "rel_sum2",
) -> pl.DataFrame:
    """Combine multiple edge layers into a single edge table.

    Parameters
    ----------
    layers : dict
        Mapping from layer name to edge DataFrame.
        Each must have uid1, uid2, rel_sum2 columns.
    strategy : str
        Combination strategy: "union", "rank", "max", "vote", "consensus".
    weights : dict, optional
        Per-layer weight multiplier, e.g. {"bc": 1.0, "cc": 0.5}.
        Default: all layers weighted equally (1.0).
    gcc : bool
        If True, filter to giant connected component after combining.
    top_k : int
        Per-node top-k filter applied to EACH layer BEFORE normalization.
        Default 30 (keep 30 strongest neighbors per node per layer).
        Set to 0 to disable.

    Returns
    -------
    pl.DataFrame
        Combined edge table with uid1, uid2, rel_sum2.
    """
    _VALID_STRATEGIES = {"union", "sum", "rank", "max", "vote", "consensus"}
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}, choose from {_VALID_STRATEGIES}")
    # "sum" is an alias for "union"
    if strategy == "sum":
        strategy = "union"

    if not layers:
        return pl.DataFrame({uid1_col: [], uid2_col: [], weight_col: []})

    # Step 0: per-node top-k filter (BEFORE normalization)
    # top_k="auto": adaptive k = sqrt(n), clamped to [5, 30]
    # top_k="balanced": adaptive k per layer so each contributes ~equal edges
    # top_k=int: same k for all layers
    filtered_layers: Dict[str, pl.DataFrame] = {}

    if top_k == "auto":
        # Estimate n_nodes from all layers combined
        _uid_parts = [
            pl.concat([df[uid1_col], df[uid2_col]])
            for df in layers.values() if df.height > 0
        ]
        if not _uid_parts:
            # All layers empty — no filtering needed
            filtered_layers = {name: df for name, df in layers.items()}
        else:
            all_uids = pl.concat(_uid_parts).unique()
            n_nodes_est = all_uids.len()
            effective_k = compute_adaptive_k(n_nodes_est)
            log.info("adaptive top_k: n=%d → k=%d (sqrt-based)", n_nodes_est, effective_k)
            for name, df in layers.items():
                if df.height == 0:
                    continue
                before = df.height
                df = filter_top_k(df, effective_k, uid1_col=uid1_col, uid2_col=uid2_col,
                                  weight_col=weight_col, mode="symmetric")
                log.info("top_k=%d on %s: %d → %d edges", effective_k, name, before, df.height)
                filtered_layers[name] = df
    elif top_k == "balanced":
        # Find sparsest layer's avg degree → use as target
        layer_stats = {}
        for name, df in layers.items():
            if df.height == 0:
                continue
            n_nodes = pl.concat([df[uid1_col], df[uid2_col]]).n_unique()
            avg_deg = 2 * df.height / n_nodes if n_nodes > 0 else 0
            layer_stats[name] = {"n_nodes": n_nodes, "avg_deg": avg_deg, "n_edges": df.height}

        if layer_stats:
            min_deg = min(s["avg_deg"] for s in layer_stats.values())
            target_deg = max(5, min(30, min_deg))  # clamp to [5, 30]
            log.info("balanced top_k: target avg_degree=%.1f (from sparsest layer)", target_deg)

            for name, df in layers.items():
                if df.height == 0:
                    continue
                stats = layer_stats.get(name)
                if stats:
                    k = max(5, min(30, round(target_deg)))
                    before = df.height
                    df = filter_top_k(df, k, uid1_col=uid1_col, uid2_col=uid2_col,
                                      weight_col=weight_col, mode="symmetric")
                    log.info("balanced top_k=%d on %s: %d → %d edges", k, name, before, df.height)
                filtered_layers[name] = df
    elif isinstance(top_k, int) and top_k > 0:
        for name, df in layers.items():
            if df.height == 0:
                continue
            before = df.height
            df = filter_top_k(df, top_k, uid1_col=uid1_col, uid2_col=uid2_col,
                              weight_col=weight_col, mode="symmetric")
            log.info("top_k=%d on %s: %d → %d edges", top_k, name, before, df.height)
            filtered_layers[name] = df
    else:
        filtered_layers = {name: df for name, df in layers.items() if df.height > 0}

    # Auto-compute layer weights if not specified:
    # inverse of edge count → layers with more edges get less weight per edge
    if weights is None and len(filtered_layers) > 1:
        edge_counts = {name: df.height for name, df in filtered_layers.items()}
        total_inv = sum(1.0 / c for c in edge_counts.values() if c > 0)
        n_layers_total = len(edge_counts)
        layer_weights = {
            name: (1.0 / c) / total_inv * n_layers_total
            for name, c in edge_counts.items() if c > 0
        }
        log.info("Auto layer weights (edge-count balanced): %s",
                 {k: f"{v:.3f}" for k, v in layer_weights.items()})
    else:
        layer_weights = weights or {name: 1.0 for name in filtered_layers}

    combined_parts: List[pl.DataFrame] = []

    for name, df in filtered_layers.items():
        lw = layer_weights.get(name, 1.0)

        if strategy == "rank":
            # 1/rank normalization: rank by weight desc, score = 1/rank
            normed = df.with_columns(
                (lw / pl.col(weight_col).rank("ordinal", descending=True).cast(pl.Float64)).alias(weight_col)
            )
            combined_parts.append(normed.select(uid1_col, uid2_col, weight_col))

        elif strategy == "vote":
            # Binary: each layer contributes 1.0 * layer_weight per edge
            voted = df.select(uid1_col, uid2_col).with_columns(
                pl.lit(lw).alias(weight_col)
            )
            combined_parts.append(voted)

        elif strategy == "max":
            # Keep original weights scaled by layer weight
            scaled = df.with_columns(
                (pl.col(weight_col) * lw).alias(weight_col)
            ).select(uid1_col, uid2_col, weight_col)
            combined_parts.append(scaled)

        else:  # "union" — raw sum
            scaled = df.with_columns(
                (pl.col(weight_col) * lw).alias(weight_col)
            ).select(uid1_col, uid2_col, weight_col)
            combined_parts.append(scaled)

    if not combined_parts:
        return pl.DataFrame({uid1_col: [], uid2_col: [], weight_col: []})

    all_edges = pl.concat(combined_parts)

    if strategy == "max":
        # Keep max weight per pair
        combined = all_edges.group_by([uid1_col, uid2_col]).agg(
            pl.col(weight_col).max()
        )
    elif strategy == "consensus":
        # Consensus: weight_sum × n_layers (single group_by, no join)
        # all_edges already has per-layer weights from the loop above
        # Add vote column, then single aggregation
        vote_tagged = all_edges.with_columns(pl.lit(1.0).alias("_v"))
        combined = vote_tagged.group_by([uid1_col, uid2_col]).agg(
            pl.col(weight_col).sum().alias("_w"),
            pl.col("_v").sum().alias("_n"),
        ).with_columns(
            (pl.col("_w") * pl.col("_n")).alias(weight_col)
        ).drop("_w", "_n")
    else:
        # Sum weights per pair (union, rank, vote)
        combined = all_edges.group_by([uid1_col, uid2_col]).agg(
            pl.col(weight_col).sum()
        )

    log.info(
        "combine_edge_layers: %d layers, strategy=%s → %d edges",
        len(layers), strategy, combined.height,
    )

    # GCC filter
    if gcc:
        before = combined.height
        combined = filter_giant_component(
            combined, uid1_col=uid1_col, uid2_col=uid2_col,
        )
        log.info(
            "GCC filter: %d → %d edges (%.1f%%)",
            before, combined.height,
            100 * combined.height / before if before else 0,
        )

    return combined


def load_and_combine(
    edge_dir: Path,
    layer_names: Sequence[str] = ("bc_cosine", "cc_cosine", "dc_fractional"),
    *,
    strategy: str = "rank",
    weights: Dict[str, float] | None = None,
    gcc: bool = True,
    top_k: int | str = "auto",
) -> pl.DataFrame:
    """Load parquet edge files from a directory and combine them.

    Parameters
    ----------
    edge_dir : Path
        Directory containing edge parquet files.
    layer_names : sequence of str
        Names of edge files (without .parquet extension).
    strategy : str
        Combination strategy.
    weights : dict, optional
        Per-layer weight multiplier.
    gcc : bool
        Filter to GCC after combining.

    Returns
    -------
    pl.DataFrame
    """
    edge_dir = Path(edge_dir)
    layers: Dict[str, pl.DataFrame] = {}
    for name in layer_names:
        path = edge_dir / f"{name}.parquet"
        if path.exists():
            df = pl.read_parquet(path)
            layers[name] = df
            log.info("Loaded %s: %d edges", name, df.height)
        else:
            log.warning("Edge file not found: %s", path)

    if not layers:
        raise FileNotFoundError(
            f"No edge files found in {edge_dir} for layers: {layer_names}"
        )

    return combine_edge_layers(
        layers, strategy=strategy, weights=weights, gcc=gcc, top_k=top_k,
    )


def load_combine_and_cluster(
    edge_dir: Path,
    layer_names: Sequence[str] = ("bc_cosine", "cc_cosine", "dc_fractional"),
    *,
    strategy: str = "consensus",
    top_k: int | str = "auto",
    target_max_pct: float = 3.0,
    min_size: int = 100,
    progress: callable | None = None,
) -> Dict[str, Any]:
    """End-to-end: load → combine → auto-γ → Leiden → postprocess.

    Returns dict with gamma, membership, n_clusters, etc.
    """
    from ..clustering.auto_gamma import find_gamma

    combined = load_and_combine(
        edge_dir, layer_names,
        strategy=strategy, gcc=True, top_k=top_k,
    )
    n = pl.concat([combined["uid1"], combined["uid2"]]).n_unique()
    if progress:
        progress(f"Combined: {n:,} nodes, {combined.height:,} edges")

    result = find_gamma(
        combined,
        target_max_pct=target_max_pct,
        min_size=min_size,
        progress=progress,
    )

    return {
        "gamma": result.gamma,
        "n_clusters": result.n_clusters,
        "max_pct": result.max_pct,
        "top5": result.top5,
        "membership": result.membership,
        "n_nodes": n,
        "n_edges": combined.height,
        "combined_edges": combined,
    }


__all__ = ["combine_edge_layers", "load_and_combine", "load_combine_and_cluster"]
