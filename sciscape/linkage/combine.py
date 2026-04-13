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
from typing import Dict, List, Optional, Sequence

import numpy as np
import polars as pl

from .filters import filter_giant_component

log = logging.getLogger(__name__)


def combine_edge_layers(
    layers: Dict[str, pl.DataFrame],
    *,
    strategy: str = "rank",
    weights: Dict[str, float] | None = None,
    gcc: bool = True,
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
        Combination strategy: "union", "rank", "max", "vote".
    weights : dict, optional
        Per-layer weight multiplier, e.g. {"bc": 1.0, "cc": 0.5}.
        Default: all layers weighted equally (1.0).
    gcc : bool
        If True, filter to giant connected component after combining.

    Returns
    -------
    pl.DataFrame
        Combined edge table with uid1, uid2, rel_sum2.
    """
    if not layers:
        return pl.DataFrame({uid1_col: [], uid2_col: [], weight_col: []})

    layer_weights = weights or {name: 1.0 for name in layers}
    combined_parts: List[pl.DataFrame] = []

    for name, df in layers.items():
        if df.height == 0:
            continue
        lw = layer_weights.get(name, 1.0)

        if strategy == "rank":
            # 1/rank normalization: sort by weight desc, assign 1/rank score
            ranked = df.sort(weight_col, descending=True).with_row_index("_rank")
            normed = ranked.with_columns(
                (lw / (pl.col("_rank") + 1).cast(pl.Float64)).alias(weight_col)
            ).drop("_rank")
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
    elif strategy == "boosted":
        # Boosted sum: weight × number of layers containing this edge
        w_sum = all_edges.group_by([uid1_col, uid2_col]).agg(
            pl.col(weight_col).sum()
        )
        # Count layers per edge (each layer contributes 1 per occurrence)
        vote_parts = []
        for name, df in layers.items():
            if df.height > 0:
                vote_parts.append(
                    df.select(uid1_col, uid2_col).with_columns(pl.lit(1.0).alias("_v"))
                )
        if vote_parts:
            v_count = pl.concat(vote_parts).group_by([uid1_col, uid2_col]).agg(
                pl.col("_v").sum().alias("_n")
            )
            combined = w_sum.join(v_count, on=[uid1_col, uid2_col], how="left").with_columns(
                (pl.col(weight_col) * pl.col("_n")).alias(weight_col)
            ).drop("_n")
        else:
            combined = w_sum
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
        layers, strategy=strategy, weights=weights, gcc=gcc,
    )


__all__ = ["combine_edge_layers", "load_and_combine"]
