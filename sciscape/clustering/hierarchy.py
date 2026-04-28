"""Hierarchy helpers for Leiden clustering results."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Mapping

import numpy as np
import polars as pl

from .config import ClusterTables

log = logging.getLogger(__name__)


def get_cluster_hierarchy(
    df: pl.DataFrame,
    *,
    levels: Iterable[str] | None = None,
) -> Dict[str, object]:
    """Build a nested cluster hierarchy from a membership table."""

    cluster_cols = [col for col in df.columns if col.startswith("cluster_")]
    if not cluster_cols:
        raise ValueError("No cluster columns found; expected columns prefixed with 'cluster_'")

    if levels is None:
        ordered_cols = cluster_cols
    else:
        ordered_cols = []
        for level in levels:
            col = level if level.startswith("cluster_") else f"cluster_{level}"
            if col not in cluster_cols:
                raise ValueError(f"Requested level '{level}' not present in membership table")
            ordered_cols.append(col)

    if not ordered_cols:
        raise ValueError("No cluster levels to build hierarchy from")

    level_names = [col.removeprefix("cluster_") for col in ordered_cols]

    counts_per_level: Dict[str, Dict[int, int]] = {}
    for col, level in zip(ordered_cols, level_names):
        counts_df = df.group_by(col).agg(pl.len().alias("size"))
        counts_per_level[level] = dict(zip(
            counts_df[col].to_list(),
            counts_df["size"].cast(int).to_list(),
        ))

    transition_counts: Dict[tuple[str, str], Dict[int, Dict[int, int]]] = {}
    for parent_col, child_col in zip(ordered_cols, ordered_cols[1:]):
        parent_level = parent_col.removeprefix("cluster_")
        child_level = child_col.removeprefix("cluster_")
        pairs = df.group_by([parent_col, child_col]).agg(pl.len().alias("size"))
        mapping: Dict[int, Dict[int, int]] = {}
        for p, c, s in zip(
            pairs[parent_col].to_list(),
            pairs[child_col].to_list(),
            pairs["size"].cast(int).to_list(),
        ):
            mapping.setdefault(p, {})[c] = s
        transition_counts[(parent_level, child_level)] = mapping

    nodes: Dict[str, Dict[int, Dict[str, object]]] = {
        level: {
            cluster_id: {"size": size}
            for cluster_id, size in cluster_sizes.items()
        }
        for level, cluster_sizes in counts_per_level.items()
    }

    for parent_level, child_level in reversed(
        list(zip(level_names, level_names[1:]))
    ):
        parent_nodes = nodes[parent_level]
        child_nodes = nodes[child_level]
        parent_to_children = transition_counts.get((parent_level, child_level), {})
        for parent_id, node in parent_nodes.items():
            children_map = parent_to_children.get(parent_id)
            if not children_map:
                continue
            node["children"] = {
                child_id: child_nodes[child_id]
                for child_id in sorted(children_map)
            }

    return {
        "levels": level_names,
        "clusters": nodes[level_names[0]],
    }


def build_cluster_tables(
    df: pl.DataFrame,
    *,
    levels: Iterable[str] | None = None,
    resolutions: Mapping[str, float] | None = None,
    qualities: Mapping[str, float] | None = None,
) -> ClusterTables:
    """Create membership and description tables with hierarchical indices."""

    if levels is None:
        level_sequence = tuple(
            col.removeprefix("cluster_")
            for col in df.columns
            if col.startswith("cluster_")
        )
    else:
        level_sequence = tuple(levels)

    if not level_sequence:
        raise ValueError("No cluster levels provided to build_cluster_tables")

    cluster_cols = [f"cluster_{level}" for level in level_sequence]
    missing = [col for col in cluster_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Membership table missing columns: {missing}")

    index_exprs = _rust_hierarchy_index_series(df, level_sequence, cluster_cols)
    if index_exprs is None:
        index_exprs = []
        for idx, (level_name, cluster_col) in enumerate(zip(level_sequence, cluster_cols)):
            if idx == 0:
                expr = pl.col(cluster_col).rank("dense")
            else:
                expr = pl.col(cluster_col).rank("dense").over(cluster_cols[:idx])
            index_exprs.append(expr.cast(pl.Int64).alias(level_name))

    membership = df.with_columns(index_exprs)

    index_columns = list(level_sequence)
    membership = membership.with_columns(
        pl.concat_str([pl.col(col).cast(pl.Utf8) for col in index_columns], separator=".")
        .alias("total_index")
    )

    membership_table = membership.select(
        ["uid", *index_columns, "total_index", *cluster_cols]
    )

    description = (
        membership.group_by(index_columns)
        .agg(pl.len().alias("number_of_nodes"))
        .with_columns(
            pl.concat_str(
                [pl.col(col).cast(pl.Utf8) for col in index_columns],
                separator=".",
            ).alias("total_index"),
            pl.col("number_of_nodes").cast(pl.Int64),
        )
        .select(index_columns + ["total_index", "number_of_nodes"])
        .sort(index_columns)
    )

    return ClusterTables(
        membership=membership_table,
        description=description,
        raw_membership=df,
        levels=level_sequence,
        resolutions=dict(resolutions) if resolutions is not None else None,
        qualities=dict(qualities) if qualities is not None else None,
    )


def _rust_hierarchy_index_series(
    df: pl.DataFrame,
    level_sequence: tuple[str, ...],
    cluster_cols: list[str],
) -> list[pl.Series] | None:
    """Compute dense hierarchy indices in Rust for compact u32 cluster columns."""
    if any(df[col].dtype != pl.UInt32 for col in cluster_cols):
        return None

    try:
        import sciscape_leiden as _rust
    except ImportError:
        return None

    index_fn = getattr(_rust, "rust_hierarchy_indices_u32", None)
    if index_fn is None:
        return None

    arrays = []
    for col in cluster_cols:
        series = df[col].rechunk()
        try:
            arr = series.to_numpy(allow_copy=False)
        except RuntimeError:
            arr = series.to_numpy(allow_copy=True)
        arrays.append(np.ascontiguousarray(arr, dtype=np.uint32))

    try:
        index_arrays = index_fn(arrays)
    except Exception as exc:  # pragma: no cover - fallback preserves behavior
        log.warning("rust hierarchy index computation failed; falling back to Polars: %s", exc)
        return None

    return [
        pl.Series(level_name, np.asarray(index_array, dtype=np.uint32))
        for level_name, index_array in zip(level_sequence, index_arrays)
    ]


__all__ = ["get_cluster_hierarchy", "build_cluster_tables"]
