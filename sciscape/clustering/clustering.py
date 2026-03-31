"""Leiden community detection helpers."""

from __future__ import annotations

from typing import Dict, Mapping

import igraph as ig
import leidenalg as la
import polars as pl

from .config import LeidenConfig
from .runner import LeidenRunner


def run_leiden_levels(graph: ig.Graph, config: LeidenConfig) -> Dict[str, la.VertexPartition]:
    """Run Leiden clustering across multiple resolution levels."""

    if not config.resolutions:
        raise ValueError("LeidenConfig.resolutions is required for run_leiden_levels")

    partitions: Dict[str, la.VertexPartition] = {}
    runner = LeidenRunner(
        graph,
        objective=config.objective,
        default_iterations=config.leiden_iterations,
        default_seed=config.seed,
    )
    initial_membership = None
    for level, gamma in config.resolutions.items():
        result = runner.run(
            gamma,
            seed=config.seed,
            n_iterations=config.leiden_iterations,
            initial_membership=initial_membership,
        )
        partitions[level] = result.partition
        initial_membership = result.membership
    return partitions


def partitions_to_polars(partitions: Mapping[str, la.VertexPartition]) -> pl.DataFrame:
    """Convert Leiden partitions to a wide Polars DataFrame."""

    if not partitions:
        raise ValueError("No partitions provided")

    membership = {
        f"cluster_{level}": part.membership for level, part in partitions.items()
    }

    df = pl.DataFrame(membership)
    df = df.with_columns(pl.arange(0, df.height).alias("_vertex_index"))
    return df


def attach_uids(df: pl.DataFrame, graph: ig.Graph) -> pl.DataFrame:
    """Add uid labels from the graph to the partition table."""

    if "uid" not in graph.vs.attributes():
        raise ValueError("Graph vertices must include 'uid' attribute")

    clusters = [col for col in df.columns if col.startswith("cluster_")]
    augmented = df.with_columns(pl.Series("uid", graph.vs["uid"]))
    return augmented.select("uid", *clusters)


__all__ = [
    "run_leiden_levels",
    "partitions_to_polars",
    "attach_uids",
]
