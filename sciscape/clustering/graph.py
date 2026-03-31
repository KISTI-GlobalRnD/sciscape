"""Graph construction utilities."""

from __future__ import annotations

from typing import Dict, Optional

import igraph as ig
import numpy as np
import polars as pl


def build_graph(
    edges: pl.DataFrame,
    *,
    min_weight: Optional[float] = None,
) -> ig.Graph:
    """Construct an undirected igraph graph from an edge table.

    Parameters
    ----------
    edges : pl.DataFrame
        Must contain columns ``uid1``, ``uid2``, ``rel_sum2``.
    min_weight : float, optional
        If set, drop edges with weight below this threshold before building.
    """
    required_columns = {"uid1", "uid2", "rel_sum2"}
    if missing := required_columns.difference(edges.columns):
        raise ValueError(f"edges missing columns: {sorted(missing)}")

    if min_weight is not None:
        edges = edges.filter(pl.col("rel_sum2") >= min_weight)

    # Build uid → index mapping using Polars categorical encoding (vectorized)
    uids = pl.concat([edges["uid1"], edges["uid2"]]).unique(maintain_order=True)
    uid_list = uids.to_list()
    n_nodes = len(uid_list)

    # Map uid strings → integer indices via Polars join (avoids Python dict loop)
    uid_idx = pl.DataFrame({"uid": uid_list, "_idx": range(n_nodes)})
    src_idx = (
        edges.select(pl.col("uid1").alias("uid"))
        .join(uid_idx, on="uid", how="left")["_idx"]
        .to_numpy()
    )
    tgt_idx = (
        edges.select(pl.col("uid2").alias("uid"))
        .join(uid_idx, on="uid", how="left")["_idx"]
        .to_numpy()
    )
    weights = edges["rel_sum2"].to_numpy()

    # Build igraph from numpy arrays (faster than Python lists)
    edge_array = np.column_stack([src_idx, tgt_idx])
    graph = ig.Graph(n=n_nodes, edges=edge_array.tolist(), directed=False)
    graph.vs["uid"] = uid_list
    graph.es["weight"] = weights.tolist()
    return graph


def giant_component(graph: ig.Graph) -> ig.Graph:
    """Return the weakly connected giant component of the graph."""

    if graph.vcount() == 0:
        return graph
    return graph.clusters(mode="WEAK").giant()


__all__ = ["build_graph", "giant_component"]
