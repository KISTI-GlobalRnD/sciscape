"""Graph construction utilities."""

from __future__ import annotations

import logging
from typing import Optional

import igraph as ig
import numpy as np
import polars as pl
from scipy import sparse

log = logging.getLogger(__name__)


def build_graph(
    edges: pl.DataFrame,
    *,
    min_weight: Optional[float] = None,
) -> ig.Graph:
    """Construct an undirected igraph graph from an edge table.

    Scalable to 50M+ nodes: uses scipy sparse matrix internally
    to avoid Python list conversion of edges.

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

    if edges.height == 0:
        raise ValueError(
            f"No edges remain after filtering (min_weight={min_weight})"
        )

    # ── Vectorized string → int mapping via Polars join ──────────
    # Note: Categorical encoding is faster for the mapping step alone,
    # but build_graph is dominated by ig.Graph.Weighted_Adjacency (~84%),
    # so the mapping method makes negligible difference here.
    all_uids = pl.concat([edges["uid1"], edges["uid2"]]).unique().sort()
    n_nodes = all_uids.len()
    log.info("build_graph: %d nodes, %d edges", n_nodes, edges.height)

    uid_idx = pl.DataFrame({
        "uid": all_uids,
        "_idx": np.arange(n_nodes, dtype=np.int32),
    })

    src = (
        edges.select(pl.col("uid1").alias("uid"))
        .join(uid_idx, on="uid", how="left")["_idx"]
        .to_numpy()
    )
    tgt = (
        edges.select(pl.col("uid2").alias("uid"))
        .join(uid_idx, on="uid", how="left")["_idx"]
        .to_numpy()
    )
    w = edges["rel_sum2"].to_numpy(allow_copy=False).astype(np.float64)

    # ── Build symmetric sparse adjacency → igraph (no .tolist()) ─
    # Self-loops: add once. Regular edges: add both (i,j) and (j,i).
    is_loop = src == tgt
    reg = ~is_loop

    row = np.concatenate([src[reg], tgt[reg], src[is_loop]])
    col = np.concatenate([tgt[reg], src[reg], tgt[is_loop]])
    data = np.concatenate([w[reg], w[reg], w[is_loop]])

    # COO → CSR sums duplicate entries (multi-edges become summed weight)
    adj = sparse.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes)).tocsr()

    graph = ig.Graph.Weighted_Adjacency(adj, mode="undirected", attr="weight", loops=True)
    graph.vs["uid"] = all_uids.to_list()
    return graph


def giant_component(graph: ig.Graph) -> ig.Graph:
    """Return the weakly connected giant component of the graph."""

    if graph.vcount() == 0:
        return graph
    return graph.connected_components(mode="WEAK").giant()


__all__ = ["build_graph", "giant_component"]
