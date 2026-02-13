"""Graph construction utilities."""

from __future__ import annotations

from typing import Dict

import igraph as ig
import polars as pl


def build_graph(edges: pl.DataFrame) -> ig.Graph:
    """Construct an undirected igraph graph from an edge table."""

    required_columns = {"uid1", "uid2", "rel_sum2"}
    if missing := required_columns.difference(edges.columns):
        raise ValueError(f"edges missing columns: {sorted(missing)}")

    uids = pl.concat([edges["uid1"], edges["uid2"]]).unique(maintain_order=True)
    uid_list = uids.to_list()
    uid_to_index: Dict[str, int] = {uid: idx for idx, uid in enumerate(uid_list)}

    sources = [uid_to_index[uid] for uid in edges["uid1"].to_list()]
    targets = [uid_to_index[uid] for uid in edges["uid2"].to_list()]
    weights = edges["rel_sum2"].to_list()

    graph = ig.Graph()
    graph.add_vertices(len(uid_list))
    graph.vs["uid"] = uid_list
    graph.add_edges(zip(sources, targets))
    graph.es["weight"] = weights
    return graph


def giant_component(graph: ig.Graph) -> ig.Graph:
    """Return the weakly connected giant component of the graph."""

    if graph.vcount() == 0:
        return graph
    return graph.clusters(mode="WEAK").giant()


__all__ = ["build_graph", "giant_component"]
