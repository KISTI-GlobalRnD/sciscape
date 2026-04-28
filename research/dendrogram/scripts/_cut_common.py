"""Shared helpers for hybrid dendrogram cut research scripts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sciscape.clustering.auto_gamma import find_gamma
from sciscape.clustering.graph import build_graph, giant_component
from sciscape.clustering.integer_remap import integer_remap_memory
from sciscape.clustering.runner import LeidenRunner
from sciscape.evaluation.stability import compute_quality_report


def membership_map_from_edges(edges: pl.DataFrame, membership: np.ndarray) -> dict[str, int]:
    """Build ``uid -> cluster`` aligned to the edge remap order."""
    _src, _dst, _w, _n_nodes, uids = integer_remap_memory(edges)
    return {uid: int(cluster) for uid, cluster in zip(uids, membership)}


def default_cut_min_size(nano_size_arr: np.ndarray) -> int:
    """Mirror the landscape micro-cut heuristic."""
    return max(
        int(nano_size_arr.sum()) // 20,
        int(nano_size_arr.max()) + 1,
    )


def project_contracted_membership(
    compact_membership: list[int],
    contracted_membership: np.ndarray,
) -> np.ndarray:
    """Map contracted-graph labels back to original graph nodes."""
    return np.asarray(
        [int(contracted_membership[compact_membership[idx]]) for idx in range(len(compact_membership))],
        dtype=np.int64,
    )


def prepare_hybrid_cut_context(
    edge_path: Path,
    *,
    target_pct: float,
    nano_min_size: int,
    seed: int,
    dendrogram_mode: str,
) -> dict[str, Any]:
    """Build the nano partition, contracted graph, and contracted dendrogram."""
    edges = pl.read_parquet(edge_path)
    graph = giant_component(build_graph(edges))
    gcc_uids = graph.vs["uid"]
    edges_gcc = edges.filter(
        pl.col("uid1").is_in(gcc_uids) & pl.col("uid2").is_in(gcc_uids)
    )

    nano = find_gamma(
        edges_gcc,
        target_max_pct=target_pct,
        min_size=nano_min_size,
        postprocess=True,
        seed=seed,
    )

    uid_to_cluster = membership_map_from_edges(edges_gcc, nano.membership)
    graph_membership = [uid_to_cluster[uid] for uid in gcc_uids]

    runner = LeidenRunner(
        graph,
        objective="cpm",
        default_seed=seed,
        default_iterations=10,
    )
    unique_ids = sorted(set(graph_membership))
    id_remap = {old: new for new, old in enumerate(unique_ids)}
    compact_membership = [id_remap[cid] for cid in graph_membership]

    contracted = runner.contract(compact_membership, combine_weights="sum", keep_loops=True)
    n_contracted = contracted.vcount()
    nano_size_arr = np.bincount(
        np.asarray(compact_membership, dtype=np.int32),
        minlength=n_contracted,
    ).astype(np.uint64)

    from sciscape.clustering.dendrogram import build_dendrogram

    linkage = build_dendrogram(
        contracted,
        mode=dendrogram_mode,
        node_sizes=nano_size_arr,
    )
    nano_qr = compute_quality_report(
        edges_gcc,
        np.asarray(graph_membership, dtype=np.int64),
        gamma=nano.gamma,
        target_pct=target_pct,
    )

    return {
        "edges_gcc": edges_gcc,
        "graph": graph,
        "gcc_uids": gcc_uids,
        "nano": nano,
        "nano_qr": nano_qr,
        "graph_membership": graph_membership,
        "compact_membership": compact_membership,
        "contracted": contracted,
        "n_contracted": n_contracted,
        "nano_size_arr": nano_size_arr,
        "linkage": linkage,
    }


def threshold_cut(
    linkage: np.ndarray,
    *,
    threshold: float,
    min_size: int,
    leaf_sizes: np.ndarray | None = None,
) -> dict[str, Any]:
    """Cut a similarity dendrogram at a fixed threshold and evaluate feasibility."""
    if linkage.ndim != 2 or linkage.shape[1] != 4:
        raise ValueError(f"linkage must be (n-1, 4), got {linkage.shape}")

    n_leaves = len(linkage) + 1
    n_internal = len(linkage)
    root_id = n_leaves + n_internal - 1
    if leaf_sizes is None:
        leaf_sizes = np.ones(n_leaves, dtype=np.int64)
    else:
        leaf_sizes = np.asarray(leaf_sizes, dtype=np.int64)

    parent_height = np.zeros(n_leaves + n_internal, dtype=np.float64)
    for row_idx, row in enumerate(linkage):
        left = int(row[0])
        right = int(row[1])
        height = float(row[2])
        parent_height[left] = height
        parent_height[right] = height

    selected_nodes: list[int] = []
    stack = [root_id]
    while stack:
        node_id = stack.pop()
        if node_id < n_leaves:
            selected_nodes.append(node_id)
            continue
        row = linkage[node_id - n_leaves]
        height = float(row[2])
        if height >= threshold:
            selected_nodes.append(node_id)
        else:
            stack.append(int(row[0]))
            stack.append(int(row[1]))

    partition: list[list[int]] = []
    membership = np.full(n_leaves, -1, dtype=np.int64)
    weighted_sizes: list[int] = []
    total_stability = 0.0

    for cluster_id, node_id in enumerate(selected_nodes):
        leaves: list[int] = []
        node_stack = [node_id]
        while node_stack:
            current = node_stack.pop()
            if current < n_leaves:
                leaves.append(current)
            else:
                row = linkage[current - n_leaves]
                node_stack.append(int(row[0]))
                node_stack.append(int(row[1]))
        partition.append(leaves)
        for leaf in leaves:
            membership[leaf] = cluster_id
        cluster_size = int(leaf_sizes[np.asarray(leaves, dtype=np.int64)].sum())
        weighted_sizes.append(cluster_size)
        if node_id >= n_leaves:
            gamma_birth = float(linkage[node_id - n_leaves, 2])
            total_stability += gamma_birth - float(parent_height[node_id])

    feasible = all(size >= min_size for size in weighted_sizes)
    return {
        "threshold": threshold,
        "partition": partition,
        "membership": membership,
        "selected_nodes": selected_nodes,
        "weighted_sizes": weighted_sizes,
        "n_clusters": len(partition),
        "total_stability": total_stability,
        "feasible": feasible,
    }


def select_threshold_candidates(linkage: np.ndarray, *, n_thresholds: int) -> list[float]:
    """Select evenly spaced similarity thresholds from linkage heights."""
    heights = np.unique(np.asarray(linkage[:, 2], dtype=np.float64))
    heights = heights[::-1]  # descending
    if len(heights) <= n_thresholds:
        return heights.tolist()
    indices = np.linspace(0, len(heights) - 1, num=n_thresholds, dtype=int)
    return heights[indices].tolist()
