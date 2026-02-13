"""Post-processing helpers for Leiden cluster assignments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import igraph as ig


@dataclass(frozen=True)
class MergeAction:
    """Record describing a merge of a tiny cluster."""

    source: int
    target: int
    size: int
    weight: float
    edge_weight: float


@dataclass(frozen=True)
class PostprocessResult:
    """Result returned by `merge_small_clusters`."""

    membership: List[int]
    merges: List[MergeAction]
    cluster_sizes: Dict[int, int]
    cluster_weights: Dict[int, float]


def _compute_cluster_stats(
    membership: Sequence[int],
    node_weights: Sequence[float] | None,
) -> tuple[Dict[int, int], Dict[int, float]]:
    sizes: Dict[int, int] = {}
    weights: Dict[int, float] = {}
    for idx, cluster in enumerate(membership):
        sizes[cluster] = sizes.get(cluster, 0) + 1
        if node_weights is not None:
            weights[cluster] = weights.get(cluster, 0.0) + float(node_weights[idx])
        else:
            weights[cluster] = weights.get(cluster, 0.0) + 1.0
    return sizes, weights


def _build_cluster_adjacency(
    graph: ig.Graph,
    membership: Sequence[int],
) -> Dict[int, Dict[int, float]]:
    adjacency: Dict[int, Dict[int, float]] = {}
    weights = graph.es["weight"] if "weight" in graph.es.attributes() else None
    for edge_index, (u, v) in enumerate(graph.get_edgelist()):
        cu = membership[u]
        cv = membership[v]
        if cu == cv:
            continue
        weight = weights[edge_index] if weights is not None else 1.0
        adjacency.setdefault(cu, {})[cv] = adjacency.get(cu, {}).get(cv, 0.0) + float(weight)
        adjacency.setdefault(cv, {})[cu] = adjacency.get(cv, {}).get(cu, 0.0) + float(weight)
    return adjacency


def merge_small_clusters(
    graph: ig.Graph,
    membership: Sequence[int],
    *,
    min_size: int | None = None,
    min_weight: float | None = None,
    node_weights: Sequence[float] | None = None,
    max_passes: int = 1,
) -> PostprocessResult:
    """Merge clusters that do not satisfy the size/weight thresholds."""

    if min_size is None and min_weight is None:
        sizes, weights = _compute_cluster_stats(membership, node_weights)
        return PostprocessResult(list(membership), [], sizes, weights)

    sizes, weights = _compute_cluster_stats(membership, node_weights)
    membership_list = list(membership)
    adjacency = _build_cluster_adjacency(graph, membership_list)
    merges: List[MergeAction] = []

    def threshold_ok(cluster: int) -> bool:
        size_ok = min_size is None or sizes.get(cluster, 0) >= min_size
        weight_ok = min_weight is None or weights.get(cluster, 0.0) >= min_weight
        return size_ok and weight_ok

    passes = 0
    while passes < max_passes:
        passes += 1
        small_clusters = [
            cid
            for cid in sorted(sizes, key=lambda c: (weights.get(c, 0.0), sizes.get(c, 0)))
            if not threshold_ok(cid) and sizes.get(cid, 0) > 0
        ]
        if not small_clusters:
            break

        merged_any = False
        for cluster in small_clusters:
            neighbours = adjacency.get(cluster, {})
            if not neighbours:
                continue
            target = max(neighbours.items(), key=lambda item: (item[1], sizes.get(item[0], 0)))[0]
            if target == cluster:
                continue

            edge_weight = neighbours[target]
            merge_size = sizes.get(cluster, 0)
            merge_weight = weights.get(cluster, 0.0)

            for idx, label in enumerate(membership_list):
                if label == cluster:
                    membership_list[idx] = target

            sizes[target] = sizes.get(target, 0) + merge_size
            weights[target] = weights.get(target, 0.0) + merge_weight
            sizes[cluster] = 0
            weights[cluster] = 0.0

            merges.append(
                MergeAction(
                    source=cluster,
                    target=target,
                    size=merge_size,
                    weight=merge_weight,
                    edge_weight=edge_weight,
                )
            )
            merged_any = True

            # Update adjacency: move all edges incident to cluster into target.
            if target in adjacency:
                adjacency[target].pop(cluster, None)

            for neighbour, weight in neighbours.items():
                if neighbour == target:
                    continue
                adjacency.setdefault(target, {})[neighbour] = (
                    adjacency.get(target, {}).get(neighbour, 0.0) + weight
                )
                adjacency.setdefault(neighbour, {})[target] = (
                    adjacency.get(neighbour, {}).get(target, 0.0) + weight
                )
                adjacency[neighbour].pop(cluster, None)
            adjacency.pop(cluster, None)

        if not merged_any:
            break

    # Renumber clusters to keep labels compact.
    mapping: Dict[int, int] = {}
    next_label = 0
    for label in membership_list:
        if label not in mapping:
            mapping[label] = next_label
            next_label += 1
    membership_compact = [mapping[label] for label in membership_list]

    remapped_sizes: Dict[int, int] = {}
    remapped_weights: Dict[int, float] = {}
    for old_label, new_label in mapping.items():
        remapped_sizes[new_label] = sizes.get(old_label, 0)
        remapped_weights[new_label] = weights.get(old_label, 0.0)

    return PostprocessResult(
        membership=membership_compact,
        merges=merges,
        cluster_sizes=remapped_sizes,
        cluster_weights=remapped_weights,
    )


__all__ = ["merge_small_clusters", "PostprocessResult", "MergeAction"]
