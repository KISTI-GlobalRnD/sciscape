"""Post-processing helpers for Leiden cluster assignments."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
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


def _component_nodes(
    clusters: Sequence[int],
    adjacency: Dict[int, Dict[int, float]],
) -> List[List[int]]:
    remaining = set(clusters)
    components: List[List[int]] = []
    while remaining:
        start = min(remaining)
        stack = [start]
        component: List[int] = []
        while stack:
            current = stack.pop()
            if current not in remaining:
                continue
            remaining.remove(current)
            component.append(current)
            for neighbour in adjacency.get(current, {}):
                if neighbour in remaining:
                    stack.append(neighbour)
        components.append(component)
    return components


def _closest_anchor_assignment(
    adjacency: Dict[int, Dict[int, float]],
    anchors: Sequence[int],
    cluster_sizes: Dict[int, int],
    cluster_weights: Dict[int, float],
) -> Dict[int, int]:
    if not anchors:
        return {}

    # Multi-source Dijkstra over cluster graph.
    # Edge cost is inverse weight to prioritize stronger inter-cluster ties.
    best: Dict[int, tuple[float, float, float, int]] = {}
    heap: List[tuple[float, float, float, int, int]] = []

    for anchor in sorted(set(anchors)):
        key = (
            0.0,
            -float(cluster_weights.get(anchor, 0.0)),
            -float(cluster_sizes.get(anchor, 0)),
            int(anchor),
        )
        best[anchor] = key
        heapq.heappush(heap, (*key, anchor))

    while heap:
        dist, neg_weight, neg_size, anchor, node = heapq.heappop(heap)
        key = (dist, neg_weight, neg_size, anchor)
        if best.get(node) != key:
            continue
        for neighbour, edge_weight in adjacency.get(node, {}).items():
            cost = 1.0 / max(float(edge_weight), 1e-12)
            candidate = (dist + cost, neg_weight, neg_size, anchor)
            previous = best.get(neighbour)
            if previous is None or candidate < previous:
                best[neighbour] = candidate
                heapq.heappush(heap, (*candidate, neighbour))

    return {cluster: key[3] for cluster, key in best.items()}


def _pick_fallback_anchor(
    component: Sequence[int],
    cluster_sizes: Dict[int, int],
    cluster_weights: Dict[int, float],
) -> int:
    return max(
        component,
        key=lambda cluster: (
            float(cluster_weights.get(cluster, 0.0)),
            int(cluster_sizes.get(cluster, 0)),
            -int(cluster),
        ),
    )


def merge_small_clusters(
    graph: ig.Graph,
    membership: Sequence[int],
    *,
    min_size: int | None = None,
    min_weight: float | None = None,
    node_weights: Sequence[float] | None = None,
    max_passes: int = 1,
) -> PostprocessResult:
    """Merge clusters that do not satisfy the size/weight thresholds.

    Targets are selected on the cluster graph first, then memberships are remapped
    in one pass. This avoids per-merge full scans of node-level membership vectors.
    """

    if min_size is None and min_weight is None:
        sizes, weights = _compute_cluster_stats(membership, node_weights)
        return PostprocessResult(list(membership), [], sizes, weights)

    sizes, weights = _compute_cluster_stats(membership, node_weights)
    membership_list = list(membership)
    if not membership_list:
        return PostprocessResult([], [], {}, {})
    merges: List[MergeAction] = []

    def threshold_ok(cluster: int) -> bool:
        size_ok = min_size is None or sizes.get(cluster, 0) >= min_size
        weight_ok = min_weight is None or weights.get(cluster, 0.0) >= min_weight
        return size_ok and weight_ok

    for _ in range(max(1, int(max_passes))):
        adjacency = _build_cluster_adjacency(graph, membership_list)
        active_clusters = sorted(cluster for cluster, size in sizes.items() if size > 0)
        if not active_clusters:
            break

        anchors = {cluster for cluster in active_clusters if threshold_ok(cluster)}
        forced_anchors: set[int] = set()
        for component in _component_nodes(active_clusters, adjacency):
            if not any(cluster in anchors for cluster in component):
                fallback = _pick_fallback_anchor(component, sizes, weights)
                anchors.add(fallback)
                forced_anchors.add(fallback)

        small_clusters = [
            cluster
            for cluster in sorted(
                active_clusters,
                key=lambda cid: (float(weights.get(cid, 0.0)), int(sizes.get(cid, 0)), int(cid)),
            )
            if not threshold_ok(cluster) and cluster not in forced_anchors
        ]
        if not small_clusters:
            break

        nearest_anchor = _closest_anchor_assignment(
            adjacency,
            sorted(anchors),
            cluster_sizes=sizes,
            cluster_weights=weights,
        )
        small_set = set(small_clusters)
        merge_targets: Dict[int, int] = {}

        for cluster in small_clusters:
            target = nearest_anchor.get(cluster)
            if target is None or target == cluster:
                neighbours = adjacency.get(cluster, {})
                if not neighbours:
                    continue
                eligible = {nbr: edge for nbr, edge in neighbours.items() if nbr not in small_set}
                target_pool = eligible if eligible else neighbours
                target = max(
                    target_pool.items(),
                    key=lambda item: (
                        float(item[1]),
                        int(sizes.get(item[0], 0)),
                        float(weights.get(item[0], 0.0)),
                        -int(item[0]),
                    ),
                )[0]
            if target == cluster:
                continue
            merge_targets[cluster] = target

        if not merge_targets:
            break

        for source, target in merge_targets.items():
            merges.append(
                MergeAction(
                    source=source,
                    target=target,
                    size=int(sizes.get(source, 0)),
                    weight=float(weights.get(source, 0.0)),
                    edge_weight=float(adjacency.get(source, {}).get(target, 0.0)),
                )
            )

        membership_list = [merge_targets.get(cluster, cluster) for cluster in membership_list]
        sizes, weights = _compute_cluster_stats(membership_list, node_weights)

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
