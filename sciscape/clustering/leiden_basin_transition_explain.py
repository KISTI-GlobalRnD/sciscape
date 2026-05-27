"""Reusable explanation helpers for Leiden basin transition diagnostics."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import pandas as pd

from sciscape.clustering.leiden_basin_profile import changed_support_nodes
from sciscape.clustering.leiden_basin_profile import endpoint_distance


def unique_sorted_u32(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    if arr.size == 0:
        return np.asarray([], dtype=np.uint32)
    return np.asarray(sorted(set(int(value) for value in arr)), dtype=np.uint32)


def node_csv(nodes: Any) -> str:
    return ",".join(str(int(node)) for node in unique_sorted_u32(nodes))


def _node_mask(node_count: int, nodes: Any) -> np.ndarray:
    mask = np.zeros(int(node_count), dtype=np.bool_)
    arr = unique_sorted_u32(nodes).astype(np.int64)
    if arr.size:
        mask[arr] = True
    return mask


def hop_distance_to_sources(
    *,
    src: np.ndarray,
    dst: np.ndarray,
    node_count: int,
    source_nodes: np.ndarray,
    max_hops: int | None = None,
) -> np.ndarray:
    """Return unweighted graph-hop distance to the nearest source node.

    Unreached nodes are marked with ``-1``.  The helper is intentionally simple
    because it is used for small diagnostic slices, not production traversal.
    """
    count = int(node_count)
    distances = np.full(count, -1, dtype=np.int32)
    sources = unique_sorted_u32(source_nodes).astype(np.int64)
    if count <= 0 or sources.size == 0:
        return distances

    adjacency: list[list[int]] = [[] for _ in range(count)]
    for left, right in zip(
        np.asarray(src, dtype=np.int64),
        np.asarray(dst, dtype=np.int64),
        strict=False,
    ):
        if 0 <= int(left) < count and 0 <= int(right) < count:
            adjacency[int(left)].append(int(right))
            adjacency[int(right)].append(int(left))

    queue: deque[int] = deque()
    for node in sources:
        idx = int(node)
        if 0 <= idx < count and distances[idx] < 0:
            distances[idx] = 0
            queue.append(idx)

    while queue:
        node = queue.popleft()
        next_distance = int(distances[node]) + 1
        if max_hops is not None and next_distance > int(max_hops):
            continue
        for neighbor in adjacency[node]:
            if distances[neighbor] >= 0:
                continue
            distances[neighbor] = next_distance
            queue.append(neighbor)
    return distances


def weighted_pull_to_nodes(
    *,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    node_count: int,
    target_nodes: np.ndarray,
) -> np.ndarray:
    """Return incident edge weight from every node to ``target_nodes``."""
    scores = np.zeros(int(node_count), dtype=np.float64)
    targets = unique_sorted_u32(target_nodes).astype(np.int64)
    if targets.size == 0:
        return scores
    mask = np.zeros(int(node_count), dtype=np.bool_)
    mask[targets] = True
    left = np.asarray(src, dtype=np.int64)
    right = np.asarray(dst, dtype=np.int64)
    weights = np.asarray(weight, dtype=np.float64)
    left_hit = mask[left]
    right_hit = mask[right]
    np.add.at(scores, right[left_hit], weights[left_hit])
    np.add.at(scores, left[right_hit], weights[right_hit])
    scores[targets] = 0.0
    return scores


def membership_change_summary(
    *,
    reference_membership: np.ndarray,
    membership: np.ndarray,
    sketch_nodes: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Summarize exact-label and label-invariant changes between partitions."""
    reference = np.asarray(reference_membership, dtype=np.uint64)
    current = np.asarray(membership, dtype=np.uint64)
    if reference.shape != current.shape:
        raise ValueError("memberships must have the same shape")
    exact = unique_sorted_u32(np.flatnonzero(reference != current))
    aligned = changed_support_nodes(reference, current)
    exact_only = np.setdiff1d(exact, aligned, assume_unique=False)
    summary: dict[str, float | int] = {
        "exact_changed_node_count": int(exact.size),
        "aligned_changed_node_count": int(aligned.size),
        "exact_only_changed_node_count": int(exact_only.size),
        "exact_to_aligned_ratio": float(exact.size) / float(max(1, int(aligned.size))),
    }
    if sketch_nodes is not None:
        summary["endpoint_distance"] = float(
            endpoint_distance(reference, current, np.asarray(sketch_nodes, dtype=np.uint32))
        )
    return summary


def build_change_node_rows(
    *,
    reference_membership: np.ndarray,
    membership: np.ndarray,
    baseline_membership: np.ndarray,
    vanilla_membership: np.ndarray,
    candidate_membership: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    target_nodes: np.ndarray,
    context_nodes: np.ndarray,
    bundle_nodes: np.ndarray,
    source_action_nodes: np.ndarray,
    source_mutable_nodes: np.ndarray,
    include_nodes: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build per-node rows for a diagnostic set around a transition."""
    reference = np.asarray(reference_membership, dtype=np.uint64)
    current = np.asarray(membership, dtype=np.uint64)
    node_count = int(reference.size)
    exact = unique_sorted_u32(np.flatnonzero(reference != current))
    aligned = changed_support_nodes(reference, current)
    if include_nodes is None:
        nodes = unique_sorted_u32(np.concatenate([aligned, bundle_nodes]))
    else:
        nodes = unique_sorted_u32(include_nodes)

    target = _node_mask(node_count, target_nodes)
    context = _node_mask(node_count, context_nodes)
    bundle = _node_mask(node_count, bundle_nodes)
    source_action = _node_mask(node_count, source_action_nodes)
    source_mutable = _node_mask(node_count, source_mutable_nodes)
    exact_mask = _node_mask(node_count, exact)
    aligned_mask = _node_mask(node_count, aligned)
    hop_target = hop_distance_to_sources(
        src=src,
        dst=dst,
        node_count=node_count,
        source_nodes=target_nodes,
    )
    hop_bundle = hop_distance_to_sources(
        src=src,
        dst=dst,
        node_count=node_count,
        source_nodes=bundle_nodes,
    )
    pull_target = weighted_pull_to_nodes(
        src=src,
        dst=dst,
        weight=weight,
        node_count=node_count,
        target_nodes=target_nodes,
    )
    pull_context = weighted_pull_to_nodes(
        src=src,
        dst=dst,
        weight=weight,
        node_count=node_count,
        target_nodes=context_nodes,
    )
    pull_bundle = weighted_pull_to_nodes(
        src=src,
        dst=dst,
        weight=weight,
        node_count=node_count,
        target_nodes=bundle_nodes,
    )
    rows: list[dict[str, Any]] = []
    baseline = np.asarray(baseline_membership, dtype=np.uint64)
    vanilla = np.asarray(vanilla_membership, dtype=np.uint64)
    candidate = np.asarray(candidate_membership, dtype=np.uint64)
    for node_value in nodes.astype(np.int64):
        node = int(node_value)
        rows.append(
            {
                "node": node,
                "in_selected_target": bool(target[node]),
                "in_context": bool(context[node]),
                "in_bundle": bool(bundle[node]),
                "in_source_action": bool(source_action[node]),
                "in_source_mutable": bool(source_mutable[node]),
                "exact_label_changed": bool(exact_mask[node]),
                "aligned_partition_changed": bool(aligned_mask[node]),
                "hop_to_selected_target": int(hop_target[node]),
                "hop_to_bundle": int(hop_bundle[node]),
                "pull_to_selected_target": float(pull_target[node]),
                "pull_to_context": float(pull_context[node]),
                "pull_to_bundle": float(pull_bundle[node]),
                "baseline_label": int(baseline[node]),
                "vanilla_label": int(vanilla[node]),
                "candidate_label": int(candidate[node]),
                "reference_label": int(reference[node]),
                "result_label": int(current[node]),
            }
        )
    return pd.DataFrame(rows)


def build_change_shell_rows(
    *,
    reference_membership: np.ndarray,
    membership: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    target_nodes: np.ndarray,
    bundle_nodes: np.ndarray,
) -> pd.DataFrame:
    """Aggregate exact and label-invariant changes by hop distance."""
    reference = np.asarray(reference_membership, dtype=np.uint64)
    current = np.asarray(membership, dtype=np.uint64)
    node_count = int(reference.size)
    exact = unique_sorted_u32(np.flatnonzero(reference != current))
    aligned = changed_support_nodes(reference, current)
    exact_only = np.setdiff1d(exact, aligned, assume_unique=False).astype(
        np.uint32,
        copy=False,
    )
    hop_target = hop_distance_to_sources(
        src=src,
        dst=dst,
        node_count=node_count,
        source_nodes=target_nodes,
    )
    hop_bundle = hop_distance_to_sources(
        src=src,
        dst=dst,
        node_count=node_count,
        source_nodes=bundle_nodes,
    )

    rows: list[dict[str, Any]] = []
    for change_kind, nodes in (
        ("exact_label_changed", exact),
        ("aligned_partition_changed", aligned),
        ("exact_only_label_changed", exact_only),
    ):
        for node in nodes.astype(np.int64):
            rows.append(
                {
                    "change_kind": change_kind,
                    "hop_to_selected_target": int(hop_target[int(node)]),
                    "hop_to_bundle": int(hop_bundle[int(node)]),
                    "node_count": 1,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "change_kind",
                "hop_to_selected_target",
                "hop_to_bundle",
                "node_count",
            ]
        )
    return (
        pd.DataFrame(rows)
        .groupby(
            ["change_kind", "hop_to_selected_target", "hop_to_bundle"],
            sort=True,
            as_index=False,
        )
        ["node_count"]
        .sum()
    )


def build_label_transition_rows(
    *,
    reference_membership: np.ndarray,
    membership: np.ndarray,
    nodes: np.ndarray,
    target_nodes: np.ndarray,
    context_nodes: np.ndarray,
    bundle_nodes: np.ndarray,
) -> pd.DataFrame:
    """Aggregate reference/result label transitions for a selected node set."""
    reference = np.asarray(reference_membership, dtype=np.uint64)
    current = np.asarray(membership, dtype=np.uint64)
    selected = unique_sorted_u32(nodes)
    if selected.size == 0:
        return pd.DataFrame()
    node_count = int(reference.size)
    target = _node_mask(node_count, target_nodes)
    context = _node_mask(node_count, context_nodes)
    bundle = _node_mask(node_count, bundle_nodes)
    rows: list[dict[str, Any]] = []
    for node_value in selected.astype(np.int64):
        node = int(node_value)
        rows.append(
            {
                "reference_label": int(reference[node]),
                "result_label": int(current[node]),
                "node_count": 1,
                "selected_target_node_count": int(target[node]),
                "context_node_count": int(context[node]),
                "bundle_node_count": int(bundle[node]),
            }
        )
    return (
        pd.DataFrame(rows)
        .groupby(["reference_label", "result_label"], sort=True, as_index=False)
        .sum()
        .sort_values(["node_count", "reference_label", "result_label"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
