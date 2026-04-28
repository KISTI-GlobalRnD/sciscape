"""Post-processing helpers for Leiden cluster assignments."""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
import heapq
from typing import TYPE_CHECKING, Dict, List, Sequence

import igraph as ig
import numpy as np

if TYPE_CHECKING:
    from .runner import LeidenRunResult, LeidenRunner

log = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class SingletonResolutionResult:
    """Result from :func:`resolve_singletons`."""

    membership: List[int]
    n_singletons_initial: int
    n_resolved: int
    n_unresolvable: int
    resolutions_used: List[float]


@dataclass(frozen=True)
class SmallClusterResolutionResult:
    """Result from :func:`resolve_small_clusters`."""

    membership: List[int]
    n_small_initial: int
    n_small_nodes_initial: int
    n_clusters_resolved: int
    n_nodes_resolved: int
    n_clusters_unresolvable: int
    n_nodes_unresolvable: int
    resolutions_used: List[float]


@dataclass(frozen=True)
class GammaSearchResult:
    """Result from :func:`gamma_search`."""

    best_gamma: float
    membership: List[int]
    n_large: int
    n_evals: int


@dataclass(frozen=True)
class LargeClusterSplitResult:
    """Result from :func:`split_large_clusters`."""

    membership: List[int]
    n_clusters_split: int
    n_nodes_affected: int
    n_new_clusters_created: int
    split_gammas: Dict[int, float]


@dataclass(frozen=True)
class RefinementResult:
    """Result from :func:`refine_clusters`."""

    membership: List[int]
    n_rounds: int
    split_results: List[LargeClusterSplitResult]
    merge_results: List[SmallClusterResolutionResult]


def _compute_cluster_stats(
    membership: Sequence[int],
    node_weights: Sequence[float] | None,
) -> tuple[Dict[int, int], Dict[int, float]]:
    mem = np.asarray(membership)
    size_arr = np.bincount(mem)
    sizes = {int(c): int(size_arr[c]) for c in range(len(size_arr)) if size_arr[c] > 0}
    if node_weights is not None:
        w = np.asarray(node_weights, dtype=np.float64)
        weight_arr = np.bincount(mem, weights=w, minlength=len(size_arr))
    else:
        weight_arr = size_arr.astype(np.float64)
    weights = {int(c): float(weight_arr[c]) for c in sizes}
    return sizes, weights


def _build_cluster_adjacency(
    graph: ig.Graph,
    membership: Sequence[int],
) -> Dict[int, Dict[int, float]]:
    mem = np.asarray(membership)
    edges = np.array(graph.get_edgelist())  # (|E|, 2)
    if edges.size == 0:
        return {}
    w = np.array(graph.es["weight"], dtype=np.float64) if "weight" in graph.es.attributes() \
        else np.ones(edges.shape[0], dtype=np.float64)

    cu = mem[edges[:, 0]]
    cv = mem[edges[:, 1]]
    inter_mask = cu != cv
    cu, cv, w = cu[inter_mask], cv[inter_mask], w[inter_mask]

    # Build sparse cluster adjacency and convert to nested dict
    if len(cu) == 0:
        return {}
    n_clusters = int(max(cu.max(), cv.max())) + 1
    from scipy.sparse import coo_matrix
    # Symmetric: add both directions
    rows = np.concatenate([cu, cv])
    cols = np.concatenate([cv, cu])
    data = np.concatenate([w, w])
    adj = coo_matrix((data, (rows, cols)), shape=(n_clusters, n_clusters)).tocsr()

    adjacency: Dict[int, Dict[int, float]] = {}
    csr = adj.tocsr()
    for i in range(n_clusters):
        start, end = csr.indptr[i], csr.indptr[i + 1]
        if start == end:
            continue
        adjacency[i] = dict(zip(csr.indices[start:end], csr.data[start:end]))
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

        mem_arr = np.array(membership_list)
        remap = np.arange(mem_arr.max() + 1)
        for src, tgt in merge_targets.items():
            remap[src] = tgt
        membership_list = remap[mem_arr].tolist()
        sizes, weights = _compute_cluster_stats(membership_list, node_weights)

    # Renumber clusters to keep labels compact (vectorized).
    mem_arr = np.array(membership_list)
    # unique in order of first appearance
    _, first_idx, inverse = np.unique(mem_arr, return_index=True, return_inverse=True)
    # Re-label by order of first appearance (stable)
    appearance_order = np.argsort(first_idx)
    compact_remap = np.empty_like(appearance_order)
    compact_remap[appearance_order] = np.arange(len(appearance_order))
    membership_compact = compact_remap[inverse].tolist()

    old_labels = np.unique(mem_arr)
    mapping = {int(old): int(compact_remap[i]) for i, old in enumerate(old_labels)}
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


def resolve_singletons(
    runner: LeidenRunner,
    membership: Sequence[int],
    gamma: float,
    *,
    coarsening_factors: Sequence[float] = (0.1, 0.01),
) -> SingletonResolutionResult:
    """Assign singletons to clusters via hierarchical CPM inheritance.

    Runs Leiden at progressively coarser γ levels.  When a singleton joins a
    cluster at coarser γ that also contains non-singleton fine-γ members, it
    inherits the dominant fine-γ cluster from that coarse group.

    This is parameter-free in the sense that CPM determines the assignment at
    each level; the *coarsening_factors* merely control how many additional
    Leiden runs to attempt (computational budget, not quality parameter).

    Parameters
    ----------
    runner : LeidenRunner
        Runner bound to the same graph used for the original clustering.
    membership : sequence of int
        Fine-γ membership vector (may contain singletons).
    gamma : float
        Resolution used for the fine-γ clustering.
    coarsening_factors : sequence of float
        Multiplicative factors applied to *gamma* for each coarser level.
        Default ``(0.1, 0.01)`` yields 2 additional Leiden runs and resolves
        ~99 % of singletons based on cross-field validation.
    """
    membership = list(membership)
    sizes = Counter(membership)

    # Identify singleton nodes (nodes whose cluster has exactly one member).
    singletons = {i for i, c in enumerate(membership) if sizes[c] == 1}
    n_initial = len(singletons)

    if not singletons:
        return SingletonResolutionResult(
            membership=membership,
            n_singletons_initial=0,
            n_resolved=0,
            n_unresolvable=0,
            resolutions_used=[],
        )

    log.info("  Resolving %d singletons via hierarchical CPM...", n_initial)
    resolutions_used: List[float] = []

    for factor in coarsening_factors:
        if not singletons:
            break

        coarse_gamma = gamma * factor
        resolutions_used.append(coarse_gamma)
        coarse_result = runner.run(coarse_gamma)
        coarse_mem = coarse_result.membership

        # Map: coarse_cluster → {fine_cluster: count} for non-singleton nodes.
        coarse_to_fine: Dict[int, Dict[int, int]] = {}
        for node_id in range(len(membership)):
            if node_id in singletons:
                continue
            cc = coarse_mem[node_id]
            fc = membership[node_id]
            bucket = coarse_to_fine.get(cc)
            if bucket is None:
                bucket = {}
                coarse_to_fine[cc] = bucket
            bucket[fc] = bucket.get(fc, 0) + 1

        # Assign each singleton to the dominant fine cluster in its coarse group.
        resolved: set[int] = set()
        for node_id in singletons:
            fine_counts = coarse_to_fine.get(coarse_mem[node_id])
            if not fine_counts:
                continue  # coarse cluster has only singletons — try next level
            best_fine = max(fine_counts, key=fine_counts.get)  # type: ignore[arg-type]
            membership[node_id] = best_fine
            resolved.add(node_id)

        singletons -= resolved
        log.info("    γ=%.2e → %d/%d resolved", coarse_gamma,
                 len(resolved), len(resolved) + len(singletons))

    n_resolved = n_initial - len(singletons)
    log.info("  Singletons: %d initial → %d resolved, %d unresolvable (%.3f%%)",
             n_initial, n_resolved, len(singletons),
             len(singletons) / len(membership) * 100 if membership else 0)

    return SingletonResolutionResult(
        membership=membership,
        n_singletons_initial=n_initial,
        n_resolved=n_resolved,
        n_unresolvable=len(singletons),
        resolutions_used=resolutions_used,
    )


def resolve_small_clusters(
    runner: LeidenRunner,
    membership: Sequence[int],
    gamma: float,
    *,
    min_size: int = 1000,
    min_factor: float = 0.0001,
    coarse_iterations: int | None = 2,
) -> SmallClusterResolutionResult:
    """Resolve small clusters via convergence-based adaptive CPM coarsening.

    Contracts the graph to a cluster-level supernode graph (one node per
    cluster) and runs Leiden at progressively coarser γ **on the contracted
    graph**.  This is orders of magnitude faster than re-running on the full
    graph because the contracted graph has only ``n_clusters`` nodes.

    Single-parameter design: *min_size* (typically ``min_docs``) drives both
    what counts as "small" and when to stop coarsening.

    The algorithm halves the γ factor each iteration (0.5, 0.25, 0.125, ...)
    and stops when:
    - all clusters have size ≥ *min_size*, or
    - two consecutive rounds resolve zero clusters (convergence), or
    - the factor drops below *min_factor* (safety floor).

    Parameters
    ----------
    runner : LeidenRunner
        Runner bound to the same graph used for the original clustering.
    membership : sequence of int
        Fine-γ membership vector.
    gamma : float
        Resolution used for the fine-γ clustering.
    min_size : int
        Target minimum cluster size.  Clusters smaller than this are resolved
        into neighbouring large clusters via coarser γ.
    min_factor : float
        Safety floor: stop coarsening if factor drops below this value.
        Default ``0.0001`` means γ is never reduced by more than 10000×.
    coarse_iterations : int or None
        Number of Leiden iterations for coarsening runs.  Default ``2``.
    """
    membership = list(membership)
    sizes = Counter(membership)

    # Partition clusters into large (anchors) and small (to resolve).
    large_cids: set[int] = set()
    small_cids: set[int] = set()
    for cid, sz in sizes.items():
        if sz >= min_size:
            large_cids.add(cid)
        else:
            small_cids.add(cid)

    # Pre-compute node lists for each small cluster.
    small_cluster_nodes: Dict[int, List[int]] = {cid: [] for cid in small_cids}
    for node_id, cid in enumerate(membership):
        if cid in small_cluster_nodes:
            small_cluster_nodes[cid].append(node_id)

    n_small_initial = len(small_cids)
    n_small_nodes_initial = sum(len(v) for v in small_cluster_nodes.values())

    if n_small_initial == 0:
        return SmallClusterResolutionResult(
            membership=membership,
            n_small_initial=0,
            n_small_nodes_initial=0,
            n_clusters_resolved=0,
            n_nodes_resolved=0,
            n_clusters_unresolvable=0,
            n_nodes_unresolvable=0,
            resolutions_used=[],
        )

    log.info("  Resolving %d small clusters (%d nodes, size < %d) via "
             "contracted-graph CPM coarsening...",
             n_small_initial, n_small_nodes_initial, min_size)

    # ── Contract: each cluster → supernode ──────────────────────────
    # Renumber cluster IDs to 0..n_clusters-1 for contraction.
    unique_cids = sorted(sizes.keys())
    # Vectorized cid → supernode remap
    max_cid = max(unique_cids) if unique_cids else 0
    cid_to_super_arr = np.empty(max_cid + 1, dtype=np.intp)
    super_to_cid_arr = np.array(unique_cids, dtype=np.intp)
    for i, c in enumerate(unique_cids):
        cid_to_super_arr[c] = i
    cid_to_super: Dict[int, int] = {c: int(cid_to_super_arr[c]) for c in unique_cids}
    super_to_cid: Dict[int, int] = {i: int(super_to_cid_arr[i]) for i in range(len(unique_cids))}
    contracted_mem = cid_to_super_arr[np.array(membership)].tolist()

    contracted = runner.contract(contracted_mem, combine_weights="sum",
                                 keep_loops=True)
    n_supers = len(unique_cids)
    node_sizes_list = [sizes[int(super_to_cid_arr[i])] for i in range(n_supers)]

    contracted_runner = runner.clone_for_graph(contracted)

    large_supers = {cid_to_super[c] for c in large_cids}
    small_supers = {cid_to_super[c] for c in small_cids}

    log.info("    Contracted graph: %d supernodes (%d large, %d small), %d edges",
             n_supers, len(large_supers), len(small_supers),
             contracted.ecount())

    # ── Adaptive coarsening on contracted graph ─────────────────────
    resolutions_used: List[float] = []
    total_clusters_resolved = 0
    total_nodes_resolved = 0
    consecutive_zero = 0

    factor = 0.5
    while small_supers and factor >= min_factor:
        coarse_gamma = gamma * factor
        resolutions_used.append(coarse_gamma)
        coarse_result = contracted_runner.run(
            coarse_gamma, n_iterations=coarse_iterations,
            node_sizes=node_sizes_list,
        )
        coarse_mem = coarse_result.membership

        # Map: coarse_community → {original_large_cid: total_node_count}
        coarse_to_large: Dict[int, Dict[int, int]] = {}
        for super_id in range(n_supers):
            if super_id not in large_supers:
                continue
            cc = coarse_mem[super_id]
            orig_cid = super_to_cid[super_id]
            bucket = coarse_to_large.get(cc)
            if bucket is None:
                bucket = {}
                coarse_to_large[cc] = bucket
            bucket[orig_cid] = bucket.get(orig_cid, 0) + sizes[orig_cid]

        # Resolve each small supernode.
        resolved_this_round: set[int] = set()
        for super_id in list(small_supers):
            cc = coarse_mem[super_id]
            large_counts = coarse_to_large.get(cc)
            if not large_counts:
                continue  # no large cluster in this coarse group — try next level

            target_cid = max(large_counts, key=large_counts.get)  # type: ignore[arg-type]
            orig_cid = super_to_cid[super_id]
            for n in small_cluster_nodes[orig_cid]:
                membership[n] = target_cid
            resolved_this_round.add(super_id)

        # Update bookkeeping.
        for super_id in resolved_this_round:
            orig_cid = super_to_cid[super_id]
            n_moved = len(small_cluster_nodes[orig_cid])
            total_clusters_resolved += 1
            total_nodes_resolved += n_moved
            sizes[membership[small_cluster_nodes[orig_cid][0]]] += n_moved
            del small_cluster_nodes[orig_cid]

        small_supers -= resolved_this_round

        n_resolved_round = len(resolved_this_round)
        log.info("    γ=%.2e (×%.3f) → %d resolved (%d remaining)",
                 coarse_gamma, factor, n_resolved_round, len(small_supers))

        # Convergence check: stop after 2 consecutive zero-resolution rounds.
        if n_resolved_round == 0:
            consecutive_zero += 1
            if consecutive_zero >= 2:
                log.info("    Converged (2 consecutive zero rounds)")
                break
        else:
            consecutive_zero = 0

        factor *= 0.5

    # ── Phase 2: edge-based fallback for remaining clusters ────────
    # Contracted CPM cannot resolve clusters whose supernodes never
    # group with a large supernode.  Fall back to original-graph edge
    # weights: assign each remaining small cluster to the large cluster
    # with the strongest total edge weight.
    n_phase1 = total_clusters_resolved
    if small_supers:
        graph = runner.graph
        weights = graph.es["weight"] if "weight" in graph.es.attributes() else None

        edge_resolved: set[int] = set()
        for super_id in list(small_supers):
            orig_cid = super_to_cid[super_id]
            nodes = small_cluster_nodes[orig_cid]

            # Sum edge weights from this cluster's nodes to each large cluster.
            weight_to_large: Dict[int, float] = {}
            for node_id in nodes:
                for eid in graph.incident(node_id):
                    e = graph.es[eid]
                    neighbor = e.target if e.source == node_id else e.source
                    neighbor_cid = membership[neighbor]
                    if neighbor_cid not in large_cids:
                        continue
                    w = weights[eid] if weights is not None else 1.0
                    weight_to_large[neighbor_cid] = (
                        weight_to_large.get(neighbor_cid, 0.0) + w
                    )

            if not weight_to_large:
                continue  # truly isolated from all large clusters

            target_cid = max(weight_to_large, key=weight_to_large.get)  # type: ignore[arg-type]
            for n in nodes:
                membership[n] = target_cid
            edge_resolved.add(super_id)

        for super_id in edge_resolved:
            orig_cid = super_to_cid[super_id]
            n_moved = len(small_cluster_nodes[orig_cid])
            total_clusters_resolved += 1
            total_nodes_resolved += n_moved
            sizes[membership[small_cluster_nodes[orig_cid][0]]] += n_moved
            del small_cluster_nodes[orig_cid]

        small_supers -= edge_resolved
        n_edge = len(edge_resolved)
        if n_edge:
            log.info("    Edge-based fallback → %d resolved (%d remaining)",
                     n_edge, len(small_supers))

    n_unresolvable_clusters = len(small_supers)
    n_unresolvable_nodes = sum(
        len(small_cluster_nodes[super_to_cid[s]]) for s in small_supers
    )

    log.info("  Small clusters: %d initial → %d resolved "
             "(%d CPM + %d edge-based, %d nodes), "
             "%d unresolvable (%d nodes, %.3f%%)",
             n_small_initial, total_clusters_resolved,
             n_phase1, total_clusters_resolved - n_phase1,
             total_nodes_resolved,
             n_unresolvable_clusters, n_unresolvable_nodes,
             n_unresolvable_nodes / len(membership) * 100 if membership else 0)

    return SmallClusterResolutionResult(
        membership=membership,
        n_small_initial=n_small_initial,
        n_small_nodes_initial=n_small_nodes_initial,
        n_clusters_resolved=total_clusters_resolved,
        n_nodes_resolved=total_nodes_resolved,
        n_clusters_unresolvable=n_unresolvable_clusters,
        n_nodes_unresolvable=n_unresolvable_nodes,
        resolutions_used=resolutions_used,
    )


def gamma_search(
    runner: LeidenRunner,
    gamma_range: tuple[float, float],
    min_size: int,
    *,
    n_coarse: int = 3,
    max_refine: int = 2,
    search_iterations: int | None = -1,
    warm_start: bool = True,
    node_sizes: Sequence[int] | None = None,
) -> GammaSearchResult:
    """Search for γ that maximises clusters ≥ *min_size*.

    Three speed optimisations over a naive grid search:

    1. **Auto-convergence** — ``search_iterations`` defaults to ``-1``
       (run until no improvement).  Combined with warm-start, each probe
       after the first converges in 1–2 iterations.
    2. **Warm-start** — each probe initialises from the nearest cached
       membership, so Leiden refines rather than searches from scratch.
    3. **Fewer refinement rounds** — ``max_refine`` (default 2) midpoint
       probes after the coarse scan.

    Parameters
    ----------
    runner : LeidenRunner
        Runner bound to the target graph.
    gamma_range : (float, float)
        ``(lo, hi)`` bounds for γ in linear scale.
    min_size : int
        Cluster size threshold; the objective is to maximise the number
        of clusters with size ≥ *min_size*.
    n_coarse : int
        Number of log-spaced probes in the initial scan.
    max_refine : int
        Maximum midpoint refinement rounds.
    search_iterations : int or None
        Leiden iterations per probe.  ``-1`` (default) = run until
        convergence.  ``None`` = use runner default.
    warm_start : bool
        If True, use the nearest cached membership as ``initial_membership``
        for each probe.
    node_sizes : sequence of int, optional
        Per-vertex sizes for contracted graphs.  When provided, cluster
        sizes are measured in original nodes (sum of node_sizes) rather
        than supernode count, and passed through to ``runner.run()``.
    """
    lo_g = math.log10(gamma_range[0])
    hi_g = math.log10(gamma_range[1])
    denom = max(n_coarse - 1, 1)
    coarse_gammas = [10 ** (lo_g + i * (hi_g - lo_g) / denom)
                     for i in range(n_coarse)]

    cache: Dict[float, tuple[int, list]] = {}

    def _nearest_membership(g: float) -> list[int] | None:
        if not cache or not warm_start:
            return None
        nearest = min(cache, key=lambda c: abs(math.log10(c) - math.log10(g)))
        return cache[nearest][1]

    def _eval(g: float) -> tuple[int, list]:
        run_kw: dict = {}
        if search_iterations is not None:
            run_kw["n_iterations"] = search_iterations
        init_mem = _nearest_membership(g)
        if init_mem is not None:
            run_kw["initial_membership"] = init_mem
        if node_sizes is not None:
            run_kw["node_sizes"] = node_sizes
        result = runner.run(g, **run_kw)
        mem = list(result.membership)
        if node_sizes is not None:
            # Weighted sizes: count original nodes per cluster
            weighted: Dict[int, int] = {}
            for v, cid in enumerate(mem):
                weighted[cid] = weighted.get(cid, 0) + node_sizes[v]
            n_large = sum(1 for s in weighted.values() if s >= min_size)
            n_cl = len(weighted)
        else:
            sizes = Counter(mem)
            n_large = sum(1 for s in sizes.values() if s >= min_size)
            n_cl = len(sizes)
        log.info("    γ=%.4e → %d clusters, %d large (≥%d)",
                 g, n_cl, n_large, min_size)
        return n_large, mem

    # Phase 1: Coarse scan
    for g in coarse_gammas:
        cache[g] = _eval(g)

    best_gamma = max(cache, key=lambda g: (cache[g][0], g))
    best_n = cache[best_gamma][0]

    # Phase 2: Midpoint refinement
    for _ in range(max_refine):
        sorted_gammas = sorted(cache.keys())
        idx = sorted_gammas.index(best_gamma)
        probes: list[float] = []
        if idx > 0:
            mid_lo = 10 ** (
                (math.log10(sorted_gammas[idx - 1]) + math.log10(best_gamma)) / 2
            )
            if mid_lo not in cache:
                probes.append(mid_lo)
        if idx < len(sorted_gammas) - 1:
            mid_hi = 10 ** (
                (math.log10(best_gamma) + math.log10(sorted_gammas[idx + 1])) / 2
            )
            if mid_hi not in cache:
                probes.append(mid_hi)
        if not probes:
            break
        for g in probes:
            cache[g] = _eval(g)

        best_gamma = max(cache, key=lambda g: (cache[g][0], g))
        new_best = cache[best_gamma][0]
        if new_best == best_n:
            break
        best_n = new_best

    log.info("    → Best: γ=%.4e, %d large clusters (%d evals)",
             best_gamma, best_n, len(cache))

    return GammaSearchResult(
        best_gamma=best_gamma,
        membership=cache[best_gamma][1],
        n_large=best_n,
        n_evals=len(cache),
    )


def find_and_run(
    runner: LeidenRunner,
    gamma_range: tuple[float, float],
    min_size: int,
    *,
    search_iterations: int | None = -1,
    warm_start: bool = True,
    n_coarse: int = 3,
    max_refine: int = 2,
) -> LeidenRunResult:
    """Search for optimal γ, then run Leiden with warm-start auto-convergence.

    Combines :func:`gamma_search` and a final Leiden run into one call.
    The search probes use ``n_iterations=-1`` (auto-convergence) with
    warm-start, so each probe after the first converges in 1–2 iterations.
    The final run also warm-starts from the search result.
    """
    sr = gamma_search(
        runner, gamma_range, min_size,
        search_iterations=search_iterations,
        warm_start=warm_start,
        n_coarse=n_coarse,
        max_refine=max_refine,
    )
    result = runner.run(
        sr.best_gamma,
        initial_membership=sr.membership,
        n_iterations=-1,
    )
    return result


def split_large_clusters(
    runner: LeidenRunner,
    membership: Sequence[int],
    gamma: float,
    *,
    max_size: int | None = None,
    min_size: int = 1000,
    split_iterations: int | None = -1,
) -> LargeClusterSplitResult:
    """Split oversized clusters via subgraph γ search.

    For each cluster larger than *max_size*, extracts an induced subgraph
    and searches for a γ > γ* that produces at least 2 sub-clusters
    ≥ *min_size*.

    Parameters
    ----------
    runner : LeidenRunner
        Runner bound to the full graph.
    membership : sequence of int
        Current membership vector.
    gamma : float
        Resolution used for the original clustering (γ*).
    max_size : int or None
        Split threshold.  Default ``2 × min_size``.
    min_size : int
        Minimum acceptable cluster size (search criterion).
    split_iterations : int or None
        Leiden iterations per γ probe on subgraphs.  Default ``-1``
        (run until convergence).
    """
    membership = list(membership)
    if max_size is None:
        max_size = 2 * min_size

    sizes = Counter(membership)
    graph = runner.graph

    oversized = sorted(cid for cid, sz in sizes.items() if sz > max_size)

    if not oversized:
        return LargeClusterSplitResult(
            membership=membership,
            n_clusters_split=0,
            n_nodes_affected=0,
            n_new_clusters_created=0,
            split_gammas={},
        )

    log.info("  Splitting %d oversized clusters (size > %d)...",
             len(oversized), max_size)

    next_cid = max(sizes.keys()) + 1
    n_split = 0
    n_nodes_affected = 0
    n_new_clusters = 0
    split_gammas: Dict[int, float] = {}

    mem_arr = np.array(membership)
    for cid in oversized:
        nodes = np.where(mem_arr == cid)[0].tolist()

        subgraph = graph.induced_subgraph(nodes)
        sub_runner = runner.clone_for_graph(subgraph)

        search_result = gamma_search(
            sub_runner,
            gamma_range=(gamma, gamma * 1000),
            min_size=min_size,
            search_iterations=split_iterations,
        )
        best_gamma_sub = search_result.best_gamma
        sub_mem = search_result.membership
        n_large = search_result.n_large

        if n_large < 2:
            log.info("    Cluster %d (%d nodes): no meaningful split (n_large=%d)",
                     cid, len(nodes), n_large)
            continue

        # Largest sub-cluster keeps original ID; others get new IDs.
        sub_sizes = Counter(sub_mem)
        largest_sub = max(sub_sizes, key=lambda k: sub_sizes[k])

        id_mapping: Dict[int, int] = {}
        for sub_cid in sorted(set(sub_mem)):
            if sub_cid == largest_sub:
                id_mapping[sub_cid] = cid
            else:
                id_mapping[sub_cid] = next_cid
                next_cid += 1
                n_new_clusters += 1

        for i, node_id in enumerate(nodes):
            membership[node_id] = id_mapping[sub_mem[i]]

        split_gammas[cid] = best_gamma_sub
        n_split += 1
        n_nodes_affected += len(nodes)

        sub_sizes_mapped = Counter(id_mapping[s] for s in sub_mem)
        log.info("    Cluster %d (%d nodes) → %d sub-clusters at γ=%.2e: %s",
                 cid, len(nodes), len(sub_sizes_mapped), best_gamma_sub,
                 sorted(sub_sizes_mapped.values(), reverse=True))

    log.info("  Split result: %d clusters split, %d new sub-clusters, "
             "%d nodes affected",
             n_split, n_new_clusters, n_nodes_affected)

    return LargeClusterSplitResult(
        membership=membership,
        n_clusters_split=n_split,
        n_nodes_affected=n_nodes_affected,
        n_new_clusters_created=n_new_clusters,
        split_gammas=split_gammas,
    )


def refine_clusters(
    runner: LeidenRunner,
    membership: Sequence[int],
    gamma: float,
    *,
    min_size: int = 1000,
    max_rounds: int = 3,
    min_factor: float = 0.0001,
    coarse_iterations: int | None = 2,
    split_iterations: int | None = -1,
) -> RefinementResult:
    """Split oversized + merge undersized clusters in a convergence loop.

    Each round:
      1. **Split** clusters larger than ``2 × min_size`` via subgraph γ search.
      2. **Merge** clusters smaller than *min_size* via contracted-graph CPM.
      3. Stop if neither phase made any changes.

    Parameters
    ----------
    runner : LeidenRunner
        Runner bound to the full graph.
    membership : sequence of int
        Initial membership vector (e.g. raw Leiden output).
    gamma : float
        Resolution used for the initial clustering (γ*).
    min_size : int
        Target minimum cluster size.
    max_rounds : int
        Safety cap on split–merge iterations.
    min_factor, coarse_iterations
        Forwarded to :func:`resolve_small_clusters`.
    split_iterations
        Forwarded to :func:`split_large_clusters`.
    """
    membership = list(membership)
    split_results: List[LargeClusterSplitResult] = []
    merge_results: List[SmallClusterResolutionResult] = []

    for round_i in range(max_rounds):
        log.info("  ── Refinement round %d ──", round_i + 1)

        # Phase 1: Split oversized
        split_result = split_large_clusters(
            runner, membership, gamma, min_size=min_size,
            split_iterations=split_iterations,
        )
        membership = split_result.membership
        split_results.append(split_result)

        # Phase 2: Merge undersized
        merge_result = resolve_small_clusters(
            runner, membership, gamma,
            min_size=min_size,
            min_factor=min_factor,
            coarse_iterations=coarse_iterations,
        )
        membership = merge_result.membership
        merge_results.append(merge_result)

        # Phase 3: Stability check
        if (split_result.n_clusters_split == 0
                and merge_result.n_clusters_resolved == 0):
            log.info("  Refinement converged at round %d", round_i + 1)
            break

    return RefinementResult(
        membership=membership,
        n_rounds=len(split_results),
        split_results=split_results,
        merge_results=merge_results,
    )


def cpm_quality(
    graph: ig.Graph,
    membership: Sequence[int],
    gamma: float,
) -> float:
    """Compute CPM quality: H(P) = Σ_c [ e_c − γ · C(n_c, 2) ].

    Parameters
    ----------
    graph : igraph.Graph
        The graph (weighted or unweighted).
    membership : sequence of int
        Cluster assignment for each node.
    gamma : float
        Resolution parameter.

    Returns
    -------
    float
        Total CPM quality.  Higher is better.
    """
    mem = np.asarray(membership)
    edges = np.array(graph.get_edgelist())  # (|E|, 2)
    if edges.size == 0:
        # No edges: quality is purely the penalty term
        sizes = np.bincount(mem)
        return -gamma * float(np.sum(sizes * (sizes - 1) / 2))

    w = np.array(graph.es["weight"], dtype=np.float64) if "weight" in graph.es.attributes() \
        else np.ones(edges.shape[0], dtype=np.float64)

    cu = mem[edges[:, 0]]
    cv = mem[edges[:, 1]]
    internal_mask = cu == cv
    # Sum internal weights per cluster
    internal = np.bincount(cu[internal_mask], weights=w[internal_mask],
                           minlength=mem.max() + 1)
    # Cluster sizes
    sizes = np.bincount(mem, minlength=mem.max() + 1).astype(np.float64)
    # CPM quality: Σ_c [ e_c - γ * n_c * (n_c - 1) / 2 ]
    return float(np.sum(internal - gamma * sizes * (sizes - 1) / 2))


__all__ = [
    "merge_small_clusters",
    "PostprocessResult",
    "MergeAction",
    "resolve_singletons",
    "SingletonResolutionResult",
    "resolve_small_clusters",
    "SmallClusterResolutionResult",
    "split_large_clusters",
    "LargeClusterSplitResult",
    "refine_clusters",
    "RefinementResult",
    "gamma_search",
    "GammaSearchResult",
    "find_and_run",
    "cpm_quality",
]
