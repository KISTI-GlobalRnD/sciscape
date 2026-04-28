"""Worst-case cluster quality sampler.

Strategy: pick nodes that are hardest for clustering to get right,
then sample their cluster neighbors from easy → hard.

Hard nodes = high cross-cluster edge ratio (boundary nodes).
Hard neighbors = far from target in the cluster (weakest link).
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


def _min_group_size(n_neighbors: int) -> int:
    """Minimum same-cluster group size required for review."""
    return max(2, (n_neighbors + 1) // 2)


@dataclass
class SampleCase:
    """One evaluation case: a target node + sampled cluster neighbors."""
    target_uid: str
    cluster_id: int
    cluster_size: int
    # Neighbors sampled from same cluster
    easy_neighbors: List[str]   # strongest intra-cluster connections
    hard_neighbors: List[str]   # weakest intra-cluster connections
    # Cross-cluster info
    cross_cluster_ratio: float  # fraction of edges going outside cluster
    n_cross_edges: int
    # Metadata
    target_title: str = ""
    target_year: int | None = None


@dataclass
class SampleSet:
    """Collection of evaluation cases."""
    cases: List[SampleCase]
    field: str
    method: str  # e.g. "rank_bc_cc_dc"
    n_clusters: int
    n_nodes: int


@dataclass
class DisagreementCase:
    """One A/B disagreement case for blind comparison."""
    target_uid: str
    method_a_cluster_id: int
    method_b_cluster_id: int
    method_a_cluster_size: int
    method_b_cluster_size: int
    method_a_cross_cluster_ratio: float
    method_b_cross_cluster_ratio: float
    group_a_uids: List[str]
    group_b_uids: List[str]
    overlap_size: int
    jaccard: float
    target_title: str = ""
    target_year: int | None = None


@dataclass
class DisagreementSampleSet:
    """Collection of disagreement cases for A/B review."""
    cases: List[DisagreementCase]
    method_a: str
    method_b: str
    n_nodes: int
    n_candidates: int


@dataclass
class RankShiftCase:
    """One target whose local neighbor ranking changed strongly across methods."""
    target_uid: str
    method_a_cluster_id: int
    method_b_cluster_id: int
    method_a_cluster_size: int
    method_b_cluster_size: int
    rank_jaccard: float
    overlap_size: int
    mean_abs_rank_shift: float
    max_abs_rank_shift: int
    cluster_overlap_coeff: float
    cluster_changed: bool
    shift_score: float
    neighbors_a: List[dict[str, Any]]
    neighbors_b: List[dict[str, Any]]
    shared_neighbors: List[dict[str, Any]]
    neighbors_only_a: List[dict[str, Any]]
    neighbors_only_b: List[dict[str, Any]]
    target_title: str = ""
    target_year: int | None = None


@dataclass
class RankShiftSampleSet:
    """Collection of rank-shift cases for local neighborhood review."""
    cases: List[RankShiftCase]
    method_a: str
    method_b: str
    n_nodes: int
    n_candidates: int


@dataclass
class BoundaryCoverageCase:
    """One coverage-aware boundary case for Protocol D v2."""
    target_uid: str
    coverage_state: str
    method_a_reviewable: bool
    method_b_reviewable: bool
    method_a_cluster_id: int | None
    method_b_cluster_id: int | None
    method_a_cluster_size: int
    method_b_cluster_size: int
    method_a_cross_cluster_ratio: float
    method_b_cross_cluster_ratio: float
    group_a_uids: List[str]
    group_b_uids: List[str]
    group_a_doc_count: int
    group_b_doc_count: int
    overlap_size: int
    jaccard: float
    cluster_size_ratio: float | None
    method_a_total_degree: int
    method_b_total_degree: int
    method_a_intra_degree: int
    method_b_intra_degree: int
    method_a_cross_degree: int
    method_b_cross_degree: int
    method_a_group_fill_rate: float
    method_b_group_fill_rate: float
    diagnostic_strata: List[str]
    target_title: str = ""
    target_year: int | None = None


@dataclass
class BoundaryCoverageSampleSet:
    """Population and diagnostic samples for coverage-aware boundary review."""
    population_cases: List[BoundaryCoverageCase]
    diagnostic_cases: List[BoundaryCoverageCase]
    method_a: str
    method_b: str
    n_nodes: int
    n_target_universe: int
    coverage_state_counts: Dict[str, int]
    sample_mode: str


@dataclass
class _MethodStats:
    """Per-method neighborhood statistics used for disagreement sampling."""
    cross_ratios: Dict[str, float]
    cluster_sizes: Counter
    cluster_members: Dict[int, List[str]]
    cluster_member_sets: Dict[int, set[str]]
    node_neighbors: Dict[str, Dict[str, float]]
    node_intra: Dict[str, float]
    node_total_degree: Dict[str, int]
    node_intra_degree: Dict[str, int]
    node_cross_degree: Dict[str, int]


def _compute_all_neighbors(edges: pl.DataFrame) -> Dict[str, Dict[str, float]]:
    """Build symmetric per-node neighbor weights from an edge table."""
    all_neighbors: Dict[str, Dict[str, float]] = defaultdict(dict)
    for row in edges.iter_rows(named=True):
        u1, u2 = row["uid1"], row["uid2"]
        w = float(row.get("rel_sum2", 1.0))
        all_neighbors[u1][u2] = all_neighbors[u1].get(u2, 0.0) + w
        all_neighbors[u2][u1] = all_neighbors[u2].get(u1, 0.0) + w
    return all_neighbors


def _top_neighbors(
    uid: str,
    neighbor_map: Dict[str, Dict[str, float]],
    membership: Dict[str, int],
    *,
    n_neighbors: int,
    allowed_uids: set[str] | None = None,
) -> List[dict[str, Any]]:
    """Return ranked local neighbors for one node."""
    ordered = sorted(
        neighbor_map.get(uid, {}).items(),
        key=lambda item: (-item[1], item[0]),
    )
    results: List[dict[str, Any]] = []
    target_cluster = membership.get(uid)
    for nbr, weight in ordered:
        if allowed_uids is not None and nbr not in allowed_uids:
            continue
        results.append(
            {
                "uid": nbr,
                "rank": len(results) + 1,
                "weight": round(weight, 6),
                "same_cluster": membership.get(nbr) == target_cluster,
            }
        )
        if len(results) >= n_neighbors:
            break
    return results


def sample_worst_case(
    edges: pl.DataFrame,
    membership: Dict[str, int],
    *,
    abstracts: pl.DataFrame | None = None,
    n_targets: int = 50,
    n_easy: int = 5,
    n_hard: int = 5,
    min_cluster_size: int = 10,
    boundary_quantile: float = 0.9,
    seed: int = 42,
) -> SampleSet:
    """Sample worst-case nodes for cluster quality evaluation.

    Parameters
    ----------
    edges : pl.DataFrame
        Edge table (uid1, uid2, rel_sum2).
    membership : dict
        UID → cluster ID mapping.
    abstracts : pl.DataFrame, optional
        Abstract table (uid, title, abstract, pubyear) for metadata.
    n_targets : int
        Number of target nodes to sample.
    n_easy, n_hard : int
        Neighbors per target from same cluster (strongest/weakest).
    min_cluster_size : int
        Only sample from clusters with >= this many nodes.
    boundary_quantile : float
        Sample targets from top quantile of cross-cluster ratio.
    """
    rng = np.random.RandomState(seed)

    # Build per-node edge stats
    node_intra: Dict[str, float] = defaultdict(float)   # intra-cluster weight
    node_cross: Dict[str, float] = defaultdict(float)   # cross-cluster weight
    node_cross_counts: Dict[str, int] = defaultdict(int)  # cross-cluster edge count
    node_neighbors: Dict[str, Dict[str, float]] = defaultdict(dict)  # same-cluster neighbors

    for row in edges.iter_rows(named=True):
        u1, u2 = row["uid1"], row["uid2"]
        w = float(row.get("rel_sum2", 1.0))
        c1 = membership.get(u1)
        c2 = membership.get(u2)
        if c1 is None or c2 is None:
            continue
        if c1 == c2:
            node_intra[u1] += w
            node_intra[u2] += w
            node_neighbors[u1][u2] = node_neighbors[u1].get(u2, 0) + w
            node_neighbors[u2][u1] = node_neighbors[u2].get(u1, 0) + w
        else:
            node_cross[u1] += w
            node_cross[u2] += w
            node_cross_counts[u1] += 1
            node_cross_counts[u2] += 1

    # Compute cross-cluster ratio per node
    cross_ratios: Dict[str, float] = {}
    for uid in membership:
        total = node_intra.get(uid, 0) + node_cross.get(uid, 0)
        if total > 0:
            cross_ratios[uid] = node_cross.get(uid, 0) / total
        else:
            cross_ratios[uid] = 0.0

    # Filter: only nodes in large enough clusters
    cluster_sizes = Counter(membership.values())
    cluster_members: Dict[int, List[str]] = defaultdict(list)
    for uid, cid in membership.items():
        cluster_members[cid].append(uid)
    eligible = [
        uid for uid, cid in membership.items()
        if cluster_sizes[cid] >= min_cluster_size and uid in cross_ratios
    ]

    if not eligible:
        log.warning("No eligible nodes for sampling")
        return SampleSet(cases=[], field="", method="", n_clusters=0, n_nodes=0)

    # Sort by cross-cluster ratio (descending) — hardest nodes first
    eligible.sort(key=lambda u: -cross_ratios[u])

    # Take from top boundary_quantile
    cutoff = max(1, int(len(eligible) * (1 - boundary_quantile)))
    boundary_pool = eligible[:cutoff]

    # Ensure diversity: sample from different clusters
    cluster_pool: Dict[int, List[str]] = defaultdict(list)
    for uid in boundary_pool:
        cluster_pool[membership[uid]].append(uid)

    # Sample targets: spread across clusters
    targets: List[str] = []
    cluster_ids = list(cluster_pool.keys())
    rng.shuffle(cluster_ids)

    per_cluster = max(1, n_targets // len(cluster_ids)) if cluster_ids else 0
    for cid in cluster_ids:
        pool = cluster_pool[cid]
        n = min(per_cluster, len(pool))
        chosen = rng.choice(pool, size=n, replace=False).tolist()
        targets.extend(chosen)
        if len(targets) >= n_targets:
            break

    targets = targets[:n_targets]

    # Load metadata
    uid_meta: Dict[str, dict] = {}
    if abstracts is not None:
        for row in abstracts.iter_rows(named=True):
            uid_meta[row["uid"]] = row

    # Build cases
    cases: List[SampleCase] = []
    for uid in targets:
        cid = membership[uid]
        neighbors = node_neighbors.get(uid, {})

        if neighbors:
            # Sort direct same-cluster neighbors by weight
            sorted_nbrs = sorted(neighbors.items(), key=lambda x: -x[1])
            easy = [n for n, _ in sorted_nbrs[:n_easy]]
            hard = [n for n, _ in sorted_nbrs[-n_hard:]] if len(sorted_nbrs) > n_easy else []
        else:
            # Postprocess may merge nodes into clusters without direct same-cluster
            # edges. Fall back to representative cluster members so review-set
            # construction still works on merged partitions.
            candidates = [v for v in cluster_members[cid] if v != uid]
            candidates.sort(key=lambda v: (-node_intra.get(v, 0.0), v))
            easy = candidates[:n_easy]
            hard = candidates[-n_hard:] if len(candidates) > n_easy else []
            if not easy and not hard:
                continue

        meta = uid_meta.get(uid, {})
        cases.append(SampleCase(
            target_uid=uid,
            cluster_id=cid,
            cluster_size=cluster_sizes[cid],
            easy_neighbors=easy,
            hard_neighbors=hard,
            cross_cluster_ratio=round(cross_ratios[uid], 4),
            n_cross_edges=int(node_cross_counts.get(uid, 0)),
            target_title=meta.get("title", ""),
            target_year=meta.get("pubyear"),
        ))

    log.info("Sampled %d worst-case targets from %d eligible (%.0f%% boundary quantile)",
             len(cases), len(eligible), boundary_quantile * 100)

    return SampleSet(
        cases=cases,
        field="",
        method="",
        n_clusters=len(cluster_sizes),
        n_nodes=len(membership),
    )


def _compute_method_stats(edges: pl.DataFrame, membership: Dict[str, int]) -> _MethodStats:
    """Build per-node stats for one clustering result."""
    node_intra: Dict[str, float] = defaultdict(float)
    node_cross: Dict[str, float] = defaultdict(float)
    node_neighbors: Dict[str, Dict[str, float]] = defaultdict(dict)
    node_total_degree: Dict[str, int] = defaultdict(int)
    node_intra_degree: Dict[str, int] = defaultdict(int)
    node_cross_degree: Dict[str, int] = defaultdict(int)

    for row in edges.iter_rows(named=True):
        u1, u2 = row["uid1"], row["uid2"]
        w = float(row.get("rel_sum2", 1.0))
        c1 = membership.get(u1)
        c2 = membership.get(u2)
        if c1 is None or c2 is None:
            continue
        node_total_degree[u1] += 1
        node_total_degree[u2] += 1
        if c1 == c2:
            node_intra[u1] += w
            node_intra[u2] += w
            node_intra_degree[u1] += 1
            node_intra_degree[u2] += 1
            node_neighbors[u1][u2] = node_neighbors[u1].get(u2, 0.0) + w
            node_neighbors[u2][u1] = node_neighbors[u2].get(u1, 0.0) + w
        else:
            node_cross[u1] += w
            node_cross[u2] += w
            node_cross_degree[u1] += 1
            node_cross_degree[u2] += 1

    cross_ratios: Dict[str, float] = {}
    for uid in membership:
        total = node_intra.get(uid, 0.0) + node_cross.get(uid, 0.0)
        cross_ratios[uid] = (node_cross.get(uid, 0.0) / total) if total > 0 else 0.0

    cluster_sizes = Counter(membership.values())
    cluster_members: Dict[int, List[str]] = defaultdict(list)
    for uid, cid in membership.items():
        cluster_members[cid].append(uid)
    cluster_member_sets = {cid: set(uids) for cid, uids in cluster_members.items()}

    return _MethodStats(
        cross_ratios=cross_ratios,
        cluster_sizes=cluster_sizes,
        cluster_members=cluster_members,
        cluster_member_sets=cluster_member_sets,
        node_neighbors=node_neighbors,
        node_intra=node_intra,
        node_total_degree=node_total_degree,
        node_intra_degree=node_intra_degree,
        node_cross_degree=node_cross_degree,
    )


def _cluster_overlap_coefficient(
    uid: str,
    membership_a: Dict[str, int],
    stats_a: _MethodStats,
    membership_b: Dict[str, int],
    stats_b: _MethodStats,
) -> float:
    """Return overlap coefficient for the target's assigned clusters across two partitions."""
    cid_a = membership_a.get(uid)
    cid_b = membership_b.get(uid)
    if cid_a is None or cid_b is None:
        return 0.0
    members_a = stats_a.cluster_member_sets.get(cid_a, set())
    members_b = stats_b.cluster_member_sets.get(cid_b, set())
    denom = min(len(members_a), len(members_b))
    if denom == 0:
        return 0.0
    return len(members_a & members_b) / denom


def _select_group(
    uid: str,
    membership: Dict[str, int],
    stats: _MethodStats,
    *,
    n_neighbors: int,
    min_group_size: int,
    allowed_uids: set[str] | None = None,
) -> List[str]:
    """Return same-cluster neighbors for one method, with postprocess fallback."""
    cid = membership.get(uid)
    if cid is None:
        return []

    neighbors = stats.node_neighbors.get(uid, {})
    if neighbors:
        ordered = sorted(neighbors.items(), key=lambda item: (-item[1], item[0]))
        if allowed_uids is not None:
            ordered = [(nbr, weight) for nbr, weight in ordered if nbr in allowed_uids]
        group = [nbr for nbr, _weight in ordered[:n_neighbors]]
        if len(group) >= min_group_size:
            return group

    # Postprocess may merge nodes into clusters without direct same-cluster edges.
    candidates = [v for v in stats.cluster_members[cid] if v != uid]
    if allowed_uids is not None:
        candidates = [v for v in candidates if v in allowed_uids]
    candidates.sort(key=lambda v: (-stats.node_intra.get(v, 0.0), v))
    return candidates[:n_neighbors]


def sample_disagreement_cases(
    edges_a: pl.DataFrame,
    membership_a: Dict[str, int],
    edges_b: pl.DataFrame,
    membership_b: Dict[str, int],
    *,
    method_a: str = "sum",
    method_b: str = "consensus",
    abstracts: pl.DataFrame | None = None,
    n_targets: int = 50,
    n_neighbors: int = 8,
    min_cluster_size: int = 10,
    boundary_quantile: float = 0.9,
    max_group_jaccard: float = 0.5,
    allowed_uids: set[str] | None = None,
    seed: int = 42,
) -> DisagreementSampleSet:
    """Sample boundary-node cases where method A and B disagree meaningfully."""
    rng = np.random.RandomState(seed)
    stats_a = _compute_method_stats(edges_a, membership_a)
    stats_b = _compute_method_stats(edges_b, membership_b)
    min_group_size = _min_group_size(n_neighbors)

    uid_meta: Dict[str, dict] = {}
    if abstracts is not None:
        for row in abstracts.iter_rows(named=True):
            uid_meta[row["uid"]] = row

    eligible = [
        uid
        for uid in set(membership_a) & set(membership_b)
        if allowed_uids is None or uid in allowed_uids
        if stats_a.cluster_sizes[membership_a[uid]] >= min_cluster_size
        and stats_b.cluster_sizes[membership_b[uid]] >= min_cluster_size
    ]
    if not eligible:
        log.warning("No eligible disagreement targets")
        return DisagreementSampleSet([], method_a=method_a, method_b=method_b, n_nodes=0, n_candidates=0)

    eligible.sort(
        key=lambda uid: (
            -max(
                stats_a.cross_ratios.get(uid, 0.0),
                stats_b.cross_ratios.get(uid, 0.0),
            ),
            uid,
        )
    )
    cutoff = max(1, int(len(eligible) * (1 - boundary_quantile)))
    boundary_pool = eligible[:cutoff]

    candidate_rows: list[DisagreementCase] = []
    for uid in boundary_pool:
        group_a = _select_group(
            uid,
            membership_a,
            stats_a,
            n_neighbors=n_neighbors,
            min_group_size=min_group_size,
            allowed_uids=allowed_uids,
        )
        group_b = _select_group(
            uid,
            membership_b,
            stats_b,
            n_neighbors=n_neighbors,
            min_group_size=min_group_size,
            allowed_uids=allowed_uids,
        )
        if len(group_a) < min_group_size or len(group_b) < min_group_size:
            continue

        set_a = set(group_a)
        set_b = set(group_b)
        union = set_a | set_b
        overlap = set_a & set_b
        jaccard = len(overlap) / len(union) if union else 1.0
        if jaccard > max_group_jaccard:
            continue
        if set_a == set_b:
            continue

        meta = uid_meta.get(uid, {})
        candidate_rows.append(
            DisagreementCase(
                target_uid=uid,
                method_a_cluster_id=membership_a[uid],
                method_b_cluster_id=membership_b[uid],
                method_a_cluster_size=stats_a.cluster_sizes[membership_a[uid]],
                method_b_cluster_size=stats_b.cluster_sizes[membership_b[uid]],
                method_a_cross_cluster_ratio=round(stats_a.cross_ratios.get(uid, 0.0), 4),
                method_b_cross_cluster_ratio=round(stats_b.cross_ratios.get(uid, 0.0), 4),
                group_a_uids=group_a,
                group_b_uids=group_b,
                overlap_size=len(overlap),
                jaccard=round(jaccard, 4),
                target_title=meta.get("title", ""),
                target_year=meta.get("pubyear"),
            )
        )

    if not candidate_rows:
        log.warning("No disagreement cases after group-difference filtering")
        return DisagreementSampleSet([], method_a=method_a, method_b=method_b, n_nodes=len(eligible), n_candidates=0)

    bucketed: Dict[tuple[int, int], List[DisagreementCase]] = defaultdict(list)
    for case in candidate_rows:
        bucketed[(case.method_a_cluster_id, case.method_b_cluster_id)].append(case)

    keys = list(bucketed)
    rng.shuffle(keys)
    selected: List[DisagreementCase] = []
    per_bucket = max(1, n_targets // max(1, len(keys)))
    for key in keys:
        bucket = bucketed[key]
        rng.shuffle(bucket)
        selected.extend(bucket[:per_bucket])
        if len(selected) >= n_targets:
            break

    if len(selected) < min(n_targets, len(candidate_rows)):
        seen = {case.target_uid for case in selected}
        remaining = [case for case in candidate_rows if case.target_uid not in seen]
        rng.shuffle(remaining)
        selected.extend(remaining[: max(0, n_targets - len(selected))])

    selected = selected[:n_targets]
    log.info(
        "Sampled %d disagreement targets from %d candidates (boundary q=%.2f, max_jaccard=%.2f)",
        len(selected),
        len(candidate_rows),
        boundary_quantile,
        max_group_jaccard,
    )

    return DisagreementSampleSet(
        cases=selected,
        method_a=method_a,
        method_b=method_b,
        n_nodes=len(eligible),
        n_candidates=len(candidate_rows),
    )


def sample_rank_shift_cases(
    edges_a: pl.DataFrame,
    membership_a: Dict[str, int],
    edges_b: pl.DataFrame,
    membership_b: Dict[str, int],
    *,
    method_a: str = "sum",
    method_b: str = "consensus",
    abstracts: pl.DataFrame | None = None,
    n_targets: int = 50,
    n_neighbors: int = 8,
    min_cluster_size: int = 10,
    max_rank_jaccard: float = 0.85,
    min_cluster_overlap: float = 0.5,
    allowed_uids: set[str] | None = None,
    target_uids: List[str] | None = None,
    strict_target_uids: bool = False,
    seed: int = 42,
) -> RankShiftSampleSet:
    """Sample targets whose local neighbor ranking changes strongly."""
    rng = np.random.RandomState(seed)
    candidate_rows, n_eligible = collect_rank_shift_cases(
        edges_a,
        membership_a,
        edges_b,
        membership_b,
        method_a=method_a,
        method_b=method_b,
        abstracts=abstracts,
        n_neighbors=n_neighbors,
        min_cluster_size=min_cluster_size,
        max_rank_jaccard=max_rank_jaccard,
        min_cluster_overlap=min_cluster_overlap,
        allowed_uids=allowed_uids,
        target_uids=target_uids,
    )
    if not candidate_rows:
        log.warning("No rank-shift cases after filtering")
        return RankShiftSampleSet([], method_a=method_a, method_b=method_b, n_nodes=n_eligible, n_candidates=0)

    if target_uids is not None:
        preferred_set = set(target_uids)
        selected = [case for case in candidate_rows if case.target_uid in preferred_set][:n_targets]
        if not strict_target_uids and len(selected) < n_targets:
            seen = {case.target_uid for case in selected}
            remaining = [case for case in candidate_rows if case.target_uid not in seen]
            selected.extend(remaining[: max(0, n_targets - len(selected))])
        selected = selected[:n_targets]
    else:
        bucketed: Dict[tuple[int, int], List[RankShiftCase]] = defaultdict(list)
        for case in candidate_rows:
            bucketed[(case.method_a_cluster_id, case.method_b_cluster_id)].append(case)

        keys = list(bucketed)
        rng.shuffle(keys)
        selected = []
        per_bucket = max(1, n_targets // max(1, len(keys)))
        for key in keys:
            bucket = bucketed[key]
            bucket.sort(key=lambda case: (-case.shift_score, case.target_uid))
            selected.extend(bucket[:per_bucket])
            if len(selected) >= n_targets:
                break

        if len(selected) < min(n_targets, len(candidate_rows)):
            seen = {case.target_uid for case in selected}
            remaining = [case for case in candidate_rows if case.target_uid not in seen]
            selected.extend(remaining[: max(0, n_targets - len(selected))])

        selected = selected[:n_targets]
    log.info(
        "Sampled %d rank-shift targets from %d candidates (max_jaccard=%.2f)",
        len(selected),
        len(candidate_rows),
        max_rank_jaccard,
    )

    return RankShiftSampleSet(
        cases=selected,
        method_a=method_a,
        method_b=method_b,
        n_nodes=n_eligible,
        n_candidates=len(candidate_rows),
    )


def collect_rank_shift_cases(
    edges_a: pl.DataFrame,
    membership_a: Dict[str, int],
    edges_b: pl.DataFrame,
    membership_b: Dict[str, int],
    *,
    method_a: str = "sum",
    method_b: str = "consensus",
    abstracts: pl.DataFrame | None = None,
    n_neighbors: int = 8,
    min_cluster_size: int = 10,
    max_rank_jaccard: float = 0.85,
    min_cluster_overlap: float = 0.5,
    allowed_uids: set[str] | None = None,
    target_uids: List[str] | None = None,
) -> tuple[List[RankShiftCase], int]:
    """Collect every rank-shift case before bucket sampling."""
    stats_a = _compute_method_stats(edges_a, membership_a)
    stats_b = _compute_method_stats(edges_b, membership_b)
    all_neighbors_a = _compute_all_neighbors(edges_a)
    all_neighbors_b = _compute_all_neighbors(edges_b)

    uid_meta: Dict[str, dict] = {}
    if abstracts is not None:
        for row in abstracts.iter_rows(named=True):
            uid_meta[row["uid"]] = row

    if target_uids is not None:
        preferred = list(dict.fromkeys(target_uids))
        remaining = sorted((set(membership_a) & set(membership_b)) - set(preferred))
        ordered_candidates = preferred + remaining
    else:
        ordered_candidates = sorted(set(membership_a) & set(membership_b))

    eligible = [
        uid
        for uid in ordered_candidates
        if uid in membership_a and uid in membership_b
        if allowed_uids is None or uid in allowed_uids
        if stats_a.cluster_sizes[membership_a[uid]] >= min_cluster_size
        and stats_b.cluster_sizes[membership_b[uid]] >= min_cluster_size
    ]
    if not eligible:
        log.warning("No eligible rank-shift targets")
        return [], 0

    candidate_rows: list[RankShiftCase] = []
    for uid in eligible:
        neighbors_a = _top_neighbors(
            uid,
            all_neighbors_a,
            membership_a,
            n_neighbors=n_neighbors,
            allowed_uids=allowed_uids,
        )
        neighbors_b = _top_neighbors(
            uid,
            all_neighbors_b,
            membership_b,
            n_neighbors=n_neighbors,
            allowed_uids=allowed_uids,
        )
        if len(neighbors_a) < n_neighbors or len(neighbors_b) < n_neighbors:
            continue

        order_a = [row["uid"] for row in neighbors_a]
        order_b = [row["uid"] for row in neighbors_b]
        cluster_overlap_coeff = round(
            _cluster_overlap_coefficient(uid, membership_a, stats_a, membership_b, stats_b), 4
        )
        cluster_changed = cluster_overlap_coeff < min_cluster_overlap
        if order_a == order_b and not cluster_changed:
            continue

        ranks_a = {row["uid"]: row["rank"] for row in neighbors_a}
        ranks_b = {row["uid"]: row["rank"] for row in neighbors_b}
        set_a = set(ranks_a)
        set_b = set(ranks_b)
        union = set_a | set_b
        overlap = set_a & set_b
        rank_jaccard = len(overlap) / len(union) if union else 1.0
        if rank_jaccard > max_rank_jaccard:
            continue

        shared_neighbors: list[dict[str, Any]] = []
        abs_rank_shifts: list[int] = []
        for nbr in sorted(overlap):
            delta = int(ranks_b[nbr] - ranks_a[nbr])
            abs_rank_shifts.append(abs(delta))
            row_a = next(row for row in neighbors_a if row["uid"] == nbr)
            row_b = next(row for row in neighbors_b if row["uid"] == nbr)
            shared_neighbors.append(
                {
                    "uid": nbr,
                    "rank_a": row_a["rank"],
                    "rank_b": row_b["rank"],
                    "delta": delta,
                    "weight_a": row_a["weight"],
                    "weight_b": row_b["weight"],
                }
            )
        shared_neighbors.sort(key=lambda row: (-abs(row["delta"]), row["uid"]))

        neighbors_only_a = [row for row in neighbors_a if row["uid"] not in set_b]
        neighbors_only_b = [row for row in neighbors_b if row["uid"] not in set_a]
        mean_abs_rank_shift = (
            round(float(np.mean(abs_rank_shifts)), 4) if abs_rank_shifts else float(n_neighbors)
        )
        max_abs_rank_shift = max(abs_rank_shifts) if abs_rank_shifts else n_neighbors
        shift_score = round(
            (1.0 - rank_jaccard)
            + (mean_abs_rank_shift / max(1, n_neighbors))
            + (0.25 if cluster_changed else 0.0),
            4,
        )

        meta = uid_meta.get(uid, {})
        candidate_rows.append(
            RankShiftCase(
                target_uid=uid,
                method_a_cluster_id=membership_a[uid],
                method_b_cluster_id=membership_b[uid],
                method_a_cluster_size=stats_a.cluster_sizes[membership_a[uid]],
                method_b_cluster_size=stats_b.cluster_sizes[membership_b[uid]],
                rank_jaccard=round(rank_jaccard, 4),
                overlap_size=len(overlap),
                mean_abs_rank_shift=mean_abs_rank_shift,
                max_abs_rank_shift=max_abs_rank_shift,
                cluster_overlap_coeff=cluster_overlap_coeff,
                cluster_changed=cluster_changed,
                shift_score=shift_score,
                neighbors_a=neighbors_a,
                neighbors_b=neighbors_b,
                shared_neighbors=shared_neighbors,
                neighbors_only_a=neighbors_only_a,
                neighbors_only_b=neighbors_only_b,
                target_title=meta.get("title", ""),
                target_year=meta.get("pubyear"),
            )
        )

    if not candidate_rows:
        return [], len(eligible)

    if target_uids is not None:
        target_order = {uid: idx for idx, uid in enumerate(target_uids)}
        candidate_rows.sort(
            key=lambda case: (
                target_order.get(case.target_uid, len(target_order)),
                case.target_uid,
            )
        )
    else:
        candidate_rows.sort(
            key=lambda case: (
                -case.shift_score,
                case.rank_jaccard,
                -case.mean_abs_rank_shift,
                case.target_uid,
            )
        )

    return candidate_rows, len(eligible)


def _is_usable_metadata(record: dict[str, Any] | None) -> bool:
    """Return whether a metadata record is useful for semantic review."""
    if not record:
        return False
    title = str(record.get("title", "") or "").strip()
    abstract = str(record.get("abstract", "") or "").strip()
    return bool(title and abstract)


def _usable_metadata_lookup(abstracts: pl.DataFrame | None) -> Dict[str, dict[str, Any]]:
    """Build a UID -> metadata map restricted to review-usable records."""
    if abstracts is None:
        return {}
    uid_meta: Dict[str, dict[str, Any]] = {}
    for row in abstracts.iter_rows(named=True):
        uid = row.get("uid")
        if uid is None:
            continue
        record = dict(row)
        if _is_usable_metadata(record):
            uid_meta[str(uid)] = record
    return uid_meta


def _coverage_state(method_a_reviewable: bool, method_b_reviewable: bool) -> str:
    if method_a_reviewable and method_b_reviewable:
        return "both_reviewable"
    if method_a_reviewable:
        return "A_only_reviewable"
    if method_b_reviewable:
        return "B_only_reviewable"
    return "neither_reviewable"


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 4)


def _reviewable_group(
    uid: str,
    membership: Dict[str, int],
    stats: _MethodStats,
    *,
    n_neighbors: int,
    min_group_size: int,
    min_cluster_size: int,
    metadata_uids: set[str],
) -> tuple[list[str], bool, int]:
    """Return metadata-covered group UIDs and whether the group is reviewable."""
    if uid not in membership:
        return [], False, 0
    cid = membership[uid]
    if stats.cluster_sizes[cid] < min_cluster_size:
        return [], False, 0
    group = _select_group(
        uid,
        membership,
        stats,
        n_neighbors=n_neighbors,
        min_group_size=min_group_size,
        allowed_uids=metadata_uids,
    )
    doc_count = len([group_uid for group_uid in group if group_uid in metadata_uids])
    return group, doc_count >= min_group_size, doc_count


def _diagnostic_strata(case: BoundaryCoverageCase, *, n_neighbors: int, max_group_jaccard: float) -> List[str]:
    """Assign diagnostic strata for mechanism-focused sampling."""
    strata = [case.coverage_state]
    a_sparse = case.method_a_group_fill_rate < 1.0 or case.method_a_total_degree < n_neighbors
    b_sparse = case.method_b_group_fill_rate < 1.0 or case.method_b_total_degree < n_neighbors
    a_dense = case.method_a_group_fill_rate >= 1.0 and case.method_a_total_degree >= n_neighbors
    b_dense = case.method_b_group_fill_rate >= 1.0 and case.method_b_total_degree >= n_neighbors
    if a_sparse and b_dense:
        strata.append("A_sparse_B_dense")
    if b_sparse and a_dense:
        strata.append("B_sparse_A_dense")
    if (
        case.coverage_state == "both_reviewable"
        and case.jaccard <= max_group_jaccard
        and a_dense
        and b_dense
    ):
        strata.append("high_disagreement_both_dense")
    return strata


def collect_boundary_coverage_cases(
    edges_a: pl.DataFrame,
    membership_a: Dict[str, int],
    edges_b: pl.DataFrame,
    membership_b: Dict[str, int],
    *,
    method_a: str = "A",
    method_b: str = "B",
    abstracts: pl.DataFrame | None = None,
    n_neighbors: int = 8,
    min_cluster_size: int = 10,
    target_uids: List[str] | None = None,
    max_group_jaccard: float = 0.5,
) -> tuple[List[BoundaryCoverageCase], int]:
    """Collect full-universe coverage-aware boundary candidates.

    Unlike disagreement sampling, this collector keeps targets where only one
    method forms a reviewable local group, and targets where neither does.
    """
    stats_a = _compute_method_stats(edges_a, membership_a)
    stats_b = _compute_method_stats(edges_b, membership_b)
    uid_meta = _usable_metadata_lookup(abstracts)
    if not uid_meta:
        log.warning("No usable metadata for boundary coverage sampling")
        return [], 0

    metadata_uids = set(uid_meta)
    if target_uids is None:
        ordered_targets = sorted((set(membership_a) | set(membership_b)) & metadata_uids)
    else:
        ordered_targets = [uid for uid in dict.fromkeys(target_uids) if uid in metadata_uids]

    min_group_size = _min_group_size(n_neighbors)
    cases: list[BoundaryCoverageCase] = []
    for uid in ordered_targets:
        group_a, a_reviewable, group_a_docs = _reviewable_group(
            uid,
            membership_a,
            stats_a,
            n_neighbors=n_neighbors,
            min_group_size=min_group_size,
            min_cluster_size=min_cluster_size,
            metadata_uids=metadata_uids,
        )
        group_b, b_reviewable, group_b_docs = _reviewable_group(
            uid,
            membership_b,
            stats_b,
            n_neighbors=n_neighbors,
            min_group_size=min_group_size,
            min_cluster_size=min_cluster_size,
            metadata_uids=metadata_uids,
        )
        set_a = set(group_a)
        set_b = set(group_b)
        union = set_a | set_b
        overlap = set_a & set_b
        jaccard = len(overlap) / len(union) if union else 0.0
        cid_a = membership_a.get(uid)
        cid_b = membership_b.get(uid)
        size_a = int(stats_a.cluster_sizes[cid_a]) if cid_a is not None else 0
        size_b = int(stats_b.cluster_sizes[cid_b]) if cid_b is not None else 0
        min_size = min(size for size in (size_a, size_b) if size > 0) if size_a and size_b else 0
        max_size = max(size_a, size_b)
        case = BoundaryCoverageCase(
            target_uid=uid,
            coverage_state=_coverage_state(a_reviewable, b_reviewable),
            method_a_reviewable=a_reviewable,
            method_b_reviewable=b_reviewable,
            method_a_cluster_id=cid_a,
            method_b_cluster_id=cid_b,
            method_a_cluster_size=size_a,
            method_b_cluster_size=size_b,
            method_a_cross_cluster_ratio=round(stats_a.cross_ratios.get(uid, 0.0), 4),
            method_b_cross_cluster_ratio=round(stats_b.cross_ratios.get(uid, 0.0), 4),
            group_a_uids=group_a,
            group_b_uids=group_b,
            group_a_doc_count=group_a_docs,
            group_b_doc_count=group_b_docs,
            overlap_size=len(overlap),
            jaccard=round(jaccard, 4),
            cluster_size_ratio=_ratio_or_none(max_size, min_size),
            method_a_total_degree=int(stats_a.node_total_degree.get(uid, 0)),
            method_b_total_degree=int(stats_b.node_total_degree.get(uid, 0)),
            method_a_intra_degree=int(stats_a.node_intra_degree.get(uid, 0)),
            method_b_intra_degree=int(stats_b.node_intra_degree.get(uid, 0)),
            method_a_cross_degree=int(stats_a.node_cross_degree.get(uid, 0)),
            method_b_cross_degree=int(stats_b.node_cross_degree.get(uid, 0)),
            method_a_group_fill_rate=round(len(group_a) / max(1, n_neighbors), 4),
            method_b_group_fill_rate=round(len(group_b) / max(1, n_neighbors), 4),
            diagnostic_strata=[],
            target_title=str(uid_meta[uid].get("title", "") or ""),
            target_year=uid_meta[uid].get("pubyear"),
        )
        case.diagnostic_strata = _diagnostic_strata(
            case,
            n_neighbors=n_neighbors,
            max_group_jaccard=max_group_jaccard,
        )
        cases.append(case)

    return cases, len(ordered_targets)


def _sample_cases(cases: list[BoundaryCoverageCase], n_cases: int, rng: np.random.RandomState) -> list[BoundaryCoverageCase]:
    ordered = list(cases)
    rng.shuffle(ordered)
    return ordered[: min(n_cases, len(ordered))]


def _sample_diagnostic_cases(
    cases: list[BoundaryCoverageCase],
    *,
    n_per_stratum: int,
    rng: np.random.RandomState,
) -> list[BoundaryCoverageCase]:
    stratum_order = [
        "A_only_reviewable",
        "B_only_reviewable",
        "both_reviewable",
        "neither_reviewable",
        "A_sparse_B_dense",
        "B_sparse_A_dense",
        "high_disagreement_both_dense",
    ]
    by_stratum: Dict[str, list[BoundaryCoverageCase]] = defaultdict(list)
    for case in cases:
        for stratum in case.diagnostic_strata:
            by_stratum[stratum].append(case)

    selected: list[BoundaryCoverageCase] = []
    seen: set[str] = set()
    for stratum in stratum_order:
        bucket = [case for case in by_stratum.get(stratum, []) if case.target_uid not in seen]
        rng.shuffle(bucket)
        chosen = bucket[: min(n_per_stratum, len(bucket))]
        selected.extend(chosen)
        seen.update(case.target_uid for case in chosen)
    return selected


def sample_boundary_coverage_cases(
    edges_a: pl.DataFrame,
    membership_a: Dict[str, int],
    edges_b: pl.DataFrame,
    membership_b: Dict[str, int],
    *,
    method_a: str = "A",
    method_b: str = "B",
    abstracts: pl.DataFrame | None = None,
    n_neighbors: int = 8,
    min_cluster_size: int = 10,
    n_population_cases: int = 30,
    n_diagnostic_per_stratum: int = 12,
    sample_mode: str = "both",
    target_uids: List[str] | None = None,
    max_group_jaccard: float = 0.5,
    seed: int = 42,
) -> BoundaryCoverageSampleSet:
    """Sample Protocol D v2 population and diagnostic boundary cases."""
    if sample_mode not in {"population", "diagnostic", "both"}:
        raise ValueError(f"unknown sample_mode: {sample_mode}")
    rng = np.random.RandomState(seed)
    cases, n_target_universe = collect_boundary_coverage_cases(
        edges_a,
        membership_a,
        edges_b,
        membership_b,
        method_a=method_a,
        method_b=method_b,
        abstracts=abstracts,
        n_neighbors=n_neighbors,
        min_cluster_size=min_cluster_size,
        target_uids=target_uids,
        max_group_jaccard=max_group_jaccard,
    )
    counts = Counter(case.coverage_state for case in cases)
    population_cases: list[BoundaryCoverageCase] = []
    diagnostic_cases: list[BoundaryCoverageCase] = []
    if sample_mode in {"population", "both"}:
        population_cases = _sample_cases(cases, n_population_cases, rng)
    if sample_mode in {"diagnostic", "both"}:
        diagnostic_cases = _sample_diagnostic_cases(
            cases,
            n_per_stratum=n_diagnostic_per_stratum,
            rng=rng,
        )

    log.info(
        "Sampled Protocol D v2 cases: population=%d diagnostic=%d universe=%d",
        len(population_cases),
        len(diagnostic_cases),
        n_target_universe,
    )
    return BoundaryCoverageSampleSet(
        population_cases=population_cases,
        diagnostic_cases=diagnostic_cases,
        method_a=method_a,
        method_b=method_b,
        n_nodes=len(set(membership_a) | set(membership_b)),
        n_target_universe=n_target_universe,
        coverage_state_counts={
            label: int(counts.get(label, 0))
            for label in (
                "A_only_reviewable",
                "B_only_reviewable",
                "both_reviewable",
                "neither_reviewable",
            )
        },
        sample_mode=sample_mode,
    )


__all__ = [
    "sample_worst_case",
    "sample_disagreement_cases",
    "collect_rank_shift_cases",
    "sample_rank_shift_cases",
    "collect_boundary_coverage_cases",
    "sample_boundary_coverage_cases",
    "SampleSet",
    "SampleCase",
    "DisagreementSampleSet",
    "DisagreementCase",
    "RankShiftSampleSet",
    "RankShiftCase",
    "BoundaryCoverageCase",
    "BoundaryCoverageSampleSet",
]
