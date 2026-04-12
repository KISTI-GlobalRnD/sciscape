"""Worst-case cluster quality sampler.

Strategy: pick nodes that are hardest for clustering to get right,
then sample their cluster neighbors from easy → hard.

Hard nodes = high cross-cluster edge ratio (boundary nodes).
Hard neighbors = far from target in the cluster (weakest link).
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


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
        if not neighbors:
            continue

        # Sort neighbors by weight
        sorted_nbrs = sorted(neighbors.items(), key=lambda x: -x[1])
        easy = [n for n, _ in sorted_nbrs[:n_easy]]
        hard = [n for n, _ in sorted_nbrs[-n_hard:]] if len(sorted_nbrs) > n_easy else []

        meta = uid_meta.get(uid, {})
        cases.append(SampleCase(
            target_uid=uid,
            cluster_id=cid,
            cluster_size=cluster_sizes[cid],
            easy_neighbors=easy,
            hard_neighbors=hard,
            cross_cluster_ratio=round(cross_ratios[uid], 4),
            n_cross_edges=int(node_cross.get(uid, 0)),
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


__all__ = ["sample_worst_case", "SampleSet", "SampleCase"]
