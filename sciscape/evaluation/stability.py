"""Clustering stability evaluation via multi-seed comparison.

Runs Leiden with different random seeds and measures agreement
between resulting partitions using AMI and ARI.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


@dataclass
class StabilityResult:
    """Result of stability evaluation."""
    n_seeds: int
    ami_mean: float
    ami_std: float
    ari_mean: float
    ari_std: float
    n_clusters_per_seed: List[int]
    max_pct_per_seed: List[float]
    elapsed: float

    def summary(self) -> str:
        return (
            f"Stability ({self.n_seeds} seeds): "
            f"AMI={self.ami_mean:.3f}±{self.ami_std:.3f}, "
            f"ARI={self.ari_mean:.3f}±{self.ari_std:.3f}, "
            f"clusters={min(self.n_clusters_per_seed)}-{max(self.n_clusters_per_seed)}"
        )


@dataclass
class QualityReport:
    """Summary quality metrics for a clustering result."""
    n_nodes: int
    n_edges: int
    n_clusters: int
    max_cluster_pct: float
    target_pct: float
    gamma: float
    singleton_pct: float  # % of clusters with size 1
    top5_sizes: List[int]
    consensus_edge_pct: Dict[int, float]  # {n_layers: pct_of_edges}
    stability: StabilityResult | None = None

    def summary(self) -> str:
        lines = [
            f"Quality Report",
            f"  Nodes: {self.n_nodes:,}, Edges: {self.n_edges:,}",
            f"  Clusters: {self.n_clusters}, γ={self.gamma:.2e}",
            f"  Max cluster: {self.max_cluster_pct:.1f}% (target: <{self.target_pct}%)",
            f"  Singletons: {self.singleton_pct:.1f}%",
            f"  Top-5 sizes: {self.top5_sizes}",
        ]
        if self.consensus_edge_pct:
            lines.append("  Consensus edges:")
            for nl, pct in sorted(self.consensus_edge_pct.items()):
                lines.append(f"    {nl}-layer: {pct:.1f}%")
        if self.stability:
            lines.append(f"  {self.stability.summary()}")
        return "\n".join(lines)


def evaluate_stability(
    edges: pl.DataFrame,
    *,
    gamma: float,
    n_seeds: int = 5,
    min_size: int = 10,
    postprocess: bool = True,
    progress: callable | None = None,
) -> StabilityResult:
    """Evaluate clustering stability by running Leiden with multiple seeds.

    Parameters
    ----------
    edges : pl.DataFrame
        Edge table (uid1, uid2, rel_sum2).
    gamma : float
        Resolution parameter.
    n_seeds : int
        Number of random seeds to test.
    min_size : int
        Minimum cluster size for postprocessing.
    postprocess : bool
        Whether to postprocess small clusters.

    Returns
    -------
    StabilityResult
    """
    try:
        from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    except ImportError:
        raise ImportError("scikit-learn required for stability evaluation: pip install scikit-learn")
    from ..clustering.integer_remap import integer_remap_memory
    from ..clustering.leiden_rust import (
        RUST_AVAILABLE,
        build_leiden_graph,
        postprocess_small_clusters_rust,
        run_leiden_rust,
    )

    if not RUST_AVAILABLE:
        raise ImportError("Rust backend required for stability evaluation")
    if edges.height == 0:
        raise ValueError("Cannot evaluate stability: edges DataFrame is empty")
    if n_seeds < 2:
        raise ValueError("n_seeds must be >= 2 for pairwise comparison")

    def _log(msg):
        log.info(msg)
        if progress:
            progress(msg)

    t0 = time.perf_counter()
    src, dst, w, n_nodes, _uids = integer_remap_memory(edges)
    try:
        graph = build_leiden_graph(
            edges_src=src,
            edges_dst=dst,
            edges_weight=w,
            n_nodes=n_nodes,
        )
    except AttributeError:
        graph = None

    memberships = []
    n_clusters_list = []
    max_pct_list = []

    for seed in range(n_seeds):
        if graph is not None:
            r = graph.run_leiden(
                resolution=gamma, seed=seed, n_iterations=10,
            )
        else:
            r = run_leiden_rust(
                edges_src=src, edges_dst=dst, edges_weight=w,
                resolution=gamma, n_nodes=n_nodes, seed=seed, n_iterations=10,
            )
        mem = r.membership
        if postprocess:
            if graph is not None:
                p = graph.postprocess_small_clusters(
                    resolution=gamma, min_size=min_size,
                    membership=mem,
                    seed=seed,
                    gamma_decay=0.5, max_rounds=3,
                    use_greedy=True, use_component_merge=True,
                )
            else:
                p = postprocess_small_clusters_rust(
                    resolution=gamma, min_size=min_size,
                    membership=mem,
                    edges_src=src, edges_dst=dst, edges_weight=w,
                    n_nodes=n_nodes, seed=seed,
                    gamma_decay=0.5, max_rounds=3,
                    use_greedy=True, use_component_merge=True,
                )
            mem = p.membership

        memberships.append(mem)
        size_arr = np.bincount(np.asarray(mem, dtype=np.int32))
        size_arr = size_arr[size_arr > 0]
        n_clusters_list.append(len(size_arr))
        max_pct_list.append(100 * int(size_arr.max()) / max(n_nodes, 1))
        _log(f"  seed={seed}: {len(size_arr)} clusters, max={max_pct_list[-1]:.1f}%")

    # Pairwise AMI/ARI
    ami_scores = []
    ari_scores = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            ami_scores.append(adjusted_mutual_info_score(memberships[i], memberships[j]))
            ari_scores.append(adjusted_rand_score(memberships[i], memberships[j]))

    elapsed = time.perf_counter() - t0
    result = StabilityResult(
        n_seeds=n_seeds,
        ami_mean=float(np.mean(ami_scores)),
        ami_std=float(np.std(ami_scores)),
        ari_mean=float(np.mean(ari_scores)),
        ari_std=float(np.std(ari_scores)),
        n_clusters_per_seed=n_clusters_list,
        max_pct_per_seed=max_pct_list,
        elapsed=round(elapsed, 1),
    )
    _log(result.summary())
    return result


def compute_quality_report(
    edges: pl.DataFrame,
    membership: np.ndarray,
    *,
    gamma: float,
    target_pct: float = 3.0,
    layer_tables: Dict[str, pl.DataFrame] | None = None,
    stability: StabilityResult | None = None,
) -> QualityReport:
    """Compute summary quality metrics for a clustering result.

    Parameters
    ----------
    edges : pl.DataFrame
        Combined edge table.
    membership : np.ndarray
        Cluster assignments (one per node).
    gamma : float
        Resolution parameter used.
    target_pct : float
        Target max cluster percentage.
    layer_tables : dict, optional
        Per-layer edge tables for consensus analysis.
    stability : StabilityResult, optional
        Pre-computed stability result.
    """
    n_nodes = len(membership)
    n_edges = edges.height
    size_arr = np.bincount(np.asarray(membership, dtype=np.int32))
    size_arr_nz = size_arr[size_arr > 0]
    n_clusters = len(size_arr_nz)
    max_pct = 100 * int(size_arr_nz.max()) / max(n_nodes, 1) if n_clusters > 0 else 0
    singleton_pct = 100 * int((size_arr_nz == 1).sum()) / max(n_clusters, 1)
    top5 = sorted(size_arr_nz.tolist(), reverse=True)[:5]

    # Consensus edge distribution
    consensus_pct: Dict[int, float] = {}
    if layer_tables and len(layer_tables) > 1:
        tagged = []
        for name, df in layer_tables.items():
            if df.height > 0:
                normed = df.select(
                    pl.min_horizontal("uid1", "uid2").alias("_lo"),
                    pl.max_horizontal("uid1", "uid2").alias("_hi"),
                )
                tagged.append(normed)
        if tagged:
            all_tagged = pl.concat(tagged)
            pair_counts = all_tagged.group_by(["_lo", "_hi"]).len(name="_n")
            total = pair_counts.height
            if total > 0:
                dist = pair_counts.group_by("_n").len(name="_cnt")
                for row in dist.iter_rows():
                    consensus_pct[row[0]] = round(100 * row[1] / total, 1)

    return QualityReport(
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_clusters=n_clusters,
        max_cluster_pct=round(max_pct, 1),
        target_pct=target_pct,
        gamma=gamma,
        singleton_pct=round(singleton_pct, 1),
        top5_sizes=top5,
        consensus_edge_pct=consensus_pct,
        stability=stability,
    )


__all__ = [
    "evaluate_stability", "StabilityResult",
    "compute_quality_report", "QualityReport",
]
