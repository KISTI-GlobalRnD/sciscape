"""Automatic γ selection for combined multi-layer edges.

Finds γ such that max cluster size < target percentage,
using binary search on log-scale with Rust Leiden.

Usage:
    from sciscape.clustering.auto_gamma import find_gamma
    gamma, result = find_gamma(edges_df, target_max_pct=3.0)
"""

from __future__ import annotations

import logging
import math
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from .leiden_rust import run_leiden_rust, postprocess_small_clusters_rust
from .integer_remap import integer_remap

log = logging.getLogger(__name__)


@dataclass
class GammaProbe:
    """Result of one γ probe."""
    gamma: float
    n_clusters: int
    max_size: int
    max_pct: float
    top5: List[int]
    elapsed: float


@dataclass
class AutoGammaResult:
    """Result of automatic γ search."""
    gamma: float
    n_clusters: int
    max_pct: float
    top5: List[int]
    probes: List[GammaProbe]
    membership: np.ndarray | None = None


def find_gamma(
    edges: pl.DataFrame,
    *,
    target_max_pct: float = 3.0,
    gamma_range: Tuple[float, float] = (1e-7, 1e-2),
    n_coarse: int = 6,
    max_refine: int = 3,
    min_size: int = 100,
    seed: int = 42,
    n_iterations: int = 10,
    postprocess: bool = True,
    progress: callable | None = None,
) -> AutoGammaResult:
    """Find γ where max cluster < target_max_pct% of total nodes.

    Strategy:
    1. Coarse log-spaced scan (n_coarse probes)
    2. Binary refinement between best bracket
    3. Return γ with max_pct closest to (but below) target

    Parameters
    ----------
    edges : pl.DataFrame
        Combined edge table (uid1, uid2, rel_sum2).
    target_max_pct : float
        Target: max cluster should be < this % of total nodes.
    gamma_range : tuple
        (lo, hi) bounds for γ search in log scale.
    n_coarse : int
        Number of initial log-spaced probes.
    max_refine : int
        Maximum binary refinement steps.
    min_size : int
        Postprocess min cluster size.
    postprocess : bool
        Whether to run postprocess after Leiden.

    Returns
    -------
    AutoGammaResult
    """
    def _log(msg):
        log.info(msg)
        if progress:
            progress(msg)

    # Prepare edges once
    with tempfile.TemporaryDirectory() as td:
        remap = integer_remap(edges, Path(td) / "remap")
        ie = pl.read_parquet(remap.int_edges_path)
        src = ie["src"].to_numpy().astype(np.uint32)
        dst = ie["dst"].to_numpy().astype(np.uint32)
        w = ie["weight"].to_numpy().astype(np.float64)
        n_nodes = remap.n_nodes

        n_total = n_nodes
        probes: List[GammaProbe] = []
        cache: Dict[float, GammaProbe] = {}

        def _probe(gamma: float) -> GammaProbe:
            if gamma in cache:
                return cache[gamma]
            t0 = time.perf_counter()
            r = run_leiden_rust(
                edges_src=src, edges_dst=dst, edges_weight=w,
                resolution=gamma, n_nodes=n_nodes, seed=seed,
                n_iterations=n_iterations,
            )
            mem = r.membership
            if postprocess:
                p = postprocess_small_clusters_rust(
                    resolution=gamma, min_size=min_size,
                    membership=mem,
                    edges_src=src, edges_dst=dst, edges_weight=w,
                    n_nodes=n_nodes, seed=seed,
                    gamma_decay=0.5, max_rounds=5,
                    use_greedy=True, use_component_merge=True,
                )
                mem = p.membership
                n_cl = p.n_clusters
            else:
                n_cl = r.n_clusters

            sizes = Counter(mem.tolist())
            mx = max(sizes.values())
            top5 = sorted(sizes.values(), reverse=True)[:5]
            elapsed = time.perf_counter() - t0
            pct = 100 * mx / n_total

            probe = GammaProbe(
                gamma=gamma, n_clusters=n_cl,
                max_size=mx, max_pct=round(pct, 2),
                top5=top5, elapsed=round(elapsed, 1),
            )
            probe._membership = mem  # stash for best result
            cache[gamma] = probe
            probes.append(probe)
            _log(f"  γ={gamma:.2e}: {n_cl} cl, max={mx} ({pct:.1f}%), {elapsed:.0f}s")
            return probe

        # Phase 1: coarse log-spaced scan
        lo_log = math.log10(gamma_range[0])
        hi_log = math.log10(gamma_range[1])
        coarse_gammas = [10 ** (lo_log + i * (hi_log - lo_log) / max(n_coarse - 1, 1))
                         for i in range(n_coarse)]

        _log(f"auto_gamma: target max < {target_max_pct}%, "
             f"range [{gamma_range[0]:.0e}, {gamma_range[1]:.0e}], {n_coarse} probes")

        for g in coarse_gammas:
            _probe(g)

        # Phase 2: binary refinement
        # Find bracket: largest γ where max_pct > target (too few clusters)
        #               smallest γ where max_pct <= target (enough clusters)
        sorted_probes = sorted(cache.values(), key=lambda p: p.gamma)

        for _ in range(max_refine):
            # Find best bracket
            below = [p for p in sorted_probes if p.max_pct <= target_max_pct]
            above = [p for p in sorted_probes if p.max_pct > target_max_pct]

            if not below:
                # All above target — need higher γ
                highest = sorted_probes[-1]
                new_g = highest.gamma * 10
                if new_g > gamma_range[1] * 100:
                    break
                _probe(new_g)
                sorted_probes = sorted(cache.values(), key=lambda p: p.gamma)
                continue

            if not above:
                # All below target — γ is already fine, pick lowest
                break

            # Bracket: highest above, lowest below
            hi_probe = max(above, key=lambda p: p.gamma)
            lo_probe = min(below, key=lambda p: p.gamma)

            if lo_probe.gamma / hi_probe.gamma < 1.5:
                break  # close enough

            mid_g = 10 ** ((math.log10(hi_probe.gamma) + math.log10(lo_probe.gamma)) / 2)
            _probe(mid_g)
            sorted_probes = sorted(cache.values(), key=lambda p: p.gamma)

        # Select best: closest to target from below
        candidates = [p for p in cache.values() if p.max_pct <= target_max_pct]
        if candidates:
            best = max(candidates, key=lambda p: p.max_pct)  # closest to target
        else:
            best = min(cache.values(), key=lambda p: p.max_pct)  # least bad

        _log(f"auto_gamma: selected γ={best.gamma:.2e} "
             f"({best.n_clusters} cl, max={best.max_pct}%)")

        return AutoGammaResult(
            gamma=best.gamma,
            n_clusters=best.n_clusters,
            max_pct=best.max_pct,
            top5=best.top5,
            probes=probes,
            membership=getattr(best, '_membership', None),
        )


__all__ = ["find_gamma", "AutoGammaResult", "GammaProbe"]
