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
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import polars as pl

from .integer_remap import RemapResult
from .runner import RustLeidenRunner

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
    gamma_range: Tuple[float, float] | str = "auto",
    n_coarse: int | str = "auto",
    max_refine: int | str = "auto",
    min_size: int = 100,
    seed: int = 42,
    n_iterations: int = 10,
    postprocess: bool = True,
    progress: callable | None = None,
    remap: RemapResult | None = None,
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

    # Input validation
    required_cols = {"uid1", "uid2", "rel_sum2"}
    missing = required_cols - set(edges.columns)
    if missing:
        raise ValueError(f"edges DataFrame missing columns: {missing}")
    if edges.height == 0:
        return AutoGammaResult(gamma=1.0, n_clusters=0, max_pct=100.0, top5=[], probes=[])
    if isinstance(gamma_range, tuple) and gamma_range[0] >= gamma_range[1]:
        raise ValueError(f"gamma_range[0] must be < gamma_range[1], got {gamma_range}")

    # Compute n_est once for reuse
    n_est = pl.concat([edges["uid1"], edges["uid2"]]).n_unique()

    # Auto gamma_range based on edge weight distribution + graph density
    if gamma_range == "auto":
        w_arr = edges["rel_sum2"].to_numpy()
        n_edges = len(w_arr)
        w_median = float(np.median(w_arr))
        w_max = float(w_arr.max())

        # Graph density: edge_count / (n*(n-1)/2)
        max_possible = n_est * (n_est - 1) / 2 if n_est > 1 else 1
        density = n_edges / max_possible

        # CPM quality function: H = Σ_c [e_c - γ * (n_c choose 2)]
        # γ must be high enough to split dense subgraphs.
        # For dense graphs: push γ_lo higher so CPM can actually split.
        #
        # Heuristic:
        #   - γ_lo = w_median / n (base) × density_boost
        #   - γ_hi = w_max × 10
        #   - density_boost: dense graphs (>5%) get 10× higher floor
        density_boost = 1.0 + 50.0 * density  # density=0.1 → 6×, density=0.01 → 1.5×

        gamma_lo = max(1e-8, w_median / max(n_est, 1) * density_boost)
        gamma_hi = max(w_max * 10, gamma_lo * 1e4)

        # For small graphs (<1000 nodes), tighten the range
        # to avoid wasting probes on extreme γ values
        if n_est < 1000:
            gamma_lo = max(gamma_lo, w_median * 0.01)
            gamma_hi = min(gamma_hi, w_max * 100)

        gamma_range = (gamma_lo, gamma_hi)
        log.info("auto gamma_range: [%.2e, %.2e] "
                 "(median=%.4f, max=%.2f, n=%d, density=%.4f, boost=%.1f)",
                 gamma_lo, gamma_hi, w_median, w_max, n_est, density, density_boost)

    # Adaptive probe count: more probes for small graphs (narrow range)
    if n_coarse == "auto":
        n_coarse = 12 if n_est < 1000 else 6
    if max_refine == "auto":
        max_refine = 5 if n_est < 1000 else 3

    # Prepare graph once and reuse the Rust CSR handle across all probes.
    if remap is not None:
        n_nodes = remap.n_nodes
        rust_runner = RustLeidenRunner.from_edge_path(
            str(remap.int_edges_path),
            n_nodes,
            default_iterations=n_iterations,
            default_seed=seed,
        )
    else:
        from .integer_remap import integer_remap_memory
        src, dst, w, n_nodes, _uids = integer_remap_memory(edges)
        rust_runner = RustLeidenRunner(
            src, dst, w, n_nodes,
            default_iterations=n_iterations,
            default_seed=seed,
        )

    n_total = n_nodes
    probes: List[GammaProbe] = []
    cache: Dict[float, GammaProbe] = {}
    import threading
    _cache_lock = threading.Lock()

    def _probe(gamma: float, *, do_postprocess: bool = True) -> GammaProbe:
        with _cache_lock:
            if gamma in cache:
                return cache[gamma]
        t0 = time.perf_counter()
        r = rust_runner.run_array(gamma, seed=seed, n_iterations=n_iterations)
        mem = r.membership
        if postprocess and do_postprocess:
            p = rust_runner.postprocess(
                resolution=gamma, min_size=min_size,
                membership=mem,
                seed=seed,
                gamma_decay=0.5, max_rounds=5,
                use_greedy=True, use_component_merge=True,
            )
            mem = p.membership
            n_cl = p.n_clusters
        else:
            n_cl = r.n_clusters

        size_arr = np.bincount(mem.astype(np.int32))
        if len(size_arr) == 0:
            log.warning("γ=%.2e: empty clustering, skipping", gamma)
            return None
        mx = int(size_arr.max())
        top5 = sorted(size_arr[size_arr > 0].tolist(), reverse=True)[:5]
        elapsed = time.perf_counter() - t0
        pct = 100 * mx / max(n_total, 1)

        probe = GammaProbe(
            gamma=gamma, n_clusters=int((size_arr > 0).sum()),
            max_size=mx, max_pct=round(pct, 2),
            top5=top5, elapsed=round(elapsed, 1),
        )
        probe._membership = mem  # stash for best result
        with _cache_lock:
            cache[gamma] = probe
            probes.append(probe)
        _log(f"  γ={gamma:.2e}: {probe.n_clusters} cl, max={mx} ({pct:.1f}%), {elapsed:.0f}s")
        return probe

    # Phase 1: coarse log-spaced scan
    lo_log = math.log10(gamma_range[0])
    hi_log = math.log10(gamma_range[1])
    coarse_gammas = [10 ** (lo_log + i * (hi_log - lo_log) / max(n_coarse - 1, 1))
                     for i in range(n_coarse)]

    _log(f"auto_gamma: target max < {target_max_pct}%, "
         f"range [{gamma_range[0]:.0e}, {gamma_range[1]:.0e}], {n_coarse} probes")

    # Parallel coarse scan WITHOUT postprocess (fast bracketing)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    n_workers = min(4, n_coarse)
    if n_workers > 1 and n_coarse > 2:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_probe, g, do_postprocess=False): g for g in coarse_gammas}
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    g = futures[future]
                    _log(f"  γ={g:.2e}: skipped (empty clustering)")
    else:
        for g in coarse_gammas:
            result = _probe(g, do_postprocess=False)
            if result is None:
                _log(f"  γ={g:.2e}: skipped (empty clustering)")

    # Phase 2: binary refinement
    if not cache:
        _log("auto_gamma: all probes failed, returning midpoint γ")
        mid_g = 10 ** ((lo_log + hi_log) / 2)
        return AutoGammaResult(
            gamma=mid_g, n_clusters=1, max_pct=100.0, top5=[n_total], probes=probes,
        )

    sorted_probes = sorted(cache.values(), key=lambda p: p.gamma)

    for _ in range(max_refine):
        below = [p for p in sorted_probes if p.max_pct <= target_max_pct]
        above = [p for p in sorted_probes if p.max_pct > target_max_pct]

        if not below:
            highest = sorted_probes[-1]
            new_g = highest.gamma * 10
            if new_g > gamma_range[1] * 100:
                break
            _probe(new_g)
            sorted_probes = sorted(cache.values(), key=lambda p: p.gamma)
            continue

        if not above:
            break

        hi_probe = max(above, key=lambda p: p.gamma)
        lo_probe = min(below, key=lambda p: p.gamma)

        if lo_probe.gamma / hi_probe.gamma < 1.5:
            break

        mid_g = 10 ** ((math.log10(hi_probe.gamma) + math.log10(lo_probe.gamma)) / 2)
        _probe(mid_g, do_postprocess=False)
        sorted_probes = sorted(cache.values(), key=lambda p: p.gamma)

    # Select best: closest to target from below
    candidates = [p for p in cache.values() if p.max_pct <= target_max_pct]
    if candidates:
        best = max(candidates, key=lambda p: p.max_pct)
    else:
        best = min(cache.values(), key=lambda p: p.max_pct)

    # Re-run best γ WITH postprocess for final membership
    if postprocess:
        best = _probe(best.gamma, do_postprocess=True)
        if best is None:
            best = min(cache.values(), key=lambda p: p.max_pct)

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
