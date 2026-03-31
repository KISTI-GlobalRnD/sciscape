"""Benchmark: gamma_search speed (old inline vs new optimised).

Compares on the same field:
  OLD: 3 coarse + 4 refine, iter=50, no warm-start  (current landscape.py)
  NEW: 3 coarse + 2 refine, iter=10 + warm-start + final iter=50
"""

from __future__ import annotations

import logging
import sys
import time
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bench_gamma")

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
_field = sys.argv[1] if len(sys.argv) > 1 else "field_34"
EDGE_FILE = DATA_ROOT / "linktype_edges" / _field / "bc_assoc_strength.parquet"
MIN_DOCS = 1000
SEED = 42
ITERATIONS = 50
GAMMA_RANGE = (1e-6, 1e-3)


def build_runner():
    import polars as pl
    from sciscape.clustering.graph import build_graph, giant_component
    from sciscape.clustering.runner import LeidenRunner

    log.info("Loading: %s", EDGE_FILE)
    edges = pl.read_parquet(EDGE_FILE)
    if "src" in edges.columns:
        edges = edges.rename({"src": "uid1", "dst": "uid2", "weight": "rel_sum2"})
    graph = build_graph(edges)
    giant = giant_component(graph)
    log.info("Graph: %d V, %d E", giant.vcount(), giant.ecount())
    runner = LeidenRunner(
        giant, objective="cpm",
        default_seed=SEED, default_iterations=ITERATIONS,
    )
    return giant, runner


def old_gamma_search(runner):
    """Old approach: 3 coarse + 4 refine, iter=50, no warm-start."""
    import numpy as np

    log.info("\n══ OLD γ search (iter=50, no warm-start, 4 refine) ══")
    t0 = time.perf_counter()

    lo_g, hi_g = np.log10(GAMMA_RANGE[0]), np.log10(GAMMA_RANGE[1])
    coarse_gammas = np.logspace(lo_g, hi_g, num=3)
    cache = {}

    def _eval(gamma):
        t1 = time.perf_counter()
        result = runner.run(gamma)
        mem = list(result.membership)
        sizes = Counter(mem)
        n_large = sum(1 for s in sizes.values() if s >= MIN_DOCS)
        elapsed = time.perf_counter() - t1
        log.info("  γ=%.4e → %d total, %d large [%.1fs]",
                 gamma, len(sizes), n_large, elapsed)
        return n_large, mem

    for g in coarse_gammas:
        cache[g] = _eval(g)

    best_gamma = max(cache, key=lambda g: cache[g][0])
    best_n = cache[best_gamma][0]

    for _ in range(4):
        sorted_gammas = sorted(cache.keys())
        idx = sorted_gammas.index(best_gamma)
        probes = []
        if idx > 0:
            mid = 10 ** ((np.log10(sorted_gammas[idx - 1]) + np.log10(best_gamma)) / 2)
            if mid not in cache:
                probes.append(mid)
        if idx < len(sorted_gammas) - 1:
            mid = 10 ** ((np.log10(best_gamma) + np.log10(sorted_gammas[idx + 1])) / 2)
            if mid not in cache:
                probes.append(mid)
        if not probes:
            break
        for g in probes:
            cache[g] = _eval(g)
        best_gamma = max(cache, key=lambda g: cache[g][0])
        new_best = cache[best_gamma][0]
        if new_best == best_n:
            break
        best_n = new_best

    elapsed = time.perf_counter() - t0
    log.info("  → Best: γ=%.4e, %d large (%d evals, %.1fs)",
             best_gamma, best_n, len(cache), elapsed)
    return best_gamma, cache[best_gamma][1], elapsed, len(cache)


def new_gamma_search(runner):
    """New approach: gamma_search() with iter=10, warm-start, 2 refine + final."""
    from sciscape.clustering.postprocess import gamma_search

    log.info("\n══ NEW γ search (iter=10, warm-start, 2 refine + final) ══")
    t0 = time.perf_counter()

    result = gamma_search(
        runner,
        gamma_range=GAMMA_RANGE,
        min_size=MIN_DOCS,
        search_iterations=10,
        max_refine=2,
        warm_start=True,
    )

    # Final full run
    log.info("  Running final at γ=%.4e (iter=%d)...", result.best_gamma, ITERATIONS)
    t1 = time.perf_counter()
    final = runner.run(result.best_gamma)
    final_mem = list(final.membership)
    final_elapsed = time.perf_counter() - t1
    log.info("  Final run: %.1fs", final_elapsed)

    total = time.perf_counter() - t0
    sizes = Counter(final_mem)
    n_large = sum(1 for s in sizes.values() if s >= MIN_DOCS)
    log.info("  → Best: γ=%.4e, %d large (search) → %d large (final), "
             "%d evals + 1 final, %.1fs total",
             result.best_gamma, result.n_large, n_large,
             result.n_evals, total)
    return result.best_gamma, final_mem, total, result.n_evals + 1


def compare(old_gamma, old_mem, old_time, old_evals,
            new_gamma, new_mem, new_time, new_evals):
    old_sizes = Counter(old_mem)
    new_sizes = Counter(new_mem)

    log.info("\n══ COMPARISON ══")
    log.info("  %-25s %12s %12s", "", "OLD", "NEW")
    log.info("  %-25s %12.4e %12.4e", "Best γ", old_gamma, new_gamma)
    log.info("  %-25s %12d %12d", "Large clusters (≥min)",
             sum(1 for s in old_sizes.values() if s >= MIN_DOCS),
             sum(1 for s in new_sizes.values() if s >= MIN_DOCS))
    log.info("  %-25s %12d %12d", "Total clusters",
             len(old_sizes), len(new_sizes))
    log.info("  %-25s %12d %12d", "Max cluster size",
             max(old_sizes.values()), max(new_sizes.values()))
    log.info("  %-25s %12d %12d", "Evals", old_evals, new_evals)
    log.info("  %-25s %11.1fs %11.1fs", "Time", old_time, new_time)
    log.info("  %-25s %11.1f× %11s", "Speedup",
             old_time / new_time if new_time > 0 else float("inf"), "")


def main():
    log.info("═══ Benchmark: γ search speed ═══")
    log.info("%s, min_docs=%d, gamma_range=%s", _field, MIN_DOCS, GAMMA_RANGE)

    _, runner = build_runner()

    old_gamma, old_mem, old_time, old_evals = old_gamma_search(runner)
    new_gamma, new_mem, new_time, new_evals = new_gamma_search(runner)

    compare(old_gamma, old_mem, old_time, old_evals,
            new_gamma, new_mem, new_time, new_evals)


if __name__ == "__main__":
    main()
