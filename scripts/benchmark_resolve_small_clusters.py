"""Benchmark: resolve_small_clusters (new) vs merge_small_clusters (old).

Runs both approaches on field 34 (Veterinary, ~15k nodes) and compares:
  - Clustering quality: cluster count, size distribution, singleton count
  - Computation time: γ search + postprocessing
  - Resolution profile: adaptive coarsening behaviour
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark")

# ── Paths ────────────────────────────────────────────────────────────────
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
import sys
_field = sys.argv[1] if len(sys.argv) > 1 else "field_34"
EDGE_FILE = DATA_ROOT / "linktype_edges" / _field / "bc_assoc_strength.parquet"

# ── Parameters (from LandscapeConfig defaults) ──────────────────────────
MIN_DOCS = 1000
GAMMA_RANGE = (1e-6, 1e-3)
SEED = 42
ITERATIONS = 50


def build_graph_and_runner():
    """Load edges, build graph, extract giant component, create runner."""
    import polars as pl
    from sciscape.clustering.graph import build_graph, giant_component
    from sciscape.clustering.runner import LeidenRunner

    log.info("Loading edges: %s", EDGE_FILE)
    edges = pl.read_parquet(EDGE_FILE)
    # Normalise column names for build_graph (expects uid1, uid2, rel_sum2)
    if "src" in edges.columns:
        edges = edges.rename({"src": "uid1", "dst": "uid2", "weight": "rel_sum2"})
    log.info("  %d edges", len(edges))

    graph = build_graph(edges)
    log.info("  Graph: %d V, %d E", graph.vcount(), graph.ecount())

    giant = giant_component(graph)
    log.info("  Giant component: %d V, %d E", giant.vcount(), giant.ecount())

    runner = LeidenRunner(
        giant, objective="cpm",
        default_seed=SEED, default_iterations=ITERATIONS,
    )
    return giant, runner


def gamma_search(runner):
    """Binary-search γ to maximise natural large clusters (≥ MIN_DOCS)."""
    lo_g, hi_g = np.log10(GAMMA_RANGE[0]), np.log10(GAMMA_RANGE[1])
    coarse_gammas = np.logspace(lo_g, hi_g, num=3)
    cache = {}

    def _eval(gamma):
        t0 = time.perf_counter()
        result = runner.run(gamma)
        mem = list(result.membership)
        sizes = Counter(mem)
        n_large = sum(1 for s in sizes.values() if s >= MIN_DOCS)
        elapsed = time.perf_counter() - t0
        log.info("  γ=%.6f → %d total, %d large (≥%d) [%.1fs]",
                 gamma, len(sizes), n_large, MIN_DOCS, elapsed)
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
            mid_lo = 10 ** ((np.log10(sorted_gammas[idx - 1]) + np.log10(best_gamma)) / 2)
            if mid_lo not in cache:
                probes.append(mid_lo)
        if idx < len(sorted_gammas) - 1:
            mid_hi = 10 ** ((np.log10(best_gamma) + np.log10(sorted_gammas[idx + 1])) / 2)
            if mid_hi not in cache:
                probes.append(mid_hi)
        if not probes:
            break
        for g in probes:
            cache[g] = _eval(g)
        best_gamma = max(cache, key=lambda g: cache[g][0])
        new_best = cache[best_gamma][0]
        if new_best == best_n:
            break
        best_n = new_best

    log.info("  → Best: γ=%.6f → %d large clusters (%d evals)",
             best_gamma, best_n, len(cache))
    return best_gamma, cache[best_gamma][1], cache


def describe_membership(name, membership):
    """Print cluster size distribution stats."""
    sizes = Counter(membership)
    size_vals = sorted(sizes.values(), reverse=True)
    n_clusters = len(sizes)
    n_singletons = sum(1 for s in size_vals if s == 1)
    n_small = sum(1 for s in size_vals if s < MIN_DOCS)
    n_large = sum(1 for s in size_vals if s >= MIN_DOCS)

    log.info("── %s ──", name)
    log.info("  Total clusters: %d", n_clusters)
    log.info("  Large (≥%d): %d", MIN_DOCS, n_large)
    log.info("  Small (<%d): %d (%d nodes)", MIN_DOCS, n_small,
             sum(s for s in size_vals if s < MIN_DOCS))
    log.info("  Singletons: %d", n_singletons)
    log.info("  Top 10 sizes: %s", size_vals[:10])
    log.info("  Bottom 10 sizes: %s", size_vals[-10:])

    pcts = [25, 50, 75, 90, 95, 99]
    for p in pcts:
        log.info("  p%d cluster size: %d", p, int(np.percentile(size_vals, 100 - p)))


def run_old_approach(graph, runner, raw_membership):
    """Old: merge_small_clusters on the raw Leiden output."""
    from sciscape.clustering.postprocess import merge_small_clusters

    log.info("\n══ OLD APPROACH: merge_small_clusters ══")
    t0 = time.perf_counter()
    result = merge_small_clusters(graph, raw_membership, min_size=MIN_DOCS)
    elapsed = time.perf_counter() - t0

    log.info("  merge_small_clusters took %.2fs", elapsed)
    log.info("  %d merge actions", len(result.merges))
    describe_membership("OLD (after merge)", result.membership)
    return result.membership, elapsed


def run_new_approach(runner, raw_membership, best_gamma, coarse_iterations=None):
    """New: resolve_small_clusters via adaptive CPM coarsening."""
    from sciscape.clustering.postprocess import resolve_small_clusters

    label = f"coarse_iterations={coarse_iterations}" if coarse_iterations else "default (2)"
    log.info("\n══ NEW APPROACH: resolve_small_clusters (%s) ══", label)
    t0 = time.perf_counter()
    kwargs = dict(min_size=MIN_DOCS)
    if coarse_iterations is not None:
        kwargs["coarse_iterations"] = coarse_iterations
    result = resolve_small_clusters(
        runner, raw_membership, best_gamma, **kwargs,
    )
    elapsed = time.perf_counter() - t0

    log.info("  resolve_small_clusters took %.2fs", elapsed)
    log.info("  %d small clusters initially (%d nodes)",
             result.n_small_initial, result.n_small_nodes_initial)
    log.info("  %d resolved (%d nodes), %d unresolvable (%d nodes)",
             result.n_clusters_resolved, result.n_nodes_resolved,
             result.n_clusters_unresolvable, result.n_nodes_unresolvable)
    log.info("  %d coarsening levels used: %s",
             len(result.resolutions_used),
             [f"{g:.2e}" for g in result.resolutions_used])

    # Mark remaining singletons as undetermined
    sizes_final = Counter(result.membership)
    undetermined = sum(1 for c in result.membership if sizes_final[c] == 1)
    log.info("  Remaining singletons (→ undetermined): %d (%.2f%%)",
             undetermined, undetermined / len(result.membership) * 100)

    describe_membership("NEW (after resolve)", result.membership)
    return result.membership, elapsed, result


def run_refine_approach(runner, raw_membership, best_gamma):
    """Newest: refine_clusters (split + merge loop)."""
    from sciscape.clustering.postprocess import refine_clusters

    log.info("\n══ REFINE APPROACH: split + merge loop ══")
    t0 = time.perf_counter()
    result = refine_clusters(
        runner, raw_membership, best_gamma, min_size=MIN_DOCS,
    )
    elapsed = time.perf_counter() - t0

    log.info("  refine_clusters took %.2fs (%d rounds)", elapsed, result.n_rounds)
    for i, sr in enumerate(result.split_results):
        log.info("  Round %d split: %d clusters split, %d new sub-clusters",
                 i + 1, sr.n_clusters_split, sr.n_new_clusters_created)
    for i, mr in enumerate(result.merge_results):
        log.info("  Round %d merge: %d resolved (%d nodes), %d unresolvable",
                 i + 1, mr.n_clusters_resolved, mr.n_nodes_resolved,
                 mr.n_clusters_unresolvable)

    sizes_final = Counter(result.membership)
    undetermined = sum(1 for c in result.membership if sizes_final[c] == 1)
    log.info("  Remaining singletons (→ undetermined): %d (%.2f%%)",
             undetermined, undetermined / len(result.membership) * 100)

    describe_membership("REFINE (after split+merge)", result.membership)
    return result.membership, elapsed, result


def compare_approaches(old_mem, new_mem, n_nodes):
    """Side-by-side comparison."""
    from collections import Counter

    old_sizes = Counter(old_mem)
    new_sizes = Counter(new_mem)

    log.info("\n══ COMPARISON ══")
    log.info("  %-25s %8s %8s", "", "OLD", "NEW")
    log.info("  %-25s %8d %8d", "Total clusters", len(old_sizes), len(new_sizes))
    log.info("  %-25s %8d %8d", "Large clusters (≥min_docs)",
             sum(1 for s in old_sizes.values() if s >= MIN_DOCS),
             sum(1 for s in new_sizes.values() if s >= MIN_DOCS))
    log.info("  %-25s %8d %8d", "Small clusters",
             sum(1 for s in old_sizes.values() if s < MIN_DOCS),
             sum(1 for s in new_sizes.values() if s < MIN_DOCS))
    log.info("  %-25s %8d %8d", "Singletons",
             sum(1 for s in old_sizes.values() if s == 1),
             sum(1 for s in new_sizes.values() if s == 1))

    # Nodes in large clusters
    old_in_large = sum(s for s in old_sizes.values() if s >= MIN_DOCS)
    new_in_large = sum(s for s in new_sizes.values() if s >= MIN_DOCS)
    log.info("  %-25s %8d %8d", "Nodes in large clusters", old_in_large, new_in_large)
    log.info("  %-25s %7.1f%% %7.1f%%", "Coverage (large)",
             old_in_large / n_nodes * 100, new_in_large / n_nodes * 100)


def main():
    log.info("═══ Benchmark: resolve_small_clusters vs merge_small_clusters ═══")
    log.info("%s, min_docs=%d", _field, MIN_DOCS)

    # Build graph
    t0_total = time.perf_counter()
    graph, runner = build_graph_and_runner()
    n_nodes = graph.vcount()

    # γ search (shared between old and new)
    log.info("\n── γ search (shared) ──")
    t0_search = time.perf_counter()
    best_gamma, raw_membership, cache = gamma_search(runner)
    search_elapsed = time.perf_counter() - t0_search
    log.info("  γ search total: %.2fs", search_elapsed)

    describe_membership("RAW Leiden output", raw_membership)

    # Run both approaches on same raw membership
    old_mem, old_time = run_old_approach(graph, runner, raw_membership)

    # Merge-only approach
    new_mem, new_time, new_result = run_new_approach(runner, raw_membership, best_gamma)

    # Refine approach (split + merge loop)
    ref_mem, ref_time, ref_result = run_refine_approach(runner, raw_membership, best_gamma)

    compare_approaches(old_mem, new_mem, n_nodes)
    log.info("\n  %-25s %7.2fs %7.2fs %7.2fs", "Postprocess time",
             old_time, new_time, ref_time)
    log.info("  %-25s %7.2fs %7.2fs %7.2fs", "Total (search + post)",
             search_elapsed + old_time, search_elapsed + new_time,
             search_elapsed + ref_time)

    # Extra: compare refine vs merge-only
    log.info("\n══ REFINE vs MERGE-ONLY ══")
    ref_sizes = Counter(ref_mem)
    new_sizes = Counter(new_mem)
    log.info("  %-25s %8s %8s", "", "MERGE", "REFINE")
    log.info("  %-25s %8d %8d", "Total clusters",
             len(new_sizes), len(ref_sizes))
    log.info("  %-25s %8d %8d", "Max cluster size",
             max(new_sizes.values()), max(ref_sizes.values()))
    log.info("  %-25s %8d %8d", "Large (≥min_docs)",
             sum(1 for s in new_sizes.values() if s >= MIN_DOCS),
             sum(1 for s in ref_sizes.values() if s >= MIN_DOCS))
    log.info("  %-25s %8d %8d", "> 2×min_docs",
             sum(1 for s in new_sizes.values() if s > 2 * MIN_DOCS),
             sum(1 for s in ref_sizes.values() if s > 2 * MIN_DOCS))

    total_elapsed = time.perf_counter() - t0_total
    log.info("\nTotal benchmark time: %.1fs", total_elapsed)


if __name__ == "__main__":
    main()
