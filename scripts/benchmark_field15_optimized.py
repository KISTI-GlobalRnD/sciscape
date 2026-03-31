"""Benchmark: optimised pipeline on field_15 (gamma_search + refine_clusters).

Runs the NEW code path only — fast gamma_search (iter=10, warm-start, 2 refine)
followed by refine_clusters (split + merge loop).
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
log = logging.getLogger("bench_opt")

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
_field = sys.argv[1] if len(sys.argv) > 1 else "field_15"
EDGE_FILE = DATA_ROOT / "linktype_edges" / _field / "bc_assoc_strength.parquet"
MIN_DOCS = 1000
SEED = 42
ITERATIONS = 50
GAMMA_RANGE = (1e-6, 1e-3)


def main():
    import polars as pl
    from sciscape.clustering.graph import build_graph, giant_component
    from sciscape.clustering.runner import LeidenRunner
    from sciscape.clustering.postprocess import gamma_search, refine_clusters

    log.info("═══ Optimised pipeline: %s ═══", _field)
    log.info("min_docs=%d, gamma_range=%s", MIN_DOCS, GAMMA_RANGE)

    # ── Build graph ──────────────────────────────────────────────
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

    # ── Phase 1: Optimised γ search ─────────────────────────────
    log.info("\n══ γ search (iter=10, warm-start, max_refine=2) ══")
    t0 = time.perf_counter()
    search_result = gamma_search(
        runner,
        gamma_range=GAMMA_RANGE,
        min_size=MIN_DOCS,
        search_iterations=10,
        warm_start=True,
        max_refine=2,
    )
    search_elapsed = time.perf_counter() - t0
    best_gamma = search_result.best_gamma
    log.info("  γ search: %.1fs (%d evals) → γ=%.4e, %d large",
             search_elapsed, search_result.n_evals, best_gamma, search_result.n_large)

    # ── Phase 2: Final full-quality Leiden ───────────────────────
    log.info("\n══ Final Leiden at γ=%.4e (iter=%d) ══", best_gamma, ITERATIONS)
    t1 = time.perf_counter()
    final_result = runner.run(best_gamma)
    raw_membership = list(final_result.membership)
    final_elapsed = time.perf_counter() - t1
    raw_sizes = Counter(raw_membership)
    log.info("  Final run: %.1fs → %d clusters, %d large (≥%d)",
             final_elapsed, len(raw_sizes),
             sum(1 for s in raw_sizes.values() if s >= MIN_DOCS), MIN_DOCS)

    # ── Phase 3: Refine (split + merge loop) ────────────────────
    log.info("\n══ refine_clusters (split + merge) ══")
    t2 = time.perf_counter()
    refine_result = refine_clusters(
        runner, raw_membership, best_gamma, min_size=MIN_DOCS,
    )
    refine_elapsed = time.perf_counter() - t2

    # Summarise
    final_mem = refine_result.membership
    final_sizes = Counter(final_mem)
    n_large = sum(1 for s in final_sizes.values() if s >= MIN_DOCS)
    n_singletons = sum(1 for s in final_sizes.values() if s == 1)
    top10 = sorted(final_sizes.values(), reverse=True)[:10]

    log.info("\n══ RESULTS ══")
    log.info("  %-25s %.4e", "Best γ", best_gamma)
    log.info("  %-25s %d", "Total clusters", len(final_sizes))
    log.info("  %-25s %d", "Large (≥%d)" % MIN_DOCS, n_large)
    log.info("  %-25s %d", "Singletons", n_singletons)
    log.info("  %-25s %s", "Top 10 sizes", top10)
    log.info("  %-25s %d", "Max cluster size", max(final_sizes.values()))
    log.info("  %-25s %d rounds", "Refinement", refine_result.n_rounds)

    for i, sr in enumerate(refine_result.split_results):
        log.info("  Round %d split: %d clusters → %d new sub-clusters",
                 i + 1, sr.n_clusters_split, sr.n_new_clusters_created)
    for i, mr in enumerate(refine_result.merge_results):
        log.info("  Round %d merge: %d resolved (%d nodes), %d unresolvable",
                 i + 1, mr.n_clusters_resolved, mr.n_nodes_resolved,
                 mr.n_clusters_unresolvable)

    total = time.perf_counter() - t0
    log.info("\n══ TIMING ══")
    log.info("  %-25s %7.1fs", "γ search", search_elapsed)
    log.info("  %-25s %7.1fs", "Final Leiden", final_elapsed)
    log.info("  %-25s %7.1fs", "Refinement", refine_elapsed)
    log.info("  %-25s %7.1fs", "Total", total)


if __name__ == "__main__":
    main()
