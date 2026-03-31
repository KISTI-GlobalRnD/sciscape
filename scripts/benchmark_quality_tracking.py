"""Benchmark: CPM quality tracking through refine_clusters stages.

Shows that each split/merge step monotonically improves (or preserves) CPM quality,
validating the use of local γ in subgraph operations.
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
log = logging.getLogger("bench_quality")

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
_field = sys.argv[1] if len(sys.argv) > 1 else "field_34"
EDGE_FILE = DATA_ROOT / "linktype_edges" / _field / "bc_assoc_strength.parquet"
MIN_DOCS = 1000
SEED = 42
ITERATIONS = 50
GAMMA_RANGE = (1e-6, 1e-3)


def describe(name, membership, gamma, graph):
    """Print cluster stats and CPM quality."""
    from sciscape.clustering.postprocess import cpm_quality

    sizes = Counter(membership)
    n_large = sum(1 for s in sizes.values() if s >= MIN_DOCS)
    n_small = sum(1 for s in sizes.values() if s < MIN_DOCS)
    Q = cpm_quality(graph, membership, gamma)

    log.info("  [%s] %d clusters (%d large, %d small), "
             "max=%d, CPM quality=%.4f",
             name, len(sizes), n_large, n_small,
             max(sizes.values()), Q)
    return Q


def main():
    import polars as pl
    from sciscape.clustering.graph import build_graph, giant_component
    from sciscape.clustering.runner import LeidenRunner
    from sciscape.clustering.postprocess import (
        gamma_search, split_large_clusters, resolve_small_clusters, cpm_quality,
    )

    log.info("═══ CPM Quality Tracking: %s ═══", _field)

    # ── Build ────────────────────────────────────────────────────
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

    # ── γ search ─────────────────────────────────────────────────
    log.info("\n══ Phase 1: γ search ══")
    t0 = time.perf_counter()
    sr = gamma_search(runner, gamma_range=GAMMA_RANGE, min_size=MIN_DOCS,
                      search_iterations=10, warm_start=True)
    best_gamma = sr.best_gamma
    log.info("  γ search: %.1fs → γ=%.4e", time.perf_counter() - t0, best_gamma)

    # ── Final Leiden ─────────────────────────────────────────────
    log.info("\n══ Phase 2: Final Leiden (iter=%d) ══", ITERATIONS)
    t0 = time.perf_counter()
    final = runner.run(best_gamma)
    raw_mem = list(final.membership)
    log.info("  Final Leiden: %.1fs", time.perf_counter() - t0)

    qualities = []
    q = describe("Raw Leiden", raw_mem, best_gamma, giant)
    qualities.append(("Raw Leiden", q))

    # ── Refine: step by step ─────────────────────────────────────
    membership = list(raw_mem)
    max_rounds = 3

    for round_i in range(max_rounds):
        log.info("\n══ Refinement round %d ══", round_i + 1)

        # Split
        log.info("  ── Split ──")
        split_result = split_large_clusters(
            runner, membership, best_gamma, min_size=MIN_DOCS,
            split_iterations=10,
        )
        membership = split_result.membership
        q = describe(f"Round {round_i+1} after split", membership, best_gamma, giant)
        qualities.append((f"R{round_i+1} split", q))

        # Merge
        log.info("  ── Merge ──")
        merge_result = resolve_small_clusters(
            runner, membership, best_gamma, min_size=MIN_DOCS,
        )
        membership = merge_result.membership
        q = describe(f"Round {round_i+1} after merge", membership, best_gamma, giant)
        qualities.append((f"R{round_i+1} merge", q))

        if (split_result.n_clusters_split == 0
                and merge_result.n_clusters_resolved == 0):
            log.info("  Converged at round %d", round_i + 1)
            break

    # ── Summary ──────────────────────────────────────────────────
    log.info("\n══ CPM QUALITY PROGRESSION (γ=%.4e) ══", best_gamma)
    log.info("  %-25s %15s %12s", "Stage", "CPM Quality", "Δ from prev")
    prev_q = None
    for name, q in qualities:
        delta = f"{q - prev_q:+.4f}" if prev_q is not None else ""
        log.info("  %-25s %15.4f %12s", name, q, delta)
        prev_q = q

    # Monotonicity checks
    qs = [q for _, q in qualities]
    monotone_all = all(qs[i] <= qs[i+1] + 1e-10 for i in range(len(qs)-1))
    log.info("\n  Quality monotonically non-decreasing (all steps): %s", monotone_all)

    # Split-only monotonicity: split should never decrease quality
    split_pairs = [(qualities[i], qualities[i+1])
                   for i in range(len(qualities) - 1)
                   if "split" in qualities[i+1][0].lower()]
    if split_pairs:
        split_monotone = all(b[1] >= a[1] - 1e-10 for a, b in split_pairs)
        log.info("  Quality non-decreasing after split only: %s", split_monotone)

    # Merge quality loss
    merge_pairs = [(qualities[i], qualities[i+1])
                   for i in range(len(qualities) - 1)
                   if "merge" in qualities[i+1][0].lower()]
    if merge_pairs:
        total_loss = sum(max(0, a[1] - b[1]) for a, b in merge_pairs)
        base_q = qualities[0][1]
        loss_pct = total_loss / base_q * 100 if base_q > 0 else 0
        log.info("  Merge quality loss: %.4f (%.2f%% of initial)", total_loss, loss_pct)

    # Per-cluster quality breakdown
    log.info("\n══ PER-CLUSTER BREAKDOWN (final) ══")
    final_sizes = Counter(membership)
    weights = giant.es["weight"] if "weight" in giant.es.attributes() else None
    internal = {}
    for eid, (u, v) in enumerate(giant.get_edgelist()):
        cu, cv = membership[u], membership[v]
        if cu == cv:
            w = weights[eid] if weights is not None else 1.0
            internal[cu] = internal.get(cu, 0.0) + float(w)

    log.info("  %-8s %8s %12s %12s %12s", "Cluster", "Size",
             "Internal e", "Penalty", "h(c)")
    for cid in sorted(final_sizes, key=lambda c: -final_sizes[c]):
        n_c = final_sizes[cid]
        e_c = internal.get(cid, 0.0)
        penalty = best_gamma * n_c * (n_c - 1) / 2
        h_c = e_c - penalty
        log.info("  %-8d %8d %12.2f %12.2f %12.2f",
                 cid, n_c, e_c, penalty, h_c)


if __name__ == "__main__":
    main()
