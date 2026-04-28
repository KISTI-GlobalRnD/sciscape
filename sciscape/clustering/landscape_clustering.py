"""Legacy igraph-based clustering pipeline for landscape analysis.

Extracted from landscape.py to reduce monolith size. This module
handles BFS subsampling, nano-level gamma search (pre-partition or
direct), micro-level dendrogram + constrained cut.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

if TYPE_CHECKING:
    import polars as pl

    from ..landscape import LandscapeConfig

log = logging.getLogger(__name__)


def _run_clustering(
    edges: "pl.DataFrame",
    cfg: "LandscapeConfig",
    output_dir: Path,
) -> "pl.DataFrame":
    """Run hierarchical Leiden level-by-level with per-level caching.

    **Nano level**: Searches for the highest γ that, after merging clusters
    below ``min_docs_per_cluster``, still maximises the number of clusters.
    **Upper levels**: Run on contracted graph with γ=1.0.

    Each level is saved to ``membership.parquet`` after completion.
    """
    import polars as pl
    from collections import Counter
    from . import (
        build_graph,
        giant_component,
    )
    from .runner import LeidenRunner
    from .postprocess import refine_clusters, gamma_search

    membership_path = output_dir / "membership.parquet"
    level_names = ["nano", "micro"][:cfg.n_hierarchy_levels]

    # Check which levels are already done (schema-only, no data read)
    existing_levels: set = set()
    if not cfg.force and membership_path.exists():
        import pyarrow.parquet as pq
        schema_cols = pq.read_schema(membership_path).names
        for col in schema_cols:
            if col.startswith("cluster_"):
                existing_levels.add(col.removeprefix("cluster_"))

    if all(name in existing_levels for name in level_names):
        log.info("All %d hierarchy levels cached, skipping clustering",
                 len(level_names))
        return pl.read_parquet(membership_path)

    # Build graph
    log.info("Building graph...")
    t0 = time.perf_counter()
    graph = build_graph(edges)
    log.info("Graph: %d vertices, %d edges (%.1fs)",
             graph.vcount(), graph.ecount(), time.perf_counter() - t0)

    log.info("Extracting giant component...")
    giant = giant_component(graph)
    log.info("Giant component: %d vertices, %d edges",
             giant.vcount(), giant.ecount())
    uids = list(giant.vs["uid"])
    min_docs = cfg.min_docs_per_cluster

    # ------------------------------------------------------------------
    # Nano: auto-search γ to maximise clusters with min_docs constraint
    # ------------------------------------------------------------------
    if "nano" not in existing_levels:
        runner = LeidenRunner(
            giant, objective=cfg.leiden_objective,
            default_seed=cfg.seed, default_iterations=cfg.leiden_iterations,
        )
        t0_search = time.perf_counter()

        # Resolve gamma_pre: "auto" → 10 × gamma_range[1]
        gamma_pre = cfg.gamma_pre
        if gamma_pre == "auto":
            gamma_pre = 10.0 * cfg.gamma_range[1]

        if gamma_pre is not None:
            # ── Pre-partition mode: parts → contraction → cascade ──
            import math as _math
            from .prepartition import (
                prepartition as _prepartition,
                cascade_search as _cascade_search,
                save_prepartition, load_prepartition, is_cache_valid,
                contract_graph,
            )

            parts_path = output_dir / "parts.parquet"

            if not cfg.force and is_cache_valid(
                parts_path, gamma_pre, giant.vcount()
            ):
                log.info("Loading cached parts from %s", parts_path)
                parts = load_prepartition(parts_path)
            else:
                log.info("Pre-partition: γ_block=%.2e...", gamma_pre)
                parts = _prepartition(runner, gamma_pre, seed=cfg.seed)
                save_prepartition(parts, parts_path, uids)

            # Singleton warning
            singleton_frac = parts.n_singletons / parts.n_nodes
            if singleton_frac > 0.8:
                log.warning(
                    "Pre-partition: %.0f%% singletons (%d/%d). "
                    "Consider lowering gamma_pre (currently %.2e).",
                    singleton_frac * 100,
                    parts.n_singletons, parts.n_nodes, gamma_pre,
                )

            contracted, contracted_runner = contract_graph(runner, parts)
            node_sizes = parts.node_sizes_list
            contraction_ratio = giant.vcount() / max(contracted.vcount(), 1)
            log.info("Contracted: %d → %d supernodes (%.1fx reduction)",
                     giant.vcount(), contracted.vcount(), contraction_ratio)

            # γ search on contracted graph (weighted sizes)
            log.info("Searching optimal γ on contracted graph (min_docs=%d)...",
                     min_docs)
            search_result = gamma_search(
                contracted_runner,
                gamma_range=cfg.gamma_range,
                min_size=min_docs,
                search_iterations=10,
                node_sizes=node_sizes,
            )
            best_gamma = search_result.best_gamma

            # Cascade targets: log-spaced from best_gamma up to near γ_block
            lo_g = _math.log10(best_gamma)
            hi_g = _math.log10(gamma_pre * cfg.gamma_pre_margin)
            if hi_g > lo_g:
                n_steps = min(5, max(2, int((hi_g - lo_g) / cfg.gamma_log_step) + 1))
                cascade_gammas = [
                    10 ** (lo_g + i * (hi_g - lo_g) / (n_steps - 1))
                    for i in range(n_steps)
                ]
            else:
                cascade_gammas = [best_gamma]

            cascade_result = _cascade_search(
                runner, parts,
                gamma_targets=cascade_gammas,
                seed=cfg.seed,
                hot_start=True,
            )
            raw_membership = list(cascade_result.membership)
            search_elapsed = time.perf_counter() - t0_search
            log.info("  Pre-partition + cascade: %.1fs (γ=%.4e, %d clusters)",
                     search_elapsed, best_gamma, cascade_result.n_clusters)

        else:
            # ── Standard mode (no pre-partition) ────────────────────
            log.info("Searching optimal γ for nano (min_docs=%d)...", min_docs)
            search_result = gamma_search(
                runner,
                gamma_range=cfg.gamma_range,
                min_size=min_docs,
                search_iterations=10,
            )
            best_gamma = search_result.best_gamma

            log.info("  Running final Leiden at γ=%.4f (full iterations)...",
                     best_gamma)
            final_result = runner.run(best_gamma)
            raw_membership = list(final_result.membership)
            search_elapsed = time.perf_counter() - t0_search
            log.info("  γ search + final: %.1fs (%d search evals + 1 final)",
                     search_elapsed, search_result.n_evals)

        # Common refinement + save path
        refinement_result = refine_clusters(
            runner, raw_membership, best_gamma, min_size=min_docs,
        )
        nano_membership = list(refinement_result.membership)

        # Mark remaining singletons as undetermined (-1) in saved output only.
        sizes_final = Counter(nano_membership)
        undetermined_nodes: set[int] = {
            i for i, c in enumerate(nano_membership) if sizes_final[c] == 1
        }

        nano_for_save = list(nano_membership)
        for i in undetermined_nodes:
            nano_for_save[i] = -1

        n_clusters = len(sizes_final)
        n_undetermined = len(undetermined_nodes)
        log.info("  → Final: %d clusters, %d undetermined (%.3f%%)",
                 n_clusters, n_undetermined,
                 n_undetermined / len(nano_membership) * 100 if nano_membership else 0)

        # Save nano
        cols: Dict[str, Any] = {"uid": uids, "cluster_nano": nano_for_save}
        pl.DataFrame(cols).write_parquet(membership_path, compression="zstd")
        log.info("  nano membership saved")
    else:
        log.info("Nano cached, loading...")
        existing_df = pl.read_parquet(membership_path)
        nano_arr = existing_df["cluster_nano"].to_numpy()
        nano_for_save = nano_arr.tolist()
        # For contraction, replace -1 (undetermined) with a valid cluster ID.
        if (nano_arr < 0).any():
            next_cid = int(nano_arr[nano_arr >= 0].max()) + 1
            nano_arr = np.where(nano_arr < 0, next_cid, nano_arr)
        nano_membership = nano_arr.tolist()

    # ------------------------------------------------------------------
    # Upper levels: dendrogram on contracted graph + constrained cut
    # ------------------------------------------------------------------
    if cfg.n_hierarchy_levels >= 2 and "micro" not in existing_levels:
        from .dendrogram import build_dendrogram
        from .constrained_cut import constrained_cut

        runner = LeidenRunner(
            giant, objective=cfg.leiden_objective,
            default_seed=cfg.seed, default_iterations=cfg.leiden_iterations,
        )

        # Compact membership IDs to 0..K-1 so igraph.contract_vertices
        # produces exactly K supernodes (no empty-cluster gaps).
        unique_ids = sorted(set(nano_membership))
        id_remap = {old: new for new, old in enumerate(unique_ids)}
        compact_membership = [id_remap[c] for c in nano_membership]

        contracted = runner.contract(compact_membership, combine_weights="sum", keep_loops=True)
        n_contracted = contracted.vcount()

        # Compute nano cluster sizes for node_sizes parameter.
        nano_size_arr = np.bincount(
            compact_membership, minlength=n_contracted,
        ).astype(np.uint64)

        log.info("Building CPM dendrogram on contracted %d-node graph...",
                 n_contracted)
        t0 = time.perf_counter()
        linkage = build_dendrogram(contracted, mode="cpm", node_sizes=nano_size_arr)
        log.info("  Dendrogram: %d merges, height range [%.6f, %.6f] (%.2fs)",
                 len(linkage),
                 linkage[-1, 2] if len(linkage) > 0 else 0,
                 linkage[0, 2] if len(linkage) > 0 else 0,
                 time.perf_counter() - t0)

        # Constrained cut: min_size in original-node terms.
        # Use a fraction of total nodes as micro min_size (same cluster
        # count maximisation objective, just at a coarser scale).

        micro_min_size = max(
            int(nano_size_arr.sum()) // 20,  # ~5% of total
            int(nano_size_arr.max()) + 1,     # larger than biggest nano
        )
        cut_result = constrained_cut(
            linkage, min_size=micro_min_size,
            leaf_sizes=nano_size_arr,
        )

        if not cut_result.feasible or cut_result.n_clusters <= 1:
            log.warning("  micro cut infeasible or trivial (%d clusters), "
                        "falling back to 1 cluster", cut_result.n_clusters)
            micro_mem_contracted = [0] * n_contracted
            n_micro = 1
        else:
            micro_mem_contracted = list(cut_result.membership)
            n_micro = cut_result.n_clusters

        log.info("  → micro: %d clusters (min_size=%d)", n_micro, micro_min_size)

        # Map back to original nodes (use compact IDs as contracted-graph indices)
        micro_membership = [micro_mem_contracted[compact_membership[i]]
                            for i in range(len(nano_membership))]

        # Save both levels (nano uses -1 for undetermined, micro keeps valid IDs)
        cols = {
            "uid": uids,
            "cluster_nano": nano_for_save,
            "cluster_micro": micro_membership,
        }
        pl.DataFrame(cols).write_parquet(membership_path, compression="zstd")
        log.info("  membership saved (nano + micro)")

    return pl.read_parquet(membership_path)
