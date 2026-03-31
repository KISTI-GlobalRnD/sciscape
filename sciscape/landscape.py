"""End-to-end landscape pipeline: edges → clustering → keywords → report.

Public API
----------
run_landscape(edge_path, abstract_path, output_dir, ...)
    Full pipeline with BFS subsampling, hierarchical Leiden, keyword
    extraction on the finest cluster level, and interactive HTML report.

LandscapeConfig
    Dataclass holding all tuneable knobs.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class LandscapeConfig:
    """All tuneable parameters for :func:`run_landscape`."""

    # Subsampling
    n_target_nodes: int = 100_000
    seed: int = 42
    force: bool = False  # True → ignore cached intermediate results

    # Clustering
    min_docs_per_cluster: int = 1000  # minimum nodes per cluster (post-merge)
    gamma_range: Tuple[float, float] = (1e-6, 1e-3)  # resolution search range for nano (CPM)
    n_hierarchy_levels: int = 2  # nano + micro (upper levels use contraction)
    leiden_objective: str = "cpm"
    leiden_iterations: int = 50

    # Block-init mode: high-γ blocks → contraction → cascade hot start
    # "auto" → 10 × gamma_range[1]; None → disabled; float → explicit value
    gamma_block: Optional[float] = "auto"  # type: ignore[assignment]

    # Keyword extraction
    top_n_unigrams: int = 200
    top_n_keywords: int = 100
    include_title: bool = True
    title_weight: float = 2.0
    min_df_unigram: int = 5
    min_df_phrase: int = 3
    ngram_range: Tuple[int, int] = (2, 3)
    normalization_enabled: bool = True
    cooccurrence_enabled: bool = True
    term_network_enabled: bool = True
    depth_enabled: bool = True
    n_jobs: int = -1

    # Report
    report_title: str = "SciScape Landscape"


# ---------------------------------------------------------------------------
# Step 1: Load & BFS subsample
# ---------------------------------------------------------------------------
def _load_and_subsample(
    edge_path: Path,
    n_target: int,
    seed: int,
) -> "pl.DataFrame":
    """Load edge list and subsample via igraph BFS to ~n_target nodes."""
    import polars as pl
    from .clustering.graph import build_graph, giant_component

    log.info("Loading edge list: %s", edge_path)
    t0 = time.perf_counter()
    edges = pl.read_parquet(edge_path) if edge_path.suffix == ".parquet" else pl.read_csv(
        edge_path,
        separator="\t",
        has_header=True,
        schema_overrides={"uid1": pl.Utf8, "uid2": pl.Utf8, "rel_sum2": pl.Float64},
    )
    log.info("Loaded %s edges in %.1fs", f"{len(edges):,}", time.perf_counter() - t0)

    # Auto-detect and rename columns to canonical schema (uid1, uid2, rel_sum2)
    col_map = {}
    cols = set(edges.columns)
    if "uid1" not in cols:
        for alias in ("src", "source", "node1", "from"):
            if alias in cols:
                col_map[alias] = "uid1"
                break
    if "uid2" not in cols:
        for alias in ("dst", "target", "node2", "to"):
            if alias in cols:
                col_map[alias] = "uid2"
                break
    if "rel_sum2" not in cols:
        for alias in ("weight", "w", "value"):
            if alias in cols:
                col_map[alias] = "rel_sum2"
                break
    if col_map:
        log.info("Column mapping: %s", col_map)
        edges = edges.rename(col_map)

    log.info("Building graph for subsampling...")
    t0 = time.perf_counter()
    graph = build_graph(edges)
    log.info("Graph: %d V, %d E (%.1fs)",
             graph.vcount(), graph.ecount(), time.perf_counter() - t0)

    giant = giant_component(graph)
    n_total = giant.vcount()
    log.info("Giant component: %d vertices", n_total)

    if n_total <= n_target:
        log.info("No subsampling needed (target=%s)", f"{n_target:,}")
        keep_uids = set(giant.vs["uid"])
        return edges.filter(
            pl.col("uid1").is_in(keep_uids) & pl.col("uid2").is_in(keep_uids)
        )

    rng = np.random.RandomState(seed)
    start = rng.randint(0, n_total)
    log.info("BFS from vertex %d (target: %s nodes)...", start, f"{n_target:,}")

    t0 = time.perf_counter()
    bfs_result = giant.bfs(start)
    keep_vertices = bfs_result[0][:n_target]
    log.info("BFS done in %.1fs, selected %d vertices",
             time.perf_counter() - t0, len(keep_vertices))

    sub = giant.subgraph(keep_vertices)
    keep_uids = set(sub.vs["uid"])
    log.info("Subgraph: %d V, %d E", sub.vcount(), sub.ecount())

    sub_edges = edges.filter(
        pl.col("uid1").is_in(keep_uids) & pl.col("uid2").is_in(keep_uids)
    )
    log.info("Filtered edges: %s", f"{len(sub_edges):,}")

    del graph, giant
    return sub_edges


def _all_levels_cached(membership_path: Path, cfg: "LandscapeConfig") -> bool:
    """Check if all hierarchy levels already exist in the membership file."""
    if not membership_path.exists():
        return False
    import pyarrow.parquet as pq
    schema = pq.read_schema(membership_path)
    existing = {f.name.removeprefix("cluster_") for f in schema if f.name.startswith("cluster_")}
    required = {"nano", "micro"}
    required = set(["nano", "micro"][:cfg.n_hierarchy_levels])
    return required.issubset(existing)


# ---------------------------------------------------------------------------
# Step 2: Hierarchical clustering
# ---------------------------------------------------------------------------
def _run_clustering(
    edges: "pl.DataFrame",
    cfg: LandscapeConfig,
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
    from .clustering import (
        build_graph,
        giant_component,
    )
    from .clustering.runner import LeidenRunner
    from .clustering.postprocess import refine_clusters, gamma_search

    membership_path = output_dir / "membership.parquet"
    level_names = ["nano", "micro"][:cfg.n_hierarchy_levels]

    # Check which levels are already done
    existing_levels: set = set()
    if not cfg.force and membership_path.exists():
        existing_df = pl.read_parquet(membership_path)
        for col in existing_df.columns:
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

        # Resolve gamma_block: "auto" → 10 × gamma_range[1]
        gamma_block = cfg.gamma_block
        if gamma_block == "auto":
            gamma_block = 10.0 * cfg.gamma_range[1]

        if gamma_block is not None:
            # ── Block-init mode: blocks → contraction → cascade ──
            import math as _math
            from .clustering.block_init import (
                block_init as _block_init,
                cascade_search as _cascade_search,
                save_blocks, load_blocks, is_cache_valid,
                contract_graph,
            )

            blocks_path = output_dir / "blocks.parquet"

            if not cfg.force and is_cache_valid(
                blocks_path, gamma_block, giant.vcount()
            ):
                log.info("Loading cached blocks from %s", blocks_path)
                blocks = load_blocks(blocks_path)
            else:
                log.info("Block init: γ_block=%.2e...", gamma_block)
                blocks = _block_init(runner, gamma_block, seed=cfg.seed)
                save_blocks(blocks, blocks_path, uids)

            # Singleton warning
            singleton_frac = blocks.n_singletons / blocks.n_nodes
            if singleton_frac > 0.8:
                log.warning(
                    "Block init: %.0f%% singletons (%d/%d). "
                    "Consider lowering gamma_block (currently %.2e).",
                    singleton_frac * 100,
                    blocks.n_singletons, blocks.n_nodes, gamma_block,
                )

            contracted, contracted_runner = contract_graph(runner, blocks)
            node_sizes = blocks.node_sizes_list
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
            hi_g = _math.log10(gamma_block * 0.9)
            if hi_g > lo_g:
                n_steps = min(5, max(2, int((hi_g - lo_g) / 0.3) + 1))
                cascade_gammas = [
                    10 ** (lo_g + i * (hi_g - lo_g) / (n_steps - 1))
                    for i in range(n_steps)
                ]
            else:
                cascade_gammas = [best_gamma]

            cascade_result = _cascade_search(
                runner, blocks,
                gamma_targets=cascade_gammas,
                seed=cfg.seed,
                hot_start=True,
            )
            raw_membership = list(cascade_result.membership)
            search_elapsed = time.perf_counter() - t0_search
            log.info("  Block-init + cascade: %.1fs (γ=%.4e, %d clusters)",
                     search_elapsed, best_gamma, cascade_result.n_clusters)

        else:
            # ── Standard mode (no block-init) ────────────────────
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
        pl.DataFrame(cols).write_parquet(membership_path)
        log.info("  nano membership saved")
    else:
        log.info("Nano cached, loading...")
        existing_df = pl.read_parquet(membership_path)
        nano_for_save = existing_df["cluster_nano"].to_list()
        # For contraction, replace -1 (undetermined) with a valid cluster ID.
        nano_membership = list(nano_for_save)
        has_undetermined = any(c < 0 for c in nano_membership)
        if has_undetermined:
            next_cid = max((c for c in nano_membership if c >= 0), default=0) + 1
            for i in range(len(nano_membership)):
                if nano_membership[i] < 0:
                    nano_membership[i] = next_cid

    # ------------------------------------------------------------------
    # Upper levels: dendrogram on contracted graph + constrained cut
    # ------------------------------------------------------------------
    if cfg.n_hierarchy_levels >= 2 and "micro" not in existing_levels:
        from .clustering.dendrogram import build_dendrogram
        from .clustering.constrained_cut import constrained_cut

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
        nano_sizes = Counter(compact_membership)
        nano_size_arr = np.array(
            [nano_sizes[i] for i in range(n_contracted)], dtype=np.uint64,
        )

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
        n_total_nodes = len(nano_membership)
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
        pl.DataFrame(cols).write_parquet(membership_path)
        log.info("  membership saved (nano + micro)")

    return pl.read_parquet(membership_path)


# ---------------------------------------------------------------------------
# Step 3: Keyword extraction (finest level, auto-detected)
# ---------------------------------------------------------------------------
def _run_keywords(
    membership_path: Path,
    abstract_path: Path,
    cfg: LandscapeConfig,
) -> Tuple[Any, Optional[Dict]]:
    """Run keyword extraction pipeline on the finest cluster level."""
    from .keyword_extraction import KeywordExtractionConfig
    from .keyword_extraction.depth import DepthConfig
    from .keyword_extraction.keyword_extraction import KeywordExtractionPipeline
    from .keyword_extraction.term_network import TermNetworkConfig

    kw_cfg = KeywordExtractionConfig(
        abstract_path=abstract_path,
        membership_path=membership_path,
        # cluster_level=None → auto-detects finest level

        include_title=cfg.include_title,
        title_weight=cfg.title_weight,

        min_df_unigram=cfg.min_df_unigram,
        min_df_phrase=cfg.min_df_phrase,
        use_phrase_vectorizer=True,
        ngram_min=cfg.ngram_range[0],
        ngram_max=cfg.ngram_range[1],
        phrase_min_count_per_cluster=5,

        top_n_unigrams=cfg.top_n_unigrams,
        top_n_keywords=cfg.top_n_keywords,
        scoring_pool_factor=1.5,

        normalization_enabled=cfg.normalization_enabled,
        norm_plural_merge_enabled=True,
        academic_stopwords_enabled=True,
        artifact_filter_enabled=True,
        cross_cluster_penalty_enabled=True,
        cross_cluster_penalty_min_count=2,
        fragment_suppression_enabled=True,

        cooccurrence_enabled=cfg.cooccurrence_enabled,
        cooccurrence_min_count=3,
        term_network=TermNetworkConfig(
            enabled=cfg.term_network_enabled,
            layers=["string", "token", "cooccurrence"],
            merge_threshold=0.5,
        ) if cfg.term_network_enabled else None,
        auto_merge_enabled=True,
        short_term_expansion_enabled=True,

        depth=DepthConfig(enabled=cfg.depth_enabled, n_levels=3) if cfg.depth_enabled else None,

        n_jobs=cfg.n_jobs,
        verbose=True,
    )

    log.info("Running keyword extraction pipeline...")
    t0 = time.perf_counter()
    pipeline = KeywordExtractionPipeline(kw_cfg)
    result = pipeline.run()
    elapsed = time.perf_counter() - t0
    log.info("Keywords: %d rows, %d clusters (%.1fs)",
             len(result), result["cluster_id"].nunique(), elapsed)

    viz_data = pipeline.get_visualization_data()
    return result, viz_data


# ---------------------------------------------------------------------------
# Step 4: Generate report
# ---------------------------------------------------------------------------
def _generate_report(
    keywords_df: Any,
    viz_data: Optional[Dict],
    out_dir: Path,
    title: str,
) -> List[str]:
    """Generate landscape visualization report."""
    from .keyword_extraction.visualization import (
        export_report,
        plot_cluster_map_with_keywords,
    )

    log.info("Generating landscape report → %s", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = export_report(
        keywords_df,
        output_dir=str(out_dir),
        title=title,
        viz_data=viz_data,
    )
    for p in paths:
        log.info("  → %s", Path(p).name)

    fig = plot_cluster_map_with_keywords(
        keywords_df,
        viz_data=viz_data,
        layout="mds",
        top_n_keywords=8,
        title=f"{title} — Cluster Map",
    )
    map_path = str(out_dir / "landscape_detailed.html")
    fig.write_html(map_path, include_plotlyjs="cdn")
    log.info("  → landscape_detailed.html")

    return paths


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_landscape(
    edge_path: Path,
    abstract_path: Path,
    output_dir: Path,
    config: Optional[LandscapeConfig] = None,
) -> Dict[str, Any]:
    """Run the full landscape pipeline.

    Parameters
    ----------
    edge_path : Path
        Edge list file (.parquet, .csv, .tsv, .txt).
    abstract_path : Path
        Parquet with uid, title, abstract, pubyear columns.
    output_dir : Path
        Directory for all outputs (membership, keywords, report).
    config : LandscapeConfig, optional
        Pipeline parameters.  Defaults are sensible for ~100k nodes.

    Returns
    -------
    dict
        ``membership_path``, ``keywords_path``, ``report_dir``,
        ``keywords_df``, ``viz_data``.
    """
    import pandas as pd
    import polars as pl

    cfg = config or LandscapeConfig()
    edge_path = Path(edge_path)
    abstract_path = Path(abstract_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Input validation ──────────────────────────────────────
    if not edge_path.exists():
        raise FileNotFoundError(f"Edge file not found: {edge_path}")
    if not abstract_path.exists():
        raise FileNotFoundError(f"Abstract file not found: {abstract_path}")

    # Check abstract columns (lightweight schema-only read)
    try:
        import pyarrow.parquet as pq
        abs_schema = pq.read_schema(str(abstract_path))
        abs_cols = {f.name for f in abs_schema}
        required = {"uid", "title", "abstract", "pubyear"}
        missing = required - abs_cols
        if missing:
            raise ValueError(
                f"Abstract file missing columns: {missing}. "
                f"Required: {required}. Found: {abs_cols}"
            )
    except ImportError:
        pass  # pyarrow not available → skip schema check

    membership_path = output_dir / "membership.parquet"
    abstract_subset_path = output_dir / "abstracts_subset.parquet"
    keywords_path = output_dir / "keywords.parquet"
    report_dir = output_dir / "report"

    # ------------------------------------------------------------------
    # Step 1–2: Clustering (level-by-level with per-level caching)
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Step 1–2: Hierarchical clustering")
    log.info("=" * 60)

    # Check if all levels already cached → skip edge loading entirely
    all_cached = _all_levels_cached(membership_path, cfg)
    if all_cached and not cfg.force:
        log.info("All hierarchy levels cached, skipping edge load + clustering")
        membership_df = pl.read_parquet(membership_path)
    else:
        edges = _load_and_subsample(edge_path, cfg.n_target_nodes, cfg.seed)
        membership_df = _run_clustering(edges, cfg, output_dir)

    # ------------------------------------------------------------------
    # Step 3: Keyword extraction (skip if keywords exist)
    # ------------------------------------------------------------------
    if not cfg.force and keywords_path.exists():
        log.info("=" * 60)
        log.info("Step 3: Reusing existing keywords: %s", keywords_path)
        log.info("=" * 60)
        keywords_df = pd.read_parquet(keywords_path)
        viz_data = None
        log.info("  %d keywords, %d clusters",
                 len(keywords_df), keywords_df["cluster_id"].nunique())
    else:
        log.info("=" * 60)
        log.info("Step 3: Keyword extraction (finest level)")
        log.info("=" * 60)

        member_uids = set(membership_df["uid"].to_list())
        log.info("Loading abstracts for %s nodes...", f"{len(member_uids):,}")

        abstract_df = pl.read_parquet(
            abstract_path,
            columns=["uid", "title", "abstract", "pubyear"],
        ).filter(pl.col("uid").is_in(member_uids))
        log.info("Matched abstracts: %s", f"{len(abstract_df):,}")

        abstract_df.write_parquet(abstract_subset_path)

        keywords_df, viz_data = _run_keywords(membership_path, abstract_subset_path, cfg)

        # Save keywords
        save_df = keywords_df.copy()
        dict_cols = ("pub_year_series", "year_denominators", "ppm_series",
                     "loglift_series", "bayesian_log_odds_series")
        for col in dict_cols:
            if col in save_df.columns:
                save_df[col] = save_df[col].apply(
                    lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v
                )
        save_df.to_parquet(keywords_path, index=False)
        log.info("Keywords saved: %s", keywords_path)

    # ------------------------------------------------------------------
    # Step 4: Report (always regenerate — cheap)
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Step 4: Generate landscape report")
    log.info("=" * 60)
    _generate_report(keywords_df, viz_data, report_dir, cfg.report_title)

    # Summary
    log.info("=" * 60)
    n_clusters = keywords_df["cluster_id"].nunique()
    log.info("DONE — %d clusters, %d keywords", n_clusters, len(keywords_df))
    for cid in sorted(keywords_df["cluster_id"].unique())[:10]:
        grp = keywords_df[keywords_df["cluster_id"] == cid]
        top3 = grp.nlargest(3, "score")["term"].tolist()
        log.info("  C%d (%d kw): %s", cid, len(grp), ", ".join(top3))
    if n_clusters > 10:
        log.info("  ... (%d more clusters)", n_clusters - 10)
    log.info("=" * 60)

    return {
        "membership_path": membership_path,
        "keywords_path": keywords_path,
        "report_dir": report_dir,
        "keywords_df": keywords_df,
        "viz_data": viz_data,
    }


__all__ = ["LandscapeConfig", "run_landscape"]
