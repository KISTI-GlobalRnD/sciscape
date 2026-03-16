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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
        HierarchyLevelConfig,
        PostprocessConfig,
        build_graph,
        giant_component,
    )
    from .clustering.hierarchy_builder import HierarchyBuilder
    from .clustering.runner import LeidenRunner
    from .clustering.postprocess import merge_small_clusters

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
        log.info("Searching optimal γ for nano (min_docs=%d)...", min_docs)
        runner = LeidenRunner(
            giant, objective=cfg.leiden_objective,
            default_seed=cfg.seed, default_iterations=cfg.leiden_iterations,
        )

        def _eval_gamma(gamma: float) -> Tuple[int, list]:
            """Run Leiden at gamma, merge small clusters, return (n_clusters, membership)."""
            t0 = time.perf_counter()
            result = runner.run(gamma)
            post = merge_small_clusters(giant, result.membership, min_size=min_docs)
            n_clusters = len(set(post.membership))
            smallest = min(Counter(post.membership).values())
            elapsed = time.perf_counter() - t0
            log.info("  γ=%.4f → %d clusters (min=%d, Q=%.0f, %.1fs)",
                     gamma, n_clusters, smallest, result.quality, elapsed)
            return n_clusters, list(post.membership)

        # Phase 1: Coarse scan (3 points) to find promising region
        lo_g, hi_g = np.log10(cfg.gamma_range[0]), np.log10(cfg.gamma_range[1])
        coarse_gammas = np.logspace(lo_g, hi_g, num=3)
        cache: Dict[float, Tuple[int, list]] = {}

        for g in coarse_gammas:
            cache[g] = _eval_gamma(g)

        # Phase 2: Binary search to find γ* maximising cluster count
        # γ↑ → raw clusters↑, but after merge clusters can peak then decline
        # (over-splitting → too many tiny clusters → all merged away)
        # Find the peak via ternary-like search.
        best_gamma = max(cache, key=lambda g: cache[g][0])
        best_n_clusters = cache[best_gamma][0]

        for _ in range(4):  # 4 refinement rounds
            # Probe midpoints around best
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
                cache[g] = _eval_gamma(g)

            best_gamma = max(cache, key=lambda g: cache[g][0])
            new_best = cache[best_gamma][0]
            if new_best == best_n_clusters:
                break  # converged
            best_n_clusters = new_best

        log.info("  → Best: γ=%.4f → %d clusters (%d evals)",
                 best_gamma, best_n_clusters, len(cache))
        nano_membership = cache[best_gamma][1]

        # Save nano
        cols: Dict[str, Any] = {"uid": uids, "cluster_nano": nano_membership}
        pl.DataFrame(cols).write_parquet(membership_path)
        log.info("  nano membership saved")
    else:
        log.info("Nano cached, loading...")
        existing_df = pl.read_parquet(membership_path)
        nano_membership = existing_df["cluster_nano"].to_list()

    # ------------------------------------------------------------------
    # Upper levels: contract and run with γ=1.0
    # ------------------------------------------------------------------
    if cfg.n_hierarchy_levels >= 2 and "micro" not in existing_levels:
        runner = LeidenRunner(
            giant, objective=cfg.leiden_objective,
            default_seed=cfg.seed, default_iterations=cfg.leiden_iterations,
        )
        contracted = runner.contract(nano_membership, combine_weights="sum", keep_loops=True)
        n_contracted = contracted.vcount()

        # Normalize edge weights to density for CPM on contracted graph.
        # Raw summed weights are proportional to cluster sizes, making CPM
        # unable to split.  Dividing by (n_i * n_j) yields inter-cluster
        # edge density, which CPM can meaningfully threshold.
        nano_sizes = Counter(nano_membership)
        for e in contracted.es:
            s, t = e.source, e.target
            ns, nt = nano_sizes[s], nano_sizes[t]
            if s != t:
                e["weight"] = e["weight"] / (ns * nt) if ns * nt > 0 else 0.0
            else:
                denom = ns * (ns - 1) / 2
                e["weight"] = e["weight"] / denom if denom > 0 else 0.0

        runner2 = runner.clone_for_graph(contracted)

        log.info("Searching optimal γ for micro on contracted %d-node graph "
                 "(density-normalized)...", n_contracted)

        # micro min_size: at least 2 nano clusters per micro (otherwise trivial)
        micro_min_size = max(2, n_contracted // 20)

        best_micro_gamma = None
        best_micro_n = 0
        best_micro_mem = None

        micro_gammas = np.logspace(-8, -3, num=10)
        for g in micro_gammas:
            t0 = time.perf_counter()
            res = runner2.run(g)
            post = merge_small_clusters(contracted, res.membership, min_size=micro_min_size)
            nc = len(set(post.membership))
            elapsed = time.perf_counter() - t0
            log.info("  γ=%.2e → %d micro clusters (%.2fs)", g, nc, elapsed)
            if nc > best_micro_n and nc < n_contracted:
                best_micro_n = nc
                best_micro_gamma = g
                best_micro_mem = list(post.membership)

        if best_micro_mem is None or best_micro_n <= 1:
            log.warning("  micro search found no useful partition, using 1 cluster")
            best_micro_mem = [0] * n_contracted
            best_micro_n = 1

        log.info("  → Best micro: γ=%.2e → %d clusters", best_micro_gamma or 0, best_micro_n)
        micro_mem_contracted = best_micro_mem
        n_micro = best_micro_n

        # Map back to original nodes
        micro_membership = [micro_mem_contracted[nano_membership[i]] for i in range(len(nano_membership))]

        # Save both levels
        cols = {
            "uid": uids,
            "cluster_nano": nano_membership,
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
