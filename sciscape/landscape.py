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
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple

if TYPE_CHECKING:
    import polars as pl
    from .clustering.hierarchy_oversize_postprocess import HierarchyPostprocessConfig

import numpy as np
import pandas as pd

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

    # Pre-partition mode: high-γ parts → contraction → cascade hot start
    # "auto" → 10 × gamma_range[1]; None → disabled; float → explicit value
    gamma_pre: Optional[float] = "auto"  # type: ignore[assignment]
    gamma_pre_margin: float = 0.9  # multiplier for gamma_pre upper bound in cascade
    gamma_log_step: float = 0.3      # log10-step size for cascade gamma spacing

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

    # Multi-layer combination
    layer_paths: Dict[str, Path] | None = None  # {"bc": path, "cc": path, ...}
    combine_strategy: str = "consensus"
    combine_top_k: int | str = "auto"
    auto_gamma: bool = False
    auto_gamma_target: float = 3.0
    hierarchy_postprocess: "HierarchyPostprocessConfig | None" = None

    # Callbacks
    progress: Any = None  # callable(str) for progress messages

    # Report
    report_title: str = "SciScape Landscape"
    edge_evidence_enabled: bool = True
    edge_evidence_max_relations: int = 300
    edge_evidence_samples_per_relation: int = 3


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
        matches = [a for a in ("src", "source", "node1", "from") if a in cols]
        if matches:
            if len(matches) > 1:
                log.warning("Ambiguous uid1 mapping: columns %s all match; using %r",
                            matches, matches[0])
            col_map[matches[0]] = "uid1"
    if "uid2" not in cols:
        matches = [a for a in ("dst", "target", "node2", "to") if a in cols]
        if matches:
            if len(matches) > 1:
                log.warning("Ambiguous uid2 mapping: columns %s all match; using %r",
                            matches, matches[0])
            col_map[matches[0]] = "uid2"
    if "rel_sum2" not in cols:
        matches = [a for a in ("weight", "w", "value") if a in cols]
        if matches:
            if len(matches) > 1:
                log.warning("Ambiguous rel_sum2 mapping: columns %s all match; using %r",
                            matches, matches[0])
            col_map[matches[0]] = "rel_sum2"
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
    if cfg.hierarchy_postprocess is not None and cfg.hierarchy_postprocess.enabled:
        return False
    if not membership_path.exists():
        return False
    import pyarrow.parquet as pq
    schema = pq.read_schema(membership_path)
    existing = {f.name.removeprefix("cluster_") for f in schema if f.name.startswith("cluster_")}
    level_names = ["nano", "micro", "meso", "macro", "mega"]
    required = set(level_names[:cfg.n_hierarchy_levels])
    return required.issubset(existing)


def _legacy_cluster(edge_path: Path, cfg: "LandscapeConfig", output_dir: Path):
    """Run legacy igraph-based clustering (subsample + Leiden + merge)."""
    edges = _load_and_subsample(edge_path, cfg.n_target_nodes, cfg.seed)
    return _run_clustering(edges, cfg, output_dir)


# ---------------------------------------------------------------------------
# Step 2: Hierarchical clustering (legacy path — extracted to own module)
# ---------------------------------------------------------------------------
from .clustering.landscape_clustering import _run_clustering  # noqa: E402


# ---------------------------------------------------------------------------
# Step 3: Keyword extraction (finest level, auto-detected)
# ---------------------------------------------------------------------------
def _run_keywords(
    membership_path: Path,
    abstract_path: Path,
    cfg: LandscapeConfig,
) -> Tuple[Any, Optional[Dict], Optional[pd.DataFrame]]:
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
        quality_diagnostics_enabled=True,
        quality_rerank_enabled=True,
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
    return result, viz_data, pipeline.abbreviation_evidence


# ---------------------------------------------------------------------------
# Step 4: Generate report
# ---------------------------------------------------------------------------
def _generate_report(
    keywords_df: Any,
    viz_data: Optional[Dict],
    out_dir: Path,
    title: str,
    selection: Mapping[str, Any] | None = None,
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
        selection=selection,
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


def _landscape_report_selection(cfg: LandscapeConfig) -> dict[str, Any]:
    """Return public-safe export selection metadata for run_landscape reports."""

    gamma_low, gamma_high = cfg.gamma_range
    return {
        "scope": "full_landscape_result",
        "view": {"mode": "html_report", "surface": "landscape_pipeline"},
        "filters": [],
        "thresholds": {
            "min_docs_per_cluster": int(cfg.min_docs_per_cluster),
            "gamma_low": float(gamma_low),
            "gamma_high": float(gamma_high),
        },
        "layer_state": {
            "pipeline": "run_landscape",
            "n_target_nodes": int(cfg.n_target_nodes),
            "n_hierarchy_levels": int(cfg.n_hierarchy_levels),
            "leiden_objective": str(cfg.leiden_objective),
            "auto_gamma": bool(cfg.auto_gamma),
            "layer_count": int(len(cfg.layer_paths or {})),
            "cooccurrence_enabled": bool(cfg.cooccurrence_enabled),
            "term_network_enabled": bool(cfg.term_network_enabled),
            "edge_evidence_enabled": bool(cfg.edge_evidence_enabled),
        },
        "focus": {},
    }


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

    # ── Multi-layer combination (if configured) ──────────────
    if cfg.layer_paths:
        from .linkage.combine import combine_edge_layers
        log.info("Multi-layer combination: %d layers, strategy=%s, top_k=%s",
                 len(cfg.layer_paths), cfg.combine_strategy, cfg.combine_top_k)
        layers = {}
        for name, path in cfg.layer_paths.items():
            path = Path(path)
            if path.exists():
                layers[name] = pl.read_parquet(path)
                log.info("  %s: %d edges", name, layers[name].height)
            else:
                log.warning("  %s: file not found: %s", name, path)
        if layers:
            combined = combine_edge_layers(
                layers,
                strategy=cfg.combine_strategy,
                gcc=True,
                top_k=cfg.combine_top_k,
            )
            combined_path = output_dir / "combined_edges.parquet"
            combined.write_parquet(combined_path)
            edge_path = combined_path
            log.info("Combined: %d edges → %s", combined.height, combined_path)

    # Auto-gamma is handled inside build_hierarchy (new path) or
    # _run_clustering (legacy path). No pre-processing needed here.

    # ── Input validation ──────────────────────────────────────
    if not edge_path.exists() and not cfg.layer_paths:
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
    abbreviation_path = output_dir / "abbreviation_pairs.parquet"
    edge_evidence_path = output_dir / "edge_evidence_samples.json"
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
    elif cfg.layer_paths or cfg.auto_gamma:
        # ── New path: build_hierarchy (Rust + consensus + auto-γ per level) ──
        from .clustering.hierarchical import build_hierarchy
        from .clustering.leiden_rust import RUST_AVAILABLE

        if RUST_AVAILABLE:
            log.info("Using build_hierarchy (Rust + consensus + auto-gamma)")
            hier_result = build_hierarchy(
                edges=pl.read_parquet(edge_path) if edge_path.exists() else None,
                layer_paths=cfg.layer_paths,
                n_levels=cfg.n_hierarchy_levels,
                combine_strategy=cfg.combine_strategy,
                combine_top_k=cfg.combine_top_k,
                seed=cfg.seed,
                cache_dir=output_dir,
                hierarchy_postprocess=cfg.hierarchy_postprocess,
                progress=cfg.progress,
            )
            # Build membership DataFrame from hierarchy result
            # Use authoritative UID list from integer_remap (NOT sorted set)
            uids = hier_result.uids
            if uids and hier_result.levels:
                membership_df = hier_result.to_dataframe(uids)
            elif hier_result.levels:
                # Fallback: use integer indices
                data = {"uid": [str(i) for i in range(hier_result.n_nodes)]}
                for level in hier_result.levels:
                    data[f"cluster_{level.name}"] = level.membership.tolist()
                membership_df = pl.DataFrame(data)

            if hier_result.levels:
                membership_df.write_parquet(membership_path)
                log.info("Hierarchy: %d levels, saved → %s",
                         len(hier_result.levels), membership_path)
            else:
                log.warning("build_hierarchy returned no levels, falling back")
                membership_df = _legacy_cluster(edge_path, cfg, output_dir)
        else:
            log.info("Rust not available, using legacy clustering")
            membership_df = _legacy_cluster(edge_path, cfg, output_dir)
    else:
        membership_df = _legacy_cluster(edge_path, cfg, output_dir)

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

        member_uids = membership_df["uid"]
        log.info("Loading abstracts for %s nodes...", f"{member_uids.len():,}")

        abstract_df = (
            pl.scan_parquet(abstract_path)
            .select("uid", "title", "abstract", "pubyear")
            .filter(pl.col("uid").is_in(member_uids))
            .collect()
        )
        log.info("Matched abstracts: %s", f"{len(abstract_df):,}")

        abstract_df.write_parquet(abstract_subset_path, compression="zstd")

        keywords_df, viz_data, abbreviation_evidence = _run_keywords(membership_path, abstract_subset_path, cfg)

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
        if abbreviation_evidence is not None and not abbreviation_evidence.empty:
            abbr_df = abbreviation_evidence.copy()
            if "cluster_supports" in abbr_df.columns:
                abbr_df["cluster_supports"] = abbr_df["cluster_supports"].apply(
                    lambda value: json.dumps(value) if isinstance(value, dict) else value
                )
            abbr_df.to_parquet(abbreviation_path, index=False)
            log.info("Abbreviation evidence saved: %s", abbreviation_path)

    # ------------------------------------------------------------------
    # Step 4: Report (always regenerate — cheap)
    # ------------------------------------------------------------------
    if cfg.edge_evidence_enabled:
        try:
            from .artifacts import write_edge_evidence_samples

            evidence_source_abstracts = abstract_subset_path if abstract_subset_path.exists() else abstract_path
            written_edge_evidence = write_edge_evidence_samples(
                edges_path=edge_path,
                membership_path=membership_path,
                abstracts_path=evidence_source_abstracts,
                output_path=edge_evidence_path,
                max_relations=cfg.edge_evidence_max_relations,
                max_samples_per_relation=cfg.edge_evidence_samples_per_relation,
            )
            if written_edge_evidence is not None:
                log.info("Edge evidence samples saved: %s", written_edge_evidence)
        except Exception as exc:  # pragma: no cover - defensive sidecar generation
            log.warning("Edge evidence sample sidecar skipped: %s", exc)

    log.info("=" * 60)
    log.info("Step 4: Generate landscape report")
    log.info("=" * 60)
    _generate_report(
        keywords_df,
        viz_data,
        report_dir,
        cfg.report_title,
        selection=_landscape_report_selection(cfg),
    )
    cooccurrence_artifacts = None
    try:
        from .artifacts import write_cooccurrence_artifacts

        cooccurrence_artifacts = write_cooccurrence_artifacts(output_dir)
        if cooccurrence_artifacts is not None:
            log.info(
                "Co-occurrence artifacts saved: %s, %s",
                cooccurrence_artifacts["table_path"],
                cooccurrence_artifacts["map_path"],
            )
    except Exception as exc:  # pragma: no cover - defensive sidecar generation
        log.warning("Co-occurrence artifact sidecars skipped: %s", exc)
    result_manifest_path = None
    try:
        from .artifacts import default_result_manifest_path, validate_result_root, write_result_manifest

        write_result_manifest(output_dir)
        result_manifest_path = default_result_manifest_path(validate_result_root(output_dir))
        log.info("Result manifest saved: %s", result_manifest_path)
    except Exception as exc:  # pragma: no cover - defensive manifest generation
        log.warning("Result manifest skipped: %s", exc)

    # Summary
    log.info("=" * 60)
    n_clusters = keywords_df["cluster_id"].nunique()
    log.info("DONE — %d clusters, %d keywords", n_clusters, len(keywords_df))
    label_col = "display_label" if "display_label" in keywords_df.columns else "term"
    score_col = (
        "representative_score"
        if "representative_score" in keywords_df.columns
        else "quality_score" if "quality_score" in keywords_df.columns else "score"
    )
    for cid in sorted(keywords_df["cluster_id"].unique())[:10]:
        grp = keywords_df[keywords_df["cluster_id"] == cid]
        top3 = grp.nlargest(3, score_col)[label_col].astype(str).tolist()
        log.info("  C%d (%d kw): %s", cid, len(grp), ", ".join(top3))
    if n_clusters > 10:
        log.info("  ... (%d more clusters)", n_clusters - 10)
    log.info("=" * 60)

    return {
        "membership_path": membership_path,
        "keywords_path": keywords_path,
        "report_dir": report_dir,
        "edge_evidence_path": edge_evidence_path if edge_evidence_path.exists() else None,
        "cooccurrence_path": (
            cooccurrence_artifacts["table_path"] if cooccurrence_artifacts is not None else None
        ),
        "cooccurrence_map_path": (
            cooccurrence_artifacts["map_path"] if cooccurrence_artifacts is not None else None
        ),
        "result_manifest_path": result_manifest_path,
        "keywords_df": keywords_df,
        "viz_data": viz_data,
    }


__all__ = ["LandscapeConfig", "run_landscape"]
