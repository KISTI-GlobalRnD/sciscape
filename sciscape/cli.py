"""SciScape CLI.

Usage:
    sciscape query     <search_query> [options]
    sciscape cluster   <zip_path> <inner_name> [options]
    sciscape keywords  <abstract_parquet> <membership_parquet> [options]
    sciscape convert   <source> <input_file> [options]
    sciscape landscape <edge_file> <abstract_parquet> [options]
    sciscape viewer    [options]
    sciscape export    <edge_parquet> <membership_parquet> [options]
    sciscape web       [options]
    sciscape gui

Examples:
    sciscape query "machine learning" --years 2020-2024 --email you@univ.edu -o ml_output/
    sciscape cluster edges.zip edges.txt --levels 5,100 80,500
    sciscape keywords abstracts.parquet membership.parquet --top-n 100 --include-title -o keywords.parquet
    sciscape convert wos savedrecs.txt -o abstracts.parquet
    sciscape landscape edges.parquet abstracts.parquet -o output/landscape
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sciscape",
        description=(
            "SciScape: full-cycle SciSci analysis and visualization for paper networks."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- cluster ----
    cl = sub.add_parser("cluster", help="Run Leiden clustering pipeline")
    cl.add_argument("zip_path", type=Path, help="Path to edge table (ZIP or Parquet)")
    cl.add_argument("inner_name", type=str, help="Filename inside ZIP (ignored for Parquet)")
    cl.add_argument(
        "--levels", nargs="+", metavar="MIN,MAX",
        help="Level constraints as min,max pairs (e.g. 5,100 80,500 400,5000)",
    )
    cl.add_argument("--resolution-bounds", type=str, default="1e-3,5.0",
                     help="Resolution search bounds (default: 1e-3,5.0)")
    cl.add_argument("--max-iterations", type=int, default=32)
    cl.add_argument("--seed", type=int, default=None)
    cl.add_argument("-o", "--output", type=Path, default=Path("membership.parquet"),
                     help="Output membership parquet path")
    cl.add_argument("-v", "--verbose", action="store_true")

    # ---- keywords ----
    kw = sub.add_parser("keywords", help="Run keyword extraction pipeline")
    kw.add_argument("abstract_path", type=Path, help="Abstract parquet file")
    kw.add_argument("membership_path", type=Path, help="Membership parquet file")
    kw.add_argument("--cluster-level", type=str, default=None,
                     help="Cluster column name in membership (default: auto-detect finest)")
    kw.add_argument("--top-n", type=int, default=100, help="Keywords per cluster (default: 100)")
    kw.add_argument("--include-title", action="store_true", help="Include title in text")
    kw.add_argument("--min-df", type=int, default=5, help="Min document frequency (default: 5)")
    kw.add_argument("--ngram-max", type=int, default=3, help="Max n-gram size (default: 3)")
    kw.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs (default: -1 = all)")
    kw.add_argument("--enable-all", action="store_true",
                     help="Enable all optional stages (vocab merge, cooccurrence, term network, depth)")
    kw.add_argument("-o", "--output", type=Path, default=Path("keywords.parquet"),
                     help="Output parquet path")
    kw.add_argument("-v", "--verbose", action="store_true")

    # ---- convert ----
    cv = sub.add_parser("convert", help="Convert external data to SciScape abstract parquet")
    cv.add_argument("source", choices=["wos", "scopus", "openalex", "bibtex"],
                     help="Data source format")
    cv.add_argument("input_file", type=Path, help="Input file path")
    cv.add_argument("-o", "--output", type=Path, default=Path("abstracts.parquet"),
                     help="Output parquet path (default: abstracts.parquet)")
    cv.add_argument("--encoding", type=str, default=None,
                     help="File encoding (default: auto per source)")
    cv.add_argument("--keep-no-abstract", action="store_true",
                     help="Keep rows without abstracts")
    cv.add_argument("-v", "--verbose", action="store_true")

    # ---- landscape ----
    ls = sub.add_parser("landscape", help="Full pipeline: edges → clustering → keywords → report")
    ls.add_argument("abstract_path", type=Path, help="Abstract parquet (uid, title, abstract, pubyear)")
    ls.add_argument("edge_path", type=Path, nargs="?", default=None,
                     help="Edge list file (optional if --layers is used)")
    ls.add_argument("-o", "--output-dir", type=Path, default=Path("landscape_output"),
                     help="Output directory (default: landscape_output)")
    ls.add_argument("--n-nodes", type=int, default=100_000,
                     help="Target node count for BFS subsampling (default: 100000)")
    ls.add_argument("--seed", type=int, default=42)
    ls.add_argument("--min-docs", type=int, default=1000,
                     help="Min documents per cluster (default: 1000)")
    ls.add_argument("--top-n", type=int, default=80, help="Keywords per cluster (default: 80)")
    ls.add_argument("--title", type=str, default="SciScape Landscape", help="Report title")
    ls.add_argument("--gamma-pre", type=str, default="auto",
                     help="Pre-partition γ: 'auto' (10×γ_range upper), 'none' (disable), or float (default: auto)")
    ls.add_argument("--gamma-range", type=str, default=None,
                     help="Resolution search bounds lo,hi (default: 1e-6,1e-3)")
    ls.add_argument("--force", action="store_true",
                     help="Ignore cached intermediate results and re-run from scratch")
    ls.add_argument("--layers", type=str, default=None,
                     help="Multi-layer edge files: name=path,name=path,... "
                          "(e.g. bc=bc.parquet,cc=cc.parquet,dc=dc.parquet)")
    ls.add_argument("--combine-strategy", type=str, default="consensus",
                     choices=["consensus", "rank", "sum", "max", "vote"],
                     help="Edge combination strategy (default: consensus)")
    ls.add_argument("--combine-top-k", type=str, default="auto",
                     help="Per-node top-k filter: 'auto' (sqrt-based), integer, or 'balanced' (default: auto)")
    ls.add_argument("--auto-gamma", action="store_true",
                     help="Auto-select γ (target max cluster < 3%%)")
    ls.add_argument("--auto-gamma-target", type=float, default=3.0,
                     help="Max cluster %% target for auto-gamma (default: 3.0)")
    ls.add_argument("--evaluate", action="store_true",
                     help="Run stability evaluation (AMI/ARI with 5 seeds) + quality report")
    ls.add_argument("-v", "--verbose", action="store_true")

    # ---- viewer ----
    vw = sub.add_parser("viewer", help="Generate standalone viewer HTML (deploy to Vercel/GitHub Pages)")
    vw.add_argument("-o", "--output", type=Path, default=Path("viewer.html"),
                     help="Output HTML path (default: viewer.html)")
    vw.add_argument("--title", type=str, default="SciScape Viewer", help="Viewer title")
    vw.add_argument("--open", action="store_true", help="Open in browser after generation")

    # ---- export ----
    ex = sub.add_parser("export", help="Export network to GEXF (Gephi) or GraphML (Cytoscape)")
    ex.add_argument("edge_path", type=Path, help="Edge parquet file")
    ex.add_argument("membership_path", type=Path, help="Membership parquet file")
    ex.add_argument("-o", "--output", type=Path, default=Path("network.gexf"),
                     help="Output file (default: network.gexf)")
    ex.add_argument("--format", choices=["gexf", "graphml"], default="gexf",
                     help="Export format (default: gexf)")
    ex.add_argument("--abstracts", type=Path, default=None,
                     help="Abstracts parquet for title/year attributes")

    # ---- query (OpenAlex) ----
    qa = sub.add_parser("query", help="Query OpenAlex → fetch → edges → landscape (all-in-one)")
    qa.add_argument("search", type=str, help="Search query (title + abstract)")
    qa.add_argument("--years", type=str, default=None,
                     help="Year range, e.g. 2020-2024")
    qa.add_argument("--max-works", type=int, default=5000,
                     help="Maximum works to fetch (default: 5000)")
    qa.add_argument("--email", type=str, default=None,
                     help="Email for OpenAlex polite pool (10x rate limit)")
    qa.add_argument("--edges", type=str, default="dc,bc",
                     help="Edge types to build (default: dc,bc)")
    qa.add_argument("--no-landscape", action="store_true",
                     help="Skip landscape pipeline (only fetch + edges)")
    qa.add_argument("-o", "--output", type=Path, default=Path("openalex_output"),
                     help="Output directory")
    qa.add_argument("-v", "--verbose", action="store_true")

    # ---- web ----
    wb = sub.add_parser("web", help="Launch web interface (FastAPI)")
    wb.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    wb.add_argument("--port", type=int, default=8000, help="Bind port")
    wb.add_argument("--reload", action="store_true", help="Auto-reload on code changes")

    # ---- gui ----
    sub.add_parser("gui", help="Launch graphical interface")

    return parser


def _run_convert(args: argparse.Namespace) -> None:
    from sciscape.adapters import read_wos, read_scopus, read_openalex, read_bibtex

    common = dict(drop_no_abstract=not args.keep_no_abstract)
    if args.encoding:
        common["encoding"] = args.encoding

    if args.source == "wos":
        df = read_wos(args.input_file, **common)
    elif args.source == "scopus":
        df = read_scopus(args.input_file, **common)
    elif args.source == "openalex":
        df = read_openalex(args.input_file, **common)
    elif args.source == "bibtex":
        df = read_bibtex(args.input_file, **common)
    else:
        raise ValueError(f"Unknown source: {args.source}")

    df.to_parquet(args.output, index=False)
    print(f"Converted {args.source}: {len(df)} documents → {args.output}")
    if args.verbose:
        years = df["pubyear"].dropna()
        if len(years):
            print(f"  Year range: {int(years.min())}–{int(years.max())}")
        print(f"  Avg abstract length: {df['abstract'].str.len().mean():.0f} chars")


def _run_cluster(args: argparse.Namespace) -> None:
    from sciscape.clustering import LeidenConfig, run_pipeline

    level_constraints = None
    if args.levels:
        level_constraints = []
        for pair in args.levels:
            try:
                lo, hi = pair.split(",")
                level_constraints.append((int(lo), int(hi)))
            except ValueError:
                print(f"Invalid --levels format: {pair!r}. Expected: min,max (e.g., 5,100)",
                      file=sys.stderr)
                sys.exit(1)

    lo, hi = args.resolution_bounds.split(",")
    resolution_bounds = (float(lo), float(hi))

    config = LeidenConfig(
        level_constraints=level_constraints,
        resolution_bounds=resolution_bounds,
        max_iterations=args.max_iterations,
        seed=args.seed,
        log_history=args.verbose,
    )

    print(f"Running clustering: {args.zip_path}")
    tables = run_pipeline(
        zip_path=args.zip_path,
        inner_name=args.inner_name,
        config=config,
    )

    tables.membership.write_parquet(args.output)
    print(f"Membership saved: {args.output} ({len(tables.membership)} rows)")

    desc_path = args.output.with_name(args.output.stem + "_description.parquet")
    tables.description.write_parquet(desc_path)
    print(f"Description saved: {desc_path} ({len(tables.description)} rows)")

    if tables.resolutions:
        for level, gamma in tables.resolutions.items():
            quality = tables.qualities.get(level, 0.0) if tables.qualities else 0.0
            print(f"  {level}: gamma={gamma:.4f}, quality={quality:.4f}")


def _run_keywords(args: argparse.Namespace) -> None:
    from sciscape.keyword_extraction import KeywordExtractionConfig, run_keyword_pipeline
    from sciscape.keyword_extraction.config import VocabMergeConfig
    from sciscape.keyword_extraction.depth import DepthConfig
    from sciscape.keyword_extraction.term_network import TermNetworkConfig

    kwargs = dict(
        abstract_path=args.abstract_path,
        membership_path=args.membership_path,
        cluster_level=args.cluster_level,
        top_n_keywords=args.top_n,
        include_title=args.include_title,
        min_df_unigram=args.min_df,
        min_df_phrase=args.min_df,
        ngram_min=2,
        ngram_max=args.ngram_max,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )

    if args.enable_all:
        kwargs.update(
            vocab_merge=VocabMergeConfig(enabled=True),
            normalization_enabled=True,
            norm_plural_merge_enabled=True,
            academic_stopwords_enabled=True,
            artifact_filter_enabled=True,
            cross_cluster_penalty_enabled=True,
            fragment_suppression_enabled=True,
            cooccurrence_enabled=True,
            term_network=TermNetworkConfig(
                enabled=True,
                layers=["string", "token", "cooccurrence"],
            ),
            auto_merge_enabled=True,
            short_term_expansion_enabled=True,
            depth=DepthConfig(enabled=True, n_levels=3),
        )

    cfg = KeywordExtractionConfig(**kwargs)

    print(f"Running keyword extraction: {args.abstract_path}")
    print(f"  cluster_level={args.cluster_level}, top_n={args.top_n}")
    keywords = run_keyword_pipeline(cfg)

    # Serialize dict columns for parquet compatibility
    import json
    save_df = keywords.copy()
    dict_cols = ("pub_year_series", "year_denominators", "ppm_series",
                 "loglift_series", "bayesian_log_odds_series")
    for col in dict_cols:
        if col in save_df.columns:
            save_df[col] = save_df[col].apply(
                lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v
            )

    save_df.to_parquet(args.output, index=False)
    print(f"Keywords saved: {args.output} ({len(keywords)} rows)")

    # Summary
    for cid, grp in keywords.groupby("cluster_id"):
        top3 = grp.nlargest(3, "score")["term"].tolist()
        print(f"  cluster {cid}: {len(grp)} keywords — {top3}")


def _run_landscape(args: argparse.Namespace) -> None:
    import logging
    from sciscape.landscape import LandscapeConfig, run_landscape

    if args.verbose:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(message)s",
                            datefmt="%H:%M:%S")

    # Parse gamma_pre: "auto" | "none" | float
    gb = args.gamma_pre.strip().lower()
    if gb == "none":
        gamma_pre = None
    elif gb == "auto":
        gamma_pre = "auto"
    else:
        gamma_pre = float(gb)

    cfg_kwargs = dict(
        n_target_nodes=args.n_nodes,
        seed=args.seed,
        force=args.force,
        min_docs_per_cluster=args.min_docs,
        top_n_keywords=args.top_n,
        report_title=args.title,
        gamma_pre=gamma_pre,
    )
    if args.gamma_range:
        try:
            lo, hi = args.gamma_range.split(",")
            cfg_kwargs["gamma_range"] = (float(lo), float(hi))
        except ValueError:
            print(f"Invalid --gamma-range format. Expected: lo,hi (e.g., 1e-5,1e-2)",
                  file=sys.stderr)
            sys.exit(1)

    # Multi-layer combination
    if args.layers:
        layer_paths = {}
        for item in args.layers.split(","):
            if "=" in item:
                name, path = item.split("=", 1)
                layer_paths[name.strip()] = Path(path.strip())
            else:
                layer_paths[Path(item).stem] = Path(item.strip())
        cfg_kwargs["layer_paths"] = layer_paths
        cfg_kwargs["combine_strategy"] = args.combine_strategy
        # Parse combine_top_k: "auto", "balanced", or integer
        tk = args.combine_top_k
        if tk not in ("auto", "balanced"):
            try:
                tk = int(tk)
            except ValueError:
                tk = "auto"
        cfg_kwargs["combine_top_k"] = tk

    if args.auto_gamma:
        cfg_kwargs["auto_gamma"] = True
        cfg_kwargs["auto_gamma_target"] = args.auto_gamma_target

    cfg = LandscapeConfig(**cfg_kwargs)

    # Determine edge_path
    edge_path = args.edge_path
    if edge_path is None and not args.layers:
        print("Error: provide either edge_path or --layers", file=sys.stderr)
        sys.exit(1)
    if edge_path is None:
        # Placeholder — will be replaced by combined edges inside run_landscape
        edge_path = args.output_dir / "_placeholder_edges.parquet"

    result = run_landscape(edge_path, args.abstract_path, args.output_dir, config=cfg)
    print(f"Landscape complete → {result['report_dir']}/report.html")

    # Optional evaluation
    if getattr(args, "evaluate", False):
        try:
            from sciscape.evaluation.stability import evaluate_stability, compute_quality_report
            import polars as pl
            import numpy as np

            # Load edges and membership
            membership_path = args.output_dir / "membership.parquet"
            if not membership_path.exists():
                print("Evaluation skipped: membership.parquet not found")
            elif not edge_path or not edge_path.exists():
                print(f"Evaluation skipped: edge file not found at {edge_path}")
            elif membership_path.exists():
                edges = pl.read_parquet(edge_path)
                mem_df = pl.read_parquet(membership_path)
                cluster_col = next((c for c in mem_df.columns if c.startswith("cluster_")), None)
                if cluster_col:
                    membership = mem_df[cluster_col].to_numpy()
                    gamma = result.get("gamma", 1.0)

                    # Stability
                    print("\n--- Stability Evaluation ---")
                    min_sz = getattr(cfg, 'min_docs_per_cluster', None) or args.min_docs or 10
                    stab = evaluate_stability(edges, gamma=gamma, n_seeds=5,
                                              min_size=min_sz)
                    print(stab.summary())

                    # Quality report
                    print("\n--- Quality Report ---")
                    qr = compute_quality_report(edges, membership, gamma=gamma,
                                                target_pct=cfg.auto_gamma_target)
                    print(qr.summary())
        except Exception as e:
            print(f"Evaluation skipped: {e}")


def _run_viewer(args: argparse.Namespace) -> None:
    from sciscape.keyword_extraction.visualization import export_viewer

    path = export_viewer(
        output_path=str(args.output),
        title=args.title,
    )
    print(f"Viewer generated: {path}")
    print("Deploy to Vercel:  vercel deploy --prod .")
    print("Or open locally:   open viewer.html")

    if args.open:
        import webbrowser
        webbrowser.open(f"file://{path}")


def _run_export(args: argparse.Namespace) -> None:
    import polars as pl
    from sciscape.export import export_gexf, export_graphml

    edges = pl.read_parquet(args.edge_path)
    membership = pl.read_parquet(args.membership_path)
    abstracts = pl.read_parquet(args.abstracts) if args.abstracts else None

    if args.format == "graphml":
        path = export_graphml(edges, membership, args.output, abstracts=abstracts)
    else:
        path = export_gexf(edges, membership, args.output, abstracts=abstracts)
    print(f"Exported → {path}")


def _run_query(args: argparse.Namespace) -> None:
    from sciscape.openalex import run_openalex_pipeline, OpenAlexPipelineConfig

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    filters = {}
    if args.years:
        filters["publication_year"] = args.years

    config = OpenAlexPipelineConfig(
        query=args.search,
        filters=filters,
        max_works=args.max_works,
        email=args.email,
        edge_types=args.edges.split(","),
        output_dir=Path(args.output),
        run_landscape=not args.no_landscape,
        progress=print,
    )
    result = run_openalex_pipeline(config)
    print(f"\nDone: {result.n_works} works, {result.n_edges} edges")
    if result.abstracts_path:
        print(f"  Abstracts: {result.abstracts_path}")
    if result.edges_path:
        print(f"  Edges: {result.edges_path}")
    if result.landscape_dir:
        print(f"  Landscape: {result.landscape_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "cluster":
        _run_cluster(args)
    elif args.command == "keywords":
        _run_keywords(args)
    elif args.command == "convert":
        _run_convert(args)
    elif args.command == "landscape":
        _run_landscape(args)
    elif args.command == "viewer":
        _run_viewer(args)
    elif args.command == "export":
        _run_export(args)
    elif args.command == "query":
        _run_query(args)
    elif args.command == "web":
        import uvicorn
        uvicorn.run(
            "sciscape.web.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    elif args.command == "gui":
        from sciscape.gui import launch
        launch()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
