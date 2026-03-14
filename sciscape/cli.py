"""SciScape CLI — minimal command-line interface.

Usage:
    sciscape cluster  <zip_path> <inner_name> [options]
    sciscape keywords <abstract_parquet> <membership_parquet> [options]
    sciscape convert  <source> <input_file> [options]

Examples:
    sciscape cluster edges.zip edges.txt --levels 5,100 80,500
    sciscape keywords abstracts.parquet membership.parquet --cluster-level cluster_micro
    sciscape keywords abstracts.parquet membership.parquet --top-n 100 --include-title -o keywords.parquet
    sciscape convert wos savedrecs.txt -o abstracts.parquet
    sciscape convert scopus scopus_export.csv -o abstracts.parquet
    sciscape convert openalex works.jsonl -o abstracts.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sciscape",
        description="SciScape: Leiden clustering + keyword extraction toolkit.",
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
    kw.add_argument("--cluster-level", type=str, default="cluster_micro",
                     help="Cluster column name in membership (default: cluster_micro)")
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
            lo, hi = pair.split(",")
            level_constraints.append((int(lo), int(hi)))

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


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "cluster":
        _run_cluster(args)
    elif args.command == "keywords":
        _run_keywords(args)
    elif args.command == "convert":
        _run_convert(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
