"""SciScape CLI.

Usage:
    sciscape query     <search_query> [options]
    sciscape cluster   <zip_path> <inner_name> [options]
    sciscape keywords  <abstract_parquet> <membership_parquet> [options]
    sciscape convert   <source> <input_file> [options]
    sciscape landscape <edge_file> <abstract_parquet> [options]
    sciscape visualize <keyword_table> [options]
    sciscape viewer    [options]
    sciscape evolution-evidence <records_table> <membership_table> [options]
    sciscape evolution-from-membership <result_root> <records_table> <membership_table> [options]
    sciscape evolution-from-slice-membership <result_root> <slice_membership_table> [options]
    sciscape evolution-from-slice-reclustering <result_root> <records_table> <edge_table> [options]
    sciscape evolution <result_root> <slices_table> <state_evidence_table> [transition_evidence_table] [options]
    sciscape export    <edge_parquet> <membership_parquet> [options]
    sciscape rule-export <rule_manifest> [options]
    sciscape matrix    wrap-term-cooccurrence <result_root> [options]
    sciscape matrix    export <result_or_matrix> [options]
    sciscape bundle    vosviewer <result_root> [options]
    sciscape web       [options]
    sciscape gui

Examples:
    sciscape query "machine learning" --years 2020-2024 --email you@univ.edu -o workspace/openalex_output/ml
    sciscape cluster edges.zip edges.txt --levels 5,100 80,500
    sciscape keywords abstracts.parquet membership.parquet --top-n 100 --include-title -o keywords.parquet
    sciscape convert wos savedrecs.txt -o abstracts.parquet
    sciscape landscape edges.parquet abstracts.parquet -o workspace/output/landscape
    sciscape visualize keywords.parquet -o workspace/reports/keywords
    sciscape evolution-evidence abstracts.parquet membership.parquet -o workspace/evolution_evidence
    sciscape evolution-from-membership result abstracts.parquet membership.parquet --periodization '{"window_years":2}'
    sciscape evolution-from-slice-membership result slice_membership.parquet --slices-table slices.parquet
    sciscape evolution-from-slice-reclustering result abstracts.parquet edges.parquet --periodization '{"window_years":2}'
    sciscape evolution result slices.parquet states.parquet transitions.parquet --metric term_overlap
    sciscape evolution result slices.parquet states.parquet --derive-transitions document-overlap --state-membership-table state_membership.parquet
    sciscape rule-export result/rules/keyword_cleaning_default_v1/rule_set_manifest.json -o result/vosviewer
    sciscape matrix wrap-term-cooccurrence result
    sciscape matrix export result --matrix-id term_cooccurrence_default --format csv-triplets
    sciscape bundle vosviewer result --ensure-term-exports
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _parse_int_csv(value: str) -> tuple[int, ...]:
    try:
        ids = tuple(sorted({int(part.strip()) for part in str(value).split(",") if part.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integer IDs") from exc
    if not ids:
        raise argparse.ArgumentTypeError("expected at least one shard ID")
    if any(item < 0 for item in ids):
        raise argparse.ArgumentTypeError("shard IDs must be non-negative")
    return ids


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
    kw.add_argument(
        "--keyword-engine",
        choices=["legacy", "cluster_sharded"],
        default="legacy",
        help="Keyword engine (default: legacy; cluster_sharded is opt-in V2)",
    )
    kw.add_argument(
        "--cluster-sharded-output-dir",
        type=Path,
        default=None,
        help="Artifact directory for --keyword-engine cluster_sharded",
    )
    kw.add_argument(
        "--keyword-preflight-only",
        action="store_true",
        help="For --keyword-engine cluster_sharded, write shard/budget manifests and exit",
    )
    kw.add_argument("--uid-col", type=str, default="uid", help="Document id column (default: uid)")
    kw.add_argument("--title-col", type=str, default="title", help="Title column (default: title)")
    kw.add_argument("--abstract-col", type=str, default="abstract", help="Abstract column (default: abstract)")
    kw.add_argument("--year-col", type=str, default="pubyear", help="Publication year column (default: pubyear)")
    kw.add_argument("--target-docs-per-shard", type=int, default=500_000)
    kw.add_argument("--max-clusters-per-shard", type=int, default=1024)
    kw.add_argument(
        "--cluster-sharded-shard-ids",
        type=_parse_int_csv,
        default=None,
        help="For --keyword-engine cluster_sharded, rerun only these comma-separated shard IDs",
    )
    kw.add_argument("--candidate-pool-floor", type=int, default=256)
    kw.add_argument("--candidate-pool-large", type=int, default=1024)
    kw.add_argument("--candidate-pool-hard-max", type=int, default=1536)
    kw.add_argument("--global-candidate-row-warning", type=int, default=80_000_000)
    kw.add_argument("--global-candidate-row-hard-stop", type=int, default=100_000_000)
    kw.add_argument("--global-unique-term-warning", type=int, default=8_000_000)
    kw.add_argument("--global-unique-term-hard-stop", type=int, default=10_000_000)
    kw.add_argument("--candidate-mining-progress-interval-docs", type=int, default=25_000)
    kw.add_argument("--candidate-mining-prune-interval-docs", type=int, default=50_000)
    kw.add_argument("--candidate-mining-prune-multiplier", type=int, default=8)
    kw.add_argument("--include-title", action="store_true", help="Include title in text")
    kw.add_argument("--min-df", type=int, default=5, help="Min document frequency (default: 5)")
    kw.add_argument("--ngram-max", type=int, default=3, help="Max n-gram size (default: 3)")
    kw.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs (default: -1 = all)")
    kw.add_argument(
        "--parallel-backend",
        choices=["auto", "loky", "threading", "sequential"],
        default="auto",
        help="Cluster task backend; auto uses threads for large cluster counts",
    )
    kw.add_argument("--parallel-large-cluster-threshold", type=int, default=1000)
    kw.add_argument("--progress-path", type=Path, default=None, help="Write progress JSON here")
    kw.add_argument("--progress-interval-clusters", type=int, default=100)
    kw.add_argument("--scoring-shard-dir", type=Path, default=None, help="Write Stage 4 scoring shards here")
    kw.add_argument(
        "--scoring-shard-size-clusters",
        type=int,
        default=0,
        help="Clusters per scoring shard (0 uses default when --scoring-shard-dir is set)",
    )
    kw.add_argument(
        "--scoring-shard-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse complete matching scoring shards when --scoring-shard-dir is set",
    )
    kw.add_argument(
        "--quality-rerank",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use representative-label quality for final top-N ranking",
    )
    kw.add_argument(
        "--keyword-rule-artifact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write replayable keyword cleaning rule artifacts when a result root is available",
    )
    kw.add_argument(
        "--keyword-rule-set-id",
        type=str,
        default="keyword_cleaning_default_v1",
        help="Rule-set ID for keyword cleaning artifacts",
    )
    kw.add_argument(
        "--keyword-rule-result-root",
        type=Path,
        default=None,
        help="Result root for keyword cleaning artifacts; inferred for <root>/landscape/keywords.parquet outputs",
    )
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

    # ---- visualize ----
    vz = sub.add_parser("visualize", help="Generate dashboard/report from a keyword table")
    vz.add_argument(
        "keyword_table",
        type=Path,
        help="Keyword table from sciscape keywords/landscape (.parquet, .csv, .tsv, .json, .jsonl)",
    )
    vz.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("workspace/reports/sciscape_report"),
        help="Output report directory, or HTML path with --dashboard-only (default: workspace/reports/sciscape_report)",
    )
    vz.add_argument("--title", type=str, default="SciScape Keyword Report", help="Report title")
    vz.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Write one standalone dashboard HTML instead of a multi-page report",
    )
    vz.add_argument("--open", action="store_true", help="Open the generated output in a browser")

    # ---- evolution evidence ----
    ee = sub.add_parser("evolution-evidence", help="Build slice-local evidence tables for evolution")
    ee.add_argument("records_table", type=Path, help="Records/abstract table with uid and publication year")
    ee.add_argument("membership_table", type=Path, help="Membership table with uid and cluster/cluster_* column")
    ee.add_argument("-o", "--output-dir", type=Path, default=Path("evolution_evidence"), help="Output directory")
    ee.add_argument("--keywords-table", type=Path, default=None, help="Optional keyword table for state labels")
    ee.add_argument("--evolution-id", type=str, default="cluster_evolution", help="Evolution artifact ID")
    ee.add_argument("--cluster-column", type=str, default=None, help="Membership cluster column (default: auto-detect)")
    ee.add_argument("--uid-column", type=str, default=None, help="Record document ID column (default: auto-detect)")
    ee.add_argument("--membership-uid-column", type=str, default=None, help="Membership document ID column (default: auto-detect)")
    ee.add_argument(
        "--representative-work-limit",
        type=int,
        default=50,
        help="Max representative IDs stored in each state row (default: 50)",
    )
    ee.add_argument("--periodization", type=str, default=None, help="Inline JSON object or path for slice metadata")
    ee.add_argument("--output-format", choices=["parquet", "csv"], default="parquet", help="Table output format")
    ee.add_argument("--json", action="store_true", help="Print a JSON summary")

    # ---- evolution from membership ----
    efm = sub.add_parser("evolution-from-membership", help="Write evolution artifacts from records and membership")
    efm.add_argument("result_root", type=Path, help="SciScape result root to receive evolution/")
    efm.add_argument("records_table", type=Path, help="Records/abstract table with uid and publication year")
    efm.add_argument("membership_table", type=Path, help="Membership table with uid and cluster/cluster_* column")
    efm.add_argument("--keywords-table", type=Path, default=None, help="Optional keyword table for state labels")
    efm.add_argument("--evolution-id", type=str, default="cluster_evolution", help="Evolution artifact ID")
    efm.add_argument("--metric", type=str, default="overlap_min", help="Document-overlap metric")
    efm.add_argument("--title", type=str, default=None, help="Evolution artifact title")
    efm.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: <result_root>/evolution)")
    efm.add_argument("--temporal-manifest", type=Path, default=None, help="Optional temporal_manifest.json source ref")
    efm.add_argument("--cluster-column", type=str, default=None, help="Membership cluster column (default: auto-detect)")
    efm.add_argument("--uid-column", type=str, default=None, help="Record document ID column (default: auto-detect)")
    efm.add_argument("--membership-uid-column", type=str, default=None, help="Membership document ID column (default: auto-detect)")
    efm.add_argument("--representative-work-limit", type=int, default=50, help="Max representative IDs stored in each state row")
    efm.add_argument("--min-transition-score", type=float, default=0.5, help="Minimum transition score")
    efm.add_argument("--min-support-count", type=int, default=1, help="Minimum transition support count")
    efm.add_argument("--matching-method", type=str, default=None, help="Inline JSON object or path for matching metadata")
    efm.add_argument("--event-rules", type=str, default=None, help="Inline JSON object or path for event rules")
    efm.add_argument(
        "--periodization",
        type=str,
        default=None,
        help="Inline JSON object or path for slice metadata (default: 2-year rolling windows)",
    )
    efm.add_argument("--entity-scope", type=str, default=None, help="Inline JSON object or path for entity-scope metadata")
    efm.add_argument(
        "--allow-incomplete-state-membership",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow state membership doc_count mismatches and flag affected transitions",
    )
    efm.add_argument("--json", action="store_true", help="Print a JSON summary")

    # ---- evolution from slice-local membership ----
    efsm = sub.add_parser(
        "evolution-from-slice-membership",
        help="Write evolution artifacts from slice-local clustering membership",
    )
    efsm.add_argument("result_root", type=Path, help="SciScape result root to receive evolution/")
    efsm.add_argument(
        "slice_membership_table",
        type=Path,
        help="Slice-local membership table with slice_id, uid, and cluster/cluster_id/cluster_* column",
    )
    efsm.add_argument("--slices-table", type=Path, default=None, help="Optional explicit time-slice table")
    efsm.add_argument("--keywords-table", type=Path, default=None, help="Optional slice-aware keyword table for state labels")
    efsm.add_argument("--evolution-id", type=str, default="cluster_evolution", help="Evolution artifact ID")
    efsm.add_argument("--metric", type=str, default="overlap_min", help="Document-overlap metric")
    efsm.add_argument("--title", type=str, default=None, help="Evolution artifact title")
    efsm.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: <result_root>/evolution)")
    efsm.add_argument("--temporal-manifest", type=Path, default=None, help="Optional temporal_manifest.json source ref")
    efsm.add_argument("--cluster-column", type=str, default=None, help="Membership cluster column (default: auto-detect)")
    efsm.add_argument("--uid-column", type=str, default=None, help="Document ID column (default: auto-detect)")
    efsm.add_argument("--slice-id-column", type=str, default="slice_id", help="Slice ID column (default: slice_id)")
    efsm.add_argument("--default-level", type=str, default="cluster", help="Default cluster level for cluster_id columns")
    efsm.add_argument("--representative-work-limit", type=int, default=50, help="Max representative IDs stored in each state row")
    efsm.add_argument("--min-transition-score", type=float, default=0.5, help="Minimum transition score")
    efsm.add_argument("--min-support-count", type=int, default=1, help="Minimum transition support count")
    efsm.add_argument("--matching-method", type=str, default=None, help="Inline JSON object or path for matching metadata")
    efsm.add_argument("--event-rules", type=str, default=None, help="Inline JSON object or path for event rules")
    efsm.add_argument("--entity-scope", type=str, default=None, help="Inline JSON object or path for entity-scope metadata")
    efsm.add_argument(
        "--allow-incomplete-state-membership",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow state membership doc_count mismatches and flag affected transitions",
    )
    efsm.add_argument("--json", action="store_true", help="Print a JSON summary")

    # ---- evolution from slice-local reclustering ----
    efsr = sub.add_parser(
        "evolution-from-slice-reclustering",
        help="Write evolution artifacts by reclustering induced slice graphs",
    )
    efsr.add_argument("result_root", type=Path, help="SciScape result root to receive evolution/")
    efsr.add_argument("records_table", type=Path, help="Records/abstract table with uid and publication year")
    efsr.add_argument("edge_table", type=Path, help="Paper-level edge table")
    efsr.add_argument("--keywords-table", type=Path, default=None, help="Optional keyword table for state labels")
    efsr.add_argument("--evolution-id", type=str, default="cluster_evolution", help="Evolution artifact ID")
    efsr.add_argument("--metric", type=str, default="overlap_min", help="Document-overlap metric")
    efsr.add_argument("--title", type=str, default=None, help="Evolution artifact title")
    efsr.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: <result_root>/evolution)")
    efsr.add_argument("--temporal-manifest", type=Path, default=None, help="Optional temporal_manifest.json source ref")
    efsr.add_argument("--uid-column", type=str, default=None, help="Record document ID column (default: auto-detect)")
    efsr.add_argument("--edge-source-column", type=str, default=None, help="Edge source document ID column (default: auto-detect)")
    efsr.add_argument("--edge-target-column", type=str, default=None, help="Edge target document ID column (default: auto-detect)")
    efsr.add_argument("--edge-weight-column", type=str, default=None, help="Edge weight column (default: rel_sum2/weight/w/value or 1.0)")
    efsr.add_argument("--resolution", type=float, default=1.0, help="Leiden/CPM resolution for each slice")
    efsr.add_argument("--objective", choices=["cpm", "modularity"], default="cpm", help="Leiden objective")
    efsr.add_argument("--backend", choices=["auto", "rust", "igraph"], default="auto", help="Slice clustering backend")
    efsr.add_argument("--seed", type=int, default=0, help="Leiden random seed")
    efsr.add_argument("--n-iterations", type=int, default=10, help="Leiden iterations per slice")
    efsr.add_argument("--min-docs-per-slice", type=int, default=1, help="Skip slices with fewer documents")
    efsr.add_argument(
        "--slice-reclustering-workers",
        type=int,
        default=1,
        help="Maximum slice reclustering workers (default: 1)",
    )
    efsr.add_argument(
        "--slice-membership-output",
        type=Path,
        default=None,
        help="Optional Parquet path for generated slice-local membership rows",
    )
    efsr.add_argument(
        "--slice-membership-parts-dir",
        type=Path,
        default=None,
        help="Optional directory for per-slice membership checkpoint parts",
    )
    efsr.add_argument(
        "--progress-path",
        type=Path,
        default=None,
        help="Progress JSON path (default: <result_root>/evolution_work/slice_reclustering_progress.json)",
    )
    efsr.add_argument("--representative-work-limit", type=int, default=50, help="Max representative IDs stored in each state row")
    efsr.add_argument("--min-transition-score", type=float, default=0.5, help="Minimum transition score")
    efsr.add_argument("--min-support-count", type=int, default=1, help="Minimum transition support count")
    efsr.add_argument("--matching-method", type=str, default=None, help="Inline JSON object or path for matching metadata")
    efsr.add_argument("--event-rules", type=str, default=None, help="Inline JSON object or path for event rules")
    efsr.add_argument(
        "--periodization",
        type=str,
        default=None,
        help="Inline JSON object or path for slice metadata (default: 2-year rolling windows)",
    )
    efsr.add_argument("--entity-scope", type=str, default=None, help="Inline JSON object or path for entity-scope metadata")
    efsr.add_argument(
        "--allow-incomplete-state-membership",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow state membership doc_count mismatches and flag affected transitions",
    )
    efsr.add_argument("--json", action="store_true", help="Print a JSON summary")

    # ---- evolution ----
    ev = sub.add_parser("evolution", help="Write artifact-backed cluster evolution tables")
    ev.add_argument("result_root", type=Path, help="SciScape result root to receive evolution/")
    ev.add_argument("slices_table", type=Path, help="Time-slice table (.parquet, .csv, .tsv, .json, .jsonl)")
    ev.add_argument("state_evidence_table", type=Path, help="Slice-local cluster state evidence table")
    ev.add_argument("transition_evidence_table", type=Path, nargs="?", help="State transition evidence table")
    ev.add_argument(
        "--derive-transitions",
        choices=["explicit", "document-overlap"],
        default="explicit",
        help="Transition source (default: explicit transition evidence table)",
    )
    ev.add_argument(
        "--state-membership-table",
        type=Path,
        default=None,
        help="State-document membership table for --derive-transitions document-overlap",
    )
    ev.add_argument(
        "--state-membership-uid-column",
        type=str,
        default=None,
        help="Document ID column in --state-membership-table (default: auto-detect)",
    )
    ev.add_argument(
        "--state-membership-state-id-column",
        type=str,
        default="state_id",
        help="State ID column in --state-membership-table (default: state_id)",
    )
    ev.add_argument(
        "--allow-incomplete-state-membership",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow state membership doc_count mismatches and flag affected transitions",
    )
    ev.add_argument("--evolution-id", type=str, default="cluster_evolution", help="Evolution artifact ID")
    ev.add_argument("--metric", type=str, default="transition_score", help="Transition metric name")
    ev.add_argument("--title", type=str, default=None, help="Evolution artifact title")
    ev.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: <result_root>/evolution)")
    ev.add_argument("--temporal-manifest", type=Path, default=None, help="Optional temporal_manifest.json source ref")
    ev.add_argument("--default-level", type=str, default="cluster", help="Default cluster level for state evidence")
    ev.add_argument(
        "--allow-skip-slices",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow transitions across non-adjacent slices",
    )
    ev.add_argument("--min-transition-score", type=float, default=0.5, help="Minimum transition score")
    ev.add_argument("--min-support-count", type=int, default=1, help="Minimum transition support count")
    ev.add_argument("--matching-method", type=str, default=None, help="Inline JSON object or path for matching metadata")
    ev.add_argument("--event-rules", type=str, default=None, help="Inline JSON object or path for event rules")
    ev.add_argument("--periodization", type=str, default=None, help="Inline JSON object or path for slice metadata")
    ev.add_argument("--entity-scope", type=str, default=None, help="Inline JSON object or path for entity-scope metadata")

    # ---- export ----
    ex = sub.add_parser("export", help="Export network to GEXF, GraphML, or VOSviewer-style files")
    ex.add_argument("edge_path", type=Path, help="Edge parquet file")
    ex.add_argument("membership_path", type=Path, help="Membership parquet file")
    ex.add_argument("-o", "--output", type=Path, default=Path("network.gexf"),
                     help="Output file for GEXF/GraphML or directory for VOSviewer-style export")
    ex.add_argument("--format", choices=["gexf", "graphml", "vosviewer"], default="gexf",
                     help="Export format (default: gexf)")
    ex.add_argument("--abstracts", type=Path, default=None,
                     help="Abstracts parquet for title/year attributes")

    # ---- rule-export ----
    rex = sub.add_parser("rule-export", help="Export keyword cleaning rules to external tool formats")
    rex.add_argument("rule_manifest", type=Path, help="Keyword rule_set_manifest.json path or rule directory")
    rex.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("vosviewer"),
        help="Output directory for rule export files",
    )
    rex.add_argument(
        "--format",
        choices=["vosviewer-thesaurus"],
        default="vosviewer-thesaurus",
        help="Rule export format (default: vosviewer-thesaurus)",
    )
    rex.add_argument(
        "--result-root",
        type=Path,
        default=None,
        help="Result root for manifest-backed exports; inferred from rules/<rule_set_id>/ when possible",
    )
    rex.add_argument(
        "--write-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a manifest-backed export artifact",
    )
    rex.add_argument(
        "--thesaurus-filename",
        type=str,
        default="vosviewer_thesaurus.txt",
        help="Output filename for the VOSviewer thesaurus table",
    )
    rex.add_argument(
        "--rule-set-filename",
        type=str,
        default="sciscape_keyword_rules.tsv",
        help="Output filename for the companion SciScape rule-set table",
    )

    # ---- matrix ----
    mx = sub.add_parser("matrix", help="Build or adapt matrix artifacts")
    mx_sub = mx.add_subparsers(dest="matrix_command", required=True)
    mtc = mx_sub.add_parser(
        "wrap-term-cooccurrence",
        help="Wrap term co-occurrence sidecars as a general matrix artifact",
    )
    mtc.add_argument(
        "result_root",
        type=Path,
        help="SciScape result root with landscape/term_cooccurrence or keywords/report data",
    )
    mtc.add_argument(
        "--matrix-id",
        type=str,
        default="term_cooccurrence_default",
        help="Matrix artifact ID under matrices/ (default: term_cooccurrence_default)",
    )
    mtc.add_argument(
        "--json",
        action="store_true",
        help="Print written artifact paths and QA as JSON",
    )
    mex = mx_sub.add_parser(
        "export",
        help="Export a matrix artifact as manifest-backed table or summary files",
    )
    mex.add_argument(
        "matrix",
        type=Path,
        help="Result root, matrix directory, or matrix_manifest.json path",
    )
    mex.add_argument(
        "--matrix-id",
        type=str,
        default="term_cooccurrence_default",
        help="Matrix ID when the input is a result root (default: term_cooccurrence_default)",
    )
    mex.add_argument(
        "--format",
        dest="export_format",
        choices=["csv-triplets", "parquet-triplets", "json-summary", "vosviewer-network"],
        default="csv-triplets",
        help="Matrix export format (default: csv-triplets)",
    )
    mex.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory inside the result root (default: exports/<matrix_id>_<format>)",
    )
    mex.add_argument(
        "--json",
        action="store_true",
        help="Print written export paths as JSON",
    )

    # ---- bundle ----
    bd = sub.add_parser("bundle", help="Package manifest-backed export files")
    bd_sub = bd.add_subparsers(dest="bundle_command", required=True)
    bvos = bd_sub.add_parser(
        "vosviewer",
        help="Package manifest-backed VOSviewer exports into one zip file",
    )
    bvos.add_argument(
        "result_root",
        type=Path,
        help="SciScape result root with manifest-backed VOSviewer exports",
    )
    bvos.add_argument(
        "--ensure-term-exports",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Before bundling, create compatible term co-occurrence VOSviewer exports when possible",
    )
    bvos.add_argument(
        "--json",
        action="store_true",
        help="Print written bundle paths as JSON",
    )

    # ---- query (OpenAlex) ----
    qa = sub.add_parser("query", help="Query OpenAlex → fetch → edges → landscape (all-in-one)")
    qa.add_argument("search", type=str, help="Search query (title + abstract)")
    qa.add_argument("--years", type=str, default=None,
                     help="Year range, e.g. 2020-2024")
    qa.add_argument("--max-works", type=int, default=5000,
                     help="Maximum works to fetch (default: 5000)")
    qa.add_argument("--email", type=str, default=None,
                     help="Email for OpenAlex polite pool (10x rate limit)")
    qa.add_argument("--request-timeout", type=float, default=30.0,
                     help="OpenAlex HTTP request timeout in seconds (default: 30)")
    qa.add_argument("--max-retries", type=int, default=3,
                     help="Maximum OpenAlex retry attempts for 429/5xx/timeouts (default: 3)")
    qa.add_argument("--backoff-base", type=float, default=1.0,
                     help="Initial OpenAlex retry backoff in seconds (default: 1)")
    qa.add_argument("--backoff-max", type=float, default=30.0,
                     help="Maximum OpenAlex retry or Retry-After sleep in seconds (default: 30)")
    qa.add_argument("--api-attempt-budget", type=int, default=None,
                     help="Abort OpenAlex query after this many HTTP attempts")
    qa.add_argument("--retry-wait-budget-seconds", type=float, default=None,
                     help="Abort OpenAlex query after this much accumulated retry sleep")
    qa.add_argument("--interruptible-requests", action=argparse.BooleanOptionalAction, default=False,
                     help="Poll cancellation checkpoints while OpenAlex HTTP requests are in flight")
    qa.add_argument("--request-poll-interval", type=float, default=0.25,
                     help="OpenAlex in-flight request checkpoint poll interval in seconds (default: 0.25)")
    qa.add_argument("--edges", type=str, default="dc,bc",
                     help="Edge types to build (default: dc,bc)")
    qa.add_argument("--no-landscape", action="store_true",
                     help="Skip landscape pipeline (only fetch + edges)")
    qa.add_argument("-o", "--output", type=Path, default=Path("workspace/openalex_output"),
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


def _infer_keyword_rule_result_root(output: Path, explicit_root: Path | None) -> Path | None:
    if explicit_root is not None:
        return explicit_root
    if output.name == "keywords.parquet" and output.parent.name == "landscape":
        return output.parent.parent
    return None


def _run_keywords(args: argparse.Namespace) -> None:
    from sciscape.keyword_extraction import (
        KeywordExtractionConfig,
        KeywordExtractionPipeline,
        run_cluster_sharded_preflight,
    )
    from sciscape.keyword_extraction.config import VocabMergeConfig
    from sciscape.keyword_extraction.depth import DepthConfig
    from sciscape.keyword_extraction.term_network import TermNetworkConfig

    keyword_rule_result_root = _infer_keyword_rule_result_root(args.output, args.keyword_rule_result_root)
    if (
        args.keyword_rule_artifact
        and keyword_rule_result_root is None
        and args.keyword_engine != "cluster_sharded"
    ):
        print(
            "Keyword rule artifact skipped: provide --keyword-rule-result-root or write to "
            "<result_root>/landscape/keywords.parquet.",
            file=sys.stderr,
        )

    kwargs = dict(
        abstract_path=args.abstract_path,
        membership_path=args.membership_path,
        cluster_level=args.cluster_level,
        top_n_keywords=args.top_n,
        keyword_engine=args.keyword_engine,
        cluster_sharded_output_dir=args.cluster_sharded_output_dir,
        uid_col=args.uid_col,
        title_col=args.title_col,
        abstract_col=args.abstract_col,
        year_col=args.year_col,
        target_docs_per_shard=args.target_docs_per_shard,
        max_clusters_per_shard=args.max_clusters_per_shard,
        cluster_sharded_shard_ids=args.cluster_sharded_shard_ids,
        candidate_pool_floor=args.candidate_pool_floor,
        candidate_pool_large=args.candidate_pool_large,
        candidate_pool_hard_max=args.candidate_pool_hard_max,
        global_candidate_row_warning=args.global_candidate_row_warning,
        global_candidate_row_hard_stop=args.global_candidate_row_hard_stop,
        global_unique_term_warning=args.global_unique_term_warning,
        global_unique_term_hard_stop=args.global_unique_term_hard_stop,
        candidate_mining_progress_interval_docs=args.candidate_mining_progress_interval_docs,
        candidate_mining_prune_interval_docs=args.candidate_mining_prune_interval_docs,
        candidate_mining_prune_multiplier=args.candidate_mining_prune_multiplier,
        include_title=args.include_title,
        min_df_unigram=args.min_df,
        min_df_phrase=args.min_df,
        ngram_min=2,
        ngram_max=args.ngram_max,
        n_jobs=args.n_jobs,
        parallel_backend=args.parallel_backend,
        parallel_large_cluster_threshold=args.parallel_large_cluster_threshold,
        progress_path=args.progress_path,
        progress_interval_clusters=args.progress_interval_clusters,
        scoring_shard_dir=args.scoring_shard_dir,
        scoring_shard_size_clusters=args.scoring_shard_size_clusters,
        scoring_shard_resume=args.scoring_shard_resume,
        quality_rerank_enabled=args.quality_rerank,
        keyword_rule_artifact_enabled=args.keyword_rule_artifact,
        keyword_rule_set_id=args.keyword_rule_set_id,
        keyword_rule_result_root=keyword_rule_result_root,
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
            quality_diagnostics_enabled=True,
            quality_rerank_enabled=True,
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

    if args.keyword_preflight_only:
        if cfg.keyword_engine != "cluster_sharded":
            raise SystemExit("--keyword-preflight-only requires --keyword-engine cluster_sharded")
        print(f"Running cluster-sharded keyword preflight: {args.membership_path}")
        summary = run_cluster_sharded_preflight(cfg)
        print(f"  status={summary['status']}")
        print(
            "  clusters={clusters}, docs={docs}, shards={shards}".format(
                clusters=summary["total_clusters"],
                docs=summary["total_docs"],
                shards=summary["shard_count"],
            )
        )
        print(
            "  candidate_upper_bound={rows} (target={target}, warning={warning}, hard_stop={hard_stop})".format(
                rows=summary["expected_candidate_rows_upper_bound"],
                target=summary["candidate_row_budget"].get("target"),
                warning=summary["candidate_row_budget"].get("warning"),
                hard_stop=summary["candidate_row_budget"].get("hard_stop"),
            )
        )
        print(f"Preflight summary saved: {summary['preflight_summary_path']}")
        return

    print(f"Running keyword extraction: {args.abstract_path}")
    print(f"  cluster_level={args.cluster_level}, top_n={args.top_n}")
    pipeline = KeywordExtractionPipeline(cfg)
    keywords = pipeline.run()

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
    if pipeline.abbreviation_evidence is not None and not pipeline.abbreviation_evidence.empty:
        abbr_path = args.output.with_name(args.output.stem + "_abbreviations.parquet")
        abbr_df = pipeline.abbreviation_evidence.copy()
        if "cluster_supports" in abbr_df.columns:
            abbr_df["cluster_supports"] = abbr_df["cluster_supports"].apply(
                lambda value: json.dumps(value) if isinstance(value, dict) else value
            )
        abbr_df.to_parquet(abbr_path, index=False)
        print(f"Abbreviation evidence saved: {abbr_path} ({len(abbr_df)} pairs)")

    # Summary
    label_col = "display_label" if "display_label" in keywords.columns else "term"
    score_col = "quality_score" if "quality_score" in keywords.columns else "score"
    for cid, grp in keywords.groupby("cluster_id"):
        top3 = grp.nlargest(3, score_col)[label_col].astype(str).tolist()
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
        selection={
            "scope": "hosted_or_uploaded_data",
            "view": {"mode": "static_viewer", "surface": "cli_viewer"},
            "filters": [],
            "thresholds": {},
            "layer_state": {
                "command": "viewer",
                "open_browser": bool(args.open),
                "output": str(args.output),
            },
            "focus": {},
        },
    )
    print(f"Viewer generated: {path}")
    print("Deploy to Vercel:  vercel deploy --prod .")
    print("Or open locally:   open viewer.html")

    if args.open:
        import webbrowser
        webbrowser.open(f"file://{path}")


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(
        f"Unsupported table format: {path.suffix!r}. "
        "Use .parquet, .csv, .tsv, .json, or .jsonl."
    )


def _write_table(df: pd.DataFrame, path: Path, *, output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "parquet":
        df.to_parquet(path, index=False)
    elif output_format == "csv":
        df.to_csv(path, index=False)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported output format: {output_format}")


def _source_ref_path(path: Path, root: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(Path(root).expanduser().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _read_keyword_table(path: Path) -> pd.DataFrame:
    try:
        return _read_table(path)
    except ValueError as exc:
        raise ValueError(str(exc).replace("table", "keyword table")) from exc


def _prepare_keyword_table_for_visualization(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "cluster_id" not in df.columns and "cluster" in df.columns:
        df["cluster_id"] = df["cluster"]
    if "term" not in df.columns and "display_label" in df.columns:
        df["term"] = df["display_label"]

    required = {"cluster_id", "term"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Keyword table is missing required column(s): "
            + ", ".join(missing)
            + ". Required minimal schema: cluster_id, term."
        )

    if "score" not in df.columns:
        df["score"] = 1.0
    if "frequency" not in df.columns:
        df["frequency"] = 1
    if "doc_coverage" not in df.columns:
        df["doc_coverage"] = df["frequency"]

    return df


def _run_visualize(args: argparse.Namespace) -> None:
    from sciscape.keyword_extraction.visualization import export_dashboard, export_report

    try:
        keywords = _prepare_keyword_table_for_visualization(
            _read_keyword_table(args.keyword_table)
        )
    except Exception as exc:
        print(f"Could not load keyword table: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.dashboard_only:
        output = args.output
        if output.suffix.lower() != ".html":
            output = output / "index.html"
        path = export_dashboard(
            keywords,
            output_path=str(output),
            title=args.title,
            open_browser=args.open,
            selection={
                "scope": "keyword_table",
                "view": {"mode": "keyword_dashboard", "surface": "cli_visualize"},
                "filters": [],
                "thresholds": {},
                "layer_state": {
                    "command": "visualize",
                    "dashboard_only": True,
                    "open_browser": bool(args.open),
                    "keyword_table": str(args.keyword_table),
                },
                "focus": {},
            },
        )
        print(f"Dashboard generated: {path}")
        return

    paths = export_report(
        keywords,
        output_dir=str(args.output),
        title=args.title,
        open_browser=args.open,
        selection={
            "scope": "keyword_table",
            "view": {"mode": "html_report", "surface": "cli_visualize"},
            "filters": [],
            "thresholds": {},
            "layer_state": {
                "command": "visualize",
                "dashboard_only": False,
                "open_browser": bool(args.open),
                "keyword_table": str(args.keyword_table),
            },
            "focus": {},
        },
    )
    print(f"Report generated: {Path(args.output) / 'report.html'}")
    print(f"Files written: {len(paths)}")


def _read_json_object_arg(value: str | None, *, label: str) -> dict | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.exists():
        text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be a JSON object or a path to one") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _run_evolution_evidence(args: argparse.Namespace) -> None:
    from sciscape.evolution import build_slice_membership_evidence

    try:
        records = _read_table(args.records_table)
        membership = _read_table(args.membership_table)
        keywords = _read_table(args.keywords_table) if args.keywords_table is not None else None
        periodization = _read_json_object_arg(args.periodization, label="--periodization")
        evidence = build_slice_membership_evidence(
            evolution_id=args.evolution_id,
            records_df=records,
            membership_df=membership,
            keywords_df=keywords,
            periodization=periodization,
            cluster_column=args.cluster_column,
            uid_column=args.uid_column,
            membership_uid_column=args.membership_uid_column,
            representative_work_limit=args.representative_work_limit,
        )
    except Exception as exc:
        print(f"Could not build evolution evidence: {exc}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    suffix = "parquet" if args.output_format == "parquet" else "csv"
    slices_path = output_dir / f"time_slices.{suffix}"
    states_path = output_dir / f"state_evidence.{suffix}"
    membership_path = output_dir / f"state_membership.{suffix}"
    manifest_path = output_dir / "evolution_evidence_manifest.json"
    try:
        _write_table(evidence.slices, slices_path, output_format=args.output_format)
        _write_table(evidence.state_evidence, states_path, output_format=args.output_format)
        _write_table(evidence.state_membership, membership_path, output_format=args.output_format)
        manifest = {
            "schema_version": "sciscape_evolution_evidence_pack_v1",
            "evolution_id": evidence.evolution_id,
            "outputs": {
                "time_slices": slices_path.name,
                "state_evidence": states_path.name,
                "state_membership": membership_path.name,
            },
            "counts": {
                "slices": int(len(evidence.slices)),
                "states": int(len(evidence.state_evidence)),
                "state_membership_rows": int(len(evidence.state_membership)),
            },
            "periodization": evidence.periodization,
            "entity_scope": evidence.entity_scope,
            "transforms": evidence.transforms,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:
        print(f"Could not write evolution evidence: {exc}", file=sys.stderr)
        sys.exit(1)

    payload = {
        "evolution_id": evidence.evolution_id,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "time_slices_path": str(slices_path),
        "state_evidence_path": str(states_path),
        "state_membership_path": str(membership_path),
        "counts": manifest["counts"],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"Evolution evidence saved: {output_dir}")
    print(f"  slices={manifest['counts']['slices']} → {slices_path}")
    print(f"  states={manifest['counts']['states']} → {states_path}")
    print(f"  state_membership_rows={manifest['counts']['state_membership_rows']} → {membership_path}")
    print(f"  manifest → {manifest_path}")


def _run_evolution_from_membership(args: argparse.Namespace) -> None:
    from sciscape.artifacts import write_slice_membership_evolution_artifacts

    try:
        records = _read_table(args.records_table)
        membership = _read_table(args.membership_table)
        keywords = _read_table(args.keywords_table) if args.keywords_table is not None else None
        matching_method = _read_json_object_arg(args.matching_method, label="--matching-method") or {}
        matching_method.update(
            {
                "metric": args.metric,
                "min_transition_score": args.min_transition_score,
                "min_support_count": args.min_support_count,
            }
        )
        event_rules = _read_json_object_arg(args.event_rules, label="--event-rules")
        periodization = _read_json_object_arg(args.periodization, label="--periodization")
        if periodization is None:
            periodization = {"window_years": 2, "step_years": 1}
        entity_scope = _read_json_object_arg(args.entity_scope, label="--entity-scope")
    except Exception as exc:
        print(f"Could not load evolution membership inputs: {exc}", file=sys.stderr)
        sys.exit(1)

    source_artifacts = [
        {"role": "records", "path": _source_ref_path(args.records_table, args.result_root)},
        {"role": "membership", "path": _source_ref_path(args.membership_table, args.result_root)},
    ]
    if args.keywords_table is not None:
        source_artifacts.append({"role": "keywords", "path": _source_ref_path(args.keywords_table, args.result_root)})
    if args.temporal_manifest is not None:
        source_artifacts.append({"role": "temporal", "path": _source_ref_path(args.temporal_manifest, args.result_root)})

    try:
        written = write_slice_membership_evolution_artifacts(
            args.result_root,
            evolution_id=args.evolution_id,
            records_df=records,
            membership_df=membership,
            keywords_df=keywords,
            metric=args.metric,
            temporal_manifest=args.temporal_manifest,
            periodization=periodization,
            matching_method=matching_method,
            event_rules=event_rules,
            entity_scope=entity_scope,
            source_artifacts=source_artifacts,
            output_dir=args.output_dir,
            title=args.title,
            cluster_column=args.cluster_column,
            uid_column=args.uid_column,
            membership_uid_column=args.membership_uid_column,
            representative_work_limit=args.representative_work_limit,
            require_complete_membership=not args.allow_incomplete_state_membership,
        )
    except Exception as exc:
        print(f"Could not write evolution artifact from membership: {exc}", file=sys.stderr)
        sys.exit(1)

    qa = written["qa"]
    counts = qa.get("counts", {})
    payload = {
        "evolution_id": written["evolution_id"],
        "manifest_path": str(written["manifest_path"]),
        "evolution_dir": str(written["evolution_dir"]),
        "qa_path": str(written["qa_path"]),
        "status": qa.get("status"),
        "counts": counts,
        "event_counts": qa.get("event_counts", {}),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return

    print(f"Evolution artifact saved: {written['manifest_path']}")
    print(
        "  status={status}, slices={slices}, states={states}, transitions={transitions}, events={events}, state_membership_rows={membership}".format(
            status=qa.get("status"),
            slices=counts.get("slices", 0),
            states=counts.get("states", 0),
            transitions=counts.get("transitions", 0),
            events=counts.get("event_rows", 0),
            membership=counts.get("state_membership_rows", 0),
        )
    )
    print(f"  QA → {written['qa_path']}")


def _run_evolution_from_slice_membership(args: argparse.Namespace) -> None:
    from sciscape.artifacts import write_slice_local_membership_evolution_artifacts

    try:
        slice_membership = _read_table(args.slice_membership_table)
        slices = _read_table(args.slices_table) if args.slices_table is not None else None
        keywords = _read_table(args.keywords_table) if args.keywords_table is not None else None
        matching_method = _read_json_object_arg(args.matching_method, label="--matching-method") or {}
        matching_method.update(
            {
                "metric": args.metric,
                "min_transition_score": args.min_transition_score,
                "min_support_count": args.min_support_count,
            }
        )
        event_rules = _read_json_object_arg(args.event_rules, label="--event-rules")
        entity_scope = _read_json_object_arg(args.entity_scope, label="--entity-scope")
    except Exception as exc:
        print(f"Could not load slice-local evolution inputs: {exc}", file=sys.stderr)
        sys.exit(1)

    source_artifacts = [
        {"role": "slice_membership", "path": _source_ref_path(args.slice_membership_table, args.result_root)},
    ]
    if args.slices_table is not None:
        source_artifacts.append({"role": "time_slices", "path": _source_ref_path(args.slices_table, args.result_root)})
    if args.keywords_table is not None:
        source_artifacts.append({"role": "keywords", "path": _source_ref_path(args.keywords_table, args.result_root)})
    if args.temporal_manifest is not None:
        source_artifacts.append({"role": "temporal", "path": _source_ref_path(args.temporal_manifest, args.result_root)})

    try:
        written = write_slice_local_membership_evolution_artifacts(
            args.result_root,
            evolution_id=args.evolution_id,
            slice_membership_df=slice_membership,
            slices_df=slices,
            keywords_df=keywords,
            metric=args.metric,
            temporal_manifest=args.temporal_manifest,
            matching_method=matching_method,
            event_rules=event_rules,
            entity_scope=entity_scope,
            source_artifacts=source_artifacts,
            output_dir=args.output_dir,
            title=args.title,
            cluster_column=args.cluster_column,
            uid_column=args.uid_column,
            slice_id_column=args.slice_id_column,
            representative_work_limit=args.representative_work_limit,
            default_level=args.default_level,
            require_complete_membership=not args.allow_incomplete_state_membership,
        )
    except Exception as exc:
        print(f"Could not write evolution artifact from slice-local membership: {exc}", file=sys.stderr)
        sys.exit(1)

    qa = written["qa"]
    counts = qa.get("counts", {})
    payload = {
        "evolution_id": written["evolution_id"],
        "manifest_path": str(written["manifest_path"]),
        "evolution_dir": str(written["evolution_dir"]),
        "qa_path": str(written["qa_path"]),
        "status": qa.get("status"),
        "counts": counts,
        "event_counts": qa.get("event_counts", {}),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return

    print(f"Evolution artifact saved: {written['manifest_path']}")
    print(
        "  status={status}, slices={slices}, states={states}, transitions={transitions}, events={events}, state_membership_rows={membership}".format(
            status=qa.get("status"),
            slices=counts.get("slices", 0),
            states=counts.get("states", 0),
            transitions=counts.get("transitions", 0),
            events=counts.get("event_rows", 0),
            membership=counts.get("state_membership_rows", 0),
        )
    )
    print(f"  QA → {written['qa_path']}")


def _run_evolution_from_slice_reclustering(args: argparse.Namespace) -> None:
    from sciscape.artifacts import write_slice_reclustering_evolution_artifacts

    try:
        records = _read_table(args.records_table)
        edges = _read_table(args.edge_table)
        keywords = _read_table(args.keywords_table) if args.keywords_table is not None else None
        matching_method = _read_json_object_arg(args.matching_method, label="--matching-method") or {}
        matching_method.update(
            {
                "metric": args.metric,
                "min_transition_score": args.min_transition_score,
                "min_support_count": args.min_support_count,
            }
        )
        event_rules = _read_json_object_arg(args.event_rules, label="--event-rules")
        periodization = _read_json_object_arg(args.periodization, label="--periodization")
        if periodization is None:
            periodization = {"window_years": 2, "step_years": 1}
        entity_scope = _read_json_object_arg(args.entity_scope, label="--entity-scope")
    except Exception as exc:
        print(f"Could not load slice-reclustering evolution inputs: {exc}", file=sys.stderr)
        sys.exit(1)

    source_artifacts = [
        {"role": "records", "path": _source_ref_path(args.records_table, args.result_root)},
        {"role": "edges", "path": _source_ref_path(args.edge_table, args.result_root)},
    ]
    if args.keywords_table is not None:
        source_artifacts.append({"role": "keywords", "path": _source_ref_path(args.keywords_table, args.result_root)})
    if args.temporal_manifest is not None:
        source_artifacts.append({"role": "temporal", "path": _source_ref_path(args.temporal_manifest, args.result_root)})
    progress_path = args.progress_path or (args.result_root / "evolution_work" / "slice_reclustering_progress.json")

    try:
        written = write_slice_reclustering_evolution_artifacts(
            args.result_root,
            evolution_id=args.evolution_id,
            records_df=records,
            edges_df=edges,
            keywords_df=keywords,
            metric=args.metric,
            temporal_manifest=args.temporal_manifest,
            periodization=periodization,
            matching_method=matching_method,
            event_rules=event_rules,
            entity_scope=entity_scope,
            source_artifacts=source_artifacts,
            output_dir=args.output_dir,
            title=args.title,
            uid_column=args.uid_column,
            edge_source_column=args.edge_source_column,
            edge_target_column=args.edge_target_column,
            edge_weight_column=args.edge_weight_column,
            resolution=args.resolution,
            objective=args.objective,
            seed=args.seed,
            n_iterations=args.n_iterations,
            backend=args.backend,
            min_docs_per_slice=args.min_docs_per_slice,
            max_workers=args.slice_reclustering_workers,
            slice_membership_output=args.slice_membership_output,
            slice_membership_parts_dir=args.slice_membership_parts_dir,
            progress_path=progress_path,
            representative_work_limit=args.representative_work_limit,
            require_complete_membership=not args.allow_incomplete_state_membership,
        )
    except Exception as exc:
        print(f"Could not write evolution artifact from slice-local reclustering: {exc}", file=sys.stderr)
        sys.exit(1)

    qa = written["qa"]
    counts = qa.get("counts", {})
    payload = {
        "evolution_id": written["evolution_id"],
        "manifest_path": str(written["manifest_path"]),
        "evolution_dir": str(written["evolution_dir"]),
        "qa_path": str(written["qa_path"]),
        "status": qa.get("status"),
        "counts": counts,
        "event_counts": qa.get("event_counts", {}),
    }
    if written.get("slice_membership_path") is not None:
        payload["slice_membership_path"] = str(written["slice_membership_path"])
    if written.get("slice_membership_parts_dir") is not None:
        payload["slice_membership_parts_dir"] = str(written["slice_membership_parts_dir"])
    if written.get("progress_path") is not None:
        payload["progress_path"] = str(written["progress_path"])
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return

    print(f"Evolution artifact saved: {written['manifest_path']}")
    print(
        "  status={status}, slices={slices}, states={states}, transitions={transitions}, events={events}, state_membership_rows={membership}".format(
            status=qa.get("status"),
            slices=counts.get("slices", 0),
            states=counts.get("states", 0),
            transitions=counts.get("transitions", 0),
            events=counts.get("event_rows", 0),
            membership=counts.get("state_membership_rows", 0),
        )
    )
    print(f"  QA → {written['qa_path']}")
    if written.get("slice_membership_path") is not None:
        print(f"  Slice membership → {written['slice_membership_path']}")
    if written.get("slice_membership_parts_dir") is not None:
        print(f"  Slice membership parts → {written['slice_membership_parts_dir']}")
    if written.get("progress_path") is not None:
        print(f"  Progress → {written['progress_path']}")


def _run_evolution(args: argparse.Namespace) -> None:
    from sciscape.artifacts import write_document_overlap_evolution_artifacts, write_evidence_backed_evolution_artifacts

    try:
        slices = _read_table(args.slices_table)
        state_evidence = _read_table(args.state_evidence_table)
        transition_evidence = None
        state_membership = None
        if args.derive_transitions == "explicit":
            if args.transition_evidence_table is None:
                raise ValueError("transition_evidence_table is required unless --derive-transitions document-overlap is used")
            transition_evidence = _read_table(args.transition_evidence_table)
        elif args.derive_transitions == "document-overlap":
            if args.state_membership_table is None:
                raise ValueError("--state-membership-table is required for --derive-transitions document-overlap")
            state_membership = _read_table(args.state_membership_table)
        else:  # pragma: no cover
            raise ValueError(f"unsupported --derive-transitions value: {args.derive_transitions}")
        matching_method = _read_json_object_arg(args.matching_method, label="--matching-method") or {}
        event_rules = _read_json_object_arg(args.event_rules, label="--event-rules")
        periodization = _read_json_object_arg(args.periodization, label="--periodization")
        entity_scope = _read_json_object_arg(args.entity_scope, label="--entity-scope")
    except Exception as exc:
        print(f"Could not load evolution evidence: {exc}", file=sys.stderr)
        sys.exit(1)

    matching_method.update(
        {
            "metric": args.metric,
            "min_transition_score": args.min_transition_score,
            "min_support_count": args.min_support_count,
        }
    )
    try:
        if args.derive_transitions == "document-overlap":
            metric = args.metric
            if metric == "transition_score":
                metric = "jaccard_doc_overlap"
                matching_method["metric"] = metric
            written = write_document_overlap_evolution_artifacts(
                args.result_root,
                evolution_id=args.evolution_id,
                slices_df=slices,
                state_evidence_df=state_evidence,
                state_membership_df=state_membership,
                metric=metric,
                temporal_manifest=args.temporal_manifest,
                uid_column=args.state_membership_uid_column,
                state_id_column=args.state_membership_state_id_column,
                periodization=periodization,
                matching_method=matching_method,
                event_rules=event_rules,
                entity_scope=entity_scope,
                output_dir=args.output_dir,
                title=args.title,
                default_level=args.default_level,
                require_complete_membership=not args.allow_incomplete_state_membership,
            )
        else:
            written = write_evidence_backed_evolution_artifacts(
                args.result_root,
                evolution_id=args.evolution_id,
                slices_df=slices,
                state_evidence_df=state_evidence,
                transition_evidence_df=transition_evidence,
                metric=args.metric,
                temporal_manifest=args.temporal_manifest,
                periodization=periodization,
                matching_method=matching_method,
                event_rules=event_rules,
                entity_scope=entity_scope,
                output_dir=args.output_dir,
                title=args.title,
                default_level=args.default_level,
                allow_skip_slices=args.allow_skip_slices,
            )
    except Exception as exc:
        print(f"Could not write evolution artifact: {exc}", file=sys.stderr)
        sys.exit(1)

    qa = written["qa"]
    counts = qa.get("counts", {})
    print(f"Evolution artifact saved: {written['manifest_path']}")
    print(
        "  status={status}, slices={slices}, states={states}, transitions={transitions}, events={events}".format(
            status=qa.get("status"),
            slices=counts.get("slices", 0),
            states=counts.get("states", 0),
            transitions=counts.get("transitions", 0),
            events=counts.get("events", counts.get("event_rows", 0)),
        )
    )
    print(f"  QA → {written['qa_path']}")


def _run_export(args: argparse.Namespace) -> None:
    import polars as pl
    from sciscape.export import export_gexf, export_graphml, export_vosviewer_network

    edges = pl.read_parquet(args.edge_path)
    membership = pl.read_parquet(args.membership_path)
    abstracts = pl.read_parquet(args.abstracts) if args.abstracts else None
    source_paths = {
        "edges": args.edge_path,
        "membership": args.membership_path,
        "abstracts": args.abstracts,
    }
    result_root = Path(args.output).expanduser().resolve().parent

    if args.format == "vosviewer":
        output_dir = args.output if not args.output.suffix else args.output.with_suffix("")
        paths = export_vosviewer_network(
            edges,
            membership,
            output_dir,
            abstracts=abstracts,
            write_manifest=True,
            result_root=Path(output_dir).expanduser().resolve(),
            source_paths=source_paths,
        )
        print(f"Exported → {paths['map_path']}")
        print(f"Network → {paths['network_path']}")
        if paths.get("manifest_path"):
            print(f"Manifest → {paths['manifest_path']}")
        return
    elif args.format == "graphml":
        path = export_graphml(
            edges,
            membership,
            args.output,
            abstracts=abstracts,
            write_manifest=True,
            result_root=result_root,
            source_paths=source_paths,
        )
    else:
        path = export_gexf(
            edges,
            membership,
            args.output,
            abstracts=abstracts,
            write_manifest=True,
            result_root=result_root,
            source_paths=source_paths,
        )
    print(f"Exported → {path}")


def _run_rule_export(args: argparse.Namespace) -> None:
    from sciscape.export import export_vosviewer_thesaurus

    if args.format != "vosviewer-thesaurus":
        raise SystemExit(f"Unsupported rule export format: {args.format}")

    paths = export_vosviewer_thesaurus(
        args.rule_manifest,
        args.output,
        thesaurus_filename=args.thesaurus_filename,
        rule_set_filename=args.rule_set_filename,
        write_manifest=args.write_manifest,
        result_root=args.result_root,
    )
    print(f"Thesaurus → {paths['thesaurus_path']}")
    print(f"Rule set → {paths['rule_set_path']}")
    if paths.get("manifest_path"):
        print(f"Manifest → {paths['manifest_path']}")


def _run_matrix(args: argparse.Namespace) -> None:
    if args.matrix_command == "export":
        from sciscape.export import export_matrix_artifact

        try:
            written = export_matrix_artifact(
                args.matrix,
                output_dir=args.output_dir,
                matrix_id=args.matrix_id,
                export_format=args.export_format,
            )
        except Exception as exc:
            print(f"Could not export matrix artifact: {exc}", file=sys.stderr)
            sys.exit(1)
        if args.json:
            print(json.dumps(written, indent=2, sort_keys=True, default=str))
            return
        print(f"Matrix export saved: {written['primary_path']}")
        print(f"  Manifest → {written['manifest_path']}")
        print(f"  QA → {written['qa_path']}")
        return

    if args.matrix_command != "wrap-term-cooccurrence":
        raise SystemExit(f"Unsupported matrix command: {args.matrix_command}")

    from sciscape.artifacts import write_matrix_from_term_cooccurrence

    try:
        written = write_matrix_from_term_cooccurrence(
            args.result_root,
            matrix_id=args.matrix_id,
        )
    except Exception as exc:
        print(f"Could not write matrix artifact: {exc}", file=sys.stderr)
        sys.exit(1)
    if written is None:
        print(
            "Could not write matrix artifact: no valid term co-occurrence source was found",
            file=sys.stderr,
        )
        sys.exit(1)

    payload = {
        "schema_version": written["schema_version"],
        "matrix_id": written["matrix_id"],
        "matrix_dir": written["matrix_dir"],
        "manifest_path": written["manifest_path"],
        "values_path": written["values_path"],
        "row_entities_path": written["row_entities_path"],
        "column_entities_path": written["column_entities_path"],
        "qa_path": written["qa_path"],
        "qa": written["qa"],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return

    qa = written["qa"]
    counts = qa.get("counts", {})
    print(f"Matrix artifact saved: {written['matrix_dir']}")
    print(f"  manifest={written['manifest_path']}")
    print(
        "  status={status}, nnz={nnz}, rows={rows}, columns={columns}".format(
            status=qa.get("status"),
            nnz=counts.get("nnz", 0),
            rows=counts.get("rows", 0),
            columns=counts.get("columns", 0),
        )
    )
    print(f"  QA → {written['qa_path']}")


def _ensure_vosviewer_term_exports_for_bundle(result_root: Path) -> None:
    from sciscape.artifacts import write_cooccurrence_artifacts, write_matrix_from_term_cooccurrence
    from sciscape.export import export_matrix_artifact, export_vosviewer_term_cooccurrence

    written = write_cooccurrence_artifacts(result_root)
    if written is None:
        return
    export_vosviewer_term_cooccurrence(result_root)
    matrix_written = write_matrix_from_term_cooccurrence(result_root)
    if matrix_written is not None:
        export_matrix_artifact(result_root, export_format="vosviewer-network")


def _run_bundle(args: argparse.Namespace) -> None:
    if args.bundle_command != "vosviewer":
        raise SystemExit(f"Unsupported bundle command: {args.bundle_command}")

    from sciscape.export import export_vosviewer_bundle

    root = Path(args.result_root)
    try:
        if args.ensure_term_exports:
            _ensure_vosviewer_term_exports_for_bundle(root)
        written = export_vosviewer_bundle(root)
    except Exception as exc:
        print(f"Could not write VOSviewer bundle: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(written, indent=2, sort_keys=True, default=str))
        return

    print(f"VOSviewer bundle saved: {written['bundle_path']}")
    print(f"  Inventory → {written['inventory_path']}")
    if written.get("manifest_path"):
        print(f"  Manifest → {written['manifest_path']}")


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
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        backoff_base=args.backoff_base,
        backoff_max=args.backoff_max,
        api_attempt_budget=args.api_attempt_budget,
        retry_wait_budget_seconds=args.retry_wait_budget_seconds,
        interruptible_requests=args.interruptible_requests,
        request_poll_interval=args.request_poll_interval,
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
    elif args.command == "visualize":
        _run_visualize(args)
    elif args.command == "evolution-evidence":
        _run_evolution_evidence(args)
    elif args.command == "evolution-from-membership":
        _run_evolution_from_membership(args)
    elif args.command == "evolution-from-slice-membership":
        _run_evolution_from_slice_membership(args)
    elif args.command == "evolution-from-slice-reclustering":
        _run_evolution_from_slice_reclustering(args)
    elif args.command == "evolution":
        _run_evolution(args)
    elif args.command == "export":
        _run_export(args)
    elif args.command == "rule-export":
        _run_rule_export(args)
    elif args.command == "matrix":
        _run_matrix(args)
    elif args.command == "bundle":
        _run_bundle(args)
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
