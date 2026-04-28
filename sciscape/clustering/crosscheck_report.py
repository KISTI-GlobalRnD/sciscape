"""Java/Rust Leiden cross-check report generation.

The profiler answers where the Rust backend spends time.  This module answers
the parity question: for the same remapped graph, seed, resolution, and
iteration count, how close are the Java and Rust clusterings in runtime,
quality, cluster counts, NMI, and ARI.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from sciscape.clustering.leiden_java import _resolve_jar, run_leiden_java
from sciscape.clustering.profile_report import (
    DEFAULT_CASES,
    ClusterSummary,
    ProfileCase,
    _fmt_float,
    _fmt_int,
    _fmt_sec,
    _parse_case,
    cases_for_preset,
    summarize_membership,
)


@dataclass
class CrosscheckRun:
    """One Java/Rust comparison run for a case and seed."""

    case: ProfileCase
    seed: int
    n_nodes: int
    directed_edges: int
    undirected_edges: int
    compare_nodes: int
    rust_sec: float
    java_sec: float
    rust_quality: float
    java_quality: float
    rust_clusters: ClusterSummary
    java_clusters: ClusterSummary
    nmi: float
    ari: float
    java_output_path: str | None = None

    @property
    def speedup(self) -> float:
        return self.java_sec / self.rust_sec if self.rust_sec > 0 else float("inf")

    @property
    def quality_delta(self) -> float:
        return self.rust_quality - self.java_quality


def parse_seeds(value: str) -> list[int]:
    """Parse comma-separated seed integers."""

    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer")
    return seeds


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "case"


def _sample_for_metrics(
    rust_membership: np.ndarray,
    java_membership: np.ndarray,
    *,
    sample_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_nodes = rust_membership.shape[0]
    if java_membership.shape[0] != n_nodes:
        raise ValueError(
            f"membership length mismatch: Rust={n_nodes}, Java={java_membership.shape[0]}"
        )
    if sample_size <= 0 or sample_size >= n_nodes:
        return rust_membership, java_membership, n_nodes

    rng = np.random.default_rng(seed)
    idx = rng.choice(n_nodes, size=sample_size, replace=False)
    return rust_membership[idx], java_membership[idx], int(sample_size)


def crosscheck_case(
    case: ProfileCase,
    output_dir: Path,
    *,
    seeds: Sequence[int],
    jar_path: Path | None = None,
    java_cmd: str = "java",
    java_heap: str = "8g",
    overwrite_remap: bool = False,
    randomness: float = 0.01,
    coalesce_undirected_edges: bool = True,
    compare_sample_size: int = 0,
    keep_memberships: bool = False,
) -> list[CrosscheckRun]:
    """Run Java/Rust comparisons for one profile case."""

    from sciscape.clustering.integer_remap import integer_remap
    from sciscape.clustering.leiden_rust import build_leiden_graph

    resolved_jar = _resolve_jar(jar_path)
    output_dir = Path(output_dir)
    remap_dir = output_dir / "remap" / case.name
    remap_dir.mkdir(parents=True, exist_ok=True)

    remap = integer_remap(
        case.edge_path,
        remap_dir,
        uid1_col=case.uid1_col,
        uid2_col=case.uid2_col,
        weight_col=case.weight_col,
        overwrite=overwrite_remap,
        write_int_edges=True,
    )
    graph = build_leiden_graph(remap.int_edges_path, n_nodes=remap.n_nodes)

    java_edges_dir = output_dir / "java_edges"
    java_edges_dir.mkdir(parents=True, exist_ok=True)
    java_tsv_path = java_edges_dir / f"{_safe_stem(case.name)}.tsv"

    membership_dir = output_dir / "memberships"
    if keep_memberships:
        membership_dir.mkdir(parents=True, exist_ok=True)

    runs: list[CrosscheckRun] = []
    for seed in seeds:
        t0 = time.perf_counter()
        rust = graph.run_leiden(
            resolution=case.resolution,
            seed=int(seed),
            n_iterations=case.n_iterations,
            randomness=randomness,
        )
        rust_sec = time.perf_counter() - t0

        java_output_path = None
        if keep_memberships:
            java_output_path = membership_dir / f"{_safe_stem(case.name)}.seed{seed}.java.tsv"

        t0 = time.perf_counter()
        java = run_leiden_java(
            remap.int_edges_path,
            resolution=case.resolution,
            n_nodes=remap.n_nodes,
            jar_path=resolved_jar,
            output_path=java_output_path,
            edge_tsv_path=java_tsv_path,
            seed=int(seed),
            iterations=case.n_iterations,
            randomness=randomness,
            weighted=True,
            coalesce_undirected_edges=coalesce_undirected_edges,
            java_cmd=java_cmd,
            java_heap=java_heap,
        )
        java_sec = time.perf_counter() - t0

        rust_membership = np.asarray(rust.membership)
        java_membership = np.asarray(java.membership)
        metric_rust, metric_java, compare_nodes = _sample_for_metrics(
            rust_membership,
            java_membership,
            sample_size=compare_sample_size,
            seed=int(seed),
        )

        java_membership_u64 = np.ascontiguousarray(java_membership, dtype=np.uint64)
        java_quality = graph.cpm_quality(java_membership_u64, resolution=case.resolution)

        runs.append(
            CrosscheckRun(
                case=case,
                seed=int(seed),
                n_nodes=int(remap.n_nodes),
                directed_edges=int(graph.n_edges),
                undirected_edges=int(remap.n_edges),
                compare_nodes=compare_nodes,
                rust_sec=float(rust_sec),
                java_sec=float(java_sec),
                rust_quality=float(rust.quality),
                java_quality=float(java_quality),
                rust_clusters=summarize_membership(rust_membership, rust.n_clusters, case.min_size),
                java_clusters=summarize_membership(java_membership, java.n_clusters, case.min_size),
                nmi=float(normalized_mutual_info_score(metric_java, metric_rust)),
                ari=float(adjusted_rand_score(metric_java, metric_rust)),
                java_output_path=None if java.output_path is None else str(java.output_path),
            )
        )
    return runs


def crosscheck_cases(
    cases: Sequence[ProfileCase],
    output_dir: Path,
    *,
    seeds: Sequence[int],
    jar_path: Path | None = None,
    java_cmd: str = "java",
    java_heap: str = "8g",
    overwrite_remap: bool = False,
    randomness: float = 0.01,
    coalesce_undirected_edges: bool = True,
    compare_sample_size: int = 0,
    keep_memberships: bool = False,
) -> list[CrosscheckRun]:
    """Run cross-checks for all cases in order."""

    runs: list[CrosscheckRun] = []
    for case in cases:
        runs.extend(
            crosscheck_case(
                case,
                output_dir,
                seeds=seeds,
                jar_path=jar_path,
                java_cmd=java_cmd,
                java_heap=java_heap,
                overwrite_remap=overwrite_remap,
                randomness=randomness,
                coalesce_undirected_edges=coalesce_undirected_edges,
                compare_sample_size=compare_sample_size,
                keep_memberships=keep_memberships,
            )
        )
    return runs


def _run_to_flat_row(run: CrosscheckRun) -> dict[str, object]:
    return {
        "case": run.case.name,
        "edge_path": str(run.case.edge_path),
        "seed": run.seed,
        "n_nodes": run.n_nodes,
        "undirected_edges": run.undirected_edges,
        "directed_edges": run.directed_edges,
        "compare_nodes": run.compare_nodes,
        "resolution": run.case.resolution,
        "n_iterations": run.case.n_iterations,
        "rust_sec": run.rust_sec,
        "java_sec": run.java_sec,
        "speedup_java_over_rust": run.speedup,
        "rust_quality": run.rust_quality,
        "java_quality": run.java_quality,
        "quality_delta_rust_minus_java": run.quality_delta,
        "rust_clusters": run.rust_clusters.n_clusters,
        "java_clusters": run.java_clusters.n_clusters,
        "cluster_delta_rust_minus_java": run.rust_clusters.n_clusters - run.java_clusters.n_clusters,
        "rust_small": run.rust_clusters.n_small,
        "java_small": run.java_clusters.n_small,
        "rust_singletons": run.rust_clusters.n_singletons,
        "java_singletons": run.java_clusters.n_singletons,
        "nmi": run.nmi,
        "ari": run.ari,
        "java_output_path": run.java_output_path or "",
    }


def write_artifacts(runs: Sequence[CrosscheckRun], output_dir: Path) -> dict[str, Path]:
    """Write Java/Rust cross-check JSON/CSV/HTML artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_json = output_dir / "crosscheck_metrics.json"
    metrics_csv = output_dir / "crosscheck_metrics.csv"
    report_html = output_dir / "crosscheck_report.html"

    serializable = []
    for run in runs:
        row = asdict(run)
        row["case"]["edge_path"] = str(run.case.edge_path)
        row["speedup_java_over_rust"] = run.speedup
        row["quality_delta_rust_minus_java"] = run.quality_delta
        serializable.append(row)
    metrics_json.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    rows = [_run_to_flat_row(run) for run in runs]
    if rows:
        with metrics_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        metrics_csv.write_text("case\n", encoding="utf-8")

    report_html.write_text(render_html_report(runs), encoding="utf-8")
    return {
        "metrics_json": metrics_json,
        "metrics_csv": metrics_csv,
        "report_html": report_html,
    }


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _speed_bar(run: CrosscheckRun, max_speedup: float) -> str:
    width = max(run.speedup / max_speedup * 100.0, 2.0) if max_speedup > 0 else 100.0
    return (
        f'<div class="speed-shell"><div class="speed-bar" style="width:{width:.3f}%"></div></div>'
        f'<div class="time">{_fmt_float(run.speedup, 2)}x</div>'
    )


def render_html_report(runs: Sequence[CrosscheckRun]) -> str:
    """Render a dependency-free Java/Rust cross-check HTML report."""

    max_speedup = max((run.speedup for run in runs), default=1.0)
    total_nodes = sum(run.n_nodes for run in runs)
    median_speedup = _median([run.speedup for run in runs])
    median_nmi = _median([run.nmi for run in runs])
    median_ari = _median([run.ari for run in runs])
    cases = sorted({run.case.name for run in runs})

    summary_cards = f"""
      <div class="card"><div class="label">Cases</div><div class="value">{len(cases)}</div></div>
      <div class="card"><div class="label">Runs</div><div class="value">{len(runs)}</div></div>
      <div class="card"><div class="label">Node-Runs</div><div class="value">{_fmt_int(total_nodes)}</div></div>
      <div class="card"><div class="label">Median Speedup</div><div class="value">{_fmt_float(median_speedup, 2)}x</div></div>
      <div class="card"><div class="label">Median NMI</div><div class="value">{_fmt_float(median_nmi, 3)}</div></div>
      <div class="card"><div class="label">Median ARI</div><div class="value">{_fmt_float(median_ari, 3)}</div></div>
    """

    table_rows = []
    for run in runs:
        table_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(run.case.name)}</strong><br><span>seed={run.seed}, gamma={run.case.resolution:g}</span></td>"
            f"<td>{_fmt_int(run.n_nodes)}</td>"
            f"<td>{_fmt_sec(run.rust_sec)}</td>"
            f"<td>{_fmt_sec(run.java_sec)}</td>"
            f"<td>{_speed_bar(run, max_speedup)}</td>"
            f"<td>{_fmt_float(run.quality_delta, 2)}</td>"
            f"<td>{_fmt_int(run.rust_clusters.n_clusters)} / {_fmt_int(run.java_clusters.n_clusters)}</td>"
            f"<td>{_fmt_int(run.rust_clusters.n_singletons)} / {_fmt_int(run.java_clusters.n_singletons)}</td>"
            f"<td>{_fmt_float(run.nmi, 3)}<br><span>{_fmt_int(run.compare_nodes)} nodes</span></td>"
            f"<td>{_fmt_float(run.ari, 3)}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SciScape Leiden Java/Rust Cross-check</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --fg: #1f2933;
      --muted: #667085;
      --line: #d9dee7;
      --panel: #ffffff;
      --accent: #0f766e;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--fg);
    }}
    header {{
      padding: 28px 36px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{ margin: 0 0 6px; font-size: 28px; }}
    h2 {{ margin: 0 0 14px; font-size: 21px; }}
    p {{ margin: 0; color: var(--muted); }}
    main {{ padding: 24px 36px 40px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-bottom: 22px;
    }}
    .card, section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .card {{ padding: 14px 16px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .value {{ font-size: 24px; font-weight: 700; margin-top: 6px; }}
    section {{ padding: 18px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: right; vertical-align: middle; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; background: #fbfcfe; }}
    td span {{ color: var(--muted); font-size: 12px; }}
    .speed-shell {{
      width: 96px;
      height: 12px;
      background: #edf0f5;
      border-radius: 4px;
      overflow: hidden;
      display: inline-block;
      vertical-align: middle;
    }}
    .speed-bar {{ height: 100%; background: var(--accent); }}
    .time {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
    @media (max-width: 960px) {{
      main, header {{ padding-left: 16px; padding-right: 16px; }}
      table {{ font-size: 12px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>SciScape Leiden Java/Rust Cross-check</h1>
    <p>Runtime, CPM quality, cluster count, singleton count, NMI, and ARI by seed.</p>
  </header>
  <main>
    <div class="cards">{summary_cards}</div>
    <section>
      <h2>Run Summary</h2>
      <table>
        <thead>
          <tr>
            <th>Case</th><th>Nodes</th><th>Rust</th><th>Java</th><th>Java/Rust</th>
            <th>Delta Q</th><th>Clusters R/J</th><th>Singletons R/J</th><th>NMI</th><th>ARI</th>
          </tr>
        </thead>
        <tbody>{"".join(table_rows)}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-check Rust Leiden against the Java backend.")
    parser.add_argument("--preset", choices=sorted(DEFAULT_CASES), default="quick")
    parser.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        help=(
            "Custom case: name:path:uid1:uid2:weight[:resolution:min_size:n_iterations]. "
            "May be passed multiple times. Overrides --preset."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/leiden_crosscheck_report"))
    parser.add_argument("--seeds", type=parse_seeds, default=[0, 1, 2])
    parser.add_argument("--jar-path", type=Path, default=None)
    parser.add_argument("--java-cmd", default="java")
    parser.add_argument("--java-heap", default="8g")
    parser.add_argument("--overwrite-remap", action="store_true")
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--compare-sample-size", type=int, default=0)
    parser.add_argument("--keep-memberships", action="store_true")
    parser.add_argument(
        "--no-coalesce-undirected-edges",
        action="store_true",
        help="Do not coalesce duplicate undirected pairs before invoking Java.",
    )
    parser.add_argument("--trace", choices=["0", "summary", "verbose"], default=None)
    parser.add_argument("--trace-rss", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.trace is not None:
        os.environ["SCISCAPE_LEIDEN_TRACE"] = args.trace
    if args.trace_rss:
        os.environ["SCISCAPE_LEIDEN_TRACE_RSS"] = "1"

    cases = args.case if args.case else cases_for_preset(args.preset)
    runs = crosscheck_cases(
        cases,
        args.output_dir,
        seeds=args.seeds,
        jar_path=args.jar_path,
        java_cmd=args.java_cmd,
        java_heap=args.java_heap,
        overwrite_remap=args.overwrite_remap,
        randomness=args.randomness,
        coalesce_undirected_edges=not args.no_coalesce_undirected_edges,
        compare_sample_size=args.compare_sample_size,
        keep_memberships=args.keep_memberships,
    )
    paths = write_artifacts(runs, args.output_dir)
    print(f"Wrote cross-check report: {paths['report_html']}")
    print(f"Wrote metrics CSV: {paths['metrics_csv']}")
    return 0


__all__ = [
    "CrosscheckRun",
    "build_arg_parser",
    "crosscheck_case",
    "crosscheck_cases",
    "main",
    "parse_seeds",
    "render_html_report",
    "write_artifacts",
]
