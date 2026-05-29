"""Profiling and HTML report generation for the Rust Leiden backend.

This module is intentionally dependency-light: it writes JSON/CSV artifacts and
an inline HTML report without requiring plotly or matplotlib.  It is meant for
large-server smoke runs where the important question is where time and memory
are going: remap, CSR build, Leiden, or postprocess.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ProfileCase:
    """One edge dataset and clustering configuration to profile."""

    name: str
    edge_path: Path
    uid1_col: str = "uid1"
    uid2_col: str = "uid2"
    weight_col: str = "rel_sum2"
    resolution: float = 0.01
    min_size: int = 50
    n_iterations: int = 5
    seed: int = 42


@dataclass
class PhaseMetric:
    """Wall-time and memory after a profiling phase."""

    phase: str
    elapsed_sec: float
    rss_mb: float | None = None
    hwm_mb: float | None = None


@dataclass
class ClusterSummary:
    """Compact cluster-size summary."""

    n_clusters: int
    n_small: int
    n_singletons: int
    min_size: int
    median_size: float
    p90_size: float
    p99_size: float
    max_size: int
    size_bins: dict[str, int] = field(default_factory=dict)


@dataclass
class ProfileResult:
    """Result for one profiled case."""

    case: ProfileCase
    n_nodes: int
    directed_edges: int
    undirected_edges: int
    phases: list[PhaseMetric]
    raw_clusters: ClusterSummary
    post_clusters: ClusterSummary
    quality_raw: float
    quality_post: float
    changed_nodes: int
    postprocess_rounds: list[dict] = field(default_factory=list)

    @property
    def total_sec(self) -> float:
        return sum(phase.elapsed_sec for phase in self.phases)

    @property
    def quality_delta(self) -> float:
        return self.quality_post - self.quality_raw


DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

CLUSTER_SIZE_BINS: tuple[tuple[str, int, int | None], ...] = (
    ("1", 1, 1),
    ("2-4", 2, 4),
    ("5-9", 5, 9),
    ("10-49", 10, 49),
    ("50-99", 50, 99),
    ("100-499", 100, 499),
    ("500-999", 500, 999),
    ("1000+", 1000, None),
)

DEFAULT_CASES: dict[str, list[ProfileCase]] = {
    "quick": [
        ProfileCase(
            name="field34_combo_dc_bc_cc_sum",
            edge_path=DATA_ROOT / "linktype_edges/field_34/combo_dc+bc+cc_sum.parquet",
            uid1_col="src",
            uid2_col="dst",
            weight_col="weight",
            resolution=0.01,
            min_size=50,
            n_iterations=5,
        ),
        ProfileCase(
            name="field15_gcc_emb_full_knn30",
            edge_path=DATA_ROOT / "linktype_edges_gcc/field_15/emb_full_knn30.parquet",
            uid1_col="uid1",
            uid2_col="uid2",
            weight_col="rel_sum2",
            resolution=0.01,
            min_size=50,
            n_iterations=5,
        ),
    ],
    "full": [
        ProfileCase(
            name="field34_combo_dc_bc_cc_sum",
            edge_path=DATA_ROOT / "linktype_edges/field_34/combo_dc+bc+cc_sum.parquet",
            uid1_col="src",
            uid2_col="dst",
            weight_col="weight",
            resolution=0.01,
            min_size=50,
            n_iterations=5,
        ),
        ProfileCase(
            name="field15_gcc_emb_full_knn30",
            edge_path=DATA_ROOT / "linktype_edges_gcc/field_15/emb_full_knn30.parquet",
            uid1_col="uid1",
            uid2_col="uid2",
            weight_col="rel_sum2",
            resolution=0.01,
            min_size=50,
            n_iterations=5,
        ),
        ProfileCase(
            name="field12_gcc_emb_full_knn30",
            edge_path=DATA_ROOT / "linktype_edges_gcc/field_12/emb_full_knn30.parquet",
            uid1_col="uid1",
            uid2_col="uid2",
            weight_col="rel_sum2",
            resolution=0.01,
            min_size=50,
            n_iterations=3,
        ),
    ],
}


def _memory_status() -> tuple[float | None, float | None]:
    """Return current RSS and high-water RSS in MB on Linux."""

    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None, None

    rss_kb = None
    hwm_kb = None
    for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("VmRSS:"):
            rss_kb = float(line.split()[1])
        elif line.startswith("VmHWM:"):
            hwm_kb = float(line.split()[1])
    rss_mb = None if rss_kb is None else rss_kb / 1024.0
    hwm_mb = None if hwm_kb is None else hwm_kb / 1024.0
    return rss_mb, hwm_mb


def _phase(name: str, start: float) -> PhaseMetric:
    rss_mb, hwm_mb = _memory_status()
    return PhaseMetric(
        phase=name,
        elapsed_sec=time.perf_counter() - start,
        rss_mb=rss_mb,
        hwm_mb=hwm_mb,
    )


def summarize_membership(membership: np.ndarray, n_clusters: int, min_size: int) -> ClusterSummary:
    """Compute cluster-size statistics from a membership vector."""

    counts = np.bincount(np.asarray(membership, dtype=np.int64), minlength=int(n_clusters))
    used = counts[counts > 0]
    if used.size == 0:
        return ClusterSummary(
            n_clusters=0,
            n_small=0,
            n_singletons=0,
            min_size=0,
            median_size=0.0,
            p90_size=0.0,
            p99_size=0.0,
            max_size=0,
            size_bins=_cluster_size_bins(used),
        )

    return ClusterSummary(
        n_clusters=int(used.size),
        n_small=int(((used > 0) & (used < min_size)).sum()),
        n_singletons=int((used == 1).sum()),
        min_size=int(used.min()),
        median_size=float(np.percentile(used, 50)),
        p90_size=float(np.percentile(used, 90)),
        p99_size=float(np.percentile(used, 99)),
        max_size=int(used.max()),
        size_bins=_cluster_size_bins(used),
    )


def _cluster_size_bins(sizes: np.ndarray) -> dict[str, int]:
    bins: dict[str, int] = {}
    for label, lo, hi in CLUSTER_SIZE_BINS:
        if hi is None:
            bins[label] = int((sizes >= lo).sum())
        else:
            bins[label] = int(((sizes >= lo) & (sizes <= hi)).sum())
    return bins


def _build_graph(case: ProfileCase, remap_dir: Path, overwrite_remap: bool) -> tuple[object, int, int, list[PhaseMetric]]:
    from sciscape.clustering.integer_remap import integer_remap
    from sciscape.clustering.leiden_rust import build_leiden_graph

    phases: list[PhaseMetric] = []
    t0 = time.perf_counter()
    remap = integer_remap(
        case.edge_path,
        remap_dir,
        uid1_col=case.uid1_col,
        uid2_col=case.uid2_col,
        weight_col=case.weight_col,
        overwrite=overwrite_remap,
        write_int_edges=False,
    )
    phases.append(_phase("remap", t0))

    t0 = time.perf_counter()
    graph = build_leiden_graph(remap.int_edges_path, n_nodes=remap.n_nodes)
    phases.append(_phase("graph_build", t0))
    return graph, int(remap.n_edges), int(remap.n_nodes), phases


def profile_case(
    case: ProfileCase,
    output_dir: Path,
    *,
    overwrite_remap: bool = False,
) -> ProfileResult:
    """Run remap/build, Leiden, and postprocess for one case."""

    output_dir = Path(output_dir)
    remap_dir = output_dir / "remap" / case.name
    remap_dir.mkdir(parents=True, exist_ok=True)

    graph, undirected_edges, n_nodes, phases = _build_graph(case, remap_dir, overwrite_remap)

    t0 = time.perf_counter()
    raw = graph.run_leiden(
        resolution=case.resolution,
        seed=case.seed,
        n_iterations=case.n_iterations,
        randomness=0.01,
    )
    phases.append(_phase("leiden", t0))

    raw_summary = summarize_membership(raw.membership, raw.n_clusters, case.min_size)
    quality_raw = float(raw.quality)

    t0 = time.perf_counter()
    post = graph.postprocess_small_clusters(
        resolution=case.resolution,
        min_size=case.min_size,
        membership=raw.membership,
        seed=case.seed,
        n_iterations=case.n_iterations,
        randomness=0.01,
        max_rounds=5,
        gamma_decay=0.5,
        use_greedy=True,
        use_component_merge=True,
    )
    phases.append(_phase("postprocess", t0))

    post_summary = summarize_membership(post.membership, post.n_clusters, case.min_size)
    quality_post = graph.cpm_quality(post.membership, resolution=case.resolution)
    changed_nodes = int(np.sum(np.asarray(post.changed_at_round) >= 0))

    return ProfileResult(
        case=case,
        n_nodes=n_nodes,
        directed_edges=int(graph.n_edges),
        undirected_edges=undirected_edges,
        phases=phases,
        raw_clusters=raw_summary,
        post_clusters=post_summary,
        quality_raw=quality_raw,
        quality_post=float(quality_post),
        changed_nodes=changed_nodes,
        postprocess_rounds=[dict(round_info) for round_info in post.rounds],
    )


def profile_cases(
    cases: Sequence[ProfileCase],
    output_dir: Path,
    *,
    overwrite_remap: bool = False,
) -> list[ProfileResult]:
    """Profile all cases in order."""

    results = []
    for case in cases:
        results.append(profile_case(case, output_dir, overwrite_remap=overwrite_remap))
    return results


def _result_to_flat_row(result: ProfileResult) -> dict[str, object]:
    phases = {phase.phase: phase for phase in result.phases}
    return {
        "case": result.case.name,
        "edge_path": str(result.case.edge_path),
        "n_nodes": result.n_nodes,
        "undirected_edges": result.undirected_edges,
        "directed_edges": result.directed_edges,
        "resolution": result.case.resolution,
        "min_size": result.case.min_size,
        "n_iterations": result.case.n_iterations,
        "remap_sec": phases.get("remap", PhaseMetric("remap", 0)).elapsed_sec,
        "graph_build_sec": phases.get("graph_build", PhaseMetric("graph_build", 0)).elapsed_sec,
        "leiden_sec": phases.get("leiden", PhaseMetric("leiden", 0)).elapsed_sec,
        "postprocess_sec": phases.get("postprocess", PhaseMetric("postprocess", 0)).elapsed_sec,
        "total_sec": result.total_sec,
        "raw_clusters": result.raw_clusters.n_clusters,
        "post_clusters": result.post_clusters.n_clusters,
        "raw_small": result.raw_clusters.n_small,
        "post_small": result.post_clusters.n_small,
        "raw_singletons": result.raw_clusters.n_singletons,
        "post_singletons": result.post_clusters.n_singletons,
        "raw_max_cluster": result.raw_clusters.max_size,
        "post_max_cluster": result.post_clusters.max_size,
        "quality_raw": result.quality_raw,
        "quality_post": result.quality_post,
        "quality_delta": result.quality_delta,
        "changed_nodes": result.changed_nodes,
        "rounds": len(result.postprocess_rounds),
        "peak_hwm_mb": max((p.hwm_mb or 0.0) for p in result.phases),
    }


def write_artifacts(results: Sequence[ProfileResult], output_dir: Path) -> dict[str, Path]:
    """Write JSON/CSV/HTML report artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_json = output_dir / "metrics.json"
    metrics_csv = output_dir / "metrics.csv"
    rounds_csv = output_dir / "postprocess_rounds.csv"
    report_html = output_dir / "report.html"

    serializable = []
    for result in results:
        row = asdict(result)
        row["case"]["edge_path"] = str(result.case.edge_path)
        serializable.append(row)
    metrics_json.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    rows = [_result_to_flat_row(result) for result in results]
    if rows:
        with metrics_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    round_rows = []
    for result in results:
        for round_info in result.postprocess_rounds:
            row = {"case": result.case.name}
            row.update(round_info)
            round_rows.append(row)
    if round_rows:
        fieldnames = sorted({key for row in round_rows for key in row})
        with rounds_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(round_rows)
    else:
        rounds_csv.write_text("case\n", encoding="utf-8")

    report_html.write_text(render_html_report(results), encoding="utf-8")
    return {
        "metrics_json": metrics_json,
        "metrics_csv": metrics_csv,
        "rounds_csv": rounds_csv,
        "report_html": report_html,
    }


def _fmt_sec(value: float) -> str:
    if value < 1.0:
        return f"{value * 1000:.0f} ms"
    if value < 60.0:
        return f"{value:.2f} s"
    return f"{value / 60.0:.1f} min"


def _fmt_int(value: int | float) -> str:
    return f"{int(value):,}"


def _fmt_float(value: float, digits: int = 3) -> str:
    return f"{value:,.{digits}f}"


def _phase_bar(result: ProfileResult, max_total: float) -> str:
    colors = {
        "remap": "#4f6bed",
        "graph_build": "#00a38a",
        "leiden": "#f59f00",
        "postprocess": "#d6336c",
    }
    labels = {
        "remap": "remap",
        "graph_build": "CSR",
        "leiden": "Leiden",
        "postprocess": "post",
    }
    width = max(result.total_sec / max_total * 100.0, 2.0) if max_total > 0 else 100.0
    inner = []
    for phase in result.phases:
        pct = phase.elapsed_sec / result.total_sec * 100.0 if result.total_sec > 0 else 0.0
        if pct <= 0:
            continue
        inner.append(
            f'<span class="seg" style="width:{pct:.3f}%;background:{colors.get(phase.phase, "#868e96")}" '
            f'title="{html.escape(labels.get(phase.phase, phase.phase))}: {_fmt_sec(phase.elapsed_sec)}"></span>'
        )
    return f'<div class="bar-shell" style="width:{width:.3f}%">{"".join(inner)}</div>'


def _small_line(result: ProfileResult) -> str:
    values = [result.raw_clusters.n_small]
    labels = ["raw"]
    for round_info in result.postprocess_rounds:
        values.append(int(round_info.get("n_small_after", values[-1])))
        labels.append(str(round_info.get("method", "round")))
    if len(values) == 1:
        values.append(result.post_clusters.n_small)
        labels.append("post")
    max_value = max(values) or 1
    points = []
    n = len(values)
    for idx, value in enumerate(values):
        x = 12 + idx * (176 / max(n - 1, 1))
        y = 52 - (value / max_value) * 40
        points.append((x, y, value, labels[idx]))
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in points)
    circles = "\n".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3"><title>{html.escape(label)}: {value}</title></circle>'
        for x, y, value, label in points
    )
    return (
        '<svg class="spark" viewBox="0 0 200 64" role="img">'
        '<line x1="12" y1="52" x2="188" y2="52" />'
        f'<polyline points="{polyline}" />{circles}</svg>'
    )


def _cluster_histogram(summary: ClusterSummary, title: str) -> str:
    bins = summary.size_bins or {label: 0 for label, _, _ in CLUSTER_SIZE_BINS}
    max_count = max(bins.values(), default=0) or 1
    rows = []
    for label, _, _ in CLUSTER_SIZE_BINS:
        count = int(bins.get(label, 0))
        width = (count / max_count) * 100.0 if count else 0.0
        rows.append(
            '<div class="hist-row">'
            f'<div class="hist-label">{html.escape(label)}</div>'
            '<div class="hist-track">'
            f'<div class="hist-bar" style="width:{width:.2f}%"></div>'
            "</div>"
            f'<div class="hist-count">{_fmt_int(count)}</div>'
            "</div>"
        )
    return (
        f'<div class="hist"><div class="hist-title">{html.escape(title)}</div>'
        f'{"".join(rows)}</div>'
    )


def _cluster_stats_table(raw: ClusterSummary, post: ClusterSummary) -> str:
    metrics = [
        ("Clusters", raw.n_clusters, post.n_clusters, _fmt_int),
        ("Small clusters", raw.n_small, post.n_small, _fmt_int),
        ("Singletons", raw.n_singletons, post.n_singletons, _fmt_int),
        ("Min size", raw.min_size, post.min_size, _fmt_int),
        ("Median size", raw.median_size, post.median_size, lambda value: _fmt_float(value, 1)),
        ("P90 size", raw.p90_size, post.p90_size, lambda value: _fmt_float(value, 1)),
        ("P99 size", raw.p99_size, post.p99_size, lambda value: _fmt_float(value, 1)),
        ("Max size", raw.max_size, post.max_size, _fmt_int),
    ]
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(label)}</td>"
        f"<td>{fmt(raw_value)}</td>"
        f"<td>{fmt(post_value)}</td>"
        "</tr>"
        for label, raw_value, post_value, fmt in metrics
    )
    return (
        "<table><thead><tr><th>Metric</th><th>Raw</th><th>Postprocess</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def render_html_report(results: Sequence[ProfileResult]) -> str:
    """Render an inline HTML report."""

    rows = [_result_to_flat_row(result) for result in results]
    max_total = max((result.total_sec for result in results), default=1.0)
    total_nodes = sum(result.n_nodes for result in results)
    total_edges = sum(result.directed_edges for result in results)
    total_time = sum(result.total_sec for result in results)
    median_post = statistics.median([r.post_clusters.n_clusters for r in results]) if results else 0

    summary_cards = f"""
      <div class="card"><div class="label">Cases</div><div class="value">{len(results)}</div></div>
      <div class="card"><div class="label">Nodes Profiled</div><div class="value">{_fmt_int(total_nodes)}</div></div>
      <div class="card"><div class="label">Directed Edges</div><div class="value">{_fmt_int(total_edges)}</div></div>
      <div class="card"><div class="label">Total Wall Time</div><div class="value">{_fmt_sec(total_time)}</div></div>
      <div class="card"><div class="label">Median Final Clusters</div><div class="value">{_fmt_int(median_post)}</div></div>
    """

    table_rows = []
    for result, row in zip(results, rows):
        post_share = result.phases[-1].elapsed_sec / result.total_sec * 100.0 if result.total_sec else 0.0
        table_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(result.case.name)}</strong><br><span>{html.escape(str(result.case.edge_path))}</span></td>"
            f"<td>{_fmt_int(result.n_nodes)}</td>"
            f"<td>{_fmt_int(result.directed_edges)}</td>"
            f"<td>{_phase_bar(result, max_total)}<div class='time'>{_fmt_sec(result.total_sec)}</div></td>"
            f"<td>{_fmt_sec(row['leiden_sec'])}</td>"
            f"<td>{_fmt_sec(row['postprocess_sec'])}<br><span>{post_share:.1f}% total</span></td>"
            f"<td>{_fmt_int(result.raw_clusters.n_clusters)} -> {_fmt_int(result.post_clusters.n_clusters)}</td>"
            f"<td>{_fmt_int(result.raw_clusters.n_small)} -> {_fmt_int(result.post_clusters.n_small)} {_small_line(result)}</td>"
            f"<td>{_fmt_float(result.quality_delta, 2)}</td>"
            f"<td>{_fmt_int(result.changed_nodes)}</td>"
            f"<td>{_fmt_float(row['peak_hwm_mb'], 1)} MB</td>"
            "</tr>"
        )

    detail_sections = []
    for result in results:
        phases = "".join(
            "<tr>"
            f"<td>{html.escape(phase.phase)}</td>"
            f"<td>{_fmt_sec(phase.elapsed_sec)}</td>"
            f"<td>{'' if phase.rss_mb is None else _fmt_float(phase.rss_mb, 1)}</td>"
            f"<td>{'' if phase.hwm_mb is None else _fmt_float(phase.hwm_mb, 1)}</td>"
            "</tr>"
            for phase in result.phases
        )
        rounds = "".join(
            "<tr>"
            f"<td>{round_info.get('round', '')}</td>"
            f"<td>{html.escape(str(round_info.get('method', '')))}</td>"
            f"<td>{round_info.get('gamma', 0):.4e}</td>"
            f"<td>{round_info.get('n_small_before', '')}</td>"
            f"<td>{round_info.get('n_small_after', '')}</td>"
            f"<td>{round_info.get('n_merged', '')}</td>"
            f"<td>{round_info.get('n_total_clusters', '')}</td>"
            "</tr>"
            for round_info in result.postprocess_rounds
        ) or "<tr><td colspan='7'>No postprocess rounds</td></tr>"
        histograms = (
            _cluster_histogram(result.raw_clusters, "Raw")
            + _cluster_histogram(result.post_clusters, "Postprocess")
        )
        detail_sections.append(
            f"""
            <section class="detail">
              <h2>{html.escape(result.case.name)}</h2>
              <div class="grid2">
                <div>
                  <h3>Phases</h3>
                  <table><thead><tr><th>Phase</th><th>Elapsed</th><th>RSS MB</th><th>HWM MB</th></tr></thead><tbody>{phases}</tbody></table>
                </div>
                <div>
                  <h3>Postprocess Rounds</h3>
                  <table><thead><tr><th>Round</th><th>Method</th><th>Gamma</th><th>Small Before</th><th>Small After</th><th>Merged</th><th>Total</th></tr></thead><tbody>{rounds}</tbody></table>
                </div>
              </div>
              <div class="grid2 lower">
                <div>
                  <h3>Cluster Size Distribution</h3>
                  <div class="hist-pair">{histograms}</div>
                </div>
                <div>
                  <h3>Cluster Size Statistics</h3>
                  {_cluster_stats_table(result.raw_clusters, result.post_clusters)}
                </div>
              </div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SciScape Leiden Profile Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --fg: #1f2933;
      --muted: #667085;
      --line: #d9dee7;
      --panel: #ffffff;
      --accent: #3451b2;
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
    h3 {{ margin: 0 0 10px; font-size: 15px; }}
    p {{ margin: 0; color: var(--muted); }}
    main {{ padding: 24px 36px 40px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin-bottom: 22px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .value {{ font-size: 24px; font-weight: 700; margin-top: 6px; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 18px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: right; vertical-align: middle; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; background: #fbfcfe; }}
    td span {{ color: var(--muted); font-size: 12px; }}
    .bar-shell {{
      height: 14px;
      min-width: 42px;
      background: #edf0f5;
      border-radius: 4px;
      overflow: hidden;
      display: inline-flex;
      vertical-align: middle;
    }}
    .seg {{ display: inline-block; height: 100%; }}
    .time {{ margin-top: 4px; color: var(--muted); font-size: 12px; }}
    .spark {{ width: 110px; height: 36px; margin-left: 6px; vertical-align: middle; }}
    .spark line {{ stroke: #d0d5dd; }}
    .spark polyline {{ fill: none; stroke: var(--accent); stroke-width: 2; }}
    .spark circle {{ fill: var(--accent); }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .lower {{ margin-top: 18px; }}
    .detail table {{ font-size: 12px; }}
    .hist-pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .hist-title {{ color: var(--muted); font-size: 12px; font-weight: 700; margin-bottom: 8px; text-transform: uppercase; }}
    .hist-row {{ display: grid; grid-template-columns: 54px 1fr 58px; gap: 8px; align-items: center; margin: 5px 0; }}
    .hist-label, .hist-count {{ color: var(--muted); font-size: 12px; }}
    .hist-count {{ text-align: right; }}
    .hist-track {{ height: 10px; background: #edf0f5; border-radius: 3px; overflow: hidden; }}
    .hist-bar {{ height: 100%; min-width: 0; background: var(--accent); }}
    @media (max-width: 960px) {{
      main, header {{ padding-left: 16px; padding-right: 16px; }}
      .grid2 {{ grid-template-columns: 1fr; }}
      .hist-pair {{ grid-template-columns: 1fr; }}
      table {{ font-size: 12px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>SciScape Leiden Profile Report</h1>
    <p>Rust remap, CSR build, Leiden, and postprocess timing across test datasets.</p>
  </header>
  <main>
    <div class="cards">{summary_cards}</div>
    <section>
      <h2>Run Summary</h2>
      <table>
        <thead>
          <tr>
            <th>Case</th><th>Nodes</th><th>Directed Edges</th><th>Total Time</th>
            <th>Leiden</th><th>Postprocess</th><th>Clusters</th><th>Small Clusters</th>
            <th>Delta Q</th><th>Changed Nodes</th><th>Peak HWM</th>
          </tr>
        </thead>
        <tbody>{"".join(table_rows)}</tbody>
      </table>
    </section>
    {"".join(detail_sections)}
  </main>
</body>
</html>
"""


def cases_for_preset(preset: str) -> list[ProfileCase]:
    try:
        return list(DEFAULT_CASES[preset])
    except KeyError as exc:
        raise ValueError(f"unknown preset {preset!r}; choose one of {sorted(DEFAULT_CASES)}") from exc


def _parse_case(value: str) -> ProfileCase:
    """Parse name:path:uid1:uid2:weight:resolution:min_size:n_iterations."""

    parts = value.split(":")
    if len(parts) not in {5, 8}:
        raise argparse.ArgumentTypeError(
            "--case must be name:path:uid1:uid2:weight or "
            "name:path:uid1:uid2:weight:resolution:min_size:n_iterations"
        )
    name, path, uid1, uid2, weight = parts[:5]
    if len(parts) == 5:
        return ProfileCase(name=name, edge_path=Path(path), uid1_col=uid1, uid2_col=uid2, weight_col=weight)
    return ProfileCase(
        name=name,
        edge_path=Path(path),
        uid1_col=uid1,
        uid2_col=uid2,
        weight_col=weight,
        resolution=float(parts[5]),
        min_size=int(parts[6]),
        n_iterations=int(parts[7]),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile Rust Leiden and write an HTML report.")
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
    parser.add_argument("--output-dir", type=Path, default=Path("workspace/output/leiden_profile_report"))
    parser.add_argument("--overwrite-remap", action="store_true")
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
    results = profile_cases(cases, args.output_dir, overwrite_remap=args.overwrite_remap)
    paths = write_artifacts(results, args.output_dir)
    print(f"Wrote profile report: {paths['report_html']}")
    print(f"Wrote metrics CSV: {paths['metrics_csv']}")
    return 0


__all__ = [
    "ClusterSummary",
    "DEFAULT_CASES",
    "PhaseMetric",
    "ProfileCase",
    "ProfileResult",
    "cases_for_preset",
    "profile_case",
    "profile_cases",
    "render_html_report",
    "summarize_membership",
    "write_artifacts",
]
