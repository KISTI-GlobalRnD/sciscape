#!/usr/bin/env python3
"""Summarize recomputed basin operator rows with exact/aligned change metrics."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


COMBINED_DIR = Path(
    "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30"
)
DEFAULT_OUTPUT_DIR = (
    COMBINED_DIR / "basin_evaluation_metric_audit_v0/recomputed_operator_metric_review"
)

SUMMARY_FILENAME = "recomputed_operator_metric_summary.csv"
TOP_ROWS_FILENAME = "recomputed_operator_metric_top_rows.csv"
REPORT_FILENAME = "recomputed_operator_metric_review.md"
PAYLOAD_FILENAME = "recomputed_operator_metric_review_summary.json"


@dataclass(frozen=True)
class ArtifactSpec:
    artifact: str
    rows_path: Path
    quality_gain_col: str
    verdict_col: str
    final_exact_col: str
    final_aligned_col: str
    endpoint_col: str
    pre_exact_col: str | None = None
    pre_aligned_col: str | None = None
    final_exact_only_col: str | None = None
    context_cols: tuple[str, ...] = ()


DEFAULT_ARTIFACTS = (
    ArtifactSpec(
        artifact="joint_bundle",
        rows_path=COMBINED_DIR
        / "basin_transition_attachment_margin_joint_bundle_field34_cc_c0_p6_p8_p10_v0"
        / "attachment_margin_joint_bundle_rows.csv",
        quality_gain_col="joint_delta_q_gain_vs_source",
        verdict_col="joint_verdict",
        pre_exact_col="joint_pre_polish_exact_changed_node_count",
        pre_aligned_col="joint_pre_polish_aligned_changed_node_count",
        final_exact_col="joint_final_exact_changed_node_count",
        final_aligned_col="joint_final_aligned_changed_node_count",
        final_exact_only_col="joint_final_exact_only_changed_node_count",
        endpoint_col="joint_final_endpoint_distance_to_source",
        context_cols=(
            "source_case",
            "target_k",
            "context_family",
            "context_multiplier",
            "move_kind",
            "state_delta_q_vs_vanilla",
            "joint_verdict",
        ),
    ),
    ArtifactSpec(
        artifact="stage2_recovery",
        rows_path=COMBINED_DIR
        / "basin_transition_attachment_margin_stage2_recovery_field34_cc_c0_p6_p8_p10_v2"
        / "attachment_margin_stage2_recovery_rows.csv",
        quality_gain_col="stage2_delta_q_gain_vs_stage1",
        verdict_col="stage2_verdict",
        pre_exact_col="stage2_pre_polish_exact_changed_node_count",
        pre_aligned_col="stage2_pre_polish_aligned_changed_node_count",
        final_exact_col="stage2_final_exact_changed_node_count",
        final_aligned_col="stage2_final_aligned_changed_node_count",
        final_exact_only_col="stage2_final_exact_only_changed_node_count",
        endpoint_col="stage2_final_endpoint_distance_to_stage1",
        context_cols=(
            "source_case",
            "context_family",
            "context_multiplier",
            "move_kind",
            "state_delta_q_vs_vanilla",
            "stage2_verdict",
        ),
    ),
    ArtifactSpec(
        artifact="gate_release_v0",
        rows_path=COMBINED_DIR
        / "basin_transition_gate_release_operator_probe_field34_cc_c0_p8_v0"
        / "gate_release_operator_probe_rows.csv",
        quality_gain_col="gate_release_delta_q_gain",
        verdict_col="gate_release_verdict",
        final_exact_col="changed_node_count",
        final_aligned_col="aligned_changed_node_count",
        endpoint_col="state_endpoint_distance_to_vanilla",
        context_cols=(
            "selector",
            "selected_k",
            "action_mode",
            "selected_node_ids",
            "state_delta_q_vs_vanilla",
            "gate_release_verdict",
        ),
    ),
    ArtifactSpec(
        artifact="gate_release_seed5",
        rows_path=COMBINED_DIR
        / "basin_transition_gate_release_operator_probe_field34_cc_c0_p8_seed5_v0"
        / "gate_release_operator_probe_rows.csv",
        quality_gain_col="gate_release_delta_q_gain",
        verdict_col="gate_release_verdict",
        final_exact_col="changed_node_count",
        final_aligned_col="aligned_changed_node_count",
        endpoint_col="state_endpoint_distance_to_vanilla",
        context_cols=(
            "selector",
            "selected_k",
            "action_mode",
            "selected_node_ids",
            "gate_release_seed",
            "state_delta_q_vs_vanilla",
            "gate_release_verdict",
        ),
    ),
    ArtifactSpec(
        artifact="gate_release_manual",
        rows_path=COMBINED_DIR
        / "basin_transition_gate_release_operator_probe_field34_cc_c0_p8_manual_v0"
        / "gate_release_operator_probe_rows.csv",
        quality_gain_col="gate_release_delta_q_gain",
        verdict_col="gate_release_verdict",
        final_exact_col="changed_node_count",
        final_aligned_col="aligned_changed_node_count",
        endpoint_col="state_endpoint_distance_to_vanilla",
        context_cols=(
            "selector",
            "selected_k",
            "action_mode",
            "selected_node_ids",
            "state_delta_q_vs_vanilla",
            "gate_release_verdict",
        ),
    ),
)


def _numeric(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None or column not in frame.columns:
        return pd.Series(index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _finite(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.6g}"
    return "" if pd.isna(value) else str(value)


def _verdict_counts(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns:
        return ""
    counts = frame[column].fillna("").astype(str).value_counts()
    return ";".join(f"{label}:{int(count)}" for label, count in counts.items() if label)


def _row_value(row: pd.Series | None, column: str) -> Any:
    if row is None or column not in row.index:
        return ""
    return row[column]


def summarize_artifact(spec: ArtifactSpec, *, top_k: int = 5) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frame = pd.read_csv(spec.rows_path)
    quality_gain = _numeric(frame, spec.quality_gain_col)
    final_exact = _numeric(frame, spec.final_exact_col)
    final_aligned = _numeric(frame, spec.final_aligned_col)
    final_exact_only = (
        _numeric(frame, spec.final_exact_only_col)
        if spec.final_exact_only_col
        else (final_exact - final_aligned)
    )
    endpoint = _numeric(frame, spec.endpoint_col)
    pre_exact = _numeric(frame, spec.pre_exact_col)
    pre_aligned = _numeric(frame, spec.pre_aligned_col)
    metric_mask = final_aligned.notna() | final_exact.notna()
    ranking_index = frame.index[metric_mask] if metric_mask.any() else frame.index
    ranking_quality = quality_gain.loc[ranking_index]
    ranking_aligned = final_aligned.loc[ranking_index]
    best_quality_row = (
        frame.loc[ranking_quality.idxmax()]
        if ranking_quality.notna().any()
        else None
    )
    best_aligned_row = (
        frame.loc[ranking_aligned.idxmax()]
        if ranking_aligned.notna().any()
        else None
    )
    summary = {
        "artifact": spec.artifact,
        "rows_path": str(spec.rows_path),
        "row_count": int(len(frame)),
        "metric_row_count": int(metric_mask.sum()),
        "quality_gain_col": spec.quality_gain_col,
        "max_quality_gain": _finite(ranking_quality.max()),
        "mean_quality_gain": _finite(ranking_quality.mean()),
        "best_quality_final_aligned_changed": _finite(
            _row_value(best_quality_row, spec.final_aligned_col)
        ),
        "best_quality_final_exact_changed": _finite(
            _row_value(best_quality_row, spec.final_exact_col)
        ),
        "best_quality_endpoint": _finite(_row_value(best_quality_row, spec.endpoint_col)),
        "max_pre_exact_changed": _finite(pre_exact.max()),
        "max_pre_aligned_changed": _finite(pre_aligned.max()),
        "max_final_exact_changed": _finite(final_exact.max()),
        "max_final_aligned_changed": _finite(final_aligned.max()),
        "mean_final_aligned_changed": _finite(final_aligned.mean()),
        "median_final_aligned_changed": _finite(final_aligned.median()),
        "max_final_exact_only_changed": _finite(final_exact_only.max()),
        "max_endpoint_distance": _finite(endpoint.max()),
        "zero_final_aligned_row_count": int(final_aligned.fillna(-1).eq(0).sum()),
        "verdict_counts": _verdict_counts(frame, spec.verdict_col),
    }
    top_rows: list[dict[str, Any]] = []
    rank_specs = (
        ("quality_gain", ranking_quality),
        ("final_aligned_changed", ranking_aligned),
    )
    for rank_kind, metric in rank_specs:
        if not metric.notna().any():
            continue
        order = metric.sort_values(ascending=False).head(top_k).index
        for rank, row_idx in enumerate(order, start=1):
            row = frame.loc[row_idx]
            item = {
                "artifact": spec.artifact,
                "rank_kind": rank_kind,
                "rank": int(rank),
                "row_index": int(row_idx),
                "quality_gain": _finite(row.get(spec.quality_gain_col)),
                "final_aligned_changed": _finite(row.get(spec.final_aligned_col)),
                "final_exact_changed": _finite(row.get(spec.final_exact_col)),
                "final_exact_only_changed": _finite(
                    row.get(spec.final_exact_only_col)
                    if spec.final_exact_only_col
                    else _finite(row.get(spec.final_exact_col))
                    - _finite(row.get(spec.final_aligned_col))
                ),
                "endpoint_distance": _finite(row.get(spec.endpoint_col)),
            }
            for column in spec.context_cols:
                if column in row.index:
                    item[column] = row[column]
            top_rows.append(item)
    return summary, top_rows


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_fmt(row[column]) for column in columns) + " |")
    return lines


def write_report(path: Path, summary: pd.DataFrame, top_rows: pd.DataFrame) -> None:
    display_cols = [
        "artifact",
        "row_count",
        "metric_row_count",
        "max_quality_gain",
        "max_final_aligned_changed",
        "mean_final_aligned_changed",
        "max_final_exact_changed",
        "max_final_exact_only_changed",
        "max_endpoint_distance",
        "verdict_counts",
    ]
    top_cols = [
        "artifact",
        "rank_kind",
        "rank",
        "quality_gain",
        "final_aligned_changed",
        "final_exact_changed",
        "final_exact_only_changed",
        "endpoint_distance",
        "state_delta_q_vs_vanilla",
        "joint_verdict",
        "stage2_verdict",
        "gate_release_verdict",
        "target_k",
        "context_family",
        "selector",
        "selected_k",
        "action_mode",
        "selected_node_ids",
    ]
    lines = [
        "# Recomputed Operator Metric Review",
        "",
        "This report restates the recomputed operator artifacts using label-invariant",
        "aligned-change and endpoint metrics. Generic exact-label counts are treated",
        "as implementation diagnostics, not basin-level movement by themselves.",
        "",
        "## Summary",
        "",
    ]
    lines.extend(_markdown_table(summary[[c for c in display_cols if c in summary.columns]]))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage2 recovery has zero final aligned movement after polish, so it remains a no-recovery path.",
            "- Gate-release rows are tiny aligned repairs, with final aligned changes of at most two nodes.",
            "- Joint-bundle is the only remaining positive QF signal, but its final aligned movement is compact; large exact-only counts are label namespace accounting.",
            "- The next operator should therefore target the compact aligned core and its boundary context, not optimize for raw exact changed-node counts.",
            "",
            "## Top Rows",
            "",
        ]
    )
    lines.extend(_markdown_table(top_rows[[c for c in top_cols if c in top_rows.columns]]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_review(*, output_dir: Path, top_k: int = 5) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    for spec in DEFAULT_ARTIFACTS:
        summary, tops = summarize_artifact(spec, top_k=top_k)
        summary_rows.append(summary)
        top_rows.extend(tops)
    summary_frame = pd.DataFrame(summary_rows)
    top_frame = pd.DataFrame(top_rows)
    summary_frame.to_csv(output_dir / SUMMARY_FILENAME, index=False)
    top_frame.to_csv(output_dir / TOP_ROWS_FILENAME, index=False)
    write_report(output_dir / REPORT_FILENAME, summary_frame, top_frame)
    payload = {
        "schema": "leiden_basin_recomputed_operator_metric_review.v0",
        "output_dir": str(output_dir),
        "artifact_count": int(len(summary_frame)),
        "paths": {
            "summary": str(output_dir / SUMMARY_FILENAME),
            "top_rows": str(output_dir / TOP_ROWS_FILENAME),
            "report": str(output_dir / REPORT_FILENAME),
        },
    }
    (output_dir / PAYLOAD_FILENAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_review(output_dir=args.output_dir, top_k=args.top_k)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
