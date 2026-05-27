#!/usr/bin/env python3
"""Summarize QF walls along diagnostic Leiden basin-transition pathways."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from sciscape.clustering.leiden_basin_search import (  # noqa: E402
    compute_pathway_wall_rows,
    select_qf_wall_frontier,
    summarize_pathway_wall_rows,
)


BASE_RESULT_DIR = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement"
    / "leiden_multibasin_crossfield_budget12_support_20260519"
    / "combined_with_field30"
)
DEFAULT_INPUT_DIRS = (
    BASE_RESULT_DIR / "basin_transition_search_field34_cc_reachability_v0",
    BASE_RESULT_DIR / "basin_transition_search_field34_cc_v0",
)
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "basin_transition_pathway_wall_stats_field34_cc_v0"

STATE_ROWS_FILENAME = "transition_search_states.csv"
PATHWAY_ROWS_FILENAME = "pathway_wall_rows.csv"
CASE_ROWS_FILENAME = "pathway_wall_case_rows.csv"
FRONTIER_ROWS_FILENAME = "pathway_wall_frontier_rows.csv"
BUCKET_ROWS_FILENAME = "pathway_wall_bucket_rows.csv"
SUMMARY_FILENAME = "pathway_wall_summary.json"
REPORT_FILENAME = "pathway_wall_report.md"
CONFIG_FILENAME = "pathway_wall_config.json"


def _source_label(path: Path) -> str:
    name = path.name
    prefix = "basin_transition_search_field34_cc_"
    return name.removeprefix(prefix)


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 40) -> list[str]:
    if frame.empty:
        return []
    display = frame.head(max_rows)
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _bucket_rows(path_rows: pd.DataFrame) -> pd.DataFrame:
    if path_rows.empty:
        return pd.DataFrame()
    return (
        path_rows.groupby(
            ["source_label", "pair_id", "path_q_wall_bucket"],
            sort=True,
            dropna=False,
        )
        .size()
        .reset_index(name="path_rows")
    )


def write_report(
    path: Path,
    *,
    path_rows: pd.DataFrame,
    case_rows: pd.DataFrame,
    frontier_rows: pd.DataFrame,
    bucket_rows: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Leiden Basin Pathway QF Wall Statistics",
        "",
        "This diagnostic reconstructs each transition-search state as a root-to-terminal pathway and measures the largest QF debt observed along that path.",
        "",
        "QF wall is accounting, not a discovery pruning gate. A pathway can be retained because it reaches a different support region, then judged later by material QF, cost, wall time, memory, and seed controls.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "input_dirs",
        "path_rows",
        "case_rows",
        "frontier_rows",
        "bucket_rows",
        "support_gate",
        "barrier_floor",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")

    lines.extend(["", "## Case Wall Rows", ""])
    case_cols = [
        "source_label",
        "pair_id",
        "path_rows",
        "support_gate_rows",
        "q_recovered_rows",
        "support_gate_q_recovered_rows",
        "wall_crossed_rows",
        "zero_wall_rows",
        "q_wall_min",
        "q_wall_median",
        "q_wall_p90",
        "q_wall_max",
        "support_gate_q_wall_min",
        "support_gate_q_wall_median",
        "support_gate_q_wall_max",
        "min_wall_gate_state_id",
        "min_wall_gate_q_wall",
        "min_wall_gate_delta_q",
        "min_wall_gate_support_distance_to_vanilla",
        "min_wall_gate_target_progress",
        "min_wall_gate_coverage",
        "min_wall_gate_mutable_nodes",
        "min_wall_gate_prefix_raw_barrier",
        "min_wall_gate_wall_reduction_vs_prefix_raw_barrier",
        "best_support_state_id",
        "best_support_q_wall",
        "best_support_delta_q",
        "best_support_support_distance_to_vanilla",
        "best_support_target_progress",
        "best_support_coverage",
        "best_support_mutable_nodes",
        "best_support_prefix_raw_barrier",
        "best_support_wall_reduction_vs_prefix_raw_barrier",
    ]
    lines.extend(
        _markdown_table(case_rows[[c for c in case_cols if c in case_rows.columns]])
    )

    lines.extend(["", "## Wall Buckets", ""])
    lines.extend(_markdown_table(bucket_rows, max_rows=80))

    lines.extend(["", "## Frontier Rows", ""])
    frontier_cols = [
        "source_label",
        "pair_id",
        "path_final_state_id",
        "path_depth",
        "path_q_wall",
        "path_min_delta_q_vs_start",
        "path_final_delta_q_vs_start",
        "path_final_support_distance_to_vanilla",
        "path_final_target_progress_from_vanilla",
        "path_final_target_coverage_fraction",
        "path_final_mutable_node_count",
        "path_target_progress_per_wall_floor",
        "path_wall_frontier_score",
        "path_final_search_recovery_label",
        "path_final_reachability_label",
        "path_applied_actions",
    ]
    lines.extend(
        _markdown_table(
            frontier_rows[[c for c in frontier_cols if c in frontier_rows.columns]],
            max_rows=100,
        )
    )

    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "- A low QF wall does not prove a better operator; it only identifies a cheaper candidate route through the basin landscape.",
            "- A high-support route with weak target progress is a reachability signal, not a material basin transition claim.",
            "- The next operator decision should compare wall height, retained support progress, final QF, mutable-node cost, wall time, and seed-control baselines together.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_summary(
    *,
    input_dirs: tuple[Path, ...],
    input_labels: tuple[str, ...],
    output_dir: Path,
    support_gate: float,
    barrier_floor: float,
    max_frontier_rows: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = input_labels or tuple(_source_label(path) for path in input_dirs)
    if len(labels) != len(input_dirs):
        raise ValueError("--input-label count must match --input-dir count")

    frames: list[pd.DataFrame] = []
    for input_dir, label in zip(input_dirs, labels, strict=True):
        state_path = input_dir / STATE_ROWS_FILENAME
        if not state_path.exists():
            raise FileNotFoundError(f"Missing transition states: {state_path}")
        states = pd.read_csv(state_path)
        paths = compute_pathway_wall_rows(
            states,
            source_label=label,
            support_gate=support_gate,
            barrier_floor=barrier_floor,
        )
        paths.insert(1, "source_dir", str(input_dir))
        frames.append(paths)
    path_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    case_rows = summarize_pathway_wall_rows(path_rows, support_gate=support_gate)
    frontier_frames: list[pd.DataFrame] = []
    if not path_rows.empty:
        for _, group in path_rows.groupby(["source_label", "pair_id"], sort=True):
            frontier_frames.append(
                select_qf_wall_frontier(group, max_rows=max_frontier_rows)
            )
    frontier_rows = (
        pd.concat(frontier_frames, ignore_index=True)
        if frontier_frames
        else pd.DataFrame()
    )
    bucket_rows = _bucket_rows(path_rows)

    path_rows.to_csv(output_dir / PATHWAY_ROWS_FILENAME, index=False)
    case_rows.to_csv(output_dir / CASE_ROWS_FILENAME, index=False)
    frontier_rows.to_csv(output_dir / FRONTIER_ROWS_FILENAME, index=False)
    bucket_rows.to_csv(output_dir / BUCKET_ROWS_FILENAME, index=False)

    config = {
        "input_dirs": [str(path) for path in input_dirs],
        "input_labels": list(labels),
        "output_dir": str(output_dir),
        "support_gate": float(support_gate),
        "barrier_floor": float(barrier_floor),
        "max_frontier_rows": int(max_frontier_rows),
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "schema": "leiden_basin_pathway_qf_wall_stats.v0",
        "path_rows": int(len(path_rows)),
        "case_rows": int(len(case_rows)),
        "frontier_rows": int(len(frontier_rows)),
        "bucket_rows": int(len(bucket_rows)),
        **config,
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(
        output_dir / REPORT_FILENAME,
        path_rows=path_rows,
        case_rows=case_rows,
        frontier_rows=frontier_rows,
        bucket_rows=bucket_rows,
        summary=summary,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        default=None,
        help="Transition-search artifact directory. Defaults compare reachability-first and state-greedy.",
    )
    parser.add_argument(
        "--input-label",
        action="append",
        default=None,
        help="Label for the corresponding --input-dir.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--support-gate", type=float, default=0.05)
    parser.add_argument("--barrier-floor", type=float, default=1.0)
    parser.add_argument("--max-frontier-rows", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_dirs = tuple(args.input_dir) if args.input_dir else DEFAULT_INPUT_DIRS
    input_labels = tuple(args.input_label or ())
    summary = run_summary(
        input_dirs=input_dirs,
        input_labels=input_labels,
        output_dir=args.output_dir,
        support_gate=args.support_gate,
        barrier_floor=args.barrier_floor,
        max_frontier_rows=args.max_frontier_rows,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
