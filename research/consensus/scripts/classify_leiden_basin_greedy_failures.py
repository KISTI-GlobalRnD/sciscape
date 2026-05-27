#!/usr/bin/env python3
"""Classify greedy failure modes for basin-transition branch pathways."""

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
    classify_branch_greedy_failure_rows,
    summarize_greedy_failure_rows,
)


COMBINED_DIR = REPO_ROOT / (
    "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30"
)
DEFAULT_BRANCH_DIR = COMBINED_DIR / "basin_transition_branch_target_growth_field34_cc_c0_v0"
DEFAULT_CONTROL_DIR = COMBINED_DIR / (
    "basin_transition_branch_candidate_controls_field34_cc_c0_v0"
)
DEFAULT_OUTPUT_DIR = COMBINED_DIR / (
    "basin_transition_greedy_failure_classifier_field34_cc_c0_v0"
)

STATE_ROWS_FILENAME = "branch_target_growth_states.csv"
PATH_ROWS_FILENAME = "branch_target_growth_path_rows.csv"
CONTROL_ROWS_FILENAME = "branch_candidate_control_rows.csv"
CLASSIFIER_ROWS_FILENAME = "greedy_failure_rows.csv"
CLASSIFIER_SUMMARY_FILENAME = "greedy_failure_summary.csv"
CONFIG_FILENAME = "greedy_failure_config.json"
SUMMARY_FILENAME = "greedy_failure_summary.json"
REPORT_FILENAME = "greedy_failure_report.md"


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


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame:
        return {}
    counts = frame[column].astype(str).value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def write_report(
    path: Path,
    *,
    rows: pd.DataFrame,
    summary_rows: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Greedy Failure Classifier",
        "",
        "This artifact classifies already-generated branch target-growth paths.",
        "It does not rerun Leiden and does not claim an accepted Dongdaemun operator.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "branch_dir",
        "control_dir",
        "path_rows",
        "candidate_directed_rows",
        "support_gate_q_recovered_rows",
        "q_greedy_miss_rows",
        "progress_greedy_miss_rows",
        "closure_compound_miss_rows",
        "polish_recovery_miss_rows",
        "unique_candidate_directed_quality_lag_rows",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")

    lines.extend(["", "## Case Summary", ""])
    summary_cols = [
        "case",
        "pair_id",
        "path_rows",
        "candidate_directed_rows",
        "support_gate_q_recovered_rows",
        "q_greedy_miss_rows",
        "progress_greedy_miss_rows",
        "closure_compound_miss_rows",
        "polish_recovery_miss_rows",
        "unique_candidate_directed_quality_lag_rows",
        "best_state_id",
        "best_failure_labels",
        "best_control_status",
        "best_delta_q",
        "best_support",
        "best_target_progress",
        "best_q_wall",
        "best_mutable",
    ]
    lines.extend(
        _markdown_table(
            summary_rows[[column for column in summary_cols if column in summary_rows]],
        )
    )

    lines.extend(["", "## Candidate-Directed Rows", ""])
    directed = rows[rows["path_candidate_directed"].astype(bool)].copy()
    if not directed.empty:
        directed = directed.sort_values(
            [
                "path_branch_discovery_score",
                "path_final_support_distance_to_vanilla",
                "path_final_delta_q_vs_start",
                "path_q_wall",
            ],
            ascending=[False, False, False, True],
        )
    directed_cols = [
        "pair_id",
        "path_final_state_id",
        "path_selection_policy",
        "failure_labels",
        "control_comparison_status",
        "path_final_delta_q_vs_start",
        "path_final_support_distance_to_vanilla",
        "path_final_target_progress_from_vanilla",
        "path_q_wall",
        "path_final_mutable_node_count",
        "branch_delta_q_minus_best_control",
        "root_greedy_failure_labels",
    ]
    lines.extend(
        _markdown_table(
            directed[[column for column in directed_cols if column in directed]],
            max_rows=40,
        )
    )

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `q_greedy_miss` and `progress_greedy_miss` come from the prefix-level ordered-flip rank evidence.",
            "- `closure_compound_miss` marks direct edits whose useful prefix is entangled with a larger label/context burden.",
            "- `polish_recovery_miss` marks paths that pay a raw QF wall but recover after bounded polish.",
            "- A branch remains diagnostic unless it beats seed controls on material, cost-adjusted value.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_classifier(
    *,
    branch_dir: Path,
    control_dir: Path | None,
    output_dir: Path,
    support_gate: float,
    progress_margin: float,
    support_margin: float,
    material_delta_q: float,
    q_wall_floor: float,
    closure_ratio_threshold: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path_rows = pd.read_csv(branch_dir / PATH_ROWS_FILENAME)
    state_rows = pd.read_csv(branch_dir / STATE_ROWS_FILENAME)
    control_rows = pd.DataFrame()
    if control_dir is not None and (control_dir / CONTROL_ROWS_FILENAME).exists():
        control_rows = pd.read_csv(control_dir / CONTROL_ROWS_FILENAME)

    rows = classify_branch_greedy_failure_rows(
        path_rows,
        state_rows=state_rows,
        control_rows=control_rows,
        support_gate=support_gate,
        progress_margin=progress_margin,
        support_margin=support_margin,
        material_delta_q=material_delta_q,
        q_wall_floor=q_wall_floor,
        closure_ratio_threshold=closure_ratio_threshold,
    )
    summary_rows = summarize_greedy_failure_rows(rows)
    rows.to_csv(output_dir / CLASSIFIER_ROWS_FILENAME, index=False)
    summary_rows.to_csv(output_dir / CLASSIFIER_SUMMARY_FILENAME, index=False)

    config = {
        "branch_dir": str(branch_dir),
        "control_dir": str(control_dir) if control_dir is not None else "",
        "support_gate": float(support_gate),
        "progress_margin": float(progress_margin),
        "support_margin": float(support_margin),
        "material_delta_q": float(material_delta_q),
        "q_wall_floor": float(q_wall_floor),
        "closure_ratio_threshold": float(closure_ratio_threshold),
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "schema": "leiden_basin_greedy_failure_classifier.v0",
        "output_dir": str(output_dir),
        "path_rows": int(len(rows)),
        "summary_rows": int(len(summary_rows)),
        "candidate_directed_rows": int(rows["path_candidate_directed"].astype(bool).sum()),
        "support_gate_q_recovered_rows": int(
            rows["path_support_gate_q_recovered"].astype(bool).sum()
        )
        if "path_support_gate_q_recovered" in rows
        else 0,
        "q_greedy_miss_rows": int(rows["q_greedy_miss"].astype(bool).sum()),
        "progress_greedy_miss_rows": int(rows["progress_greedy_miss"].astype(bool).sum()),
        "closure_compound_miss_rows": int(
            rows["closure_compound_miss"].astype(bool).sum()
        ),
        "polish_recovery_miss_rows": int(rows["polish_recovery_miss"].astype(bool).sum()),
        "unique_candidate_directed_quality_lag_rows": int(
            rows["control_comparison_status"]
            .astype(str)
            .eq("branch_unique_candidate_directed_quality_lag")
            .sum()
        ),
        "failure_label_counts": _value_counts(rows, "failure_labels"),
        "control_status_counts": _value_counts(rows, "control_comparison_status"),
        **config,
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(
        output_dir / REPORT_FILENAME,
        rows=rows,
        summary_rows=summary_rows,
        summary=summary,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-dir", type=Path, default=DEFAULT_BRANCH_DIR)
    parser.add_argument("--control-dir", type=Path, default=DEFAULT_CONTROL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--support-gate", type=float, default=0.05)
    parser.add_argument("--progress-margin", type=float, default=0.005)
    parser.add_argument("--support-margin", type=float, default=0.01)
    parser.add_argument("--material-delta-q", type=float, default=1.0)
    parser.add_argument("--q-wall-floor", type=float, default=0.1)
    parser.add_argument("--closure-ratio-threshold", type=float, default=4.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_classifier(
        branch_dir=args.branch_dir,
        control_dir=args.control_dir,
        output_dir=args.output_dir,
        support_gate=args.support_gate,
        progress_margin=args.progress_margin,
        support_margin=args.support_margin,
        material_delta_q=args.material_delta_q,
        q_wall_floor=args.q_wall_floor,
        closure_ratio_threshold=args.closure_ratio_threshold,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
