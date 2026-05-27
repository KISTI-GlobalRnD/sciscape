#!/usr/bin/env python3
"""Profile observed QF wall route families and lower-wall side routes."""

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
    annotate_wall_route_families,
    summarize_wall_route_families,
)


COMBINED_DIR = REPO_ROOT / (
    "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30"
)
DEFAULT_CLASSIFIER_DIR = COMBINED_DIR / (
    "basin_transition_greedy_failure_classifier_field34_cc_c0_v0"
)
DEFAULT_OUTPUT_DIR = COMBINED_DIR / (
    "basin_transition_wall_route_family_profile_field34_cc_c0_v0"
)

CLASSIFIER_ROWS_FILENAME = "greedy_failure_rows.csv"
ANNOTATED_ROWS_FILENAME = "wall_route_annotated_rows.csv"
FAMILY_ROWS_FILENAME = "wall_route_family_rows.csv"
PREFIX_ROWS_FILENAME = "wall_route_prefix_rows.csv"
SUMMARY_ROWS_FILENAME = "wall_route_summary_rows.csv"
CONFIG_FILENAME = "wall_route_config.json"
SUMMARY_FILENAME = "wall_route_summary.json"
REPORT_FILENAME = "wall_route_report.md"


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


def write_report(
    path: Path,
    *,
    rows: pd.DataFrame,
    family_rows: pd.DataFrame,
    prefix_rows: pd.DataFrame,
    summary_rows: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Wall Route Family Profile",
        "",
        "This artifact asks whether the current branch search found one observed wall or visible side routes.",
        "It is diagnostic-only and works from classifier/path rows without rerunning Leiden.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "classifier_dir",
        "path_rows",
        "family_rows",
        "prefix_rows",
        "candidate_directed_rows",
        "side_route_candidate_rows",
        "lower_wall_side_route_rows",
        "support_gate",
        "progress_margin",
        "side_support_fraction",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")

    lines.extend(["", "## Case Verdict", ""])
    verdict_cols = [
        "case",
        "pair_id",
        "path_rows",
        "candidate_directed_rows",
        "candidate_directed_wall_entries",
        "candidate_directed_wall_values",
        "candidate_directed_prefixes",
        "side_route_candidate_rows",
        "lower_wall_side_route_rows",
        "min_candidate_directed_wall",
        "max_candidate_directed_support",
        "max_side_route_support",
        "max_side_route_progress",
        "wall_route_verdict",
    ]
    lines.extend(
        _markdown_table(
            summary_rows[[column for column in verdict_cols if column in summary_rows]]
        )
    )

    lines.extend(["", "## Prefix Rows", ""])
    prefix_cols = [
        "pair_id",
        "prefix_rank",
        "rows",
        "candidate_directed_rows",
        "side_route_candidate_rows",
        "support_gate_rows",
        "partial_progress_rows",
        "support_max",
        "target_progress_max",
        "delta_q_max",
        "q_wall_min",
        "q_wall_max",
    ]
    lines.extend(
        _markdown_table(
            prefix_rows[[column for column in prefix_cols if column in prefix_rows]],
            max_rows=80,
        )
    )

    lines.extend(["", "## Wall Families", ""])
    family_cols = [
        "pair_id",
        "wall_entry_key",
        "prefix_rank",
        "wall_key",
        "rows",
        "candidate_directed_rows",
        "side_route_candidate_rows",
        "support_gate_rows",
        "partial_progress_rows",
        "support_max",
        "target_progress_max",
        "delta_q_max",
        "q_wall_min",
        "q_wall_max",
        "mutable_min",
        "mutable_max",
        "best_control_status",
    ]
    family_display = family_rows.sort_values(
        [
            "candidate_directed_rows",
            "side_route_candidate_rows",
            "support_max",
            "target_progress_max",
        ],
        ascending=[False, False, False, False],
    )
    lines.extend(
        _markdown_table(
            family_display[[column for column in family_cols if column in family_display]],
            max_rows=80,
        )
    )

    side = rows[rows["wall_side_route_candidate"].astype(bool)].copy()
    if not side.empty:
        side = side.sort_values(
            [
                "path_final_support_distance_to_vanilla",
                "path_final_target_progress_from_vanilla",
                "path_final_delta_q_vs_start",
            ],
            ascending=[False, False, False],
        )
    lines.extend(["", "## Lower-Wall Side-Route Candidates", ""])
    side_cols = [
        "pair_id",
        "path_final_state_id",
        "path_prefix_rank",
        "path_selection_policy",
        "wall_entry_key",
        "path_q_wall",
        "path_final_delta_q_vs_start",
        "path_final_support_distance_to_vanilla",
        "path_final_target_progress_from_vanilla",
        "path_final_mutable_node_count",
        "failure_labels",
    ]
    lines.extend(
        _markdown_table(
            side[[column for column in side_cols if column in side]],
            max_rows=40,
        )
    )

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `candidate_directed_wall_entries` counts observed support-gate, target-progress wall entries; it is not a proof of wall uniqueness.",
            "- `side_route_candidate_rows` are lower-support routes with QF recovery and partial target progress; they are the first places to test for hidden detours.",
            "- If side routes exist below the gate, the next search should expand around them instead of assuming the observed candidate wall is the only route.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_profile(
    *,
    classifier_dir: Path,
    output_dir: Path,
    support_gate: float,
    progress_margin: float,
    side_support_fraction: float,
    wall_round_digits: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    classifier_rows = pd.read_csv(classifier_dir / CLASSIFIER_ROWS_FILENAME)
    rows = annotate_wall_route_families(
        classifier_rows,
        support_gate=support_gate,
        progress_margin=progress_margin,
        side_support_fraction=side_support_fraction,
        wall_round_digits=wall_round_digits,
    )
    family_rows, prefix_rows, summary_rows = summarize_wall_route_families(
        rows,
        support_gate=support_gate,
        progress_margin=progress_margin,
        side_support_fraction=side_support_fraction,
        wall_round_digits=wall_round_digits,
    )
    rows.to_csv(output_dir / ANNOTATED_ROWS_FILENAME, index=False)
    family_rows.to_csv(output_dir / FAMILY_ROWS_FILENAME, index=False)
    prefix_rows.to_csv(output_dir / PREFIX_ROWS_FILENAME, index=False)
    summary_rows.to_csv(output_dir / SUMMARY_ROWS_FILENAME, index=False)
    config = {
        "classifier_dir": str(classifier_dir),
        "support_gate": float(support_gate),
        "progress_margin": float(progress_margin),
        "side_support_fraction": float(side_support_fraction),
        "wall_round_digits": int(wall_round_digits),
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "schema": "leiden_basin_wall_route_family_profile.v0",
        "output_dir": str(output_dir),
        "path_rows": int(len(rows)),
        "family_rows": int(len(family_rows)),
        "prefix_rows": int(len(prefix_rows)),
        "summary_rows": int(len(summary_rows)),
        "candidate_directed_rows": int(rows["path_candidate_directed"].astype(bool).sum()),
        "side_route_candidate_rows": int(rows["wall_side_route_candidate"].sum()),
        "lower_wall_side_route_rows": int(
            summary_rows["lower_wall_side_route_rows"].sum()
        )
        if "lower_wall_side_route_rows" in summary_rows
        else 0,
        **config,
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(
        output_dir / REPORT_FILENAME,
        rows=rows,
        family_rows=family_rows,
        prefix_rows=prefix_rows,
        summary_rows=summary_rows,
        summary=summary,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-dir", type=Path, default=DEFAULT_CLASSIFIER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--support-gate", type=float, default=0.05)
    parser.add_argument("--progress-margin", type=float, default=0.005)
    parser.add_argument("--side-support-fraction", type=float, default=0.75)
    parser.add_argument("--wall-round-digits", type=int, default=6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_profile(
        classifier_dir=args.classifier_dir,
        output_dir=args.output_dir,
        support_gate=args.support_gate,
        progress_margin=args.progress_margin,
        side_support_fraction=args.side_support_fraction,
        wall_round_digits=args.wall_round_digits,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
