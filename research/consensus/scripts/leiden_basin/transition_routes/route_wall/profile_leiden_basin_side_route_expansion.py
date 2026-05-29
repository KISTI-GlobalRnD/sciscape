#!/usr/bin/env python3
"""Profile focused side-route expansion from target-elbow state rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
SCRIPT_ROOT = REPO_ROOT / "research/consensus/scripts"
_SCRIPT_PATHS = [REPO_ROOT, SCRIPT_ROOT]
_SCRIPT_PATHS.extend(path for path in SCRIPT_ROOT.rglob("*") if path.is_dir())
for _script_path in reversed(_SCRIPT_PATHS):
    _script_path_str = str(_script_path)
    if _script_path_str not in sys.path:
        sys.path.insert(0, _script_path_str)


import pandas as pd

from sciscape.clustering.leiden_basin_search import (  # noqa: E402
    classify_branch_greedy_failure_rows,
    compute_pathway_wall_rows,
    summarize_wall_route_families,
)

COMBINED_DIR = REPO_ROOT / (
    "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30"
)
DEFAULT_EXPANSION_DIR = COMBINED_DIR / "basin_transition_side_route_expansion_field34_cc_c0_v0"
DEFAULT_CONTROL_DIR = COMBINED_DIR / "basin_transition_branch_candidate_controls_field34_cc_c0_v0"

STATE_ROWS_FILENAME = "target_elbow_polish_states.csv"
CONTROL_ROWS_FILENAME = "branch_candidate_control_rows.csv"
PATH_ROWS_FILENAME = "side_route_expansion_path_rows.csv"
CLASSIFIED_ROWS_FILENAME = "side_route_expansion_classified_rows.csv"
WALL_FAMILY_ROWS_FILENAME = "side_route_expansion_wall_family_rows.csv"
WALL_PREFIX_ROWS_FILENAME = "side_route_expansion_wall_prefix_rows.csv"
WALL_SUMMARY_ROWS_FILENAME = "side_route_expansion_wall_summary_rows.csv"
SUMMARY_FILENAME = "side_route_expansion_profile_summary.json"
CONFIG_FILENAME = "side_route_expansion_profile_config.json"
REPORT_FILENAME = "side_route_expansion_profile_report.md"

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
    classified_rows: pd.DataFrame,
    family_rows: pd.DataFrame,
    prefix_rows: pd.DataFrame,
    summary_rows: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Side-Route Expansion Profile",
        "",
        "This artifact profiles focused expansion from lower-wall side-route prefixes.",
        "It asks whether those routes can cross the support gate and whether QF recovery survives.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "expansion_dir",
        "state_rows",
        "path_rows",
        "candidate_directed_rows",
        "support_gate_q_recovered_rows",
        "quality_loss_rows",
        "candidate_directed_wall_entries",
        "min_candidate_directed_wall",
        "max_candidate_directed_support",
        "max_candidate_directed_progress",
        "best_candidate_directed_delta_q",
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

    lines.extend(["", "## Prefix Summary", ""])
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

    lines.extend(["", "## Best Candidate-Directed Rows", ""])
    directed = classified_rows[classified_rows["path_candidate_directed"].astype(bool)].copy()
    if not directed.empty:
        directed = directed.sort_values(
            [
                "path_final_delta_q_vs_start",
                "path_final_support_distance_to_vanilla",
                "path_final_target_progress_from_vanilla",
                "path_q_wall",
            ],
            ascending=[False, False, False, True],
        )
    directed_cols = [
        "pair_id",
        "path_prefix_rank",
        "path_policy",
        "path_final_state_id",
        "path_q_wall",
        "path_final_delta_q_vs_start",
        "path_final_support_distance_to_vanilla",
        "path_final_target_progress_from_vanilla",
        "path_final_mutable_node_count",
        "path_support_gate_q_recovered",
        "failure_labels",
        "control_comparison_status",
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
            "- Crossing the support gate is not enough; QF recovery is reported separately.",
            "- A lower-wall route that reaches the gate but remains quality-negative is a detour candidate, not an operator candidate.",
            "- The next useful search should focus on QF recovery around these lower-wall gate-crossing states.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_profile(
    *,
    expansion_dir: Path,
    control_dir: Path | None,
    support_gate: float,
    progress_margin: float,
    support_margin: float,
    material_delta_q: float,
    side_support_fraction: float,
) -> dict[str, Any]:
    states = pd.read_csv(expansion_dir / STATE_ROWS_FILENAME)
    controls = pd.DataFrame()
    if control_dir is not None and (control_dir / CONTROL_ROWS_FILENAME).exists():
        controls = pd.read_csv(control_dir / CONTROL_ROWS_FILENAME)
    path_rows = compute_pathway_wall_rows(
        states,
        source_label="side_route_expansion_v0",
        support_gate=support_gate,
    )
    classified = classify_branch_greedy_failure_rows(
        path_rows,
        state_rows=states,
        control_rows=controls,
        support_gate=support_gate,
        progress_margin=progress_margin,
        support_margin=support_margin,
        material_delta_q=material_delta_q,
    )
    family_rows, prefix_rows, summary_rows = summarize_wall_route_families(
        classified,
        support_gate=support_gate,
        progress_margin=progress_margin,
        side_support_fraction=side_support_fraction,
    )
    path_rows.to_csv(expansion_dir / PATH_ROWS_FILENAME, index=False)
    classified.to_csv(expansion_dir / CLASSIFIED_ROWS_FILENAME, index=False)
    family_rows.to_csv(expansion_dir / WALL_FAMILY_ROWS_FILENAME, index=False)
    prefix_rows.to_csv(expansion_dir / WALL_PREFIX_ROWS_FILENAME, index=False)
    summary_rows.to_csv(expansion_dir / WALL_SUMMARY_ROWS_FILENAME, index=False)

    directed = classified[classified["path_candidate_directed"].astype(bool)].copy()
    support_gate_q_recovered = int(
        classified.get(
            "path_support_gate_q_recovered",
            pd.Series(False, index=classified.index),
        )
        .astype(bool)
        .sum()
    )
    quality_loss_rows = int(
        classified.get(
            "path_final_search_recovery_label",
            pd.Series("", index=classified.index),
        )
        .astype(str)
        .eq("quality_loss")
        .sum()
    )
    summary = {
        "schema": "leiden_basin_side_route_expansion_profile.v0",
        "expansion_dir": str(expansion_dir),
        "control_dir": str(control_dir) if control_dir is not None else "",
        "state_rows": int(len(states)),
        "path_rows": int(len(path_rows)),
        "classified_rows": int(len(classified)),
        "family_rows": int(len(family_rows)),
        "prefix_rows": int(len(prefix_rows)),
        "candidate_directed_rows": int(len(directed)),
        "support_gate_q_recovered_rows": support_gate_q_recovered,
        "quality_loss_rows": quality_loss_rows,
        "candidate_directed_wall_entries": int(
            summary_rows["candidate_directed_wall_entries"].sum()
        )
        if "candidate_directed_wall_entries" in summary_rows
        else 0,
        "min_candidate_directed_wall": float(
            summary_rows["min_candidate_directed_wall"].min()
        )
        if "min_candidate_directed_wall" in summary_rows and not summary_rows.empty
        else math.nan,
        "max_candidate_directed_support": float(
            directed["path_final_support_distance_to_vanilla"].max()
        )
        if not directed.empty
        else math.nan,
        "max_candidate_directed_progress": float(
            directed["path_final_target_progress_from_vanilla"].max()
        )
        if not directed.empty
        else math.nan,
        "best_candidate_directed_delta_q": float(
            directed["path_final_delta_q_vs_start"].max()
        )
        if not directed.empty
        else math.nan,
        "support_gate": float(support_gate),
        "progress_margin": float(progress_margin),
        "support_margin": float(support_margin),
        "material_delta_q": float(material_delta_q),
        "side_support_fraction": float(side_support_fraction),
    }
    (expansion_dir / CONFIG_FILENAME).write_text(
        json.dumps(
            {
                "expansion_dir": str(expansion_dir),
                "control_dir": str(control_dir) if control_dir is not None else "",
                "support_gate": float(support_gate),
                "progress_margin": float(progress_margin),
                "support_margin": float(support_margin),
                "material_delta_q": float(material_delta_q),
                "side_support_fraction": float(side_support_fraction),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (expansion_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(
        expansion_dir / REPORT_FILENAME,
        classified_rows=classified,
        family_rows=family_rows,
        prefix_rows=prefix_rows,
        summary_rows=summary_rows,
        summary=summary,
    )
    return summary

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansion-dir", type=Path, default=DEFAULT_EXPANSION_DIR)
    parser.add_argument("--control-dir", type=Path, default=DEFAULT_CONTROL_DIR)
    parser.add_argument("--support-gate", type=float, default=0.05)
    parser.add_argument("--progress-margin", type=float, default=0.005)
    parser.add_argument("--support-margin", type=float, default=0.01)
    parser.add_argument("--material-delta-q", type=float, default=1.0)
    parser.add_argument("--side-support-fraction", type=float, default=0.75)
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_profile(
        expansion_dir=args.expansion_dir,
        control_dir=args.control_dir,
        support_gate=args.support_gate,
        progress_margin=args.progress_margin,
        support_margin=args.support_margin,
        material_delta_q=args.material_delta_q,
        side_support_fraction=args.side_support_fraction,
    )
    print(summary)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
