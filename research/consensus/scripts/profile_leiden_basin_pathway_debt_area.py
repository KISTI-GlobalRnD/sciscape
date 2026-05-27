#!/usr/bin/env python3
"""Compare basin-transition paths by QF debt area, not only peak wall."""

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
    annotate_pathway_debt_area_rows,
    compute_pathway_wall_rows,
)


COMBINED_DIR = REPO_ROOT / (
    "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30"
)
DEFAULT_BRANCH_DIR = COMBINED_DIR / "basin_transition_branch_target_growth_field34_cc_c0_v0"
DEFAULT_SIDE_ROUTE_DIR = COMBINED_DIR / "basin_transition_side_route_expansion_field34_cc_c0_v0"
DEFAULT_OUTPUT_DIR = COMBINED_DIR / "basin_transition_pathway_debt_area_compare_field34_cc_c0_v0"

BRANCH_STATES_FILENAME = "branch_target_growth_states.csv"
SIDE_ROUTE_STATES_FILENAME = "target_elbow_polish_states.csv"
ROWS_FILENAME = "pathway_debt_area_rows.csv"
SUMMARY_ROWS_FILENAME = "pathway_debt_area_summary_rows.csv"
FRONTIER_ROWS_FILENAME = "pathway_debt_area_frontier_rows.csv"
CONFIG_FILENAME = "pathway_debt_area_config.json"
SUMMARY_FILENAME = "pathway_debt_area_summary.json"
REPORT_FILENAME = "pathway_debt_area_report.md"


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _quantile(values: pd.Series, q: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return math.nan
    return float(numeric.quantile(float(q)))


def _best_row(
    rows: pd.DataFrame,
    *,
    sort_columns: list[str],
    ascending: list[bool],
) -> pd.Series | None:
    if rows.empty:
        return None
    return rows.sort_values(sort_columns, ascending=ascending).iloc[0]


def _row_value(row: pd.Series | None, column: str) -> Any:
    if row is None:
        return math.nan
    return row.get(column, math.nan)


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


def _annotate_route_labels(
    rows: pd.DataFrame,
    *,
    support_gate: float,
    progress_margin: float,
) -> pd.DataFrame:
    rows = rows.copy()
    support = _numeric(rows, "path_final_support_distance_to_vanilla")
    progress = _numeric(rows, "path_final_target_progress_from_vanilla")
    delta_q = _numeric(rows, "path_final_delta_q_vs_start", default=math.nan)
    rows["path_area_candidate_directed"] = (
        (support >= float(support_gate)) & (progress > float(progress_margin))
    )
    rows["path_area_quality_recovered"] = rows["path_area_candidate_directed"] & (
        delta_q >= 0.0
    )
    rows["path_area_quality_loss"] = rows["path_area_candidate_directed"] & (
        delta_q < 0.0
    )
    labels: list[str] = []
    for _, row in rows.iterrows():
        if bool(row["path_area_quality_recovered"]):
            labels.append("candidate_directed_q_recovered")
        elif bool(row["path_area_quality_loss"]):
            labels.append("candidate_directed_quality_loss")
        elif float(row.get("path_final_support_distance_to_vanilla", 0.0)) >= float(
            support_gate
        ):
            labels.append("support_gate_without_target_progress")
        elif float(row.get("path_final_target_progress_from_vanilla", 0.0)) > float(
            progress_margin
        ):
            labels.append("partial_target_progress")
        else:
            labels.append("stalled")
    rows["path_area_route_label"] = labels
    return rows


def _load_artifact_paths(
    *,
    artifact_label: str,
    artifact_dir: Path,
    state_filename: str,
    support_gate: float,
    progress_margin: float,
) -> pd.DataFrame:
    states = pd.read_csv(artifact_dir / state_filename)
    path_rows = compute_pathway_wall_rows(
        states,
        source_label=artifact_label,
        support_gate=support_gate,
    )
    rows = annotate_pathway_debt_area_rows(
        path_rows,
        state_rows=states,
        support_gate=support_gate,
    )
    rows.insert(0, "artifact_label", artifact_label)
    rows.insert(1, "artifact_dir", str(artifact_dir))
    rows = _annotate_route_labels(
        rows,
        support_gate=support_gate,
        progress_margin=progress_margin,
    )
    return rows


def _summary_row(group: pd.DataFrame, artifact_label: str) -> dict[str, Any]:
    directed = group[group["path_area_candidate_directed"].astype(bool)].copy()
    recovered = group[group["path_area_quality_recovered"].astype(bool)].copy()
    quality_loss = group[group["path_area_quality_loss"].astype(bool)].copy()
    shortcut_scope = directed if not directed.empty else group

    best_short_step = _best_row(
        shortcut_scope,
        sort_columns=[
            "path_shortcut_score_step",
            "path_final_delta_q_vs_start",
            "path_final_support_distance_to_vanilla",
            "path_q_debt_area_step",
        ],
        ascending=[False, False, False, True],
    )
    best_short_mutable = _best_row(
        shortcut_scope,
        sort_columns=[
            "path_shortcut_score_mutable",
            "path_final_delta_q_vs_start",
            "path_final_support_distance_to_vanilla",
            "path_q_debt_area_mutable",
        ],
        ascending=[False, False, False, True],
    )
    best_directed = _best_row(
        directed,
        sort_columns=[
            "path_area_quality_recovered",
            "path_final_delta_q_vs_start",
            "path_final_support_distance_to_vanilla",
            "path_q_debt_area_step",
        ],
        ascending=[False, False, False, True],
    )
    lowest_directed_area = _best_row(
        directed,
        sort_columns=[
            "path_q_debt_area_step",
            "path_final_delta_q_vs_start",
            "path_final_support_distance_to_vanilla",
        ],
        ascending=[True, False, False],
    )

    return {
        "artifact_label": artifact_label,
        "path_rows": int(len(group)),
        "candidate_directed_rows": int(len(directed)),
        "support_gate_q_recovered_rows": int(len(recovered)),
        "candidate_directed_quality_loss_rows": int(len(quality_loss)),
        "q_wall_min": _quantile(group["path_q_wall"], 0.0),
        "q_wall_median": _quantile(group["path_q_wall"], 0.5),
        "q_wall_max": _quantile(group["path_q_wall"], 1.0),
        "candidate_q_wall_min": _quantile(directed["path_q_wall"], 0.0),
        "candidate_q_wall_median": _quantile(directed["path_q_wall"], 0.5),
        "candidate_q_wall_max": _quantile(directed["path_q_wall"], 1.0),
        "candidate_debt_area_step_min": _quantile(
            directed["path_q_debt_area_step"], 0.0
        ),
        "candidate_debt_area_step_median": _quantile(
            directed["path_q_debt_area_step"], 0.5
        ),
        "candidate_debt_area_mutable_min": _quantile(
            directed["path_q_debt_area_mutable"], 0.0
        ),
        "candidate_debt_area_mutable_median": _quantile(
            directed["path_q_debt_area_mutable"], 0.5
        ),
        "max_candidate_support": _quantile(
            directed["path_final_support_distance_to_vanilla"], 1.0
        ),
        "max_candidate_progress": _quantile(
            directed["path_final_target_progress_from_vanilla"], 1.0
        ),
        "best_candidate_delta_q": _quantile(
            directed["path_final_delta_q_vs_start"], 1.0
        ),
        "best_shortcut_step_state_id": _row_value(
            best_short_step, "path_final_state_id"
        ),
        "best_shortcut_step_score": _row_value(
            best_short_step, "path_shortcut_score_step"
        ),
        "best_shortcut_step_delta_q": _row_value(
            best_short_step, "path_final_delta_q_vs_start"
        ),
        "best_shortcut_step_q_wall": _row_value(best_short_step, "path_q_wall"),
        "best_shortcut_step_debt_area": _row_value(
            best_short_step, "path_q_debt_area_step"
        ),
        "best_shortcut_mutable_state_id": _row_value(
            best_short_mutable, "path_final_state_id"
        ),
        "best_shortcut_mutable_score": _row_value(
            best_short_mutable, "path_shortcut_score_mutable"
        ),
        "best_directed_state_id": _row_value(best_directed, "path_final_state_id"),
        "best_directed_delta_q": _row_value(
            best_directed, "path_final_delta_q_vs_start"
        ),
        "best_directed_q_wall": _row_value(best_directed, "path_q_wall"),
        "best_directed_debt_area_step": _row_value(
            best_directed, "path_q_debt_area_step"
        ),
        "lowest_directed_area_state_id": _row_value(
            lowest_directed_area, "path_final_state_id"
        ),
        "lowest_directed_area_delta_q": _row_value(
            lowest_directed_area, "path_final_delta_q_vs_start"
        ),
        "lowest_directed_area_q_wall": _row_value(
            lowest_directed_area, "path_q_wall"
        ),
        "lowest_directed_area_step": _row_value(
            lowest_directed_area, "path_q_debt_area_step"
        ),
    }


def summarize_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    out = []
    for artifact_label, group in rows.groupby("artifact_label", sort=True):
        out.append(_summary_row(group, str(artifact_label)))
    return pd.DataFrame(out)


def select_frontier_rows(rows: pd.DataFrame, *, max_rows: int = 80) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    candidates = rows.sort_values(
        [
            "path_area_quality_recovered",
            "path_area_candidate_directed",
            "path_shortcut_score_step",
            "path_final_delta_q_vs_start",
            "path_final_support_distance_to_vanilla",
            "path_q_debt_area_step",
        ],
        ascending=[False, False, False, False, False, True],
    ).copy()
    return candidates.head(max_rows)


def write_report(
    path: Path,
    *,
    rows: pd.DataFrame,
    summary_rows: pd.DataFrame,
    frontier_rows: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Pathway Debt-Area Comparison",
        "",
        "This artifact compares transition paths by peak QF wall and by the area under the QF debt curve.",
        "A high wall can still be a plausible shortcut if it is short and recovers quickly; a low wall can still be a poor detour if it stays quality-negative.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "output_dir",
        "path_rows",
        "candidate_directed_rows",
        "support_gate_q_recovered_rows",
        "candidate_directed_quality_loss_rows",
        "support_gate",
        "progress_margin",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")

    lines.extend(["", "## Artifact Comparison", ""])
    comparison_cols = [
        "artifact_label",
        "path_rows",
        "candidate_directed_rows",
        "support_gate_q_recovered_rows",
        "candidate_directed_quality_loss_rows",
        "candidate_q_wall_min",
        "candidate_q_wall_median",
        "candidate_debt_area_step_min",
        "candidate_debt_area_step_median",
        "max_candidate_support",
        "max_candidate_progress",
        "best_candidate_delta_q",
        "best_directed_state_id",
        "best_directed_delta_q",
        "best_directed_q_wall",
        "best_directed_debt_area_step",
        "lowest_directed_area_state_id",
        "lowest_directed_area_delta_q",
        "lowest_directed_area_q_wall",
        "lowest_directed_area_step",
    ]
    lines.extend(
        _markdown_table(
            summary_rows[[column for column in comparison_cols if column in summary_rows]],
            max_rows=20,
        )
    )

    lines.extend(["", "## Shortcut Frontier", ""])
    frontier_cols = [
        "artifact_label",
        "path_area_route_label",
        "pair_id",
        "path_prefix_rank",
        "path_selection_policy",
        "path_policy",
        "path_final_state_id",
        "path_q_wall",
        "path_q_debt_area_step",
        "path_q_debt_area_mutable",
        "path_wall_duration_steps",
        "path_final_delta_q_vs_start",
        "path_final_support_distance_to_vanilla",
        "path_final_target_progress_from_vanilla",
        "path_shortcut_score_step",
        "path_recovery_slope_per_step",
    ]
    lines.extend(
        _markdown_table(
            frontier_rows[[column for column in frontier_cols if column in frontier_rows]],
            max_rows=40,
        )
    )

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `path_q_wall` is the peak debt: the highest instantaneous wall.",
            "- `path_q_debt_area_step` is the path-integrated debt over visited states.",
            "- `path_q_debt_area_mutable` weights debt by newly mutable nodes; it approximates how much state mass paid the debt.",
            "- `path_shortcut_score_*` is diagnostic only. It ranks routes that buy support/progress/final QF with less debt area.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_profile(
    *,
    branch_dir: Path,
    side_route_dir: Path,
    output_dir: Path,
    support_gate: float,
    progress_margin: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    branch = _load_artifact_paths(
        artifact_label="branch_target_growth",
        artifact_dir=branch_dir,
        state_filename=BRANCH_STATES_FILENAME,
        support_gate=support_gate,
        progress_margin=progress_margin,
    )
    side = _load_artifact_paths(
        artifact_label="side_route_expansion",
        artifact_dir=side_route_dir,
        state_filename=SIDE_ROUTE_STATES_FILENAME,
        support_gate=support_gate,
        progress_margin=progress_margin,
    )
    rows = pd.concat([branch, side], axis=0, ignore_index=True)
    summary_rows = summarize_rows(rows)
    frontier_rows = select_frontier_rows(rows)

    rows.to_csv(output_dir / ROWS_FILENAME, index=False)
    summary_rows.to_csv(output_dir / SUMMARY_ROWS_FILENAME, index=False)
    frontier_rows.to_csv(output_dir / FRONTIER_ROWS_FILENAME, index=False)
    config = {
        "branch_dir": str(branch_dir),
        "side_route_dir": str(side_route_dir),
        "output_dir": str(output_dir),
        "support_gate": float(support_gate),
        "progress_margin": float(progress_margin),
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "schema": "leiden_basin_pathway_debt_area_compare.v0",
        "path_rows": int(len(rows)),
        "candidate_directed_rows": int(
            rows["path_area_candidate_directed"].astype(bool).sum()
        ),
        "support_gate_q_recovered_rows": int(
            rows["path_area_quality_recovered"].astype(bool).sum()
        ),
        "candidate_directed_quality_loss_rows": int(
            rows["path_area_quality_loss"].astype(bool).sum()
        ),
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
        frontier_rows=frontier_rows,
        summary=summary,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-dir", type=Path, default=DEFAULT_BRANCH_DIR)
    parser.add_argument("--side-route-dir", type=Path, default=DEFAULT_SIDE_ROUTE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--support-gate", type=float, default=0.05)
    parser.add_argument("--progress-margin", type=float, default=0.005)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_profile(
        branch_dir=args.branch_dir,
        side_route_dir=args.side_route_dir,
        output_dir=args.output_dir,
        support_gate=args.support_gate,
        progress_margin=args.progress_margin,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
