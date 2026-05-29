#!/usr/bin/env python3
"""Collect diagnostic evidence for recoverable basin-wall tunneling."""

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
    TUNNEL_ROUTE_RECOVERABLE,
    TUNNEL_ROUTE_UNRECOVERED_DETOUR,
    annotate_pathway_debt_area_rows,
    annotate_tunneling_evidence_rows,
    compute_pathway_wall_rows,
    summarize_tunneling_evidence_rows,
    trace_tunneling_path_states,
)

COMBINED_DIR = REPO_ROOT / (
    "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30"
)
DEFAULT_BRANCH_DIR = COMBINED_DIR / "basin_transition_branch_target_growth_field34_cc_c0_v0"
DEFAULT_SIDE_ROUTE_DIR = COMBINED_DIR / "basin_transition_side_route_expansion_field34_cc_c0_v0"
DEFAULT_OUTPUT_DIR = COMBINED_DIR / "basin_transition_tunneling_evidence_field34_cc_c0_v0"

BRANCH_STATES_FILENAME = "branch_target_growth_states.csv"
SIDE_ROUTE_STATES_FILENAME = "target_elbow_polish_states.csv"
ROWS_FILENAME = "tunneling_evidence_rows.csv"
TRACE_ROWS_FILENAME = "tunneling_trace_rows.csv"
SUMMARY_ROWS_FILENAME = "tunneling_summary_rows.csv"
CONTRAST_ROWS_FILENAME = "tunneling_contrast_rows.csv"
CONFIG_FILENAME = "tunneling_evidence_config.json"
SUMMARY_FILENAME = "tunneling_evidence_summary.json"
REPORT_FILENAME = "tunneling_evidence_report.md"

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

def _best_row(
    rows: pd.DataFrame,
    *,
    sort_columns: list[str],
    ascending: list[bool],
    label: str,
) -> dict[str, Any] | None:
    if rows.empty:
        return None
    row = rows.sort_values(sort_columns, ascending=ascending).iloc[0]
    out = row.to_dict()
    out["contrast_label"] = label
    return out

def _load_tunnel_rows(
    *,
    artifact_label: str,
    artifact_dir: Path,
    state_filename: str,
    support_gate: float,
    progress_margin: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    states = pd.read_csv(artifact_dir / state_filename)
    paths = compute_pathway_wall_rows(
        states,
        source_label=artifact_label,
        support_gate=support_gate,
    )
    paths = annotate_pathway_debt_area_rows(
        paths,
        state_rows=states,
        support_gate=support_gate,
    )
    paths.insert(0, "artifact_label", artifact_label)
    paths.insert(1, "artifact_dir", str(artifact_dir))
    paths = annotate_tunneling_evidence_rows(
        paths,
        support_gate=support_gate,
        progress_margin=progress_margin,
    )
    states.insert(0, "artifact_label", artifact_label)
    return paths, states

def _select_contrast_rows(rows: pd.DataFrame) -> pd.DataFrame:
    recovered = rows[rows["tunnel_route_label"].eq(TUNNEL_ROUTE_RECOVERABLE)]
    detours = rows[rows["tunnel_route_label"].eq(TUNNEL_ROUTE_UNRECOVERED_DETOUR)]
    selected: list[dict[str, Any]] = []
    for candidate in (
        _best_row(
            recovered,
            sort_columns=[
                "tunnel_recovered_shortcut_score",
                "path_final_delta_q_vs_start",
                "path_final_support_distance_to_vanilla",
                "path_q_debt_area_step",
            ],
            ascending=[False, False, False, True],
            label="best_recoverable_shortcut",
        ),
        _best_row(
            recovered,
            sort_columns=[
                "path_final_support_distance_to_vanilla",
                "path_final_target_progress_from_vanilla",
                "path_final_delta_q_vs_start",
                "path_q_debt_area_step",
            ],
            ascending=[False, False, False, True],
            label="widest_recoverable_support",
        ),
        _best_row(
            detours,
            sort_columns=[
                "path_q_debt_area_step",
                "path_final_delta_q_vs_start",
                "path_final_support_distance_to_vanilla",
            ],
            ascending=[True, False, False],
            label="lowest_area_unrecovered_detour",
        ),
        _best_row(
            detours,
            sort_columns=[
                "path_final_delta_q_vs_start",
                "path_final_support_distance_to_vanilla",
                "path_q_debt_area_step",
            ],
            ascending=[False, False, True],
            label="best_quality_unrecovered_detour",
        ),
        _best_row(
            detours,
            sort_columns=[
                "path_final_support_distance_to_vanilla",
                "path_final_target_progress_from_vanilla",
                "path_final_delta_q_vs_start",
            ],
            ascending=[False, False, False],
            label="widest_unrecovered_detour",
        ),
    ):
        if candidate is not None:
            selected.append(candidate)
    if not selected:
        return pd.DataFrame()
    out = pd.DataFrame(selected)
    return out.drop_duplicates(subset=["contrast_label", "path_final_state_id"])

def _trace_contrast_rows(
    contrast_rows: pd.DataFrame,
    *,
    state_rows_by_artifact: dict[str, pd.DataFrame],
    support_gate: float,
    progress_margin: float,
) -> pd.DataFrame:
    traces: list[pd.DataFrame] = []
    for artifact_label, group in contrast_rows.groupby("artifact_label", sort=True):
        states = state_rows_by_artifact[str(artifact_label)]
        trace = trace_tunneling_path_states(
            group,
            state_rows=states,
            support_gate=support_gate,
            progress_margin=progress_margin,
        )
        if not trace.empty:
            contrast_map = dict(
                zip(
                    group["path_final_state_id"].astype(str),
                    group["contrast_label"].astype(str),
                    strict=False,
                )
            )
            trace["contrast_label"] = trace["path_final_state_id"].map(contrast_map)
            traces.append(trace)
    return pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()

def _summary_dict(
    rows: pd.DataFrame,
    contrast_rows: pd.DataFrame,
    *,
    output_dir: Path,
    support_gate: float,
    progress_margin: float,
) -> dict[str, Any]:
    recovered = rows[rows["tunnel_route_label"].eq(TUNNEL_ROUTE_RECOVERABLE)]
    detours = rows[rows["tunnel_route_label"].eq(TUNNEL_ROUTE_UNRECOVERED_DETOUR)]
    best_recovered = contrast_rows[
        contrast_rows["contrast_label"].eq("best_recoverable_shortcut")
    ]
    best_detour = contrast_rows[
        contrast_rows["contrast_label"].eq("lowest_area_unrecovered_detour")
    ]
    summary: dict[str, Any] = {
        "schema": "leiden_basin_tunneling_evidence.v0",
        "output_dir": str(output_dir),
        "path_rows": int(len(rows)),
        "recoverable_tunnel_rows": int(len(recovered)),
        "unrecovered_detour_rows": int(len(detours)),
        "support_gate": float(support_gate),
        "progress_margin": float(progress_margin),
    }
    if not best_recovered.empty:
        row = best_recovered.iloc[0]
        summary.update(
            {
                "best_recoverable_state_id": row["path_final_state_id"],
                "best_recoverable_q_wall": float(row["path_q_wall"]),
                "best_recoverable_debt_area_step": float(
                    row["path_q_debt_area_step"]
                ),
                "best_recoverable_delta_q": float(row["path_final_delta_q_vs_start"]),
                "best_recoverable_support": float(
                    row["path_final_support_distance_to_vanilla"]
                ),
                "best_recoverable_progress": float(
                    row["path_final_target_progress_from_vanilla"]
                ),
            }
        )
    if not best_detour.empty:
        row = best_detour.iloc[0]
        summary.update(
            {
                "lowest_detour_state_id": row["path_final_state_id"],
                "lowest_detour_q_wall": float(row["path_q_wall"]),
                "lowest_detour_debt_area_step": float(row["path_q_debt_area_step"]),
                "lowest_detour_delta_q": float(row["path_final_delta_q_vs_start"]),
                "lowest_detour_support": float(
                    row["path_final_support_distance_to_vanilla"]
                ),
                "lowest_detour_progress": float(
                    row["path_final_target_progress_from_vanilla"]
                ),
            }
        )
    return summary

def write_report(
    path: Path,
    *,
    rows: pd.DataFrame,
    summary_rows: pd.DataFrame,
    contrast_rows: pd.DataFrame,
    trace_rows: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Tunneling Evidence Profile",
        "",
        "This artifact treats tunneling as a diagnostic basin-transition pattern: a candidate-directed path pays temporary QF debt, keeps the debt area short, and recovers QF by the terminal state.",
        "It contrasts recoverable tunnels against low-wall detours that cross the support gate but remain quality-negative.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "output_dir",
        "path_rows",
        "recoverable_tunnel_rows",
        "unrecovered_detour_rows",
        "best_recoverable_q_wall",
        "best_recoverable_debt_area_step",
        "best_recoverable_delta_q",
        "lowest_detour_q_wall",
        "lowest_detour_debt_area_step",
        "lowest_detour_delta_q",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")

    lines.extend(["", "## Route Class Summary", ""])
    summary_cols = [
        "artifact_label",
        "tunnel_route_label",
        "rows",
        "candidate_directed_rows",
        "q_recovered_rows",
        "q_wall_min",
        "q_wall_median",
        "debt_area_step_min",
        "debt_area_step_median",
        "final_delta_q_max",
        "support_max",
        "target_progress_max",
        "recovered_shortcut_score_max",
        "recovery_slope_step_max",
    ]
    lines.extend(
        _markdown_table(
            summary_rows[[column for column in summary_cols if column in summary_rows]],
            max_rows=30,
        )
    )

    lines.extend(["", "## Contrast Rows", ""])
    contrast_cols = [
        "contrast_label",
        "artifact_label",
        "tunnel_route_label",
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
        "tunnel_recovered_shortcut_score",
        "path_recovery_slope_per_step",
    ]
    lines.extend(
        _markdown_table(
            contrast_rows[[column for column in contrast_cols if column in contrast_rows]],
            max_rows=20,
        )
    )

    lines.extend(["", "## Contrast Traces", ""])
    trace_cols = [
        "contrast_label",
        "artifact_label",
        "trace_step_index",
        "trace_phase",
        "trace_action_type",
        "trace_q_debt",
        "trace_cumulative_q_debt_area_step",
        "trace_delta_q_vs_start",
        "trace_support_distance_to_vanilla",
        "trace_target_progress_from_vanilla",
        "trace_mutable_node_count",
        "trace_marginal_mutable_node_count",
        "trace_first_candidate_directed_step",
        "trace_first_q_recovered_step",
    ]
    lines.extend(
        _markdown_table(
            trace_rows[[column for column in trace_cols if column in trace_rows]],
            max_rows=80,
        )
    )

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- A recovered tunnel is not a low-wall route; it is a recoverable short route through a wall.",
            "- An unrecovered detour can have a lower wall and still be a weaker operator target if its terminal QF remains negative.",
            "- The trace rows identify where a future operator needs to intervene: before the wall, at the wall peak, or after the gate-crossing but before QF recovery.",
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
    branch_rows, branch_states = _load_tunnel_rows(
        artifact_label="branch_target_growth",
        artifact_dir=branch_dir,
        state_filename=BRANCH_STATES_FILENAME,
        support_gate=support_gate,
        progress_margin=progress_margin,
    )
    side_rows, side_states = _load_tunnel_rows(
        artifact_label="side_route_expansion",
        artifact_dir=side_route_dir,
        state_filename=SIDE_ROUTE_STATES_FILENAME,
        support_gate=support_gate,
        progress_margin=progress_margin,
    )
    rows = pd.concat([branch_rows, side_rows], axis=0, ignore_index=True)
    summary_rows = summarize_tunneling_evidence_rows(rows)
    contrast_rows = _select_contrast_rows(rows)
    trace_rows = _trace_contrast_rows(
        contrast_rows,
        state_rows_by_artifact={
            "branch_target_growth": branch_states,
            "side_route_expansion": side_states,
        },
        support_gate=support_gate,
        progress_margin=progress_margin,
    )

    rows.to_csv(output_dir / ROWS_FILENAME, index=False)
    summary_rows.to_csv(output_dir / SUMMARY_ROWS_FILENAME, index=False)
    contrast_rows.to_csv(output_dir / CONTRAST_ROWS_FILENAME, index=False)
    trace_rows.to_csv(output_dir / TRACE_ROWS_FILENAME, index=False)
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
    summary = _summary_dict(
        rows,
        contrast_rows,
        output_dir=output_dir,
        support_gate=support_gate,
        progress_margin=progress_margin,
    )
    summary.update(config)
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(
        output_dir / REPORT_FILENAME,
        rows=rows,
        summary_rows=summary_rows,
        contrast_rows=contrast_rows,
        trace_rows=trace_rows,
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
