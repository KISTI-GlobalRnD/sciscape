#!/usr/bin/env python3
"""Rank existing basin-transition paths for Dongdaemun tunneling design."""

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
    TUNNEL_OPERATOR_BACKGROUND,
    TUNNEL_OPERATOR_RECOVERABLE_SEED,
    TUNNEL_OPERATOR_RECOVERY_TARGET,
    annotate_pathway_debt_area_rows,
    annotate_tunneling_evidence_rows,
    compute_pathway_wall_rows,
    rank_tunneling_operator_candidates,
    select_tunneling_operator_candidates,
    summarize_tunneling_evidence_rows,
    trace_tunneling_path_states,
)


COMBINED_DIR = REPO_ROOT / (
    "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30"
)
DEFAULT_OUTPUT_DIR = COMBINED_DIR / "basin_transition_tunneling_path_rank_field34_cc_v0"

DEFAULT_ARTIFACTS = (
    (
        "branch_target_growth_c0",
        "basin_transition_branch_target_growth_field34_cc_c0_v0",
        "branch_target_growth_states.csv",
    ),
    (
        "branch_target_growth_c2",
        "basin_transition_branch_target_growth_field34_cc_c2_v0",
        "branch_target_growth_states.csv",
    ),
    (
        "side_route_expansion_c0",
        "basin_transition_side_route_expansion_field34_cc_c0_v0",
        "target_elbow_polish_states.csv",
    ),
    (
        "target_elbow_c0_top10",
        "basin_transition_target_elbow_polish_field34_cc_c0_top10_v0",
        "target_elbow_polish_states.csv",
    ),
    (
        "target_elbow_c0_backfill",
        "basin_transition_target_elbow_polish_field34_cc_c0_backfill_v0",
        "target_elbow_polish_states.csv",
    ),
    (
        "target_elbow_c0_escalate",
        "basin_transition_target_elbow_polish_field34_cc_c0_escalate_v0",
        "target_elbow_polish_states.csv",
    ),
    (
        "target_elbow_c2_top10",
        "basin_transition_target_elbow_polish_field34_cc_c2_top10_v0",
        "target_elbow_polish_states.csv",
    ),
    (
        "target_elbow_c2_backfill",
        "basin_transition_target_elbow_polish_field34_cc_c2_backfill_v0",
        "target_elbow_polish_states.csv",
    ),
    (
        "target_elbow_c2_escalate",
        "basin_transition_target_elbow_polish_field34_cc_c2_escalate_v0",
        "target_elbow_polish_states.csv",
    ),
    (
        "transition_search_reachability",
        "basin_transition_search_field34_cc_reachability_v0",
        "transition_search_states.csv",
    ),
    (
        "transition_search_v0",
        "basin_transition_search_field34_cc_v0",
        "transition_search_states.csv",
    ),
)

CANDIDATE_ROWS_FILENAME = "tunneling_path_candidate_rows.csv"
SELECTED_ROWS_FILENAME = "tunneling_path_selected_rows.csv"
SUMMARY_ROWS_FILENAME = "tunneling_path_summary_rows.csv"
TRACE_ROWS_FILENAME = "tunneling_path_trace_rows.csv"
DESIGN_ROWS_FILENAME = "tunneling_operator_design_rows.csv"
CONFIG_FILENAME = "tunneling_path_rank_config.json"
SUMMARY_FILENAME = "tunneling_path_rank_summary.json"
REPORT_FILENAME = "tunneling_path_rank_report.md"


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


def _load_artifact(
    *,
    artifact_label: str,
    artifact_dir: Path,
    state_filename: str,
    support_gate: float,
    progress_margin: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_path = artifact_dir / state_filename
    states = pd.read_csv(state_path)
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


def _load_all_artifacts(
    *,
    combined_dir: Path,
    support_gate: float,
    progress_margin: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[dict[str, str]]]:
    rows: list[pd.DataFrame] = []
    states_by_artifact: dict[str, pd.DataFrame] = {}
    loaded: list[dict[str, str]] = []
    for artifact_label, dirname, state_filename in DEFAULT_ARTIFACTS:
        artifact_dir = combined_dir / dirname
        state_path = artifact_dir / state_filename
        if not state_path.exists():
            continue
        path_rows, states = _load_artifact(
            artifact_label=artifact_label,
            artifact_dir=artifact_dir,
            state_filename=state_filename,
            support_gate=support_gate,
            progress_margin=progress_margin,
        )
        rows.append(path_rows)
        states_by_artifact[artifact_label] = states
        loaded.append(
            {
                "artifact_label": artifact_label,
                "artifact_dir": str(artifact_dir),
                "state_filename": state_filename,
                "state_rows": str(len(states)),
                "path_rows": str(len(path_rows)),
            }
        )
    if not rows:
        return pd.DataFrame(), states_by_artifact, loaded
    return pd.concat(rows, ignore_index=True), states_by_artifact, loaded


def _trace_selected(
    selected_rows: pd.DataFrame,
    *,
    states_by_artifact: dict[str, pd.DataFrame],
    support_gate: float,
    progress_margin: float,
) -> pd.DataFrame:
    traces: list[pd.DataFrame] = []
    for artifact_label, group in selected_rows.groupby("artifact_label", sort=True):
        states = states_by_artifact.get(str(artifact_label))
        if states is None:
            continue
        trace = trace_tunneling_path_states(
            group,
            state_rows=states,
            support_gate=support_gate,
            progress_margin=progress_margin,
        )
        if not trace.empty:
            category_map = dict(
                zip(
                    group["path_final_state_id"].astype(str),
                    group["tunnel_operator_category"].astype(str),
                    strict=False,
                )
            )
            trace["tunnel_operator_category"] = trace["path_final_state_id"].map(
                category_map
            )
            traces.append(trace)
    return pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()


def _design_row(row: pd.Series) -> dict[str, Any]:
    category = str(row.get("tunnel_operator_category", ""))
    if category == TUNNEL_OPERATOR_RECOVERABLE_SEED:
        hook = "use_as_tunnel_seed"
        mechanism = "Does the prefix plus bounded polish reproduce the fast wall-exit?"
        next_test = "material_quality_and_seed_control_retest"
    elif category == TUNNEL_OPERATOR_RECOVERY_TARGET:
        hook = "post_gate_recovery_move"
        mechanism = "Which local context or label repair turns gate crossing into QF recovery?"
        next_test = "recovery_context_probe_after_gate"
    else:
        hook = "entrance_probe"
        mechanism = "Can this below-gate prefix be extended into a wall entry without losing support direction?"
        next_test = "bounded_extension_search"
    return {
        "artifact_label": row.get("artifact_label", ""),
        "operator_hook": hook,
        "operator_category": category,
        "path_final_state_id": row.get("path_final_state_id", ""),
        "path_prefix_rank": row.get("path_prefix_rank", math.nan),
        "path_selection_policy": row.get("path_selection_policy", ""),
        "path_policy": row.get("path_policy", ""),
        "action_hint": row.get("tunnel_operator_action_hint", ""),
        "mechanism_question": mechanism,
        "next_test": next_test,
        "score": row.get("tunnel_operator_score", math.nan),
        "q_wall": row.get("path_q_wall", math.nan),
        "debt_area_step": row.get("path_q_debt_area_step", math.nan),
        "delta_q": row.get("path_final_delta_q_vs_start", math.nan),
        "support": row.get("path_final_support_distance_to_vanilla", math.nan),
        "target_progress": row.get("path_final_target_progress_from_vanilla", math.nan),
        "mutable_nodes": row.get("path_final_mutable_node_count", math.nan),
    }


def _build_design_rows(selected_rows: pd.DataFrame) -> pd.DataFrame:
    if selected_rows.empty:
        return pd.DataFrame()
    design_rows = [_design_row(row) for _, row in selected_rows.iterrows()]
    return pd.DataFrame(design_rows)


def _summary_dict(
    *,
    output_dir: Path,
    loaded_artifacts: list[dict[str, str]],
    candidate_rows: pd.DataFrame,
    selected_rows: pd.DataFrame,
    support_gate: float,
    progress_margin: float,
) -> dict[str, Any]:
    summary = {
        "schema": "leiden_basin_tunneling_path_rank.v0",
        "output_dir": str(output_dir),
        "loaded_artifacts": int(len(loaded_artifacts)),
        "candidate_rows": int(len(candidate_rows)),
        "selected_rows": int(len(selected_rows)),
        "support_gate": float(support_gate),
        "progress_margin": float(progress_margin),
    }
    for category, key in (
        (TUNNEL_OPERATOR_RECOVERABLE_SEED, "recoverable_seed_rows"),
        (TUNNEL_OPERATOR_RECOVERY_TARGET, "recovery_target_rows"),
    ):
        summary[key] = int(
            candidate_rows.get(
                "tunnel_operator_category",
                pd.Series("", index=candidate_rows.index),
            )
            .astype(str)
            .eq(category)
            .sum()
        )
    top = selected_rows[
        selected_rows["tunnel_operator_category"].eq(TUNNEL_OPERATOR_RECOVERABLE_SEED)
    ]
    if not top.empty:
        row = top.iloc[0]
        summary.update(
            {
                "top_recoverable_state_id": row["path_final_state_id"],
                "top_recoverable_artifact": row["artifact_label"],
                "top_recoverable_score": float(row["tunnel_operator_score"]),
                "top_recoverable_delta_q": float(row["path_final_delta_q_vs_start"]),
                "top_recoverable_support": float(
                    row["path_final_support_distance_to_vanilla"]
                ),
                "top_recoverable_progress": float(
                    row["path_final_target_progress_from_vanilla"]
                ),
                "top_recoverable_debt_area_step": float(
                    row["path_q_debt_area_step"]
                ),
            }
        )
    return summary


def write_report(
    path: Path,
    *,
    summary: dict[str, Any],
    loaded_artifacts: list[dict[str, str]],
    summary_rows: pd.DataFrame,
    selected_rows: pd.DataFrame,
    trace_rows: pd.DataFrame,
    design_rows: pd.DataFrame,
) -> None:
    lines = [
        "# Tunneling Path Rank",
        "",
        "This artifact ranks existing transition-search paths as operator seeds, recovery targets, or entrance probes.",
        "It is still diagnostic: rows are design inputs, not accepted production policies.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "output_dir",
        "loaded_artifacts",
        "candidate_rows",
        "selected_rows",
        "recoverable_seed_rows",
        "recovery_target_rows",
        "top_recoverable_artifact",
        "top_recoverable_score",
        "top_recoverable_delta_q",
        "top_recoverable_support",
        "top_recoverable_progress",
        "top_recoverable_debt_area_step",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")

    lines.extend(["", "## Loaded Artifacts", ""])
    lines.extend(_markdown_table(pd.DataFrame(loaded_artifacts), max_rows=20))

    lines.extend(["", "## Category Summary", ""])
    summary_cols = [
        "artifact_label",
        "tunnel_route_label",
        "rows",
        "candidate_directed_rows",
        "q_recovered_rows",
        "q_wall_median",
        "debt_area_step_median",
        "final_delta_q_max",
        "support_max",
        "target_progress_max",
        "recovered_shortcut_score_max",
    ]
    lines.extend(
        _markdown_table(
            summary_rows[[column for column in summary_cols if column in summary_rows]],
            max_rows=80,
        )
    )

    lines.extend(["", "## Selected Operator Candidates", ""])
    selected_cols = [
        "artifact_label",
        "tunnel_operator_category",
        "tunnel_route_label",
        "path_prefix_rank",
        "path_selection_policy",
        "path_policy",
        "path_q_wall",
        "path_q_debt_area_step",
        "path_final_delta_q_vs_start",
        "path_final_support_distance_to_vanilla",
        "path_final_target_progress_from_vanilla",
        "tunnel_operator_score",
        "tunnel_operator_action_hint",
        "path_final_state_id",
    ]
    lines.extend(
        _markdown_table(
            selected_rows[[column for column in selected_cols if column in selected_rows]],
            max_rows=60,
        )
    )

    lines.extend(["", "## Operator Design Rows", ""])
    design_cols = [
        "operator_hook",
        "operator_category",
        "artifact_label",
        "path_prefix_rank",
        "path_selection_policy",
        "mechanism_question",
        "next_test",
        "score",
        "q_wall",
        "debt_area_step",
        "delta_q",
        "support",
        "target_progress",
    ]
    lines.extend(
        _markdown_table(
            design_rows[[column for column in design_cols if column in design_rows]],
            max_rows=60,
        )
    )

    lines.extend(["", "## Selected Trace Rows", ""])
    trace_cols = [
        "artifact_label",
        "tunnel_operator_category",
        "trace_step_index",
        "trace_phase",
        "trace_action_type",
        "trace_q_debt",
        "trace_cumulative_q_debt_area_step",
        "trace_delta_q_vs_start",
        "trace_support_distance_to_vanilla",
        "trace_target_progress_from_vanilla",
        "trace_first_candidate_directed_step",
        "trace_first_q_recovered_step",
        "path_final_state_id",
    ]
    lines.extend(
        _markdown_table(
            trace_rows[[column for column in trace_cols if column in trace_rows]],
            max_rows=120,
        )
    )

    lines.extend(
        [
            "",
            "## Algorithm Design Reading",
            "",
            "- Recoverable seeds are the first candidates for a Dongdaemun tunneling operator: replay the entrance prefix, run bounded polish, then test tail growth.",
            "- Recovery targets are not failures to discard. They identify support-gate states where a separate recovery move is needed.",
            "- Entrance probes are below-gate prefixes that may become useful if a later wall-entry step can be found.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_rank(
    *,
    combined_dir: Path,
    output_dir: Path,
    support_gate: float,
    progress_margin: float,
    max_rows_per_category: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, states_by_artifact, loaded_artifacts = _load_all_artifacts(
        combined_dir=combined_dir,
        support_gate=support_gate,
        progress_margin=progress_margin,
    )
    ranked = rank_tunneling_operator_candidates(
        rows,
        support_gate=support_gate,
        progress_margin=progress_margin,
    )
    ranked_for_selection = ranked.drop_duplicates(
        subset=["tunnel_operator_category", "path_final_state_id"],
        keep="first",
    )
    selected = select_tunneling_operator_candidates(
        ranked_for_selection,
        max_rows_per_category=max_rows_per_category,
    )
    selected = selected[
        selected["tunnel_operator_category"].ne(TUNNEL_OPERATOR_BACKGROUND)
    ].copy()
    trace_rows = _trace_selected(
        selected,
        states_by_artifact=states_by_artifact,
        support_gate=support_gate,
        progress_margin=progress_margin,
    )
    summary_rows = summarize_tunneling_evidence_rows(ranked)
    design_rows = _build_design_rows(selected)

    ranked.to_csv(output_dir / CANDIDATE_ROWS_FILENAME, index=False)
    selected.to_csv(output_dir / SELECTED_ROWS_FILENAME, index=False)
    summary_rows.to_csv(output_dir / SUMMARY_ROWS_FILENAME, index=False)
    trace_rows.to_csv(output_dir / TRACE_ROWS_FILENAME, index=False)
    design_rows.to_csv(output_dir / DESIGN_ROWS_FILENAME, index=False)

    config = {
        "combined_dir": str(combined_dir),
        "output_dir": str(output_dir),
        "support_gate": float(support_gate),
        "progress_margin": float(progress_margin),
        "max_rows_per_category": int(max_rows_per_category),
        "loaded_artifacts": loaded_artifacts,
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = _summary_dict(
        output_dir=output_dir,
        loaded_artifacts=loaded_artifacts,
        candidate_rows=ranked,
        selected_rows=selected,
        support_gate=support_gate,
        progress_margin=progress_margin,
    )
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(
        output_dir / REPORT_FILENAME,
        summary=summary,
        loaded_artifacts=loaded_artifacts,
        summary_rows=summary_rows,
        selected_rows=selected,
        trace_rows=trace_rows,
        design_rows=design_rows,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-dir", type=Path, default=COMBINED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--support-gate", type=float, default=0.05)
    parser.add_argument("--progress-margin", type=float, default=0.005)
    parser.add_argument("--max-rows-per-category", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_rank(
        combined_dir=args.combined_dir,
        output_dir=args.output_dir,
        support_gate=args.support_gate,
        progress_margin=args.progress_margin,
        max_rows_per_category=args.max_rows_per_category,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
