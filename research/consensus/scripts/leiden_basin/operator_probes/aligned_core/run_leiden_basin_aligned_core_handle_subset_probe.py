#!/usr/bin/env python3
"""Probe which direct aligned-core handles are sufficient.

This diagnostic keeps the context closed and exhaustively tests candidate-label
transplants over subsets of the direct target handles discovered by the
aligned-core frontier.
"""

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


import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

from run_leiden_basin_aligned_core_boundary_operator_probe import (  # noqa: E402
    DEFAULT_FRONTIER_ROWS,
)
from run_leiden_basin_attachment_margin_cross_prefix_probe import (  # noqa: E402
    DEFAULT_SOURCE_RECOVERY_POLICY,
    SUMMARY_ROWS_FILENAME as ATTACHMENT_SUMMARY_ROWS_FILENAME,
)
from run_leiden_basin_attachment_margin_joint_bundle_probe import (  # noqa: E402
    ACTION_JOINT_CANDIDATE_TRANSPLANT,
    DEFAULT_ATTACHMENT_DIR,
)
from run_leiden_basin_attachment_margin_stage2_recovery import (  # noqa: E402
    DEFAULT_CONTROL_DIR,
    _control_context,
    _load_control_summary,
    _polished_child_with_reference,
    _rebuild_source_state,
)
from sciscape.clustering.leiden_basin_profile import (  # noqa: E402
    changed_support_nodes,
    compact_membership,
    parse_node_ids,
)
from sciscape.clustering.leiden_basin_search import (  # noqa: E402
    TransitionAction,
    build_aligned_core_handle_subset_plan_rows,
    intersect_sorted_u32,
    node_csv,
    select_aligned_core_boundary_nodes,
    setdiff_sorted_u32,
    transplant_action_nodes,
    unique_sorted_u32,
)
from sciscape.clustering.leiden_basin_transition_explain import (  # noqa: E402
    membership_change_summary,
)
from search_leiden_basin_transitions import _evaluate_state  # noqa: E402

COMBINED_DIR = DEFAULT_ATTACHMENT_DIR.parent
DEFAULT_OUTPUT_DIR = COMBINED_DIR / (
    "basin_transition_aligned_core_handle_subset_field34_cc_c0_p8_v0"
)

PLAN_ROWS_FILENAME = "aligned_core_handle_subset_plan_rows.csv"
ROWS_FILENAME = "aligned_core_handle_subset_rows.csv"
SUMMARY_ROWS_FILENAME = "aligned_core_handle_subset_summary_rows.csv"
CONFIG_FILENAME = "aligned_core_handle_subset_config.json"
SUMMARY_FILENAME = "aligned_core_handle_subset_summary.json"
REPORT_FILENAME = "aligned_core_handle_subset_report.md"

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
                values.append("" if pd.isna(value) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines

def _row_verdict(row: pd.Series, *, quality_tolerance: float) -> str:
    if bool(row["recovers_required_aligned_core"]) and float(
        row["quality_gap_vs_full_handle_set"]
    ) >= -float(quality_tolerance):
        return "sufficient_full_core_quality_match"
    if bool(row["recovers_required_aligned_core"]):
        return "recovers_full_core_quality_lag"
    if float(row["operator_delta_q_gain_vs_source"]) > 0.0:
        return "partial_core_quality_gain"
    return "partial_core_no_quality_gain"

def _add_full_set_comparisons(
    rows: pd.DataFrame,
    *,
    required_nodes: np.ndarray,
    quality_tolerance: float,
) -> pd.DataFrame:
    if rows.empty:
        return rows
    out = rows.copy()
    handle_count_col = (
        "direct_handle_count" if "direct_handle_count" in out.columns else "target_node_count"
    )
    full = out[out["subset_size"].astype(int).eq(int(out[handle_count_col].max()))]
    if full.empty:
        full_quality = float(out["state_quality"].max())
    else:
        full_quality = float(full.sort_values(["state_quality"], ascending=False).iloc[0]["state_quality"])
    required = unique_sorted_u32(required_nodes)
    hit_counts: list[int] = []
    unexpected_counts: list[int] = []
    for ids in out["operator_final_aligned_changed_support_node_ids"]:
        aligned = unique_sorted_u32(parse_node_ids(ids))
        hit_counts.append(int(intersect_sorted_u32(aligned, required).size))
        unexpected_counts.append(int(setdiff_sorted_u32(aligned, required).size))
    out["required_aligned_core_node_ids"] = node_csv(required)
    out["required_aligned_core_count"] = int(required.size)
    out["required_aligned_core_hit_count"] = hit_counts
    out["required_aligned_core_hit_fraction"] = (
        out["required_aligned_core_hit_count"].astype(float)
        / float(max(1, int(required.size)))
    )
    out["unexpected_aligned_changed_count"] = unexpected_counts
    out["recovers_required_aligned_core"] = out["required_aligned_core_hit_count"].astype(
        int
    ).eq(int(required.size))
    out["quality_gap_vs_full_handle_set"] = out["state_quality"].astype(float) - full_quality
    out["handle_subset_verdict"] = [
        _row_verdict(row, quality_tolerance=quality_tolerance)
        for _, row in out.iterrows()
    ]
    return out

def _evaluate_subsets(
    *,
    plans: pd.DataFrame,
    config: dict[str, Any],
    case_ctx: dict[str, Any],
    source_state: Any,
    source_row: dict[str, Any],
    meta: dict[str, Any],
    control_ctx: dict[str, Any],
    recovery_seed: int,
    polish_seed_offset: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, plan in plans.iterrows():
        subset_nodes = unique_sorted_u32(parse_node_ids(plan["subset_node_ids"]))
        action = TransitionAction(
            action_type=ACTION_JOINT_CANDIDATE_TRANSPLANT,
            action_params=(
                f"subset_size={int(subset_nodes.size)};"
                f"subset_node_ids={node_csv(subset_nodes)}"
            ),
            context_nodes=np.asarray([], dtype=np.uint32),
            action_nodes=subset_nodes,
        )
        pre_membership = transplant_action_nodes(
            membership=source_state.membership,
            donor_membership=case_ctx["candidate"].recreated.membership,
            action_nodes=subset_nodes,
            reference_nodes=np.asarray([], dtype=np.uint32),
        )
        pre_membership = compact_membership(pre_membership)
        pre_change = membership_change_summary(
            reference_membership=source_state.membership,
            membership=pre_membership,
            sketch_nodes=case_ctx["sketch_nodes"],
        )
        pre_quality = float(
            case_ctx["graph"].cpm_quality(
                pre_membership,
                resolution=float(config.get("resolution", 0.01)),
            )
        )
        child = _polished_child_with_reference(
            parent=source_state,
            action=action,
            graph=case_ctx["graph"],
            donor_membership=case_ctx["candidate"].recreated.membership,
            reference_nodes=np.asarray([], dtype=np.uint32),
            resolution=float(config.get("resolution", 0.01)),
            seed=int(recovery_seed) + int(polish_seed_offset),
            n_iterations=int(config.get("recovery_polish_iterations", 6)),
            randomness=float(config.get("randomness", 0.01)),
            child_index=int(plan["plan_rank"]),
        )
        final_change = membership_change_summary(
            reference_membership=source_state.membership,
            membership=child.membership,
            sketch_nodes=case_ctx["sketch_nodes"],
        )
        final_aligned = changed_support_nodes(source_state.membership, child.membership)
        row = _evaluate_state(
            state=child,
            baseline_membership=case_ctx["baseline"].membership,
            candidate_membership=case_ctx["candidate"].recreated.membership,
            vanilla_membership=case_ctx["vanilla"].membership,
            sketch_nodes=case_ctx["sketch_nodes"],
            start_quality=case_ctx["vanilla"].quality,
            candidate_quality=case_ctx["candidate"].recreated.quality,
            vanilla_quality=case_ctx["vanilla"].quality,
            vanilla_support_distance_to_candidate=case_ctx[
                "vanilla_support_distance_to_candidate"
            ],
            context={
                **meta,
                "path_policy": "aligned_core_handle_subset",
                "source_state_id": source_state.state_id,
                "plan_rank": int(plan["plan_rank"]),
                "plan_kind": str(plan["plan_kind"]),
                "subset_size": int(plan["subset_size"]),
                "move_kind": "candidate_bundle_transplant",
            },
            parent_row=source_row,
            min_support_shift_from_vanilla=0.01,
            min_material_q_gain=0.01,
        )
        row.update(
            {
                "subset_node_ids": node_csv(subset_nodes),
                "direct_handle_count": int(plan["target_node_count"]),
                "full_handle_node_ids": str(plan["full_target_node_ids"]),
                "bundle_node_count": int(subset_nodes.size),
                "bundle_node_ids": node_csv(subset_nodes),
                "operator_pre_polish_exact_label_delta_count": int(
                    pre_change["exact_changed_node_count"]
                ),
                "operator_pre_polish_aligned_changed_support_node_count": int(
                    pre_change["aligned_changed_node_count"]
                ),
                "operator_pre_polish_exact_only_label_delta_count": int(
                    pre_change["exact_only_changed_node_count"]
                ),
                "operator_pre_polish_delta_q_gain_vs_source": pre_quality
                - float(source_row["state_quality"]),
                "operator_final_exact_label_delta_count": int(
                    final_change["exact_changed_node_count"]
                ),
                "operator_final_aligned_changed_support_node_count": int(
                    final_change["aligned_changed_node_count"]
                ),
                "operator_final_exact_only_label_delta_count": int(
                    final_change["exact_only_changed_node_count"]
                ),
                "operator_final_endpoint_distance_to_source": float(
                    final_change["endpoint_distance"]
                ),
                "operator_final_aligned_changed_support_node_ids": node_csv(final_aligned),
                "operator_delta_q_gain_vs_source": float(row["state_quality"])
                - float(source_row["state_quality"]),
                "operator_target_progress_gain_vs_source": float(
                    row["state_target_progress_from_vanilla"]
                )
                - float(source_row["state_target_progress_from_vanilla"]),
                **{
                    key: value
                    for key, value in control_ctx.items()
                    if key.startswith("best_")
                },
            }
        )
        if control_ctx:
            row["quality_minus_best_control"] = float(row["state_quality"]) - float(
                control_ctx["best_quality_control_quality"]
            )
            row["quality_minus_best_same_randomness_control"] = float(
                row["state_quality"]
            ) - float(control_ctx["best_same_randomness_control_quality"])
        row["quality_gain_per_bundle_node"] = float(
            row["operator_delta_q_gain_vs_source"]
        ) / float(max(1, int(subset_nodes.size)))
        rows.append(row)
    return pd.DataFrame(rows)

def _summary_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    sufficient = rows[
        rows["handle_subset_verdict"].astype(str).eq("sufficient_full_core_quality_match")
    ].copy()
    best = rows.sort_values(
        ["state_quality", "required_aligned_core_hit_fraction", "subset_size"],
        ascending=[False, False, True],
    ).iloc[0]
    minimal = (
        sufficient.sort_values(["subset_size", "state_quality"], ascending=[True, False]).iloc[0]
        if not sufficient.empty
        else best
    )
    return pd.DataFrame(
        [
            {
                "row_count": int(len(rows)),
                "direct_handle_count": int(rows["direct_handle_count"].max()),
                "sufficient_row_count": int(len(sufficient)),
                "minimal_sufficient_subset_size": (
                    int(minimal["subset_size"]) if not sufficient.empty else math.nan
                ),
                "minimal_sufficient_subset_node_ids": (
                    str(minimal["subset_node_ids"]) if not sufficient.empty else ""
                ),
                "minimal_sufficient_delta_q_gain_vs_source": (
                    float(minimal["operator_delta_q_gain_vs_source"])
                    if not sufficient.empty
                    else math.nan
                ),
                "best_subset_size": int(best["subset_size"]),
                "best_subset_node_ids": str(best["subset_node_ids"]),
                "best_delta_q_gain_vs_source": float(
                    best["operator_delta_q_gain_vs_source"]
                ),
                "best_quality_gap_vs_full_handle_set": float(
                    best["quality_gap_vs_full_handle_set"]
                ),
                "verdict_counts": json.dumps(
                    rows["handle_subset_verdict"].value_counts().to_dict(),
                    sort_keys=True,
                ),
            }
        ]
    )

def _write_report(
    path: Path,
    *,
    rows: pd.DataFrame,
    summary_rows: pd.DataFrame,
) -> None:
    summary_cols = [
        "row_count",
        "direct_handle_count",
        "sufficient_row_count",
        "minimal_sufficient_subset_size",
        "minimal_sufficient_subset_node_ids",
        "minimal_sufficient_delta_q_gain_vs_source",
        "best_subset_size",
        "best_subset_node_ids",
        "best_delta_q_gain_vs_source",
        "best_quality_gap_vs_full_handle_set",
        "verdict_counts",
    ]
    row_cols = [
        "subset_size",
        "subset_node_ids",
        "required_aligned_core_hit_count",
        "required_aligned_core_hit_fraction",
        "recovers_required_aligned_core",
        "operator_final_aligned_changed_support_node_count",
        "operator_delta_q_gain_vs_source",
        "quality_gap_vs_full_handle_set",
        "state_delta_q_vs_vanilla",
        "state_target_progress_from_vanilla",
        "quality_minus_best_same_randomness_control",
        "quality_gain_per_bundle_node",
        "handle_subset_verdict",
        "operator_final_aligned_changed_support_node_ids",
    ]
    lines = [
        "# Aligned-Core Handle Subset Probe",
        "",
        "This diagnostic keeps context closed and asks which direct candidate-label",
        "transplant handles are sufficient to reproduce the required aligned core.",
        "",
        "## Summary",
        "",
    ]
    lines.extend(
        _markdown_table(
            summary_rows[[c for c in summary_cols if c in summary_rows]],
            max_rows=10,
        )
    )
    lines.extend(["", "## Minimal/Best Rows", ""])
    display = rows.sort_values(
        [
            "handle_subset_verdict",
            "subset_size",
            "quality_gap_vs_full_handle_set",
            "operator_delta_q_gain_vs_source",
        ],
        ascending=[False, True, False, False],
    )
    lines.extend(_markdown_table(display[[c for c in row_cols if c in display]], max_rows=80))
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "- This is a sufficiency diagnostic over one p8 source, not a default policy.",
            "- A smaller subset is interesting only if it recovers the required aligned core and matches the full-handle quality within tolerance.",
            "- Quality gain is reported with bundle size so tiny positive gains are not overread.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_probe(
    *,
    frontier_rows_path: Path,
    attachment_dir: Path,
    control_dir: Path | None,
    output_dir: Path,
    source_case: str,
    source_recovery_policy: str,
    recovery_seed: int,
    polish_seed_offset: int,
    min_target_change_count: int,
    min_boundary_change_count: int,
    min_subset_size: int,
    max_subset_size: int | None,
    quality_tolerance: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frontier_rows = pd.read_csv(frontier_rows_path)
    selection = select_aligned_core_boundary_nodes(
        frontier_rows,
        min_target_change_count=min_target_change_count,
        min_boundary_change_count=min_boundary_change_count,
        max_context_core_nodes=0,
    )
    if selection.target_nodes.size == 0:
        raise ValueError("No target core nodes selected from frontier rows")
    required_nodes = unique_sorted_u32(
        np.concatenate([selection.target_nodes, selection.boundary_core_nodes])
    )
    plans = build_aligned_core_handle_subset_plan_rows(
        target_nodes=selection.target_nodes,
        min_subset_size=min_subset_size,
        max_subset_size=max_subset_size,
    )
    stage1_rows = pd.read_csv(attachment_dir / ATTACHMENT_SUMMARY_ROWS_FILENAME)
    stage1 = stage1_rows[stage1_rows["source_case"].astype(str).eq(str(source_case))]
    if stage1.empty:
        raise ValueError(f"No attachment summary row for source_case={source_case}")
    config, case_ctx, source_state, source_row, effective_seed, meta = _rebuild_source_state(
        source_move_dir=Path(str(stage1.iloc[0]["source_move_dir"])),
        source_case=source_case,
        source_recovery_policy=source_recovery_policy,
        requested_recovery_seed=recovery_seed,
    )
    rows = _evaluate_subsets(
        plans=plans,
        config=config,
        case_ctx=case_ctx,
        source_state=source_state,
        source_row=source_row,
        meta=meta,
        control_ctx=_control_context(_load_control_summary(control_dir), source_case),
        recovery_seed=effective_seed,
        polish_seed_offset=polish_seed_offset,
    )
    rows = _add_full_set_comparisons(
        rows,
        required_nodes=required_nodes,
        quality_tolerance=quality_tolerance,
    )
    summary_rows = _summary_rows(rows)
    plans.to_csv(output_dir / PLAN_ROWS_FILENAME, index=False)
    rows.to_csv(output_dir / ROWS_FILENAME, index=False)
    summary_rows.to_csv(output_dir / SUMMARY_ROWS_FILENAME, index=False)
    run_config = {
        "frontier_rows_path": str(frontier_rows_path),
        "attachment_dir": str(attachment_dir),
        "control_dir": str(control_dir) if control_dir is not None else "",
        "output_dir": str(output_dir),
        "source_case": source_case,
        "source_recovery_policy": source_recovery_policy,
        "requested_recovery_seed": int(recovery_seed),
        "effective_recovery_seed": int(effective_seed),
        "polish_seed_offset": int(polish_seed_offset),
        "target_node_ids": node_csv(selection.target_nodes),
        "required_aligned_core_node_ids": node_csv(required_nodes),
        "min_subset_size": int(min_subset_size),
        "max_subset_size": (
            int(max_subset_size) if max_subset_size is not None else int(selection.target_nodes.size)
        ),
        "quality_tolerance": float(quality_tolerance),
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verdict_counts = rows["handle_subset_verdict"].value_counts().to_dict()
    payload = {
        "schema": "leiden_basin_aligned_core_handle_subset_probe.v0",
        "output_dir": str(output_dir),
        "plan_count": int(len(plans)),
        "row_count": int(len(rows)),
        "summary_row_count": int(len(summary_rows)),
        "verdict_counts": {str(key): int(value) for key, value in verdict_counts.items()},
        "minimal_sufficient_subset_size": (
            None
            if summary_rows.empty
            or pd.isna(summary_rows.iloc[0]["minimal_sufficient_subset_size"])
            else int(summary_rows.iloc[0]["minimal_sufficient_subset_size"])
        ),
        "paths": {
            "plan_rows": str(output_dir / PLAN_ROWS_FILENAME),
            "rows": str(output_dir / ROWS_FILENAME),
            "summary_rows": str(output_dir / SUMMARY_ROWS_FILENAME),
            "report": str(output_dir / REPORT_FILENAME),
        },
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / REPORT_FILENAME, rows=rows, summary_rows=summary_rows)
    return payload

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-rows", type=Path, default=DEFAULT_FRONTIER_ROWS)
    parser.add_argument("--attachment-dir", type=Path, default=DEFAULT_ATTACHMENT_DIR)
    parser.add_argument("--control-dir", type=Path, default=DEFAULT_CONTROL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-case", default="p8_fullctx")
    parser.add_argument("--source-recovery-policy", default=DEFAULT_SOURCE_RECOVERY_POLICY)
    parser.add_argument("--recovery-seed", type=int, default=0)
    parser.add_argument("--polish-seed-offset", type=int, default=2000)
    parser.add_argument("--min-target-change-count", type=int, default=5)
    parser.add_argument("--min-boundary-change-count", type=int, default=5)
    parser.add_argument("--min-subset-size", type=int, default=1)
    parser.add_argument("--max-subset-size", type=int, default=0)
    parser.add_argument("--quality-tolerance", type=float, default=1e-9)
    return parser

def main() -> None:
    args = build_parser().parse_args()
    result = run_probe(
        frontier_rows_path=args.frontier_rows,
        attachment_dir=args.attachment_dir,
        control_dir=args.control_dir,
        output_dir=args.output_dir,
        source_case=args.source_case,
        source_recovery_policy=args.source_recovery_policy,
        recovery_seed=args.recovery_seed,
        polish_seed_offset=args.polish_seed_offset,
        min_target_change_count=args.min_target_change_count,
        min_boundary_change_count=args.min_boundary_change_count,
        min_subset_size=args.min_subset_size,
        max_subset_size=(None if int(args.max_subset_size) <= 0 else int(args.max_subset_size)),
        quality_tolerance=args.quality_tolerance,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))

if __name__ == "__main__":
    main()
