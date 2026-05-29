#!/usr/bin/env python3
"""Probe compact aligned-core plus boundary-context operator plans.

This diagnostic follows the joint-bundle aligned-core frontier. It tests
whether the stable direct target handles need the non-target boundary core and
how quickly candidate-label context becomes broad again.
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

from run_leiden_basin_attachment_margin_cross_prefix_probe import (  # noqa: E402
    DEFAULT_SOURCE_RECOVERY_POLICY,
    SUMMARY_ROWS_FILENAME as ATTACHMENT_SUMMARY_ROWS_FILENAME,
)
from run_leiden_basin_attachment_margin_joint_bundle_probe import (  # noqa: E402
    ACTION_JOINT_CANDIDATE_TRANSPLANT,
    ACTION_JOINT_MUTABLE,
    DEFAULT_ATTACHMENT_DIR,
    _context_candidates,
    _source_context_nodes,
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
    build_aligned_core_boundary_plan_rows,
    node_csv,
    select_aligned_core_boundary_nodes,
    setdiff_sorted_u32,
    topk_by_pull,
    transplant_action_nodes,
    unique_sorted_u32,
    weighted_pull_to_nodes,
)
from sciscape.clustering.leiden_basin_transition_explain import (  # noqa: E402
    membership_change_summary,
)
from search_leiden_basin_transitions import (  # noqa: E402
    _evaluate_state,
    _polished_child,
)

COMBINED_DIR = DEFAULT_ATTACHMENT_DIR.parent
DEFAULT_FRONTIER_DIR = COMBINED_DIR / "joint_bundle_aligned_core_frontier_v0"
DEFAULT_FRONTIER_ROWS = DEFAULT_FRONTIER_DIR / "joint_bundle_aligned_core_node_frontier_rows.csv"
DEFAULT_OUTPUT_DIR = COMBINED_DIR / (
    "basin_transition_aligned_core_boundary_operator_field34_cc_c0_p8_v0"
)

PLAN_ROWS_FILENAME = "aligned_core_boundary_operator_plan_rows.csv"
ROWS_FILENAME = "aligned_core_boundary_operator_rows.csv"
SUMMARY_ROWS_FILENAME = "aligned_core_boundary_operator_summary_rows.csv"
CONFIG_FILENAME = "aligned_core_boundary_operator_config.json"
SUMMARY_FILENAME = "aligned_core_boundary_operator_summary.json"
REPORT_FILENAME = "aligned_core_boundary_operator_report.md"

def _parse_int_tuple(value: str, default: tuple[int, ...]) -> tuple[int, ...]:
    text = str(value).strip()
    if not text:
        return default
    parsed = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    return parsed or default

def _parse_move_kinds(value: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in str(value).split(",") if part.strip())
    return parts or ("joint_mutable", "candidate_bundle_transplant")

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

def _candidate_context_by_cap(
    *,
    caps: tuple[int, ...],
    target_nodes: np.ndarray,
    source_context_nodes: np.ndarray,
    source_state: Any,
    case_ctx: dict[str, Any],
) -> dict[int, np.ndarray]:
    if not caps:
        return {}
    candidates = _context_candidates(
        family="candidate_label",
        source_context_nodes=source_context_nodes,
        source_state=source_state,
        target_nodes=target_nodes,
        case_ctx=case_ctx,
    )
    if candidates.size == 0:
        return {}
    arrays = case_ctx["arrays"]
    pull = weighted_pull_to_nodes(
        src=np.asarray(arrays.src, dtype=np.uint32),
        dst=np.asarray(arrays.dst, dtype=np.uint32),
        weight=np.asarray(arrays.weight, dtype=np.float64),
        target_nodes=target_nodes,
        node_count=int(source_state.membership.size),
    )
    return {
        int(cap): topk_by_pull(
            candidate_nodes=candidates,
            pull_scores=pull,
            max_nodes=int(cap),
        )
        for cap in sorted(set(int(cap) for cap in caps if int(cap) > 0))
    }

def _operator_verdict(
    row: dict[str, Any],
    *,
    control_ctx: dict[str, Any],
    min_progress: float,
) -> str:
    directed = float(row["state_target_progress_from_vanilla"]) >= float(min_progress)
    if not directed:
        return "aligned_core_not_candidate_directed"
    if (
        control_ctx
        and float(row.get("quality_minus_best_same_randomness_control", -math.inf))
        >= 0.0
    ):
        return "aligned_core_beats_same_randomness_control"
    if control_ctx and float(row.get("quality_minus_best_control", -math.inf)) >= 0.0:
        return "aligned_core_beats_broad_control"
    if float(row["state_delta_q_vs_vanilla"]) >= 0.0:
        return "aligned_core_recovered_to_vanilla_quality"
    if float(row["operator_delta_q_gain_vs_source"]) > 0.0:
        return "aligned_core_directed_quality_lag"
    return "aligned_core_directed_quality_loss"

def _apply_target_only_deltas(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    out = rows.copy()
    out["quality_gain_vs_target_only_same_move_kind"] = math.nan
    out["aligned_gain_vs_target_only_same_move_kind"] = math.nan
    baselines = out[out["plan_kind"].astype(str).eq("target_core_only")]
    for _, base in baselines.iterrows():
        mask = out["move_kind"].astype(str).eq(str(base["move_kind"]))
        out.loc[mask, "quality_gain_vs_target_only_same_move_kind"] = (
            out.loc[mask, "state_quality"].astype(float) - float(base["state_quality"])
        )
        out.loc[mask, "aligned_gain_vs_target_only_same_move_kind"] = (
            out.loc[mask, "operator_final_aligned_changed_support_node_count"].astype(float)
            - float(base["operator_final_aligned_changed_support_node_count"])
        )
    return out

def _evaluate_plan_rows(
    *,
    plans: pd.DataFrame,
    config: dict[str, Any],
    case_ctx: dict[str, Any],
    source_state: Any,
    source_row: dict[str, Any],
    meta: dict[str, Any],
    control_ctx: dict[str, Any],
    recovery_seed: int,
    move_kinds: tuple[str, ...],
    min_progress: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    trial_index = 0
    source_mutable = unique_sorted_u32(source_state.mutable_nodes)
    for _, plan in plans.iterrows():
        target_nodes = unique_sorted_u32(parse_node_ids(plan["target_node_ids"]))
        context_nodes = unique_sorted_u32(parse_node_ids(plan["context_node_ids"]))
        bundle_nodes = unique_sorted_u32(parse_node_ids(plan["bundle_node_ids"]))
        boundary_nodes = unique_sorted_u32(parse_node_ids(plan["boundary_core_node_ids"]))
        for move_kind in move_kinds:
            trial_index += 1
            action = TransitionAction(
                action_type=(
                    ACTION_JOINT_MUTABLE
                    if move_kind == "joint_mutable"
                    else ACTION_JOINT_CANDIDATE_TRANSPLANT
                ),
                action_params=(
                    f"plan_kind={plan['plan_kind']};"
                    f"candidate_context_cap={int(plan['candidate_context_cap'])};"
                    f"move_kind={move_kind};"
                    f"target_k={int(target_nodes.size)};"
                    f"context_k={int(context_nodes.size)};"
                    f"bundle_k={int(bundle_nodes.size)}"
                ),
                context_nodes=(
                    bundle_nodes
                    if move_kind == "joint_mutable"
                    else np.asarray([], dtype=np.uint32)
                ),
                action_nodes=(
                    None
                    if move_kind == "joint_mutable"
                    else bundle_nodes
                ),
            )
            pre_membership = source_state.membership
            if move_kind == "candidate_bundle_transplant":
                pre_membership = transplant_action_nodes(
                    membership=source_state.membership,
                    donor_membership=case_ctx["candidate"].recreated.membership,
                    action_nodes=bundle_nodes,
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
            child_kwargs = {
                "parent": source_state,
                "action": action,
                "graph": case_ctx["graph"],
                "donor_membership": case_ctx["candidate"].recreated.membership,
                "resolution": float(config.get("resolution", 0.01)),
                "seed": int(recovery_seed) + int(trial_index) * 1000,
                "n_iterations": int(config.get("recovery_polish_iterations", 6)),
                "randomness": float(config.get("randomness", 0.01)),
                "child_index": trial_index,
            }
            child = (
                _polished_child_with_reference(
                    **child_kwargs,
                    reference_nodes=np.asarray([], dtype=np.uint32),
                )
                if move_kind == "candidate_bundle_transplant"
                else _polished_child(**child_kwargs)
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
                    "path_policy": "aligned_core_boundary_operator",
                    "source_state_id": source_state.state_id,
                    "plan_rank": int(plan["plan_rank"]),
                    "plan_kind": str(plan["plan_kind"]),
                    "move_kind": move_kind,
                },
                parent_row=source_row,
                min_support_shift_from_vanilla=0.01,
                min_material_q_gain=0.01,
            )
            boundary_overlap = np.intersect1d(
                boundary_nodes,
                source_mutable,
                assume_unique=True,
            ).astype(np.uint32, copy=False)
            plan_context_new = setdiff_sorted_u32(context_nodes, source_mutable)
            row.update(
                {
                    "target_node_count": int(target_nodes.size),
                    "target_node_ids": node_csv(target_nodes),
                    "boundary_core_node_count": int(boundary_nodes.size),
                    "boundary_core_node_ids": node_csv(boundary_nodes),
                    "boundary_core_source_mutable_overlap_count": int(
                        boundary_overlap.size
                    ),
                    "boundary_core_source_mutable_overlap_node_ids": node_csv(
                        boundary_overlap
                    ),
                    "candidate_context_cap": int(plan["candidate_context_cap"]),
                    "context_node_count": int(context_nodes.size),
                    "context_node_ids": node_csv(context_nodes),
                    "new_context_node_count": int(plan_context_new.size),
                    "new_context_node_ids": node_csv(plan_context_new),
                    "bundle_node_count": int(bundle_nodes.size),
                    "bundle_node_ids": node_csv(bundle_nodes),
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
            row["aligned_core_operator_verdict"] = _operator_verdict(
                row,
                control_ctx=control_ctx,
                min_progress=min_progress,
            )
            row["quality_gain_per_bundle_node"] = float(
                row["operator_delta_q_gain_vs_source"]
            ) / float(max(1, int(bundle_nodes.size)))
            rows.append(row)
    return _apply_target_only_deltas(pd.DataFrame(rows))

def _summary_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    summary: list[dict[str, Any]] = []
    for plan_kind, group in rows.groupby("plan_kind", sort=False):
        best = group.sort_values(
            [
                "state_quality",
                "state_target_progress_from_vanilla",
                "bundle_node_count",
            ],
            ascending=[False, False, True],
        ).iloc[0]
        summary.append(
            {
                "plan_kind": plan_kind,
                "row_count": int(len(group)),
                "best_move_kind": str(best["move_kind"]),
                "best_delta_q_gain_vs_source": float(
                    best["operator_delta_q_gain_vs_source"]
                ),
                "best_delta_q_vs_vanilla": float(best["state_delta_q_vs_vanilla"]),
                "best_target_progress": float(
                    best["state_target_progress_from_vanilla"]
                ),
                "best_final_aligned_changed": int(
                    best["operator_final_aligned_changed_support_node_count"]
                ),
                "best_bundle_node_count": int(best["bundle_node_count"]),
                "best_quality_gain_vs_target_only_same_move_kind": float(
                    best.get("quality_gain_vs_target_only_same_move_kind", math.nan)
                ),
                "best_verdict": str(best["aligned_core_operator_verdict"]),
            }
        )
    return pd.DataFrame(summary)

def _write_report(
    path: Path,
    *,
    plans: pd.DataFrame,
    rows: pd.DataFrame,
    summary_rows: pd.DataFrame,
) -> None:
    plan_cols = [
        "plan_rank",
        "plan_kind",
        "candidate_context_cap",
        "target_node_ids",
        "boundary_core_node_ids",
        "included_boundary_core_node_ids",
        "candidate_context_node_ids",
        "context_node_count",
        "context_node_ids",
        "bundle_node_count",
    ]
    summary_cols = [
        "plan_kind",
        "best_move_kind",
        "best_delta_q_gain_vs_source",
        "best_delta_q_vs_vanilla",
        "best_target_progress",
        "best_final_aligned_changed",
        "best_bundle_node_count",
        "best_quality_gain_vs_target_only_same_move_kind",
        "best_verdict",
    ]
    row_cols = [
        "plan_kind",
        "move_kind",
        "candidate_context_cap",
        "bundle_node_count",
        "new_context_node_count",
        "boundary_core_source_mutable_overlap_count",
        "operator_pre_polish_aligned_changed_support_node_count",
        "operator_final_aligned_changed_support_node_count",
        "operator_final_exact_only_label_delta_count",
        "operator_delta_q_gain_vs_source",
        "quality_gain_vs_target_only_same_move_kind",
        "state_delta_q_vs_vanilla",
        "state_target_progress_from_vanilla",
        "quality_minus_best_same_randomness_control",
        "quality_gain_per_bundle_node",
        "aligned_core_operator_verdict",
        "operator_final_aligned_changed_support_node_ids",
    ]
    lines = [
        "# Aligned-Core Boundary Operator Probe",
        "",
        "This diagnostic compares compact target handles, explicit boundary-core",
        "inclusion, and bounded candidate-label context under the same source",
        "state and polish path.",
        "",
        "## Plans",
        "",
    ]
    lines.extend(_markdown_table(plans[[c for c in plan_cols if c in plans]], max_rows=40))
    lines.extend(["", "## Summary", ""])
    lines.extend(
        _markdown_table(
            summary_rows[[c for c in summary_cols if c in summary_rows]],
            max_rows=40,
        )
    )
    lines.extend(["", "## Rows", ""])
    display = rows.sort_values(
        ["state_quality", "state_target_progress_from_vanilla", "bundle_node_count"],
        ascending=[False, False, True],
    )
    lines.extend(_markdown_table(display[[c for c in row_cols if c in display]], max_rows=80))
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "- Boundary-core nodes are priced separately from direct target handles.",
            "- Candidate-label context is a bounded expansion test, not a default policy.",
            "- Exact-only movement remains an implementation diagnostic; aligned movement, QF, target progress, and bundle cost are the basin-level signal.",
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
    move_kinds: tuple[str, ...],
    candidate_context_caps: tuple[int, ...],
    min_target_change_count: int,
    min_boundary_change_count: int,
    max_context_core_nodes: int,
    min_progress: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frontier_rows = pd.read_csv(frontier_rows_path)
    selection = select_aligned_core_boundary_nodes(
        frontier_rows,
        min_target_change_count=min_target_change_count,
        min_boundary_change_count=min_boundary_change_count,
        max_context_core_nodes=max_context_core_nodes,
    )
    if selection.target_nodes.size == 0:
        raise ValueError("No target core nodes selected from frontier rows")
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
    source_context = _source_context_nodes(
        source_move_dir=Path(str(stage1.iloc[0]["source_move_dir"])),
        source_recovery_policy=source_recovery_policy,
    )
    candidate_by_cap = _candidate_context_by_cap(
        caps=candidate_context_caps,
        target_nodes=selection.target_nodes,
        source_context_nodes=source_context,
        source_state=source_state,
        case_ctx=case_ctx,
    )
    plans = build_aligned_core_boundary_plan_rows(
        target_nodes=selection.target_nodes,
        boundary_core_nodes=selection.boundary_core_nodes,
        context_core_nodes=selection.context_core_nodes,
        candidate_context_by_cap=candidate_by_cap,
    )
    control_ctx = _control_context(_load_control_summary(control_dir), source_case)
    rows = _evaluate_plan_rows(
        plans=plans,
        config=config,
        case_ctx=case_ctx,
        source_state=source_state,
        source_row=source_row,
        meta=meta,
        control_ctx=control_ctx,
        recovery_seed=effective_seed,
        move_kinds=move_kinds,
        min_progress=min_progress,
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
        "move_kinds": list(move_kinds),
        "candidate_context_caps": list(candidate_context_caps),
        "min_target_change_count": int(min_target_change_count),
        "min_boundary_change_count": int(min_boundary_change_count),
        "max_context_core_nodes": int(max_context_core_nodes),
        "min_progress": float(min_progress),
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verdict_counts = (
        rows["aligned_core_operator_verdict"].value_counts().to_dict()
        if not rows.empty
        else {}
    )
    payload = {
        "schema": "leiden_basin_aligned_core_boundary_operator_probe.v0",
        "output_dir": str(output_dir),
        "plan_count": int(len(plans)),
        "row_count": int(len(rows)),
        "summary_row_count": int(len(summary_rows)),
        "verdict_counts": {str(key): int(value) for key, value in verdict_counts.items()},
        "best_delta_q_gain_vs_source": (
            float(rows["operator_delta_q_gain_vs_source"].max()) if not rows.empty else math.nan
        ),
        "best_quality_gain_vs_target_only_same_move_kind": (
            float(rows["quality_gain_vs_target_only_same_move_kind"].max())
            if not rows.empty
            else math.nan
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
    _write_report(
        output_dir / REPORT_FILENAME,
        plans=plans,
        rows=rows,
        summary_rows=summary_rows,
    )
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
    parser.add_argument(
        "--move-kinds",
        default="joint_mutable,candidate_bundle_transplant",
    )
    parser.add_argument("--candidate-context-caps", default="8,32")
    parser.add_argument("--min-target-change-count", type=int, default=5)
    parser.add_argument("--min-boundary-change-count", type=int, default=5)
    parser.add_argument("--max-context-core-nodes", type=int, default=3)
    parser.add_argument("--min-progress", type=float, default=0.0)
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
        move_kinds=_parse_move_kinds(args.move_kinds),
        candidate_context_caps=_parse_int_tuple(
            args.candidate_context_caps,
            default=(8, 32),
        ),
        min_target_change_count=args.min_target_change_count,
        min_boundary_change_count=args.min_boundary_change_count,
        max_context_core_nodes=args.max_context_core_nodes,
        min_progress=args.min_progress,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))

if __name__ == "__main__":
    main()
