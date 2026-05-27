#!/usr/bin/env python3
"""Probe joint target/context bundles before local polish.

This diagnostic follows the negative stage2 result: opening context after the
compact target move was a no-op.  Here, target nodes and companion context are
activated together from the source state before polish.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from probe_leiden_basin_post_gate_recovery_subsets import (  # noqa: E402
    SOURCE_MOVE_ROWS_FILENAME,
    _select_source_move,
)
from run_leiden_basin_attachment_margin_cross_prefix_probe import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_ATTACHMENT_DIR,
    DEFAULT_SOURCE_RECOVERY_POLICY,
    SCORE_ROWS_FILENAME as ATTACHMENT_SCORE_ROWS_FILENAME,
    SUMMARY_ROWS_FILENAME as ATTACHMENT_SUMMARY_ROWS_FILENAME,
    _load_json,
)
from run_leiden_basin_attachment_margin_stage2_recovery import (  # noqa: E402
    DEFAULT_CONTROL_DIR,
    _control_context,
    _load_control_summary,
    _polished_child_with_reference,
    _rebuild_source_state,
)
from sciscape.clustering.leiden_basin_profile import compact_membership  # noqa: E402
from sciscape.clustering.leiden_basin_profile import changed_support_nodes  # noqa: E402
from sciscape.clustering.leiden_basin_profile import parse_node_ids  # noqa: E402
from sciscape.clustering.leiden_basin_search import (  # noqa: E402
    TransitionAction,
    boundary_shell_context_nodes,
    edge_public_row,
    label_closure_context_nodes,
    node_csv,
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
DEFAULT_OUTPUT_DIR = COMBINED_DIR / (
    "basin_transition_attachment_margin_joint_bundle_field34_cc_c0_p6_p8_p10_v0"
)

ACTION_JOINT_MUTABLE = "joint_attachment_margin_mutable_bundle"
ACTION_JOINT_CANDIDATE_TRANSPLANT = "joint_attachment_margin_candidate_bundle"

ROWS_FILENAME = "attachment_margin_joint_bundle_rows.csv"
SUMMARY_ROWS_FILENAME = "attachment_margin_joint_bundle_summary_rows.csv"
EDGE_ROWS_FILENAME = "attachment_margin_joint_bundle_edges.csv"
CONFIG_FILENAME = "attachment_margin_joint_bundle_config.json"
SUMMARY_FILENAME = "attachment_margin_joint_bundle_summary.json"
REPORT_FILENAME = "attachment_margin_joint_bundle_report.md"


def _parse_int_tuple(value: str, default: tuple[int, ...]) -> tuple[int, ...]:
    text = str(value).strip()
    if not text:
        return default
    parsed = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    return parsed or default


def _parse_float_tuple(value: str, default: tuple[float, ...]) -> tuple[float, ...]:
    text = str(value).strip()
    if not text:
        return default
    parsed = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    return parsed or default


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


def _source_context_nodes(
    *,
    source_move_dir: Path,
    source_recovery_policy: str,
) -> np.ndarray:
    source_moves = pd.read_csv(source_move_dir / SOURCE_MOVE_ROWS_FILENAME)
    source_move, _source_recovery_index = _select_source_move(
        source_moves,
        recovery_policy=source_recovery_policy,
    )
    return unique_sorted_u32(parse_node_ids(source_move["selected_node_ids"]))


def _target_nodes_from_scores(
    score_rows: pd.DataFrame,
    *,
    source_case: str,
    selected_k: int,
) -> np.ndarray:
    rows = score_rows[score_rows["source_case"].astype(str).eq(source_case)].copy()
    rows = rows.sort_values(
        ["gate_pull_margin_vs_current_source", "node"],
        ascending=[False, True],
    )
    return unique_sorted_u32(rows.head(int(selected_k))["node"].to_numpy(dtype=np.uint32))


def _context_candidates(
    *,
    family: str,
    source_context_nodes: np.ndarray,
    source_state: Any,
    target_nodes: np.ndarray,
    case_ctx: dict[str, Any],
) -> np.ndarray:
    selected = unique_sorted_u32(target_nodes)
    exclude = unique_sorted_u32(np.concatenate([source_state.mutable_nodes, selected]))
    if family == "none":
        return np.asarray([], dtype=np.uint32)
    if family == "source_context":
        return unique_sorted_u32(np.setdiff1d(source_context_nodes, exclude))
    if family == "candidate_label":
        return label_closure_context_nodes(
            membership=case_ctx["candidate"].recreated.membership,
            direct_nodes=selected,
            exclude_nodes=exclude,
        )
    if family == "current_label":
        return label_closure_context_nodes(
            membership=source_state.membership,
            direct_nodes=selected,
            exclude_nodes=exclude,
        )
    if family == "boundary_shell":
        arrays = case_ctx["arrays"]
        return boundary_shell_context_nodes(
            src=np.asarray(arrays.src, dtype=np.uint32),
            dst=np.asarray(arrays.dst, dtype=np.uint32),
            direct_nodes=selected,
            exclude_nodes=exclude,
            node_count=int(source_state.membership.size),
        )
    raise ValueError(f"Unknown context family: {family}")


def _select_context(
    *,
    candidates: np.ndarray,
    target_nodes: np.ndarray,
    case_ctx: dict[str, Any],
    context_multiplier: float,
    max_context_nodes: int,
) -> np.ndarray:
    nodes = unique_sorted_u32(candidates)
    if nodes.size == 0:
        return nodes
    cap = min(
        int(max_context_nodes),
        max(1, int(math.ceil(float(context_multiplier) * max(1, int(target_nodes.size))))),
    )
    arrays = case_ctx["arrays"]
    pull = weighted_pull_to_nodes(
        src=np.asarray(arrays.src, dtype=np.uint32),
        dst=np.asarray(arrays.dst, dtype=np.uint32),
        weight=np.asarray(arrays.weight, dtype=np.float64),
        target_nodes=target_nodes,
        node_count=int(case_ctx["baseline"].membership.size),
    )
    return topk_by_pull(candidate_nodes=nodes, pull_scores=pull, max_nodes=cap)


def _row_verdict(
    row: dict[str, Any],
    *,
    control_ctx: dict[str, Any],
    min_progress: float,
) -> str:
    directed = float(row["state_target_progress_from_vanilla"]) >= float(min_progress)
    if not directed:
        return "joint_not_candidate_directed"
    if control_ctx and float(row.get("quality_minus_best_control", -math.inf)) >= 0.0:
        return "joint_beats_broad_control"
    if (
        control_ctx
        and float(row.get("quality_minus_best_same_randomness_control", -math.inf))
        >= 0.0
    ):
        return "joint_beats_same_randomness_control"
    if float(row["state_delta_q_vs_vanilla"]) >= 0.0:
        return "joint_recovered_to_vanilla_quality"
    if float(row["joint_delta_q_gain_vs_source"]) > 0.0:
        return "joint_directed_quality_lag"
    return "joint_directed_quality_loss"


def _probe_one_source(
    *,
    stage1_summary_row: pd.Series,
    score_rows: pd.DataFrame,
    control_rows: pd.DataFrame,
    source_recovery_policy: str,
    requested_recovery_seed: int,
    target_ks: tuple[int, ...],
    context_families: tuple[str, ...],
    context_multipliers: tuple[float, ...],
    max_context_nodes: int,
    min_progress: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_case = str(stage1_summary_row["source_case"])
    source_move_dir = Path(str(stage1_summary_row["source_move_dir"]))
    config, case_ctx, source_state, source_row, recovery_seed, meta = _rebuild_source_state(
        source_move_dir=source_move_dir,
        source_case=source_case,
        source_recovery_policy=source_recovery_policy,
        requested_recovery_seed=requested_recovery_seed,
    )
    control_ctx = _control_context(control_rows, source_case)
    source_context = _source_context_nodes(
        source_move_dir=source_move_dir,
        source_recovery_policy=source_recovery_policy,
    )
    base_context = {
        **meta,
        "path_policy": "attachment_margin_joint_bundle",
        "source_state_id": source_state.state_id,
    }
    rows: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    trial_index = 0

    for target_k in target_ks:
        target_nodes = _target_nodes_from_scores(
            score_rows,
            source_case=source_case,
            selected_k=int(target_k),
        )
        if target_nodes.size == 0:
            continue
        for family in context_families:
            candidates = _context_candidates(
                family=family,
                source_context_nodes=source_context,
                source_state=source_state,
                target_nodes=target_nodes,
                case_ctx=case_ctx,
            )
            multipliers = (0.0,) if family == "none" else context_multipliers
            for multiplier in multipliers:
                context_nodes = (
                    np.asarray([], dtype=np.uint32)
                    if family == "none"
                    else _select_context(
                        candidates=candidates,
                        target_nodes=target_nodes,
                        case_ctx=case_ctx,
                        context_multiplier=float(multiplier),
                        max_context_nodes=max_context_nodes,
                    )
                )
                if family != "none" and context_nodes.size == 0:
                    continue
                bundle_nodes = unique_sorted_u32(np.concatenate([target_nodes, context_nodes]))
                for move_kind in ("joint_mutable", "candidate_bundle_transplant"):
                    key = (str(target_k), family, move_kind, node_csv(bundle_nodes))
                    if key in seen:
                        continue
                    seen.add(key)
                    trial_index += 1
                    action = TransitionAction(
                        action_type=(
                            ACTION_JOINT_MUTABLE
                            if move_kind == "joint_mutable"
                            else ACTION_JOINT_CANDIDATE_TRANSPLANT
                        ),
                        action_params=(
                            f"target_k={int(target_k)};"
                            f"context_family={family};"
                            f"move_kind={move_kind};"
                            f"context_multiplier={float(multiplier):g};"
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
                    final_aligned = changed_support_nodes(
                        source_state.membership,
                        child.membership,
                    )
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
                            **base_context,
                            "target_k": int(target_k),
                            "target_node_ids": node_csv(target_nodes),
                            "context_family": family,
                            "context_multiplier": float(multiplier),
                            "move_kind": move_kind,
                        },
                        parent_row=source_row,
                        min_support_shift_from_vanilla=0.01,
                        min_material_q_gain=0.01,
                    )
                    row.update(
                        {
                            "target_node_ids": node_csv(target_nodes),
                            "context_node_count": int(context_nodes.size),
                            "context_node_ids": node_csv(context_nodes),
                            "bundle_node_count": int(bundle_nodes.size),
                            "bundle_node_ids": node_csv(bundle_nodes),
                            "joint_pre_polish_changed_node_count": int(
                                pre_change["exact_changed_node_count"]
                            ),
                            "joint_pre_polish_exact_changed_node_count": int(
                                pre_change["exact_changed_node_count"]
                            ),
                            "joint_pre_polish_aligned_changed_node_count": int(
                                pre_change["aligned_changed_node_count"]
                            ),
                            "joint_pre_polish_exact_only_changed_node_count": int(
                                pre_change["exact_only_changed_node_count"]
                            ),
                            "joint_pre_polish_delta_q_gain_vs_source": pre_quality
                            - float(source_row["state_quality"]),
                            "joint_final_changed_node_count": int(
                                final_change["exact_changed_node_count"]
                            ),
                            "joint_final_exact_changed_node_count": int(
                                final_change["exact_changed_node_count"]
                            ),
                            "joint_final_aligned_changed_node_count": int(
                                final_change["aligned_changed_node_count"]
                            ),
                            "joint_final_exact_only_changed_node_count": int(
                                final_change["exact_only_changed_node_count"]
                            ),
                            "joint_final_endpoint_distance_to_source": float(
                                final_change["endpoint_distance"]
                            ),
                            "joint_final_aligned_changed_node_ids": node_csv(final_aligned),
                            "joint_delta_q_gain_vs_source": float(row["state_quality"])
                            - float(source_row["state_quality"]),
                            "joint_target_progress_gain_vs_source": float(
                                row["state_target_progress_from_vanilla"]
                            )
                            - float(source_row["state_target_progress_from_vanilla"]),
                            "joint_support_gain_vs_source": float(
                                row["state_support_distance_to_vanilla"]
                            )
                            - float(source_row["state_support_distance_to_vanilla"]),
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
                    row["joint_verdict"] = _row_verdict(
                        row,
                        control_ctx=control_ctx,
                        min_progress=min_progress,
                    )
                    rows.append(row)
                    edges.append(
                        edge_public_row(
                            parent_state_id=source_state.state_id,
                            child_state_id=child.state_id,
                            action=action,
                            context={
                                **case_ctx["public_context"],
                                "source_case": source_case,
                                "target_k": int(target_k),
                                "context_family": family,
                                "move_kind": move_kind,
                            },
                        )
                    )

    probe_rows = pd.DataFrame(rows)
    edge_rows = pd.DataFrame(edges)
    directed = probe_rows[
        probe_rows["state_target_progress_from_vanilla"].astype(float) >= float(min_progress)
    ].copy()
    best_pool = directed if not directed.empty else probe_rows
    best = best_pool.sort_values(
        [
            "state_quality",
            "state_target_progress_from_vanilla",
            "mutable_node_count",
        ],
        ascending=[False, False, True],
    ).iloc[0]
    summary = pd.DataFrame(
        [
            {
                "source_case": source_case,
                "source_delta_q_vs_vanilla": float(source_row["state_delta_q_vs_vanilla"]),
                "source_target_progress": float(
                    source_row["state_target_progress_from_vanilla"]
                ),
                "best_target_k": int(best["target_k"]),
                "best_target_node_ids": str(best["target_node_ids"]),
                "best_context_family": str(best["context_family"]),
                "best_move_kind": str(best["move_kind"]),
                "best_context_node_count": int(best["context_node_count"]),
                "best_bundle_node_count": int(best["bundle_node_count"]),
                "best_delta_q_vs_vanilla": float(best["state_delta_q_vs_vanilla"]),
                "best_delta_q_gain_vs_source": float(
                    best["joint_delta_q_gain_vs_source"]
                ),
                "best_target_progress": float(
                    best["state_target_progress_from_vanilla"]
                ),
                "best_target_progress_gain_vs_source": float(
                    best["joint_target_progress_gain_vs_source"]
                ),
                "best_mutable_node_count": int(best["mutable_node_count"]),
                "best_quality_minus_best_control": float(
                    best.get("quality_minus_best_control", math.nan)
                ),
                "best_quality_minus_best_same_randomness_control": float(
                    best.get("quality_minus_best_same_randomness_control", math.nan)
                ),
                "best_verdict": str(best["joint_verdict"]),
                "joint_action_count": int(len(probe_rows)),
                "joint_directed_count": int(len(directed)),
            }
        ]
    )
    return probe_rows, edge_rows, summary


def _write_report(
    path: Path,
    *,
    rows: pd.DataFrame,
    summary_rows: pd.DataFrame,
) -> None:
    summary_cols = [
        "source_case",
        "source_delta_q_vs_vanilla",
        "best_target_k",
        "best_target_node_ids",
        "best_context_family",
        "best_move_kind",
        "best_bundle_node_count",
        "best_delta_q_vs_vanilla",
        "best_delta_q_gain_vs_source",
        "best_target_progress",
        "best_quality_minus_best_control",
        "best_quality_minus_best_same_randomness_control",
        "best_verdict",
    ]
    row_cols = [
        "source_case",
        "target_k",
        "context_family",
        "move_kind",
        "context_node_count",
        "bundle_node_count",
        "joint_pre_polish_changed_node_count",
        "joint_pre_polish_aligned_changed_node_count",
        "joint_pre_polish_delta_q_gain_vs_source",
        "joint_final_changed_node_count",
        "joint_final_aligned_changed_node_count",
        "joint_final_exact_only_changed_node_count",
        "joint_final_endpoint_distance_to_source",
        "state_delta_q_vs_vanilla",
        "joint_delta_q_gain_vs_source",
        "state_target_progress_from_vanilla",
        "mutable_node_count",
        "quality_minus_best_control",
        "quality_minus_best_same_randomness_control",
        "joint_verdict",
    ]
    lines = [
        "# Attachment-Margin Joint Bundle Probe",
        "",
        "This diagnostic activates target and companion context together before",
        "bounded polish. It follows the negative stage2 result where post-target",
        "context opening was a no-op.",
        "",
        "## Summary",
        "",
    ]
    lines.extend(
        _markdown_table(summary_rows[[c for c in summary_cols if c in summary_rows]], max_rows=30)
    )
    lines.extend(["", "## Top Rows", ""])
    display = rows.sort_values(
        ["source_case", "state_quality", "state_target_progress_from_vanilla"],
        ascending=[True, False, False],
    )
    lines.extend(_markdown_table(display[[c for c in row_cols if c in display]], max_rows=120))
    same_randomness_wins = rows[
        (rows["state_target_progress_from_vanilla"].astype(float) > 0.0)
        & (rows["quality_minus_best_same_randomness_control"].astype(float) >= 0.0)
    ].copy()
    if not same_randomness_wins.empty:
        lines.extend(["", "## Same-Randomness Wins", ""])
        win_cols = [
            "source_case",
            "target_k",
            "target_node_ids",
            "context_family",
            "move_kind",
            "context_node_count",
            "bundle_node_count",
            "state_delta_q_vs_vanilla",
            "state_target_progress_from_vanilla",
            "mutable_node_count",
            "quality_minus_best_control",
            "quality_minus_best_same_randomness_control",
            "joint_verdict",
        ]
        wins = same_randomness_wins.sort_values(
            ["source_case", "mutable_node_count", "state_quality"],
            ascending=[True, True, False],
        )
        lines.extend(_markdown_table(wins[[c for c in win_cols if c in wins]], max_rows=80))
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- Joint bundle rows remain diagnostic unless they beat seed/iteration controls on material and cost-adjusted value.",
            "- Candidate-directed movement and QF recovery are reported separately.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_probe(
    *,
    attachment_dir: Path,
    control_dir: Path | None,
    output_dir: Path,
    source_recovery_policy: str,
    recovery_seed: int,
    target_ks: tuple[int, ...],
    context_families: tuple[str, ...],
    context_multipliers: tuple[float, ...],
    max_context_nodes: int,
    min_progress: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_rows = pd.read_csv(attachment_dir / ATTACHMENT_SUMMARY_ROWS_FILENAME)
    score_rows = pd.read_csv(attachment_dir / ATTACHMENT_SCORE_ROWS_FILENAME)
    control_rows = _load_control_summary(control_dir)
    all_rows: list[pd.DataFrame] = []
    all_edges: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    for _, stage1 in stage1_rows.sort_values(["source_case"]).iterrows():
        rows, edges, summary = _probe_one_source(
            stage1_summary_row=stage1,
            score_rows=score_rows,
            control_rows=control_rows,
            source_recovery_policy=source_recovery_policy,
            requested_recovery_seed=recovery_seed,
            target_ks=target_ks,
            context_families=context_families,
            context_multipliers=context_multipliers,
            max_context_nodes=max_context_nodes,
            min_progress=min_progress,
        )
        all_rows.append(rows)
        all_edges.append(edges)
        all_summaries.append(summary)

    rows = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    edges = pd.concat(all_edges, ignore_index=True) if all_edges else pd.DataFrame()
    summary_rows = (
        pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    )
    rows.to_csv(output_dir / ROWS_FILENAME, index=False)
    edges.to_csv(output_dir / EDGE_ROWS_FILENAME, index=False)
    summary_rows.to_csv(output_dir / SUMMARY_ROWS_FILENAME, index=False)
    config = {
        "attachment_dir": str(attachment_dir),
        "control_dir": "" if control_dir is None else str(control_dir),
        "output_dir": str(output_dir),
        "source_recovery_policy": source_recovery_policy,
        "requested_recovery_seed": int(recovery_seed),
        "target_ks": [int(value) for value in target_ks],
        "context_families": list(context_families),
        "context_multipliers": [float(value) for value in context_multipliers],
        "max_context_nodes": int(max_context_nodes),
        "min_progress": float(min_progress),
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "schema": "leiden_basin_attachment_margin_joint_bundle.v0",
        "output_dir": str(output_dir),
        "source_count": int(len(summary_rows)),
        "row_count": int(len(rows)),
        "verdict_counts": rows["joint_verdict"].value_counts().to_dict()
        if not rows.empty
        else {},
        "best_verdict_counts": summary_rows["best_verdict"].value_counts().to_dict()
        if not summary_rows.empty
        else {},
        "paths": {
            "rows": str(output_dir / ROWS_FILENAME),
            "summary_rows": str(output_dir / SUMMARY_ROWS_FILENAME),
            "report": str(output_dir / REPORT_FILENAME),
        },
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / REPORT_FILENAME, rows=rows, summary_rows=summary_rows)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attachment-dir", type=Path, default=DEFAULT_ATTACHMENT_DIR)
    parser.add_argument("--control-dir", type=Path, default=DEFAULT_CONTROL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-recovery-policy", default=DEFAULT_SOURCE_RECOVERY_POLICY)
    parser.add_argument(
        "--recovery-seed",
        type=int,
        default=0,
        help="0 reuses each source row's original seed, 21000 + source_recovery_index.",
    )
    parser.add_argument("--target-ks", default="1,2,4,8")
    parser.add_argument(
        "--context-families",
        default="none,source_context,candidate_label,current_label,boundary_shell",
    )
    parser.add_argument("--context-multipliers", default="8,32")
    parser.add_argument("--max-context-nodes", type=int, default=256)
    parser.add_argument("--min-progress", type=float, default=0.005)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_probe(
        attachment_dir=args.attachment_dir,
        control_dir=args.control_dir,
        output_dir=args.output_dir,
        source_recovery_policy=args.source_recovery_policy,
        recovery_seed=args.recovery_seed,
        target_ks=_parse_int_tuple(args.target_ks, (1, 2, 4, 8)),
        context_families=tuple(
            part.strip()
            for part in str(args.context_families).split(",")
            if part.strip()
        ),
        context_multipliers=_parse_float_tuple(args.context_multipliers, (8.0, 32.0)),
        max_context_nodes=args.max_context_nodes,
        min_progress=args.min_progress,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
