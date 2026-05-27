#!/usr/bin/env python3
"""Probe second-stage QF recovery after compact attachment-margin tunneling."""

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

from analyze_leiden_basin_barrier_aware_pathways import (  # noqa: E402
    PREFIX_ROWS_FILENAME as BARRIER_PREFIX_ROWS_FILENAME,
)
from evaluate_leiden_basin_polish_prefixes import select_prefix_rows  # noqa: E402
from evaluate_leiden_basin_target_elbow_polish import (  # noqa: E402
    _rank_and_filter_prefix_rows,
)
from probe_leiden_basin_post_gate_recovery_moves import (  # noqa: E402
    POST_GATE_PATH_SUMMARY_FILENAME,
    _load_case_context,
    _markdown_table,
    _prefix_context,
    _replay_to_source_state,
    _select_source_path,
)
from probe_leiden_basin_post_gate_recovery_subsets import (  # noqa: E402
    SOURCE_MOVE_ROWS_FILENAME,
    _select_source_move,
)
from run_leiden_basin_attachment_margin_cross_prefix_probe import (  # noqa: E402
    ACTION_ATTACHMENT_MARGIN_TARGET_ONLY,
    DEFAULT_OUTPUT_DIR as DEFAULT_ATTACHMENT_DIR,
    DEFAULT_SOURCE_RECOVERY_POLICY,
    SUMMARY_ROWS_FILENAME as ATTACHMENT_SUMMARY_ROWS_FILENAME,
    _load_json,
)
from sciscape.clustering.leiden_basin_profile import parse_node_ids  # noqa: E402
from sciscape.clustering.leiden_basin_profile import compact_membership  # noqa: E402
from sciscape.clustering.leiden_basin_profile import changed_support_nodes  # noqa: E402
from sciscape.clustering.leiden_basin_search import (  # noqa: E402
    POST_GATE_VERDICT_NEAR_MISS,
    TransitionAction,
    boundary_shell_context_nodes,
    edge_public_row,
    label_closure_context_nodes,
    make_child_state,
    node_csv,
    polish_state,
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
DEFAULT_CONTROL_DIR = COMBINED_DIR / (
    "basin_transition_attachment_margin_controls_field34_cc_c0_p6_p8_p10_v0"
)
DEFAULT_OUTPUT_DIR = COMBINED_DIR / (
    "basin_transition_attachment_margin_stage2_recovery_field34_cc_c0_p6_p8_p10_v2"
)

ACTION_STAGE2_CANDIDATE_LABEL_CONTEXT = "stage2_candidate_label_context_topk"
ACTION_STAGE2_CURRENT_LABEL_CONTEXT = "stage2_current_label_context_topk"
ACTION_STAGE2_VANILLA_LABEL_CONTEXT = "stage2_vanilla_label_context_topk"
ACTION_STAGE2_BOUNDARY_CONTEXT = "stage2_boundary_context_topk"

ROWS_FILENAME = "attachment_margin_stage2_recovery_rows.csv"
SUMMARY_ROWS_FILENAME = "attachment_margin_stage2_recovery_summary_rows.csv"
EDGE_ROWS_FILENAME = "attachment_margin_stage2_recovery_edges.csv"
CONFIG_FILENAME = "attachment_margin_stage2_recovery_config.json"
SUMMARY_FILENAME = "attachment_margin_stage2_recovery_summary.json"
REPORT_FILENAME = "attachment_margin_stage2_recovery_report.md"


def _parse_float_tuple(value: str, default: tuple[float, ...]) -> tuple[float, ...]:
    text = str(value).strip()
    if not text:
        return default
    parsed = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    return parsed or default


def _load_control_summary(control_dir: Path | None) -> pd.DataFrame:
    if control_dir is None:
        return pd.DataFrame()
    path = control_dir / "attachment_margin_control_summary_rows.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _control_context(control_rows: pd.DataFrame, source_case: str) -> dict[str, Any]:
    if control_rows.empty:
        return {}
    matched = control_rows[control_rows["source_case"].astype(str).eq(source_case)]
    if matched.empty:
        return {}
    row = matched.iloc[0]
    return {
        "best_quality_control_quality": float(row["best_quality_control_quality"]),
        "best_quality_control_target_progress": float(
            row["best_quality_control_target_progress"]
        ),
        "best_same_randomness_control_quality": float(
            row.get("best_same_randomness_control_quality", math.nan)
        ),
        "best_same_randomness_control_target_progress": float(
            row.get("best_same_randomness_control_target_progress", math.nan)
        ),
    }


def _select_stage1_rows(attachment_dir: Path) -> pd.DataFrame:
    path = attachment_dir / ATTACHMENT_SUMMARY_ROWS_FILENAME
    rows = pd.read_csv(path)
    if rows.empty:
        raise ValueError(f"Attachment summary rows are empty: {path}")
    return rows.sort_values(["source_case"]).reset_index(drop=True)


def _rebuild_source_state(
    *,
    source_move_dir: Path,
    source_case: str,
    source_recovery_policy: str,
    requested_recovery_seed: int,
) -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, Any], int]:
    config = _load_json(source_move_dir / "post_gate_recovery_move_config.json")
    if not config:
        raise ValueError(f"Missing source config in {source_move_dir}")

    pair_id = str(config.get("pair_id", "c0-s11-r0.001"))
    prefix_rank = int(config["prefix_rank"])
    post_gate_dir = Path(config["post_gate_dir"])
    prefix_dir = Path(config["prefix_dir"])
    profile_batch_dir = Path(config["profile_batch_dir"])
    vanilla_dir = Path(config["vanilla_dir"])
    candidate_dirs = tuple(Path(path) for path in config["candidate_dirs"])
    source_verdict = str(config.get("source_verdict", POST_GATE_VERDICT_NEAR_MISS))

    source_moves = pd.read_csv(source_move_dir / SOURCE_MOVE_ROWS_FILENAME)
    _source_move, source_recovery_index = _select_source_move(
        source_moves,
        recovery_policy=source_recovery_policy,
    )
    effective_recovery_seed = (
        int(requested_recovery_seed)
        if int(requested_recovery_seed) > 0
        else 21000 + int(source_recovery_index)
    )

    post_gate_paths = pd.read_csv(post_gate_dir / POST_GATE_PATH_SUMMARY_FILENAME)
    source_path = _select_source_path(
        post_gate_paths,
        pair_id=pair_id,
        prefix_rank=prefix_rank,
        verdict=source_verdict,
    )
    prefixes = select_prefix_rows(
        pd.read_csv(prefix_dir / BARRIER_PREFIX_ROWS_FILENAME),
        pair_ids=(pair_id,),
        top_prefixes_per_case=max(prefix_rank, 10),
    )
    prefixes = _rank_and_filter_prefix_rows(
        prefixes,
        selected_prefix_ranks=(prefix_rank,),
    )
    if prefixes.empty:
        raise ValueError(f"No prefix row selected for {pair_id} rank {prefix_rank}")
    prefix_row = prefixes.iloc[0]

    case_ctx = _load_case_context(
        prefix_row=prefix_row,
        profile_batch_dir=profile_batch_dir,
        candidate_dirs=candidate_dirs,
        vanilla_dir=vanilla_dir,
        baseline_iterations=int(config.get("baseline_iterations", 10)),
        candidate_polish_iterations=int(config.get("candidate_polish_iterations", 5)),
        resolution=float(config.get("resolution", 0.01)),
        randomness=float(config.get("randomness", 0.01)),
        perturb_seed_offset=1000,
    )
    source_state, source_row, _, _ = _replay_to_source_state(
        prefix_row=prefix_row,
        source_path=source_path,
        case_ctx=case_ctx,
        target_action_multiplier=float(config.get("target_action_multiplier", 0.5)),
        max_target_action_nodes=int(config.get("max_target_action_nodes", 64)),
        cumulative_fraction=0.80,
        min_score_fraction=0.05,
        min_gap_fraction=0.25,
        min_guarded_pull_fraction=0.50,
        local_polish_iterations=int(config.get("local_polish_iterations", 3)),
        resolution=float(config.get("resolution", 0.01)),
        randomness=float(config.get("randomness", 0.01)),
        polish_seed_offset=2000,
        min_support_shift_from_vanilla=0.01,
        min_material_q_gain=0.01,
    )
    meta = {
        "source_case": source_case,
        "source_move_dir": str(source_move_dir),
        "source_recovery_index": int(source_recovery_index),
        "effective_recovery_seed": int(effective_recovery_seed),
        "prefix_rank": int(prefix_rank),
        "resolution": float(config.get("resolution", 0.01)),
        "randomness": float(config.get("randomness", 0.01)),
        "recovery_polish_iterations": int(
            config.get("recovery_polish_iterations", 6)
        ),
        **case_ctx["public_context"],
        **_prefix_context(prefix_row),
    }
    return config, case_ctx, source_state, source_row, effective_recovery_seed, meta


def _stage2_actions(
    *,
    stage1_state: Any,
    selected_nodes: np.ndarray,
    case_ctx: dict[str, Any],
    context_multipliers: tuple[float, ...],
    max_context_nodes: int,
) -> list[tuple[str, str, float, np.ndarray, TransitionAction]]:
    arrays = case_ctx["arrays"]
    src = np.asarray(arrays.src, dtype=np.uint32)
    dst = np.asarray(arrays.dst, dtype=np.uint32)
    weight = np.asarray(arrays.weight, dtype=np.float64)
    node_count = int(stage1_state.membership.size)
    selected = unique_sorted_u32(selected_nodes)
    exclude = unique_sorted_u32(
        np.concatenate([stage1_state.mutable_nodes, stage1_state.context_nodes])
    )
    pull = weighted_pull_to_nodes(
        src=src,
        dst=dst,
        weight=weight,
        target_nodes=selected,
        node_count=node_count,
    )
    family_candidates = {
        "candidate_label": (
            ACTION_STAGE2_CANDIDATE_LABEL_CONTEXT,
            label_closure_context_nodes(
                membership=case_ctx["candidate"].recreated.membership,
                direct_nodes=selected,
                exclude_nodes=exclude,
            ),
        ),
        "current_label": (
            ACTION_STAGE2_CURRENT_LABEL_CONTEXT,
            label_closure_context_nodes(
                membership=stage1_state.membership,
                direct_nodes=selected,
                exclude_nodes=exclude,
            ),
        ),
        "vanilla_label": (
            ACTION_STAGE2_VANILLA_LABEL_CONTEXT,
            label_closure_context_nodes(
                membership=case_ctx["vanilla"].membership,
                direct_nodes=selected,
                exclude_nodes=exclude,
            ),
        ),
        "boundary_shell": (
            ACTION_STAGE2_BOUNDARY_CONTEXT,
            boundary_shell_context_nodes(
                src=src,
                dst=dst,
                direct_nodes=selected,
                exclude_nodes=exclude,
                node_count=node_count,
            ),
        ),
    }
    actions: list[tuple[str, str, float, np.ndarray, TransitionAction]] = []
    seen: set[tuple[str, str, str]] = set()
    for family, (action_type, candidates) in family_candidates.items():
        for multiplier in context_multipliers:
            cap = min(
                int(max_context_nodes),
                max(1, int(math.ceil(float(multiplier) * max(1, selected.size)))),
            )
            context = topk_by_pull(
                candidate_nodes=candidates,
                pull_scores=pull,
                max_nodes=cap,
            )
            if context.size == 0:
                continue
            for move_kind, action_nodes in (
                ("context_only", None),
                ("candidate_transplant", context),
            ):
                key = (family, move_kind, node_csv(context))
                if key in seen:
                    continue
                seen.add(key)
                action = TransitionAction(
                    action_type=action_type,
                    action_params=(
                        f"stage2_family={family};"
                        f"stage2_move_kind={move_kind};"
                        f"context_multiplier={float(multiplier):g};"
                        f"max_context_nodes={int(max_context_nodes)};"
                        f"context_k={int(context.size)}"
                    ),
                    context_nodes=(
                        context
                        if move_kind == "context_only"
                        else np.asarray([], dtype=np.uint32)
                    ),
                    action_nodes=action_nodes,
                )
                actions.append((family, move_kind, float(multiplier), context, action))
    return actions


def _polished_child_with_reference(
    *,
    parent: Any,
    action: TransitionAction,
    graph: Any,
    donor_membership: np.ndarray,
    reference_nodes: np.ndarray,
    resolution: float,
    seed: int,
    n_iterations: int,
    randomness: float,
    child_index: int,
) -> Any:
    action_nodes = unique_sorted_u32(
        [] if action.action_nodes is None else action.action_nodes
    )
    membership = (
        transplant_action_nodes(
            membership=parent.membership,
            donor_membership=donor_membership,
            action_nodes=action_nodes,
            reference_nodes=reference_nodes,
        )
        if action_nodes.size
        else parent.membership
    )
    mutable_nodes = unique_sorted_u32(
        np.concatenate([parent.mutable_nodes, action.context_nodes, action_nodes])
    )
    membership, quality, elapsed = polish_state(
        graph=graph,
        membership=membership,
        mutable_nodes=mutable_nodes,
        resolution=resolution,
        seed=seed,
        n_iterations=n_iterations,
        randomness=randomness,
    )
    return make_child_state(
        parent=parent,
        action=action,
        membership=membership,
        quality=quality,
        elapsed_sec=elapsed,
        child_index=child_index,
    )


def _verdict(
    *,
    row: dict[str, Any],
    stage1_row: dict[str, Any],
    control_ctx: dict[str, Any],
    progress_tolerance: float,
) -> str:
    progress_retained = float(row["state_target_progress_from_vanilla"]) >= (
        float(stage1_row["state_target_progress_from_vanilla"]) - progress_tolerance
    )
    if not progress_retained:
        return "stage2_progress_lost"
    if control_ctx and float(row.get("quality_minus_best_control", -math.inf)) >= 0.0:
        return "stage2_beats_broad_control"
    if control_ctx and float(row.get("quality_minus_best_same_randomness_control", -math.inf)) >= 0.0:
        return "stage2_beats_same_randomness_control"
    if float(row["state_delta_q_vs_vanilla"]) >= 0.0:
        return "stage2_recovered_to_vanilla_quality"
    if (
        float(row["stage2_delta_q_gain_vs_stage1"]) > 0.0
        and progress_retained
    ):
        return "stage2_q_gain_progress_retained"
    return "stage2_no_recovery"


def _probe_one_source(
    *,
    stage1_summary_row: pd.Series,
    control_rows: pd.DataFrame,
    source_recovery_policy: str,
    requested_recovery_seed: int,
    context_multipliers: tuple[float, ...],
    max_context_nodes: int,
    progress_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_case = str(stage1_summary_row["source_case"])
    source_move_dir = Path(str(stage1_summary_row["source_move_dir"]))
    selected_nodes = unique_sorted_u32(
        parse_node_ids(stage1_summary_row["best_selected_node_ids"])
    )
    config, case_ctx, source_state, source_row, recovery_seed, meta = _rebuild_source_state(
        source_move_dir=source_move_dir,
        source_case=source_case,
        source_recovery_policy=source_recovery_policy,
        requested_recovery_seed=requested_recovery_seed,
    )
    control_ctx = _control_context(control_rows, source_case)
    base_context = {
        **meta,
        "path_policy": "attachment_margin_stage2_recovery",
        "source_state_id": source_state.state_id,
        "stage1_selected_node_ids": node_csv(selected_nodes),
        "stage1_selected_k": int(selected_nodes.size),
    }
    rows: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    stage1_action = TransitionAction(
        action_type=ACTION_ATTACHMENT_MARGIN_TARGET_ONLY,
        action_params=(
            f"source_case={source_case};mode=stage1_target_only;"
            f"selected_k={int(selected_nodes.size)}"
        ),
        context_nodes=selected_nodes,
        action_nodes=None,
    )
    stage1_state = _polished_child(
        parent=source_state,
        action=stage1_action,
        graph=case_ctx["graph"],
        donor_membership=case_ctx["candidate"].recreated.membership,
        resolution=float(config.get("resolution", 0.01)),
        seed=int(recovery_seed),
        n_iterations=int(config.get("recovery_polish_iterations", 6)),
        randomness=float(config.get("randomness", 0.01)),
        child_index=1,
    )
    stage1_row = _evaluate_state(
        state=stage1_state,
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
        context={**base_context, "stage": "stage1_target_only"},
        parent_row=source_row,
        min_support_shift_from_vanilla=0.01,
        min_material_q_gain=0.01,
    )
    stage1_row.update(
        {
            "stage2_family": "",
            "stage2_context_multiplier": math.nan,
            "stage2_context_node_count": 0,
            "stage2_context_node_ids": "",
            "stage2_delta_q_gain_vs_stage1": 0.0,
            "stage2_target_progress_gain_vs_stage1": 0.0,
            "stage2_support_gain_vs_stage1": 0.0,
            "stage2_progress_retained": True,
            **{
                key: value
                for key, value in control_ctx.items()
                if key.startswith("best_")
            },
        }
    )
    if control_ctx:
        stage1_row["quality_minus_best_control"] = float(stage1_row["state_quality"]) - float(
            control_ctx["best_quality_control_quality"]
        )
        stage1_row["quality_minus_best_same_randomness_control"] = float(
            stage1_row["state_quality"]
        ) - float(control_ctx["best_same_randomness_control_quality"])
    stage1_row["stage2_verdict"] = "stage1_reference"
    rows.append(stage1_row)
    edges.append(
        edge_public_row(
            parent_state_id=source_state.state_id,
            child_state_id=stage1_state.state_id,
            action=stage1_action,
            context={**case_ctx["public_context"], "source_case": source_case},
        )
    )

    actions = _stage2_actions(
        stage1_state=stage1_state,
        selected_nodes=selected_nodes,
        case_ctx=case_ctx,
        context_multipliers=context_multipliers,
        max_context_nodes=max_context_nodes,
    )
    action_rows: list[dict[str, Any]] = []
    for action_index, (family, move_kind, multiplier, context_nodes, action) in enumerate(
        actions,
        start=2,
    ):
        pre_membership = stage1_state.membership
        if move_kind == "candidate_transplant":
            pre_membership = transplant_action_nodes(
                membership=stage1_state.membership,
                donor_membership=case_ctx["candidate"].recreated.membership,
                action_nodes=unique_sorted_u32(action.action_nodes),
                reference_nodes=selected_nodes,
            )
        pre_membership = compact_membership(pre_membership)
        pre_change = membership_change_summary(
            reference_membership=stage1_state.membership,
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
            "parent": stage1_state,
            "action": action,
            "graph": case_ctx["graph"],
            "donor_membership": case_ctx["candidate"].recreated.membership,
            "resolution": float(config.get("resolution", 0.01)),
            "seed": int(recovery_seed) + int(action_index) * 1000,
            "n_iterations": int(config.get("recovery_polish_iterations", 6)),
            "randomness": float(config.get("randomness", 0.01)),
            "child_index": action_index,
        }
        child = (
            _polished_child_with_reference(
                **child_kwargs,
                reference_nodes=selected_nodes,
            )
            if move_kind == "candidate_transplant"
            else _polished_child(**child_kwargs)
        )
        final_change = membership_change_summary(
            reference_membership=stage1_state.membership,
            membership=child.membership,
            sketch_nodes=case_ctx["sketch_nodes"],
        )
        final_aligned = changed_support_nodes(stage1_state.membership, child.membership)
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
                "stage": "stage2_context_recovery",
                "stage2_family": family,
                "stage2_move_kind": move_kind,
                "stage2_context_multiplier": float(multiplier),
            },
            parent_row=stage1_row,
            min_support_shift_from_vanilla=0.01,
            min_material_q_gain=0.01,
        )
        row.update(
            {
                "stage2_context_node_count": int(context_nodes.size),
                "stage2_context_node_ids": node_csv(context_nodes),
                "stage2_move_kind": move_kind,
                "stage2_pre_polish_changed_node_count": int(
                    pre_change["exact_changed_node_count"]
                ),
                "stage2_pre_polish_exact_changed_node_count": int(
                    pre_change["exact_changed_node_count"]
                ),
                "stage2_pre_polish_aligned_changed_node_count": int(
                    pre_change["aligned_changed_node_count"]
                ),
                "stage2_pre_polish_exact_only_changed_node_count": int(
                    pre_change["exact_only_changed_node_count"]
                ),
                "stage2_pre_polish_delta_q_gain_vs_stage1": pre_quality
                - float(stage1_row["state_quality"]),
                "stage2_final_changed_node_count": int(
                    final_change["exact_changed_node_count"]
                ),
                "stage2_final_exact_changed_node_count": int(
                    final_change["exact_changed_node_count"]
                ),
                "stage2_final_aligned_changed_node_count": int(
                    final_change["aligned_changed_node_count"]
                ),
                "stage2_final_exact_only_changed_node_count": int(
                    final_change["exact_only_changed_node_count"]
                ),
                "stage2_final_endpoint_distance_to_stage1": float(
                    final_change["endpoint_distance"]
                ),
                "stage2_final_aligned_changed_node_ids": node_csv(final_aligned),
                "stage2_delta_q_gain_vs_stage1": float(row["state_quality"])
                - float(stage1_row["state_quality"]),
                "stage2_target_progress_gain_vs_stage1": float(
                    row["state_target_progress_from_vanilla"]
                )
                - float(stage1_row["state_target_progress_from_vanilla"]),
                "stage2_support_gain_vs_stage1": float(
                    row["state_support_distance_to_vanilla"]
                )
                - float(stage1_row["state_support_distance_to_vanilla"]),
                "stage2_progress_retained": bool(
                    float(row["state_target_progress_from_vanilla"])
                    >= (
                        float(stage1_row["state_target_progress_from_vanilla"])
                        - float(progress_tolerance)
                    )
                ),
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
        row["stage2_verdict"] = _verdict(
            row=row,
            stage1_row=stage1_row,
            control_ctx=control_ctx,
            progress_tolerance=progress_tolerance,
        )
        rows.append(row)
        action_rows.append(row)
        edges.append(
            edge_public_row(
                parent_state_id=stage1_state.state_id,
                child_state_id=child.state_id,
                action=action,
                context={
                    **case_ctx["public_context"],
                    "source_case": source_case,
                    "stage2_family": family,
                    "stage2_move_kind": move_kind,
                },
            )
        )

    probe_rows = pd.DataFrame(rows)
    edge_rows = pd.DataFrame(edges)
    action_frame = pd.DataFrame(action_rows)
    retained = action_frame[action_frame["stage2_progress_retained"].astype(bool)]
    best_pool = retained if not retained.empty else action_frame
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
                "stage1_selected_node_ids": node_csv(selected_nodes),
                "stage1_quality": float(stage1_row["state_quality"]),
                "stage1_delta_q_vs_vanilla": float(
                    stage1_row["state_delta_q_vs_vanilla"]
                ),
                "stage1_target_progress": float(
                    stage1_row["state_target_progress_from_vanilla"]
                ),
                "stage1_mutable_node_count": int(stage1_row["mutable_node_count"]),
                "best_stage2_family": str(best["stage2_family"]),
                "best_stage2_move_kind": str(best["stage2_move_kind"]),
                "best_stage2_context_multiplier": float(
                    best["stage2_context_multiplier"]
                ),
                "best_stage2_context_node_count": int(
                    best["stage2_context_node_count"]
                ),
                "best_stage2_quality": float(best["state_quality"]),
                "best_stage2_delta_q_vs_vanilla": float(
                    best["state_delta_q_vs_vanilla"]
                ),
                "best_stage2_delta_q_gain_vs_stage1": float(
                    best["stage2_delta_q_gain_vs_stage1"]
                ),
                "best_stage2_target_progress": float(
                    best["state_target_progress_from_vanilla"]
                ),
                "best_stage2_target_progress_gain_vs_stage1": float(
                    best["stage2_target_progress_gain_vs_stage1"]
                ),
                "best_stage2_mutable_node_count": int(best["mutable_node_count"]),
                "best_stage2_quality_minus_best_control": float(
                    best.get("quality_minus_best_control", math.nan)
                ),
                "best_stage2_quality_minus_best_same_randomness_control": float(
                    best.get("quality_minus_best_same_randomness_control", math.nan)
                ),
                "best_stage2_verdict": str(best["stage2_verdict"]),
                "stage2_action_count": int(len(action_frame)),
                "stage2_retained_progress_count": int(len(retained)),
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
        "stage1_selected_node_ids",
        "stage1_delta_q_vs_vanilla",
        "stage1_target_progress",
        "best_stage2_family",
        "best_stage2_move_kind",
        "best_stage2_context_node_count",
        "best_stage2_delta_q_vs_vanilla",
        "best_stage2_delta_q_gain_vs_stage1",
        "best_stage2_target_progress",
        "best_stage2_quality_minus_best_control",
        "best_stage2_quality_minus_best_same_randomness_control",
        "best_stage2_verdict",
    ]
    row_cols = [
        "source_case",
        "stage",
        "stage2_family",
        "stage2_move_kind",
        "stage2_context_multiplier",
        "stage2_context_node_count",
        "stage2_pre_polish_changed_node_count",
        "stage2_pre_polish_aligned_changed_node_count",
        "stage2_pre_polish_delta_q_gain_vs_stage1",
        "stage2_final_changed_node_count",
        "stage2_final_aligned_changed_node_count",
        "stage2_final_exact_only_changed_node_count",
        "stage2_final_endpoint_distance_to_stage1",
        "state_delta_q_vs_vanilla",
        "stage2_delta_q_gain_vs_stage1",
        "state_target_progress_from_vanilla",
        "stage2_target_progress_gain_vs_stage1",
        "mutable_node_count",
        "quality_minus_best_control",
        "quality_minus_best_same_randomness_control",
        "stage2_verdict",
    ]
    lines = [
        "# Attachment-Margin Stage2 Recovery Probe",
        "",
        "This diagnostic starts from compact target-only tunneling, then opens",
        "small label or boundary context around the selected target nodes to test",
        "whether QF can recover while target progress is retained.",
        "",
        "## Summary",
        "",
    ]
    lines.extend(
        _markdown_table(summary_rows[[c for c in summary_cols if c in summary_rows]], max_rows=30)
    )
    lines.extend(["", "## Probe Rows", ""])
    display = rows.sort_values(
        ["source_case", "stage", "state_quality", "mutable_node_count"],
        ascending=[True, True, False, True],
    )
    lines.extend(_markdown_table(display[[c for c in row_cols if c in display]], max_rows=120))
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- Stage2 is a recovery diagnostic, not a default policy.",
            "- A useful row must improve QF without erasing target progress, then be compared against the seed/iteration controls.",
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
    context_multipliers: tuple[float, ...],
    max_context_nodes: int,
    progress_tolerance: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    control_rows = _load_control_summary(control_dir)
    stage1_rows = _select_stage1_rows(attachment_dir)
    all_rows: list[pd.DataFrame] = []
    all_edges: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    for _, stage1 in stage1_rows.iterrows():
        rows, edges, summary = _probe_one_source(
            stage1_summary_row=stage1,
            control_rows=control_rows,
            source_recovery_policy=source_recovery_policy,
            requested_recovery_seed=recovery_seed,
            context_multipliers=context_multipliers,
            max_context_nodes=max_context_nodes,
            progress_tolerance=progress_tolerance,
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
        "context_multipliers": [float(value) for value in context_multipliers],
        "max_context_nodes": int(max_context_nodes),
        "progress_tolerance": float(progress_tolerance),
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "schema": "leiden_basin_attachment_margin_stage2_recovery.v0",
        "output_dir": str(output_dir),
        "source_count": int(len(summary_rows)),
        "row_count": int(len(rows)),
        "verdict_counts": summary_rows["best_stage2_verdict"].value_counts().to_dict()
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
    parser.add_argument("--context-multipliers", default="4,16,64")
    parser.add_argument("--max-context-nodes", type=int, default=256)
    parser.add_argument("--progress-tolerance", type=float, default=0.001)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_probe(
        attachment_dir=args.attachment_dir,
        control_dir=args.control_dir,
        output_dir=args.output_dir,
        source_recovery_policy=args.source_recovery_policy,
        recovery_seed=args.recovery_seed,
        context_multipliers=_parse_float_tuple(
            args.context_multipliers,
            (4.0, 16.0, 64.0),
        ),
        max_context_nodes=args.max_context_nodes,
        progress_tolerance=args.progress_tolerance,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
