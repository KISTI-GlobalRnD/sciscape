#!/usr/bin/env python3
"""Reusable basin-transition search primitives for Leiden diagnostics.

This module builds on :mod:`sciscape.clustering.leiden_basin_profile`.
It is intentionally diagnostic-only: it defines small, auditable transition
actions and state classification helpers, but it does not promote any
Dongdaemun operator into the production Leiden path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from sciscape.clustering.leiden_basin_profile import (
    BARRIER_CLOSURE_COMPOUND_MISS,
    BARRIER_GREEDY_VISIBLE,
    BARRIER_POLISH_RECOVERY_MISS,
    BARRIER_PROGRESS_GREEDY_MISS,
    BARRIER_Q_GREEDY_MISS,
    compact_membership,
    endpoint_distance,
    fixed_outside,
    membership_metric_row,
    parse_node_ids,
    parse_unit_ids,
    support_progress_from_vanilla,
)

ACTION_PREFIX_ONLY = "prefix_only"
ACTION_CANDIDATE_CLOSURE_TOPK = "candidate_closure_topk"
ACTION_VANILLA_CLOSURE_TOPK = "vanilla_closure_topk"
ACTION_BOUNDARY_SHELL_TOPK = "boundary_shell_topk"
ACTION_REMAINING_TARGET_TOPK = "remaining_target_topk"
ACTION_REMAINING_TARGET_UNIT_TOPK = "remaining_target_unit_topk"
ACTION_RECOVERY_CANDIDATE_CONTEXT_TOPK = "recovery_candidate_context_topk"
ACTION_RECOVERY_VANILLA_CONTEXT_TOPK = "recovery_vanilla_context_topk"
ACTION_RECOVERY_BOUNDARY_CONTEXT_TOPK = "recovery_boundary_context_topk"
ACTION_RECOVERY_CANDIDATE_TRANSPLANT_TOPK = "recovery_candidate_transplant_topk"
ACTION_RECOVERY_BOUNDARY_TRANSPLANT_TOPK = "recovery_boundary_transplant_topk"

TUNNEL_ROUTE_RECOVERABLE = "recoverable_tunnel"
TUNNEL_ROUTE_DIRECT_RECOVERED = "directed_recovered_flat"
TUNNEL_ROUTE_UNRECOVERED_DETOUR = "unrecovered_detour"
TUNNEL_ROUTE_SUPPORT_GATE_NO_TARGET = "support_gate_no_target_progress"
TUNNEL_ROUTE_PARTIAL_PROGRESS = "partial_progress_probe"
TUNNEL_ROUTE_STALLED = "stalled"

TUNNEL_OPERATOR_RECOVERABLE_SEED = "recoverable_tunnel_seed"
TUNNEL_OPERATOR_RECOVERY_TARGET = "unrecovered_detour_recovery_target"
TUNNEL_OPERATOR_ENTRANCE_PROBE = "partial_progress_entrance_probe"
TUNNEL_OPERATOR_BACKGROUND = "background"

POST_GATE_STEP_NO_GATE = "no_gate"
POST_GATE_STEP_PRE_GATE = "pre_gate"
POST_GATE_STEP_GATE_ENTRY = "gate_entry"
POST_GATE_STEP_RECOVERED = "post_gate_recovered"
POST_GATE_STEP_RECOVERY_TREND = "post_gate_q_recovery_trend"
POST_GATE_STEP_SUPPORT_DEEPENING = "post_gate_support_deepening"
POST_GATE_STEP_SUPPORT_QUALITY_TRADEOFF = "post_gate_support_quality_tradeoff"
POST_GATE_STEP_QUALITY_REGRESSION = "post_gate_quality_regression"
POST_GATE_STEP_PLATEAU = "post_gate_plateau"

POST_GATE_VERDICT_NO_GATE = "no_gate"
POST_GATE_VERDICT_GATE_TERMINAL = "gate_terminal_no_recovery"
POST_GATE_VERDICT_RECOVERED = "post_gate_recovered"
POST_GATE_VERDICT_NEAR_MISS = "near_miss_recovery_trend"
POST_GATE_VERDICT_SUPPORT_TRADEOFF = "support_deepening_quality_tradeoff"
POST_GATE_VERDICT_SUPPORT_ONLY = "support_deepening_no_recovery"
POST_GATE_VERDICT_QUALITY_ONLY = "quality_recovery_without_support"
POST_GATE_VERDICT_PLATEAU = "post_gate_plateau"

POST_GATE_RECOVERY_MOVE_RECOVERED = "q_recovered_support_retained"
POST_GATE_RECOVERY_MOVE_Q_GAIN = "q_gain_support_retained"
POST_GATE_RECOVERY_MOVE_SUPPORT_TRADEOFF = "support_gain_quality_tradeoff"
POST_GATE_RECOVERY_MOVE_SUPPORT_LOSS_Q_GAIN = "q_gain_support_loss"
POST_GATE_RECOVERY_MOVE_PLATEAU = "plateau"
POST_GATE_RECOVERY_MOVE_REGRESSION = "quality_regression"

TARGET_SELECTION_RAW_PREFIX = "raw_prefix"
TARGET_SELECTION_PREFIX_POLISH = "prefix_polish"
TARGET_SELECTION_FIXED_CAP = "fixed_cap"
TARGET_SELECTION_GUARDED_ELBOW = "guarded_elbow"
TARGET_SELECTION_FIXED_TAIL_BACKFILL = "fixed_tail_backfill"
TARGET_SELECTION_POLICIES = (
    TARGET_SELECTION_GUARDED_ELBOW,
    TARGET_SELECTION_FIXED_CAP,
    TARGET_SELECTION_FIXED_TAIL_BACKFILL,
)

SEARCH_LABEL_SUPPORT_SHIFT_Q_RECOVERED = "support_shift_q_recovered"
SEARCH_LABEL_VANILLA_COLLAPSE = "vanilla_collapse"
SEARCH_LABEL_QUALITY_LOSS = "quality_loss"
SEARCH_LABEL_LOW_ROI_SUPPORT_SHIFT = "low_roi_support_shift"
SEARCH_LABEL_RAW_ONLY = "raw_only"

REACHABILITY_LABEL_SUPPORT_GATE_REACHED = "support_gate_reached"
REACHABILITY_LABEL_TARGET_PROGRESS = "target_progress"
REACHABILITY_LABEL_SOURCE_ESCAPE = "source_escape"
REACHABILITY_LABEL_COVERAGE_ONLY = "coverage_only"
REACHABILITY_LABEL_STALLED = "stalled"

SEARCH_POLICY_QUALITY = "quality"
SEARCH_POLICY_PROGRESS = "progress"
SEARCH_POLICY_BALANCED = "balanced"
SEARCH_POLICY_STATE_GREEDY = "state_greedy"
SEARCH_POLICY_REACHABILITY_FIRST = "reachability_first"
SEARCH_POLICIES = (
    SEARCH_POLICY_STATE_GREEDY,
    SEARCH_POLICY_QUALITY,
    SEARCH_POLICY_PROGRESS,
    SEARCH_POLICY_BALANCED,
    SEARCH_POLICY_REACHABILITY_FIRST,
)

TARGET_UNIT_LABEL_INTERSECTION_BLOCK = "label_intersection_block"
TARGET_UNIT_CONNECTED_COMPONENT = "target_connected_component"
TARGET_UNIT_TRIANGLE_SUPPORTED_COMPONENT = "triangle_supported_component"
TARGET_UNIT_TYPES = (
    TARGET_UNIT_LABEL_INTERSECTION_BLOCK,
    TARGET_UNIT_CONNECTED_COMPONENT,
    TARGET_UNIT_TRIANGLE_SUPPORTED_COMPONENT,
)

GREEDY_CONTROL_NOT_CHECKED = "control_not_checked"
GREEDY_CONTROL_NOT_CANDIDATE_DIRECTED = "not_candidate_directed"
GREEDY_CONTROL_REPRODUCED = "candidate_directed_control_reproduced"
GREEDY_CONTROL_BRANCH_UNIQUE = "branch_unique_candidate_directed"
GREEDY_CONTROL_BRANCH_UNIQUE_QUALITY_LAG = (
    "branch_unique_candidate_directed_quality_lag"
)
GREEDY_CONTROL_BRANCH_LAGS_CANDIDATE_DIRECTED_CONTROL = (
    "branch_lags_candidate_directed_control"
)

ALIGNED_CORE_PLAN_TARGET_ONLY = "target_core_only"
ALIGNED_CORE_PLAN_BOUNDARY_CORE = "target_core_plus_boundary_core"
ALIGNED_CORE_PLAN_CONTEXT_CORE = "target_core_plus_boundary_context_core"
ALIGNED_CORE_PLAN_CANDIDATE_CONTEXT = "target_core_plus_candidate_context"

ALIGNED_CORE_HANDLE_SELECTOR_ALIGNED_FREQUENCY = "aligned_frequency"
ALIGNED_CORE_HANDLE_SELECTOR_CONTEXT_PULL = "context_pull"
ALIGNED_CORE_HANDLE_SELECTOR_MUTABLE_PENALIZED_CONTEXT_PULL = (
    "mutable_penalized_context_pull"
)
ALIGNED_CORE_HANDLE_SELECTOR_MUTABLE_PENALIZED_ALIGNED = (
    "mutable_penalized_aligned_frequency"
)
ALIGNED_CORE_HANDLE_SELECTOR_POLICIES = (
    ALIGNED_CORE_HANDLE_SELECTOR_ALIGNED_FREQUENCY,
    ALIGNED_CORE_HANDLE_SELECTOR_CONTEXT_PULL,
    ALIGNED_CORE_HANDLE_SELECTOR_MUTABLE_PENALIZED_CONTEXT_PULL,
    ALIGNED_CORE_HANDLE_SELECTOR_MUTABLE_PENALIZED_ALIGNED,
)
ALIGNED_CORE_HANDLE_SELECTOR_REPLAY_POLICIES = {
    ALIGNED_CORE_HANDLE_SELECTOR_ALIGNED_FREQUENCY,
    ALIGNED_CORE_HANDLE_SELECTOR_MUTABLE_PENALIZED_ALIGNED,
}

LOCAL_HANDLE_SELECTOR_ATTACHMENT_MARGIN = "attachment_margin"
LOCAL_HANDLE_SELECTOR_GATE_PULL = "gate_pull"
LOCAL_HANDLE_SELECTOR_NON_SOURCE_GATE_PULL = "non_source_gate_pull"
LOCAL_HANDLE_SELECTOR_RANK_CONSENSUS = "rank_consensus"
LOCAL_HANDLE_SELECTOR_CANDIDATE_LABEL_MARGIN_COHERENT = (
    "candidate_label_margin_coherent"
)
LOCAL_HANDLE_SELECTOR_POLICIES = (
    LOCAL_HANDLE_SELECTOR_ATTACHMENT_MARGIN,
    LOCAL_HANDLE_SELECTOR_GATE_PULL,
    LOCAL_HANDLE_SELECTOR_NON_SOURCE_GATE_PULL,
    LOCAL_HANDLE_SELECTOR_CANDIDATE_LABEL_MARGIN_COHERENT,
    LOCAL_HANDLE_SELECTOR_RANK_CONSENSUS,
)

LOCAL_SELECTOR_READINESS_READY = "selector_test_ready"
LOCAL_SELECTOR_READINESS_LABEL_COMPLETION = "coherent_label_completion_probe"
LOCAL_SELECTOR_READINESS_ALREADY_RECOVERED = "already_recovered_control"
LOCAL_SELECTOR_READINESS_TOO_FEW_HANDLES = "too_few_handles"
LOCAL_SELECTOR_READINESS_NO_LABEL_COMPETITION = "no_label_competition"
LOCAL_SELECTOR_READINESS_WEAK_CANDIDATE_DIRECTION = "weak_candidate_direction"
LOCAL_SELECTOR_READINESS_MISSING_SOURCE = "missing_source_artifact"


@dataclass(frozen=True)
class TransitionSearchState:
    state_id: str
    parent_state_id: str
    depth: int
    prefix_rank: int
    prefix_unit_ids: str
    action_type: str
    action_params: str
    membership: np.ndarray
    quality: float
    direct_nodes: np.ndarray
    target_nodes: np.ndarray
    action_nodes: np.ndarray
    covered_target_nodes: np.ndarray
    mutable_nodes: np.ndarray
    context_nodes: np.ndarray
    applied_actions: tuple[str, ...]
    elapsed_sec: float


@dataclass(frozen=True)
class TransitionAction:
    action_type: str
    action_params: str
    context_nodes: np.ndarray
    action_nodes: np.ndarray | None = None


@dataclass(frozen=True)
class BranchTargetActionCandidate:
    selection_policy: str
    escalation_reason: str
    target_stage_index: int
    selected_nodes: np.ndarray
    action: TransitionAction
    elbow_summary: dict[str, Any]


@dataclass(frozen=True)
class PostGateRecoveryActionCandidate:
    recovery_policy: str
    source_action_type: str
    move_kind: str
    selected_nodes: np.ndarray
    action: TransitionAction


@dataclass(frozen=True)
class AlignedCoreBoundarySelection:
    """Node sets selected from a joint-bundle aligned-core frontier."""

    target_nodes: np.ndarray
    boundary_core_nodes: np.ndarray
    context_core_nodes: np.ndarray


def unique_sorted_u32(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    if arr.size == 0:
        return np.asarray([], dtype=np.uint32)
    return np.asarray(sorted(set(int(value) for value in arr)), dtype=np.uint32)


def intersect_sorted_u32(left: Any, right: Any) -> np.ndarray:
    left_arr = unique_sorted_u32(left)
    right_arr = unique_sorted_u32(right)
    if left_arr.size == 0 or right_arr.size == 0:
        return np.asarray([], dtype=np.uint32)
    return np.intersect1d(left_arr, right_arr, assume_unique=True).astype(
        np.uint32,
        copy=False,
    )


def setdiff_sorted_u32(left: Any, right: Any) -> np.ndarray:
    left_arr = unique_sorted_u32(left)
    right_arr = unique_sorted_u32(right)
    if left_arr.size == 0:
        return np.asarray([], dtype=np.uint32)
    if right_arr.size == 0:
        return left_arr
    return np.setdiff1d(left_arr, right_arr, assume_unique=True).astype(
        np.uint32,
        copy=False,
    )


def node_csv(nodes: Any) -> str:
    return ",".join(str(int(node)) for node in unique_sorted_u32(nodes))


def prefix_direct_nodes(units: pd.DataFrame, prefix_unit_ids: Any) -> np.ndarray:
    unit_ids = parse_unit_ids(prefix_unit_ids)
    if not unit_ids:
        return np.asarray([], dtype=np.uint32)
    units_by_id = {str(row["unit_id"]): row for _, row in units.iterrows()}
    nodes: list[int] = []
    for unit_id in unit_ids:
        if unit_id not in units_by_id:
            raise KeyError(f"Missing unit_id in units table: {unit_id}")
        nodes.extend(int(node) for node in parse_node_ids(units_by_id[unit_id]["node_ids"]))
    return unique_sorted_u32(nodes)


def weighted_pull_to_nodes(
    *,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    target_nodes: np.ndarray,
    node_count: int,
) -> np.ndarray:
    """Return incident edge weight from every node to ``target_nodes``."""
    scores = np.zeros(int(node_count), dtype=np.float64)
    targets = np.asarray(target_nodes, dtype=np.int64)
    if targets.size == 0:
        return scores
    target_mask = np.zeros(int(node_count), dtype=np.bool_)
    target_mask[targets] = True
    src_arr = np.asarray(src, dtype=np.int64)
    dst_arr = np.asarray(dst, dtype=np.int64)
    weights = np.asarray(weight, dtype=np.float64)
    src_hit = target_mask[src_arr]
    dst_hit = target_mask[dst_arr]
    np.add.at(scores, dst_arr[src_hit], weights[src_hit])
    np.add.at(scores, src_arr[dst_hit], weights[dst_hit])
    scores[targets] = 0.0
    return scores


def topk_by_pull(
    *,
    candidate_nodes: np.ndarray,
    pull_scores: np.ndarray,
    max_nodes: int,
) -> np.ndarray:
    candidates = unique_sorted_u32(candidate_nodes)
    if candidates.size == 0 or int(max_nodes) <= 0:
        return np.asarray([], dtype=np.uint32)
    scores = np.asarray(pull_scores, dtype=np.float64)[candidates.astype(np.int64)]
    frame = pd.DataFrame(
        {
            "node": candidates.astype(np.uint32),
            "score": scores,
        }
    )
    frame = frame.sort_values(["score", "node"], ascending=[False, True])
    return np.asarray(frame.head(int(max_nodes))["node"], dtype=np.uint32)


def select_aligned_core_boundary_nodes(
    frontier_rows: pd.DataFrame,
    *,
    min_target_change_count: int = 5,
    min_boundary_change_count: int = 5,
    max_context_core_nodes: int = 3,
) -> AlignedCoreBoundarySelection:
    """Select target and boundary roles from an aligned-core frontier table.

    The selector deliberately separates direct target handles from non-target
    aligned movers.  That keeps the next diagnostic from silently treating
    every useful polish-side movement as another target node.
    """
    if frontier_rows.empty:
        return AlignedCoreBoundarySelection(
            target_nodes=np.asarray([], dtype=np.uint32),
            boundary_core_nodes=np.asarray([], dtype=np.uint32),
            context_core_nodes=np.asarray([], dtype=np.uint32),
        )
    rows = frontier_rows.copy()
    required = {"node", "frontier_role", "aligned_change_count"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"frontier rows are missing required columns: {sorted(missing)}")
    rows["_aligned_change_count"] = pd.to_numeric(
        rows["aligned_change_count"],
        errors="coerce",
    ).fillna(0)
    rows["_max_pull_to_bundle"] = pd.to_numeric(
        rows.get("max_pull_to_bundle", 0.0),
        errors="coerce",
    ).fillna(0.0)
    rows["_context_count"] = pd.to_numeric(
        rows.get("context_count", 0),
        errors="coerce",
    ).fillna(0)
    rows["_source_mutable_count"] = pd.to_numeric(
        rows.get("source_mutable_count", 0),
        errors="coerce",
    ).fillna(0)

    target = rows[
        rows["frontier_role"].astype(str).eq("target_core")
        & (rows["_aligned_change_count"] >= int(min_target_change_count))
    ].sort_values(["_aligned_change_count", "node"], ascending=[False, True])
    boundary = rows[
        rows["frontier_role"].astype(str).isin({"source_mutable_core", "context_core"})
        & (rows["_aligned_change_count"] >= int(min_boundary_change_count))
    ].sort_values(
        ["_aligned_change_count", "_source_mutable_count", "_max_pull_to_bundle", "node"],
        ascending=[False, False, False, True],
    )
    context = rows[
        rows["frontier_role"].astype(str).eq("context_core")
        & (rows["_aligned_change_count"] > 0)
    ].sort_values(
        ["_aligned_change_count", "_context_count", "_max_pull_to_bundle", "node"],
        ascending=[False, False, False, True],
    )
    return AlignedCoreBoundarySelection(
        target_nodes=unique_sorted_u32(target["node"].to_numpy(dtype=np.uint32)),
        boundary_core_nodes=unique_sorted_u32(boundary["node"].to_numpy(dtype=np.uint32)),
        context_core_nodes=unique_sorted_u32(
            context.head(max(0, int(max_context_core_nodes)))["node"].to_numpy(
                dtype=np.uint32
            )
        ),
    )


def build_aligned_core_boundary_plan_rows(
    *,
    target_nodes: np.ndarray,
    boundary_core_nodes: np.ndarray,
    context_core_nodes: np.ndarray | None = None,
    candidate_context_by_cap: dict[int, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Build auditable plan rows for a compact aligned-core operator probe."""
    target = unique_sorted_u32(target_nodes)
    boundary = unique_sorted_u32(boundary_core_nodes)
    context_core = unique_sorted_u32(
        [] if context_core_nodes is None else context_core_nodes
    )
    candidate_by_cap = candidate_context_by_cap or {}
    rows: list[dict[str, Any]] = []

    def add_plan(
        *,
        plan_kind: str,
        context_nodes: np.ndarray,
        candidate_context_cap: int = 0,
    ) -> None:
        context = unique_sorted_u32(context_nodes)
        bundle = unique_sorted_u32(np.concatenate([target, context]))
        included_boundary = intersect_sorted_u32(context, boundary)
        included_context_core = intersect_sorted_u32(context, context_core)
        extra_candidate_context = (
            setdiff_sorted_u32(
                context,
                unique_sorted_u32(np.concatenate([boundary, context_core])),
            )
            if plan_kind == ALIGNED_CORE_PLAN_CANDIDATE_CONTEXT
            else np.asarray([], dtype=np.uint32)
        )
        rows.append(
            {
                "plan_rank": int(len(rows) + 1),
                "plan_kind": plan_kind,
                "candidate_context_cap": int(candidate_context_cap),
                "target_node_count": int(target.size),
                "target_node_ids": node_csv(target),
                "boundary_core_node_count": int(boundary.size),
                "boundary_core_node_ids": node_csv(boundary),
                "context_core_node_count": int(context_core.size),
                "context_core_node_ids": node_csv(context_core),
                "included_boundary_core_node_count": int(included_boundary.size),
                "included_boundary_core_node_ids": node_csv(included_boundary),
                "included_context_core_node_count": int(included_context_core.size),
                "included_context_core_node_ids": node_csv(included_context_core),
                "candidate_context_node_count": int(extra_candidate_context.size),
                "candidate_context_node_ids": node_csv(extra_candidate_context),
                "context_node_count": int(context.size),
                "context_node_ids": node_csv(context),
                "bundle_node_count": int(bundle.size),
                "bundle_node_ids": node_csv(bundle),
                "includes_boundary_core": bool(
                    intersect_sorted_u32(context, boundary).size == boundary.size
                    and boundary.size > 0
                ),
            }
        )

    add_plan(plan_kind=ALIGNED_CORE_PLAN_TARGET_ONLY, context_nodes=np.asarray([], dtype=np.uint32))
    if boundary.size:
        add_plan(plan_kind=ALIGNED_CORE_PLAN_BOUNDARY_CORE, context_nodes=boundary)
    if boundary.size and context_core.size:
        add_plan(
            plan_kind=ALIGNED_CORE_PLAN_CONTEXT_CORE,
            context_nodes=unique_sorted_u32(np.concatenate([boundary, context_core])),
        )
    for cap, nodes in sorted(candidate_by_cap.items()):
        candidate_context = unique_sorted_u32(nodes)
        if candidate_context.size == 0:
            continue
        add_plan(
            plan_kind=ALIGNED_CORE_PLAN_CANDIDATE_CONTEXT,
            context_nodes=unique_sorted_u32(np.concatenate([boundary, candidate_context])),
            candidate_context_cap=int(cap),
        )
    return pd.DataFrame(rows)


def build_aligned_core_handle_subset_plan_rows(
    *,
    target_nodes: np.ndarray,
    min_subset_size: int = 1,
    max_subset_size: int | None = None,
) -> pd.DataFrame:
    """Enumerate direct-handle subset plans for a sufficiency probe."""
    target = unique_sorted_u32(target_nodes)
    if target.size == 0:
        return pd.DataFrame()
    upper = int(target.size) if max_subset_size is None else int(max_subset_size)
    lower = max(1, int(min_subset_size))
    upper = max(lower, min(int(target.size), upper))
    rows: list[dict[str, Any]] = []
    for subset_size in range(lower, upper + 1):
        for subset in combinations([int(node) for node in target], subset_size):
            subset_nodes = unique_sorted_u32(subset)
            rows.append(
                {
                    "plan_rank": int(len(rows) + 1),
                    "plan_kind": "direct_handle_subset",
                    "subset_size": int(subset_nodes.size),
                    "target_node_count": int(target.size),
                    "full_target_node_ids": node_csv(target),
                    "subset_node_ids": node_csv(subset_nodes),
                    "bundle_node_count": int(subset_nodes.size),
                    "bundle_node_ids": node_csv(subset_nodes),
                }
            )
    return pd.DataFrame(rows)


def score_aligned_core_handle_nodes(
    frontier_rows: pd.DataFrame,
    *,
    selector_policy: str,
    min_target_change_count: int = 5,
) -> pd.DataFrame:
    """Rank direct aligned-core handles using inspectable feature policies.

    The policies are diagnostic selectors.  Some use replay-derived frontier
    counts, while the context-pull policy family is closer to a local graph
    proxy.  The returned table keeps those families explicit so reports do not
    confuse retrospective frontier evidence with an operator-ready rule.
    """
    if selector_policy not in ALIGNED_CORE_HANDLE_SELECTOR_POLICIES:
        raise ValueError(f"Unknown aligned-core handle selector: {selector_policy}")
    if frontier_rows.empty:
        return pd.DataFrame()
    required = {"node", "frontier_role", "aligned_change_count"}
    missing = required - set(frontier_rows.columns)
    if missing:
        raise ValueError(f"frontier rows are missing required columns: {sorted(missing)}")
    rows = frontier_rows.copy()
    numeric_defaults = {
        "aligned_change_count": 0,
        "selected_target_count": 0,
        "context_count": 0,
        "source_action_count": 0,
        "source_mutable_count": 0,
        "max_pull_to_target": 0.0,
        "max_pull_to_context": 0.0,
        "max_pull_to_bundle": 0.0,
        "aligned_change_fraction": 0.0,
    }
    for column, default in numeric_defaults.items():
        values = (
            rows[column]
            if column in rows.columns
            else pd.Series(default, index=rows.index)
        )
        rows[column] = pd.to_numeric(
            values,
            errors="coerce",
        ).fillna(default)
    rows = rows[
        rows["frontier_role"].astype(str).eq("target_core")
        & (rows["aligned_change_count"].astype(float) >= int(min_target_change_count))
    ].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["_source_mutable_free"] = rows["source_mutable_count"].astype(float).eq(0).astype(int)
    rows["_source_action_free"] = rows["source_action_count"].astype(float).eq(0).astype(int)
    if selector_policy == ALIGNED_CORE_HANDLE_SELECTOR_ALIGNED_FREQUENCY:
        sort_cols = [
            "aligned_change_count",
            "selected_target_count",
            "max_pull_to_context",
            "node",
        ]
        ascending = [False, False, False, True]
        feature_family = "replay_frontier"
    elif selector_policy == ALIGNED_CORE_HANDLE_SELECTOR_CONTEXT_PULL:
        sort_cols = [
            "max_pull_to_context",
            "max_pull_to_target",
            "aligned_change_count",
            "node",
        ]
        ascending = [False, False, False, True]
        feature_family = "local_graph_proxy"
    elif selector_policy == ALIGNED_CORE_HANDLE_SELECTOR_MUTABLE_PENALIZED_CONTEXT_PULL:
        sort_cols = [
            "_source_mutable_free",
            "_source_action_free",
            "max_pull_to_context",
            "aligned_change_count",
            "node",
        ]
        ascending = [False, False, False, False, True]
        feature_family = "local_graph_proxy"
    else:
        sort_cols = [
            "_source_mutable_free",
            "_source_action_free",
            "aligned_change_count",
            "selected_target_count",
            "max_pull_to_context",
            "node",
        ]
        ascending = [False, False, False, False, False, True]
        feature_family = "replay_frontier"
    ranked = rows.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    ranked["selector_policy"] = selector_policy
    ranked["selector_rank"] = np.arange(1, len(ranked) + 1, dtype=np.int64)
    ranked["selector_feature_family"] = feature_family
    ranked["selector_uses_replay_features"] = selector_policy in (
        ALIGNED_CORE_HANDLE_SELECTOR_REPLAY_POLICIES
    )
    keep = [
        "selector_policy",
        "selector_rank",
        "selector_feature_family",
        "selector_uses_replay_features",
        "node",
        "aligned_change_count",
        "selected_target_count",
        "context_count",
        "source_action_count",
        "source_mutable_count",
        "max_pull_to_target",
        "max_pull_to_context",
        "max_pull_to_bundle",
        "baseline_label",
        "vanilla_label",
        "candidate_label",
    ]
    return ranked[[column for column in keep if column in ranked]].copy()


def build_aligned_core_handle_selector_plan_rows(
    frontier_rows: pd.DataFrame,
    *,
    selector_policies: tuple[str, ...] = ALIGNED_CORE_HANDLE_SELECTOR_POLICIES,
    min_target_change_count: int = 5,
    min_subset_size: int = 1,
    max_subset_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build top-k selector plans and node score rows for handle diagnostics."""
    score_frames: list[pd.DataFrame] = []
    plan_rows: list[dict[str, Any]] = []
    plan_rank = 0
    for policy in selector_policies:
        scores = score_aligned_core_handle_nodes(
            frontier_rows,
            selector_policy=policy,
            min_target_change_count=min_target_change_count,
        )
        if scores.empty:
            continue
        score_frames.append(scores)
        upper = int(len(scores)) if max_subset_size is None else int(max_subset_size)
        lower = max(1, int(min_subset_size))
        upper = max(lower, min(int(len(scores)), upper))
        ordered_nodes = [int(node) for node in scores["node"]]
        for subset_size in range(lower, upper + 1):
            selected = unique_sorted_u32(ordered_nodes[:subset_size])
            plan_rank += 1
            plan_rows.append(
                {
                    "plan_rank": int(plan_rank),
                    "selector_policy": policy,
                    "selector_feature_family": str(
                        scores.iloc[0]["selector_feature_family"]
                    ),
                    "selector_uses_replay_features": bool(
                        scores.iloc[0]["selector_uses_replay_features"]
                    ),
                    "subset_size": int(selected.size),
                    "selector_ordered_node_ids": ",".join(
                        str(int(node)) for node in ordered_nodes[:subset_size]
                    ),
                    "subset_node_ids": node_csv(selected),
                    "available_handle_count": int(len(scores)),
                }
            )
    score_rows = (
        pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    )
    return pd.DataFrame(plan_rows), score_rows


def score_local_handle_candidates(
    score_rows: pd.DataFrame,
    *,
    selector_policy: str,
    source_case: str | None = None,
) -> pd.DataFrame:
    """Rank handle candidates from source-local graph feature rows.

    Unlike :func:`score_aligned_core_handle_nodes`, this helper does not use
    aligned frontier replay counts.  It consumes rows such as the attachment
    margin score table, where each source case has local pull/margin features
    computed before the selector knows the aligned-core endpoint.
    """
    if selector_policy not in LOCAL_HANDLE_SELECTOR_POLICIES:
        raise ValueError(f"Unknown local handle selector: {selector_policy}")
    if score_rows.empty:
        return pd.DataFrame()
    rows = score_rows.copy()
    if source_case is not None:
        rows = rows[rows["source_case"].astype(str).eq(str(source_case))].copy()
    required = {"node"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"local score rows are missing required columns: {sorted(missing)}")
    if rows.empty:
        return pd.DataFrame()
    numeric_defaults = {
        "gate_pull_margin_vs_current_source": 0.0,
        "gate_pull_share_vs_current_source": 0.0,
        "gate_pull_margin_vs_source_action": 0.0,
        "pull_to_gate_context": 0.0,
        "mean_edge_weight_to_gate_context": 0.0,
        "pull_to_current_source_label": 0.0,
        "rank_mean_consensus": math.inf,
        "rank_best_consensus": math.inf,
    }
    for column, default in numeric_defaults.items():
        values = (
            rows[column]
            if column in rows.columns
            else pd.Series(default, index=rows.index)
        )
        rows[column] = pd.to_numeric(values, errors="coerce").fillna(default)
    for column in ("in_source_action", "in_source_mutable", "in_direct_nodes"):
        values = (
            rows[column]
            if column in rows.columns
            else pd.Series(False, index=rows.index)
        )
        rows[column] = values.map(
            lambda value: str(value).strip().lower() in {"true", "1", "yes"}
        )
    rows["_not_source_action"] = (~rows["in_source_action"]).astype(int)
    rows["_not_source_mutable"] = (~rows["in_source_mutable"]).astype(int)

    selected_label: int | None = None
    if selector_policy == LOCAL_HANDLE_SELECTOR_CANDIDATE_LABEL_MARGIN_COHERENT:
        if "candidate_label" not in rows.columns:
            raise ValueError(
                "candidate_label_margin_coherent requires a candidate_label column"
            )
        candidates = rows[~rows["in_source_mutable"]].copy()
        if candidates.empty:
            candidates = rows.copy()
        label_scores: list[dict[str, Any]] = []
        for label, group in candidates.groupby("candidate_label", sort=False):
            positive_group = group[
                group["gate_pull_margin_vs_current_source"].astype(float) > 0.0
            ]
            scoring_group = positive_group if not positive_group.empty else group
            ordered = scoring_group.sort_values(
                [
                    "gate_pull_margin_vs_current_source",
                    "pull_to_gate_context",
                    "node",
                ],
                ascending=[False, False, True],
            ).head(4)
            label_scores.append(
                {
                    "candidate_label": label,
                    "positive_node_count": int(len(positive_group)),
                    "top4_margin_sum": float(
                        ordered["gate_pull_margin_vs_current_source"].sum()
                    ),
                    "top4_pull_sum": float(ordered["pull_to_gate_context"].sum()),
                    "node_count": int(len(group)),
                }
            )
        if label_scores:
            label_frame = pd.DataFrame(label_scores).sort_values(
                [
                    "positive_node_count",
                    "top4_margin_sum",
                    "top4_pull_sum",
                    "node_count",
                    "candidate_label",
                ],
                ascending=[False, False, False, False, True],
            )
            selected_label = int(label_frame.iloc[0]["candidate_label"])
            rows = candidates[
                candidates["candidate_label"].astype(int).eq(selected_label)
            ].copy()
        sort_cols = [
            "gate_pull_margin_vs_current_source",
            "pull_to_gate_context",
            "node",
        ]
        ascending = [False, False, True]
    elif selector_policy == LOCAL_HANDLE_SELECTOR_ATTACHMENT_MARGIN:
        sort_cols = [
            "gate_pull_margin_vs_current_source",
            "gate_pull_share_vs_current_source",
            "pull_to_gate_context",
            "node",
        ]
        ascending = [False, False, False, True]
    elif selector_policy == LOCAL_HANDLE_SELECTOR_GATE_PULL:
        sort_cols = [
            "pull_to_gate_context",
            "mean_edge_weight_to_gate_context",
            "gate_pull_margin_vs_current_source",
            "node",
        ]
        ascending = [False, False, False, True]
    elif selector_policy == LOCAL_HANDLE_SELECTOR_NON_SOURCE_GATE_PULL:
        sort_cols = [
            "_not_source_mutable",
            "_not_source_action",
            "pull_to_gate_context",
            "gate_pull_margin_vs_current_source",
            "node",
        ]
        ascending = [False, False, False, False, True]
    else:
        sort_cols = [
            "rank_best_consensus",
            "rank_mean_consensus",
            "gate_pull_margin_vs_current_source",
            "node",
        ]
        ascending = [True, True, False, True]

    ranked = rows.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    ranked["selector_policy"] = selector_policy
    ranked["selector_rank"] = np.arange(1, len(ranked) + 1, dtype=np.int64)
    ranked["selector_feature_family"] = "local_graph_proxy"
    ranked["selector_uses_replay_features"] = False
    ranked["selector_candidate_label"] = (
        "" if selected_label is None else str(int(selected_label))
    )
    keep = [
        "source_case",
        "selector_policy",
        "selector_rank",
        "selector_feature_family",
        "selector_uses_replay_features",
        "selector_candidate_label",
        "node",
        "gate_pull_margin_vs_current_source",
        "gate_pull_share_vs_current_source",
        "gate_pull_margin_vs_source_action",
        "pull_to_gate_context",
        "mean_edge_weight_to_gate_context",
        "pull_to_current_source_label",
        "in_source_action",
        "in_source_mutable",
        "in_direct_nodes",
        "baseline_label",
        "candidate_label",
        "vanilla_label",
        "source_label",
    ]
    return ranked[[column for column in keep if column in ranked]].copy()


def build_local_handle_selector_plan_rows(
    score_rows: pd.DataFrame,
    *,
    selector_policies: tuple[str, ...] = LOCAL_HANDLE_SELECTOR_POLICIES,
    selected_ks: tuple[int, ...] = (1, 2, 3, 4, 5, 8),
    source_case: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build source-local top-k selector plans plus node score rows."""
    score_frames: list[pd.DataFrame] = []
    plan_rows: list[dict[str, Any]] = []
    plan_rank = 0
    for policy in selector_policies:
        scores = score_local_handle_candidates(
            score_rows,
            selector_policy=policy,
            source_case=source_case,
        )
        if scores.empty:
            continue
        score_frames.append(scores)
        ordered_nodes = [int(node) for node in scores["node"]]
        for selected_k in sorted(set(int(k) for k in selected_ks if int(k) > 0)):
            if selected_k > len(ordered_nodes):
                continue
            selected_ordered = ordered_nodes[:selected_k]
            selected = unique_sorted_u32(selected_ordered)
            plan_rank += 1
            plan_rows.append(
                {
                    "plan_rank": int(plan_rank),
                    "source_case": (
                        str(source_case)
                        if source_case is not None
                        else str(scores.iloc[0].get("source_case", ""))
                    ),
                    "plan_kind": "local_handle_selector",
                    "selector_policy": policy,
                    "selector_feature_family": "local_graph_proxy",
                    "selector_uses_replay_features": False,
                    "selector_candidate_label": str(
                        scores.iloc[0].get("selector_candidate_label", "")
                    ),
                    "selected_k": int(selected_k),
                    "subset_size": int(selected.size),
                    "target_node_count": int(len(scores)),
                    "full_target_node_ids": node_csv(ordered_nodes),
                    "selector_ordered_node_ids": ",".join(
                        str(int(node)) for node in selected_ordered
                    ),
                    "subset_node_ids": node_csv(selected),
                    "bundle_node_count": int(selected.size),
                    "bundle_node_ids": node_csv(selected),
                }
            )
    score_out = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    return pd.DataFrame(plan_rows), score_out


def summarize_local_selector_readiness_rows(
    score_rows: pd.DataFrame,
    *,
    source_summary_rows: pd.DataFrame | None = None,
    min_positive_margin_nodes: int = 2,
    min_positive_margin_non_source_nodes: int = 2,
    min_positive_margin_candidate_labels: int = 2,
    min_source_support_distance: float = 0.01,
    recovered_quality_threshold: float = 0.01,
    recovered_support_threshold: float = 0.05,
) -> pd.DataFrame:
    """Summarize whether source-local scores can really test a selector.

    A useful selector-readiness row is not a success claim.  It identifies
    cases where a local selector has enough non-source positive-margin handles
    and label competition to be meaningfully tested, while source rows that
    already recovered QF/support are marked as controls.
    """
    def label_text(value: Any) -> str:
        if pd.isna(value):
            return ""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        return str(int(number)) if number.is_integer() else str(value)

    if score_rows.empty:
        return pd.DataFrame()
    rows = score_rows.copy()
    if "source_case" not in rows.columns:
        rows["source_case"] = ""
    for column in (
        "gate_pull_margin_vs_current_source",
        "pull_to_gate_context",
        "pull_to_current_source_label",
    ):
        values = rows[column] if column in rows.columns else pd.Series(0.0, index=rows.index)
        rows[column] = pd.to_numeric(values, errors="coerce").fillna(0.0)
    for column in ("in_source_action", "in_source_mutable", "in_direct_nodes"):
        values = rows[column] if column in rows.columns else pd.Series(False, index=rows.index)
        rows[column] = values.map(
            lambda value: str(value).strip().lower() in {"true", "1", "yes"}
        )
    if "candidate_label" not in rows.columns:
        rows["candidate_label"] = ""
    summaries = (
        source_summary_rows.copy()
        if source_summary_rows is not None and not source_summary_rows.empty
        else pd.DataFrame()
    )
    summary_by_case = {
        str(row["source_case"]): row for _, row in summaries.iterrows()
    } if "source_case" in summaries.columns else {}

    out: list[dict[str, Any]] = []
    for source_case, group in rows.groupby("source_case", sort=True):
        group = group.copy()
        positive = group[group["gate_pull_margin_vs_current_source"].astype(float) > 0.0]
        non_source = positive[
            ~positive["in_source_action"].astype(bool)
            & ~positive["in_source_mutable"].astype(bool)
        ].copy()
        candidate_pool = group[
            ~group["in_source_action"].astype(bool)
            & ~group["in_source_mutable"].astype(bool)
        ].copy()
        label_scores: list[dict[str, Any]] = []
        for label, label_group in non_source.groupby("candidate_label", sort=False):
            full_label_group = candidate_pool[
                candidate_pool["candidate_label"].astype(str).eq(str(label))
            ]
            label_scores.append(
                {
                    "candidate_label": label,
                    "positive_margin_sum": float(
                        label_group["gate_pull_margin_vs_current_source"].sum()
                    ),
                    "positive_pull_sum": float(label_group["pull_to_gate_context"].sum()),
                    "positive_node_count": int(len(label_group)),
                    "node_count": int(len(full_label_group)),
                }
            )
        label_frame = (
            pd.DataFrame(label_scores).sort_values(
                [
                    "positive_margin_sum",
                    "positive_pull_sum",
                    "positive_node_count",
                    "candidate_label",
                ],
                ascending=[False, False, False, True],
            )
            if label_scores
            else pd.DataFrame()
        )
        source_summary = summary_by_case.get(str(source_case))

        def source_float(column: str, default: float = math.nan) -> float:
            if source_summary is None or column not in source_summary.index:
                return default
            value = pd.to_numeric(pd.Series([source_summary[column]]), errors="coerce").iloc[0]
            return float(value) if not pd.isna(value) else default

        def source_text(column: str) -> str:
            if source_summary is None or column not in source_summary.index:
                return ""
            value = source_summary[column]
            return "" if pd.isna(value) else str(value)

        source_delta = source_float("source_delta_q_vs_start")
        source_support = source_float("source_support_distance_to_vanilla")
        source_progress = source_float("source_target_progress_from_vanilla")
        has_source = source_summary is not None
        already_recovered = bool(
            has_source
            and source_delta >= float(recovered_quality_threshold)
            and source_support >= float(recovered_support_threshold)
        )
        top = label_frame.iloc[0] if not label_frame.empty else {}
        second = label_frame.iloc[1] if len(label_frame) > 1 else {}
        top_margin = float(top.get("positive_margin_sum", math.nan))
        second_margin = float(second.get("positive_margin_sum", math.nan))
        top_label_node_count = int(top.get("node_count", 0))
        if not has_source:
            verdict = LOCAL_SELECTOR_READINESS_MISSING_SOURCE
        elif already_recovered:
            verdict = LOCAL_SELECTOR_READINESS_ALREADY_RECOVERED
        elif source_support < float(min_source_support_distance):
            verdict = LOCAL_SELECTOR_READINESS_WEAK_CANDIDATE_DIRECTION
        elif (
            non_source.shape[0] >= int(min_positive_margin_non_source_nodes)
            and label_frame.shape[0] >= int(min_positive_margin_candidate_labels)
        ):
            verdict = LOCAL_SELECTOR_READINESS_READY
        elif non_source.shape[0] >= 1 and top_label_node_count >= 4:
            verdict = LOCAL_SELECTOR_READINESS_LABEL_COMPLETION
        elif positive.shape[0] < int(min_positive_margin_nodes):
            verdict = LOCAL_SELECTOR_READINESS_TOO_FEW_HANDLES
        elif non_source.shape[0] < int(min_positive_margin_non_source_nodes):
            verdict = LOCAL_SELECTOR_READINESS_TOO_FEW_HANDLES
        elif label_frame.shape[0] < int(min_positive_margin_candidate_labels):
            verdict = LOCAL_SELECTOR_READINESS_NO_LABEL_COMPETITION
        else:
            verdict = LOCAL_SELECTOR_READINESS_READY

        best_non_source = non_source.sort_values(
            ["gate_pull_margin_vs_current_source", "pull_to_gate_context", "node"],
            ascending=[False, False, True],
        ).head(1)
        out.append(
            {
                "source_case": str(source_case),
                "readiness_verdict": verdict,
                "has_source_summary": bool(has_source),
                "already_recovered": bool(already_recovered),
                "score_row_count": int(len(group)),
                "positive_margin_node_count": int(len(positive)),
                "positive_margin_non_source_count": int(len(non_source)),
                "positive_margin_candidate_label_count": int(len(label_frame)),
                "top_candidate_label": (
                    label_text(top.get("candidate_label", ""))
                    if len(label_frame)
                    else ""
                ),
                "top_label_positive_margin_sum": top_margin,
                "top_label_positive_pull_sum": float(
                    top.get("positive_pull_sum", math.nan)
                ),
                "top_label_positive_node_count": int(
                    top.get("positive_node_count", 0)
                ),
                "top_label_node_count": top_label_node_count,
                "second_label_positive_margin_sum": second_margin,
                "label_competition_gap": (
                    top_margin - second_margin
                    if not math.isnan(top_margin) and not math.isnan(second_margin)
                    else math.nan
                ),
                "best_non_source_node": (
                    int(best_non_source.iloc[0]["node"])
                    if not best_non_source.empty
                    else -1
                ),
                "best_non_source_margin": (
                    float(best_non_source.iloc[0]["gate_pull_margin_vs_current_source"])
                    if not best_non_source.empty
                    else math.nan
                ),
                "best_non_source_pull": (
                    float(best_non_source.iloc[0]["pull_to_gate_context"])
                    if not best_non_source.empty
                    else math.nan
                ),
                "source_delta_q_vs_start": source_delta,
                "source_support_distance_to_vanilla": source_support,
                "source_target_progress_from_vanilla": source_progress,
                "prefix_rank": source_float("prefix_rank"),
                "source_recovery_index": source_float("source_recovery_index"),
                "source_move_dir": source_text("source_move_dir"),
                "min_positive_margin_nodes": int(min_positive_margin_nodes),
                "min_positive_margin_non_source_nodes": int(
                    min_positive_margin_non_source_nodes
                ),
                "min_positive_margin_candidate_labels": int(
                    min_positive_margin_candidate_labels
                ),
                "min_source_support_distance": float(min_source_support_distance),
                "recovered_quality_threshold": float(recovered_quality_threshold),
                "recovered_support_threshold": float(recovered_support_threshold),
            }
        )
    return pd.DataFrame(out)


def transplant_action_nodes(
    *,
    membership: np.ndarray,
    donor_membership: np.ndarray,
    action_nodes: np.ndarray,
    reference_nodes: np.ndarray | None = None,
) -> np.ndarray:
    """Apply donor labels to action nodes, reusing reference-label mappings."""
    out = np.asarray(membership, dtype=np.uint64).copy()
    donor = np.asarray(donor_membership, dtype=np.uint64)
    action = unique_sorted_u32(action_nodes)
    if action.size == 0:
        return out
    mapping: dict[int, int] = {}
    reference = unique_sorted_u32([] if reference_nodes is None else reference_nodes)
    if reference.size:
        frame = pd.DataFrame(
            {
                "donor": donor[reference.astype(np.int64)].astype(np.int64),
                "current": out[reference.astype(np.int64)].astype(np.int64),
            }
        )
        counts = (
            frame.groupby(["donor", "current"], sort=False)
            .size()
            .reset_index(name="count")
        )
        for donor_label, group in counts.groupby("donor", sort=False):
            best = group.sort_values(["count", "current"], ascending=[False, True]).iloc[0]
            mapping[int(donor_label)] = int(best["current"])
    next_label = int(out.max(initial=0)) + 1
    for node in action.astype(np.int64):
        donor_label = int(donor[int(node)])
        target = mapping.get(donor_label)
        if target is None:
            target = next_label
            mapping[donor_label] = target
            next_label += 1
        out[int(node)] = np.uint64(target)
    return out


def target_edge_support_rows(
    *,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    target_nodes: np.ndarray,
) -> pd.DataFrame:
    """Return target-induced edge rows with common-neighbor support counts."""
    targets = unique_sorted_u32(target_nodes)
    if targets.size == 0:
        return pd.DataFrame(
            columns=["src", "dst", "edge_weight", "edge_support"]
        )
    target_set = set(int(node) for node in targets)
    pair_weight: dict[tuple[int, int], float] = {}
    adjacency: dict[int, set[int]] = {int(node): set() for node in targets}
    for left, right, edge_weight in zip(src, dst, weight, strict=False):
        u = int(left)
        v = int(right)
        if u == v or u not in target_set or v not in target_set:
            continue
        a, b = (u, v) if u < v else (v, u)
        pair_weight[(a, b)] = pair_weight.get((a, b), 0.0) + float(edge_weight)
        adjacency[a].add(b)
        adjacency[b].add(a)
    rows: list[dict[str, Any]] = []
    for (u, v), edge_weight in sorted(pair_weight.items()):
        rows.append(
            {
                "src": int(u),
                "dst": int(v),
                "edge_weight": float(edge_weight),
                "edge_support": int(len(adjacency[u] & adjacency[v])),
            }
        )
    return pd.DataFrame(rows)


def components_from_edges(
    *,
    nodes: np.ndarray,
    edges: pd.DataFrame,
) -> list[np.ndarray]:
    """Return connected components over ``nodes`` using edge rows with src/dst."""
    node_arr = unique_sorted_u32(nodes)
    adjacency: dict[int, set[int]] = {int(node): set() for node in node_arr}
    if not edges.empty:
        for _, edge in edges.iterrows():
            u = int(edge["src"])
            v = int(edge["dst"])
            if u in adjacency and v in adjacency:
                adjacency[u].add(v)
                adjacency[v].add(u)
    seen: set[int] = set()
    components: list[np.ndarray] = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(np.asarray(sorted(component), dtype=np.uint32))
    components.sort(key=lambda values: (-int(values.size), int(values[0]) if values.size else -1))
    return components


def _label_closure_stats(
    *,
    membership: np.ndarray,
    nodes: np.ndarray,
    prefix: str,
) -> dict[str, Any]:
    labels = np.asarray(membership, dtype=np.uint64)
    node_arr = unique_sorted_u32(nodes)
    if node_arr.size == 0:
        return {
            f"{prefix}_label_count": 0,
            f"{prefix}_closure_node_count": 0,
            f"{prefix}_closure_extra_count": 0,
            f"{prefix}_closure_ratio": math.nan,
        }
    touched = np.unique(labels[node_arr.astype(np.int64)])
    closure_count = int(np.count_nonzero(np.isin(labels, touched)))
    extra = max(0, closure_count - int(node_arr.size))
    return {
        f"{prefix}_label_count": int(touched.size),
        f"{prefix}_closure_node_count": closure_count,
        f"{prefix}_closure_extra_count": int(extra),
        f"{prefix}_closure_ratio": float(extra) / float(max(1, int(node_arr.size))),
    }


def _unit_graph_metrics(
    *,
    nodes: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    node_count: int,
    target_nodes: np.ndarray,
    target_edge_support: pd.DataFrame,
    candidate_pull_scores: np.ndarray,
) -> dict[str, Any]:
    unit = unique_sorted_u32(nodes)
    unit_size = int(unit.size)
    possible_edges = unit_size * (unit_size - 1) / 2.0
    unit_mask = np.zeros(int(node_count), dtype=np.bool_)
    target_mask = np.zeros(int(node_count), dtype=np.bool_)
    if unit_size:
        unit_mask[unit.astype(np.int64)] = True
    target = unique_sorted_u32(target_nodes)
    if target.size:
        target_mask[target.astype(np.int64)] = True
    src_arr = np.asarray(src, dtype=np.int64)
    dst_arr = np.asarray(dst, dtype=np.int64)
    weights = np.asarray(weight, dtype=np.float64)
    src_unit = unit_mask[src_arr]
    dst_unit = unit_mask[dst_arr]
    internal = src_unit & dst_unit
    boundary = src_unit ^ dst_unit
    target_boundary = (
        (src_unit & target_mask[dst_arr] & ~dst_unit)
        | (dst_unit & target_mask[src_arr] & ~src_unit)
    )
    edge_supports: list[int] = []
    if not target_edge_support.empty and unit_size:
        unit_set = set(int(node) for node in unit)
        for _, edge in target_edge_support.iterrows():
            if int(edge["src"]) in unit_set and int(edge["dst"]) in unit_set:
                edge_supports.append(int(edge["edge_support"]))
    internal_weight = float(weights[internal].sum())
    boundary_weight = float(weights[boundary].sum())
    conductance_denom = 2.0 * internal_weight + boundary_weight
    candidate_pull = (
        float(np.asarray(candidate_pull_scores, dtype=np.float64)[unit.astype(np.int64)].sum())
        if unit_size
        else 0.0
    )
    return {
        "unit_node_count": unit_size,
        "unit_internal_edge_count": int(np.count_nonzero(internal)),
        "unit_boundary_edge_count": int(np.count_nonzero(boundary)),
        "unit_target_boundary_edge_count": int(np.count_nonzero(target_boundary)),
        "unit_internal_weight": internal_weight,
        "unit_boundary_weight": boundary_weight,
        "unit_target_boundary_weight": float(weights[target_boundary].sum()),
        "unit_density": (
            float(np.count_nonzero(internal)) / possible_edges
            if possible_edges > 0.0
            else 0.0
        ),
        "unit_conductance": (
            boundary_weight / conductance_denom
            if conductance_denom > 0.0
            else math.nan
        ),
        "triangle_edge_fraction": (
            float(sum(1 for support in edge_supports if support > 0))
            / float(len(edge_supports))
            if edge_supports
            else 0.0
        ),
        "mean_edge_support": (
            float(np.mean(edge_supports)) if edge_supports else 0.0
        ),
        "max_edge_support": int(max(edge_supports)) if edge_supports else 0,
        "pull_to_candidate_support_weight": candidate_pull,
        "pull_to_candidate_support_mean": (
            candidate_pull / float(unit_size) if unit_size else 0.0
        ),
    }


def _target_unit_row(
    *,
    unit_type: str,
    unit_index: int,
    nodes: np.ndarray,
    baseline_membership: np.ndarray,
    candidate_membership: np.ndarray,
    vanilla_membership: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    node_count: int,
    target_nodes: np.ndarray,
    target_edge_support: pd.DataFrame,
    candidate_pull_scores: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    unit = unique_sorted_u32(nodes)
    graph_metrics = _unit_graph_metrics(
        nodes=unit,
        src=src,
        dst=dst,
        weight=weight,
        node_count=node_count,
        target_nodes=target_nodes,
        target_edge_support=target_edge_support,
        candidate_pull_scores=candidate_pull_scores,
    )
    return {
        "unit_type": unit_type,
        "unit_id": f"{unit_type}_{int(unit_index):05d}",
        "node_ids": node_csv(unit),
        "target_node_count": int(unique_sorted_u32(target_nodes).size),
        "unit_target_fraction": (
            float(unit.size) / float(max(1, int(unique_sorted_u32(target_nodes).size)))
        ),
        **graph_metrics,
        **_label_closure_stats(
            membership=baseline_membership,
            nodes=unit,
            prefix="baseline",
        ),
        **_label_closure_stats(
            membership=candidate_membership,
            nodes=unit,
            prefix="candidate",
        ),
        **_label_closure_stats(
            membership=vanilla_membership,
            nodes=unit,
            prefix="vanilla",
        ),
        **(extra or {}),
    }


def build_target_unit_rows(
    *,
    target_nodes: np.ndarray,
    candidate_support_nodes: np.ndarray,
    baseline_membership: np.ndarray,
    candidate_membership: np.ndarray,
    vanilla_membership: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    node_count: int,
    unit_types: tuple[str, ...] = TARGET_UNIT_TYPES,
    triangle_support_min: int = 1,
) -> pd.DataFrame:
    """Build simple, auditable unit candidates over a target node set."""
    target = unique_sorted_u32(target_nodes)
    if target.size == 0:
        return pd.DataFrame()
    target_edges = target_edge_support_rows(
        src=src,
        dst=dst,
        weight=weight,
        target_nodes=target,
    )
    candidate_pull_scores = weighted_pull_to_nodes(
        src=src,
        dst=dst,
        weight=weight,
        target_nodes=candidate_support_nodes,
        node_count=node_count,
    )
    rows: list[dict[str, Any]] = []
    if TARGET_UNIT_LABEL_INTERSECTION_BLOCK in unit_types:
        labels = pd.DataFrame(
            {
                "node": target.astype(np.uint32),
                "baseline_label": np.asarray(baseline_membership, dtype=np.uint64)[
                    target.astype(np.int64)
                ].astype(np.int64),
                "candidate_label": np.asarray(candidate_membership, dtype=np.uint64)[
                    target.astype(np.int64)
                ].astype(np.int64),
                "vanilla_label": np.asarray(vanilla_membership, dtype=np.uint64)[
                    target.astype(np.int64)
                ].astype(np.int64),
            }
        )
        grouped = labels.groupby(
            ["baseline_label", "candidate_label", "vanilla_label"],
            sort=True,
        )
        blocks = [
            (
                key,
                unique_sorted_u32(group["node"].to_numpy(dtype=np.uint32)),
            )
            for key, group in grouped
        ]
        blocks.sort(key=lambda item: (-int(item[1].size), int(item[1][0])))
        for index, (key, nodes) in enumerate(blocks):
            baseline_label, candidate_label, vanilla_label = key
            rows.append(
                _target_unit_row(
                    unit_type=TARGET_UNIT_LABEL_INTERSECTION_BLOCK,
                    unit_index=index,
                    nodes=nodes,
                    baseline_membership=baseline_membership,
                    candidate_membership=candidate_membership,
                    vanilla_membership=vanilla_membership,
                    src=src,
                    dst=dst,
                    weight=weight,
                    node_count=node_count,
                    target_nodes=target,
                    target_edge_support=target_edges,
                    candidate_pull_scores=candidate_pull_scores,
                    extra={
                        "baseline_label": int(baseline_label),
                        "candidate_label": int(candidate_label),
                        "vanilla_label": int(vanilla_label),
                    },
                )
            )
    if TARGET_UNIT_CONNECTED_COMPONENT in unit_types:
        for index, nodes in enumerate(
            components_from_edges(nodes=target, edges=target_edges)
        ):
            rows.append(
                _target_unit_row(
                    unit_type=TARGET_UNIT_CONNECTED_COMPONENT,
                    unit_index=index,
                    nodes=nodes,
                    baseline_membership=baseline_membership,
                    candidate_membership=candidate_membership,
                    vanilla_membership=vanilla_membership,
                    src=src,
                    dst=dst,
                    weight=weight,
                    node_count=node_count,
                    target_nodes=target,
                    target_edge_support=target_edges,
                    candidate_pull_scores=candidate_pull_scores,
                )
            )
    if TARGET_UNIT_TRIANGLE_SUPPORTED_COMPONENT in unit_types:
        supported_edges = (
            target_edges[target_edges["edge_support"] >= int(triangle_support_min)]
            if not target_edges.empty
            else target_edges
        )
        for index, nodes in enumerate(
            components_from_edges(nodes=target, edges=supported_edges)
        ):
            rows.append(
                _target_unit_row(
                    unit_type=TARGET_UNIT_TRIANGLE_SUPPORTED_COMPONENT,
                    unit_index=index,
                    nodes=nodes,
                    baseline_membership=baseline_membership,
                    candidate_membership=candidate_membership,
                    vanilla_membership=vanilla_membership,
                    src=src,
                    dst=dst,
                    weight=weight,
                    node_count=node_count,
                    target_nodes=target,
                    target_edge_support=target_edges,
                    candidate_pull_scores=candidate_pull_scores,
                    extra={"triangle_support_min": int(triangle_support_min)},
                )
            )
    return pd.DataFrame(rows)


def cap_context_count(
    *,
    direct_node_count: int,
    context_multiplier: float,
    max_context_nodes: int,
) -> int:
    scaled = int(math.ceil(max(1, int(direct_node_count)) * float(context_multiplier)))
    return max(0, min(int(max_context_nodes), scaled))


def label_closure_context_nodes(
    *,
    membership: np.ndarray,
    direct_nodes: np.ndarray,
    exclude_nodes: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(membership, dtype=np.uint64)
    direct = np.asarray(direct_nodes, dtype=np.int64)
    if direct.size == 0:
        return np.asarray([], dtype=np.uint32)
    touched_labels = set(int(label) for label in labels[direct])
    if not touched_labels:
        return np.asarray([], dtype=np.uint32)
    label_mask = np.isin(labels, np.asarray(sorted(touched_labels), dtype=np.uint64))
    excluded = np.zeros(labels.shape[0], dtype=np.bool_)
    excluded[np.asarray(exclude_nodes, dtype=np.int64)] = True
    nodes = np.flatnonzero(label_mask & ~excluded)
    return np.asarray(nodes, dtype=np.uint32)


def boundary_shell_context_nodes(
    *,
    src: np.ndarray,
    dst: np.ndarray,
    direct_nodes: np.ndarray,
    exclude_nodes: np.ndarray,
    node_count: int,
) -> np.ndarray:
    direct = np.asarray(direct_nodes, dtype=np.int64)
    if direct.size == 0:
        return np.asarray([], dtype=np.uint32)
    direct_mask = np.zeros(int(node_count), dtype=np.bool_)
    direct_mask[direct] = True
    src_arr = np.asarray(src, dtype=np.int64)
    dst_arr = np.asarray(dst, dtype=np.int64)
    neighbors = np.concatenate([dst_arr[direct_mask[src_arr]], src_arr[direct_mask[dst_arr]]])
    if neighbors.size == 0:
        return np.asarray([], dtype=np.uint32)
    excluded = np.zeros(int(node_count), dtype=np.bool_)
    excluded[np.asarray(exclude_nodes, dtype=np.int64)] = True
    neighbors = unique_sorted_u32(neighbors)
    return neighbors[~excluded[neighbors.astype(np.int64)]]


def build_context_actions(
    *,
    state: TransitionSearchState,
    candidate_membership: np.ndarray,
    vanilla_membership: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    node_count: int,
    action_types: tuple[str, ...],
    context_multiplier: float,
    max_context_nodes: int,
) -> list[TransitionAction]:
    """Build next context-expansion actions for one search state."""
    action_nodes = unique_sorted_u32(state.action_nodes)
    direct_nodes = action_nodes if action_nodes.size else unique_sorted_u32(state.direct_nodes)
    exclude = unique_sorted_u32(np.concatenate([state.mutable_nodes, state.context_nodes]))
    cap = cap_context_count(
        direct_node_count=int(direct_nodes.size),
        context_multiplier=context_multiplier,
        max_context_nodes=max_context_nodes,
    )
    if cap <= 0:
        return []
    pull = weighted_pull_to_nodes(
        src=src,
        dst=dst,
        weight=weight,
        target_nodes=direct_nodes,
        node_count=node_count,
    )
    actions: list[TransitionAction] = []
    for action_type in action_types:
        if action_type == ACTION_CANDIDATE_CLOSURE_TOPK:
            candidates = label_closure_context_nodes(
                membership=candidate_membership,
                direct_nodes=direct_nodes,
                exclude_nodes=exclude,
            )
            context = topk_by_pull(
                candidate_nodes=candidates,
                pull_scores=pull,
                max_nodes=cap,
            )
        elif action_type == ACTION_VANILLA_CLOSURE_TOPK:
            candidates = label_closure_context_nodes(
                membership=vanilla_membership,
                direct_nodes=direct_nodes,
                exclude_nodes=exclude,
            )
            context = topk_by_pull(
                candidate_nodes=candidates,
                pull_scores=pull,
                max_nodes=cap,
            )
        elif action_type == ACTION_BOUNDARY_SHELL_TOPK:
            candidates = boundary_shell_context_nodes(
                src=src,
                dst=dst,
                direct_nodes=direct_nodes,
                exclude_nodes=exclude,
                node_count=node_count,
            )
            context = topk_by_pull(
                candidate_nodes=candidates,
                pull_scores=pull,
                max_nodes=cap,
            )
        else:
            raise ValueError(f"Unsupported transition action type: {action_type}")
        if context.size:
            actions.append(
                TransitionAction(
                    action_type=action_type,
                    action_params=(
                        f"context_multiplier={float(context_multiplier):g};"
                        f"max_context_nodes={int(max_context_nodes)}"
                    ),
                    context_nodes=context,
                )
            )
    return actions


def build_post_gate_recovery_actions(
    *,
    state: TransitionSearchState,
    candidate_membership: np.ndarray,
    vanilla_membership: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    node_count: int,
    action_types: tuple[str, ...] = (
        ACTION_CANDIDATE_CLOSURE_TOPK,
        ACTION_VANILLA_CLOSURE_TOPK,
        ACTION_BOUNDARY_SHELL_TOPK,
    ),
    context_multiplier: float = 0.5,
    max_context_nodes: int = 64,
    include_context_only: bool = True,
    include_candidate_transplant: bool = True,
    include_boundary_transplant: bool = False,
) -> list[PostGateRecoveryActionCandidate]:
    """Build one-step recovery probes around a post-gate transition state.

    Context-only actions expand the mutable region and let local polish decide
    the labels.  Transplant actions additionally force the selected nodes to
    candidate donor labels before polish.  The latter is diagnostic: it asks
    whether the missed recovery is hidden in nearby candidate-labeled context
    rather than in the next target-node frontier.
    """
    action_nodes = unique_sorted_u32(state.action_nodes)
    direct_nodes = action_nodes if action_nodes.size else unique_sorted_u32(state.direct_nodes)
    exclude = unique_sorted_u32(np.concatenate([state.mutable_nodes, state.context_nodes]))
    cap = cap_context_count(
        direct_node_count=int(direct_nodes.size),
        context_multiplier=context_multiplier,
        max_context_nodes=max_context_nodes,
    )
    if cap <= 0:
        return []
    pull = weighted_pull_to_nodes(
        src=src,
        dst=dst,
        weight=weight,
        target_nodes=direct_nodes,
        node_count=node_count,
    )

    def selected_for(action_type: str) -> np.ndarray:
        if action_type == ACTION_CANDIDATE_CLOSURE_TOPK:
            candidates = label_closure_context_nodes(
                membership=candidate_membership,
                direct_nodes=direct_nodes,
                exclude_nodes=exclude,
            )
        elif action_type == ACTION_VANILLA_CLOSURE_TOPK:
            candidates = label_closure_context_nodes(
                membership=vanilla_membership,
                direct_nodes=direct_nodes,
                exclude_nodes=exclude,
            )
        elif action_type == ACTION_BOUNDARY_SHELL_TOPK:
            candidates = boundary_shell_context_nodes(
                src=src,
                dst=dst,
                direct_nodes=direct_nodes,
                exclude_nodes=exclude,
                node_count=node_count,
            )
        else:
            raise ValueError(f"Unsupported recovery action type: {action_type}")
        return topk_by_pull(
            candidate_nodes=candidates,
            pull_scores=pull,
            max_nodes=cap,
        )

    actions: list[PostGateRecoveryActionCandidate] = []
    seen: set[tuple[str, str, str]] = set()

    def add_action(
        *,
        recovery_policy: str,
        source_action_type: str,
        move_kind: str,
        selected: np.ndarray,
        action_type: str,
        context_nodes: np.ndarray,
        action_nodes_value: np.ndarray | None,
    ) -> None:
        nodes = unique_sorted_u32(selected)
        if nodes.size == 0:
            return
        key = (recovery_policy, move_kind, node_csv(nodes))
        if key in seen:
            return
        seen.add(key)
        action = TransitionAction(
            action_type=action_type,
            action_params=(
                f"recovery_policy={recovery_policy};"
                f"source_action_type={source_action_type};"
                f"move_kind={move_kind};"
                f"context_multiplier={float(context_multiplier):g};"
                f"max_context_nodes={int(max_context_nodes)};"
                f"selected_k={int(nodes.size)}"
            ),
            context_nodes=unique_sorted_u32(context_nodes),
            action_nodes=(
                None
                if action_nodes_value is None
                else unique_sorted_u32(action_nodes_value)
            ),
        )
        actions.append(
            PostGateRecoveryActionCandidate(
                recovery_policy=recovery_policy,
                source_action_type=source_action_type,
                move_kind=move_kind,
                selected_nodes=nodes,
                action=action,
            )
        )

    for action_type in tuple(str(value) for value in action_types):
        selected = selected_for(action_type)
        if include_context_only:
            if action_type == ACTION_CANDIDATE_CLOSURE_TOPK:
                recovery_action_type = ACTION_RECOVERY_CANDIDATE_CONTEXT_TOPK
            elif action_type == ACTION_VANILLA_CLOSURE_TOPK:
                recovery_action_type = ACTION_RECOVERY_VANILLA_CONTEXT_TOPK
            else:
                recovery_action_type = ACTION_RECOVERY_BOUNDARY_CONTEXT_TOPK
            add_action(
                recovery_policy=f"{action_type}:context_only",
                source_action_type=action_type,
                move_kind="context_only",
                selected=selected,
                action_type=recovery_action_type,
                context_nodes=selected,
                action_nodes_value=None,
            )
        if include_candidate_transplant and action_type == ACTION_CANDIDATE_CLOSURE_TOPK:
            add_action(
                recovery_policy=f"{action_type}:candidate_transplant",
                source_action_type=action_type,
                move_kind="candidate_transplant",
                selected=selected,
                action_type=ACTION_RECOVERY_CANDIDATE_TRANSPLANT_TOPK,
                context_nodes=np.asarray([], dtype=np.uint32),
                action_nodes_value=selected,
            )
        if include_boundary_transplant and action_type == ACTION_BOUNDARY_SHELL_TOPK:
            add_action(
                recovery_policy=f"{action_type}:candidate_transplant",
                source_action_type=action_type,
                move_kind="candidate_transplant",
                selected=selected,
                action_type=ACTION_RECOVERY_BOUNDARY_TRANSPLANT_TOPK,
                context_nodes=np.asarray([], dtype=np.uint32),
                action_nodes_value=selected,
            )
    return actions


def build_remaining_target_actions(
    *,
    state: TransitionSearchState,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    node_count: int,
    target_action_multiplier: float,
    max_target_action_nodes: int,
) -> list[TransitionAction]:
    """Select the next uncovered target subset adjacent to the current action."""
    remaining = setdiff_sorted_u32(state.target_nodes, state.covered_target_nodes)
    if remaining.size == 0:
        return []
    anchor = unique_sorted_u32(state.action_nodes)
    if anchor.size == 0:
        anchor = unique_sorted_u32(state.covered_target_nodes)
    if anchor.size == 0:
        return []
    cap = cap_context_count(
        direct_node_count=int(anchor.size),
        context_multiplier=target_action_multiplier,
        max_context_nodes=max_target_action_nodes,
    )
    if cap <= 0:
        return []
    pull = weighted_pull_to_nodes(
        src=src,
        dst=dst,
        weight=weight,
        target_nodes=anchor,
        node_count=node_count,
    )
    action_nodes = topk_by_pull(
        candidate_nodes=remaining,
        pull_scores=pull,
        max_nodes=cap,
    )
    if action_nodes.size == 0:
        return []
    return [
        TransitionAction(
            action_type=ACTION_REMAINING_TARGET_TOPK,
            action_params=(
                f"target_action_multiplier={float(target_action_multiplier):g};"
                f"max_target_action_nodes={int(max_target_action_nodes)}"
            ),
            context_nodes=np.asarray([], dtype=np.uint32),
            action_nodes=action_nodes,
        )
    ]


def remaining_target_pull_frame(
    *,
    state: TransitionSearchState,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    node_count: int,
) -> pd.DataFrame:
    """Rank uncovered target nodes by weighted pull to current action nodes."""
    remaining = setdiff_sorted_u32(state.target_nodes, state.covered_target_nodes)
    anchor = unique_sorted_u32(state.action_nodes)
    if anchor.size == 0:
        anchor = unique_sorted_u32(state.covered_target_nodes)
    if remaining.size == 0 or anchor.size == 0:
        return pd.DataFrame(
            columns=[
                "rank",
                "node",
                "pull_score",
                "cumulative_pull",
                "cumulative_pull_fraction",
                "score_fraction_of_top",
                "next_pull_score",
                "next_gap",
                "next_gap_fraction_of_top",
            ]
        )
    pull = weighted_pull_to_nodes(
        src=src,
        dst=dst,
        weight=weight,
        target_nodes=anchor,
        node_count=node_count,
    )
    frame = pd.DataFrame(
        {
            "node": remaining.astype(np.uint32),
            "pull_score": pull[remaining.astype(np.int64)].astype(np.float64),
        }
    ).sort_values(["pull_score", "node"], ascending=[False, True])
    frame = frame.reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1, dtype=np.int64))
    total_pull = float(frame.loc[frame["pull_score"] > 0.0, "pull_score"].sum())
    top_pull = float(frame["pull_score"].iloc[0]) if not frame.empty else 0.0
    frame["cumulative_pull"] = frame["pull_score"].clip(lower=0.0).cumsum()
    frame["cumulative_pull_fraction"] = (
        frame["cumulative_pull"] / total_pull if total_pull > 0.0 else 0.0
    )
    frame["score_fraction_of_top"] = (
        frame["pull_score"] / top_pull if top_pull > 0.0 else 0.0
    )
    frame["next_pull_score"] = frame["pull_score"].shift(-1, fill_value=0.0)
    frame["next_gap"] = frame["pull_score"] - frame["next_pull_score"]
    frame["next_gap_fraction_of_top"] = (
        frame["next_gap"] / top_pull if top_pull > 0.0 else 0.0
    )
    return frame


def remaining_target_elbow_summary(
    pull_frame: pd.DataFrame,
    *,
    fixed_k: int,
    cumulative_fraction: float = 0.80,
    min_score_fraction: float = 0.05,
    min_gap_fraction: float = 0.25,
    min_guarded_pull_fraction: float = 0.50,
) -> dict[str, Any]:
    """Summarize candidate elbow cut points for a pull-ranked target frontier."""
    remaining_count = int(len(pull_frame))
    if pull_frame.empty:
        return {
            "remaining_count": 0,
            "positive_pull_count": 0,
            "zero_pull_count": 0,
            "fixed_k": int(fixed_k),
            "fixed_effective_k": 0,
            "top_pull": 0.0,
            "total_positive_pull": 0.0,
            "gap_elbow_k": 0,
            "gap_elbow_drop": 0.0,
            "gap_elbow_drop_fraction_of_top": 0.0,
            "gap_elbow_next_ratio": 0.0,
            "cumulative_elbow_k": 0,
            "score_floor_k": 0,
            "guarded_elbow_k": 0,
            "guarded_elbow_reason": "none",
            "fixed_pull_fraction": 0.0,
            "gap_elbow_pull_fraction": 0.0,
            "cumulative_elbow_pull_fraction": 0.0,
            "score_floor_pull_fraction": 0.0,
            "guarded_elbow_pull_fraction": 0.0,
        }
    positive = pull_frame[pull_frame["pull_score"] > 0.0].copy()
    positive_count = int(len(positive))
    zero_count = remaining_count - positive_count
    fixed_effective_k = min(max(0, int(fixed_k)), positive_count)
    if positive.empty:
        return {
            "remaining_count": remaining_count,
            "positive_pull_count": 0,
            "zero_pull_count": zero_count,
            "fixed_k": int(fixed_k),
            "fixed_effective_k": 0,
            "top_pull": 0.0,
            "total_positive_pull": 0.0,
            "gap_elbow_k": 0,
            "gap_elbow_drop": 0.0,
            "gap_elbow_drop_fraction_of_top": 0.0,
            "gap_elbow_next_ratio": 0.0,
            "cumulative_elbow_k": 0,
            "score_floor_k": 0,
            "guarded_elbow_k": 0,
            "guarded_elbow_reason": "none",
            "fixed_pull_fraction": 0.0,
            "gap_elbow_pull_fraction": 0.0,
            "cumulative_elbow_pull_fraction": 0.0,
            "score_floor_pull_fraction": 0.0,
            "guarded_elbow_pull_fraction": 0.0,
        }
    scores = positive["pull_score"].to_numpy(dtype=np.float64)
    top_pull = float(scores[0])
    total_pull = float(scores.sum())
    gaps = positive["next_gap"].to_numpy(dtype=np.float64)
    gap_index = int(np.argmax(gaps))
    gap_elbow_k = int(gap_index + 1)
    gap_drop = float(gaps[gap_index])
    gap_fraction = gap_drop / top_pull if top_pull > 0.0 else 0.0
    next_score = (
        float(scores[gap_index + 1])
        if gap_index + 1 < int(scores.size)
        else 0.0
    )
    gap_next_ratio = gap_drop / max(next_score, 1e-12)
    cumulative = positive["cumulative_pull_fraction"].to_numpy(dtype=np.float64)
    cumulative_elbow_k = int(
        np.searchsorted(cumulative, float(cumulative_fraction), side="left") + 1
    )
    cumulative_elbow_k = min(cumulative_elbow_k, positive_count)
    score_floor_k = int(np.count_nonzero(scores >= top_pull * float(min_score_fraction)))
    score_floor_k = max(1, score_floor_k)

    def pull_fraction(k_value: int) -> float:
        if k_value <= 0 or total_pull <= 0.0:
            return 0.0
        index = min(int(k_value), positive_count) - 1
        return float(positive["cumulative_pull_fraction"].iloc[index])

    gap_pull_fraction = pull_fraction(gap_elbow_k)
    if (
        gap_fraction >= float(min_gap_fraction)
        and gap_pull_fraction >= float(min_guarded_pull_fraction)
    ):
        guarded_elbow_k = gap_elbow_k
        guarded_elbow_reason = "gap"
    else:
        guarded_elbow_k = cumulative_elbow_k
        guarded_elbow_reason = "cumulative"
    guarded_elbow_k = min(max(1, guarded_elbow_k), positive_count)
    return {
        "remaining_count": remaining_count,
        "positive_pull_count": positive_count,
        "zero_pull_count": zero_count,
        "fixed_k": int(fixed_k),
        "fixed_effective_k": fixed_effective_k,
        "top_pull": top_pull,
        "total_positive_pull": total_pull,
        "gap_elbow_k": gap_elbow_k,
        "gap_elbow_drop": gap_drop,
        "gap_elbow_drop_fraction_of_top": float(gap_fraction),
        "gap_elbow_next_ratio": float(gap_next_ratio),
        "cumulative_elbow_k": cumulative_elbow_k,
        "score_floor_k": score_floor_k,
        "guarded_elbow_k": int(guarded_elbow_k),
        "guarded_elbow_reason": guarded_elbow_reason,
        "fixed_pull_fraction": pull_fraction(fixed_effective_k),
        "gap_elbow_pull_fraction": gap_pull_fraction,
        "cumulative_elbow_pull_fraction": pull_fraction(cumulative_elbow_k),
        "score_floor_pull_fraction": pull_fraction(score_floor_k),
        "guarded_elbow_pull_fraction": pull_fraction(guarded_elbow_k),
    }


def _selected_nodes_from_pull_frame(pull_frame: pd.DataFrame, start: int, stop: int) -> np.ndarray:
    if pull_frame.empty or int(stop) <= int(start):
        return np.asarray([], dtype=np.uint32)
    return np.asarray(
        pull_frame.iloc[int(start) : int(stop)]["node"],
        dtype=np.uint32,
    )


def _target_selection_action_params(
    *,
    selection_policy: str,
    escalation_reason: str,
    target_stage_index: int,
    selected_k: int,
    elbow_summary: dict[str, Any],
) -> str:
    return (
        f"selection_policy={selection_policy};"
        f"escalation_reason={escalation_reason};"
        f"target_stage={int(target_stage_index)};"
        f"selected_k={int(selected_k)};"
        f"fixed_effective_k={int(elbow_summary.get('fixed_effective_k', 0))};"
        f"guarded_elbow_k={int(elbow_summary.get('guarded_elbow_k', 0))};"
        f"guarded_elbow_reason={elbow_summary.get('guarded_elbow_reason', '')}"
    )


def build_branching_target_growth_actions(
    *,
    state: TransitionSearchState,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    node_count: int,
    target_stage_index: int,
    target_action_multiplier: float,
    max_target_action_nodes: int,
    selection_policies: tuple[str, ...] = TARGET_SELECTION_POLICIES,
    cumulative_fraction: float = 0.80,
    min_score_fraction: float = 0.05,
    min_gap_fraction: float = 0.25,
    min_guarded_pull_fraction: float = 0.50,
) -> list[BranchTargetActionCandidate]:
    """Build mutually visible target-growth branch actions for one state.

    This keeps fixed-cap and guarded-elbow choices in the same frontier rather
    than committing to a single staged policy before path-level evidence is
    available. ``fixed_tail_backfill`` is the fixed-cap tail omitted by the
    guarded cut; it is useful as a diagnostic branch, not a standalone
    acceptance rule.
    """
    anchor_count = int(unique_sorted_u32(state.action_nodes).size)
    fixed_k = cap_context_count(
        direct_node_count=anchor_count,
        context_multiplier=target_action_multiplier,
        max_context_nodes=max_target_action_nodes,
    )
    pull_frame = remaining_target_pull_frame(
        state=state,
        src=src,
        dst=dst,
        weight=weight,
        node_count=node_count,
    )
    elbow = remaining_target_elbow_summary(
        pull_frame,
        fixed_k=fixed_k,
        cumulative_fraction=cumulative_fraction,
        min_score_fraction=min_score_fraction,
        min_gap_fraction=min_gap_fraction,
        min_guarded_pull_fraction=min_guarded_pull_fraction,
    )
    if pull_frame.empty:
        return []

    actions: list[BranchTargetActionCandidate] = []
    seen: set[tuple[str, str]] = set()

    def add_candidate(
        *,
        selection_policy: str,
        escalation_reason: str,
        selected: np.ndarray,
    ) -> None:
        nodes = unique_sorted_u32(selected)
        if nodes.size == 0:
            return
        key = (selection_policy, node_csv(nodes))
        if key in seen:
            return
        seen.add(key)
        action = TransitionAction(
            action_type=ACTION_REMAINING_TARGET_TOPK,
            action_params=_target_selection_action_params(
                selection_policy=selection_policy,
                escalation_reason=escalation_reason,
                target_stage_index=target_stage_index,
                selected_k=int(nodes.size),
                elbow_summary=elbow,
            ),
            context_nodes=np.asarray([], dtype=np.uint32),
            action_nodes=nodes,
        )
        actions.append(
            BranchTargetActionCandidate(
                selection_policy=selection_policy,
                escalation_reason=escalation_reason,
                target_stage_index=int(target_stage_index),
                selected_nodes=nodes,
                action=action,
                elbow_summary=dict(elbow),
            )
        )

    policies = tuple(str(policy) for policy in selection_policies)
    fixed_effective_k = int(elbow.get("fixed_effective_k", 0))
    guarded_elbow_k = int(elbow.get("guarded_elbow_k", 0))
    if TARGET_SELECTION_GUARDED_ELBOW in policies:
        add_candidate(
            selection_policy=TARGET_SELECTION_GUARDED_ELBOW,
            escalation_reason="guarded_branch",
            selected=_selected_nodes_from_pull_frame(pull_frame, 0, guarded_elbow_k),
        )
    if TARGET_SELECTION_FIXED_CAP in policies:
        add_candidate(
            selection_policy=TARGET_SELECTION_FIXED_CAP,
            escalation_reason="fixed_branch",
            selected=_selected_nodes_from_pull_frame(pull_frame, 0, fixed_effective_k),
        )
    if TARGET_SELECTION_FIXED_TAIL_BACKFILL in policies and fixed_effective_k > guarded_elbow_k:
        add_candidate(
            selection_policy=TARGET_SELECTION_FIXED_TAIL_BACKFILL,
            escalation_reason="fixed_tail_after_guarded_cut",
            selected=_selected_nodes_from_pull_frame(
                pull_frame,
                guarded_elbow_k,
                fixed_effective_k,
            ),
        )
    return actions


def branch_target_action_context(
    candidate: BranchTargetActionCandidate,
) -> dict[str, Any]:
    elbow = candidate.elbow_summary
    return {
        "selection_policy": str(candidate.selection_policy),
        "escalation_reason": str(candidate.escalation_reason),
        "escalated_to_fixed": bool(
            candidate.selection_policy
            in {
                TARGET_SELECTION_FIXED_CAP,
                TARGET_SELECTION_FIXED_TAIL_BACKFILL,
            }
        ),
        "target_stage_index": int(candidate.target_stage_index),
        "selected_k": int(unique_sorted_u32(candidate.selected_nodes).size),
        "selected_node_ids": node_csv(candidate.selected_nodes),
        "remaining_count_before_selection": int(elbow.get("remaining_count", 0)),
        "positive_pull_count": int(elbow.get("positive_pull_count", 0)),
        "fixed_effective_k": int(elbow.get("fixed_effective_k", 0)),
        "guarded_elbow_k": int(elbow.get("guarded_elbow_k", 0)),
        "guarded_elbow_reason": str(elbow.get("guarded_elbow_reason", "")),
        "fixed_pull_fraction": float(elbow.get("fixed_pull_fraction", 0.0)),
        "guarded_elbow_pull_fraction": float(
            elbow.get("guarded_elbow_pull_fraction", 0.0)
        ),
        "gap_elbow_k": int(elbow.get("gap_elbow_k", 0)),
        "gap_elbow_drop_fraction_of_top": float(
            elbow.get("gap_elbow_drop_fraction_of_top", 0.0)
        ),
        "cumulative_elbow_k": int(elbow.get("cumulative_elbow_k", 0)),
        "score_floor_k": int(elbow.get("score_floor_k", 0)),
    }


def build_remaining_target_unit_actions(
    *,
    state: TransitionSearchState,
    target_unit_rows: pd.DataFrame,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    node_count: int,
    target_unit_types: tuple[str, ...] = TARGET_UNIT_TYPES,
    max_target_unit_actions: int = 3,
    max_target_unit_nodes: int = 64,
) -> list[TransitionAction]:
    """Select uncovered target subsets by coherent precomputed target units."""
    remaining = setdiff_sorted_u32(state.target_nodes, state.covered_target_nodes)
    if remaining.size == 0 or target_unit_rows.empty:
        return []
    anchor = unique_sorted_u32(state.action_nodes)
    if anchor.size == 0:
        anchor = unique_sorted_u32(state.covered_target_nodes)
    if anchor.size == 0:
        return []
    pull = weighted_pull_to_nodes(
        src=src,
        dst=dst,
        weight=weight,
        target_nodes=anchor,
        node_count=node_count,
    )
    allowed = set(str(unit_type) for unit_type in target_unit_types)
    candidates: list[dict[str, Any]] = []
    for _, row in target_unit_rows.iterrows():
        unit_type = str(row.get("unit_type", ""))
        if allowed and unit_type not in allowed:
            continue
        unit_nodes = intersect_sorted_u32(parse_node_ids(row.get("node_ids", "")), remaining)
        unit_node_count = int(unit_nodes.size)
        if unit_node_count == 0 or unit_node_count > int(max_target_unit_nodes):
            continue
        pull_weight = float(pull[unit_nodes.astype(np.int64)].sum())
        pull_mean = pull_weight / float(unit_node_count)
        density = _finite_or_zero(row.get("unit_density"))
        triangle_fraction = _finite_or_zero(row.get("triangle_edge_fraction"))
        cohesion = 0.5 * density + 0.5 * triangle_fraction
        score = pull_weight + 0.10 * pull_mean + 0.01 * cohesion
        candidates.append(
            {
                "unit_type": unit_type,
                "unit_id": str(row.get("unit_id", "")),
                "action_nodes": unit_nodes,
                "unit_node_count": unit_node_count,
                "pull_weight": pull_weight,
                "pull_mean": pull_mean,
                "cohesion": cohesion,
                "score": score,
            }
        )
    if not candidates:
        return []
    frame = pd.DataFrame(candidates).sort_values(
        [
            "score",
            "pull_weight",
            "cohesion",
            "unit_node_count",
            "unit_id",
        ],
        ascending=[False, False, False, True, True],
    )
    actions: list[TransitionAction] = []
    for _, row in frame.head(int(max_target_unit_actions)).iterrows():
        actions.append(
            TransitionAction(
                action_type=ACTION_REMAINING_TARGET_UNIT_TOPK,
                action_params=(
                    f"unit_type={row['unit_type']};"
                    f"unit_id={row['unit_id']};"
                    f"unit_score={float(row['score']):.6g};"
                    f"pull_weight={float(row['pull_weight']):.6g};"
                    f"unit_node_count={int(row['unit_node_count'])};"
                    f"max_target_unit_nodes={int(max_target_unit_nodes)}"
                ),
                context_nodes=np.asarray([], dtype=np.uint32),
                action_nodes=unique_sorted_u32(row["action_nodes"]),
            )
        )
    return actions


def classify_search_state(
    *,
    delta_q_vs_start: float,
    candidate_progress_from_vanilla: float,
    support_distance_to_vanilla: float,
    min_support_shift_from_vanilla: float = 0.05,
    min_material_q_gain: float = 0.0,
) -> str:
    if float(delta_q_vs_start) < -abs(float(min_material_q_gain)):
        return SEARCH_LABEL_QUALITY_LOSS
    if float(candidate_progress_from_vanilla) <= 0.0:
        return SEARCH_LABEL_RAW_ONLY
    if float(support_distance_to_vanilla) < float(min_support_shift_from_vanilla):
        return SEARCH_LABEL_VANILLA_COLLAPSE
    if float(delta_q_vs_start) >= float(min_material_q_gain):
        return SEARCH_LABEL_SUPPORT_SHIFT_Q_RECOVERED
    return SEARCH_LABEL_LOW_ROI_SUPPORT_SHIFT


def classify_reachability_state(
    *,
    target_progress_from_vanilla: float,
    support_distance_to_vanilla: float,
    target_coverage_fraction: float,
    min_support_shift_from_vanilla: float = 0.05,
    min_target_progress: float = 1e-9,
    min_coverage_fraction: float = 1e-9,
) -> str:
    """Classify pathway discovery without using QF as a gate."""
    if float(support_distance_to_vanilla) >= float(min_support_shift_from_vanilla):
        return REACHABILITY_LABEL_SUPPORT_GATE_REACHED
    if float(target_progress_from_vanilla) > float(min_target_progress):
        return REACHABILITY_LABEL_TARGET_PROGRESS
    if float(support_distance_to_vanilla) > 0.0:
        return REACHABILITY_LABEL_SOURCE_ESCAPE
    if float(target_coverage_fraction) > float(min_coverage_fraction):
        return REACHABILITY_LABEL_COVERAGE_ONLY
    return REACHABILITY_LABEL_STALLED


def _finite_or_zero(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _first_finite(values: list[Any]) -> float:
    for value in values:
        number = _finite_or_nan(value)
        if math.isfinite(number):
            return number
    return math.nan


def _missing_parent_id(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    return str(value).strip() in {"", "nan", "None"}


def state_distance(
    *,
    support_distance_value: float,
    endpoint_distance_value: float,
    support_weight: float = 1.0,
    endpoint_weight: float = 0.25,
) -> float:
    """Distance to a basin endpoint using label-invariant support and endpoint terms."""
    return (
        float(support_weight) * _finite_or_zero(support_distance_value)
        + float(endpoint_weight) * _finite_or_zero(endpoint_distance_value)
    )


def search_policy_score_column(search_policy: str) -> str:
    if search_policy == SEARCH_POLICY_STATE_GREEDY:
        return "state_greedy_score"
    if search_policy == SEARCH_POLICY_REACHABILITY_FIRST:
        return "reachability_search_score"
    if search_policy == SEARCH_POLICY_QUALITY:
        return "quality_search_score"
    if search_policy == SEARCH_POLICY_PROGRESS:
        return "progress_search_score"
    if search_policy == SEARCH_POLICY_BALANCED:
        return "balanced_search_score"
    raise ValueError(f"Unsupported search policy: {search_policy}")


def search_state_metrics(
    *,
    state: TransitionSearchState,
    baseline_membership: np.ndarray,
    candidate_membership: np.ndarray,
    vanilla_membership: np.ndarray,
    sketch_nodes: np.ndarray,
    start_quality: float,
    candidate_quality: float,
    vanilla_quality: float,
    vanilla_support_distance_to_candidate: float,
    prefix: str = "state",
    min_support_shift_from_vanilla: float = 0.05,
    min_material_q_gain: float = 0.0,
) -> dict[str, Any]:
    metrics = membership_metric_row(
        membership=state.membership,
        quality=state.quality,
        baseline_membership=baseline_membership,
        candidate_membership=candidate_membership,
        vanilla_membership=vanilla_membership,
        sketch_nodes=sketch_nodes,
        start_quality=start_quality,
        candidate_quality=candidate_quality,
        vanilla_quality=vanilla_quality,
        prefix=prefix,
    )
    progress = support_progress_from_vanilla(
        support_distance_to_candidate=metrics[f"{prefix}_support_distance_to_candidate"],
        vanilla_support_distance_to_candidate=vanilla_support_distance_to_candidate,
    )
    label = classify_search_state(
        delta_q_vs_start=float(metrics[f"{prefix}_delta_q_vs_start"]),
        candidate_progress_from_vanilla=float(progress),
        support_distance_to_vanilla=float(metrics[f"{prefix}_support_distance_to_vanilla"]),
        min_support_shift_from_vanilla=min_support_shift_from_vanilla,
        min_material_q_gain=min_material_q_gain,
    )
    mutable_count = int(unique_sorted_u32(state.mutable_nodes).size)
    context_count = int(unique_sorted_u32(state.context_nodes).size)
    target_nodes = unique_sorted_u32(state.target_nodes)
    action_nodes = unique_sorted_u32(state.action_nodes)
    covered_target_nodes = unique_sorted_u32(state.covered_target_nodes)
    action_target_nodes = intersect_sorted_u32(action_nodes, target_nodes)
    target_count = int(target_nodes.size)
    covered_count = int(covered_target_nodes.size)
    action_target_count = int(action_target_nodes.size)
    action_off_target_count = int(action_nodes.size) - action_target_count
    remaining_count = max(0, target_count - covered_count)
    target_coverage_fraction = (
        float(covered_count) / float(target_count) if target_count else 0.0
    )
    context_to_action_ratio = (
        float(context_count) / float(max(1, int(action_nodes.size)))
    )
    target_distance = state_distance(
        support_distance_value=float(metrics[f"{prefix}_support_distance_to_candidate"]),
        endpoint_distance_value=float(metrics[f"{prefix}_endpoint_distance_to_candidate"]),
    )
    source_distance = state_distance(
        support_distance_value=float(metrics[f"{prefix}_support_distance_to_vanilla"]),
        endpoint_distance_value=float(metrics[f"{prefix}_endpoint_distance_to_vanilla"]),
    )
    vanilla_target_distance = state_distance(
        support_distance_value=float(vanilla_support_distance_to_candidate),
        endpoint_distance_value=endpoint_distance(
            vanilla_membership,
            candidate_membership,
            sketch_nodes,
        ),
    )
    target_progress = float(vanilla_target_distance - target_distance)
    q_delta = float(metrics[f"{prefix}_delta_q_vs_start"])
    q_debt = float(metrics[f"{prefix}_q_debt_vs_start"])
    quality_score = q_delta - 1e-3 * float(mutable_count)
    progress_score = target_progress + 0.25 * source_distance - 1e-3 * float(mutable_count)
    state_greedy_score = (
        target_progress
        + 0.50 * source_distance
        + 0.10 * max(q_delta, 0.0)
        - 0.50 * q_debt
        - 1e-3 * float(mutable_count)
        - 2e-3 * float(context_count)
    )
    balanced_score = (
        0.50 * state_greedy_score
        + 0.30 * quality_score
        + 0.20 * progress_score
    )
    reachability_score = (
        target_progress
        + 0.75 * source_distance
        + 0.25 * target_coverage_fraction
    )
    reachability_label = classify_reachability_state(
        target_progress_from_vanilla=target_progress,
        support_distance_to_vanilla=float(metrics[f"{prefix}_support_distance_to_vanilla"]),
        target_coverage_fraction=target_coverage_fraction,
        min_support_shift_from_vanilla=min_support_shift_from_vanilla,
    )
    return {
        **metrics,
        f"{prefix}_candidate_progress_from_vanilla": float(progress),
        f"{prefix}_target_distance": float(target_distance),
        f"{prefix}_source_distance": float(source_distance),
        f"{prefix}_target_progress_from_vanilla": float(target_progress),
        f"{prefix}_source_escape_from_vanilla": float(source_distance),
        "search_recovery_label": label,
        "quality_search_score": float(quality_score),
        "progress_search_score": float(progress_score),
        "balanced_search_score": float(balanced_score),
        "state_greedy_score": float(state_greedy_score),
        "reachability_search_score": float(reachability_score),
        "reachability_label": reachability_label,
        "search_score": float(state_greedy_score),
        "target_node_count": target_count,
        "action_node_count": int(action_nodes.size),
        "action_target_node_count": action_target_count,
        "action_off_target_node_count": action_off_target_count,
        "covered_target_count": covered_count,
        "remaining_target_count": remaining_count,
        "target_coverage_fraction": float(target_coverage_fraction),
        "context_to_action_ratio": float(context_to_action_ratio),
        "mutable_node_count": mutable_count,
        "context_node_count": context_count,
    }


def pathway_marginal_metrics(
    row: dict[str, Any],
    parent_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return parent-relative pathway accounting for a public state row."""
    current_distance = _finite_or_zero(row.get("state_target_distance"))
    current_debt = _finite_or_zero(row.get("state_q_debt_vs_start"))
    current_mutable = int(_finite_or_zero(row.get("mutable_node_count")))
    current_covered = int(_finite_or_zero(row.get("covered_target_count")))
    if parent_row is None:
        parent_distance = current_distance + _finite_or_zero(
            row.get("state_target_progress_from_vanilla")
        )
        parent_debt = 0.0
        parent_mutable = 0
        parent_covered = 0
    else:
        parent_distance = _finite_or_zero(parent_row.get("state_target_distance"))
        parent_debt = _finite_or_zero(parent_row.get("state_q_debt_vs_start"))
        parent_mutable = int(_finite_or_zero(parent_row.get("mutable_node_count")))
        parent_covered = int(_finite_or_zero(parent_row.get("covered_target_count")))
    marginal_covered = current_covered - parent_covered
    marginal_mutable = current_mutable - parent_mutable
    cost_per_target = (
        float(marginal_mutable) / float(marginal_covered)
        if marginal_covered > 0
        else math.nan
    )
    return {
        "marginal_target_distance_reduction": float(parent_distance - current_distance),
        "marginal_q_debt": float(current_debt - parent_debt),
        "marginal_mutable_node_count": int(marginal_mutable),
        "marginal_covered_target_count": int(marginal_covered),
        "marginal_cost_per_target_node": float(cost_per_target),
    }


def _pathway_wall_bucket(q_wall: float) -> str:
    wall = _finite_or_zero(q_wall)
    if wall <= 0.0:
        return "zero"
    if wall <= 0.1:
        return "le_0.1"
    if wall <= 1.0:
        return "le_1"
    if wall <= 5.0:
        return "le_5"
    return "gt_5"


def compute_pathway_wall_rows(
    rows: pd.DataFrame,
    *,
    source_label: str = "",
    support_gate: float = 0.05,
    barrier_floor: float = 1.0,
) -> pd.DataFrame:
    """Reconstruct transition paths and report QF wall statistics.

    Each input row is treated as a terminal state. Its parent chain is followed
    back to the root, and the path wall is the maximum QF debt versus the
    starting basin observed anywhere along that chain. This is a diagnostic
    path statistic: QF debt is measured, not used as an acceptance gate.
    """
    if rows.empty:
        return pd.DataFrame()
    if "state_id" not in rows.columns:
        raise KeyError("transition state rows must include state_id")
    if rows["state_id"].duplicated().any():
        duplicates = rows.loc[rows["state_id"].duplicated(), "state_id"].head(5)
        raise ValueError(f"Duplicate state_id rows: {duplicates.tolist()}")

    indexed = rows.set_index("state_id", drop=False)
    out: list[dict[str, Any]] = []
    for state_id, terminal in indexed.iterrows():
        chain: list[pd.Series] = []
        current_id = str(state_id)
        seen: set[str] = set()
        parent_complete = True
        while True:
            if current_id in seen:
                parent_complete = False
                break
            seen.add(current_id)
            if current_id not in indexed.index:
                parent_complete = False
                break
            current = indexed.loc[current_id]
            chain.append(current)
            parent_id = current.get("parent_state_id", "")
            if _missing_parent_id(parent_id):
                break
            parent_id = str(parent_id)
            if parent_id not in indexed.index:
                parent_complete = False
                break
            current_id = parent_id
        root_to_terminal = list(reversed(chain))
        root = root_to_terminal[0]
        final = root_to_terminal[-1]

        delta_values = [
            _finite_or_nan(row.get("state_delta_q_vs_start", math.nan))
            for row in root_to_terminal
        ]
        delta_values = [value for value in delta_values if math.isfinite(value)]
        min_delta = min(delta_values) if delta_values else math.nan
        final_delta = _finite_or_nan(final.get("state_delta_q_vs_start", math.nan))
        debt_values: list[float] = []
        for row in root_to_terminal:
            debt = _finite_or_nan(row.get("state_q_debt_vs_start", math.nan))
            if not math.isfinite(debt):
                delta = _finite_or_nan(row.get("state_delta_q_vs_start", math.nan))
                debt = max(0.0, -delta) if math.isfinite(delta) else math.nan
            if math.isfinite(debt):
                debt_values.append(max(0.0, debt))
        q_wall = max(debt_values) if debt_values else max(0.0, -_finite_or_zero(min_delta))
        wall_step = 0
        if debt_values:
            wall_step = int(np.argmax(np.asarray(debt_values, dtype=np.float64)))

        denom = max(float(barrier_floor), q_wall)
        root_target_progress = _finite_or_zero(
            root.get("state_target_progress_from_vanilla", 0.0)
        )
        final_target_progress = _finite_or_zero(
            final.get("state_target_progress_from_vanilla", 0.0)
        )
        root_source_escape = _finite_or_zero(
            root.get("state_support_distance_to_vanilla", 0.0)
        )
        final_source_escape = _finite_or_zero(
            final.get("state_support_distance_to_vanilla", 0.0)
        )
        root_coverage = _finite_or_zero(root.get("target_coverage_fraction", 0.0))
        final_coverage = _finite_or_zero(final.get("target_coverage_fraction", 0.0))
        prefix_raw_barrier = _first_finite(
            [row.get("peak_raw_barrier_input", math.nan) for row in root_to_terminal]
        )

        elapsed_sum = sum(
            _finite_or_zero(row.get("elapsed_sec", 0.0)) for row in root_to_terminal
        )
        final_mutable = int(_finite_or_zero(final.get("mutable_node_count", 0)))
        final_support_gate = final_source_escape >= float(support_gate)
        final_q_recovered = math.isfinite(final_delta) and final_delta >= 0.0
        action_types = [
            str(row.get("action_type", "")) for row in root_to_terminal if str(row.get("action_type", ""))
        ]
        path_state_ids = [str(row.get("state_id", "")) for row in root_to_terminal]

        out.append(
            {
                "source_label": str(source_label),
                "case": final.get("case", ""),
                "field": final.get("field", ""),
                "method": final.get("method", ""),
                "pair_id": final.get("pair_id", ""),
                "candidate_index": final.get("candidate_index", math.nan),
                "vanilla_seed": final.get("vanilla_seed", math.nan),
                "path_root_state_id": root.get("state_id", ""),
                "path_final_state_id": final.get("state_id", ""),
                "path_parent_state_id": final.get("parent_state_id", ""),
                "path_prefix_rank": int(_finite_or_zero(final.get("prefix_rank", 0))),
                "path_prefix_unit_ids": final.get("prefix_unit_ids", ""),
                "path_policy": final.get("path_policy", ""),
                "path_selection_policy": final.get("selection_policy", ""),
                "path_escalation_reason": final.get("escalation_reason", ""),
                "path_escalated_to_fixed": bool(final.get("escalated_to_fixed", False)),
                "path_target_stage_index": int(
                    _finite_or_zero(final.get("target_stage_index", 0))
                ),
                "path_selected_k": int(_finite_or_zero(final.get("selected_k", 0))),
                "path_fixed_effective_k": int(
                    _finite_or_zero(final.get("fixed_effective_k", 0))
                ),
                "path_guarded_elbow_k": int(
                    _finite_or_zero(final.get("guarded_elbow_k", 0))
                ),
                "path_depth": int(_finite_or_zero(final.get("depth", 0))),
                "path_state_count": int(len(root_to_terminal)),
                "path_edge_count": max(0, int(len(root_to_terminal)) - 1),
                "path_parent_complete": bool(parent_complete),
                "path_action_types": ",".join(action_types),
                "path_applied_actions": final.get("applied_actions", ""),
                "path_state_ids": "|".join(path_state_ids),
                "path_q_wall": float(q_wall),
                "path_q_wall_bucket": _pathway_wall_bucket(q_wall),
                "path_wall_step_index": int(wall_step),
                "path_wall_crossed": bool(q_wall > 0.0),
                "path_min_delta_q_vs_start": float(min_delta),
                "path_final_delta_q_vs_start": float(final_delta),
                "path_q_recovery_from_wall": float(final_delta - min_delta)
                if math.isfinite(final_delta) and math.isfinite(min_delta)
                else math.nan,
                "path_prefix_raw_barrier_input": float(prefix_raw_barrier),
                "path_wall_minus_prefix_raw_barrier": float(q_wall - prefix_raw_barrier)
                if math.isfinite(prefix_raw_barrier)
                else math.nan,
                "path_wall_reduction_vs_prefix_raw_barrier": float(prefix_raw_barrier - q_wall)
                if math.isfinite(prefix_raw_barrier)
                else math.nan,
                "path_final_support_distance_to_vanilla": float(final_source_escape),
                "path_final_support_distance_to_candidate": _finite_or_nan(
                    final.get("state_support_distance_to_candidate", math.nan)
                ),
                "path_final_target_progress_from_vanilla": float(final_target_progress),
                "path_final_candidate_progress_from_vanilla": _finite_or_nan(
                    final.get("state_candidate_progress_from_vanilla", math.nan)
                ),
                "path_final_target_coverage_fraction": float(final_coverage),
                "path_final_covered_target_count": int(
                    _finite_or_zero(final.get("covered_target_count", 0))
                ),
                "path_final_remaining_target_count": int(
                    _finite_or_zero(final.get("remaining_target_count", 0))
                ),
                "path_final_mutable_node_count": final_mutable,
                "path_elapsed_sec_sum": float(elapsed_sum),
                "path_target_progress_gain_from_root": float(
                    final_target_progress - root_target_progress
                ),
                "path_source_escape_gain_from_root": float(
                    final_source_escape - root_source_escape
                ),
                "path_coverage_gain_from_root": float(final_coverage - root_coverage),
                "path_target_progress_per_wall_floor": float(
                    final_target_progress / denom
                ),
                "path_source_escape_per_wall_floor": float(final_source_escape / denom),
                "path_coverage_per_wall_floor": float(final_coverage / denom),
                "path_delta_q_per_wall_floor": float(final_delta / denom)
                if math.isfinite(final_delta)
                else math.nan,
                "path_support_gate_reached": bool(final_support_gate),
                "path_q_recovered": bool(final_q_recovered),
                "path_support_gate_q_recovered": bool(
                    final_support_gate and final_q_recovered
                ),
                "path_final_search_recovery_label": final.get("search_recovery_label", ""),
                "path_final_reachability_label": final.get("reachability_label", ""),
                "path_final_reachability_search_score": _finite_or_nan(
                    final.get("reachability_search_score", math.nan)
                ),
                "path_final_state_greedy_score": _finite_or_nan(
                    final.get("state_greedy_score", math.nan)
                ),
            }
        )
    return pd.DataFrame(out)


def _path_chain(indexed: pd.DataFrame, state_id: Any) -> tuple[list[pd.Series], bool]:
    chain: list[pd.Series] = []
    current_id = str(state_id)
    seen: set[str] = set()
    parent_complete = True
    while True:
        if current_id in seen:
            parent_complete = False
            break
        seen.add(current_id)
        if current_id not in indexed.index:
            parent_complete = False
            break
        current = indexed.loc[current_id]
        if isinstance(current, pd.DataFrame):
            current = current.iloc[0]
        chain.append(current)
        parent_id = current.get("parent_state_id", "")
        if _missing_parent_id(parent_id):
            break
        parent_id = str(parent_id)
        if parent_id not in indexed.index:
            parent_complete = False
            break
        current_id = parent_id
    return list(reversed(chain)), parent_complete


def annotate_pathway_debt_area_rows(
    path_rows: pd.DataFrame,
    *,
    state_rows: pd.DataFrame,
    support_gate: float = 0.05,
) -> pd.DataFrame:
    """Add QF debt area, duration, and recovery-slope metrics to path rows."""
    if path_rows.empty:
        return path_rows.copy()
    if state_rows.empty:
        raise ValueError("state_rows must be non-empty")
    if "state_id" not in state_rows.columns:
        raise KeyError("state_rows must include state_id")
    indexed = state_rows.set_index("state_id", drop=False)
    out = path_rows.copy()
    metric_rows: list[dict[str, Any]] = []
    for _, path in out.iterrows():
        state_ids = _path_state_ids(path.get("path_state_ids", ""))
        terminal_id = (
            state_ids[-1] if state_ids else str(path.get("path_final_state_id", ""))
        )
        chain, parent_complete = _path_chain(indexed, terminal_id)
        debt_values: list[float] = []
        delta_values: list[float] = []
        elapsed_values: list[float] = []
        mutable_values: list[int] = []
        marginal_mutable_values: list[int] = []
        for index, row in enumerate(chain):
            delta = _finite_or_nan(row.get("state_delta_q_vs_start", math.nan))
            debt = _finite_or_nan(row.get("state_q_debt_vs_start", math.nan))
            if not math.isfinite(debt):
                debt = max(0.0, -delta) if math.isfinite(delta) else 0.0
            debt = max(0.0, _finite_or_zero(debt))
            mutable = int(_finite_or_zero(row.get("mutable_node_count", 0)))
            if "marginal_mutable_node_count" in row:
                marginal_mutable = int(
                    max(0.0, _finite_or_zero(row.get("marginal_mutable_node_count", 0)))
                )
            elif index == 0:
                marginal_mutable = mutable
            else:
                previous = mutable_values[-1] if mutable_values else 0
                marginal_mutable = max(0, mutable - previous)
            debt_values.append(float(debt))
            delta_values.append(float(delta) if math.isfinite(delta) else math.nan)
            elapsed_values.append(max(0.0, _finite_or_zero(row.get("elapsed_sec", 0.0))))
            mutable_values.append(mutable)
            marginal_mutable_values.append(marginal_mutable)

        state_count = len(chain)
        q_wall = max(debt_values) if debt_values else 0.0
        below_start_count = sum(1 for debt in debt_values if debt > 0.0)
        debt_area_step = float(sum(debt_values))
        debt_area_elapsed = float(
            sum(debt * elapsed for debt, elapsed in zip(debt_values, elapsed_values, strict=False))
        )
        debt_area_mutable = float(
            sum(
                debt * float(max(1, marginal_mutable))
                for debt, marginal_mutable in zip(
                    debt_values,
                    marginal_mutable_values,
                    strict=False,
                )
                if debt > 0.0
            )
        )
        wall_step_index = (
            int(np.argmax(np.asarray(debt_values, dtype=np.float64)))
            if debt_values
            else 0
        )
        final_delta = _finite_or_nan(path.get("path_final_delta_q_vs_start", math.nan))
        if not math.isfinite(final_delta) and delta_values:
            final_delta = delta_values[-1]
        min_delta = min([value for value in delta_values if math.isfinite(value)], default=math.nan)
        recovery_from_wall = (
            float(final_delta - min_delta)
            if math.isfinite(final_delta) and math.isfinite(min_delta)
            else math.nan
        )
        post_wall_steps = max(0, state_count - wall_step_index - 1)
        post_wall_elapsed = float(sum(elapsed_values[wall_step_index + 1 :]))
        post_wall_mutable = float(sum(marginal_mutable_values[wall_step_index + 1 :]))
        support = _finite_or_zero(path.get("path_final_support_distance_to_vanilla", 0.0))
        progress = _finite_or_zero(path.get("path_final_target_progress_from_vanilla", 0.0))

        def _ratio(numerator: float, denominator: float) -> float:
            if denominator <= 0.0:
                return math.inf if numerator > 0.0 else math.nan
            return float(numerator / denominator)

        metric_rows.append(
            {
                "path_chain_parent_complete": bool(parent_complete),
                "path_debt_below_start_state_count": int(below_start_count),
                "path_debt_below_start_fraction": (
                    float(below_start_count) / float(state_count) if state_count else 0.0
                ),
                "path_q_debt_area_step": debt_area_step,
                "path_q_debt_area_elapsed": debt_area_elapsed,
                "path_q_debt_area_mutable": debt_area_mutable,
                "path_wall_duration_steps": int(below_start_count),
                "path_post_wall_steps": int(post_wall_steps),
                "path_post_wall_elapsed_sec": post_wall_elapsed,
                "path_post_wall_marginal_mutable": post_wall_mutable,
                "path_q_recovery_from_wall_area": recovery_from_wall,
                "path_recovery_slope_per_step": _ratio(
                    _finite_or_zero(recovery_from_wall),
                    float(max(1, post_wall_steps)),
                ),
                "path_recovery_slope_per_elapsed": _ratio(
                    _finite_or_zero(recovery_from_wall),
                    max(post_wall_elapsed, 1e-12),
                ),
                "path_recovery_slope_per_mutable": _ratio(
                    _finite_or_zero(recovery_from_wall),
                    max(post_wall_mutable, 1.0),
                ),
                "path_support_per_debt_area_step": _ratio(support, debt_area_step),
                "path_progress_per_debt_area_step": _ratio(progress, debt_area_step),
                "path_support_per_debt_area_mutable": _ratio(support, debt_area_mutable),
                "path_progress_per_debt_area_mutable": _ratio(progress, debt_area_mutable),
                "path_support_per_debt_area_elapsed": _ratio(support, debt_area_elapsed),
                "path_progress_per_debt_area_elapsed": _ratio(progress, debt_area_elapsed),
                "path_shortcut_score_step": _ratio(
                    progress + 0.5 * support + max(0.0, final_delta),
                    max(debt_area_step, 1e-12),
                ),
                "path_shortcut_score_mutable": _ratio(
                    progress + 0.5 * support + max(0.0, final_delta),
                    max(debt_area_mutable, 1e-12),
                ),
                "path_shortcut_score_elapsed": _ratio(
                    progress + 0.5 * support + max(0.0, final_delta),
                    max(debt_area_elapsed, 1e-12),
                ),
                "path_gate_quality_recovered_by_area": bool(
                    support >= float(support_gate) and math.isfinite(final_delta) and final_delta >= 0.0
                ),
                "path_q_wall_for_area": float(q_wall),
            }
        )
    metrics = pd.DataFrame(metric_rows, index=out.index)
    return pd.concat([out, metrics], axis=1)


def _numeric_path_column(
    frame: pd.DataFrame,
    column: str,
    *,
    default: float = 0.0,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def annotate_tunneling_evidence_rows(
    path_rows: pd.DataFrame,
    *,
    support_gate: float = 0.05,
    progress_margin: float = 0.005,
) -> pd.DataFrame:
    """Classify path rows as recoverable tunnels, detours, or stalled probes.

    This is diagnostic-only. A "tunnel" here means a non-monotone,
    candidate-directed path that pays temporary QF debt and recovers by the
    terminal state. It does not imply a production Dongdaemun policy.
    """
    if path_rows.empty:
        return path_rows.copy()
    if "path_q_debt_area_step" not in path_rows.columns:
        raise KeyError(
            "path_rows must include path_q_debt_area_step; call "
            "annotate_pathway_debt_area_rows first"
        )
    out = path_rows.copy()
    support = _numeric_path_column(out, "path_final_support_distance_to_vanilla")
    progress = _numeric_path_column(out, "path_final_target_progress_from_vanilla")
    delta_q = _numeric_path_column(
        out,
        "path_final_delta_q_vs_start",
        default=math.nan,
    )
    q_wall = _numeric_path_column(out, "path_q_wall")
    debt_area = _numeric_path_column(out, "path_q_debt_area_step")
    mutable_area = _numeric_path_column(out, "path_q_debt_area_mutable")

    candidate_directed = (support >= float(support_gate)) & (
        progress > float(progress_margin)
    )
    q_recovered = delta_q >= 0.0
    nonmonotone = (q_wall > 0.0) & (debt_area > 0.0)
    route_labels: list[str] = []
    for idx in out.index:
        if bool(candidate_directed.loc[idx]) and bool(q_recovered.loc[idx]):
            if bool(nonmonotone.loc[idx]):
                route_labels.append(TUNNEL_ROUTE_RECOVERABLE)
            else:
                route_labels.append(TUNNEL_ROUTE_DIRECT_RECOVERED)
        elif bool(candidate_directed.loc[idx]) and bool(nonmonotone.loc[idx]):
            route_labels.append(TUNNEL_ROUTE_UNRECOVERED_DETOUR)
        elif float(support.loc[idx]) >= float(support_gate):
            route_labels.append(TUNNEL_ROUTE_SUPPORT_GATE_NO_TARGET)
        elif float(progress.loc[idx]) > float(progress_margin):
            route_labels.append(TUNNEL_ROUTE_PARTIAL_PROGRESS)
        else:
            route_labels.append(TUNNEL_ROUTE_STALLED)

    out["tunnel_route_label"] = route_labels
    out["tunnel_candidate_directed"] = candidate_directed.astype(bool)
    out["tunnel_q_recovered"] = q_recovered.astype(bool)
    out["tunnel_requires_nonmonotone"] = nonmonotone.astype(bool)
    out["tunnel_recoverable"] = (
        out["tunnel_route_label"].astype(str).eq(TUNNEL_ROUTE_RECOVERABLE)
    )
    out["tunnel_unrecovered_detour"] = (
        out["tunnel_route_label"].astype(str).eq(TUNNEL_ROUTE_UNRECOVERED_DETOUR)
    )
    positive_delta = delta_q.clip(lower=0.0)
    out["tunnel_q_recovery_per_debt_area_step"] = positive_delta / debt_area.replace(
        0.0,
        np.nan,
    )
    out["tunnel_q_recovery_per_debt_area_mutable"] = (
        positive_delta / mutable_area.replace(0.0, np.nan)
    )
    out["tunnel_support_progress_score"] = support + progress
    out["tunnel_recovered_shortcut_score"] = (
        positive_delta + support + progress
    ) / debt_area.replace(0.0, np.nan)
    out["tunnel_debt_area_gap_to_wall"] = debt_area - q_wall
    out["tunnel_wall_concentration"] = q_wall / debt_area.replace(0.0, np.nan)
    return out


def trace_tunneling_path_states(
    path_rows: pd.DataFrame,
    *,
    state_rows: pd.DataFrame,
    support_gate: float = 0.05,
    progress_margin: float = 0.005,
    max_paths: int | None = None,
) -> pd.DataFrame:
    """Expand selected path rows into per-state traces for tunnel diagnosis."""
    if path_rows.empty:
        return pd.DataFrame()
    if state_rows.empty:
        raise ValueError("state_rows must be non-empty")
    if "state_id" not in state_rows.columns:
        raise KeyError("state_rows must include state_id")
    selected = path_rows.copy()
    if max_paths is not None:
        selected = selected.head(int(max_paths))
    indexed = state_rows.set_index("state_id", drop=False)
    out: list[dict[str, Any]] = []
    for _, path in selected.iterrows():
        state_ids = _path_state_ids(path.get("path_state_ids", ""))
        terminal_id = (
            state_ids[-1] if state_ids else str(path.get("path_final_state_id", ""))
        )
        chain, parent_complete = _path_chain(indexed, terminal_id)
        if not chain:
            continue
        debts: list[float] = []
        deltas: list[float] = []
        supports: list[float] = []
        progresses: list[float] = []
        for row in chain:
            delta = _finite_or_nan(row.get("state_delta_q_vs_start", math.nan))
            debt = _finite_or_nan(row.get("state_q_debt_vs_start", math.nan))
            if not math.isfinite(debt):
                debt = max(0.0, -delta) if math.isfinite(delta) else 0.0
            debts.append(max(0.0, _finite_or_zero(debt)))
            deltas.append(float(delta) if math.isfinite(delta) else math.nan)
            supports.append(
                _finite_or_zero(row.get("state_support_distance_to_vanilla", 0.0))
            )
            progresses.append(
                _finite_or_zero(row.get("state_target_progress_from_vanilla", 0.0))
            )

        q_wall = max(debts) if debts else 0.0
        wall_step_index = int(np.argmax(np.asarray(debts, dtype=np.float64)))
        first_support_gate_step = next(
            (idx for idx, value in enumerate(supports) if value >= float(support_gate)),
            -1,
        )
        first_target_progress_step = next(
            (
                idx
                for idx, value in enumerate(progresses)
                if value > float(progress_margin)
            ),
            -1,
        )
        first_candidate_directed_step = next(
            (
                idx
                for idx, (support, progress) in enumerate(zip(supports, progresses))
                if support >= float(support_gate)
                and progress > float(progress_margin)
            ),
            -1,
        )
        first_q_recovered_step = next(
            (
                idx
                for idx, delta in enumerate(deltas)
                if math.isfinite(delta) and delta >= 0.0
            ),
            -1,
        )
        cumulative_debt = 0.0
        for step_index, row in enumerate(chain):
            debt = debts[step_index]
            delta = deltas[step_index]
            support = supports[step_index]
            progress = progresses[step_index]
            cumulative_debt += debt
            is_wall_peak = q_wall > 0.0 and math.isclose(
                debt,
                q_wall,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            candidate_directed = (
                support >= float(support_gate)
                and progress > float(progress_margin)
            )
            q_recovered = math.isfinite(delta) and delta >= 0.0
            if is_wall_peak:
                phase = "wall_peak"
            elif debt > 0.0:
                phase = "under_q_debt"
            elif candidate_directed and q_recovered:
                phase = "candidate_recovered"
            elif support >= float(support_gate):
                phase = "support_gate"
            elif progress > float(progress_margin):
                phase = "partial_progress"
            else:
                phase = "pre_tunnel"
            out.append(
                {
                    "artifact_label": path.get("artifact_label", ""),
                    "pair_id": path.get("pair_id", ""),
                    "path_final_state_id": path.get("path_final_state_id", ""),
                    "tunnel_route_label": path.get("tunnel_route_label", ""),
                    "tunnel_operator_category": path.get(
                        "tunnel_operator_category",
                        "",
                    ),
                    "path_prefix_rank": path.get("path_prefix_rank", math.nan),
                    "path_selection_policy": path.get("path_selection_policy", ""),
                    "path_policy": path.get("path_policy", ""),
                    "path_parent_complete": bool(parent_complete),
                    "trace_step_index": int(step_index),
                    "trace_state_id": row.get("state_id", ""),
                    "trace_parent_state_id": row.get("parent_state_id", ""),
                    "trace_action_type": row.get("action_type", ""),
                    "trace_phase": phase,
                    "trace_is_wall_peak": bool(is_wall_peak),
                    "trace_in_q_debt": bool(debt > 0.0),
                    "trace_q_debt": float(debt),
                    "trace_cumulative_q_debt_area_step": float(cumulative_debt),
                    "trace_delta_q_vs_start": float(delta),
                    "trace_support_distance_to_vanilla": float(support),
                    "trace_support_distance_to_candidate": _finite_or_nan(
                        row.get("state_support_distance_to_candidate", math.nan)
                    ),
                    "trace_target_progress_from_vanilla": float(progress),
                    "trace_candidate_progress_from_vanilla": _finite_or_nan(
                        row.get("state_candidate_progress_from_vanilla", math.nan)
                    ),
                    "trace_mutable_node_count": int(
                        _finite_or_zero(row.get("mutable_node_count", 0))
                    ),
                    "trace_marginal_mutable_node_count": int(
                        _finite_or_zero(row.get("marginal_mutable_node_count", 0))
                    ),
                    "trace_elapsed_sec": float(
                        _finite_or_zero(row.get("elapsed_sec", 0.0))
                    ),
                    "trace_wall_step_index": int(wall_step_index),
                    "trace_first_support_gate_step": int(first_support_gate_step),
                    "trace_first_target_progress_step": int(first_target_progress_step),
                    "trace_first_candidate_directed_step": int(
                        first_candidate_directed_step
                    ),
                    "trace_first_q_recovered_step": int(first_q_recovered_step),
                    "trace_candidate_directed": bool(candidate_directed),
                    "trace_q_recovered": bool(q_recovered),
                }
            )
    return pd.DataFrame(out)


def annotate_post_gate_recovery_step_rows(
    trace_rows: pd.DataFrame,
    *,
    min_q_recovery_gain: float = 1e-9,
    min_support_gain: float = 1e-9,
    min_progress_gain: float = 1e-9,
) -> pd.DataFrame:
    """Add parent-relative post-gate recovery labels to tunnel trace rows.

    The input is the per-state output from :func:`trace_tunneling_path_states`.
    The first candidate-directed step is treated as the gate when available;
    otherwise the first support-gate step is used.  This keeps the diagnostic
    focused on what happens after the path has already paid the basin-wall cost.
    """
    if trace_rows.empty:
        return trace_rows.copy()
    required = {
        "path_final_state_id",
        "trace_step_index",
        "trace_delta_q_vs_start",
        "trace_q_debt",
        "trace_support_distance_to_vanilla",
        "trace_target_progress_from_vanilla",
        "trace_first_candidate_directed_step",
        "trace_first_support_gate_step",
        "trace_candidate_directed",
        "trace_q_recovered",
    }
    missing = sorted(required.difference(trace_rows.columns))
    if missing:
        raise KeyError(f"trace_rows missing required post-gate columns: {missing}")

    frames: list[pd.DataFrame] = []
    for _, group in trace_rows.groupby("path_final_state_id", sort=False):
        ordered = group.sort_values("trace_step_index").copy()
        candidate_gate = int(
            _finite_or_zero(ordered["trace_first_candidate_directed_step"].iloc[0])
        )
        support_gate = int(
            _finite_or_zero(ordered["trace_first_support_gate_step"].iloc[0])
        )
        gate_step = candidate_gate if candidate_gate >= 0 else support_gate
        if gate_step < 0:
            ordered["post_gate_step_index"] = -1
            ordered["post_gate_step_label"] = POST_GATE_STEP_NO_GATE
            ordered["post_gate_delta_q_change"] = math.nan
            ordered["post_gate_q_debt_change"] = math.nan
            ordered["post_gate_support_change"] = math.nan
            ordered["post_gate_target_progress_change"] = math.nan
            ordered["post_gate_is_after_gate"] = False
            frames.append(ordered)
            continue

        deltas = pd.to_numeric(
            ordered["trace_delta_q_vs_start"],
            errors="coerce",
        ).astype(float)
        debts = pd.to_numeric(ordered["trace_q_debt"], errors="coerce").fillna(0.0)
        supports = pd.to_numeric(
            ordered["trace_support_distance_to_vanilla"],
            errors="coerce",
        ).fillna(0.0)
        progresses = pd.to_numeric(
            ordered["trace_target_progress_from_vanilla"],
            errors="coerce",
        ).fillna(0.0)
        ordered["post_gate_step_index"] = (
            pd.to_numeric(ordered["trace_step_index"], errors="coerce").astype(int)
            - int(gate_step)
        )
        ordered["post_gate_delta_q_change"] = deltas.diff()
        ordered["post_gate_q_debt_change"] = debts.diff()
        ordered["post_gate_support_change"] = supports.diff()
        ordered["post_gate_target_progress_change"] = progresses.diff()
        ordered["post_gate_is_after_gate"] = ordered["post_gate_step_index"] > 0

        labels: list[str] = []
        for _, row in ordered.iterrows():
            rel_step = int(row["post_gate_step_index"])
            if rel_step < 0:
                labels.append(POST_GATE_STEP_PRE_GATE)
                continue
            if rel_step == 0:
                labels.append(POST_GATE_STEP_GATE_ENTRY)
                continue
            delta_change = _finite_or_zero(row["post_gate_delta_q_change"])
            debt_change = _finite_or_zero(row["post_gate_q_debt_change"])
            support_change = _finite_or_zero(row["post_gate_support_change"])
            progress_change = _finite_or_zero(
                row["post_gate_target_progress_change"]
            )
            has_support_gain = (
                support_change > float(min_support_gain)
                or progress_change > float(min_progress_gain)
            )
            has_q_recovery = (
                delta_change > float(min_q_recovery_gain)
                or debt_change < -float(min_q_recovery_gain)
            )
            has_q_regression = (
                delta_change < -float(min_q_recovery_gain)
                or debt_change > float(min_q_recovery_gain)
            )
            if bool(row["trace_candidate_directed"]) and bool(row["trace_q_recovered"]):
                labels.append(POST_GATE_STEP_RECOVERED)
            elif has_q_recovery and has_support_gain:
                labels.append(POST_GATE_STEP_RECOVERY_TREND)
            elif has_support_gain and has_q_regression:
                labels.append(POST_GATE_STEP_SUPPORT_QUALITY_TRADEOFF)
            elif has_support_gain:
                labels.append(POST_GATE_STEP_SUPPORT_DEEPENING)
            elif has_q_regression:
                labels.append(POST_GATE_STEP_QUALITY_REGRESSION)
            else:
                labels.append(POST_GATE_STEP_PLATEAU)
        ordered["post_gate_step_label"] = labels
        frames.append(ordered)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_post_gate_recovery_paths(
    trace_rows: pd.DataFrame,
    *,
    min_q_recovery_gain: float = 1e-9,
    min_support_gain: float = 1e-9,
    min_progress_gain: float = 1e-9,
) -> pd.DataFrame:
    """Summarize whether post-gate steps recover QF or only deepen support."""
    if trace_rows.empty:
        return pd.DataFrame()
    steps = annotate_post_gate_recovery_step_rows(
        trace_rows,
        min_q_recovery_gain=min_q_recovery_gain,
        min_support_gain=min_support_gain,
        min_progress_gain=min_progress_gain,
    )
    out: list[dict[str, Any]] = []
    for final_state_id, group in steps.groupby("path_final_state_id", sort=False):
        ordered = group.sort_values("trace_step_index").copy()
        gate_candidates = ordered[ordered["post_gate_step_index"].eq(0)]
        final = ordered.iloc[-1]
        if gate_candidates.empty:
            out.append(
                {
                    "artifact_label": final.get("artifact_label", ""),
                    "pair_id": final.get("pair_id", ""),
                    "path_final_state_id": final_state_id,
                    "path_prefix_rank": final.get("path_prefix_rank", math.nan),
                    "path_selection_policy": final.get("path_selection_policy", ""),
                    "path_policy": final.get("path_policy", ""),
                    "tunnel_route_label": final.get("tunnel_route_label", ""),
                    "tunnel_operator_category": final.get(
                        "tunnel_operator_category",
                        "",
                    ),
                    "post_gate_verdict": POST_GATE_VERDICT_NO_GATE,
                    "post_gate_step_count": 0,
                    "post_gate_gate_step": -1,
                    "post_gate_wall_step": int(
                        _finite_or_zero(final.get("trace_wall_step_index", -1))
                    ),
                }
            )
            continue

        gate = gate_candidates.iloc[0]
        post = ordered[ordered["post_gate_step_index"] > 0].copy()
        gate_delta = _finite_or_nan(gate.get("trace_delta_q_vs_start", math.nan))
        gate_debt = _finite_or_zero(gate.get("trace_q_debt", 0.0))
        gate_support = _finite_or_zero(
            gate.get("trace_support_distance_to_vanilla", 0.0)
        )
        gate_progress = _finite_or_zero(
            gate.get("trace_target_progress_from_vanilla", 0.0)
        )
        final_delta = _finite_or_nan(final.get("trace_delta_q_vs_start", math.nan))
        final_support = _finite_or_zero(
            final.get("trace_support_distance_to_vanilla", 0.0)
        )
        final_progress = _finite_or_zero(
            final.get("trace_target_progress_from_vanilla", 0.0)
        )
        if post.empty:
            best_delta = final_delta
            best_delta_step = int(_finite_or_zero(final.get("trace_step_index", -1)))
            best_support = final_support
            best_progress = final_progress
            recovered_after_gate = False
            recovery_step_count = 0
            support_deepening_step_count = 0
            tradeoff_step_count = 0
            plateau_step_count = 0
            verdict = POST_GATE_VERDICT_GATE_TERMINAL
        else:
            post_deltas = pd.to_numeric(
                post["trace_delta_q_vs_start"],
                errors="coerce",
            )
            best_idx = post_deltas.idxmax()
            best = post.loc[best_idx]
            best_delta = _finite_or_nan(best.get("trace_delta_q_vs_start", math.nan))
            best_delta_step = int(_finite_or_zero(best.get("trace_step_index", -1)))
            best_support = float(
                pd.to_numeric(
                    post["trace_support_distance_to_vanilla"],
                    errors="coerce",
                )
                .fillna(0.0)
                .max()
            )
            best_progress = float(
                pd.to_numeric(
                    post["trace_target_progress_from_vanilla"],
                    errors="coerce",
                )
                .fillna(0.0)
                .max()
            )
            labels = post["post_gate_step_label"].astype(str)
            recovered_after_gate = bool(
                labels.eq(POST_GATE_STEP_RECOVERED).any()
                or (
                    post["trace_candidate_directed"].astype(bool)
                    & post["trace_q_recovered"].astype(bool)
                ).any()
            )
            recovery_step_count = int(
                labels.isin(
                    [
                        POST_GATE_STEP_RECOVERED,
                        POST_GATE_STEP_RECOVERY_TREND,
                    ]
                ).sum()
            )
            support_deepening_step_count = int(
                labels.isin(
                    [
                        POST_GATE_STEP_RECOVERY_TREND,
                        POST_GATE_STEP_SUPPORT_DEEPENING,
                        POST_GATE_STEP_SUPPORT_QUALITY_TRADEOFF,
                    ]
                ).sum()
            )
            tradeoff_step_count = int(
                labels.eq(POST_GATE_STEP_SUPPORT_QUALITY_TRADEOFF).sum()
            )
            plateau_step_count = int(labels.eq(POST_GATE_STEP_PLATEAU).sum())

            q_gain = (
                best_delta - gate_delta
                if math.isfinite(best_delta) and math.isfinite(gate_delta)
                else math.nan
            )
            final_q_loss_from_best = (
                best_delta - final_delta
                if math.isfinite(best_delta) and math.isfinite(final_delta)
                else math.nan
            )
            support_gain = best_support - gate_support
            progress_gain = best_progress - gate_progress
            if recovered_after_gate:
                verdict = POST_GATE_VERDICT_RECOVERED
            elif (
                math.isfinite(q_gain)
                and q_gain > float(min_q_recovery_gain)
                and (
                    support_gain > float(min_support_gain)
                    or progress_gain > float(min_progress_gain)
                )
                and not (
                    math.isfinite(final_q_loss_from_best)
                    and final_q_loss_from_best > float(min_q_recovery_gain)
                    and final_support > gate_support + float(min_support_gain)
                )
            ):
                verdict = POST_GATE_VERDICT_NEAR_MISS
            elif (
                math.isfinite(final_q_loss_from_best)
                and final_q_loss_from_best > float(min_q_recovery_gain)
                and (
                    final_support > gate_support + float(min_support_gain)
                    or final_progress > gate_progress + float(min_progress_gain)
                )
            ):
                verdict = POST_GATE_VERDICT_SUPPORT_TRADEOFF
            elif (
                best_support > gate_support + float(min_support_gain)
                or best_progress > gate_progress + float(min_progress_gain)
            ):
                verdict = POST_GATE_VERDICT_SUPPORT_ONLY
            elif math.isfinite(q_gain) and q_gain > float(min_q_recovery_gain):
                verdict = POST_GATE_VERDICT_QUALITY_ONLY
            else:
                verdict = POST_GATE_VERDICT_PLATEAU

        best_delta_gain = (
            best_delta - gate_delta
            if math.isfinite(best_delta) and math.isfinite(gate_delta)
            else math.nan
        )
        final_delta_gain = (
            final_delta - gate_delta
            if math.isfinite(final_delta) and math.isfinite(gate_delta)
            else math.nan
        )
        out.append(
            {
                "artifact_label": final.get("artifact_label", ""),
                "pair_id": final.get("pair_id", ""),
                "path_final_state_id": final_state_id,
                "path_prefix_rank": final.get("path_prefix_rank", math.nan),
                "path_selection_policy": final.get("path_selection_policy", ""),
                "path_policy": final.get("path_policy", ""),
                "tunnel_route_label": final.get("tunnel_route_label", ""),
                "tunnel_operator_category": final.get(
                    "tunnel_operator_category",
                    "",
                ),
                "post_gate_verdict": verdict,
                "post_gate_step_count": int(len(post)),
                "post_gate_gate_step": int(
                    _finite_or_zero(gate.get("trace_step_index", -1))
                ),
                "post_gate_wall_step": int(
                    _finite_or_zero(final.get("trace_wall_step_index", -1))
                ),
                "post_gate_gate_delta_q": float(gate_delta),
                "post_gate_gate_q_debt": float(gate_debt),
                "post_gate_gate_support": float(gate_support),
                "post_gate_gate_target_progress": float(gate_progress),
                "post_gate_final_delta_q": float(final_delta),
                "post_gate_final_support": float(final_support),
                "post_gate_final_target_progress": float(final_progress),
                "post_gate_best_delta_q": float(best_delta),
                "post_gate_best_delta_q_step": int(best_delta_step),
                "post_gate_best_support": float(best_support),
                "post_gate_best_target_progress": float(best_progress),
                "post_gate_best_delta_q_gain_from_gate": float(best_delta_gain),
                "post_gate_final_delta_q_gain_from_gate": float(final_delta_gain),
                "post_gate_best_support_gain_from_gate": float(
                    best_support - gate_support
                ),
                "post_gate_best_target_progress_gain_from_gate": float(
                    best_progress - gate_progress
                ),
                "post_gate_final_support_gain_from_gate": float(
                    final_support - gate_support
                ),
                "post_gate_final_target_progress_gain_from_gate": float(
                    final_progress - gate_progress
                ),
                "post_gate_recovered_after_gate": bool(recovered_after_gate),
                "post_gate_recovery_step_count": int(recovery_step_count),
                "post_gate_support_deepening_step_count": int(
                    support_deepening_step_count
                ),
                "post_gate_support_quality_tradeoff_step_count": int(
                    tradeoff_step_count
                ),
                "post_gate_plateau_step_count": int(plateau_step_count),
            }
        )
    return pd.DataFrame(out)


def classify_post_gate_recovery_move_rows(
    move_rows: pd.DataFrame,
    *,
    target_delta_q: float,
    target_support: float,
    target_progress: float,
    support_gate: float = 0.05,
    progress_margin: float = 0.005,
    min_q_recovery_gain: float = 1e-9,
    min_support_gain: float = 1e-9,
    min_progress_gain: float = 1e-9,
) -> pd.DataFrame:
    """Classify one-step recovery moves relative to a post-gate target state."""
    if move_rows.empty:
        return move_rows.copy()
    required = {
        "state_delta_q_vs_start",
        "state_support_distance_to_vanilla",
        "state_target_progress_from_vanilla",
    }
    missing = sorted(required.difference(move_rows.columns))
    if missing:
        raise KeyError(f"move_rows missing recovery columns: {missing}")
    out = move_rows.copy()
    delta = pd.to_numeric(out["state_delta_q_vs_start"], errors="coerce")
    support = pd.to_numeric(
        out["state_support_distance_to_vanilla"],
        errors="coerce",
    ).fillna(0.0)
    progress = pd.to_numeric(
        out["state_target_progress_from_vanilla"],
        errors="coerce",
    ).fillna(0.0)
    target_delta = float(target_delta_q)
    target_support_value = float(target_support)
    target_progress_value = float(target_progress)
    out["post_gate_move_delta_q_gain"] = delta - target_delta
    out["post_gate_move_support_gain"] = support - target_support_value
    out["post_gate_move_target_progress_gain"] = progress - target_progress_value
    out["post_gate_move_candidate_directed"] = (
        support >= float(support_gate)
    ) & (progress > float(progress_margin))
    out["post_gate_move_q_recovered"] = delta >= 0.0
    out["post_gate_move_support_retained"] = (
        support + float(min_support_gain) >= target_support_value
    )
    verdicts: list[str] = []
    for idx in out.index:
        q_gain = _finite_or_zero(out.loc[idx, "post_gate_move_delta_q_gain"])
        support_gain = _finite_or_zero(out.loc[idx, "post_gate_move_support_gain"])
        progress_gain = _finite_or_zero(
            out.loc[idx, "post_gate_move_target_progress_gain"]
        )
        support_retained = bool(out.loc[idx, "post_gate_move_support_retained"])
        q_recovered = bool(out.loc[idx, "post_gate_move_q_recovered"])
        if q_recovered and support_retained:
            verdicts.append(POST_GATE_RECOVERY_MOVE_RECOVERED)
        elif q_gain > float(min_q_recovery_gain) and support_retained:
            verdicts.append(POST_GATE_RECOVERY_MOVE_Q_GAIN)
        elif (
            support_gain > float(min_support_gain)
            or progress_gain > float(min_progress_gain)
        ) and q_gain < -float(min_q_recovery_gain):
            verdicts.append(POST_GATE_RECOVERY_MOVE_SUPPORT_TRADEOFF)
        elif q_gain > float(min_q_recovery_gain):
            verdicts.append(POST_GATE_RECOVERY_MOVE_SUPPORT_LOSS_Q_GAIN)
        elif q_gain < -float(min_q_recovery_gain):
            verdicts.append(POST_GATE_RECOVERY_MOVE_REGRESSION)
        else:
            verdicts.append(POST_GATE_RECOVERY_MOVE_PLATEAU)
    out["post_gate_move_verdict"] = verdicts
    return out


def summarize_tunneling_evidence_rows(path_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize tunneling labels by artifact and route class."""
    if path_rows.empty:
        return pd.DataFrame()
    group_columns = [
        column
        for column in ("artifact_label", "tunnel_route_label")
        if column in path_rows.columns
    ]
    if not group_columns:
        group_columns = ["tunnel_route_label"]
    out: list[dict[str, Any]] = []
    for keys, group in path_rows.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        context = dict(zip(group_columns, keys, strict=False))
        out.append(
            {
                **context,
                "rows": int(len(group)),
                "candidate_directed_rows": int(
                    group.get(
                        "tunnel_candidate_directed",
                        pd.Series(False, index=group.index),
                    )
                    .astype(bool)
                    .sum()
                ),
                "q_recovered_rows": int(
                    group.get("tunnel_q_recovered", pd.Series(False, index=group.index))
                    .astype(bool)
                    .sum()
                ),
                "q_wall_min": _pathway_quantile(group["path_q_wall"], 0.0),
                "q_wall_median": _pathway_quantile(group["path_q_wall"], 0.5),
                "q_wall_max": _pathway_quantile(group["path_q_wall"], 1.0),
                "debt_area_step_min": _pathway_quantile(
                    group["path_q_debt_area_step"],
                    0.0,
                ),
                "debt_area_step_median": _pathway_quantile(
                    group["path_q_debt_area_step"],
                    0.5,
                ),
                "debt_area_mutable_median": _pathway_quantile(
                    group["path_q_debt_area_mutable"],
                    0.5,
                ),
                "final_delta_q_max": _pathway_quantile(
                    group["path_final_delta_q_vs_start"],
                    1.0,
                ),
                "support_max": _pathway_quantile(
                    group["path_final_support_distance_to_vanilla"],
                    1.0,
                ),
                "target_progress_max": _pathway_quantile(
                    group["path_final_target_progress_from_vanilla"],
                    1.0,
                ),
                "recovered_shortcut_score_max": _pathway_quantile(
                    group["tunnel_recovered_shortcut_score"],
                    1.0,
                ),
                "recovery_slope_step_max": _pathway_quantile(
                    group["path_recovery_slope_per_step"],
                    1.0,
                ),
            }
        )
    return pd.DataFrame(out)


def rank_tunneling_operator_candidates(
    path_rows: pd.DataFrame,
    *,
    support_gate: float = 0.05,
    progress_margin: float = 0.005,
) -> pd.DataFrame:
    """Rank tunnel-like paths by operator-development usefulness.

    The score is deliberately diagnostic. Recovered tunnels are ranked first as
    seed candidates; unrecovered detours are kept as recovery-target candidates;
    below-gate partial-progress paths are kept as entrance probes.
    """
    if path_rows.empty:
        return path_rows.copy()
    required = {
        "tunnel_route_label",
        "path_q_debt_area_step",
        "path_final_delta_q_vs_start",
        "path_final_support_distance_to_vanilla",
        "path_final_target_progress_from_vanilla",
    }
    missing = sorted(required.difference(path_rows.columns))
    if missing:
        raise KeyError(f"path_rows missing required tunneling columns: {missing}")
    out = path_rows.copy()
    route = out["tunnel_route_label"].astype(str)
    support = _numeric_path_column(out, "path_final_support_distance_to_vanilla")
    progress = _numeric_path_column(out, "path_final_target_progress_from_vanilla")
    delta_q = _numeric_path_column(
        out,
        "path_final_delta_q_vs_start",
        default=math.nan,
    )
    debt_area = _numeric_path_column(out, "path_q_debt_area_step")
    mutable_area = _numeric_path_column(out, "path_q_debt_area_mutable")
    q_wall = _numeric_path_column(out, "path_q_wall")
    area_denom = debt_area.mask(debt_area <= 0.0, 1.0)
    mutable_denom = mutable_area.mask(mutable_area <= 0.0, 1.0)
    positive_delta = delta_q.clip(lower=0.0)
    quality_debt = (-delta_q).clip(lower=0.0)
    support_progress = support + progress

    categories: list[str] = []
    priorities: list[int] = []
    hints: list[str] = []
    for idx in out.index:
        label = route.loc[idx]
        if label == TUNNEL_ROUTE_RECOVERABLE:
            categories.append(TUNNEL_OPERATOR_RECOVERABLE_SEED)
            priorities.append(0)
            hints.append("seed_prefix_then_bounded_polish_and_tail_growth")
        elif label == TUNNEL_ROUTE_UNRECOVERED_DETOUR:
            categories.append(TUNNEL_OPERATOR_RECOVERY_TARGET)
            priorities.append(1)
            hints.append("gate_crossed_add_targeted_recovery_move")
        elif label == TUNNEL_ROUTE_PARTIAL_PROGRESS:
            categories.append(TUNNEL_OPERATOR_ENTRANCE_PROBE)
            priorities.append(2)
            hints.append("extend_partial_prefix_until_wall_entry")
        else:
            categories.append(TUNNEL_OPERATOR_BACKGROUND)
            priorities.append(3)
            hints.append("not_a_primary_tunneling_candidate")

    out["tunnel_operator_category"] = categories
    out["tunnel_operator_priority"] = priorities
    out["tunnel_operator_action_hint"] = hints
    out["tunnel_support_progress_score"] = support_progress
    out["tunnel_operator_efficiency_score"] = (
        positive_delta + support_progress
    ) / area_denom
    out["tunnel_operator_quality_debt"] = quality_debt
    out["tunnel_operator_quality_debt_per_area"] = quality_debt / area_denom
    out["tunnel_operator_mutable_area_penalty"] = mutable_area / mutable_denom.max()
    out["tunnel_operator_score"] = (
        out["tunnel_operator_efficiency_score"]
        - out["tunnel_operator_quality_debt_per_area"]
        - 0.05 * out["tunnel_operator_mutable_area_penalty"].fillna(0.0)
    )
    out["tunnel_operator_wall_to_area_ratio"] = q_wall / area_denom
    out["tunnel_operator_support_gate_margin"] = support - float(support_gate)
    out["tunnel_operator_progress_margin"] = progress - float(progress_margin)
    out["tunnel_operator_acceptance_ready"] = (
        route.eq(TUNNEL_ROUTE_RECOVERABLE)
        & (support >= float(support_gate))
        & (progress > float(progress_margin))
        & (delta_q >= 0.0)
    )
    return out.sort_values(
        [
            "tunnel_operator_priority",
            "tunnel_operator_score",
            "path_final_delta_q_vs_start",
            "path_final_support_distance_to_vanilla",
            "path_q_debt_area_step",
            "path_final_mutable_node_count",
        ],
        ascending=[True, False, False, False, True, True],
    ).reset_index(drop=True)


def select_tunneling_operator_candidates(
    path_rows: pd.DataFrame,
    *,
    max_rows_per_category: int = 20,
) -> pd.DataFrame:
    """Select top-ranked rows from each tunneling operator category."""
    if path_rows.empty:
        return path_rows.copy()
    if "tunnel_operator_category" not in path_rows.columns:
        rows = rank_tunneling_operator_candidates(path_rows)
    else:
        rows = path_rows.copy()
    selected: list[pd.DataFrame] = []
    for _, group in rows.groupby("tunnel_operator_category", sort=False):
        if group["tunnel_operator_category"].iloc[0] == TUNNEL_OPERATOR_BACKGROUND:
            continue
        selected.append(group.head(int(max_rows_per_category)))
    if not selected:
        return pd.DataFrame(columns=rows.columns)
    return pd.concat(selected, ignore_index=True)


def _pathway_quantile(values: pd.Series, q: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return math.nan
    return float(numeric.quantile(float(q)))


def _best_path_row(
    rows: pd.DataFrame,
    *,
    sort_columns: list[str],
    ascending: list[bool],
) -> pd.Series | None:
    if rows.empty:
        return None
    return rows.sort_values(sort_columns, ascending=ascending).iloc[0]


def summarize_pathway_wall_rows(
    path_rows: pd.DataFrame,
    *,
    support_gate: float = 0.05,
) -> pd.DataFrame:
    """Summarize path-level QF walls by source artifact and pair."""
    if path_rows.empty:
        return pd.DataFrame()
    path_rows = path_rows.copy()
    if "path_target_progress_per_wall_floor" not in path_rows.columns:
        denom = np.maximum(
            pd.to_numeric(path_rows["path_q_wall"], errors="coerce").fillna(0.0),
            1.0,
        )
        path_rows["path_target_progress_per_wall_floor"] = (
            pd.to_numeric(
                path_rows["path_final_target_progress_from_vanilla"],
                errors="coerce",
            ).fillna(0.0)
            / denom
        )
    group_columns = [column for column in ("source_label", "pair_id") if column in path_rows.columns]
    out: list[dict[str, Any]] = []
    for keys, group in path_rows.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        context = dict(zip(group_columns, keys, strict=False))
        support_gate_rows = group[
            group["path_final_support_distance_to_vanilla"] >= float(support_gate)
        ].copy()
        q_recovered = group[group["path_final_delta_q_vs_start"] >= 0.0].copy()
        support_q_recovered = support_gate_rows[
            support_gate_rows["path_final_delta_q_vs_start"] >= 0.0
        ].copy()
        min_wall_gate = _best_path_row(
            support_gate_rows,
            sort_columns=[
                "path_q_wall",
                "path_final_support_distance_to_vanilla",
                "path_final_target_progress_from_vanilla",
                "path_final_delta_q_vs_start",
            ],
            ascending=[True, False, False, False],
        )
        best_efficiency = _best_path_row(
            group,
            sort_columns=[
                "path_target_progress_per_wall_floor",
                "path_final_support_distance_to_vanilla",
                "path_final_delta_q_vs_start",
                "path_final_mutable_node_count",
            ],
            ascending=[False, False, False, True],
        )
        best_support = _best_path_row(
            group,
            sort_columns=[
                "path_final_support_distance_to_vanilla",
                "path_final_target_progress_from_vanilla",
                "path_final_delta_q_vs_start",
                "path_q_wall",
            ],
            ascending=[False, False, False, True],
        )

        row: dict[str, Any] = {
            **context,
            "path_rows": int(len(group)),
            "support_gate_rows": int(len(support_gate_rows)),
            "q_recovered_rows": int(len(q_recovered)),
            "support_gate_q_recovered_rows": int(len(support_q_recovered)),
            "wall_crossed_rows": int(group["path_wall_crossed"].sum()),
            "zero_wall_rows": int((group["path_q_wall"] <= 0.0).sum()),
            "q_wall_min": _pathway_quantile(group["path_q_wall"], 0.0),
            "q_wall_median": _pathway_quantile(group["path_q_wall"], 0.5),
            "q_wall_p90": _pathway_quantile(group["path_q_wall"], 0.9),
            "q_wall_max": _pathway_quantile(group["path_q_wall"], 1.0),
            "support_gate_q_wall_min": _pathway_quantile(
                support_gate_rows["path_q_wall"], 0.0
            ),
            "support_gate_q_wall_median": _pathway_quantile(
                support_gate_rows["path_q_wall"], 0.5
            ),
            "support_gate_q_wall_max": _pathway_quantile(
                support_gate_rows["path_q_wall"], 1.0
            ),
        }
        for prefix, best in (
            ("min_wall_gate", min_wall_gate),
            ("best_efficiency", best_efficiency),
            ("best_support", best_support),
        ):
            if best is None:
                row.update(
                    {
                        f"{prefix}_state_id": "",
                        f"{prefix}_q_wall": math.nan,
                        f"{prefix}_delta_q": math.nan,
                        f"{prefix}_support_distance_to_vanilla": math.nan,
                        f"{prefix}_target_progress": math.nan,
                        f"{prefix}_coverage": math.nan,
                        f"{prefix}_mutable_nodes": math.nan,
                        f"{prefix}_prefix_raw_barrier": math.nan,
                        f"{prefix}_wall_reduction_vs_prefix_raw_barrier": math.nan,
                    }
                )
                continue
            row.update(
                {
                    f"{prefix}_state_id": best["path_final_state_id"],
                    f"{prefix}_q_wall": float(best["path_q_wall"]),
                    f"{prefix}_delta_q": float(best["path_final_delta_q_vs_start"]),
                    f"{prefix}_support_distance_to_vanilla": float(
                        best["path_final_support_distance_to_vanilla"]
                    ),
                    f"{prefix}_target_progress": float(
                        best["path_final_target_progress_from_vanilla"]
                    ),
                    f"{prefix}_coverage": float(
                        best["path_final_target_coverage_fraction"]
                    ),
                    f"{prefix}_mutable_nodes": int(
                        _finite_or_zero(best["path_final_mutable_node_count"])
                    ),
                    f"{prefix}_prefix_raw_barrier": _finite_or_nan(
                        best.get("path_prefix_raw_barrier_input", math.nan)
                    ),
                    f"{prefix}_wall_reduction_vs_prefix_raw_barrier": _finite_or_nan(
                        best.get("path_wall_reduction_vs_prefix_raw_barrier", math.nan)
                    ),
                }
            )
        out.append(row)
    return pd.DataFrame(out)


def select_qf_wall_frontier(
    path_rows: pd.DataFrame,
    *,
    max_rows: int = 100,
) -> pd.DataFrame:
    """Return a Pareto-style frontier over low QF wall and final progress."""
    if path_rows.empty:
        return path_rows.copy()
    candidates = path_rows.sort_values(
        [
            "path_q_wall",
            "path_final_target_progress_from_vanilla",
            "path_final_support_distance_to_vanilla",
            "path_final_delta_q_vs_start",
            "path_final_mutable_node_count",
        ],
        ascending=[True, False, False, False, True],
    ).copy()
    maximize_columns = (
        "path_final_target_progress_from_vanilla",
        "path_final_support_distance_to_vanilla",
        "path_final_target_coverage_fraction",
        "path_final_delta_q_vs_start",
    )
    minimize_columns = ("path_q_wall", "path_final_mutable_node_count")
    kept: list[int] = []
    for idx, row in candidates.iterrows():
        dominated = False
        for kept_idx in kept:
            other = candidates.loc[kept_idx]
            no_worse = (
                all(float(other[column]) >= float(row[column]) for column in maximize_columns)
                and all(float(other[column]) <= float(row[column]) for column in minimize_columns)
            )
            strictly_better = (
                any(float(other[column]) > float(row[column]) for column in maximize_columns)
                or any(float(other[column]) < float(row[column]) for column in minimize_columns)
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            kept.append(idx)
    frontier = candidates.loc[kept].copy()
    frontier["path_wall_frontier_score"] = (
        frontier["path_final_target_progress_from_vanilla"].astype(float)
        + 0.50 * frontier["path_final_support_distance_to_vanilla"].astype(float)
        + 0.25 * frontier["path_final_target_coverage_fraction"].astype(float)
        - 0.01 * frontier["path_q_wall"].astype(float)
    )
    return frontier.sort_values(
        [
            "path_wall_frontier_score",
            "path_final_support_distance_to_vanilla",
            "path_final_delta_q_vs_start",
            "path_q_wall",
        ],
        ascending=[False, False, False, True],
    ).head(int(max_rows))


def score_branch_path_rows(
    path_rows: pd.DataFrame,
    *,
    barrier_floor: float = 1.0,
    mutable_floor: float = 1.0,
    wall_penalty: float = 0.01,
    mutable_penalty: float = 0.001,
    q_recovery_bonus: float = 0.05,
) -> pd.DataFrame:
    """Add diagnostic branch-search scores to path-level rows.

    The score is intentionally not a final acceptance metric. It favors source
    escape, target progress, and coverage while accounting for wall height and
    mutable-node cost. QF recovery is a small bonus so discovery can still keep
    unrecovered wall-crossing paths visible.
    """
    if path_rows.empty:
        return path_rows.copy()
    out = path_rows.copy()

    def numeric(column: str) -> pd.Series:
        return pd.to_numeric(out.get(column, 0.0), errors="coerce").fillna(0.0)

    support = numeric("path_final_support_distance_to_vanilla")
    progress = numeric("path_final_target_progress_from_vanilla")
    coverage = numeric("path_final_target_coverage_fraction")
    wall = numeric("path_q_wall").clip(lower=0.0)
    mutable = numeric("path_final_mutable_node_count").clip(lower=0.0)
    delta_q = numeric("path_final_delta_q_vs_start")
    wall_denom = np.maximum(float(barrier_floor), wall.to_numpy(dtype=np.float64))
    mutable_denom = np.maximum(float(mutable_floor), mutable.to_numpy(dtype=np.float64))
    out["path_support_per_wall_floor"] = support.to_numpy(dtype=np.float64) / wall_denom
    out["path_target_progress_per_wall_floor"] = (
        progress.to_numpy(dtype=np.float64) / wall_denom
    )
    out["path_support_per_mutable_node"] = (
        support.to_numpy(dtype=np.float64) / mutable_denom
    )
    out["path_target_progress_per_mutable_node"] = (
        progress.to_numpy(dtype=np.float64) / mutable_denom
    )
    out["path_q_recovered_flag"] = delta_q.to_numpy(dtype=np.float64) >= 0.0
    out["path_branch_discovery_score"] = (
        3.0 * support.to_numpy(dtype=np.float64)
        + 8.0 * progress.to_numpy(dtype=np.float64)
        + 0.25 * coverage.to_numpy(dtype=np.float64)
        + 0.50 * out["path_support_per_wall_floor"].to_numpy(dtype=np.float64)
        + 1.00 * out["path_target_progress_per_wall_floor"].to_numpy(dtype=np.float64)
        + 1.00 * out["path_support_per_mutable_node"].to_numpy(dtype=np.float64)
        + 5.00 * out["path_target_progress_per_mutable_node"].to_numpy(dtype=np.float64)
        + np.where(out["path_q_recovered_flag"].to_numpy(dtype=np.bool_), float(q_recovery_bonus), 0.0)
        - float(wall_penalty) * wall.to_numpy(dtype=np.float64)
        - float(mutable_penalty) * mutable.to_numpy(dtype=np.float64)
    )
    return out


def select_branch_path_rows(
    path_rows: pd.DataFrame,
    *,
    candidate_state_ids: list[str] | tuple[str, ...] | set[str] | None = None,
    beam_width: int = 5,
    diversity_columns: tuple[str, ...] = ("path_selection_policy",),
) -> pd.DataFrame:
    """Select branch-search terminal paths while preserving policy diversity."""
    if path_rows.empty or int(beam_width) <= 0:
        return path_rows.head(0).copy()
    scored = score_branch_path_rows(path_rows)
    if candidate_state_ids is not None:
        allowed = {str(value) for value in candidate_state_ids}
        scored = scored[
            scored["path_final_state_id"].astype(str).isin(allowed)
        ].copy()
    if scored.empty:
        return scored
    sort_columns = [
        "path_branch_discovery_score",
        "path_final_support_distance_to_vanilla",
        "path_final_target_progress_from_vanilla",
        "path_final_delta_q_vs_start",
        "path_q_wall",
        "path_final_mutable_node_count",
    ]
    ascending = [False, False, False, False, True, True]
    scored = scored.sort_values(sort_columns, ascending=ascending).copy()
    kept_indices: list[Any] = []
    seen_diversity: set[tuple[str, ...]] = set()
    for idx, row in scored.iterrows():
        if len(kept_indices) >= int(beam_width):
            break
        key = tuple(str(row.get(column, "")) for column in diversity_columns)
        if key in seen_diversity:
            continue
        seen_diversity.add(key)
        kept_indices.append(idx)
    if len(kept_indices) < int(beam_width):
        for idx in scored.index:
            if len(kept_indices) >= int(beam_width):
                break
            if idx not in kept_indices:
                kept_indices.append(idx)
    return scored.loc[kept_indices].sort_values(sort_columns, ascending=ascending)


def _split_semicolon_labels(value: Any) -> set[str]:
    text = str(value or "")
    if text.lower() == "nan":
        return set()
    return {part.strip() for part in text.split(";") if part.strip()}


def _path_state_ids(value: Any) -> tuple[str, ...]:
    text = str(value or "")
    if not text or text.lower() == "nan":
        return ()
    return tuple(part.strip() for part in text.split("|") if part.strip())


def classify_branch_greedy_failure_rows(
    path_rows: pd.DataFrame,
    *,
    state_rows: pd.DataFrame | None = None,
    control_rows: pd.DataFrame | None = None,
    support_gate: float = 0.05,
    progress_margin: float = 0.005,
    support_margin: float = 0.01,
    material_delta_q: float = 1.0,
    q_wall_floor: float = 0.1,
    closure_ratio_threshold: float = 4.0,
) -> pd.DataFrame:
    """Classify why promising branch paths are not found by simple greedies.

    The classifier is diagnostic-only. It explains already-generated branch
    paths by combining prefix-level greedy labels, path-level QF wall recovery,
    and optional same-case seed/iteration controls.
    """
    if path_rows.empty:
        return path_rows.copy()

    out = score_branch_path_rows(path_rows).copy()
    state_index = (
        state_rows.set_index("state_id", drop=False)
        if state_rows is not None and not state_rows.empty and "state_id" in state_rows
        else pd.DataFrame()
    )
    control_by_pair: dict[str, pd.DataFrame] = {}
    if control_rows is not None and not control_rows.empty and "pair_id" in control_rows:
        controls = control_rows.copy()
        known_pair_ids = [
            str(value)
            for value in controls["pair_id"].dropna().unique().tolist()
            if str(value).strip() and str(value).lower() != "nan"
        ]
        if len(known_pair_ids) == 1:
            controls["pair_id"] = controls["pair_id"].fillna(known_pair_ids[0])
        if "row_type" in controls:
            controls = controls[controls["row_type"].astype(str).eq("control")].copy()
        for pair_id, group in controls.groupby("pair_id", sort=False, dropna=False):
            control_by_pair[str(pair_id)] = group.copy()

    root_labels: list[str] = []
    max_context_ratios: list[float] = []
    max_marginal_q_debts: list[float] = []
    failure_labels: list[str] = []
    q_greedy_misses: list[bool] = []
    progress_greedy_misses: list[bool] = []
    closure_compound_misses: list[bool] = []
    polish_recovery_misses: list[bool] = []
    path_candidate_directed: list[bool] = []
    control_statuses: list[str] = []
    candidate_directed_control_counts: list[int] = []
    best_control_delta_qs: list[float] = []
    branch_delta_q_minus_best_controls: list[float] = []
    best_candidate_directed_control_delta_qs: list[float] = []
    branch_delta_q_minus_best_candidate_controls: list[float] = []

    for _, row in out.iterrows():
        root_id = str(row.get("path_root_state_id", ""))
        path_ids = _path_state_ids(row.get("path_state_ids", ""))
        path_states = (
            state_index.loc[[sid for sid in path_ids if sid in state_index.index]]
            if not state_index.empty and path_ids
            else pd.DataFrame()
        )
        root_label_set: set[str] = set()
        if not state_index.empty and root_id in state_index.index:
            root = state_index.loc[root_id]
            if isinstance(root, pd.DataFrame):
                root = root.iloc[0]
            root_label_set = _split_semicolon_labels(root.get("greedy_failure_labels", ""))
        if not root_label_set and not path_states.empty and "greedy_failure_labels" in path_states:
            root_label_set = set().union(
                *[
                    _split_semicolon_labels(value)
                    for value in path_states["greedy_failure_labels"].tolist()
                ]
            )

        if not path_states.empty and "context_to_action_ratio" in path_states:
            max_context_ratio = float(
                pd.to_numeric(
                    path_states["context_to_action_ratio"],
                    errors="coerce",
                )
                .fillna(0.0)
                .max()
            )
        else:
            max_context_ratio = math.nan
        if not path_states.empty and "marginal_q_debt" in path_states:
            max_marginal_q_debt = float(
                pd.to_numeric(path_states["marginal_q_debt"], errors="coerce")
                .fillna(0.0)
                .max()
            )
        else:
            max_marginal_q_debt = math.nan

        support = _finite_or_zero(row.get("path_final_support_distance_to_vanilla"))
        progress = _finite_or_zero(row.get("path_final_target_progress_from_vanilla"))
        delta_q = _finite_or_zero(row.get("path_final_delta_q_vs_start"))
        q_wall = _finite_or_zero(row.get("path_q_wall"))
        q_recovery = _finite_or_zero(row.get("path_q_recovery_from_wall"))
        candidate_directed = (
            progress > float(progress_margin) and support >= float(support_gate)
        )

        q_miss = BARRIER_Q_GREEDY_MISS in root_label_set and progress > 0.0
        progress_miss = (
            BARRIER_PROGRESS_GREEDY_MISS in root_label_set and progress > 0.0
        )
        closure_miss = BARRIER_CLOSURE_COMPOUND_MISS in root_label_set or (
            math.isfinite(max_context_ratio)
            and max_context_ratio >= float(closure_ratio_threshold)
        )
        polish_miss = (
            BARRIER_POLISH_RECOVERY_MISS in root_label_set
            and q_wall > float(q_wall_floor)
            and q_recovery > float(q_wall_floor)
            and delta_q >= 0.0
        )

        labels: list[str] = []
        if q_miss:
            labels.append(BARRIER_Q_GREEDY_MISS)
        if progress_miss:
            labels.append(BARRIER_PROGRESS_GREEDY_MISS)
        if closure_miss:
            labels.append(BARRIER_CLOSURE_COMPOUND_MISS)
        if polish_miss:
            labels.append(BARRIER_POLISH_RECOVERY_MISS)
        if not labels:
            labels.append(BARRIER_GREEDY_VISIBLE)

        pair_controls = control_by_pair.get(str(row.get("pair_id", "")), pd.DataFrame())
        if pair_controls.empty:
            control_status = GREEDY_CONTROL_NOT_CHECKED
            candidate_control_count = 0
            best_control_delta_q = math.nan
            branch_minus_best = math.nan
            best_candidate_delta = math.nan
            branch_minus_candidate = math.nan
        else:
            control_delta = pd.to_numeric(
                pair_controls.get("delta_q_vs_vanilla", pd.Series(dtype=float)),
                errors="coerce",
            )
            best_control_delta_q = (
                float(control_delta.max()) if not control_delta.dropna().empty else math.nan
            )
            branch_minus_best = (
                float(delta_q - best_control_delta_q)
                if math.isfinite(best_control_delta_q)
                else math.nan
            )
            control_progress = pd.to_numeric(
                pair_controls.get(
                    "target_progress_from_vanilla",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            )
            control_support = pd.to_numeric(
                pair_controls.get(
                    "support_distance_to_vanilla",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            )
            candidate_controls = pair_controls[
                control_progress.gt(float(progress_margin))
                & control_support.ge(max(0.0, support - float(support_margin)))
            ].copy()
            candidate_control_count = int(len(candidate_controls))
            if candidate_controls.empty:
                best_candidate_delta = math.nan
                branch_minus_candidate = math.nan
            else:
                candidate_delta = pd.to_numeric(
                    candidate_controls.get("delta_q_vs_vanilla", pd.Series(dtype=float)),
                    errors="coerce",
                )
                best_candidate_delta = (
                    float(candidate_delta.max())
                    if not candidate_delta.dropna().empty
                    else math.nan
                )
                branch_minus_candidate = (
                    float(delta_q - best_candidate_delta)
                    if math.isfinite(best_candidate_delta)
                    else math.nan
                )

            if not candidate_directed:
                control_status = GREEDY_CONTROL_NOT_CANDIDATE_DIRECTED
            elif candidate_control_count > 0 and branch_minus_candidate >= -float(
                material_delta_q
            ):
                control_status = GREEDY_CONTROL_REPRODUCED
            elif candidate_control_count > 0:
                control_status = GREEDY_CONTROL_BRANCH_LAGS_CANDIDATE_DIRECTED_CONTROL
            elif math.isfinite(best_control_delta_q) and (
                best_control_delta_q - delta_q
            ) >= float(material_delta_q):
                control_status = GREEDY_CONTROL_BRANCH_UNIQUE_QUALITY_LAG
            else:
                control_status = GREEDY_CONTROL_BRANCH_UNIQUE

        root_labels.append(";".join(sorted(root_label_set)) if root_label_set else "")
        max_context_ratios.append(max_context_ratio)
        max_marginal_q_debts.append(max_marginal_q_debt)
        failure_labels.append(";".join(labels))
        q_greedy_misses.append(q_miss)
        progress_greedy_misses.append(progress_miss)
        closure_compound_misses.append(closure_miss)
        polish_recovery_misses.append(polish_miss)
        path_candidate_directed.append(candidate_directed)
        control_statuses.append(control_status)
        candidate_directed_control_counts.append(candidate_control_count)
        best_control_delta_qs.append(best_control_delta_q)
        branch_delta_q_minus_best_controls.append(branch_minus_best)
        best_candidate_directed_control_delta_qs.append(best_candidate_delta)
        branch_delta_q_minus_best_candidate_controls.append(branch_minus_candidate)

    out["root_greedy_failure_labels"] = root_labels
    out["path_max_context_to_action_ratio"] = max_context_ratios
    out["path_max_marginal_q_debt"] = max_marginal_q_debts
    out["path_candidate_directed"] = path_candidate_directed
    out["failure_labels"] = failure_labels
    out["q_greedy_miss"] = q_greedy_misses
    out["progress_greedy_miss"] = progress_greedy_misses
    out["closure_compound_miss"] = closure_compound_misses
    out["polish_recovery_miss"] = polish_recovery_misses
    out["candidate_directed_control_count"] = candidate_directed_control_counts
    out["best_control_delta_q_vs_vanilla"] = best_control_delta_qs
    out["branch_delta_q_minus_best_control"] = branch_delta_q_minus_best_controls
    out["best_candidate_directed_control_delta_q"] = (
        best_candidate_directed_control_delta_qs
    )
    out["branch_delta_q_minus_best_candidate_directed_control"] = (
        branch_delta_q_minus_best_candidate_controls
    )
    out["control_comparison_status"] = control_statuses
    return out


def summarize_greedy_failure_rows(classified_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize path-level greedy failure labels by case and pair."""
    if classified_rows.empty:
        return pd.DataFrame()
    out: list[dict[str, Any]] = []
    group_columns = [
        column
        for column in ("case", "pair_id")
        if column in classified_rows.columns
    ]
    grouped = (
        classified_rows.groupby(group_columns, sort=True, dropna=False)
        if group_columns
        else [((), classified_rows)]
    )
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, Any] = {
            column: value for column, value in zip(group_columns, keys, strict=False)
        }
        candidate_directed = group[
            group.get("path_candidate_directed", pd.Series(False, index=group.index)).astype(bool)
        ].copy()
        recovered = group[
            group.get(
                "path_support_gate_q_recovered",
                pd.Series(False, index=group.index),
            ).astype(bool)
        ].copy()
        row.update(
            {
                "path_rows": int(len(group)),
                "candidate_directed_rows": int(len(candidate_directed)),
                "support_gate_q_recovered_rows": int(len(recovered)),
                "q_greedy_miss_rows": int(group.get("q_greedy_miss", False).sum()),
                "progress_greedy_miss_rows": int(
                    group.get("progress_greedy_miss", False).sum()
                ),
                "closure_compound_miss_rows": int(
                    group.get("closure_compound_miss", False).sum()
                ),
                "polish_recovery_miss_rows": int(
                    group.get("polish_recovery_miss", False).sum()
                ),
                "unique_candidate_directed_quality_lag_rows": int(
                    group.get("control_comparison_status", pd.Series("", index=group.index))
                    .astype(str)
                    .eq(GREEDY_CONTROL_BRANCH_UNIQUE_QUALITY_LAG)
                    .sum()
                ),
            }
        )
        best = group.sort_values(
            [
                "path_branch_discovery_score",
                "path_final_support_distance_to_vanilla",
                "path_final_delta_q_vs_start",
                "path_q_wall",
            ],
            ascending=[False, False, False, True],
        ).iloc[0]
        row.update(
            {
                "best_state_id": best.get("path_final_state_id", ""),
                "best_failure_labels": best.get("failure_labels", ""),
                "best_control_status": best.get("control_comparison_status", ""),
                "best_delta_q": _finite_or_nan(
                    best.get("path_final_delta_q_vs_start", math.nan)
                ),
                "best_support": _finite_or_nan(
                    best.get("path_final_support_distance_to_vanilla", math.nan)
                ),
                "best_target_progress": _finite_or_nan(
                    best.get("path_final_target_progress_from_vanilla", math.nan)
                ),
                "best_q_wall": _finite_or_nan(best.get("path_q_wall", math.nan)),
                "best_mutable": int(
                    _finite_or_zero(best.get("path_final_mutable_node_count", 0))
                ),
            }
        )
        out.append(row)
    return pd.DataFrame(out)


def annotate_wall_route_families(
    classified_rows: pd.DataFrame,
    *,
    support_gate: float = 0.05,
    progress_margin: float = 0.005,
    side_support_fraction: float = 0.75,
    wall_round_digits: int = 6,
) -> pd.DataFrame:
    """Annotate path rows with wall-family and side-route diagnostics."""
    if classified_rows.empty:
        return classified_rows.copy()
    out = classified_rows.copy()
    support = pd.to_numeric(
        out.get("path_final_support_distance_to_vanilla", 0.0),
        errors="coerce",
    ).fillna(0.0)
    progress = pd.to_numeric(
        out.get("path_final_target_progress_from_vanilla", 0.0),
        errors="coerce",
    ).fillna(0.0)
    q_recovered = out.get(
        "path_q_recovered",
        out.get("path_support_gate_q_recovered", pd.Series(False, index=out.index)),
    ).astype(bool)
    candidate_directed = out.get(
        "path_candidate_directed",
        pd.Series(False, index=out.index),
    ).astype(bool)
    side_support_floor = float(support_gate) * float(side_support_fraction)
    out["wall_support_gate"] = support.ge(float(support_gate))
    out["wall_partial_progress"] = progress.gt(float(progress_margin))
    out["wall_side_route_candidate"] = (
        ~candidate_directed
        & q_recovered
        & progress.gt(float(progress_margin))
        & support.ge(float(side_support_floor))
    )
    wall = pd.to_numeric(out.get("path_q_wall", 0.0), errors="coerce").fillna(0.0)
    out["wall_key"] = wall.round(int(wall_round_digits)).map(
        lambda value: f"{float(value):.{int(wall_round_digits)}f}"
    )
    out["wall_entry_key"] = (
        out.get("path_prefix_rank", pd.Series("", index=out.index)).astype(str)
        + "|"
        + out["wall_key"].astype(str)
    )
    return out


def summarize_wall_route_families(
    classified_rows: pd.DataFrame,
    *,
    support_gate: float = 0.05,
    progress_margin: float = 0.005,
    side_support_fraction: float = 0.75,
    wall_round_digits: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize whether the observed wall has alternate lower-wall routes."""
    rows = annotate_wall_route_families(
        classified_rows,
        support_gate=support_gate,
        progress_margin=progress_margin,
        side_support_fraction=side_support_fraction,
        wall_round_digits=wall_round_digits,
    )
    if rows.empty:
        return rows, pd.DataFrame(), pd.DataFrame()

    family_out: list[dict[str, Any]] = []
    family_groups = rows.groupby(
        ["case", "pair_id", "wall_entry_key"],
        sort=True,
        dropna=False,
    )
    for (case, pair_id, wall_entry_key), group in family_groups:
        candidate_directed = group[group["path_candidate_directed"].astype(bool)]
        side_routes = group[group["wall_side_route_candidate"].astype(bool)]
        best = group.sort_values(
            [
                "path_branch_discovery_score",
                "path_final_support_distance_to_vanilla",
                "path_final_delta_q_vs_start",
                "path_q_wall",
            ],
            ascending=[False, False, False, True],
        ).iloc[0]
        family_out.append(
            {
                "case": case,
                "pair_id": pair_id,
                "wall_entry_key": wall_entry_key,
                "prefix_rank": int(_finite_or_zero(best.get("path_prefix_rank", 0))),
                "wall_key": best.get("wall_key", ""),
                "rows": int(len(group)),
                "candidate_directed_rows": int(len(candidate_directed)),
                "side_route_candidate_rows": int(len(side_routes)),
                "support_gate_rows": int(group["wall_support_gate"].sum()),
                "partial_progress_rows": int(group["wall_partial_progress"].sum()),
                "q_recovered_rows": int(
                    group.get("path_q_recovered", pd.Series(False, index=group.index))
                    .astype(bool)
                    .sum()
                ),
                "support_max": _finite_or_nan(
                    group["path_final_support_distance_to_vanilla"].max()
                ),
                "target_progress_max": _finite_or_nan(
                    group["path_final_target_progress_from_vanilla"].max()
                ),
                "delta_q_max": _finite_or_nan(group["path_final_delta_q_vs_start"].max()),
                "q_wall_min": _finite_or_nan(group["path_q_wall"].min()),
                "q_wall_max": _finite_or_nan(group["path_q_wall"].max()),
                "mutable_min": int(
                    _finite_or_zero(group["path_final_mutable_node_count"].min())
                ),
                "mutable_max": int(
                    _finite_or_zero(group["path_final_mutable_node_count"].max())
                ),
                "best_state_id": best.get("path_final_state_id", ""),
                "best_failure_labels": best.get("failure_labels", ""),
                "best_control_status": best.get("control_comparison_status", ""),
            }
        )
    family_rows = pd.DataFrame(family_out)

    prefix_out: list[dict[str, Any]] = []
    for (case, pair_id, prefix_rank), group in rows.groupby(
        ["case", "pair_id", "path_prefix_rank"],
        sort=True,
        dropna=False,
    ):
        prefix_out.append(
            {
                "case": case,
                "pair_id": pair_id,
                "prefix_rank": int(_finite_or_zero(prefix_rank)),
                "rows": int(len(group)),
                "candidate_directed_rows": int(
                    group["path_candidate_directed"].astype(bool).sum()
                ),
                "side_route_candidate_rows": int(group["wall_side_route_candidate"].sum()),
                "support_gate_rows": int(group["wall_support_gate"].sum()),
                "partial_progress_rows": int(group["wall_partial_progress"].sum()),
                "support_max": _finite_or_nan(
                    group["path_final_support_distance_to_vanilla"].max()
                ),
                "target_progress_max": _finite_or_nan(
                    group["path_final_target_progress_from_vanilla"].max()
                ),
                "delta_q_max": _finite_or_nan(group["path_final_delta_q_vs_start"].max()),
                "q_wall_min": _finite_or_nan(group["path_q_wall"].min()),
                "q_wall_max": _finite_or_nan(group["path_q_wall"].max()),
            }
        )
    prefix_rows = pd.DataFrame(prefix_out)

    summary_out: list[dict[str, Any]] = []
    for (case, pair_id), group in rows.groupby(["case", "pair_id"], sort=True, dropna=False):
        directed = group[group["path_candidate_directed"].astype(bool)]
        side = group[group["wall_side_route_candidate"].astype(bool)]
        directed_entries = (
            directed["wall_entry_key"].nunique() if not directed.empty else 0
        )
        directed_walls = directed["wall_key"].nunique() if not directed.empty else 0
        directed_prefixes = (
            directed["path_prefix_rank"].nunique() if not directed.empty else 0
        )
        min_directed_wall = (
            _finite_or_nan(directed["path_q_wall"].min())
            if not directed.empty
            else math.nan
        )
        lower_wall_side = (
            side[pd.to_numeric(side["path_q_wall"], errors="coerce") < min_directed_wall]
            if math.isfinite(min_directed_wall)
            else side.head(0)
        )
        if directed_entries == 0 and not side.empty:
            verdict = "no_candidate_wall_partial_side_routes"
        elif directed_entries <= 1 and not lower_wall_side.empty:
            verdict = "single_observed_candidate_wall_with_lower_wall_side_routes"
        elif directed_entries <= 1:
            verdict = "single_observed_candidate_wall_no_lower_wall_side_route"
        else:
            verdict = "multiple_observed_candidate_wall_entries"
        summary_out.append(
            {
                "case": case,
                "pair_id": pair_id,
                "path_rows": int(len(group)),
                "candidate_directed_rows": int(len(directed)),
                "candidate_directed_wall_entries": int(directed_entries),
                "candidate_directed_wall_values": int(directed_walls),
                "candidate_directed_prefixes": int(directed_prefixes),
                "side_route_candidate_rows": int(len(side)),
                "lower_wall_side_route_rows": int(len(lower_wall_side)),
                "min_candidate_directed_wall": float(min_directed_wall),
                "max_candidate_directed_support": _finite_or_nan(
                    directed["path_final_support_distance_to_vanilla"].max()
                )
                if not directed.empty
                else math.nan,
                "max_side_route_support": _finite_or_nan(
                    side["path_final_support_distance_to_vanilla"].max()
                )
                if not side.empty
                else math.nan,
                "max_side_route_progress": _finite_or_nan(
                    side["path_final_target_progress_from_vanilla"].max()
                )
                if not side.empty
                else math.nan,
                "wall_route_verdict": verdict,
            }
        )
    summary_rows = pd.DataFrame(summary_out)
    return family_rows, prefix_rows, summary_rows


def state_public_row(
    *,
    state: TransitionSearchState,
    metrics: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    return {
        **context,
        "state_id": state.state_id,
        "parent_state_id": state.parent_state_id,
        "depth": int(state.depth),
        "prefix_rank": int(state.prefix_rank),
        "prefix_unit_ids": state.prefix_unit_ids,
        "action_type": state.action_type,
        "action_params": state.action_params,
        "applied_actions": ",".join(state.applied_actions),
        "elapsed_sec": float(state.elapsed_sec),
        **metrics,
    }


def edge_public_row(
    *,
    parent_state_id: str,
    child_state_id: str,
    action: TransitionAction,
    context: dict[str, Any],
) -> dict[str, Any]:
    return {
        **context,
        "parent_state_id": parent_state_id,
        "child_state_id": child_state_id,
        "action_type": action.action_type,
        "action_params": action.action_params,
        "action_node_count": int(
            unique_sorted_u32(
                [] if action.action_nodes is None else action.action_nodes
            ).size
        ),
        "context_node_count": int(unique_sorted_u32(action.context_nodes).size),
    }


def select_search_beam(
    states: list[TransitionSearchState],
    rows: pd.DataFrame,
    *,
    beam_width: int,
    search_policy: str = SEARCH_POLICY_STATE_GREEDY,
) -> list[TransitionSearchState]:
    if not states or rows.empty:
        return []
    score_column = search_policy_score_column(search_policy)
    row_by_state = rows.set_index("state_id")
    valid = [state for state in states if state.state_id in row_by_state.index]
    if search_policy == SEARCH_POLICY_REACHABILITY_FIRST:
        valid.sort(
            key=lambda state: (
                float(row_by_state.loc[state.state_id, score_column]),
                float(row_by_state.loc[state.state_id, "state_support_distance_to_vanilla"]),
                float(row_by_state.loc[state.state_id, "state_target_progress_from_vanilla"]),
                float(row_by_state.loc[state.state_id, "target_coverage_fraction"]),
                float(row_by_state.loc[state.state_id, "state_delta_q_vs_start"]),
            ),
            reverse=True,
        )
    else:
        valid.sort(
            key=lambda state: (
                float(row_by_state.loc[state.state_id, score_column]),
                float(row_by_state.loc[state.state_id, "state_target_progress_from_vanilla"]),
                float(row_by_state.loc[state.state_id, "state_support_distance_to_vanilla"]),
                float(row_by_state.loc[state.state_id, "state_delta_q_vs_start"]),
                -float(row_by_state.loc[state.state_id, "mutable_node_count"]),
            ),
            reverse=True,
        )
    return valid[: int(beam_width)]


def select_pareto_rows(
    rows: pd.DataFrame,
    *,
    max_rows: int = 50,
    search_policy: str = SEARCH_POLICY_STATE_GREEDY,
) -> pd.DataFrame:
    """Select non-dominated search rows using quality, progress, shift, and cost."""
    if rows.empty:
        return rows.copy()
    score_column = search_policy_score_column(search_policy)
    if search_policy == SEARCH_POLICY_REACHABILITY_FIRST:
        candidates = rows.sort_values(
            [
                "state_support_distance_to_vanilla",
                "state_target_progress_from_vanilla",
                "target_coverage_fraction",
                "state_delta_q_vs_start",
            ],
            ascending=[False, False, False, False],
        ).copy()
        dominance_columns = (
            "state_support_distance_to_vanilla",
            "state_target_progress_from_vanilla",
            "target_coverage_fraction",
        )
    else:
        candidates = rows.sort_values(
            [
                "state_delta_q_vs_start",
                "state_target_progress_from_vanilla",
                "state_support_distance_to_vanilla",
                "mutable_node_count",
            ],
            ascending=[False, False, False, True],
        ).copy()
        dominance_columns = (
            "state_delta_q_vs_start",
            "state_target_progress_from_vanilla",
            "state_support_distance_to_vanilla",
        )
    kept: list[int] = []
    for idx, row in candidates.iterrows():
        dominated = False
        for kept_idx in kept:
            other = candidates.loc[kept_idx]
            no_worse = (
                all(float(other[column]) >= float(row[column]) for column in dominance_columns)
                and int(other["mutable_node_count"]) <= int(row["mutable_node_count"])
            )
            strictly_better = (
                any(float(other[column]) > float(row[column]) for column in dominance_columns)
                or int(other["mutable_node_count"]) < int(row["mutable_node_count"])
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            kept.append(idx)
    if search_policy == SEARCH_POLICY_REACHABILITY_FIRST:
        sort_columns = [
            score_column,
            "state_support_distance_to_vanilla",
            "state_target_progress_from_vanilla",
            "target_coverage_fraction",
        ]
    else:
        sort_columns = [
            score_column,
            "state_delta_q_vs_start",
            "state_target_progress_from_vanilla",
            "state_support_distance_to_vanilla",
        ]
    return candidates.loc[kept].sort_values(
        sort_columns,
        ascending=[False] * len(sort_columns),
    ).head(int(max_rows))


def make_child_state(
    *,
    parent: TransitionSearchState,
    action: TransitionAction,
    membership: np.ndarray,
    quality: float,
    elapsed_sec: float,
    child_index: int,
) -> TransitionSearchState:
    added_action_nodes = unique_sorted_u32(
        [] if action.action_nodes is None else action.action_nodes
    )
    action_nodes = unique_sorted_u32(
        np.concatenate([parent.action_nodes, added_action_nodes])
    )
    covered_target_nodes = unique_sorted_u32(
        np.concatenate(
            [
                parent.covered_target_nodes,
                intersect_sorted_u32(added_action_nodes, parent.target_nodes),
            ]
        )
    )
    context_nodes = unique_sorted_u32(
        np.concatenate([parent.context_nodes, action.context_nodes])
    )
    mutable_nodes = unique_sorted_u32(
        np.concatenate([parent.mutable_nodes, action.context_nodes, added_action_nodes])
    )
    action_name = (
        f"{action.action_type}:a{int(added_action_nodes.size)}:"
        f"c{int(context_nodes.size)}"
    )
    return TransitionSearchState(
        state_id=f"{parent.state_id}/{action.action_type}:{int(child_index)}",
        parent_state_id=parent.state_id,
        depth=int(parent.depth + 1),
        prefix_rank=int(parent.prefix_rank),
        prefix_unit_ids=parent.prefix_unit_ids,
        action_type=action.action_type,
        action_params=action.action_params,
        membership=compact_membership(membership),
        quality=float(quality),
        direct_nodes=action_nodes,
        mutable_nodes=mutable_nodes,
        context_nodes=context_nodes,
        applied_actions=tuple([*parent.applied_actions, action_name]),
        elapsed_sec=float(elapsed_sec),
        target_nodes=unique_sorted_u32(parent.target_nodes),
        action_nodes=action_nodes,
        covered_target_nodes=covered_target_nodes,
    )


def make_prefix_state(
    *,
    state_id: str,
    prefix_rank: int,
    prefix_unit_ids: str,
    membership: np.ndarray,
    quality: float,
    direct_nodes: np.ndarray,
    mutable_nodes: np.ndarray,
    target_nodes: np.ndarray | None = None,
    action_nodes: np.ndarray | None = None,
    covered_target_nodes: np.ndarray | None = None,
) -> TransitionSearchState:
    direct = unique_sorted_u32(direct_nodes)
    target = unique_sorted_u32(direct if target_nodes is None else target_nodes)
    action = unique_sorted_u32(direct if action_nodes is None else action_nodes)
    covered = unique_sorted_u32(
        intersect_sorted_u32(action, target)
        if covered_target_nodes is None
        else covered_target_nodes
    )
    mutable = unique_sorted_u32(mutable_nodes)
    return TransitionSearchState(
        state_id=state_id,
        parent_state_id="",
        depth=0,
        prefix_rank=int(prefix_rank),
        prefix_unit_ids=str(prefix_unit_ids),
        action_type=ACTION_PREFIX_ONLY,
        action_params="",
        membership=compact_membership(membership),
        quality=float(quality),
        direct_nodes=direct,
        target_nodes=target,
        action_nodes=action,
        covered_target_nodes=covered,
        mutable_nodes=mutable,
        context_nodes=np.asarray([], dtype=np.uint32),
        applied_actions=(ACTION_PREFIX_ONLY,),
        elapsed_sec=0.0,
    )


def polish_state(
    *,
    graph: Any,
    membership: np.ndarray,
    mutable_nodes: np.ndarray,
    resolution: float,
    seed: int,
    n_iterations: int,
    randomness: float,
) -> tuple[np.ndarray, float, float]:
    import time

    compacted = compact_membership(membership)
    if int(n_iterations) <= 0 or np.asarray(mutable_nodes).size == 0:
        quality = float(graph.cpm_quality(compacted, resolution=float(resolution)))
        return compacted, quality, 0.0
    start = time.perf_counter()
    result = graph.run_leiden(
        resolution=float(resolution),
        seed=int(seed),
        n_iterations=int(n_iterations),
        randomness=float(randomness),
        initial_membership=compacted,
        fixed_nodes=fixed_outside(int(compacted.size), mutable_nodes),
    )
    elapsed = float(time.perf_counter() - start)
    return (
        compact_membership(np.asarray(result.membership, dtype=np.uint64)),
        float(result.quality),
        elapsed,
    )
