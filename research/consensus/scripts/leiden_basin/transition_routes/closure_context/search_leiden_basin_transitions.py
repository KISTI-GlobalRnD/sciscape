#!/usr/bin/env python3
"""Search small closure/context combinations for Leiden basin transitions."""

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

from collect_leiden_vanilla_reachability_sweep import (  # noqa: E402
    _load_graph,
    _read_candidate_rows,
    compatible_sketch_nodes,
)
from evaluate_leiden_basin_polish_prefixes import (  # noqa: E402
    select_prefix_rows,
)
from analyze_leiden_basin_barrier_aware_pathways import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PREFIX_DIR,
    PREFIX_ROWS_FILENAME as BARRIER_PREFIX_ROWS_FILENAME,
)
from profile_leiden_basin_ordered_flips import (  # noqa: E402
    UNIT_ROWS_FILENAME,
)
from profile_leiden_basin_ordered_flips_batch import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PROFILE_BATCH_DIR,
)
from run_leiden_basin_transition_operator_pilot import (  # noqa: E402
    DEFAULT_CANDIDATE_DIRS,
    DEFAULT_VANILLA_DIR,
    VANILLA_ROWS_FILENAME,
    _find_candidate_row,
    _find_vanilla_row,
    _recreate_candidate,
    _run_leiden,
    _safe_int,
)
from sciscape.clustering.leiden_basin_profile import (  # noqa: E402
    apply_prefix_units,
    score_membership,
    support_distance,
    v_only_support_nodes,
)
from sciscape.clustering.leiden_basin_search import (  # noqa: E402
    ACTION_BOUNDARY_SHELL_TOPK,
    ACTION_CANDIDATE_CLOSURE_TOPK,
    ACTION_PREFIX_ONLY,
    ACTION_REMAINING_TARGET_TOPK,
    ACTION_REMAINING_TARGET_UNIT_TOPK,
    ACTION_VANILLA_CLOSURE_TOPK,
    SEARCH_POLICIES,
    SEARCH_POLICY_STATE_GREEDY,
    TARGET_UNIT_TYPES,
    TransitionAction,
    TransitionSearchState,
    build_context_actions,
    build_remaining_target_actions,
    build_remaining_target_unit_actions,
    build_target_unit_rows,
    edge_public_row,
    make_child_state,
    make_prefix_state,
    pathway_marginal_metrics,
    polish_state,
    prefix_direct_nodes,
    search_state_metrics,
    search_policy_score_column,
    select_pareto_rows,
    select_search_beam,
    state_public_row,
    transplant_action_nodes,
    unique_sorted_u32,
)

DEFAULT_OUTPUT_DIR = DEFAULT_PROFILE_BATCH_DIR.parent / "basin_transition_search_field34_cc_v0"
STATE_ROWS_FILENAME = "transition_search_states.csv"
EDGE_ROWS_FILENAME = "transition_search_edges.csv"
PARETO_ROWS_FILENAME = "transition_search_pareto_rows.csv"
CASE_ROWS_FILENAME = "transition_search_case_rows.csv"
SUMMARY_FILENAME = "transition_search_summary.json"
REPORT_FILENAME = "transition_search_report.md"
CONFIG_FILENAME = "transition_search_config.json"

def _parse_csv_tuple(value: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if not value.strip():
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip())

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
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines

def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def _evaluate_state(
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
    context: dict[str, Any],
    min_support_shift_from_vanilla: float,
    min_material_q_gain: float,
    parent_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = search_state_metrics(
        state=state,
        baseline_membership=baseline_membership,
        candidate_membership=candidate_membership,
        vanilla_membership=vanilla_membership,
        sketch_nodes=sketch_nodes,
        start_quality=start_quality,
        candidate_quality=candidate_quality,
        vanilla_quality=vanilla_quality,
        vanilla_support_distance_to_candidate=vanilla_support_distance_to_candidate,
        min_support_shift_from_vanilla=min_support_shift_from_vanilla,
        min_material_q_gain=min_material_q_gain,
    )
    row = state_public_row(state=state, metrics=metrics, context=context)
    row.update(pathway_marginal_metrics(row, parent_row=parent_row))
    return row

def _polished_child(
    *,
    parent: TransitionSearchState,
    action: TransitionAction,
    graph: Any,
    donor_membership: np.ndarray,
    resolution: float,
    seed: int,
    n_iterations: int,
    randomness: float,
    child_index: int,
) -> TransitionSearchState:
    action_nodes = unique_sorted_u32([] if action.action_nodes is None else action.action_nodes)
    membership = (
        transplant_action_nodes(
            membership=parent.membership,
            donor_membership=donor_membership,
            action_nodes=action_nodes,
            reference_nodes=parent.action_nodes,
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

def _search_case(
    *,
    case_prefix_rows: pd.DataFrame,
    profile_batch_dir: Path,
    candidate_dirs: tuple[Path, ...],
    vanilla_dir: Path,
    action_types: tuple[str, ...],
    baseline_iterations: int,
    candidate_polish_iterations: int,
    local_polish_iterations: int,
    max_depth: int,
    beam_width: int,
    search_policy: str,
    context_multiplier: float,
    max_context_nodes: int,
    target_action_multiplier: float,
    max_target_action_nodes: int,
    target_unit_types: tuple[str, ...],
    max_target_unit_actions: int,
    max_target_unit_nodes: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
    polish_seed_offset: int,
    min_support_shift_from_vanilla: float,
    min_material_q_gain: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first = case_prefix_rows.iloc[0]
    case = str(first["case"])
    pair_id = str(first["pair_id"])
    candidate_index = int(first["candidate_index"])
    vanilla_seed = int(first["vanilla_seed"])
    vanilla_randomness = float(first["vanilla_randomness"])
    vanilla_n = str(first["vanilla_requested_n_iterations"])
    profile_dir = profile_batch_dir / pair_id
    units = pd.read_csv(profile_dir / UNIT_ROWS_FILENAME)
    candidate_rows = _read_candidate_rows(candidate_dirs)
    vanilla_rows = pd.read_csv(vanilla_dir / VANILLA_ROWS_FILENAME)
    candidate_row = _find_candidate_row(
        candidate_rows,
        case=case,
        candidate_index=candidate_index,
    )
    vanilla_row = _find_vanilla_row(
        vanilla_rows,
        case=case,
        seed=vanilla_seed,
        randomness=vanilla_randomness,
        n_iterations=vanilla_n,
    )
    graph_dir = Path(str(vanilla_row["graph_dir"]))
    graph, node_weights, arrays = _load_graph(graph_dir)
    baseline = _run_leiden(
        graph,
        resolution=resolution,
        seed=int(candidate_row.get("seed", 0)),
        n_iterations=baseline_iterations,
        randomness=randomness,
    )
    candidate = _recreate_candidate(
        graph=graph,
        arrays=arrays,
        node_weights=node_weights,
        baseline_membership=baseline.membership,
        baseline_quality=baseline.quality,
        row=candidate_row,
        resolution=resolution,
        randomness=randomness,
        perturb_seed_offset=perturb_seed_offset,
        polish_iterations=candidate_polish_iterations,
    )
    vanilla = _run_leiden(
        graph,
        resolution=resolution,
        seed=vanilla_seed,
        n_iterations=int(_safe_int(vanilla_n, baseline_iterations) or baseline_iterations),
        randomness=vanilla_randomness,
    )
    sketch_nodes, sketch_context = compatible_sketch_nodes(
        arrays=arrays,
        baseline_membership=baseline.membership,
        node_weights=node_weights,
        candidate_rows=candidate_rows[candidate_rows["case"].astype(str) == case],
    )
    if not bool(sketch_context.get("sketch_context_hash_matches_candidate", False)):
        raise RuntimeError(f"sketch context mismatch for {case}")
    candidate_support, vanilla_support, target_nodes = v_only_support_nodes(
        baseline.membership,
        candidate.recreated.membership,
        vanilla.membership,
    )
    vanilla_support_distance_to_candidate = support_distance(
        vanilla_support,
        candidate_support,
    )[0]
    src = np.asarray(arrays.src, dtype=np.uint32)
    dst = np.asarray(arrays.dst, dtype=np.uint32)
    weight = np.asarray(arrays.weight, dtype=np.float64)
    node_count = int(baseline.membership.size)
    if ACTION_REMAINING_TARGET_UNIT_TOPK in action_types:
        target_unit_rows = build_target_unit_rows(
            target_nodes=target_nodes,
            candidate_support_nodes=candidate_support,
            baseline_membership=baseline.membership,
            candidate_membership=candidate.recreated.membership,
            vanilla_membership=vanilla.membership,
            src=src,
            dst=dst,
            weight=weight,
            node_count=node_count,
            unit_types=target_unit_types,
        )
    else:
        target_unit_rows = pd.DataFrame()
    rows: list[dict[str, Any]] = []
    row_by_state: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    roots: list[TransitionSearchState] = []
    case_context = {
        "case": case,
        "field": first.get("field", ""),
        "method": first.get("method", ""),
        "pair_id": pair_id,
        "candidate_index": candidate_index,
        "vanilla_seed": vanilla_seed,
        "vanilla_randomness": vanilla_randomness,
        "vanilla_requested_n_iterations": vanilla_n,
        "candidate_quality": float(candidate.recreated.quality),
        "vanilla_quality": float(vanilla.quality),
        "vanilla_support_distance_to_candidate": float(
            vanilla_support_distance_to_candidate
        ),
    }
    for prefix_rank, (_, prefix_row) in enumerate(case_prefix_rows.iterrows(), start=1):
        raw_membership, mutable_nodes = apply_prefix_units(
            membership=vanilla.membership,
            donor_membership=candidate.recreated.membership,
            units=units,
            prefix_unit_ids=prefix_row["prefix_unit_ids"],
        )
        raw_quality = score_membership(
            graph,
            raw_membership,
            resolution=resolution,
        )
        direct_nodes = prefix_direct_nodes(units, prefix_row["prefix_unit_ids"])
        root = make_prefix_state(
            state_id=f"{pair_id}:p{prefix_rank}:raw",
            prefix_rank=prefix_rank,
            prefix_unit_ids=str(prefix_row["prefix_unit_ids"]),
            membership=raw_membership,
            quality=raw_quality,
            direct_nodes=direct_nodes,
            target_nodes=target_nodes,
            action_nodes=direct_nodes,
            mutable_nodes=mutable_nodes,
        )
        root_context = {
            **case_context,
            "barrier_aware_score": float(prefix_row["barrier_aware_score"]),
            "peak_raw_barrier_input": float(prefix_row["peak_raw_barrier"]),
            "support_progress_fraction_input": float(
                prefix_row["support_progress_fraction"]
            ),
            "greedy_failure_labels": prefix_row["greedy_failure_labels"],
        }
        root_row = _evaluate_state(
            state=root,
            baseline_membership=baseline.membership,
            candidate_membership=candidate.recreated.membership,
            vanilla_membership=vanilla.membership,
            sketch_nodes=sketch_nodes,
            start_quality=vanilla.quality,
            candidate_quality=candidate.recreated.quality,
            vanilla_quality=vanilla.quality,
            vanilla_support_distance_to_candidate=vanilla_support_distance_to_candidate,
            context=root_context,
            min_support_shift_from_vanilla=min_support_shift_from_vanilla,
            min_material_q_gain=min_material_q_gain,
        )
        rows.append(root_row)
        row_by_state[root.state_id] = root_row
        roots.append(root)

    current_states = roots
    context_action_types = tuple(
        action_type
        for action_type in action_types
        if action_type
        not in {
            ACTION_REMAINING_TARGET_TOPK,
            ACTION_REMAINING_TARGET_UNIT_TOPK,
        }
    )
    for depth in range(1, int(max_depth) + 1):
        child_states: list[TransitionSearchState] = []
        child_rows: list[dict[str, Any]] = []
        for parent_index, parent in enumerate(current_states, start=1):
            actions: list[TransitionAction] = []
            if int(parent.depth) == 0:
                actions.append(
                    TransitionAction(
                        action_type=ACTION_PREFIX_ONLY,
                        action_params="local_polish",
                        context_nodes=np.asarray([], dtype=np.uint32),
                    )
                )
            if ACTION_REMAINING_TARGET_TOPK in action_types:
                actions.extend(
                    build_remaining_target_actions(
                        state=parent,
                        src=src,
                        dst=dst,
                        weight=weight,
                        node_count=node_count,
                        target_action_multiplier=target_action_multiplier,
                        max_target_action_nodes=max_target_action_nodes,
                    )
                )
            if ACTION_REMAINING_TARGET_UNIT_TOPK in action_types:
                actions.extend(
                    build_remaining_target_unit_actions(
                        state=parent,
                        target_unit_rows=target_unit_rows,
                        src=src,
                        dst=dst,
                        weight=weight,
                        node_count=node_count,
                        target_unit_types=target_unit_types,
                        max_target_unit_actions=max_target_unit_actions,
                        max_target_unit_nodes=max_target_unit_nodes,
                    )
                )
            actions.extend(
                build_context_actions(
                    state=parent,
                    candidate_membership=candidate.recreated.membership,
                    vanilla_membership=vanilla.membership,
                    src=src,
                    dst=dst,
                    weight=weight,
                    node_count=node_count,
                    action_types=context_action_types,
                    context_multiplier=context_multiplier,
                    max_context_nodes=max_context_nodes,
                )
            )
            for action_index, action in enumerate(actions, start=1):
                child = _polished_child(
                    parent=parent,
                    action=action,
                    graph=graph,
                    donor_membership=candidate.recreated.membership,
                    resolution=resolution,
                    seed=(
                        int(polish_seed_offset)
                        + int(depth) * 1000
                        + int(parent_index) * 100
                        + int(action_index)
                    ),
                    n_iterations=local_polish_iterations,
                    randomness=randomness,
                    child_index=action_index,
                )
                row_context = {
                    **case_context,
                    "barrier_aware_score": math.nan,
                    "peak_raw_barrier_input": math.nan,
                    "support_progress_fraction_input": math.nan,
                    "greedy_failure_labels": "",
                }
                row = _evaluate_state(
                    state=child,
                    baseline_membership=baseline.membership,
                    candidate_membership=candidate.recreated.membership,
                    vanilla_membership=vanilla.membership,
                    sketch_nodes=sketch_nodes,
                    start_quality=vanilla.quality,
                    candidate_quality=candidate.recreated.quality,
                    vanilla_quality=vanilla.quality,
                    vanilla_support_distance_to_candidate=vanilla_support_distance_to_candidate,
                    context=row_context,
                    parent_row=row_by_state.get(parent.state_id),
                    min_support_shift_from_vanilla=min_support_shift_from_vanilla,
                    min_material_q_gain=min_material_q_gain,
                )
                child_states.append(child)
                child_rows.append(row)
                row_by_state[child.state_id] = row
                edges.append(
                    edge_public_row(
                        parent_state_id=parent.state_id,
                        child_state_id=child.state_id,
                        action=action,
                        context=case_context,
                    )
                )
        if not child_states:
            break
        rows.extend(child_rows)
        current_states = select_search_beam(
            child_states,
            pd.DataFrame(child_rows),
            beam_width=beam_width,
            search_policy=search_policy,
        )
        if not current_states:
            break
    return pd.DataFrame(rows), pd.DataFrame(edges)

def _case_rows(rows: pd.DataFrame, *, search_policy: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    score_column = search_policy_score_column(search_policy)
    out: list[dict[str, Any]] = []
    for pair_id, group in rows.groupby("pair_id", sort=True):
        labels = group["search_recovery_label"].value_counts().to_dict()
        reachability_labels = group["reachability_label"].value_counts().to_dict()
        best = group.sort_values(
            [
                score_column,
                "state_target_progress_from_vanilla",
                "state_support_distance_to_vanilla",
                "state_delta_q_vs_start",
                "mutable_node_count",
            ],
            ascending=[False, False, False, False, True],
        ).iloc[0]
        shifted = group[
            group["search_recovery_label"].astype(str)
            == "support_shift_q_recovered"
        ].copy()
        if shifted.empty:
            best_shift = best
        else:
            best_shift = shifted.sort_values(
                [
                    score_column,
                    "state_target_progress_from_vanilla",
                    "state_delta_q_vs_start",
                    "mutable_node_count",
                ],
                ascending=[False, False, False, True],
            ).iloc[0]
        out.append(
            {
                "pair_id": pair_id,
                "state_rows": int(len(group)),
                "support_shift_q_recovered_rows": int(
                    labels.get("support_shift_q_recovered", 0)
                ),
                "vanilla_collapse_rows": int(labels.get("vanilla_collapse", 0)),
                "quality_loss_rows": int(labels.get("quality_loss", 0)),
                "low_roi_support_shift_rows": int(
                    labels.get("low_roi_support_shift", 0)
                ),
                "raw_only_rows": int(labels.get("raw_only", 0)),
                "support_gate_reached_rows": int(
                    reachability_labels.get("support_gate_reached", 0)
                ),
                "target_progress_rows": int(
                    reachability_labels.get("target_progress", 0)
                ),
                "source_escape_rows": int(
                    reachability_labels.get("source_escape", 0)
                ),
                "coverage_only_rows": int(
                    reachability_labels.get("coverage_only", 0)
                ),
                "stalled_rows": int(reachability_labels.get("stalled", 0)),
                "best_state_id": best["state_id"],
                "best_search_score": float(best[score_column]),
                "best_reachability_score": float(best["reachability_search_score"]),
                "best_delta_q_vs_start": float(best["state_delta_q_vs_start"]),
                "best_target_progress_from_vanilla": float(
                    best["state_target_progress_from_vanilla"]
                ),
                "best_candidate_progress_from_vanilla": float(
                    best["state_candidate_progress_from_vanilla"]
                ),
                "best_support_distance_to_vanilla": float(
                    best["state_support_distance_to_vanilla"]
                ),
                "best_target_coverage_fraction": float(
                    best["target_coverage_fraction"]
                ),
                "best_covered_target_count": int(best["covered_target_count"]),
                "best_remaining_target_count": int(best["remaining_target_count"]),
                "best_marginal_target_distance_reduction": float(
                    best["marginal_target_distance_reduction"]
                ),
                "best_marginal_cost_per_target_node": float(
                    best["marginal_cost_per_target_node"]
                ),
                "best_label": best["search_recovery_label"],
                "best_reachability_label": best["reachability_label"],
                "best_shift_state_id": best_shift["state_id"],
                "best_shift_search_score": float(best_shift[score_column]),
                "best_shift_reachability_score": float(
                    best_shift["reachability_search_score"]
                ),
                "best_shift_delta_q_vs_start": float(
                    best_shift["state_delta_q_vs_start"]
                ),
                "best_shift_target_progress_from_vanilla": float(
                    best_shift["state_target_progress_from_vanilla"]
                ),
                "best_shift_candidate_progress_from_vanilla": float(
                    best_shift["state_candidate_progress_from_vanilla"]
                ),
                "best_shift_support_distance_to_vanilla": float(
                    best_shift["state_support_distance_to_vanilla"]
                ),
                "best_shift_target_coverage_fraction": float(
                    best_shift["target_coverage_fraction"]
                ),
                "best_shift_covered_target_count": int(
                    best_shift["covered_target_count"]
                ),
                "best_shift_remaining_target_count": int(
                    best_shift["remaining_target_count"]
                ),
                "best_shift_label": best_shift["search_recovery_label"],
                "best_shift_reachability_label": best_shift["reachability_label"],
            }
        )
    return pd.DataFrame(out)

def write_report(
    path: Path,
    *,
    rows: pd.DataFrame,
    edges: pd.DataFrame,
    pareto_rows: pd.DataFrame,
    case_rows: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Basin Transition Search v0",
        "",
        "This artifact searches small target/action/context combinations around barrier-aware prefixes.",
        "",
        "It is diagnostic-only. It records QF recovery and support movement separately so discovery policies can cross low-quality intermediate states without hiding the debt.",
        "",
        "For the current `V -> C` fixture, `target_nodes` is the vanilla-extra support set `S_V - S_C`; `action_nodes` is the prefix-selected subset, and `context_nodes` are bounded closure or boundary nodes released for local polish.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "prefix_dir",
        "profile_batch_dir",
        "state_rows",
        "edge_rows",
        "pareto_rows",
        "case_rows",
        "top_prefixes_per_case",
        "search_policy",
        "max_depth",
        "beam_width",
        "context_multiplier",
        "max_context_nodes",
        "target_action_multiplier",
        "max_target_action_nodes",
        "target_unit_types",
        "max_target_unit_actions",
        "max_target_unit_nodes",
        "local_polish_iterations",
        "action_types",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")
    lines.extend(["", "## Case Rows", ""])
    case_cols = [
        "pair_id",
        "state_rows",
        "support_shift_q_recovered_rows",
        "vanilla_collapse_rows",
        "quality_loss_rows",
        "support_gate_reached_rows",
        "target_progress_rows",
        "source_escape_rows",
        "best_state_id",
        "best_search_score",
        "best_reachability_score",
        "best_delta_q_vs_start",
        "best_target_progress_from_vanilla",
        "best_candidate_progress_from_vanilla",
        "best_support_distance_to_vanilla",
        "best_target_coverage_fraction",
        "best_covered_target_count",
        "best_remaining_target_count",
        "best_marginal_target_distance_reduction",
        "best_marginal_cost_per_target_node",
        "best_label",
        "best_reachability_label",
        "best_shift_state_id",
        "best_shift_search_score",
        "best_shift_reachability_score",
        "best_shift_delta_q_vs_start",
        "best_shift_target_progress_from_vanilla",
        "best_shift_candidate_progress_from_vanilla",
        "best_shift_support_distance_to_vanilla",
        "best_shift_target_coverage_fraction",
        "best_shift_covered_target_count",
        "best_shift_remaining_target_count",
        "best_shift_label",
        "best_shift_reachability_label",
    ]
    lines.extend(
        _markdown_table(case_rows[[c for c in case_cols if c in case_rows.columns]])
    )
    lines.extend(["", "## Pareto Rows", ""])
    pareto_cols = [
        "pair_id",
        "state_id",
        "parent_state_id",
        "depth",
        "action_type",
        "target_node_count",
        "action_node_count",
        "action_target_node_count",
        "action_off_target_node_count",
        "covered_target_count",
        "remaining_target_count",
        "target_coverage_fraction",
        "mutable_node_count",
        "context_node_count",
        "context_to_action_ratio",
        "marginal_target_distance_reduction",
        "marginal_q_debt",
        "marginal_mutable_node_count",
        "marginal_covered_target_count",
        "marginal_cost_per_target_node",
        "state_target_distance",
        "state_source_distance",
        "state_target_progress_from_vanilla",
        "state_delta_q_vs_start",
        "state_candidate_progress_from_vanilla",
        "state_support_distance_to_vanilla",
        "state_support_distance_to_candidate",
        "state_greedy_score",
        "reachability_search_score",
        "quality_search_score",
        "progress_search_score",
        "balanced_search_score",
        "search_recovery_label",
        "reachability_label",
        "applied_actions",
    ]
    lines.extend(
        _markdown_table(
            pareto_rows[[c for c in pareto_cols if c in pareto_rows.columns]],
            max_rows=80,
        )
    )
    lines.extend(["", "## Action Counts", ""])
    if not rows.empty:
        counts = (
            rows.groupby(["action_type", "search_recovery_label"], sort=True)
            .size()
            .reset_index(name="rows")
        )
        lines.extend(_markdown_table(counts, max_rows=80))
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- `support_shift_q_recovered` is a search diagnostic, not an operator acceptance.",
            "- Under `reachability_first`, QF is reported as debt or recovery, not used as a pruning gate.",
            "- Rows with positive QF but low support-distance-to-vanilla are treated as vanilla collapse.",
            "- The useful next step is determined by which primitive changes the label mix on the Pareto frontier, not by a single best-QF row.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_search(
    *,
    prefix_dir: Path,
    profile_batch_dir: Path,
    output_dir: Path,
    candidate_dirs: tuple[Path, ...],
    vanilla_dir: Path,
    pair_ids: tuple[str, ...],
    top_prefixes_per_case: int,
    action_types: tuple[str, ...],
    baseline_iterations: int,
    candidate_polish_iterations: int,
    local_polish_iterations: int,
    max_depth: int,
    beam_width: int,
    search_policy: str,
    context_multiplier: float,
    max_context_nodes: int,
    target_action_multiplier: float,
    max_target_action_nodes: int,
    target_unit_types: tuple[str, ...],
    max_target_unit_actions: int,
    max_target_unit_nodes: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
    polish_seed_offset: int,
    min_support_shift_from_vanilla: float,
    min_material_q_gain: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefixes = select_prefix_rows(
        pd.read_csv(prefix_dir / BARRIER_PREFIX_ROWS_FILENAME),
        pair_ids=pair_ids,
        top_prefixes_per_case=top_prefixes_per_case,
    )
    if prefixes.empty:
        raise ValueError("No prefix rows selected for transition search")
    frames: list[pd.DataFrame] = []
    edge_frames: list[pd.DataFrame] = []
    for _, case_prefixes in prefixes.groupby("pair_id", sort=True):
        states, edges = _search_case(
            case_prefix_rows=case_prefixes,
            profile_batch_dir=profile_batch_dir,
            candidate_dirs=candidate_dirs,
            vanilla_dir=vanilla_dir,
            action_types=action_types,
            baseline_iterations=baseline_iterations,
            candidate_polish_iterations=candidate_polish_iterations,
            local_polish_iterations=local_polish_iterations,
            max_depth=max_depth,
            beam_width=beam_width,
            search_policy=search_policy,
            context_multiplier=context_multiplier,
            max_context_nodes=max_context_nodes,
            target_action_multiplier=target_action_multiplier,
            max_target_action_nodes=max_target_action_nodes,
            target_unit_types=target_unit_types,
            max_target_unit_actions=max_target_unit_actions,
            max_target_unit_nodes=max_target_unit_nodes,
            resolution=resolution,
            randomness=randomness,
            perturb_seed_offset=perturb_seed_offset,
            polish_seed_offset=polish_seed_offset,
            min_support_shift_from_vanilla=min_support_shift_from_vanilla,
            min_material_q_gain=min_material_q_gain,
        )
        frames.append(states)
        edge_frames.append(edges)
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    edges = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    pareto_rows = select_pareto_rows(rows, max_rows=100, search_policy=search_policy)
    case_rows = _case_rows(rows, search_policy=search_policy)
    rows.to_csv(output_dir / STATE_ROWS_FILENAME, index=False)
    edges.to_csv(output_dir / EDGE_ROWS_FILENAME, index=False)
    pareto_rows.to_csv(output_dir / PARETO_ROWS_FILENAME, index=False)
    case_rows.to_csv(output_dir / CASE_ROWS_FILENAME, index=False)
    config = {
        "prefix_dir": str(prefix_dir),
        "profile_batch_dir": str(profile_batch_dir),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "vanilla_dir": str(vanilla_dir),
        "pair_ids": list(pair_ids),
        "top_prefixes_per_case": int(top_prefixes_per_case),
        "action_types": list(action_types),
        "baseline_iterations": int(baseline_iterations),
        "candidate_polish_iterations": int(candidate_polish_iterations),
        "local_polish_iterations": int(local_polish_iterations),
        "max_depth": int(max_depth),
        "beam_width": int(beam_width),
        "search_policy": str(search_policy),
        "context_multiplier": float(context_multiplier),
        "max_context_nodes": int(max_context_nodes),
        "target_action_multiplier": float(target_action_multiplier),
        "max_target_action_nodes": int(max_target_action_nodes),
        "target_unit_types": list(target_unit_types),
        "max_target_unit_actions": int(max_target_unit_actions),
        "max_target_unit_nodes": int(max_target_unit_nodes),
        "resolution": float(resolution),
        "randomness": float(randomness),
        "perturb_seed_offset": int(perturb_seed_offset),
        "polish_seed_offset": int(polish_seed_offset),
        "min_support_shift_from_vanilla": float(min_support_shift_from_vanilla),
        "min_material_q_gain": float(min_material_q_gain),
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "schema": "leiden_basin_transition_search.v0",
        "prefix_dir": str(prefix_dir),
        "profile_batch_dir": str(profile_batch_dir),
        "output_dir": str(output_dir),
        "state_rows": int(len(rows)),
        "edge_rows": int(len(edges)),
        "pareto_rows": int(len(pareto_rows)),
        "case_rows": int(len(case_rows)),
        **{
            key: value
            for key, value in config.items()
            if key
            in {
                "pair_ids",
                "top_prefixes_per_case",
                "action_types",
                "baseline_iterations",
                "candidate_polish_iterations",
                "local_polish_iterations",
                "max_depth",
                "beam_width",
                "search_policy",
                "context_multiplier",
                "max_context_nodes",
                "target_action_multiplier",
                "max_target_action_nodes",
                "target_unit_types",
                "max_target_unit_actions",
                "max_target_unit_nodes",
                "resolution",
                "randomness",
                "min_support_shift_from_vanilla",
                "min_material_q_gain",
            }
        },
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(
        output_dir / REPORT_FILENAME,
        rows=rows,
        edges=edges,
        pareto_rows=pareto_rows,
        case_rows=case_rows,
        summary=summary,
    )
    return summary

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--profile-batch-dir", type=Path, default=DEFAULT_PROFILE_BATCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-dir", type=Path, action="append", default=None)
    parser.add_argument("--vanilla-dir", type=Path, default=DEFAULT_VANILLA_DIR)
    parser.add_argument("--pair-ids", default="c0-s11-r0.001,c2-s11-r0")
    parser.add_argument("--top-prefixes-per-case", type=int, default=10)
    parser.add_argument(
        "--action-types",
        default=",".join(
            (
                ACTION_REMAINING_TARGET_TOPK,
                ACTION_CANDIDATE_CLOSURE_TOPK,
                ACTION_BOUNDARY_SHELL_TOPK,
            )
        ),
    )
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--candidate-polish-iterations", type=int, default=5)
    parser.add_argument("--local-polish-iterations", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument(
        "--search-policy",
        choices=SEARCH_POLICIES,
        default=SEARCH_POLICY_STATE_GREEDY,
    )
    parser.add_argument("--context-multiplier", type=float, default=2.0)
    parser.add_argument("--max-context-nodes", type=int, default=128)
    parser.add_argument("--target-action-multiplier", type=float, default=0.5)
    parser.add_argument("--max-target-action-nodes", type=int, default=64)
    parser.add_argument(
        "--target-unit-types",
        default="label_intersection_block",
        help=(
            "Comma-separated target unit types for remaining_target_unit_topk. "
            f"Available: {','.join(TARGET_UNIT_TYPES)}"
        ),
    )
    parser.add_argument("--max-target-unit-actions", type=int, default=1)
    parser.add_argument("--max-target-unit-nodes", type=int, default=64)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    parser.add_argument("--polish-seed-offset", type=int, default=11000)
    parser.add_argument("--min-support-shift-from-vanilla", type=float, default=0.05)
    parser.add_argument("--min-material-q-gain", type=float, default=0.0)
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_search(
        prefix_dir=args.prefix_dir,
        profile_batch_dir=args.profile_batch_dir,
        output_dir=args.output_dir,
        candidate_dirs=tuple(args.candidate_dir or DEFAULT_CANDIDATE_DIRS),
        vanilla_dir=args.vanilla_dir,
        pair_ids=_parse_csv_tuple(args.pair_ids),
        top_prefixes_per_case=args.top_prefixes_per_case,
        action_types=_parse_csv_tuple(args.action_types),
        baseline_iterations=args.baseline_iterations,
        candidate_polish_iterations=args.candidate_polish_iterations,
        local_polish_iterations=args.local_polish_iterations,
        max_depth=args.max_depth,
        beam_width=args.beam_width,
        search_policy=args.search_policy,
        context_multiplier=args.context_multiplier,
        max_context_nodes=args.max_context_nodes,
        target_action_multiplier=args.target_action_multiplier,
        max_target_action_nodes=args.max_target_action_nodes,
        target_unit_types=_parse_csv_tuple(args.target_unit_types, TARGET_UNIT_TYPES),
        max_target_unit_actions=args.max_target_unit_actions,
        max_target_unit_nodes=args.max_target_unit_nodes,
        resolution=args.resolution,
        randomness=args.randomness,
        perturb_seed_offset=args.perturb_seed_offset,
        polish_seed_offset=args.polish_seed_offset,
        min_support_shift_from_vanilla=args.min_support_shift_from_vanilla,
        min_material_q_gain=args.min_material_q_gain,
    )
    print(summary)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
