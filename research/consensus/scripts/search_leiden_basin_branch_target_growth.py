#!/usr/bin/env python3
"""Branch target-growth choices before committing to one basin pathway."""

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
    DEFAULT_OUTPUT_DIR as DEFAULT_PREFIX_DIR,
    PREFIX_ROWS_FILENAME as BARRIER_PREFIX_ROWS_FILENAME,
)
from collect_leiden_vanilla_reachability_sweep import (  # noqa: E402
    _load_graph,
    _read_candidate_rows,
    compatible_sketch_nodes,
)
from evaluate_leiden_basin_polish_prefixes import select_prefix_rows  # noqa: E402
from profile_leiden_basin_ordered_flips import UNIT_ROWS_FILENAME  # noqa: E402
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
    ACTION_PREFIX_ONLY,
    TARGET_SELECTION_FIXED_CAP,
    TARGET_SELECTION_FIXED_TAIL_BACKFILL,
    TARGET_SELECTION_GUARDED_ELBOW,
    TARGET_SELECTION_POLICIES,
    TARGET_SELECTION_PREFIX_POLISH,
    TARGET_SELECTION_RAW_PREFIX,
    TransitionAction,
    branch_target_action_context,
    build_branching_target_growth_actions,
    compute_pathway_wall_rows,
    edge_public_row,
    make_prefix_state,
    prefix_direct_nodes,
    score_branch_path_rows,
    select_branch_path_rows,
    select_qf_wall_frontier,
    summarize_pathway_wall_rows,
    unique_sorted_u32,
)
from search_leiden_basin_transitions import (  # noqa: E402
    _evaluate_state,
    _polished_child,
)


DEFAULT_OUTPUT_DIR = DEFAULT_PROFILE_BATCH_DIR.parent / (
    "basin_transition_branch_target_growth_field34_cc_c0_v0"
)
STATE_ROWS_FILENAME = "branch_target_growth_states.csv"
EDGE_ROWS_FILENAME = "branch_target_growth_edges.csv"
PATH_ROWS_FILENAME = "branch_target_growth_path_rows.csv"
FRONTIER_ROWS_FILENAME = "branch_target_growth_frontier_rows.csv"
CASE_ROWS_FILENAME = "branch_target_growth_case_rows.csv"
SUMMARY_FILENAME = "branch_target_growth_summary.json"
CONFIG_FILENAME = "branch_target_growth_config.json"
REPORT_FILENAME = "branch_target_growth_report.md"


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
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _prefix_context_from_row(row: dict[str, Any] | pd.Series) -> dict[str, Any]:
    return {
        "barrier_aware_score": row.get("barrier_aware_score", math.nan),
        "peak_raw_barrier_input": row.get("peak_raw_barrier_input", math.nan),
        "support_progress_fraction_input": row.get(
            "support_progress_fraction_input",
            math.nan,
        ),
        "greedy_failure_labels": row.get("greedy_failure_labels", ""),
    }


def _selection_context(selection_policy: str) -> dict[str, Any]:
    return {
        "path_policy": "branch_target_growth",
        "selection_policy": selection_policy,
        "escalation_reason": "not_applicable",
        "escalated_to_fixed": False,
        "target_stage_index": 0,
        "selected_k": 0,
        "selected_node_ids": "",
        "remaining_count_before_selection": 0,
        "positive_pull_count": 0,
        "fixed_effective_k": 0,
        "guarded_elbow_k": 0,
        "guarded_elbow_reason": "",
        "fixed_pull_fraction": 0.0,
        "guarded_elbow_pull_fraction": 0.0,
        "gap_elbow_k": 0,
        "gap_elbow_drop_fraction_of_top": 0.0,
        "cumulative_elbow_k": 0,
        "score_floor_k": 0,
    }


def _case_rows(path_rows: pd.DataFrame, *, support_gate: float) -> pd.DataFrame:
    if path_rows.empty:
        return pd.DataFrame()
    scored = score_branch_path_rows(path_rows)
    out: list[dict[str, Any]] = []
    group_columns = ["pair_id", "path_selection_policy"]
    for keys, group in scored.groupby(group_columns, sort=True, dropna=False):
        pair_id, selection_policy = keys
        support_gate_rows = group[
            group["path_final_support_distance_to_vanilla"] >= float(support_gate)
        ].copy()
        recovered_gate = support_gate_rows[
            support_gate_rows["path_final_delta_q_vs_start"] >= 0.0
        ].copy()
        best = group.sort_values(
            [
                "path_branch_discovery_score",
                "path_final_support_distance_to_vanilla",
                "path_final_target_progress_from_vanilla",
                "path_final_delta_q_vs_start",
                "path_q_wall",
                "path_final_mutable_node_count",
            ],
            ascending=[False, False, False, False, True, True],
        ).iloc[0]
        if recovered_gate.empty:
            best_recovered = best
        else:
            best_recovered = recovered_gate.sort_values(
                [
                    "path_branch_discovery_score",
                    "path_final_support_distance_to_vanilla",
                    "path_final_target_progress_from_vanilla",
                    "path_q_wall",
                    "path_final_mutable_node_count",
                ],
                ascending=[False, False, False, True, True],
            ).iloc[0]
        if support_gate_rows.empty:
            min_wall_gate = best
        else:
            min_wall_gate = support_gate_rows.sort_values(
                [
                    "path_q_wall",
                    "path_final_support_distance_to_vanilla",
                    "path_final_target_progress_from_vanilla",
                    "path_final_delta_q_vs_start",
                ],
                ascending=[True, False, False, False],
            ).iloc[0]
        out.append(
            {
                "pair_id": pair_id,
                "path_selection_policy": selection_policy,
                "path_rows": int(len(group)),
                "support_gate_rows": int(len(support_gate_rows)),
                "support_gate_q_recovered_rows": int(len(recovered_gate)),
                "best_state_id": best["path_final_state_id"],
                "best_score": float(best["path_branch_discovery_score"]),
                "best_q_wall": float(best["path_q_wall"]),
                "best_delta_q": float(best["path_final_delta_q_vs_start"]),
                "best_support": float(best["path_final_support_distance_to_vanilla"]),
                "best_target_progress": float(
                    best["path_final_target_progress_from_vanilla"]
                ),
                "best_coverage": float(best["path_final_target_coverage_fraction"]),
                "best_mutable": int(best["path_final_mutable_node_count"]),
                "best_recovered_state_id": best_recovered["path_final_state_id"],
                "best_recovered_q_wall": float(best_recovered["path_q_wall"]),
                "best_recovered_delta_q": float(
                    best_recovered["path_final_delta_q_vs_start"]
                ),
                "best_recovered_support": float(
                    best_recovered["path_final_support_distance_to_vanilla"]
                ),
                "best_recovered_target_progress": float(
                    best_recovered["path_final_target_progress_from_vanilla"]
                ),
                "best_recovered_mutable": int(
                    best_recovered["path_final_mutable_node_count"]
                ),
                "min_wall_gate_state_id": min_wall_gate["path_final_state_id"],
                "min_wall_gate_q_wall": float(min_wall_gate["path_q_wall"]),
                "min_wall_gate_delta_q": float(
                    min_wall_gate["path_final_delta_q_vs_start"]
                ),
                "min_wall_gate_support": float(
                    min_wall_gate["path_final_support_distance_to_vanilla"]
                ),
                "min_wall_gate_target_progress": float(
                    min_wall_gate["path_final_target_progress_from_vanilla"]
                ),
                "min_wall_gate_mutable": int(
                    min_wall_gate["path_final_mutable_node_count"]
                ),
            }
        )
    return pd.DataFrame(out)


def _evaluate_case(
    *,
    case_prefix_rows: pd.DataFrame,
    profile_batch_dir: Path,
    candidate_dirs: tuple[Path, ...],
    vanilla_dir: Path,
    baseline_iterations: int,
    candidate_polish_iterations: int,
    local_polish_iterations: int,
    max_target_stages: int,
    beam_width: int,
    target_action_multiplier: float,
    max_target_action_nodes: int,
    selection_policies: tuple[str, ...],
    cumulative_fraction: float,
    min_score_fraction: float,
    min_gap_fraction: float,
    min_guarded_pull_fraction: float,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
    polish_seed_offset: int,
    min_support_shift_from_vanilla: float,
    min_material_q_gain: float,
    support_gate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    rows: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    state_by_id: dict[str, Any] = {}
    row_by_state: dict[str, dict[str, Any]] = {}
    active_states: list[Any] = []
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
        "target_node_count_input": int(unique_sorted_u32(target_nodes).size),
        "candidate_support_node_count": int(candidate_support.size),
        "vanilla_support_node_count": int(vanilla_support.size),
    }
    for prefix_rank, (_, prefix_row) in enumerate(case_prefix_rows.iterrows(), start=1):
        raw_membership, mutable_nodes = apply_prefix_units(
            membership=vanilla.membership,
            donor_membership=candidate.recreated.membership,
            units=units,
            prefix_unit_ids=prefix_row["prefix_unit_ids"],
        )
        raw_quality = score_membership(graph, raw_membership, resolution=resolution)
        direct_nodes = prefix_direct_nodes(units, prefix_row["prefix_unit_ids"])
        prefix_context = {
            **case_context,
            "barrier_aware_score": float(prefix_row["barrier_aware_score"]),
            "peak_raw_barrier_input": float(prefix_row["peak_raw_barrier"]),
            "support_progress_fraction_input": float(
                prefix_row["support_progress_fraction"]
            ),
            "greedy_failure_labels": prefix_row["greedy_failure_labels"],
        }
        root = make_prefix_state(
            state_id=f"{pair_id}:p{prefix_rank}:branch:raw",
            prefix_rank=prefix_rank,
            prefix_unit_ids=str(prefix_row["prefix_unit_ids"]),
            membership=raw_membership,
            quality=raw_quality,
            direct_nodes=direct_nodes,
            target_nodes=target_nodes,
            action_nodes=direct_nodes,
            mutable_nodes=mutable_nodes,
        )
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
            context={
                **prefix_context,
                **_selection_context(TARGET_SELECTION_RAW_PREFIX),
            },
            min_support_shift_from_vanilla=min_support_shift_from_vanilla,
            min_material_q_gain=min_material_q_gain,
        )
        root_row["path_elapsed_sec"] = 0.0
        rows.append(root_row)
        row_by_state[root.state_id] = root_row
        state_by_id[root.state_id] = root
        prefix_action = TransitionAction(
            action_type=ACTION_PREFIX_ONLY,
            action_params="branch_target_growth;local_polish",
            context_nodes=np.asarray([], dtype=np.uint32),
        )
        prefix_polished = _polished_child(
            parent=root,
            action=prefix_action,
            graph=graph,
            donor_membership=candidate.recreated.membership,
            resolution=resolution,
            seed=int(polish_seed_offset) + int(prefix_rank) * 1000,
            n_iterations=local_polish_iterations,
            randomness=randomness,
            child_index=1,
        )
        prefix_row_public = _evaluate_state(
            state=prefix_polished,
            baseline_membership=baseline.membership,
            candidate_membership=candidate.recreated.membership,
            vanilla_membership=vanilla.membership,
            sketch_nodes=sketch_nodes,
            start_quality=vanilla.quality,
            candidate_quality=candidate.recreated.quality,
            vanilla_quality=vanilla.quality,
            vanilla_support_distance_to_candidate=vanilla_support_distance_to_candidate,
            context={
                **prefix_context,
                **_selection_context(TARGET_SELECTION_PREFIX_POLISH),
            },
            parent_row=root_row,
            min_support_shift_from_vanilla=min_support_shift_from_vanilla,
            min_material_q_gain=min_material_q_gain,
        )
        prefix_row_public["path_elapsed_sec"] = float(prefix_polished.elapsed_sec)
        rows.append(prefix_row_public)
        row_by_state[prefix_polished.state_id] = prefix_row_public
        state_by_id[prefix_polished.state_id] = prefix_polished
        active_states.append(prefix_polished)
        edges.append(
            edge_public_row(
                parent_state_id=root.state_id,
                child_state_id=prefix_polished.state_id,
                action=prefix_action,
                context={**case_context, "path_policy": "branch_target_growth"},
            )
        )

    for target_stage_index in range(1, int(max_target_stages) + 1):
        child_states: list[Any] = []
        child_rows: list[dict[str, Any]] = []
        for parent_index, parent in enumerate(active_states, start=1):
            parent_row = row_by_state[parent.state_id]
            branch_actions = build_branching_target_growth_actions(
                state=parent,
                src=src,
                dst=dst,
                weight=weight,
                node_count=node_count,
                target_stage_index=target_stage_index,
                target_action_multiplier=target_action_multiplier,
                max_target_action_nodes=max_target_action_nodes,
                selection_policies=selection_policies,
                cumulative_fraction=cumulative_fraction,
                min_score_fraction=min_score_fraction,
                min_gap_fraction=min_gap_fraction,
                min_guarded_pull_fraction=min_guarded_pull_fraction,
            )
            for action_index, branch in enumerate(branch_actions, start=1):
                child = _polished_child(
                    parent=parent,
                    action=branch.action,
                    graph=graph,
                    donor_membership=candidate.recreated.membership,
                    resolution=resolution,
                    seed=(
                        int(polish_seed_offset)
                        + int(target_stage_index) * 100000
                        + int(parent_index) * 1000
                        + int(action_index)
                    ),
                    n_iterations=local_polish_iterations,
                    randomness=randomness,
                    child_index=action_index,
                )
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
                    context={
                        **case_context,
                        **_prefix_context_from_row(parent_row),
                        "path_policy": "branch_target_growth",
                        **branch_target_action_context(branch),
                    },
                    parent_row=parent_row,
                    min_support_shift_from_vanilla=min_support_shift_from_vanilla,
                    min_material_q_gain=min_material_q_gain,
                )
                row["path_elapsed_sec"] = float(
                    parent_row.get("path_elapsed_sec", 0.0)
                ) + float(child.elapsed_sec)
                child_states.append(child)
                child_rows.append(row)
                row_by_state[child.state_id] = row
                state_by_id[child.state_id] = child
                edges.append(
                    edge_public_row(
                        parent_state_id=parent.state_id,
                        child_state_id=child.state_id,
                        action=branch.action,
                        context={
                            **case_context,
                            "path_policy": "branch_target_growth",
                            "selection_policy": branch.selection_policy,
                        },
                    )
                )
        if not child_states:
            break
        rows.extend(child_rows)
        path_rows = compute_pathway_wall_rows(
            pd.DataFrame(rows),
            source_label="branch_target_growth_v0",
            support_gate=support_gate,
        )
        selected_paths = select_branch_path_rows(
            path_rows,
            candidate_state_ids=[state.state_id for state in child_states],
            beam_width=beam_width,
            diversity_columns=("path_selection_policy",),
        )
        selected_ids = set(selected_paths["path_final_state_id"].astype(str))
        active_states = [state for state in child_states if state.state_id in selected_ids]
        if not active_states:
            break

    path_rows = score_branch_path_rows(
        compute_pathway_wall_rows(
            pd.DataFrame(rows),
            source_label="branch_target_growth_v0",
            support_gate=support_gate,
        )
    )
    return pd.DataFrame(rows), pd.DataFrame(edges), path_rows


def write_report(
    path: Path,
    *,
    rows: pd.DataFrame,
    path_rows: pd.DataFrame,
    frontier_rows: pd.DataFrame,
    case_rows: pd.DataFrame,
    wall_summary_rows: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Branch Target-Growth Basin Search v0",
        "",
        "This artifact keeps guarded, fixed-cap, and fixed-tail target-growth branches alive before choosing a path-level frontier.",
        "",
        "It is diagnostic-only. QF wall is measured as debt; QF recovery, support movement, target progress, and mutable-node cost are reported separately.",
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
        "path_rows",
        "frontier_rows",
        "case_rows",
        "pair_ids",
        "top_prefixes_per_case",
        "max_target_stages",
        "beam_width",
        "selection_policies",
        "local_polish_iterations",
        "target_action_multiplier",
        "max_target_action_nodes",
        "support_gate",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")
    lines.extend(["", "## Case Rows", ""])
    case_cols = [
        "pair_id",
        "path_selection_policy",
        "path_rows",
        "support_gate_rows",
        "support_gate_q_recovered_rows",
        "best_score",
        "best_q_wall",
        "best_delta_q",
        "best_support",
        "best_target_progress",
        "best_coverage",
        "best_mutable",
        "best_recovered_q_wall",
        "best_recovered_delta_q",
        "best_recovered_support",
        "best_recovered_target_progress",
        "best_recovered_mutable",
        "min_wall_gate_q_wall",
        "min_wall_gate_delta_q",
        "min_wall_gate_support",
        "min_wall_gate_target_progress",
        "min_wall_gate_mutable",
    ]
    lines.extend(
        _markdown_table(case_rows[[c for c in case_cols if c in case_rows.columns]])
    )
    lines.extend(["", "## Wall Summary", ""])
    wall_cols = [
        "source_label",
        "pair_id",
        "path_rows",
        "support_gate_rows",
        "support_gate_q_recovered_rows",
        "support_gate_q_wall_min",
        "support_gate_q_wall_median",
        "best_support_q_wall",
        "best_support_delta_q",
        "best_support_support_distance_to_vanilla",
        "best_support_target_progress",
        "best_support_mutable_nodes",
        "best_efficiency_q_wall",
        "best_efficiency_delta_q",
        "best_efficiency_support_distance_to_vanilla",
        "best_efficiency_target_progress",
        "best_efficiency_mutable_nodes",
    ]
    lines.extend(
        _markdown_table(
            wall_summary_rows[[c for c in wall_cols if c in wall_summary_rows.columns]]
        )
    )
    lines.extend(["", "## Branch Frontier Rows", ""])
    frontier_cols = [
        "pair_id",
        "path_final_state_id",
        "path_selection_policy",
        "path_prefix_rank",
        "path_branch_discovery_score",
        "path_q_wall",
        "path_final_delta_q_vs_start",
        "path_final_support_distance_to_vanilla",
        "path_final_target_progress_from_vanilla",
        "path_final_target_coverage_fraction",
        "path_final_mutable_node_count",
        "path_applied_actions",
    ]
    lines.extend(
        _markdown_table(
            frontier_rows[[c for c in frontier_cols if c in frontier_rows.columns]],
            max_rows=80,
        )
    )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- A branch is promising only if support/target progress survives bounded polish with recoverable QF debt.",
            "- `fixed_tail_backfill` is diagnostic context for what guarded-elbow skipped; it is not yet an operator policy.",
            "- Compare this against seed controls before using any row as a Dongdaemun-refinement claim.",
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
    baseline_iterations: int,
    candidate_polish_iterations: int,
    local_polish_iterations: int,
    max_target_stages: int,
    beam_width: int,
    target_action_multiplier: float,
    max_target_action_nodes: int,
    selection_policies: tuple[str, ...],
    cumulative_fraction: float,
    min_score_fraction: float,
    min_gap_fraction: float,
    min_guarded_pull_fraction: float,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
    polish_seed_offset: int,
    min_support_shift_from_vanilla: float,
    min_material_q_gain: float,
    support_gate: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefixes = select_prefix_rows(
        pd.read_csv(prefix_dir / BARRIER_PREFIX_ROWS_FILENAME),
        pair_ids=pair_ids,
        top_prefixes_per_case=top_prefixes_per_case,
    )
    if prefixes.empty:
        raise ValueError("No prefix rows selected for branch target-growth search")
    state_frames: list[pd.DataFrame] = []
    edge_frames: list[pd.DataFrame] = []
    path_frames: list[pd.DataFrame] = []
    for _, case_prefixes in prefixes.groupby("pair_id", sort=True):
        states, edges, paths = _evaluate_case(
            case_prefix_rows=case_prefixes,
            profile_batch_dir=profile_batch_dir,
            candidate_dirs=candidate_dirs,
            vanilla_dir=vanilla_dir,
            baseline_iterations=baseline_iterations,
            candidate_polish_iterations=candidate_polish_iterations,
            local_polish_iterations=local_polish_iterations,
            max_target_stages=max_target_stages,
            beam_width=beam_width,
            target_action_multiplier=target_action_multiplier,
            max_target_action_nodes=max_target_action_nodes,
            selection_policies=selection_policies,
            cumulative_fraction=cumulative_fraction,
            min_score_fraction=min_score_fraction,
            min_gap_fraction=min_gap_fraction,
            min_guarded_pull_fraction=min_guarded_pull_fraction,
            resolution=resolution,
            randomness=randomness,
            perturb_seed_offset=perturb_seed_offset,
            polish_seed_offset=polish_seed_offset,
            min_support_shift_from_vanilla=min_support_shift_from_vanilla,
            min_material_q_gain=min_material_q_gain,
            support_gate=support_gate,
        )
        state_frames.append(states)
        edge_frames.append(edges)
        path_frames.append(paths)
    rows = pd.concat(state_frames, ignore_index=True) if state_frames else pd.DataFrame()
    edges = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    path_rows = pd.concat(path_frames, ignore_index=True) if path_frames else pd.DataFrame()
    frontier_rows = select_branch_path_rows(
        path_rows,
        beam_width=100,
        diversity_columns=("path_selection_policy",),
    )
    wall_frontier = select_qf_wall_frontier(path_rows, max_rows=100)
    frontier_rows = pd.concat([frontier_rows, wall_frontier], ignore_index=True)
    if not frontier_rows.empty:
        frontier_rows = (
            frontier_rows.drop_duplicates(subset=["path_final_state_id"])
            .sort_values(
                [
                    "path_branch_discovery_score",
                    "path_final_support_distance_to_vanilla",
                    "path_final_delta_q_vs_start",
                    "path_q_wall",
                ],
                ascending=[False, False, False, True],
            )
            .head(100)
        )
    case_rows = _case_rows(path_rows, support_gate=support_gate)
    wall_summary_rows = summarize_pathway_wall_rows(
        path_rows,
        support_gate=support_gate,
    )
    rows.to_csv(output_dir / STATE_ROWS_FILENAME, index=False)
    edges.to_csv(output_dir / EDGE_ROWS_FILENAME, index=False)
    path_rows.to_csv(output_dir / PATH_ROWS_FILENAME, index=False)
    frontier_rows.to_csv(output_dir / FRONTIER_ROWS_FILENAME, index=False)
    case_rows.to_csv(output_dir / CASE_ROWS_FILENAME, index=False)
    config = {
        "prefix_dir": str(prefix_dir),
        "profile_batch_dir": str(profile_batch_dir),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "vanilla_dir": str(vanilla_dir),
        "pair_ids": list(pair_ids),
        "top_prefixes_per_case": int(top_prefixes_per_case),
        "baseline_iterations": int(baseline_iterations),
        "candidate_polish_iterations": int(candidate_polish_iterations),
        "local_polish_iterations": int(local_polish_iterations),
        "max_target_stages": int(max_target_stages),
        "beam_width": int(beam_width),
        "target_action_multiplier": float(target_action_multiplier),
        "max_target_action_nodes": int(max_target_action_nodes),
        "selection_policies": list(selection_policies),
        "cumulative_fraction": float(cumulative_fraction),
        "min_score_fraction": float(min_score_fraction),
        "min_gap_fraction": float(min_gap_fraction),
        "min_guarded_pull_fraction": float(min_guarded_pull_fraction),
        "resolution": float(resolution),
        "randomness": float(randomness),
        "perturb_seed_offset": int(perturb_seed_offset),
        "polish_seed_offset": int(polish_seed_offset),
        "min_support_shift_from_vanilla": float(min_support_shift_from_vanilla),
        "min_material_q_gain": float(min_material_q_gain),
        "support_gate": float(support_gate),
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "schema": "leiden_basin_branch_target_growth.v0",
        "output_dir": str(output_dir),
        "state_rows": int(len(rows)),
        "edge_rows": int(len(edges)),
        "path_rows": int(len(path_rows)),
        "frontier_rows": int(len(frontier_rows)),
        "case_rows": int(len(case_rows)),
        **{
            key: value
            for key, value in config.items()
            if key
            in {
                "prefix_dir",
                "profile_batch_dir",
                "pair_ids",
                "top_prefixes_per_case",
                "local_polish_iterations",
                "max_target_stages",
                "beam_width",
                "target_action_multiplier",
                "max_target_action_nodes",
                "selection_policies",
                "support_gate",
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
        path_rows=path_rows,
        frontier_rows=frontier_rows,
        case_rows=case_rows,
        wall_summary_rows=wall_summary_rows,
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
    parser.add_argument("--pair-ids", default="c0-s11-r0.001")
    parser.add_argument("--top-prefixes-per-case", type=int, default=10)
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--candidate-polish-iterations", type=int, default=5)
    parser.add_argument("--local-polish-iterations", type=int, default=3)
    parser.add_argument("--max-target-stages", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--target-action-multiplier", type=float, default=0.5)
    parser.add_argument("--max-target-action-nodes", type=int, default=64)
    parser.add_argument(
        "--selection-policies",
        default=",".join(
            (
                TARGET_SELECTION_GUARDED_ELBOW,
                TARGET_SELECTION_FIXED_CAP,
                TARGET_SELECTION_FIXED_TAIL_BACKFILL,
            )
        ),
    )
    parser.add_argument("--cumulative-fraction", type=float, default=0.80)
    parser.add_argument("--min-score-fraction", type=float, default=0.05)
    parser.add_argument("--min-gap-fraction", type=float, default=0.25)
    parser.add_argument("--min-guarded-pull-fraction", type=float, default=0.50)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    parser.add_argument("--polish-seed-offset", type=int, default=19000)
    parser.add_argument("--min-support-shift-from-vanilla", type=float, default=0.05)
    parser.add_argument("--min-material-q-gain", type=float, default=0.0)
    parser.add_argument("--support-gate", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selection_policies = _parse_csv_tuple(
        args.selection_policies,
        TARGET_SELECTION_POLICIES,
    )
    summary = run_search(
        prefix_dir=args.prefix_dir,
        profile_batch_dir=args.profile_batch_dir,
        output_dir=args.output_dir,
        candidate_dirs=tuple(args.candidate_dir or DEFAULT_CANDIDATE_DIRS),
        vanilla_dir=args.vanilla_dir,
        pair_ids=_parse_csv_tuple(args.pair_ids),
        top_prefixes_per_case=args.top_prefixes_per_case,
        baseline_iterations=args.baseline_iterations,
        candidate_polish_iterations=args.candidate_polish_iterations,
        local_polish_iterations=args.local_polish_iterations,
        max_target_stages=args.max_target_stages,
        beam_width=args.beam_width,
        target_action_multiplier=args.target_action_multiplier,
        max_target_action_nodes=args.max_target_action_nodes,
        selection_policies=selection_policies,
        cumulative_fraction=args.cumulative_fraction,
        min_score_fraction=args.min_score_fraction,
        min_gap_fraction=args.min_gap_fraction,
        min_guarded_pull_fraction=args.min_guarded_pull_fraction,
        resolution=args.resolution,
        randomness=args.randomness,
        perturb_seed_offset=args.perturb_seed_offset,
        polish_seed_offset=args.polish_seed_offset,
        min_support_shift_from_vanilla=args.min_support_shift_from_vanilla,
        min_material_q_gain=args.min_material_q_gain,
        support_gate=args.support_gate,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
