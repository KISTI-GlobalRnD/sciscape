#!/usr/bin/env python3
"""Probe one-step recovery moves from a post-gate near-miss basin state."""

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
from evaluate_leiden_basin_target_elbow_polish import (  # noqa: E402
    PATH_POLICIES,
    PATH_POLICY_GUARDED_BACKFILL,
    SELECTION_FIXED_TAIL_BACKFILL,
    _rank_and_filter_prefix_rows,
    _selected_k_for_policy,
    _selection_context,
    _target_selection_policy,
)
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
    ACTION_BOUNDARY_SHELL_TOPK,
    ACTION_CANDIDATE_CLOSURE_TOPK,
    ACTION_PREFIX_ONLY,
    ACTION_REMAINING_TARGET_TOPK,
    ACTION_VANILLA_CLOSURE_TOPK,
    POST_GATE_RECOVERY_MOVE_RECOVERED,
    POST_GATE_RECOVERY_MOVE_Q_GAIN,
    POST_GATE_VERDICT_NEAR_MISS,
    TARGET_SELECTION_FIXED_CAP,
    TARGET_SELECTION_GUARDED_ELBOW,
    TARGET_SELECTION_PREFIX_POLISH,
    TARGET_SELECTION_RAW_PREFIX,
    TransitionAction,
    annotate_pathway_debt_area_rows,
    annotate_post_gate_recovery_step_rows,
    annotate_tunneling_evidence_rows,
    build_post_gate_recovery_actions,
    cap_context_count,
    classify_post_gate_recovery_move_rows,
    compute_pathway_wall_rows,
    edge_public_row,
    make_prefix_state,
    node_csv,
    prefix_direct_nodes,
    remaining_target_elbow_summary,
    remaining_target_pull_frame,
    summarize_post_gate_recovery_paths,
    trace_tunneling_path_states,
    unique_sorted_u32,
)
from search_leiden_basin_transitions import (  # noqa: E402
    _evaluate_state,
    _polished_child,
)

COMBINED_DIR = REPO_ROOT / (
    "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30"
)
DEFAULT_POST_GATE_DIR = COMBINED_DIR / "basin_transition_post_gate_recovery_field34_cc_c0_v0"
DEFAULT_OUTPUT_DIR = COMBINED_DIR / "basin_transition_post_gate_recovery_moves_field34_cc_c0_p8_v0"
PATH_POLICY_BRANCH_TARGET_GROWTH = "branch_target_growth"

POST_GATE_PATH_SUMMARY_FILENAME = "post_gate_recovery_path_summary_rows.csv"
STATE_ROWS_FILENAME = "post_gate_recovery_move_states.csv"
EDGE_ROWS_FILENAME = "post_gate_recovery_move_edges.csv"
MOVE_ROWS_FILENAME = "post_gate_recovery_move_rows.csv"
PATH_ROWS_FILENAME = "post_gate_recovery_move_path_rows.csv"
TRACE_ROWS_FILENAME = "post_gate_recovery_move_trace_rows.csv"
STEP_ROWS_FILENAME = "post_gate_recovery_move_step_rows.csv"
SUMMARY_ROWS_FILENAME = "post_gate_recovery_move_path_summary_rows.csv"
SUMMARY_FILENAME = "post_gate_recovery_move_summary.json"
CONFIG_FILENAME = "post_gate_recovery_move_config.json"
REPORT_FILENAME = "post_gate_recovery_move_report.md"

def _parse_csv_tuple(value: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if not value.strip():
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip())

def _parse_node_ids(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.uint32)
    if isinstance(value, float) and math.isnan(value):
        return np.asarray([], dtype=np.uint32)
    text = str(value).strip()
    if not text:
        return np.asarray([], dtype=np.uint32)
    return unique_sorted_u32([int(part) for part in text.split(",") if part.strip()])

def _recorded_path_to_state(
    recorded_state_rows: pd.DataFrame,
    final_state_id: str,
) -> pd.DataFrame:
    if recorded_state_rows is None or recorded_state_rows.empty:
        raise ValueError("Recorded state rows are required for branch replay")
    rows_by_id = {
        str(row["state_id"]): row
        for _, row in recorded_state_rows.iterrows()
    }
    path_rows: list[pd.Series] = []
    current_id = str(final_state_id)
    while current_id:
        row = rows_by_id.get(current_id)
        if row is None:
            raise ValueError(f"Missing recorded state row for {current_id}")
        path_rows.append(row)
        parent = row.get("parent_state_id", "")
        if parent is None or (isinstance(parent, float) and math.isnan(parent)):
            break
        parent_id = str(parent)
        if not parent_id or parent_id == "nan":
            break
        current_id = parent_id
    path_rows.reverse()
    return pd.DataFrame(path_rows)

def _child_index_from_state_id(state_id: str) -> int:
    tail = str(state_id).rsplit("/", 1)[-1]
    if ":" not in tail:
        return 1
    try:
        return int(tail.rsplit(":", 1)[-1])
    except ValueError:
        return 1

def _recorded_int(row: pd.Series, key: str, default: int) -> int:
    value = row.get(key, default)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return int(default)
    return int(value)

def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

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

def _count_target_steps(state_id: str) -> int:
    return str(state_id).count(f"/{ACTION_REMAINING_TARGET_TOPK}:")

def _prefix_context(prefix_row: pd.Series) -> dict[str, Any]:
    return {
        "barrier_aware_score": float(prefix_row["barrier_aware_score"]),
        "peak_raw_barrier_input": float(prefix_row["peak_raw_barrier"]),
        "support_progress_fraction_input": float(prefix_row["support_progress_fraction"]),
        "greedy_failure_labels": prefix_row["greedy_failure_labels"],
    }

def _select_source_path(
    path_summary: pd.DataFrame,
    *,
    pair_id: str,
    prefix_rank: int,
    verdict: str,
) -> pd.Series:
    rows = path_summary[
        path_summary["pair_id"].astype(str).eq(str(pair_id))
        & path_summary["path_prefix_rank"].astype(int).eq(int(prefix_rank))
        & path_summary["post_gate_verdict"].astype(str).eq(str(verdict))
    ].copy()
    if rows.empty:
        raise ValueError(
            f"No post-gate source row for pair={pair_id}, prefix={prefix_rank}, verdict={verdict}"
        )
    rows = rows.sort_values(
        [
            "post_gate_best_delta_q_gain_from_gate",
            "post_gate_final_delta_q",
            "post_gate_final_support",
            "post_gate_final_target_progress",
            "post_gate_step_count",
        ],
        ascending=[False, False, False, False, True],
    )
    return rows.iloc[0]

def _load_case_context(
    *,
    prefix_row: pd.Series,
    profile_batch_dir: Path,
    candidate_dirs: tuple[Path, ...],
    vanilla_dir: Path,
    baseline_iterations: int,
    candidate_polish_iterations: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
) -> dict[str, Any]:
    case = str(prefix_row["case"])
    pair_id = str(prefix_row["pair_id"])
    candidate_index = int(prefix_row["candidate_index"])
    vanilla_seed = int(prefix_row["vanilla_seed"])
    vanilla_randomness = float(prefix_row["vanilla_randomness"])
    vanilla_n = str(prefix_row["vanilla_requested_n_iterations"])
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
        candidate_rows=candidate_rows[candidate_rows["case"].astype(str).eq(case)],
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
    return {
        "case": case,
        "pair_id": pair_id,
        "candidate_index": candidate_index,
        "vanilla_seed": vanilla_seed,
        "vanilla_randomness": vanilla_randomness,
        "vanilla_n": vanilla_n,
        "profile_dir": profile_dir,
        "units": units,
        "graph": graph,
        "arrays": arrays,
        "baseline": baseline,
        "candidate": candidate,
        "vanilla": vanilla,
        "sketch_nodes": sketch_nodes,
        "target_nodes": target_nodes,
        "candidate_support_node_count": int(candidate_support.size),
        "vanilla_support_node_count": int(vanilla_support.size),
        "vanilla_support_distance_to_candidate": float(
            vanilla_support_distance_to_candidate
        ),
        "public_context": {
            "case": case,
            "field": prefix_row.get("field", ""),
            "method": prefix_row.get("method", ""),
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
        },
    }

def _replay_recorded_branch_source_state(
    *,
    prefix_row: pd.Series,
    source_path: pd.Series,
    case_ctx: dict[str, Any],
    recorded_state_rows: pd.DataFrame,
    local_polish_iterations: int,
    resolution: float,
    randomness: float,
    polish_seed_offset: int,
    min_support_shift_from_vanilla: float,
    min_material_q_gain: float,
) -> tuple[Any, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    graph = case_ctx["graph"]
    baseline = case_ctx["baseline"]
    candidate = case_ctx["candidate"]
    vanilla = case_ctx["vanilla"]
    target_nodes = case_ctx["target_nodes"]
    units = case_ctx["units"]
    pair_id = str(prefix_row["pair_id"])
    prefix_rank = int(prefix_row["selected_prefix_rank"])
    prefix_context = {
        **case_ctx["public_context"],
        **_prefix_context(prefix_row),
    }
    recorded_path = _recorded_path_to_state(
        recorded_state_rows,
        str(source_path["path_final_state_id"]),
    )

    raw_membership, mutable_nodes = apply_prefix_units(
        membership=vanilla.membership,
        donor_membership=candidate.recreated.membership,
        units=units,
        prefix_unit_ids=prefix_row["prefix_unit_ids"],
    )
    raw_quality = score_membership(graph, raw_membership, resolution=resolution)
    direct_nodes = prefix_direct_nodes(units, prefix_row["prefix_unit_ids"])
    rows: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
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
        sketch_nodes=case_ctx["sketch_nodes"],
        start_quality=vanilla.quality,
        candidate_quality=candidate.recreated.quality,
        vanilla_quality=vanilla.quality,
        vanilla_support_distance_to_candidate=case_ctx[
            "vanilla_support_distance_to_candidate"
        ],
        context={
            **prefix_context,
            **_selection_context(
                path_policy=PATH_POLICY_BRANCH_TARGET_GROWTH,
                selection_policy=TARGET_SELECTION_RAW_PREFIX,
                escalation_reason="recorded_branch_replay",
                target_stage_index=0,
                selected=np.asarray([], dtype=np.uint32),
            ),
        },
        min_support_shift_from_vanilla=min_support_shift_from_vanilla,
        min_material_q_gain=min_material_q_gain,
    )
    root_row["path_elapsed_sec"] = 0.0
    rows.append(root_row)

    current = root
    current_row = root_row
    current_path_elapsed = 0.0
    for step_index, (_, recorded) in enumerate(recorded_path.iloc[1:].iterrows(), start=1):
        action_type = str(recorded["action_type"])
        if action_type not in {ACTION_PREFIX_ONLY, ACTION_REMAINING_TARGET_TOPK}:
            raise ValueError(f"Unsupported recorded branch action: {action_type}")
        selected = _parse_node_ids(recorded.get("selected_node_ids"))
        action_params = recorded.get("action_params", "")
        if action_params is None or (isinstance(action_params, float) and math.isnan(action_params)):
            action_params = "branch_target_growth;recorded_replay"
        action = TransitionAction(
            action_type=action_type,
            action_params=str(action_params),
            context_nodes=np.asarray([], dtype=np.uint32),
            action_nodes=selected if action_type == ACTION_REMAINING_TARGET_TOPK else None,
        )
        child_index = _child_index_from_state_id(str(recorded["state_id"]))
        child = _polished_child(
            parent=current,
            action=action,
            graph=graph,
            donor_membership=candidate.recreated.membership,
            resolution=resolution,
            seed=(
                int(polish_seed_offset)
                + int(prefix_rank) * 1000
                + int(step_index) * 100
                + int(child_index)
            ),
            n_iterations=local_polish_iterations,
            randomness=randomness,
            child_index=child_index,
        )
        row = _evaluate_state(
            state=child,
            baseline_membership=baseline.membership,
            candidate_membership=candidate.recreated.membership,
            vanilla_membership=vanilla.membership,
            sketch_nodes=case_ctx["sketch_nodes"],
            start_quality=vanilla.quality,
            candidate_quality=candidate.recreated.quality,
            vanilla_quality=vanilla.quality,
            vanilla_support_distance_to_candidate=case_ctx[
                "vanilla_support_distance_to_candidate"
            ],
            context={
                **prefix_context,
                "path_policy": PATH_POLICY_BRANCH_TARGET_GROWTH,
                "selection_policy": recorded.get("selection_policy", ""),
                "escalation_reason": recorded.get(
                    "escalation_reason",
                    "recorded_branch_replay",
                ),
                "escalated_to_fixed": bool(recorded.get("escalated_to_fixed", False)),
                "target_stage_index": _recorded_int(
                    recorded,
                    "target_stage_index",
                    0,
                ),
                "selected_k": _recorded_int(recorded, "selected_k", selected.size),
                "selected_node_ids": node_csv(selected),
                "fixed_effective_k": _recorded_int(
                    recorded,
                    "fixed_effective_k",
                    selected.size,
                ),
                "guarded_elbow_k": _recorded_int(
                    recorded,
                    "guarded_elbow_k",
                    selected.size,
                ),
            },
            parent_row=current_row,
            min_support_shift_from_vanilla=min_support_shift_from_vanilla,
            min_material_q_gain=min_material_q_gain,
        )
        current_path_elapsed += float(child.elapsed_sec)
        row["path_elapsed_sec"] = current_path_elapsed
        rows.append(row)
        edges.append(
            edge_public_row(
                parent_state_id=current.state_id,
                child_state_id=child.state_id,
                action=action,
                context={
                    **case_ctx["public_context"],
                    "path_policy": PATH_POLICY_BRANCH_TARGET_GROWTH,
                },
            )
        )
        current = child
        current_row = row
    return current, current_row, pd.DataFrame(rows), pd.DataFrame(edges)

def _replay_to_source_state(
    *,
    prefix_row: pd.Series,
    source_path: pd.Series,
    case_ctx: dict[str, Any],
    target_action_multiplier: float,
    max_target_action_nodes: int,
    cumulative_fraction: float,
    min_score_fraction: float,
    min_gap_fraction: float,
    min_guarded_pull_fraction: float,
    local_polish_iterations: int,
    resolution: float,
    randomness: float,
    polish_seed_offset: int,
    min_support_shift_from_vanilla: float,
    min_material_q_gain: float,
    recorded_state_rows: pd.DataFrame | None = None,
) -> tuple[Any, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if str(source_path.get("path_policy", "")) == PATH_POLICY_BRANCH_TARGET_GROWTH:
        return _replay_recorded_branch_source_state(
            prefix_row=prefix_row,
            source_path=source_path,
            case_ctx=case_ctx,
            recorded_state_rows=pd.DataFrame()
            if recorded_state_rows is None
            else recorded_state_rows,
            local_polish_iterations=local_polish_iterations,
            resolution=resolution,
            randomness=randomness,
            polish_seed_offset=polish_seed_offset,
            min_support_shift_from_vanilla=min_support_shift_from_vanilla,
            min_material_q_gain=min_material_q_gain,
        )
    graph = case_ctx["graph"]
    arrays = case_ctx["arrays"]
    baseline = case_ctx["baseline"]
    candidate = case_ctx["candidate"]
    vanilla = case_ctx["vanilla"]
    target_nodes = case_ctx["target_nodes"]
    units = case_ctx["units"]
    src = np.asarray(arrays.src, dtype=np.uint32)
    dst = np.asarray(arrays.dst, dtype=np.uint32)
    weight = np.asarray(arrays.weight, dtype=np.float64)
    node_count = int(baseline.membership.size)
    pair_id = str(prefix_row["pair_id"])
    prefix_rank = int(prefix_row["selected_prefix_rank"])
    path_policy = str(source_path["path_policy"])
    policy_index = list(PATH_POLICIES).index(path_policy) + 1
    target_stage_count = _count_target_steps(str(source_path["path_final_state_id"]))
    prefix_context = {
        **case_ctx["public_context"],
        **_prefix_context(prefix_row),
    }

    raw_membership, mutable_nodes = apply_prefix_units(
        membership=vanilla.membership,
        donor_membership=candidate.recreated.membership,
        units=units,
        prefix_unit_ids=prefix_row["prefix_unit_ids"],
    )
    raw_quality = score_membership(graph, raw_membership, resolution=resolution)
    direct_nodes = prefix_direct_nodes(units, prefix_row["prefix_unit_ids"])
    rows: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    root = make_prefix_state(
        state_id=f"{pair_id}:p{prefix_rank}:{path_policy}:raw",
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
        sketch_nodes=case_ctx["sketch_nodes"],
        start_quality=vanilla.quality,
        candidate_quality=candidate.recreated.quality,
        vanilla_quality=vanilla.quality,
        vanilla_support_distance_to_candidate=case_ctx[
            "vanilla_support_distance_to_candidate"
        ],
        context={
            **prefix_context,
            **_selection_context(
                path_policy=path_policy,
                selection_policy=TARGET_SELECTION_RAW_PREFIX,
                escalation_reason="not_applicable",
                target_stage_index=0,
                selected=np.asarray([], dtype=np.uint32),
            ),
        },
        min_support_shift_from_vanilla=min_support_shift_from_vanilla,
        min_material_q_gain=min_material_q_gain,
    )
    root_row["path_elapsed_sec"] = 0.0
    rows.append(root_row)
    prefix_action = TransitionAction(
        action_type=ACTION_PREFIX_ONLY,
        action_params=f"path_policy={path_policy};local_polish",
        context_nodes=np.asarray([], dtype=np.uint32),
    )
    prefix_polished = _polished_child(
        parent=root,
        action=prefix_action,
        graph=graph,
        donor_membership=candidate.recreated.membership,
        resolution=resolution,
        seed=int(polish_seed_offset)
        + int(policy_index) * 100000
        + int(prefix_rank) * 1000,
        n_iterations=local_polish_iterations,
        randomness=randomness,
        child_index=1,
    )
    current = prefix_polished
    current_row = _evaluate_state(
        state=current,
        baseline_membership=baseline.membership,
        candidate_membership=candidate.recreated.membership,
        vanilla_membership=vanilla.membership,
        sketch_nodes=case_ctx["sketch_nodes"],
        start_quality=vanilla.quality,
        candidate_quality=candidate.recreated.quality,
        vanilla_quality=vanilla.quality,
        vanilla_support_distance_to_candidate=case_ctx[
            "vanilla_support_distance_to_candidate"
        ],
        context={
            **prefix_context,
            **_selection_context(
                path_policy=path_policy,
                selection_policy=TARGET_SELECTION_PREFIX_POLISH,
                escalation_reason="not_applicable",
                target_stage_index=0,
                selected=np.asarray([], dtype=np.uint32),
            ),
        },
        parent_row=root_row,
        min_support_shift_from_vanilla=min_support_shift_from_vanilla,
        min_material_q_gain=min_material_q_gain,
    )
    current_path_elapsed = float(current.elapsed_sec)
    current_row["path_elapsed_sec"] = current_path_elapsed
    rows.append(current_row)
    edges.append(
        edge_public_row(
            parent_state_id=root.state_id,
            child_state_id=current.state_id,
            action=prefix_action,
            context={**case_ctx["public_context"], "path_policy": path_policy},
        )
    )

    pending_backfill_nodes = np.asarray([], dtype=np.uint32)
    for target_stage_index in range(1, int(target_stage_count) + 1):
        anchor_count = int(unique_sorted_u32(current.action_nodes).size)
        fixed_k = cap_context_count(
            direct_node_count=anchor_count,
            context_multiplier=target_action_multiplier,
            max_context_nodes=max_target_action_nodes,
        )
        frame = remaining_target_pull_frame(
            state=current,
            src=src,
            dst=dst,
            weight=weight,
            node_count=node_count,
        )
        elbow = remaining_target_elbow_summary(
            frame,
            fixed_k=fixed_k,
            cumulative_fraction=cumulative_fraction,
            min_score_fraction=min_score_fraction,
            min_gap_fraction=min_gap_fraction,
            min_guarded_pull_fraction=min_guarded_pull_fraction,
        )
        remaining = set(
            int(node)
            for node in unique_sorted_u32(
                frame["node"].to_numpy(dtype=np.uint32)
                if not frame.empty
                else np.asarray([], dtype=np.uint32)
            )
        )
        backfill = np.asarray(
            [int(node) for node in pending_backfill_nodes if int(node) in remaining],
            dtype=np.uint32,
        )
        parent_support_shift = float(
            current_row.get("state_support_distance_to_vanilla", 0.0)
        )
        use_backfill = (
            path_policy == PATH_POLICY_GUARDED_BACKFILL
            and backfill.size > 0
            and parent_support_shift < float(min_support_shift_from_vanilla)
        )
        if use_backfill:
            selection_policy = SELECTION_FIXED_TAIL_BACKFILL
            escalation_reason = "below_support_gate_backfill"
            selected = unique_sorted_u32(backfill)
        else:
            selection_policy, escalation_reason = _target_selection_policy(
                path_policy=path_policy,
                target_stage_index=target_stage_index,
                parent_row=current_row,
                min_support_shift_from_vanilla=min_support_shift_from_vanilla,
            )
            selected_k = _selected_k_for_policy(elbow, selection_policy)
            selected = (
                np.asarray(frame.head(selected_k)["node"], dtype=np.uint32)
                if selected_k > 0 and not frame.empty
                else np.asarray([], dtype=np.uint32)
            )
        if selected.size == 0:
            break
        select_context = _selection_context(
            path_policy=path_policy,
            selection_policy=selection_policy,
            escalation_reason=escalation_reason,
            target_stage_index=target_stage_index,
            selected=selected,
            elbow=elbow,
        )
        action = TransitionAction(
            action_type=ACTION_REMAINING_TARGET_TOPK,
            action_params=(
                f"path_policy={path_policy};selection_policy={selection_policy};"
                f"escalation_reason={escalation_reason};"
                f"target_stage={int(target_stage_index)};"
                f"selected_k={int(selected.size)}"
            ),
            context_nodes=np.asarray([], dtype=np.uint32),
            action_nodes=selected,
        )
        child = _polished_child(
            parent=current,
            action=action,
            graph=graph,
            donor_membership=candidate.recreated.membership,
            resolution=resolution,
            seed=int(polish_seed_offset)
            + int(policy_index) * 100000
            + int(prefix_rank) * 1000
            + int(target_stage_index),
            n_iterations=local_polish_iterations,
            randomness=randomness,
            child_index=target_stage_index,
        )
        row = _evaluate_state(
            state=child,
            baseline_membership=baseline.membership,
            candidate_membership=candidate.recreated.membership,
            vanilla_membership=vanilla.membership,
            sketch_nodes=case_ctx["sketch_nodes"],
            start_quality=vanilla.quality,
            candidate_quality=candidate.recreated.quality,
            vanilla_quality=vanilla.quality,
            vanilla_support_distance_to_candidate=case_ctx[
                "vanilla_support_distance_to_candidate"
            ],
            context={**prefix_context, **select_context},
            parent_row=current_row,
            min_support_shift_from_vanilla=min_support_shift_from_vanilla,
            min_material_q_gain=min_material_q_gain,
        )
        current_path_elapsed += float(child.elapsed_sec)
        row["path_elapsed_sec"] = current_path_elapsed
        rows.append(row)
        edges.append(
            edge_public_row(
                parent_state_id=current.state_id,
                child_state_id=child.state_id,
                action=action,
                context={**case_ctx["public_context"], "path_policy": path_policy},
            )
        )
        if (
            path_policy == PATH_POLICY_GUARDED_BACKFILL
            and selection_policy == TARGET_SELECTION_GUARDED_ELBOW
            and int(elbow["fixed_effective_k"]) > int(elbow["guarded_elbow_k"])
            and not frame.empty
        ):
            pending_backfill_nodes = np.asarray(
                frame.iloc[
                    int(elbow["guarded_elbow_k"]) : int(elbow["fixed_effective_k"])
                ]["node"],
                dtype=np.uint32,
            )
        else:
            pending_backfill_nodes = np.asarray([], dtype=np.uint32)
        current = child
        current_row = row
    return current, current_row, pd.DataFrame(rows), pd.DataFrame(edges)

def _write_report(
    path: Path,
    *,
    summary: dict[str, Any],
    source_path: pd.Series,
    move_rows: pd.DataFrame,
    recovery_path_rows: pd.DataFrame,
) -> None:
    lines = [
        "# Post-Gate Recovery Move Probe",
        "",
        "This artifact starts from one p8 near-miss state and probes bounded",
        "recovery moves around the gate-crossed state.  It is diagnostic-only.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "output_dir",
        "source_pair_id",
        "source_prefix_rank",
        "source_path_policy",
        "source_state_id",
        "source_delta_q",
        "source_support",
        "source_target_progress",
        "move_rows",
        "q_recovered_support_retained_rows",
        "q_gain_support_retained_rows",
        "quality_regression_rows",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")
    lines.extend(["", "## Source Path", ""])
    source_cols = [
        "path_prefix_rank",
        "path_policy",
        "path_selection_policy",
        "post_gate_verdict",
        "post_gate_gate_delta_q",
        "post_gate_best_delta_q",
        "post_gate_final_delta_q",
        "post_gate_final_support",
        "post_gate_final_target_progress",
        "path_final_state_id",
    ]
    lines.extend(
        _markdown_table(
            pd.DataFrame([source_path])[
                [column for column in source_cols if column in source_path.index]
            ]
        )
    )
    lines.extend(["", "## Recovery Moves", ""])
    move_cols = [
        "recovery_policy",
        "recovery_move_kind",
        "recovery_selected_node_count",
        "post_gate_move_verdict",
        "post_gate_move_delta_q_gain",
        "state_delta_q_vs_start",
        "post_gate_move_support_gain",
        "state_support_distance_to_vanilla",
        "post_gate_move_target_progress_gain",
        "state_target_progress_from_vanilla",
        "mutable_node_count",
        "elapsed_sec",
        "state_id",
    ]
    display_moves = move_rows.sort_values(
        [
            "post_gate_move_q_recovered",
            "post_gate_move_delta_q_gain",
            "state_support_distance_to_vanilla",
            "state_target_progress_from_vanilla",
            "mutable_node_count",
        ],
        ascending=[False, False, False, False, True],
    )
    lines.extend(
        _markdown_table(
            display_moves[[column for column in move_cols if column in display_moves]],
            max_rows=80,
        )
    )
    lines.extend(["", "## Recovery Path Rows", ""])
    path_cols = [
        "path_final_state_id",
        "path_q_wall",
        "path_q_debt_area_step",
        "path_final_delta_q_vs_start",
        "path_final_support_distance_to_vanilla",
        "path_final_target_progress_from_vanilla",
        "path_final_mutable_node_count",
        "tunnel_route_label",
    ]
    lines.extend(
        _markdown_table(
            recovery_path_rows[[column for column in path_cols if column in recovery_path_rows]],
            max_rows=80,
        )
    )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `q_gain_support_retained` is a useful near-miss improvement, not success.",
            "- `q_recovered_support_retained` would be the first strong recovery signal.",
            "- A regression row means the candidate move made the basin wall worse even if it added context.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_probe(
    *,
    post_gate_dir: Path,
    prefix_dir: Path,
    profile_batch_dir: Path,
    output_dir: Path,
    candidate_dirs: tuple[Path, ...],
    vanilla_dir: Path,
    pair_id: str,
    prefix_rank: int,
    source_verdict: str,
    baseline_iterations: int,
    candidate_polish_iterations: int,
    local_polish_iterations: int,
    recovery_polish_iterations: int,
    target_action_multiplier: float,
    max_target_action_nodes: int,
    recovery_context_multiplier: float,
    max_recovery_context_nodes: int,
    recovery_action_types: tuple[str, ...],
    include_boundary_transplant: bool,
    cumulative_fraction: float,
    min_score_fraction: float,
    min_gap_fraction: float,
    min_guarded_pull_fraction: float,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
    polish_seed_offset: int,
    recovery_seed_offset: int,
    min_support_shift_from_vanilla: float,
    min_material_q_gain: float,
    support_gate: float,
    progress_margin: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    post_gate_paths = pd.read_csv(post_gate_dir / POST_GATE_PATH_SUMMARY_FILENAME)
    post_gate_config = _load_json(post_gate_dir / "post_gate_recovery_config.json")
    recorded_state_rows: pd.DataFrame | None = None
    recorded_state_dir = post_gate_config.get("state_dir")
    recorded_state_filename = post_gate_config.get("state_rows_filename")
    if recorded_state_dir and recorded_state_filename:
        recorded_state_path = Path(str(recorded_state_dir)) / str(recorded_state_filename)
        if recorded_state_path.exists():
            recorded_state_rows = pd.read_csv(recorded_state_path)
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
        baseline_iterations=baseline_iterations,
        candidate_polish_iterations=candidate_polish_iterations,
        resolution=resolution,
        randomness=randomness,
        perturb_seed_offset=perturb_seed_offset,
    )
    source_state, source_row, replay_rows, replay_edges = _replay_to_source_state(
        prefix_row=prefix_row,
        source_path=source_path,
        case_ctx=case_ctx,
        recorded_state_rows=recorded_state_rows,
        target_action_multiplier=target_action_multiplier,
        max_target_action_nodes=max_target_action_nodes,
        cumulative_fraction=cumulative_fraction,
        min_score_fraction=min_score_fraction,
        min_gap_fraction=min_gap_fraction,
        min_guarded_pull_fraction=min_guarded_pull_fraction,
        local_polish_iterations=local_polish_iterations,
        resolution=resolution,
        randomness=randomness,
        polish_seed_offset=polish_seed_offset,
        min_support_shift_from_vanilla=min_support_shift_from_vanilla,
        min_material_q_gain=min_material_q_gain,
    )
    arrays = case_ctx["arrays"]
    recovery_candidates = build_post_gate_recovery_actions(
        state=source_state,
        candidate_membership=case_ctx["candidate"].recreated.membership,
        vanilla_membership=case_ctx["vanilla"].membership,
        src=np.asarray(arrays.src, dtype=np.uint32),
        dst=np.asarray(arrays.dst, dtype=np.uint32),
        weight=np.asarray(arrays.weight, dtype=np.float64),
        node_count=int(case_ctx["baseline"].membership.size),
        action_types=recovery_action_types,
        context_multiplier=recovery_context_multiplier,
        max_context_nodes=max_recovery_context_nodes,
        include_context_only=True,
        include_candidate_transplant=True,
        include_boundary_transplant=include_boundary_transplant,
    )
    move_rows: list[dict[str, Any]] = []
    move_edges: list[dict[str, Any]] = []
    for index, candidate in enumerate(recovery_candidates, start=1):
        child = _polished_child(
            parent=source_state,
            action=candidate.action,
            graph=case_ctx["graph"],
            donor_membership=case_ctx["candidate"].recreated.membership,
            resolution=resolution,
            seed=int(recovery_seed_offset) + index,
            n_iterations=recovery_polish_iterations,
            randomness=randomness,
            child_index=index,
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
                **case_ctx["public_context"],
                **_prefix_context(prefix_row),
                "path_policy": "post_gate_recovery_move",
                "selection_policy": candidate.recovery_policy,
                "escalation_reason": "post_gate_probe",
                "target_stage_index": int(source_row.get("target_stage_index", 0)),
                "selected_k": int(candidate.selected_nodes.size),
                "selected_node_ids": node_csv(candidate.selected_nodes),
                "recovery_policy": candidate.recovery_policy,
                "recovery_source_action_type": candidate.source_action_type,
                "recovery_move_kind": candidate.move_kind,
                "recovery_selected_node_count": int(candidate.selected_nodes.size),
                "recovery_context_node_count": int(
                    unique_sorted_u32(candidate.action.context_nodes).size
                ),
                "recovery_action_node_count": int(
                    unique_sorted_u32(
                        []
                        if candidate.action.action_nodes is None
                        else candidate.action.action_nodes
                    ).size
                ),
                "recovery_source_state_id": source_state.state_id,
            },
            parent_row=source_row,
            min_support_shift_from_vanilla=min_support_shift_from_vanilla,
            min_material_q_gain=min_material_q_gain,
        )
        row["path_elapsed_sec"] = float(source_row.get("path_elapsed_sec", 0.0)) + float(
            child.elapsed_sec
        )
        move_rows.append(row)
        move_edges.append(
            edge_public_row(
                parent_state_id=source_state.state_id,
                child_state_id=child.state_id,
                action=candidate.action,
                context={
                    **case_ctx["public_context"],
                    "path_policy": "post_gate_recovery_move",
                    "recovery_policy": candidate.recovery_policy,
                },
            )
        )
    moves = pd.DataFrame(move_rows)
    moves = classify_post_gate_recovery_move_rows(
        moves,
        target_delta_q=float(source_row["state_delta_q_vs_start"]),
        target_support=float(source_row["state_support_distance_to_vanilla"]),
        target_progress=float(source_row["state_target_progress_from_vanilla"]),
        support_gate=support_gate,
        progress_margin=progress_margin,
    )
    state_rows = pd.concat([replay_rows, moves], ignore_index=True)
    edge_rows = pd.concat([replay_edges, pd.DataFrame(move_edges)], ignore_index=True)
    path_rows = compute_pathway_wall_rows(
        state_rows,
        source_label="post_gate_recovery_move_v0",
        support_gate=support_gate,
    )
    path_rows = annotate_pathway_debt_area_rows(
        path_rows,
        state_rows=state_rows,
        support_gate=support_gate,
    )
    path_rows = annotate_tunneling_evidence_rows(
        path_rows,
        support_gate=support_gate,
        progress_margin=progress_margin,
    )
    recovery_ids = set(moves["state_id"].astype(str))
    recovery_path_rows = path_rows[
        path_rows["path_final_state_id"].astype(str).isin(recovery_ids)
    ].copy()
    recovery_trace_rows = trace_tunneling_path_states(
        recovery_path_rows,
        state_rows=state_rows,
        support_gate=support_gate,
        progress_margin=progress_margin,
    )
    recovery_step_rows = annotate_post_gate_recovery_step_rows(recovery_trace_rows)
    recovery_summary_rows = summarize_post_gate_recovery_paths(recovery_trace_rows)

    state_rows.to_csv(output_dir / STATE_ROWS_FILENAME, index=False)
    edge_rows.to_csv(output_dir / EDGE_ROWS_FILENAME, index=False)
    moves.to_csv(output_dir / MOVE_ROWS_FILENAME, index=False)
    recovery_path_rows.to_csv(output_dir / PATH_ROWS_FILENAME, index=False)
    recovery_trace_rows.to_csv(output_dir / TRACE_ROWS_FILENAME, index=False)
    recovery_step_rows.to_csv(output_dir / STEP_ROWS_FILENAME, index=False)
    recovery_summary_rows.to_csv(output_dir / SUMMARY_ROWS_FILENAME, index=False)
    config = {
        "post_gate_dir": str(post_gate_dir),
        "prefix_dir": str(prefix_dir),
        "profile_batch_dir": str(profile_batch_dir),
        "output_dir": str(output_dir),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "vanilla_dir": str(vanilla_dir),
        "pair_id": pair_id,
        "prefix_rank": int(prefix_rank),
        "source_verdict": source_verdict,
        "baseline_iterations": int(baseline_iterations),
        "candidate_polish_iterations": int(candidate_polish_iterations),
        "local_polish_iterations": int(local_polish_iterations),
        "recovery_polish_iterations": int(recovery_polish_iterations),
        "target_action_multiplier": float(target_action_multiplier),
        "max_target_action_nodes": int(max_target_action_nodes),
        "recovery_context_multiplier": float(recovery_context_multiplier),
        "max_recovery_context_nodes": int(max_recovery_context_nodes),
        "recovery_action_types": list(recovery_action_types),
        "include_boundary_transplant": bool(include_boundary_transplant),
        "resolution": float(resolution),
        "randomness": float(randomness),
        "perturb_seed_offset": int(perturb_seed_offset),
        "polish_seed_offset": int(polish_seed_offset),
        "recovery_seed_offset": int(recovery_seed_offset),
        "support_gate": float(support_gate),
        "progress_margin": float(progress_margin),
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verdict_counts = moves["post_gate_move_verdict"].astype(str).value_counts().to_dict()
    summary = {
        "schema": "leiden_basin_post_gate_recovery_move_probe.v0",
        **config,
        "source_pair_id": pair_id,
        "source_prefix_rank": int(prefix_rank),
        "source_path_policy": str(source_path["path_policy"]),
        "source_state_id": source_state.state_id,
        "source_delta_q": float(source_row["state_delta_q_vs_start"]),
        "source_support": float(source_row["state_support_distance_to_vanilla"]),
        "source_target_progress": float(
            source_row["state_target_progress_from_vanilla"]
        ),
        "state_rows": int(len(state_rows)),
        "edge_rows": int(len(edge_rows)),
        "move_rows": int(len(moves)),
        "recovery_path_rows": int(len(recovery_path_rows)),
        "q_recovered_support_retained_rows": int(
            verdict_counts.get(POST_GATE_RECOVERY_MOVE_RECOVERED, 0)
        ),
        "q_gain_support_retained_rows": int(
            verdict_counts.get(POST_GATE_RECOVERY_MOVE_Q_GAIN, 0)
        ),
        "quality_regression_rows": int(verdict_counts.get("quality_regression", 0)),
        "verdict_counts": verdict_counts,
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir / REPORT_FILENAME,
        summary=summary,
        source_path=source_path,
        move_rows=moves,
        recovery_path_rows=recovery_path_rows,
    )
    return summary

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post-gate-dir", type=Path, default=DEFAULT_POST_GATE_DIR)
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--profile-batch-dir", type=Path, default=DEFAULT_PROFILE_BATCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-dir", type=Path, action="append", default=None)
    parser.add_argument("--vanilla-dir", type=Path, default=DEFAULT_VANILLA_DIR)
    parser.add_argument("--pair-id", default="c0-s11-r0.001")
    parser.add_argument("--prefix-rank", type=int, default=8)
    parser.add_argument("--source-verdict", default=POST_GATE_VERDICT_NEAR_MISS)
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--candidate-polish-iterations", type=int, default=5)
    parser.add_argument("--local-polish-iterations", type=int, default=3)
    parser.add_argument("--recovery-polish-iterations", type=int, default=3)
    parser.add_argument("--target-action-multiplier", type=float, default=0.5)
    parser.add_argument("--max-target-action-nodes", type=int, default=64)
    parser.add_argument("--recovery-context-multiplier", type=float, default=0.5)
    parser.add_argument("--max-recovery-context-nodes", type=int, default=64)
    parser.add_argument(
        "--recovery-action-types",
        default="candidate_closure_topk,vanilla_closure_topk,boundary_shell_topk",
    )
    parser.add_argument("--include-boundary-transplant", action="store_true")
    parser.add_argument("--cumulative-fraction", type=float, default=0.80)
    parser.add_argument("--min-score-fraction", type=float, default=0.05)
    parser.add_argument("--min-gap-fraction", type=float, default=0.25)
    parser.add_argument("--min-guarded-pull-fraction", type=float, default=0.50)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    parser.add_argument("--polish-seed-offset", type=int, default=11000)
    parser.add_argument("--recovery-seed-offset", type=int, default=21000)
    parser.add_argument("--min-support-shift-from-vanilla", type=float, default=0.05)
    parser.add_argument("--min-material-q-gain", type=float, default=0.0)
    parser.add_argument("--support-gate", type=float, default=0.05)
    parser.add_argument("--progress-margin", type=float, default=0.005)
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    action_types = _parse_csv_tuple(
        args.recovery_action_types,
        (
            ACTION_CANDIDATE_CLOSURE_TOPK,
            ACTION_VANILLA_CLOSURE_TOPK,
            ACTION_BOUNDARY_SHELL_TOPK,
        ),
    )
    unsupported_actions = sorted(
        set(action_types)
        - {
            ACTION_CANDIDATE_CLOSURE_TOPK,
            ACTION_VANILLA_CLOSURE_TOPK,
            ACTION_BOUNDARY_SHELL_TOPK,
        }
    )
    if unsupported_actions:
        raise ValueError(f"Unsupported recovery action types: {unsupported_actions}")
    candidate_dirs = (
        tuple(args.candidate_dir)
        if args.candidate_dir
        else tuple(DEFAULT_CANDIDATE_DIRS)
    )
    summary = run_probe(
        post_gate_dir=args.post_gate_dir,
        prefix_dir=args.prefix_dir,
        profile_batch_dir=args.profile_batch_dir,
        output_dir=args.output_dir,
        candidate_dirs=candidate_dirs,
        vanilla_dir=args.vanilla_dir,
        pair_id=args.pair_id,
        prefix_rank=args.prefix_rank,
        source_verdict=args.source_verdict,
        baseline_iterations=args.baseline_iterations,
        candidate_polish_iterations=args.candidate_polish_iterations,
        local_polish_iterations=args.local_polish_iterations,
        recovery_polish_iterations=args.recovery_polish_iterations,
        target_action_multiplier=args.target_action_multiplier,
        max_target_action_nodes=args.max_target_action_nodes,
        recovery_context_multiplier=args.recovery_context_multiplier,
        max_recovery_context_nodes=args.max_recovery_context_nodes,
        recovery_action_types=action_types,
        include_boundary_transplant=bool(args.include_boundary_transplant),
        cumulative_fraction=args.cumulative_fraction,
        min_score_fraction=args.min_score_fraction,
        min_gap_fraction=args.min_gap_fraction,
        min_guarded_pull_fraction=args.min_guarded_pull_fraction,
        resolution=args.resolution,
        randomness=args.randomness,
        perturb_seed_offset=args.perturb_seed_offset,
        polish_seed_offset=args.polish_seed_offset,
        recovery_seed_offset=args.recovery_seed_offset,
        min_support_shift_from_vanilla=args.min_support_shift_from_vanilla,
        min_material_q_gain=args.min_material_q_gain,
        support_gate=args.support_gate,
        progress_margin=args.progress_margin,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
