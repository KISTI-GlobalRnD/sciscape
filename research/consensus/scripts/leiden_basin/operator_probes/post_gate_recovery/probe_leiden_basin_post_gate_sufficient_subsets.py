#!/usr/bin/env python3
"""Search for sufficient post-gate recovery context subsets.

This diagnostic treats the full p8 vanilla-closure context release as an oracle
and asks whether structured group removal can preserve most of its QF gain with
fewer mutable nodes.  It is a search-scope probe, not a default operator.
"""

from __future__ import annotations

import argparse
import json
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
from evaluate_leiden_basin_polish_prefixes import select_prefix_rows  # noqa: E402
from evaluate_leiden_basin_target_elbow_polish import (  # noqa: E402
    _rank_and_filter_prefix_rows,
)
from probe_leiden_basin_post_gate_recovery_moves import (  # noqa: E402
    DEFAULT_CANDIDATE_DIRS,
    DEFAULT_POST_GATE_DIR,
    DEFAULT_PROFILE_BATCH_DIR,
    DEFAULT_VANILLA_DIR,
    POST_GATE_PATH_SUMMARY_FILENAME,
    _load_case_context,
    _markdown_table,
    _prefix_context,
    _replay_to_source_state,
    _select_source_path,
)
from probe_leiden_basin_post_gate_recovery_subsets import (  # noqa: E402
    DEFAULT_SOURCE_MOVE_DIR,
    SOURCE_MOVE_ROWS_FILENAME,
    _load_source_config,
    _rank_selected_nodes,
    _select_source_move,
)
from sciscape.clustering.leiden_basin_profile import parse_node_ids  # noqa: E402
from sciscape.clustering.leiden_basin_search import (  # noqa: E402
    ACTION_RECOVERY_VANILLA_CONTEXT_TOPK,
    POST_GATE_RECOVERY_MOVE_Q_GAIN,
    POST_GATE_RECOVERY_MOVE_RECOVERED,
    POST_GATE_VERDICT_NEAR_MISS,
    TransitionAction,
    annotate_pathway_debt_area_rows,
    annotate_post_gate_recovery_step_rows,
    annotate_tunneling_evidence_rows,
    classify_post_gate_recovery_move_rows,
    compute_pathway_wall_rows,
    edge_public_row,
    node_csv,
    summarize_post_gate_recovery_paths,
    trace_tunneling_path_states,
    unique_sorted_u32,
)
from search_leiden_basin_transitions import (  # noqa: E402
    _evaluate_state,
    _polished_child,
)

DEFAULT_OUTPUT_DIR = DEFAULT_SOURCE_MOVE_DIR.parent / (
    "basin_transition_post_gate_sufficient_subset_field34_cc_c0_p8_v0"
)
DEFAULT_SOURCE_RECOVERY_POLICY = "vanilla_closure_topk:context_only"

GROUP_POLICY_VANILLA_LABEL = "vanilla_label"
GROUP_POLICY_BASELINE_VANILLA_LABEL = "baseline_vanilla_label"
GROUP_POLICY_CANDIDATE_VANILLA_LABEL = "candidate_vanilla_label"
GROUP_POLICY_SELECTED_COMPONENT = "selected_component"
GROUP_POLICY_VANILLA_LABEL_COMPONENT = "vanilla_label_component"
GROUP_POLICY_VANILLA_LABEL_PULL_BAND = "vanilla_label_pull_band"
GROUP_POLICIES = (
    GROUP_POLICY_VANILLA_LABEL,
    GROUP_POLICY_BASELINE_VANILLA_LABEL,
    GROUP_POLICY_CANDIDATE_VANILLA_LABEL,
    GROUP_POLICY_SELECTED_COMPONENT,
    GROUP_POLICY_VANILLA_LABEL_COMPONENT,
    GROUP_POLICY_VANILLA_LABEL_PULL_BAND,
)

STATE_ROWS_FILENAME = "post_gate_sufficient_subset_states.csv"
EDGE_ROWS_FILENAME = "post_gate_sufficient_subset_edges.csv"
GROUP_ROWS_FILENAME = "post_gate_sufficient_subset_groups.csv"
TRIAL_ROWS_FILENAME = "post_gate_sufficient_subset_trial_rows.csv"
COMMIT_ROWS_FILENAME = "post_gate_sufficient_subset_commit_rows.csv"
PATH_ROWS_FILENAME = "post_gate_sufficient_subset_path_rows.csv"
TRACE_ROWS_FILENAME = "post_gate_sufficient_subset_trace_rows.csv"
STEP_ROWS_FILENAME = "post_gate_sufficient_subset_step_rows.csv"
SUMMARY_ROWS_FILENAME = "post_gate_sufficient_subset_path_summary_rows.csv"
SUMMARY_FILENAME = "post_gate_sufficient_subset_summary.json"
CONFIG_FILENAME = "post_gate_sufficient_subset_config.json"
REPORT_FILENAME = "post_gate_sufficient_subset_report.md"

def _component_ids(
    *,
    selected_nodes: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    node_count: int,
    labels: np.ndarray | None = None,
) -> dict[int, int]:
    nodes = unique_sorted_u32(selected_nodes)
    index_by_node = {int(node): idx for idx, node in enumerate(nodes)}
    parent = list(range(int(nodes.size)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    selected_mask = np.zeros(int(node_count), dtype=np.bool_)
    selected_mask[nodes.astype(np.int64)] = True
    src_arr = np.asarray(src, dtype=np.int64)
    dst_arr = np.asarray(dst, dtype=np.int64)
    keep = selected_mask[src_arr] & selected_mask[dst_arr]
    if labels is not None:
        label_arr = np.asarray(labels)
        keep &= label_arr[src_arr] == label_arr[dst_arr]
    for left, right in zip(src_arr[keep], dst_arr[keep], strict=False):
        union(index_by_node[int(left)], index_by_node[int(right)])
    root_to_component: dict[int, int] = {}
    component_by_node: dict[int, int] = {}
    for node in nodes:
        root = find(index_by_node[int(node)])
        if root not in root_to_component:
            root_to_component[root] = len(root_to_component) + 1
        component_by_node[int(node)] = root_to_component[root]
    return component_by_node

def _build_group_rows(
    *,
    selected_nodes: np.ndarray,
    ranked_nodes: pd.DataFrame,
    baseline_membership: np.ndarray,
    candidate_membership: np.ndarray,
    vanilla_membership: np.ndarray,
    arrays: Any,
    node_count: int,
    group_policy: str,
    pull_band_size: int = 16,
) -> pd.DataFrame:
    if group_policy not in GROUP_POLICIES:
        raise ValueError(f"Unsupported group policy: {group_policy}")
    nodes = unique_sorted_u32(selected_nodes)
    pull_by_node = dict(
        zip(
            ranked_nodes["node"].astype(int),
            ranked_nodes["pull_score"].astype(float),
            strict=False,
        )
    )
    rank_by_node = dict(
        zip(
            ranked_nodes["node"].astype(int),
            ranked_nodes["pull_rank"].astype(int),
            strict=False,
        )
    )
    baseline = np.asarray(baseline_membership)
    candidate = np.asarray(candidate_membership)
    vanilla = np.asarray(vanilla_membership)
    component_by_node: dict[int, int] = {}
    if group_policy == GROUP_POLICY_SELECTED_COMPONENT:
        component_by_node = _component_ids(
            selected_nodes=nodes,
            src=np.asarray(arrays.src, dtype=np.uint32),
            dst=np.asarray(arrays.dst, dtype=np.uint32),
            node_count=node_count,
        )
    elif group_policy == GROUP_POLICY_VANILLA_LABEL_COMPONENT:
        component_by_node = _component_ids(
            selected_nodes=nodes,
            src=np.asarray(arrays.src, dtype=np.uint32),
            dst=np.asarray(arrays.dst, dtype=np.uint32),
            node_count=node_count,
            labels=vanilla,
        )
    band_by_node: dict[int, int] = {}
    if group_policy == GROUP_POLICY_VANILLA_LABEL_PULL_BAND:
        ranked = ranked_nodes.copy()
        ranked["vanilla_label"] = vanilla[ranked["node"].to_numpy(dtype=np.int64)]
        ranked = ranked.sort_values(
            ["vanilla_label", "pull_score", "node"],
            ascending=[True, False, True],
        )
        band_size = max(1, int(pull_band_size))
        ranked["label_rank_index"] = ranked.groupby("vanilla_label").cumcount()
        ranked["label_pull_band"] = (
            ranked["label_rank_index"].astype(int) // band_size
        ) + 1
        band_by_node = dict(
            zip(
                ranked["node"].astype(int),
                ranked["label_pull_band"].astype(int),
                strict=False,
            )
        )

    rows: list[dict[str, Any]] = []
    for node in nodes:
        node_i = int(node)
        base_label = int(baseline[node_i])
        cand_label = int(candidate[node_i])
        van_label = int(vanilla[node_i])
        if group_policy == GROUP_POLICY_VANILLA_LABEL:
            key = (van_label,)
        elif group_policy == GROUP_POLICY_BASELINE_VANILLA_LABEL:
            key = (base_label, van_label)
        elif group_policy == GROUP_POLICY_CANDIDATE_VANILLA_LABEL:
            key = (cand_label, van_label)
        elif group_policy == GROUP_POLICY_SELECTED_COMPONENT:
            key = (component_by_node[node_i],)
        elif group_policy == GROUP_POLICY_VANILLA_LABEL_COMPONENT:
            key = (van_label, component_by_node[node_i])
        else:
            key = (van_label, band_by_node[node_i])
        rows.append(
            {
                "node": node_i,
                "group_key": "|".join(str(value) for value in key),
                "baseline_label": base_label,
                "candidate_label": cand_label,
                "vanilla_label": van_label,
                "component_id": int(component_by_node.get(node_i, 0)),
                "pull_score": float(pull_by_node.get(node_i, 0.0)),
                "pull_rank": int(rank_by_node.get(node_i, 0)),
            }
        )
    frame = pd.DataFrame(rows)
    out: list[dict[str, Any]] = []
    for index, (group_key, group) in enumerate(
        frame.groupby("group_key", sort=True),
        start=1,
    ):
        group_nodes = np.asarray(sorted(group["node"].astype(int)), dtype=np.uint32)
        out.append(
            {
                "group_id": f"{group_policy}:{index:04d}",
                "group_policy": group_policy,
                "group_key": str(group_key),
                "node_count": int(group_nodes.size),
                "node_ids": node_csv(group_nodes),
                "pull_sum": float(group["pull_score"].sum()),
                "pull_mean": float(group["pull_score"].mean()),
                "pull_max": float(group["pull_score"].max()),
                "min_pull_rank": int(group["pull_rank"].min()),
                "max_pull_rank": int(group["pull_rank"].max()),
                "baseline_label_count": int(group["baseline_label"].nunique()),
                "candidate_label_count": int(group["candidate_label"].nunique()),
                "vanilla_label_count": int(group["vanilla_label"].nunique()),
                "component_count": int(
                    group["component_id"].replace(0, np.nan).nunique()
                ),
            }
        )
    groups = pd.DataFrame(out)
    if groups.empty:
        return groups
    return groups.sort_values(
        ["node_count", "pull_sum", "pull_max", "group_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

def _sufficient_verdict(
    row: pd.Series,
    *,
    full_gain: float,
    source_support: float,
    source_progress: float,
    retain_full_gain_fraction: float,
    min_q_gain: float,
    support_tolerance: float,
    progress_tolerance: float,
) -> tuple[bool, str]:
    q_gain = float(row["post_gate_move_delta_q_gain"])
    support = float(row["state_support_distance_to_vanilla"])
    progress = float(row["state_target_progress_from_vanilla"])
    min_required_gain = max(float(min_q_gain), float(full_gain) * retain_full_gain_fraction)
    if q_gain + 1e-12 < min_required_gain:
        return False, "below_q_gain_floor"
    if support + float(support_tolerance) < float(source_support):
        return False, "support_not_retained"
    if progress + float(progress_tolerance) < float(source_progress):
        return False, "progress_not_retained"
    return True, "accepted"

def _evaluate_trial(
    *,
    selected_nodes: np.ndarray,
    source_state: Any,
    source_row: pd.Series,
    prefix_row: pd.Series,
    case_ctx: dict[str, Any],
    source_recovery_policy: str,
    group_policy: str,
    round_index: int,
    trial_index: int,
    removed_group_ids: tuple[str, ...],
    candidate_removed_group_id: str,
    candidate_removed_node_count: int,
    recovery_seed: int,
    recovery_polish_iterations: int,
    resolution: float,
    randomness: float,
    min_support_shift_from_vanilla: float,
    min_material_q_gain: float,
    support_gate: float,
    progress_margin: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = unique_sorted_u32(selected_nodes)
    action = TransitionAction(
        action_type=ACTION_RECOVERY_VANILLA_CONTEXT_TOPK,
        action_params=(
            f"source_recovery_policy={source_recovery_policy};"
            f"group_policy={group_policy};"
            f"round={int(round_index)};"
            f"trial={int(trial_index)};"
            f"selected_k={int(selected.size)};"
            f"removed_group={candidate_removed_group_id}"
        ),
        context_nodes=selected,
        action_nodes=None,
    )
    child = _polished_child(
        parent=source_state,
        action=action,
        graph=case_ctx["graph"],
        donor_membership=case_ctx["candidate"].recreated.membership,
        resolution=resolution,
        seed=int(recovery_seed),
        n_iterations=recovery_polish_iterations,
        randomness=randomness,
        child_index=trial_index,
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
            "path_policy": "post_gate_sufficient_subset",
            "selection_policy": source_recovery_policy,
            "escalation_reason": "sufficient_subset_group_drop",
            "target_stage_index": int(source_row.get("target_stage_index", 0)),
            "selected_k": int(selected.size),
            "selected_node_ids": node_csv(selected),
            "recovery_policy": source_recovery_policy,
            "recovery_source_action_type": "vanilla_closure_topk",
            "recovery_move_kind": "context_only",
            "recovery_selected_node_count": int(selected.size),
            "recovery_context_node_count": int(selected.size),
            "recovery_action_node_count": 0,
            "recovery_source_state_id": source_state.state_id,
            "group_policy": group_policy,
            "sufficient_round_index": int(round_index),
            "sufficient_trial_index": int(trial_index),
            "candidate_removed_group_id": candidate_removed_group_id,
            "candidate_removed_node_count": int(candidate_removed_node_count),
            "removed_group_ids": ",".join(removed_group_ids),
            "removed_group_count": int(len(removed_group_ids)),
        },
        parent_row=source_row,
        min_support_shift_from_vanilla=min_support_shift_from_vanilla,
        min_material_q_gain=min_material_q_gain,
    )
    row["path_elapsed_sec"] = float(source_row.get("path_elapsed_sec", 0.0)) + float(
        child.elapsed_sec
    )
    classified = classify_post_gate_recovery_move_rows(
        pd.DataFrame([row]),
        target_delta_q=float(source_row["state_delta_q_vs_start"]),
        target_support=float(source_row["state_support_distance_to_vanilla"]),
        target_progress=float(source_row["state_target_progress_from_vanilla"]),
        support_gate=support_gate,
        progress_margin=progress_margin,
    )
    row = classified.iloc[0].to_dict()
    edge = edge_public_row(
        parent_state_id=source_state.state_id,
        child_state_id=child.state_id,
        action=action,
        context={
            **case_ctx["public_context"],
            "path_policy": "post_gate_sufficient_subset",
            "recovery_policy": source_recovery_policy,
            "group_policy": group_policy,
            "round": int(round_index),
            "trial": int(trial_index),
        },
    )
    return row, edge

def _write_report(
    path: Path,
    *,
    summary: dict[str, Any],
    group_rows: pd.DataFrame,
    trial_rows: pd.DataFrame,
    commit_rows: pd.DataFrame,
) -> None:
    lines = [
        "# Post-Gate Sufficient Subset Probe",
        "",
        "This artifact starts from the full p8 vanilla-closure context release and",
        "greedily removes structured groups while requiring QF gain, support, and",
        "target progress retention.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "output_dir",
        "group_policy",
        "source_pair_id",
        "source_prefix_rank",
        "source_full_selected_node_count",
        "full_q_gain",
        "retain_full_gain_fraction",
        "min_required_q_gain",
        "group_rows",
        "trial_rows",
        "accepted_trial_rows",
        "committed_rounds",
        "final_selected_node_count",
        "final_scope_fraction",
        "final_q_gain",
        "final_q_gain_retention_fraction",
        "final_support",
        "final_progress",
        "stop_reason",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")
    lines.extend(["", "## Group Summary", ""])
    group_cols = [
        "group_id",
        "group_key",
        "node_count",
        "pull_sum",
        "pull_max",
        "min_pull_rank",
        "max_pull_rank",
        "baseline_label_count",
        "candidate_label_count",
        "vanilla_label_count",
        "component_count",
    ]
    lines.extend(
        _markdown_table(
            group_rows[[column for column in group_cols if column in group_rows]].head(40),
            max_rows=40,
        )
    )
    lines.extend(["", "## Committed Removals", ""])
    commit_cols = [
        "sufficient_round_index",
        "candidate_removed_group_id",
        "candidate_removed_node_count",
        "recovery_selected_node_count",
        "sufficient_accepted",
        "post_gate_move_delta_q_gain",
        "q_gain_retention_fraction",
        "state_delta_q_vs_start",
        "state_support_distance_to_vanilla",
        "state_target_progress_from_vanilla",
        "mutable_node_count",
    ]
    lines.extend(
        _markdown_table(
            commit_rows[[column for column in commit_cols if column in commit_rows]],
            max_rows=80,
        )
    )
    lines.extend(["", "## Best Accepted Trials", ""])
    if trial_rows.empty or "sufficient_accepted" not in trial_rows:
        accepted = pd.DataFrame()
    else:
        accepted = trial_rows[trial_rows["sufficient_accepted"].astype(bool)]
    if not accepted.empty:
        accepted = accepted.sort_values(
            [
                "recovery_selected_node_count",
                "post_gate_move_delta_q_gain",
                "state_support_distance_to_vanilla",
            ],
            ascending=[True, False, False],
        ).head(20)
    trial_cols = [
        "sufficient_round_index",
        "candidate_removed_group_id",
        "candidate_removed_node_count",
        "recovery_selected_node_count",
        "sufficient_rejection_reason",
        "post_gate_move_delta_q_gain",
        "q_gain_retention_fraction",
        "state_support_distance_to_vanilla",
        "state_target_progress_from_vanilla",
        "mutable_node_count",
    ]
    lines.extend(
        _markdown_table(
            accepted[[column for column in trial_cols if column in accepted]],
            max_rows=20,
        )
    )
    lines.extend(["", "## Reading", ""])
    lines.extend(
        [
            "- If no group can be removed, this group policy is too coarse or the context is tightly coupled.",
            "- If many groups can be removed, the final row is a candidate narrowed search scope.",
            "- A narrowed scope is still diagnostic until it passes seed controls and material cost gates.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_probe(
    *,
    source_move_dir: Path,
    post_gate_dir: Path,
    prefix_dir: Path,
    profile_batch_dir: Path,
    output_dir: Path,
    candidate_dirs: tuple[Path, ...],
    vanilla_dir: Path,
    pair_id: str,
    prefix_rank: int,
    source_verdict: str,
    source_recovery_policy: str,
    group_policy: str,
    pull_band_size: int,
    max_groups: int,
    min_group_size: int,
    max_rounds: int,
    retain_full_gain_fraction: float,
    min_q_gain: float,
    support_tolerance: float,
    progress_tolerance: float,
    baseline_iterations: int,
    candidate_polish_iterations: int,
    local_polish_iterations: int,
    recovery_polish_iterations: int,
    target_action_multiplier: float,
    max_target_action_nodes: int,
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
    source_moves = pd.read_csv(source_move_dir / SOURCE_MOVE_ROWS_FILENAME)
    source_move, source_recovery_index = _select_source_move(
        source_moves,
        recovery_policy=source_recovery_policy,
    )
    full_selected_nodes = parse_node_ids(source_move["selected_node_ids"])
    if full_selected_nodes.size == 0:
        raise ValueError("Source recovery move does not contain selected_node_ids")

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
    expected_source_id = str(source_move.get("recovery_source_state_id", ""))
    if expected_source_id and expected_source_id != source_state.state_id:
        raise RuntimeError(
            "Replayed source state does not match source move artifact: "
            f"{source_state.state_id} != {expected_source_id}"
        )

    ranked_nodes = _rank_selected_nodes(
        selected_nodes=full_selected_nodes,
        source_state=source_state,
        arrays=case_ctx["arrays"],
        node_count=int(case_ctx["baseline"].membership.size),
    )
    group_rows = _build_group_rows(
        selected_nodes=full_selected_nodes,
        ranked_nodes=ranked_nodes,
        baseline_membership=case_ctx["baseline"].membership,
        candidate_membership=case_ctx["candidate"].recreated.membership,
        vanilla_membership=case_ctx["vanilla"].membership,
        arrays=case_ctx["arrays"],
        node_count=int(case_ctx["baseline"].membership.size),
        group_policy=group_policy,
        pull_band_size=pull_band_size,
    )
    group_rows = group_rows[group_rows["node_count"].astype(int) >= int(min_group_size)]
    if int(max_groups) > 0:
        group_rows = group_rows.head(int(max_groups)).copy()
    group_rows = group_rows.reset_index(drop=True)
    if group_rows.empty:
        raise ValueError(f"No groups selected for group_policy={group_policy}")

    node_by_group = {
        str(row["group_id"]): set(int(node) for node in parse_node_ids(row["node_ids"]))
        for _, row in group_rows.iterrows()
    }
    full_gain = float(source_move["post_gate_move_delta_q_gain"])
    source_support = float(source_row["state_support_distance_to_vanilla"])
    source_progress = float(source_row["state_target_progress_from_vanilla"])
    min_required_q_gain = max(float(min_q_gain), full_gain * retain_full_gain_fraction)
    recovery_seed = int(recovery_seed_offset) + int(source_recovery_index)

    current_nodes = set(int(node) for node in unique_sorted_u32(full_selected_nodes))
    removed_group_ids: list[str] = []
    trial_rows: list[dict[str, Any]] = []
    trial_edges: list[dict[str, Any]] = []
    commit_rows: list[dict[str, Any]] = []
    trial_index = 0
    stop_reason = "max_rounds"
    for round_index in range(1, int(max_rounds) + 1):
        accepted_this_round: list[dict[str, Any]] = []
        remaining_groups = [
            str(group_id)
            for group_id in group_rows["group_id"]
            if str(group_id) not in removed_group_ids
        ]
        for group_id in remaining_groups:
            group_nodes = node_by_group[group_id]
            trial_nodes = current_nodes.difference(group_nodes)
            if len(trial_nodes) == len(current_nodes):
                continue
            if not trial_nodes:
                continue
            trial_index += 1
            row, edge = _evaluate_trial(
                selected_nodes=np.asarray(sorted(trial_nodes), dtype=np.uint32),
                source_state=source_state,
                source_row=source_row,
                prefix_row=prefix_row,
                case_ctx=case_ctx,
                source_recovery_policy=source_recovery_policy,
                group_policy=group_policy,
                round_index=round_index,
                trial_index=trial_index,
                removed_group_ids=tuple([*removed_group_ids, group_id]),
                candidate_removed_group_id=group_id,
                candidate_removed_node_count=len(group_nodes),
                recovery_seed=recovery_seed,
                recovery_polish_iterations=recovery_polish_iterations,
                resolution=resolution,
                randomness=randomness,
                min_support_shift_from_vanilla=min_support_shift_from_vanilla,
                min_material_q_gain=min_material_q_gain,
                support_gate=support_gate,
                progress_margin=progress_margin,
            )
            accepted, reason = _sufficient_verdict(
                pd.Series(row),
                full_gain=full_gain,
                source_support=source_support,
                source_progress=source_progress,
                retain_full_gain_fraction=retain_full_gain_fraction,
                min_q_gain=min_q_gain,
                support_tolerance=support_tolerance,
                progress_tolerance=progress_tolerance,
            )
            row["sufficient_accepted"] = bool(accepted)
            row["sufficient_rejection_reason"] = reason
            row["full_q_gain"] = float(full_gain)
            row["min_required_q_gain"] = float(min_required_q_gain)
            row["q_gain_retention_fraction"] = (
                float(row["post_gate_move_delta_q_gain"]) / full_gain
                if full_gain > 0
                else 0.0
            )
            row["source_full_selected_node_count"] = int(full_selected_nodes.size)
            row["selected_scope_fraction"] = float(len(trial_nodes) / full_selected_nodes.size)
            trial_rows.append(row)
            trial_edges.append(edge)
            if accepted:
                accepted_this_round.append(row)
        if not accepted_this_round:
            stop_reason = "no_accepted_group_removal"
            break
        accepted_frame = pd.DataFrame(accepted_this_round).sort_values(
            [
                "recovery_selected_node_count",
                "post_gate_move_delta_q_gain",
                "state_support_distance_to_vanilla",
                "state_target_progress_from_vanilla",
                "candidate_removed_node_count",
            ],
            ascending=[True, False, False, False, False],
        )
        chosen = accepted_frame.iloc[0].to_dict()
        chosen_group = str(chosen["candidate_removed_group_id"])
        removed_group_ids.append(chosen_group)
        current_nodes = current_nodes.difference(node_by_group[chosen_group])
        commit_rows.append(chosen)
    else:
        stop_reason = "max_rounds"

    trials = pd.DataFrame(trial_rows)
    commits = pd.DataFrame(commit_rows)
    state_rows = pd.concat([replay_rows, trials], ignore_index=True)
    edge_rows = pd.concat([replay_edges, pd.DataFrame(trial_edges)], ignore_index=True)
    path_rows = compute_pathway_wall_rows(
        state_rows,
        source_label="post_gate_sufficient_subset_v0",
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
    trial_ids = set(trials["state_id"].astype(str)) if not trials.empty else set()
    trial_path_rows = path_rows[
        path_rows["path_final_state_id"].astype(str).isin(trial_ids)
    ].copy()
    trace_rows = trace_tunneling_path_states(
        trial_path_rows,
        state_rows=state_rows,
        support_gate=support_gate,
        progress_margin=progress_margin,
    )
    step_rows = annotate_post_gate_recovery_step_rows(trace_rows)
    summary_rows = summarize_post_gate_recovery_paths(trace_rows)

    group_rows.to_csv(output_dir / GROUP_ROWS_FILENAME, index=False)
    state_rows.to_csv(output_dir / STATE_ROWS_FILENAME, index=False)
    edge_rows.to_csv(output_dir / EDGE_ROWS_FILENAME, index=False)
    trials.to_csv(output_dir / TRIAL_ROWS_FILENAME, index=False)
    commits.to_csv(output_dir / COMMIT_ROWS_FILENAME, index=False)
    trial_path_rows.to_csv(output_dir / PATH_ROWS_FILENAME, index=False)
    trace_rows.to_csv(output_dir / TRACE_ROWS_FILENAME, index=False)
    step_rows.to_csv(output_dir / STEP_ROWS_FILENAME, index=False)
    summary_rows.to_csv(output_dir / SUMMARY_ROWS_FILENAME, index=False)

    final_row = commits.iloc[-1].to_dict() if not commits.empty else source_move.to_dict()
    final_selected_node_count = (
        int(final_row.get("recovery_selected_node_count", full_selected_nodes.size))
        if commits.empty
        else int(commits.iloc[-1]["recovery_selected_node_count"])
    )
    final_q_gain = (
        float(final_row.get("post_gate_move_delta_q_gain", full_gain))
        if commits.empty
        else float(commits.iloc[-1]["post_gate_move_delta_q_gain"])
    )
    accepted_count = (
        int(trials["sufficient_accepted"].astype(bool).sum()) if not trials.empty else 0
    )
    verdict_counts = (
        trials["post_gate_move_verdict"].astype(str).value_counts().to_dict()
        if not trials.empty
        else {}
    )
    config = {
        "source_move_dir": str(source_move_dir),
        "post_gate_dir": str(post_gate_dir),
        "prefix_dir": str(prefix_dir),
        "profile_batch_dir": str(profile_batch_dir),
        "output_dir": str(output_dir),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "vanilla_dir": str(vanilla_dir),
        "pair_id": pair_id,
        "prefix_rank": int(prefix_rank),
        "source_verdict": source_verdict,
        "source_recovery_policy": source_recovery_policy,
        "group_policy": group_policy,
        "pull_band_size": int(pull_band_size),
        "max_groups": int(max_groups),
        "min_group_size": int(min_group_size),
        "max_rounds": int(max_rounds),
        "retain_full_gain_fraction": float(retain_full_gain_fraction),
        "min_q_gain": float(min_q_gain),
        "support_tolerance": float(support_tolerance),
        "progress_tolerance": float(progress_tolerance),
        "baseline_iterations": int(baseline_iterations),
        "candidate_polish_iterations": int(candidate_polish_iterations),
        "local_polish_iterations": int(local_polish_iterations),
        "recovery_polish_iterations": int(recovery_polish_iterations),
        "target_action_multiplier": float(target_action_multiplier),
        "max_target_action_nodes": int(max_target_action_nodes),
        "resolution": float(resolution),
        "randomness": float(randomness),
        "support_gate": float(support_gate),
        "progress_margin": float(progress_margin),
        "recovery_seed": int(recovery_seed),
    }
    summary = {
        "schema": "leiden_basin_post_gate_sufficient_subset_probe.v0",
        **config,
        "source_pair_id": pair_id,
        "source_prefix_rank": int(prefix_rank),
        "source_state_id": source_state.state_id,
        "source_full_selected_node_count": int(full_selected_nodes.size),
        "source_delta_q": float(source_row["state_delta_q_vs_start"]),
        "source_support": source_support,
        "source_target_progress": source_progress,
        "full_q_gain": full_gain,
        "min_required_q_gain": min_required_q_gain,
        "group_rows": int(len(group_rows)),
        "trial_rows": int(len(trials)),
        "accepted_trial_rows": accepted_count,
        "committed_rounds": int(len(commits)),
        "final_selected_node_count": final_selected_node_count,
        "final_scope_fraction": float(final_selected_node_count / full_selected_nodes.size),
        "final_q_gain": final_q_gain,
        "final_q_gain_retention_fraction": (
            float(final_q_gain / full_gain) if full_gain > 0 else 0.0
        ),
        "final_delta_q": float(
            final_row.get("state_delta_q_vs_start", source_move["state_delta_q_vs_start"])
        ),
        "final_support": float(
            final_row.get(
                "state_support_distance_to_vanilla",
                source_move["state_support_distance_to_vanilla"],
            )
        ),
        "final_progress": float(
            final_row.get(
                "state_target_progress_from_vanilla",
                source_move["state_target_progress_from_vanilla"],
            )
        ),
        "stop_reason": stop_reason,
        "verdict_counts": verdict_counts,
        "q_recovered_support_retained_rows": int(
            verdict_counts.get(POST_GATE_RECOVERY_MOVE_RECOVERED, 0)
        ),
        "q_gain_support_retained_rows": int(
            verdict_counts.get(POST_GATE_RECOVERY_MOVE_Q_GAIN, 0)
        ),
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir / REPORT_FILENAME,
        summary=summary,
        group_rows=group_rows,
        trial_rows=trials,
        commit_rows=commits,
    )
    return summary

def build_parser() -> argparse.ArgumentParser:
    source_config = _load_source_config(DEFAULT_SOURCE_MOVE_DIR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-move-dir", type=Path, default=DEFAULT_SOURCE_MOVE_DIR)
    parser.add_argument(
        "--post-gate-dir",
        type=Path,
        default=Path(source_config.get("post_gate_dir", DEFAULT_POST_GATE_DIR)),
    )
    parser.add_argument(
        "--prefix-dir",
        type=Path,
        default=Path(source_config.get("prefix_dir", DEFAULT_PREFIX_DIR)),
    )
    parser.add_argument(
        "--profile-batch-dir",
        type=Path,
        default=Path(source_config.get("profile_batch_dir", DEFAULT_PROFILE_BATCH_DIR)),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-dir", type=Path, action="append", default=None)
    parser.add_argument(
        "--vanilla-dir",
        type=Path,
        default=Path(source_config.get("vanilla_dir", DEFAULT_VANILLA_DIR)),
    )
    parser.add_argument("--pair-id", default=str(source_config.get("pair_id", "c0-s11-r0.001")))
    parser.add_argument("--prefix-rank", type=int, default=int(source_config.get("prefix_rank", 8)))
    parser.add_argument(
        "--source-verdict",
        default=str(source_config.get("source_verdict", POST_GATE_VERDICT_NEAR_MISS)),
    )
    parser.add_argument("--source-recovery-policy", default=DEFAULT_SOURCE_RECOVERY_POLICY)
    parser.add_argument("--group-policy", default=GROUP_POLICY_VANILLA_LABEL)
    parser.add_argument("--pull-band-size", type=int, default=16)
    parser.add_argument("--max-groups", type=int, default=80)
    parser.add_argument("--min-group-size", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--retain-full-gain-fraction", type=float, default=0.70)
    parser.add_argument("--min-q-gain", type=float, default=0.0)
    parser.add_argument("--support-tolerance", type=float, default=1e-12)
    parser.add_argument("--progress-tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--baseline-iterations",
        type=int,
        default=int(source_config.get("baseline_iterations", 10)),
    )
    parser.add_argument(
        "--candidate-polish-iterations",
        type=int,
        default=int(source_config.get("candidate_polish_iterations", 5)),
    )
    parser.add_argument(
        "--local-polish-iterations",
        type=int,
        default=int(source_config.get("local_polish_iterations", 3)),
    )
    parser.add_argument(
        "--recovery-polish-iterations",
        type=int,
        default=int(source_config.get("recovery_polish_iterations", 10)),
    )
    parser.add_argument(
        "--target-action-multiplier",
        type=float,
        default=float(source_config.get("target_action_multiplier", 0.5)),
    )
    parser.add_argument(
        "--max-target-action-nodes",
        type=int,
        default=int(source_config.get("max_target_action_nodes", 64)),
    )
    parser.add_argument("--cumulative-fraction", type=float, default=0.80)
    parser.add_argument("--min-score-fraction", type=float, default=0.05)
    parser.add_argument("--min-gap-fraction", type=float, default=0.25)
    parser.add_argument("--min-guarded-pull-fraction", type=float, default=0.50)
    parser.add_argument(
        "--resolution",
        type=float,
        default=float(source_config.get("resolution", 0.01)),
    )
    parser.add_argument(
        "--randomness",
        type=float,
        default=float(source_config.get("randomness", 0.01)),
    )
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    parser.add_argument("--polish-seed-offset", type=int, default=11000)
    parser.add_argument("--recovery-seed-offset", type=int, default=21000)
    parser.add_argument("--min-support-shift-from-vanilla", type=float, default=0.05)
    parser.add_argument("--min-material-q-gain", type=float, default=0.0)
    parser.add_argument(
        "--support-gate",
        type=float,
        default=float(source_config.get("support_gate", 0.05)),
    )
    parser.add_argument(
        "--progress-margin",
        type=float,
        default=float(source_config.get("progress_margin", 0.005)),
    )
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_config = _load_source_config(args.source_move_dir)
    candidate_dirs = (
        tuple(args.candidate_dir)
        if args.candidate_dir
        else tuple(
            Path(path)
            for path in source_config.get("candidate_dirs", DEFAULT_CANDIDATE_DIRS)
        )
    )
    summary = run_probe(
        source_move_dir=args.source_move_dir,
        post_gate_dir=args.post_gate_dir,
        prefix_dir=args.prefix_dir,
        profile_batch_dir=args.profile_batch_dir,
        output_dir=args.output_dir,
        candidate_dirs=candidate_dirs,
        vanilla_dir=args.vanilla_dir,
        pair_id=args.pair_id,
        prefix_rank=args.prefix_rank,
        source_verdict=args.source_verdict,
        source_recovery_policy=args.source_recovery_policy,
        group_policy=args.group_policy,
        pull_band_size=args.pull_band_size,
        max_groups=args.max_groups,
        min_group_size=args.min_group_size,
        max_rounds=args.max_rounds,
        retain_full_gain_fraction=args.retain_full_gain_fraction,
        min_q_gain=args.min_q_gain,
        support_tolerance=args.support_tolerance,
        progress_tolerance=args.progress_tolerance,
        baseline_iterations=args.baseline_iterations,
        candidate_polish_iterations=args.candidate_polish_iterations,
        local_polish_iterations=args.local_polish_iterations,
        recovery_polish_iterations=args.recovery_polish_iterations,
        target_action_multiplier=args.target_action_multiplier,
        max_target_action_nodes=args.max_target_action_nodes,
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
