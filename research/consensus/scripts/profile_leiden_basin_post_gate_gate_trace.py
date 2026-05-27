#!/usr/bin/env python3
"""Trace membership changes behind a post-gate sufficient context gate."""

from __future__ import annotations

import argparse
import json
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
from sciscape.clustering.leiden_basin_profile import (  # noqa: E402
    changed_support_nodes,
    endpoint_distance,
    parse_node_ids,
)
from sciscape.clustering.leiden_basin_search import (  # noqa: E402
    ACTION_RECOVERY_VANILLA_CONTEXT_TOPK,
    POST_GATE_VERDICT_NEAR_MISS,
    TransitionAction,
    edge_public_row,
    node_csv,
    unique_sorted_u32,
    weighted_pull_to_nodes,
)
from search_leiden_basin_transitions import (  # noqa: E402
    _evaluate_state,
    _polished_child,
)


DEFAULT_SUFFICIENT_DIR = DEFAULT_SOURCE_MOVE_DIR.parent / (
    "basin_transition_post_gate_sufficient_subset_field34_cc_c0_p8_v1"
)
DEFAULT_OUTPUT_DIR = DEFAULT_SOURCE_MOVE_DIR.parent / (
    "basin_transition_post_gate_gate_trace_field34_cc_c0_p8_v0"
)
DEFAULT_SOURCE_RECOVERY_POLICY = "vanilla_closure_topk:context_only"

SUMMARY_FILENAME = "post_gate_gate_trace_summary.json"
CONFIG_FILENAME = "post_gate_gate_trace_config.json"
NODE_ROWS_FILENAME = "post_gate_gate_trace_node_rows.csv"
REGION_ROWS_FILENAME = "post_gate_gate_trace_region_rows.csv"
LABEL_TRANSITIONS_FILENAME = "post_gate_gate_trace_label_transitions.csv"
EDGE_ROWS_FILENAME = "post_gate_gate_trace_edge_rows.csv"
STATE_ROWS_FILENAME = "post_gate_gate_trace_state_rows.csv"
REPORT_FILENAME = "post_gate_gate_trace_report.md"


def _mode_int(values: pd.Series) -> int | None:
    if values.empty:
        return None
    counts = values.astype(int).value_counts()
    if counts.empty:
        return None
    return int(counts.index[0])


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _best_partner_maps(
    left_membership: np.ndarray,
    right_membership: np.ndarray,
) -> tuple[dict[int, int], dict[int, int]]:
    left = np.asarray(left_membership, dtype=np.uint64)
    right = np.asarray(right_membership, dtype=np.uint64)
    frame = pd.DataFrame({"left": left.astype(np.int64), "right": right.astype(np.int64)})
    counts = frame.groupby(["left", "right"], sort=False).size().reset_index(name="count")
    left_best: dict[int, int] = {}
    right_best: dict[int, int] = {}
    for left_label, group in counts.groupby("left", sort=False):
        best = group.sort_values(["count", "right"], ascending=[False, True]).iloc[0]
        left_best[int(left_label)] = int(best["right"])
    for right_label, group in counts.groupby("right", sort=False):
        best = group.sort_values(["count", "left"], ascending=[False, True]).iloc[0]
        right_best[int(right_label)] = int(best["left"])
    return left_best, right_best


def _selected_gate_nodes(sufficient_dir: Path) -> tuple[np.ndarray, pd.Series]:
    commits = pd.read_csv(sufficient_dir / "post_gate_sufficient_subset_commit_rows.csv")
    if commits.empty:
        raise ValueError(f"No committed sufficient subset rows in {sufficient_dir}")
    final = commits.iloc[-1]
    nodes = parse_node_ids(final["selected_node_ids"])
    if nodes.size == 0:
        raise ValueError("Final sufficient subset row has empty selected_node_ids")
    return unique_sorted_u32(nodes), final


def _edge_change_rows(
    *,
    arrays: Any,
    source_membership: np.ndarray,
    gate_membership: np.ndarray,
    full_membership: np.ndarray,
    focus_nodes: np.ndarray,
) -> pd.DataFrame:
    src = np.asarray(arrays.src, dtype=np.int64)
    dst = np.asarray(arrays.dst, dtype=np.int64)
    weight = np.asarray(arrays.weight, dtype=np.float64)
    focus = unique_sorted_u32(focus_nodes)
    focus_mask = np.zeros(int(source_membership.size), dtype=np.bool_)
    focus_mask[focus.astype(np.int64)] = True
    keep = focus_mask[src] | focus_mask[dst]
    if not np.any(keep):
        return pd.DataFrame()
    source_same = source_membership[src[keep]] == source_membership[dst[keep]]
    gate_same = gate_membership[src[keep]] == gate_membership[dst[keep]]
    full_same = full_membership[src[keep]] == full_membership[dst[keep]]
    categories = {
        "gate_gained_internal": (~source_same) & gate_same,
        "gate_lost_internal": source_same & (~gate_same),
        "gate_unchanged_internal": source_same & gate_same,
        "gate_unchanged_external": (~source_same) & (~gate_same),
        "full_gained_internal": (~source_same) & full_same,
        "full_lost_internal": source_same & (~full_same),
    }
    rows: list[dict[str, Any]] = []
    edge_weights = weight[keep]
    for category, mask in categories.items():
        rows.append(
            {
                "edge_category": category,
                "edge_count": int(np.count_nonzero(mask)),
                "edge_weight_sum": float(edge_weights[mask].sum()),
                "edge_weight_mean": (
                    float(edge_weights[mask].mean()) if np.any(mask) else 0.0
                ),
            }
        )
    rows.append(
        {
            "edge_category": "focus_incident_total",
            "edge_count": int(np.count_nonzero(keep)),
            "edge_weight_sum": float(edge_weights.sum()),
            "edge_weight_mean": float(edge_weights.mean()) if edge_weights.size else 0.0,
        }
    )
    return pd.DataFrame(rows)


def _node_rows(
    *,
    source_state: Any,
    gate_child: Any,
    full_child: Any,
    gate_nodes: np.ndarray,
    full_nodes: np.ndarray,
    ranked_nodes: pd.DataFrame,
    case_ctx: dict[str, Any],
) -> pd.DataFrame:
    source = np.asarray(source_state.membership, dtype=np.uint64)
    gate = np.asarray(gate_child.membership, dtype=np.uint64)
    full = np.asarray(full_child.membership, dtype=np.uint64)
    baseline = np.asarray(case_ctx["baseline"].membership, dtype=np.uint64)
    candidate = np.asarray(case_ctx["candidate"].recreated.membership, dtype=np.uint64)
    vanilla = np.asarray(case_ctx["vanilla"].membership, dtype=np.uint64)
    gate_set = set(int(node) for node in unique_sorted_u32(gate_nodes))
    full_set = set(int(node) for node in unique_sorted_u32(full_nodes))
    source_mutable_set = set(int(node) for node in unique_sorted_u32(source_state.mutable_nodes))
    source_action_set = set(int(node) for node in unique_sorted_u32(source_state.action_nodes))
    target_set = set(int(node) for node in unique_sorted_u32(source_state.target_nodes))
    changed_gate = set(changed_support_nodes(source, gate).astype(int).tolist())
    changed_full = set(changed_support_nodes(source, full).astype(int).tolist())
    source_to_gate_best, gate_to_source_best = _best_partner_maps(source, gate)
    source_to_full_best, full_to_source_best = _best_partner_maps(source, full)
    gate_to_full_best, full_to_gate_best = _best_partner_maps(gate, full)
    focus_nodes = unique_sorted_u32(
        [
            *gate_set,
            *full_set,
            *source_mutable_set,
            *source_action_set,
            *target_set,
            *changed_gate,
            *changed_full,
        ]
    )
    pull_to_action = weighted_pull_to_nodes(
        src=np.asarray(case_ctx["arrays"].src, dtype=np.uint32),
        dst=np.asarray(case_ctx["arrays"].dst, dtype=np.uint32),
        weight=np.asarray(case_ctx["arrays"].weight, dtype=np.float64),
        target_nodes=source_state.action_nodes,
        node_count=int(source.size),
    )
    pull_to_gate = weighted_pull_to_nodes(
        src=np.asarray(case_ctx["arrays"].src, dtype=np.uint32),
        dst=np.asarray(case_ctx["arrays"].dst, dtype=np.uint32),
        weight=np.asarray(case_ctx["arrays"].weight, dtype=np.float64),
        target_nodes=gate_nodes,
        node_count=int(source.size),
    )
    rank_by_node = dict(
        zip(
            ranked_nodes["node"].astype(int),
            ranked_nodes["pull_rank"].astype(int),
            strict=False,
        )
    )
    pull_by_node = dict(
        zip(
            ranked_nodes["node"].astype(int),
            ranked_nodes["pull_score"].astype(float),
            strict=False,
        )
    )
    rows: list[dict[str, Any]] = []
    for node in focus_nodes:
        node_i = int(node)
        source_label = int(source[node_i])
        gate_label = int(gate[node_i])
        full_label = int(full[node_i])
        source_gate_aligned = (
            source_to_gate_best.get(source_label) == gate_label
            and gate_to_source_best.get(gate_label) == source_label
        )
        source_full_aligned = (
            source_to_full_best.get(source_label) == full_label
            and full_to_source_best.get(full_label) == source_label
        )
        gate_full_aligned = (
            gate_to_full_best.get(gate_label) == full_label
            and full_to_gate_best.get(full_label) == gate_label
        )
        rows.append(
            {
                "node": node_i,
                "in_gate_context": node_i in gate_set,
                "in_full_context": node_i in full_set,
                "in_source_mutable": node_i in source_mutable_set,
                "in_source_action": node_i in source_action_set,
                "in_target_nodes": node_i in target_set,
                "moved_source_to_gate": node_i in changed_gate,
                "moved_source_to_full": node_i in changed_full,
                "gate_and_full_same_label": bool(gate[node_i] == full[node_i]),
                "source_gate_aligned_label": bool(source_gate_aligned),
                "source_full_aligned_label": bool(source_full_aligned),
                "gate_full_aligned_label": bool(gate_full_aligned),
                "baseline_label": int(baseline[node_i]),
                "candidate_label": int(candidate[node_i]),
                "vanilla_label": int(vanilla[node_i]),
                "source_label": source_label,
                "gate_child_label": gate_label,
                "full_child_label": full_label,
                "pull_rank_within_full_context": rank_by_node.get(node_i, 0),
                "pull_to_source_action": float(pull_to_action[node_i]),
                "pull_to_gate_context": float(pull_to_gate[node_i]),
                "full_context_pull_score": float(pull_by_node.get(node_i, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _region_rows(node_rows: pd.DataFrame) -> pd.DataFrame:
    region_masks = {
        "gate_context": node_rows["in_gate_context"].astype(bool),
        "full_context_not_gate": node_rows["in_full_context"].astype(bool)
        & ~node_rows["in_gate_context"].astype(bool),
        "source_mutable_not_gate": node_rows["in_source_mutable"].astype(bool)
        & ~node_rows["in_gate_context"].astype(bool),
        "source_action_nodes": node_rows["in_source_action"].astype(bool),
        "target_nodes": node_rows["in_target_nodes"].astype(bool),
        "moved_source_to_gate": node_rows["moved_source_to_gate"].astype(bool),
        "moved_source_to_full": node_rows["moved_source_to_full"].astype(bool),
    }
    rows: list[dict[str, Any]] = []
    for region, mask in region_masks.items():
        subset = node_rows[mask].copy()
        rows.append(
            {
                "region": region,
                "node_count": int(len(subset)),
                "moved_source_to_gate_count": int(
                    subset["moved_source_to_gate"].astype(bool).sum()
                ),
                "moved_source_to_full_count": int(
                    subset["moved_source_to_full"].astype(bool).sum()
                ),
                "source_label_count": int(subset["source_label"].nunique()),
                "gate_child_label_count": int(subset["gate_child_label"].nunique()),
                "full_child_label_count": int(subset["full_child_label"].nunique()),
                "dominant_source_label": _mode_int(subset["source_label"]),
                "dominant_gate_child_label": _mode_int(subset["gate_child_label"]),
                "dominant_full_child_label": _mode_int(subset["full_child_label"]),
                "pull_to_source_action_sum": float(
                    subset["pull_to_source_action"].sum()
                ),
                "pull_to_gate_context_sum": float(subset["pull_to_gate_context"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _label_transition_rows(node_rows: pd.DataFrame) -> pd.DataFrame:
    changed = node_rows[node_rows["moved_source_to_gate"].astype(bool)].copy()
    if changed.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (source_label, child_label), group in changed.groupby(
        ["source_label", "gate_child_label"],
        sort=True,
    ):
        rows.append(
            {
                "source_label": int(source_label),
                "gate_child_label": int(child_label),
                "node_count": int(len(group)),
                "gate_context_count": int(group["in_gate_context"].astype(bool).sum()),
                "source_action_count": int(group["in_source_action"].astype(bool).sum()),
                "target_count": int(group["in_target_nodes"].astype(bool).sum()),
                "dominant_vanilla_label": _mode_int(group["vanilla_label"]),
                "dominant_baseline_label": _mode_int(group["baseline_label"]),
                "dominant_candidate_label": _mode_int(group["candidate_label"]),
                "pull_to_source_action_sum": float(
                    group["pull_to_source_action"].sum()
                ),
                "pull_to_gate_context_sum": float(group["pull_to_gate_context"].sum()),
                "node_ids": node_csv(group["node"].to_numpy(dtype=np.uint32)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["node_count", "pull_to_source_action_sum"],
        ascending=[False, False],
    )


def _write_report(
    path: Path,
    *,
    summary: dict[str, Any],
    state_rows: pd.DataFrame,
    region_rows: pd.DataFrame,
    transition_rows: pd.DataFrame,
    edge_rows: pd.DataFrame,
) -> None:
    lines = [
        "# Post-Gate Gate Trace",
        "",
        "This artifact traces the membership changes caused by the narrowed",
        "209-node source-side vanilla-label gate.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "output_dir",
        "pair_id",
        "prefix_rank",
        "gate_node_count",
        "full_context_node_count",
        "source_mutable_node_count",
        "gate_mutable_node_count",
        "gate_state_delta_q",
        "full_state_delta_q",
        "gate_q_gain",
        "full_q_gain",
        "source_to_gate_changed_nodes",
        "source_to_gate_changed_node_ids",
        "source_to_gate_changed_gate_nodes",
        "source_to_gate_changed_source_action_nodes",
        "source_to_full_changed_nodes",
        "gate_gained_internal_edge_weight",
        "gate_lost_internal_edge_weight",
        "gate_full_endpoint_distance_affected",
        "gate_full_endpoint_distance_sketch",
        "gate_full_aligned_label_focus_fraction",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")
    lines.extend(["", "## State Rows", ""])
    state_cols = [
        "state_name",
        "state_delta_q_vs_start",
        "state_support_distance_to_vanilla",
        "state_target_progress_from_vanilla",
        "mutable_node_count",
        "context_node_count",
        "elapsed_sec",
    ]
    lines.extend(
        _markdown_table(
            state_rows[[column for column in state_cols if column in state_rows]],
            max_rows=20,
        )
    )
    lines.extend(["", "## Regions", ""])
    region_cols = [
        "region",
        "node_count",
        "moved_source_to_gate_count",
        "moved_source_to_full_count",
        "source_label_count",
        "gate_child_label_count",
        "dominant_source_label",
        "dominant_gate_child_label",
        "pull_to_source_action_sum",
        "pull_to_gate_context_sum",
    ]
    lines.extend(
        _markdown_table(
            region_rows[[column for column in region_cols if column in region_rows]],
            max_rows=30,
        )
    )
    lines.extend(["", "## Label Transitions", ""])
    transition_cols = [
        "source_label",
        "gate_child_label",
        "node_count",
        "gate_context_count",
        "source_action_count",
        "target_count",
        "dominant_vanilla_label",
        "dominant_baseline_label",
        "dominant_candidate_label",
        "pull_to_source_action_sum",
    ]
    lines.extend(
        _markdown_table(
            transition_rows[
                [column for column in transition_cols if column in transition_rows]
            ],
            max_rows=30,
        )
    )
    lines.extend(["", "## Edge Incidence", ""])
    lines.extend(_markdown_table(edge_rows, max_rows=20))
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- If moved nodes are mostly outside the gate, the gate enables existing action nodes.",
            "- If moved nodes are mostly inside the gate, the gate itself is the recovered component.",
            "- Edge incidence is descriptive only; CPM also includes resolution penalties.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_trace(
    *,
    sufficient_dir: Path,
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
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_moves = pd.read_csv(source_move_dir / SOURCE_MOVE_ROWS_FILENAME)
    source_move, source_recovery_index = _select_source_move(
        source_moves,
        recovery_policy=source_recovery_policy,
    )
    full_nodes = parse_node_ids(source_move["selected_node_ids"])
    gate_nodes, sufficient_final = _selected_gate_nodes(sufficient_dir)

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
    source_state, source_row, _, _ = _replay_to_source_state(
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
    recovery_seed = int(recovery_seed_offset) + int(source_recovery_index)
    common_context = {
        **case_ctx["public_context"],
        **_prefix_context(prefix_row),
        "path_policy": "post_gate_gate_trace",
        "selection_policy": source_recovery_policy,
        "escalation_reason": "gate_trace",
        "target_stage_index": int(source_row.get("target_stage_index", 0)),
        "recovery_policy": source_recovery_policy,
        "recovery_source_action_type": "vanilla_closure_topk",
        "recovery_move_kind": "context_only",
        "recovery_source_state_id": source_state.state_id,
    }

    def child_for(nodes: np.ndarray, *, name: str, child_index: int) -> tuple[Any, dict[str, Any]]:
        selected = unique_sorted_u32(nodes)
        action = TransitionAction(
            action_type=ACTION_RECOVERY_VANILLA_CONTEXT_TOPK,
            action_params=f"trace_state={name};selected_k={int(selected.size)}",
            context_nodes=selected,
            action_nodes=None,
        )
        child = _polished_child(
            parent=source_state,
            action=action,
            graph=case_ctx["graph"],
            donor_membership=case_ctx["candidate"].recreated.membership,
            resolution=resolution,
            seed=recovery_seed,
            n_iterations=recovery_polish_iterations,
            randomness=randomness,
            child_index=child_index,
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
                **common_context,
                "state_name": name,
                "selected_k": int(selected.size),
                "selected_node_ids": node_csv(selected),
            },
            parent_row=source_row,
            min_support_shift_from_vanilla=min_support_shift_from_vanilla,
            min_material_q_gain=min_material_q_gain,
        )
        return child, row

    gate_child, gate_row = child_for(gate_nodes, name="sufficient_gate", child_index=1)
    full_child, full_row = child_for(full_nodes, name="full_context", child_index=2)
    source_named_row = {**source_row, "state_name": "source_post_gate"}
    state_rows = pd.DataFrame([source_named_row, gate_row, full_row])

    ranked_nodes = _rank_selected_nodes(
        selected_nodes=full_nodes,
        source_state=source_state,
        arrays=case_ctx["arrays"],
        node_count=int(case_ctx["baseline"].membership.size),
    )
    nodes = _node_rows(
        source_state=source_state,
        gate_child=gate_child,
        full_child=full_child,
        gate_nodes=gate_nodes,
        full_nodes=full_nodes,
        ranked_nodes=ranked_nodes,
        case_ctx=case_ctx,
    )
    regions = _region_rows(nodes)
    transitions = _label_transition_rows(nodes)
    focus_nodes = unique_sorted_u32(
        np.concatenate([gate_child.mutable_nodes, full_child.mutable_nodes])
    )
    edge_rows = _edge_change_rows(
        arrays=case_ctx["arrays"],
        source_membership=source_state.membership,
        gate_membership=gate_child.membership,
        full_membership=full_child.membership,
        focus_nodes=focus_nodes,
    )
    moved_gate_nodes = nodes[nodes["moved_source_to_gate"].astype(bool)].copy()
    edge_weight_by_category = dict(
        zip(
            edge_rows["edge_category"].astype(str),
            edge_rows["edge_weight_sum"].astype(float),
            strict=False,
        )
    )
    affected_nodes = unique_sorted_u32(nodes["node"].to_numpy(dtype=np.uint32))
    summary = {
        "schema": "leiden_basin_post_gate_gate_trace.v0",
        "output_dir": str(output_dir),
        "sufficient_dir": str(sufficient_dir),
        "source_move_dir": str(source_move_dir),
        "pair_id": pair_id,
        "prefix_rank": int(prefix_rank),
        "source_recovery_policy": source_recovery_policy,
        "recovery_seed": int(recovery_seed),
        "gate_node_count": int(gate_nodes.size),
        "full_context_node_count": int(full_nodes.size),
        "source_mutable_node_count": int(unique_sorted_u32(source_state.mutable_nodes).size),
        "gate_mutable_node_count": int(unique_sorted_u32(gate_child.mutable_nodes).size),
        "full_mutable_node_count": int(unique_sorted_u32(full_child.mutable_nodes).size),
        "gate_state_delta_q": float(gate_row["state_delta_q_vs_start"]),
        "full_state_delta_q": float(full_row["state_delta_q_vs_start"]),
        "gate_q_gain": float(
            gate_row["state_delta_q_vs_start"] - source_row["state_delta_q_vs_start"]
        ),
        "full_q_gain": float(
            full_row["state_delta_q_vs_start"] - source_row["state_delta_q_vs_start"]
        ),
        "gate_support": float(gate_row["state_support_distance_to_vanilla"]),
        "full_support": float(full_row["state_support_distance_to_vanilla"]),
        "gate_progress": float(gate_row["state_target_progress_from_vanilla"]),
        "full_progress": float(full_row["state_target_progress_from_vanilla"]),
        "source_to_gate_changed_nodes": int(
            nodes["moved_source_to_gate"].astype(bool).sum()
        ),
        "source_to_gate_changed_node_ids": node_csv(
            moved_gate_nodes["node"].to_numpy(dtype=np.uint32)
            if not moved_gate_nodes.empty
            else np.asarray([], dtype=np.uint32)
        ),
        "source_to_gate_changed_gate_nodes": int(
            (
                nodes["moved_source_to_gate"].astype(bool)
                & nodes["in_gate_context"].astype(bool)
            ).sum()
        ),
        "source_to_gate_changed_source_action_nodes": int(
            (
                nodes["moved_source_to_gate"].astype(bool)
                & nodes["in_source_action"].astype(bool)
            ).sum()
        ),
        "source_to_full_changed_nodes": int(
            nodes["moved_source_to_full"].astype(bool).sum()
        ),
        "gate_full_same_label_focus_fraction": float(
            nodes["gate_and_full_same_label"].astype(bool).mean()
        ),
        "gate_full_aligned_label_focus_fraction": float(
            nodes["gate_full_aligned_label"].astype(bool).mean()
        ),
        "gate_full_endpoint_distance_affected": endpoint_distance(
            gate_child.membership,
            full_child.membership,
            affected_nodes,
        ),
        "gate_full_endpoint_distance_sketch": endpoint_distance(
            gate_child.membership,
            full_child.membership,
            case_ctx["sketch_nodes"],
        ),
        "gate_gained_internal_edge_weight": float(
            edge_weight_by_category.get("gate_gained_internal", 0.0)
        ),
        "gate_lost_internal_edge_weight": float(
            edge_weight_by_category.get("gate_lost_internal", 0.0)
        ),
        "sufficient_final_removed_group_count": int(
            sufficient_final.get("removed_group_count", 0)
        ),
    }
    config = {
        "sufficient_dir": str(sufficient_dir),
        "source_move_dir": str(source_move_dir),
        "post_gate_dir": str(post_gate_dir),
        "prefix_dir": str(prefix_dir),
        "profile_batch_dir": str(profile_batch_dir),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "vanilla_dir": str(vanilla_dir),
        "pair_id": pair_id,
        "prefix_rank": int(prefix_rank),
        "source_verdict": source_verdict,
        "source_recovery_policy": source_recovery_policy,
        "baseline_iterations": int(baseline_iterations),
        "candidate_polish_iterations": int(candidate_polish_iterations),
        "local_polish_iterations": int(local_polish_iterations),
        "recovery_polish_iterations": int(recovery_polish_iterations),
        "target_action_multiplier": float(target_action_multiplier),
        "max_target_action_nodes": int(max_target_action_nodes),
        "resolution": float(resolution),
        "randomness": float(randomness),
        "recovery_seed": int(recovery_seed),
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    nodes.to_csv(output_dir / NODE_ROWS_FILENAME, index=False)
    regions.to_csv(output_dir / REGION_ROWS_FILENAME, index=False)
    transitions.to_csv(output_dir / LABEL_TRANSITIONS_FILENAME, index=False)
    edge_rows.to_csv(output_dir / EDGE_ROWS_FILENAME, index=False)
    state_rows.to_csv(output_dir / STATE_ROWS_FILENAME, index=False)
    _write_report(
        output_dir / REPORT_FILENAME,
        summary=summary,
        state_rows=state_rows,
        region_rows=regions,
        transition_rows=transitions,
        edge_rows=edge_rows,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    sufficient_config = _load_json(
        DEFAULT_SUFFICIENT_DIR / "post_gate_sufficient_subset_config.json"
    )
    source_config = _load_source_config(DEFAULT_SOURCE_MOVE_DIR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sufficient-dir", type=Path, default=DEFAULT_SUFFICIENT_DIR)
    parser.add_argument("--source-move-dir", type=Path, default=DEFAULT_SOURCE_MOVE_DIR)
    parser.add_argument(
        "--post-gate-dir",
        type=Path,
        default=Path(sufficient_config.get("post_gate_dir", DEFAULT_POST_GATE_DIR)),
    )
    parser.add_argument(
        "--prefix-dir",
        type=Path,
        default=Path(sufficient_config.get("prefix_dir", DEFAULT_PREFIX_DIR)),
    )
    parser.add_argument(
        "--profile-batch-dir",
        type=Path,
        default=Path(
            sufficient_config.get("profile_batch_dir", DEFAULT_PROFILE_BATCH_DIR)
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-dir", type=Path, action="append", default=None)
    parser.add_argument(
        "--vanilla-dir",
        type=Path,
        default=Path(sufficient_config.get("vanilla_dir", DEFAULT_VANILLA_DIR)),
    )
    parser.add_argument(
        "--pair-id",
        default=str(sufficient_config.get("pair_id", "c0-s11-r0.001")),
    )
    parser.add_argument(
        "--prefix-rank",
        type=int,
        default=int(sufficient_config.get("prefix_rank", 8)),
    )
    parser.add_argument(
        "--source-verdict",
        default=str(
            sufficient_config.get("source_verdict", POST_GATE_VERDICT_NEAR_MISS)
        ),
    )
    parser.add_argument("--source-recovery-policy", default=DEFAULT_SOURCE_RECOVERY_POLICY)
    parser.add_argument(
        "--baseline-iterations",
        type=int,
        default=int(sufficient_config.get("baseline_iterations", 10)),
    )
    parser.add_argument(
        "--candidate-polish-iterations",
        type=int,
        default=int(sufficient_config.get("candidate_polish_iterations", 5)),
    )
    parser.add_argument(
        "--local-polish-iterations",
        type=int,
        default=int(sufficient_config.get("local_polish_iterations", 3)),
    )
    parser.add_argument(
        "--recovery-polish-iterations",
        type=int,
        default=int(sufficient_config.get("recovery_polish_iterations", 10)),
    )
    parser.add_argument(
        "--target-action-multiplier",
        type=float,
        default=float(sufficient_config.get("target_action_multiplier", 0.5)),
    )
    parser.add_argument(
        "--max-target-action-nodes",
        type=int,
        default=int(sufficient_config.get("max_target_action_nodes", 64)),
    )
    parser.add_argument("--cumulative-fraction", type=float, default=0.80)
    parser.add_argument("--min-score-fraction", type=float, default=0.05)
    parser.add_argument("--min-gap-fraction", type=float, default=0.25)
    parser.add_argument("--min-guarded-pull-fraction", type=float, default=0.50)
    parser.add_argument(
        "--resolution",
        type=float,
        default=float(sufficient_config.get("resolution", 0.01)),
    )
    parser.add_argument(
        "--randomness",
        type=float,
        default=float(sufficient_config.get("randomness", 0.01)),
    )
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    parser.add_argument("--polish-seed-offset", type=int, default=11000)
    parser.add_argument("--recovery-seed-offset", type=int, default=21000)
    parser.add_argument("--min-support-shift-from-vanilla", type=float, default=0.05)
    parser.add_argument("--min-material-q-gain", type=float, default=0.0)
    parser.set_defaults(source_config=source_config)
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
    summary = run_trace(
        sufficient_dir=args.sufficient_dir,
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
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
