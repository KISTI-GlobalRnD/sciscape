#!/usr/bin/env python3
"""Compare fixed-cap and guarded-elbow target growth with bounded polish."""

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
    ACTION_REMAINING_TARGET_TOPK,
    SEARCH_POLICY_STATE_GREEDY,
    TransitionAction,
    cap_context_count,
    edge_public_row,
    make_prefix_state,
    node_csv,
    prefix_direct_nodes,
    remaining_target_elbow_summary,
    remaining_target_pull_frame,
    select_pareto_rows,
    unique_sorted_u32,
)
from search_leiden_basin_transitions import (  # noqa: E402
    _evaluate_state,
    _polished_child,
)

DEFAULT_OUTPUT_DIR = DEFAULT_PROFILE_BATCH_DIR.parent / (
    "basin_transition_target_elbow_polish_field34_cc_v0"
)
STATE_ROWS_FILENAME = "target_elbow_polish_states.csv"
EDGE_ROWS_FILENAME = "target_elbow_polish_edges.csv"
PARETO_ROWS_FILENAME = "target_elbow_polish_pareto_rows.csv"
CASE_ROWS_FILENAME = "target_elbow_polish_case_rows.csv"
SUMMARY_FILENAME = "target_elbow_polish_summary.json"
CONFIG_FILENAME = "target_elbow_polish_config.json"
REPORT_FILENAME = "target_elbow_polish_report.md"

PATH_POLICY_FIXED_CAP = "fixed_cap"
PATH_POLICY_GUARDED_ELBOW = "guarded_elbow"
PATH_POLICY_GUARDED_ESCALATE = "guarded_escalate"
PATH_POLICY_GUARDED_BACKFILL = "guarded_backfill"
SELECTION_FIXED_TAIL_BACKFILL = "fixed_tail_backfill"
PATH_POLICIES = (
    PATH_POLICY_FIXED_CAP,
    PATH_POLICY_GUARDED_ELBOW,
    PATH_POLICY_GUARDED_ESCALATE,
    PATH_POLICY_GUARDED_BACKFILL,
)

def _parse_csv_tuple(value: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if not value.strip():
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip())

def _parse_int_tuple(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())

def _rank_and_filter_prefix_rows(
    prefixes: pd.DataFrame,
    *,
    selected_prefix_ranks: tuple[int, ...] = (),
) -> pd.DataFrame:
    """Attach stable selected-prefix ranks and optionally keep only those ranks."""
    if prefixes.empty:
        return prefixes.copy()
    out = prefixes.copy()
    out["selected_prefix_rank"] = (
        out.groupby("pair_id", sort=False).cumcount().astype(int) + 1
    )
    if selected_prefix_ranks:
        keep = {int(rank) for rank in selected_prefix_ranks}
        out = out[out["selected_prefix_rank"].isin(keep)].copy()
    return out.reset_index(drop=True)

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

def _safe_divide(numerator: Any, denominator: Any) -> float:
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(num) or not math.isfinite(den) or den == 0.0:
        return math.nan
    return float(num / den)

def _target_selection_policy(
    *,
    path_policy: str,
    target_stage_index: int,
    parent_row: dict[str, Any],
    min_support_shift_from_vanilla: float,
) -> tuple[str, str]:
    """Return the concrete selector and reason for this staged target step."""
    if path_policy == PATH_POLICY_FIXED_CAP:
        return PATH_POLICY_FIXED_CAP, "fixed_policy"
    if path_policy == PATH_POLICY_GUARDED_ELBOW:
        return PATH_POLICY_GUARDED_ELBOW, "guarded_policy"
    if path_policy not in {PATH_POLICY_GUARDED_ESCALATE, PATH_POLICY_GUARDED_BACKFILL}:
        raise ValueError(f"Unsupported target elbow path policy: {path_policy}")
    if int(target_stage_index) <= 1:
        return PATH_POLICY_GUARDED_ELBOW, "initial_guarded"
    support_shift = float(parent_row.get("state_support_distance_to_vanilla", 0.0))
    if support_shift < float(min_support_shift_from_vanilla):
        return PATH_POLICY_FIXED_CAP, "below_support_gate"
    return PATH_POLICY_GUARDED_ELBOW, "support_gate_reached"

def _selected_k_for_policy(summary: dict[str, Any], selection_policy: str) -> int:
    if selection_policy == PATH_POLICY_FIXED_CAP:
        return int(summary["fixed_effective_k"])
    if selection_policy == PATH_POLICY_GUARDED_ELBOW:
        return int(summary["guarded_elbow_k"])
    raise ValueError(f"Unsupported target elbow selection policy: {selection_policy}")

def _selection_context(
    *,
    path_policy: str,
    selection_policy: str,
    escalation_reason: str,
    target_stage_index: int,
    selected: np.ndarray,
    elbow: dict[str, Any] | None = None,
) -> dict[str, Any]:
    elbow = elbow or {}
    return {
        "path_policy": path_policy,
        "selection_policy": selection_policy,
        "escalation_reason": escalation_reason,
        "escalated_to_fixed": bool(
            path_policy in {PATH_POLICY_GUARDED_ESCALATE, PATH_POLICY_GUARDED_BACKFILL}
            and selection_policy in {PATH_POLICY_FIXED_CAP, SELECTION_FIXED_TAIL_BACKFILL}
        ),
        "target_stage_index": int(target_stage_index),
        "selected_k": int(unique_sorted_u32(selected).size),
        "selected_node_ids": node_csv(selected),
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

def _case_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    out: list[dict[str, Any]] = []
    for (pair_id, path_policy), group in rows.groupby(
        ["pair_id", "path_policy"],
        sort=True,
    ):
        labels = group["search_recovery_label"].value_counts().to_dict()
        best = group.sort_values(
            [
                "state_greedy_score",
                "state_target_progress_from_vanilla",
                "state_support_distance_to_vanilla",
                "state_delta_q_vs_start",
                "mutable_node_count",
            ],
            ascending=[False, False, False, False, True],
        ).iloc[0]
        shifted = group[
            group["search_recovery_label"].astype(str) == "support_shift_q_recovered"
        ].copy()
        if shifted.empty:
            best_shift = best
        else:
            best_shift = shifted.sort_values(
                [
                    "state_greedy_score",
                    "state_target_progress_from_vanilla",
                    "state_delta_q_vs_start",
                    "mutable_node_count",
                    "path_elapsed_sec",
                ],
                ascending=[False, False, False, True, True],
            ).iloc[0]
        target_steps = group[group["target_stage_index"].astype(int) > 0]
        escalated_steps = (
            int(target_steps["escalated_to_fixed"].astype(bool).sum())
            if not target_steps.empty and "escalated_to_fixed" in target_steps.columns
            else 0
        )
        out.append(
            {
                "pair_id": pair_id,
                "path_policy": path_policy,
                "state_rows": int(len(group)),
                "target_step_rows": int(len(target_steps)),
                "escalated_target_step_rows": escalated_steps,
                "support_shift_q_recovered_rows": int(
                    labels.get("support_shift_q_recovered", 0)
                ),
                "vanilla_collapse_rows": int(labels.get("vanilla_collapse", 0)),
                "quality_loss_rows": int(labels.get("quality_loss", 0)),
                "total_elapsed_sec": float(group["elapsed_sec"].sum()),
                "median_selected_k": (
                    float(target_steps["selected_k"].median())
                    if not target_steps.empty
                    else math.nan
                ),
                "median_mutable_node_count": float(group["mutable_node_count"].median()),
                "best_state_id": best["state_id"],
                "best_search_score": float(best["state_greedy_score"]),
                "best_delta_q_vs_start": float(best["state_delta_q_vs_start"]),
                "best_target_progress_from_vanilla": float(
                    best["state_target_progress_from_vanilla"]
                ),
                "best_support_distance_to_vanilla": float(
                    best["state_support_distance_to_vanilla"]
                ),
                "best_target_coverage_fraction": float(
                    best["target_coverage_fraction"]
                ),
                "best_mutable_node_count": int(best["mutable_node_count"]),
                "best_path_elapsed_sec": float(best.get("path_elapsed_sec", 0.0)),
                "best_delta_q_per_mutable_node": _safe_divide(
                    best["state_delta_q_vs_start"],
                    best["mutable_node_count"],
                ),
                "best_label": best["search_recovery_label"],
                "best_shift_state_id": best_shift["state_id"],
                "best_shift_search_score": float(best_shift["state_greedy_score"]),
                "best_shift_delta_q_vs_start": float(
                    best_shift["state_delta_q_vs_start"]
                ),
                "best_shift_target_progress_from_vanilla": float(
                    best_shift["state_target_progress_from_vanilla"]
                ),
                "best_shift_support_distance_to_vanilla": float(
                    best_shift["state_support_distance_to_vanilla"]
                ),
                "best_shift_target_coverage_fraction": float(
                    best_shift["target_coverage_fraction"]
                ),
                "best_shift_mutable_node_count": int(best_shift["mutable_node_count"]),
                "best_shift_path_elapsed_sec": float(
                    best_shift.get("path_elapsed_sec", 0.0)
                ),
                "best_shift_progress_per_mutable_node": _safe_divide(
                    best_shift["state_target_progress_from_vanilla"],
                    best_shift["mutable_node_count"],
                ),
                "best_shift_delta_q_per_elapsed_sec": _safe_divide(
                    best_shift["state_delta_q_vs_start"],
                    best_shift.get("path_elapsed_sec", 0.0),
                ),
                "best_shift_label": best_shift["search_recovery_label"],
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
    path_policies: tuple[str, ...],
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
    rows: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
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
    for fallback_prefix_rank, (_, prefix_row) in enumerate(
        case_prefix_rows.iterrows(),
        start=1,
    ):
        prefix_rank = int(prefix_row.get("selected_prefix_rank", fallback_prefix_rank))
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
        for policy_index, path_policy in enumerate(path_policies, start=1):
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
                sketch_nodes=sketch_nodes,
                start_quality=vanilla.quality,
                candidate_quality=candidate.recreated.quality,
                vanilla_quality=vanilla.quality,
                vanilla_support_distance_to_candidate=vanilla_support_distance_to_candidate,
                context={
                    **prefix_context,
                    **_selection_context(
                        path_policy=path_policy,
                        selection_policy="raw_prefix",
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
                    **_selection_context(
                        path_policy=path_policy,
                        selection_policy="prefix_polish",
                        escalation_reason="not_applicable",
                        target_stage_index=0,
                        selected=np.asarray([], dtype=np.uint32),
                    ),
                },
                parent_row=root_row,
                min_support_shift_from_vanilla=min_support_shift_from_vanilla,
                min_material_q_gain=min_material_q_gain,
            )
            prefix_path_elapsed = float(prefix_polished.elapsed_sec)
            prefix_row_public["path_elapsed_sec"] = prefix_path_elapsed
            rows.append(prefix_row_public)
            edges.append(
                edge_public_row(
                    parent_state_id=root.state_id,
                    child_state_id=prefix_polished.state_id,
                    action=prefix_action,
                    context={**case_context, "path_policy": path_policy},
                )
            )
            current = prefix_polished
            current_row = prefix_row_public
            current_path_elapsed = prefix_path_elapsed
            pending_backfill_nodes = np.asarray([], dtype=np.uint32)
            for target_stage_index in range(1, int(max_target_stages) + 1):
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
                    [
                        int(node)
                        for node in pending_backfill_nodes
                        if int(node) in remaining
                    ],
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
                        f"path_policy={path_policy};"
                        f"selection_policy={selection_policy};"
                        f"escalation_reason={escalation_reason};"
                        f"target_stage={int(target_stage_index)};"
                        f"selected_k={int(selected.size)};"
                        f"fixed_effective_k={int(elbow['fixed_effective_k'])};"
                        f"guarded_elbow_k={int(elbow['guarded_elbow_k'])};"
                        f"guarded_elbow_reason={elbow['guarded_elbow_reason']}"
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
                    sketch_nodes=sketch_nodes,
                    start_quality=vanilla.quality,
                    candidate_quality=candidate.recreated.quality,
                    vanilla_quality=vanilla.quality,
                    vanilla_support_distance_to_candidate=vanilla_support_distance_to_candidate,
                    context={
                        **prefix_context,
                        **select_context,
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
                        context={**case_context, "path_policy": path_policy},
                    )
                )
                if (
                    path_policy == PATH_POLICY_GUARDED_BACKFILL
                    and selection_policy == PATH_POLICY_GUARDED_ELBOW
                    and int(elbow["fixed_effective_k"]) > int(elbow["guarded_elbow_k"])
                    and not frame.empty
                ):
                    pending_backfill_nodes = np.asarray(
                        frame.iloc[
                            int(elbow["guarded_elbow_k"]) : int(
                                elbow["fixed_effective_k"]
                            )
                        ]["node"],
                        dtype=np.uint32,
                    )
                else:
                    pending_backfill_nodes = np.asarray([], dtype=np.uint32)
                current = child
                current_row = row
    return pd.DataFrame(rows), pd.DataFrame(edges)

def write_report(
    path: Path,
    *,
    rows: pd.DataFrame,
    case_rows: pd.DataFrame,
    pareto_rows: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Basin Target Elbow Polish v0",
        "",
        "This artifact compares fixed-cap target growth and guarded-elbow target growth after the same bounded local polish.",
        "",
        "It is diagnostic-only. A smaller selected target set is useful only if it keeps material support shift and QF recovery at lower mutable-node or wall-time cost.",
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
        "pair_ids",
        "path_policies",
        "top_prefixes_per_case",
        "selected_prefix_ranks",
        "max_target_stages",
        "local_polish_iterations",
        "target_action_multiplier",
        "max_target_action_nodes",
        "cumulative_fraction",
        "min_gap_fraction",
        "min_guarded_pull_fraction",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")
    lines.extend(["", "## Case Rows", ""])
    case_cols = [
        "pair_id",
        "path_policy",
        "state_rows",
        "target_step_rows",
        "escalated_target_step_rows",
        "support_shift_q_recovered_rows",
        "vanilla_collapse_rows",
        "quality_loss_rows",
        "total_elapsed_sec",
        "median_selected_k",
        "median_mutable_node_count",
        "best_search_score",
        "best_delta_q_vs_start",
        "best_target_progress_from_vanilla",
        "best_support_distance_to_vanilla",
        "best_target_coverage_fraction",
        "best_mutable_node_count",
        "best_path_elapsed_sec",
        "best_label",
        "best_shift_search_score",
        "best_shift_delta_q_vs_start",
        "best_shift_target_progress_from_vanilla",
        "best_shift_support_distance_to_vanilla",
        "best_shift_target_coverage_fraction",
        "best_shift_mutable_node_count",
        "best_shift_path_elapsed_sec",
        "best_shift_progress_per_mutable_node",
        "best_shift_label",
    ]
    lines.extend(_markdown_table(case_rows[[c for c in case_cols if c in case_rows.columns]]))
    lines.extend(["", "## Best Shift Rows", ""])
    if not rows.empty:
        shift_rows = rows[
            rows["search_recovery_label"].astype(str) == "support_shift_q_recovered"
        ].sort_values(
            [
                "pair_id",
                "path_policy",
                "state_greedy_score",
                "state_target_progress_from_vanilla",
            ],
            ascending=[True, True, False, False],
        )
        shift_cols = [
            "pair_id",
            "path_policy",
            "state_id",
            "target_stage_index",
            "selection_policy",
            "escalation_reason",
            "escalated_to_fixed",
            "selected_k",
            "fixed_effective_k",
            "guarded_elbow_k",
            "guarded_elbow_reason",
            "state_greedy_score",
            "state_delta_q_vs_start",
            "state_target_progress_from_vanilla",
            "state_support_distance_to_vanilla",
            "target_coverage_fraction",
            "mutable_node_count",
            "path_elapsed_sec",
            "marginal_target_distance_reduction",
            "marginal_cost_per_target_node",
        ]
        lines.extend(
            _markdown_table(
                shift_rows[[c for c in shift_cols if c in shift_rows.columns]],
                max_rows=80,
            )
        )
    lines.extend(["", "## Pareto Rows", ""])
    pareto_cols = [
        "pair_id",
        "path_policy",
        "state_id",
        "depth",
        "target_stage_index",
        "selection_policy",
        "escalation_reason",
        "escalated_to_fixed",
        "selected_k",
        "state_greedy_score",
        "state_delta_q_vs_start",
        "state_target_progress_from_vanilla",
        "state_support_distance_to_vanilla",
        "target_coverage_fraction",
        "mutable_node_count",
        "path_elapsed_sec",
        "search_recovery_label",
    ]
    lines.extend(
        _markdown_table(
            pareto_rows[[c for c in pareto_cols if c in pareto_rows.columns]],
            max_rows=80,
        )
    )
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- This report does not promote the elbow rule into Leiden.",
            "- The acceptance question is cost-adjusted support/QF recovery, not `selected_k` reduction alone.",
            "- If guarded elbow saves nodes but loses support shift or score, it is only a runtime heuristic candidate.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_evaluation(
    *,
    prefix_dir: Path,
    profile_batch_dir: Path,
    output_dir: Path,
    candidate_dirs: tuple[Path, ...],
    vanilla_dir: Path,
    pair_ids: tuple[str, ...],
    top_prefixes_per_case: int,
    selected_prefix_ranks: tuple[int, ...],
    baseline_iterations: int,
    candidate_polish_iterations: int,
    local_polish_iterations: int,
    max_target_stages: int,
    path_policies: tuple[str, ...],
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
    min_support_shift_from_vanilla: float,
    min_material_q_gain: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefixes = select_prefix_rows(
        pd.read_csv(prefix_dir / BARRIER_PREFIX_ROWS_FILENAME),
        pair_ids=pair_ids,
        top_prefixes_per_case=top_prefixes_per_case,
    )
    prefixes = _rank_and_filter_prefix_rows(
        prefixes,
        selected_prefix_ranks=selected_prefix_ranks,
    )
    if prefixes.empty:
        raise ValueError("No prefix rows selected for target-elbow polish evaluation")
    frames: list[pd.DataFrame] = []
    edge_frames: list[pd.DataFrame] = []
    for _, case_prefixes in prefixes.groupby("pair_id", sort=True):
        rows, edges = _evaluate_case(
            case_prefix_rows=case_prefixes,
            profile_batch_dir=profile_batch_dir,
            candidate_dirs=candidate_dirs,
            vanilla_dir=vanilla_dir,
            baseline_iterations=baseline_iterations,
            candidate_polish_iterations=candidate_polish_iterations,
            local_polish_iterations=local_polish_iterations,
            max_target_stages=max_target_stages,
            path_policies=path_policies,
            target_action_multiplier=target_action_multiplier,
            max_target_action_nodes=max_target_action_nodes,
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
        )
        frames.append(rows)
        edge_frames.append(edges)
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    edges = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    pareto_rows = select_pareto_rows(
        rows,
        max_rows=100,
        search_policy=SEARCH_POLICY_STATE_GREEDY,
    )
    case_rows = _case_rows(rows)
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
        "selected_prefix_ranks": list(selected_prefix_ranks),
        "baseline_iterations": int(baseline_iterations),
        "candidate_polish_iterations": int(candidate_polish_iterations),
        "local_polish_iterations": int(local_polish_iterations),
        "max_target_stages": int(max_target_stages),
        "path_policies": list(path_policies),
        "target_action_multiplier": float(target_action_multiplier),
        "max_target_action_nodes": int(max_target_action_nodes),
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
    }
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "schema": "leiden_basin_target_elbow_polish.v0",
        "output_dir": str(output_dir),
        "state_rows": int(len(rows)),
        "edge_rows": int(len(edges)),
        "pareto_rows": int(len(pareto_rows)),
        "case_rows": int(len(case_rows)),
        **config,
    }
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(
        output_dir / REPORT_FILENAME,
        rows=rows,
        case_rows=case_rows,
        pareto_rows=pareto_rows,
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
        "--selected-prefix-ranks",
        default="",
        help="Comma-separated 1-based ranks after top-prefix selection, e.g. 6,8,10.",
    )
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--candidate-polish-iterations", type=int, default=5)
    parser.add_argument("--local-polish-iterations", type=int, default=3)
    parser.add_argument("--max-target-stages", type=int, default=3)
    parser.add_argument("--path-policies", default="fixed_cap,guarded_elbow")
    parser.add_argument("--target-action-multiplier", type=float, default=0.5)
    parser.add_argument("--max-target-action-nodes", type=int, default=64)
    parser.add_argument("--cumulative-fraction", type=float, default=0.80)
    parser.add_argument("--min-score-fraction", type=float, default=0.05)
    parser.add_argument("--min-gap-fraction", type=float, default=0.25)
    parser.add_argument("--min-guarded-pull-fraction", type=float, default=0.50)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    parser.add_argument("--polish-seed-offset", type=int, default=11000)
    parser.add_argument("--min-support-shift-from-vanilla", type=float, default=0.05)
    parser.add_argument("--min-material-q-gain", type=float, default=0.0)
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path_policies = _parse_csv_tuple(args.path_policies, PATH_POLICIES)
    unsupported = sorted(set(path_policies) - set(PATH_POLICIES))
    if unsupported:
        raise ValueError(f"Unsupported path policies: {unsupported}")
    summary = run_evaluation(
        prefix_dir=args.prefix_dir,
        profile_batch_dir=args.profile_batch_dir,
        output_dir=args.output_dir,
        candidate_dirs=tuple(args.candidate_dir or DEFAULT_CANDIDATE_DIRS),
        vanilla_dir=args.vanilla_dir,
        pair_ids=_parse_csv_tuple(args.pair_ids),
        top_prefixes_per_case=args.top_prefixes_per_case,
        selected_prefix_ranks=_parse_int_tuple(args.selected_prefix_ranks),
        baseline_iterations=args.baseline_iterations,
        candidate_polish_iterations=args.candidate_polish_iterations,
        local_polish_iterations=args.local_polish_iterations,
        max_target_stages=args.max_target_stages,
        path_policies=path_policies,
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
        min_support_shift_from_vanilla=args.min_support_shift_from_vanilla,
        min_material_q_gain=args.min_material_q_gain,
    )
    print(summary)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
