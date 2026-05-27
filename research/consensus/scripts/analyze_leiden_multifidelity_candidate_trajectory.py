#!/usr/bin/env python3
"""Replay labeled Leiden multi-fidelity candidates with phase traces.

This is a focused attribution tool for cases where p1 prescreen ranks miss the
full p5 winner. It replays the existing candidate perturbations under traced
iteration budgets so we can see when the late-improving candidate enters the
p1 top2/top3 set without changing the production Leiden path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_leiden_multifidelity_candidate_misses import (  # noqa: E402
    build_candidate_rank_diagnostics,
)
from run_leiden_hysteresis_work_acceleration_monitor import (  # noqa: E402
    DEFAULT_GRAPH_DIRS,
    _case_name,
    _compact_membership,
    _extract_phase_checkpoints,
    _finite_float,
    _load_graph_arrays,
    _parse_graph_dirs,
    _quality_points,
    _reconstruct_external_group,
    _trace_disabled_context,
    _trace_file_context,
    _trace_run_context,
)
from sciscape.clustering.leiden_rust import build_leiden_graph  # noqa: E402


DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_hysteresis_multifidelity_label_field30_20260513"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_hysteresis_multifidelity_candidate_trajectory_cc11_20260513"
)

LOCAL_MOVE_MARGIN_SUMMARY_COLUMNS = [
    "case",
    "seed",
    "candidate_index",
    "replay_iterations",
    "run_id",
    "iteration",
    "depth",
    "event_count",
    "moved_count",
    "near_zero_margin_count",
    "margin_min",
    "margin_p10",
    "margin_p50",
    "best_increment_min",
    "best_increment_p50",
    "best_increment_max",
    "second_increment_min",
    "second_increment_p50",
    "second_increment_max",
    "top_low_margin_node_ids",
    "top_moved_low_margin_node_ids",
]

LOCAL_MERGE_PARENT_SUMMARY_COLUMNS = [
    "case",
    "seed",
    "candidate_index",
    "replay_iterations",
    "run_id",
    "iteration",
    "depth",
    "parent_row_count",
    "decision_count",
    "low_margin_count",
    "changed_count",
    "min_margin_min",
    "p10_margin_min",
    "p50_margin_min",
    "largest_child_fraction_max",
    "top_low_margin_parent_ids",
    "top_decision_parent_ids",
    "top_changed_parent_ids",
    "top_small_margin_parent_ids",
    "top_largest_child_fraction_parent_ids",
]

TARGET_PARENT_EVENT_COLUMNS = [
    "context_role",
    "case",
    "seed",
    "candidate_index",
    "replay_iterations",
    "run_id",
    "iteration",
    "depth",
    "parent_id",
    "parent_visit_index",
    "source",
    "parent_size",
    "parent_weight",
    "decision_count",
    "low_margin_decision_count",
    "changed_decision_count",
    "min_margin",
    "p10_margin",
    "p50_margin",
    "selected_child_count",
    "largest_child_fraction",
    "after_local_move_quality",
    "after_local_move_membership_hash",
    "after_refinement_quality",
    "after_refinement_membership_hash",
    "after_aggregation_phase",
    "after_aggregation_quality",
    "after_aggregation_membership_hash",
    "quality_gain_since_previous_local_move",
]

TARGET_PARENT_CONTRAST_COLUMNS = [
    "context_role",
    "case",
    "seed",
    "candidate_index",
    "replay_iterations",
    "run_id",
    "iteration",
    "depth",
    "parent_id",
    "parent_seen",
    "parent_size",
    "parent_weight",
    "decision_count",
    "low_margin_decision_count",
    "changed_decision_count",
    "min_margin",
    "p10_margin",
    "p50_margin",
    "selected_child_count",
    "largest_child_fraction",
    "after_local_move_quality",
    "after_refinement_quality",
    "after_aggregation_quality",
    "quality_gain_since_previous_local_move",
    "target_low_margin_delta",
    "target_min_margin_delta",
]

TARGET_CANDIDATE_INDEX = 2
TARGET_REPLAY_ITERATIONS = 2
TARGET_ITERATION = 2
TARGET_DEPTH = 1
TARGET_PARENT_IDS = [1931, 2678, 2867, 5121]
TOP_ID_LIMIT = 16
LOCAL_MOVE_FOCUS_NODE_ENV = "SCISCAPE_DDM_LOCAL_MOVE_FOCUS_NODES"
LOCAL_MOVE_NEIGHBOR_NODE_ENV = "SCISCAPE_DDM_LOCAL_MOVE_NEIGHBOR_NODES"

LOCAL_MOVE_FOCUS_EVENT_COLUMNS = [
    "case",
    "seed",
    "candidate_index",
    "replay_iterations",
    "run_id",
    "iteration",
    "depth",
    "node",
    "role",
    "current_cluster",
    "best_cluster",
    "second_cluster",
    "best_increment",
    "second_increment",
    "margin",
    "moved",
    "quality_gain_since_previous_local_move",
]

LOCAL_MOVE_FOCUS_SUMMARY_COLUMNS = [
    "case",
    "seed",
    "candidate_index",
    "replay_iterations",
    "run_id",
    "iteration",
    "depth",
    "quality_gain_since_previous_local_move",
    "target_event_count",
    "target_moved_count",
    "target_moved_node_ids",
    "target_margin_min",
    "target_margin_p50",
    "neighbor_event_count",
    "neighbor_moved_count",
    "neighbor_moved_node_ids",
    "neighbor_margin_min",
    "neighbor_margin_p50",
    "moved_count",
    "moved_node_ids",
    "moved_margin_min",
    "moved_margin_p50",
    "best_increment_min",
    "best_increment_p50",
    "best_increment_max",
    "second_increment_min",
    "second_increment_p50",
    "second_increment_max",
    "moved_overlap_previous_window_count",
    "moved_overlap_next_window_count",
    "moved_overlap_target_window_count",
]

PERTURBATION_FOOTPRINT_EVENT_COLUMNS = [
    "case",
    "seed",
    "candidate_index",
    "replay_iterations",
    "run_id",
    "contrast_kind",
    "contrast_run_id",
    "hop_bucket",
    "changed_node_count",
    "changed_node_fraction",
    "changed_reference_clusters_count",
    "changed_perturb_clusters_count",
    "sample_node_ids",
]

PERTURBATION_FOOTPRINT_SUMMARY_COLUMNS = [
    "case",
    "seed",
    "candidate_index",
    "replay_iterations",
    "run_id",
    "contrast_kind",
    "contrast_run_id",
    "source_cluster",
    "target_cluster",
    "target_node_count",
    "neighbor_node_count",
    "two_hop_node_count",
    "changed_nodes_total",
    "changed_nodes_fraction",
    "changed_hop0_count",
    "changed_hop1_count",
    "changed_hop2_count",
    "changed_hop3plus_count",
    "fraction_changed_within_1hop",
    "fraction_changed_within_2hop",
    "changed_source_or_target_initial_count",
    "fraction_changed_source_or_target_initial",
    "changed_reference_clusters_total",
    "changed_perturb_clusters_total",
    "max_hop_reached",
    "classification",
    "classification_confidence",
    "baseline_quality",
    "reference_quality",
    "perturb_quality",
    "delta_q_vs_baseline",
    "delta_q_vs_reference",
    "sample_changed_node_ids",
]


def _parse_int_list(value: str) -> list[int]:
    if value.strip().lower() == "all":
        return []
    out = [int(part) for part in value.split(",") if part.strip()]
    if not out:
        raise ValueError("expected at least one integer or 'all'")
    return out


def _parse_positive_int_list(value: str) -> list[int]:
    out = _parse_int_list(value)
    invalid = [item for item in out if item <= 0]
    if invalid:
        raise ValueError(
            f"replay iterations must be positive because run_leiden n_iterations=0 "
            f"does not mean initial-only here: {invalid}"
        )
    return out


def _parse_parent_ids(value: str) -> list[int]:
    parsed = _parse_int_list(value)
    return parsed or list(TARGET_PARENT_IDS)


def _node_env_value(nodes: Any) -> str:
    values = sorted({int(node) for node in np.asarray(nodes, dtype=np.int64).tolist()})
    return ",".join(str(node) for node in values)


def _one_hop_neighbor_nodes(src: Any, dst: Any, target_nodes: Any) -> list[int]:
    targets = {int(node) for node in np.asarray(target_nodes, dtype=np.int64).tolist()}
    if not targets:
        return []
    neighbors: set[int] = set()
    for left, right in zip(np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64), strict=False):
        left_int = int(left)
        right_int = int(right)
        if left_int in targets and right_int not in targets:
            neighbors.add(right_int)
        if right_int in targets and left_int not in targets:
            neighbors.add(left_int)
    return sorted(neighbors)


def _hop_distance_sets(
    src: Any,
    dst: Any,
    target_nodes: Any,
    *,
    max_hop: int = 2,
) -> dict[int, set[int]]:
    targets = {int(node) for node in np.asarray(target_nodes, dtype=np.int64).tolist()}
    hops: dict[int, set[int]] = {0: set(targets)}
    if not targets or max_hop <= 0:
        return hops
    src_arr = np.asarray(src, dtype=np.int64)
    dst_arr = np.asarray(dst, dtype=np.int64)
    visited = set(targets)
    frontier = set(targets)
    for hop in range(1, max_hop + 1):
        next_frontier: set[int] = set()
        if not frontier:
            hops[hop] = next_frontier
            continue
        for left, right in zip(src_arr, dst_arr, strict=False):
            left_int = int(left)
            right_int = int(right)
            if left_int in frontier and right_int not in visited:
                next_frontier.add(right_int)
            if right_int in frontier and left_int not in visited:
                next_frontier.add(left_int)
        visited.update(next_frontier)
        hops[hop] = next_frontier
        frontier = next_frontier
    return hops


@contextmanager
def _local_move_focus_context(
    *,
    target_nodes: Any,
    neighbor_nodes: Any,
) -> Iterator[None]:
    previous = {
        LOCAL_MOVE_FOCUS_NODE_ENV: os.environ.get(LOCAL_MOVE_FOCUS_NODE_ENV),
        LOCAL_MOVE_NEIGHBOR_NODE_ENV: os.environ.get(LOCAL_MOVE_NEIGHBOR_NODE_ENV),
    }
    target_value = _node_env_value(target_nodes)
    neighbor_value = _node_env_value(neighbor_nodes)
    if target_value:
        os.environ[LOCAL_MOVE_FOCUS_NODE_ENV] = target_value
    else:
        os.environ.pop(LOCAL_MOVE_FOCUS_NODE_ENV, None)
    if neighbor_value:
        os.environ[LOCAL_MOVE_NEIGHBOR_NODE_ENV] = neighbor_value
    else:
        os.environ.pop(LOCAL_MOVE_NEIGHBOR_NODE_ENV, None)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _focused_trace_run_context(
    run_id: str,
    *,
    target_max_weight: float,
    target_nodes: Any,
    neighbor_nodes: Any,
) -> Iterator[None]:
    with _local_move_focus_context(
        target_nodes=target_nodes,
        neighbor_nodes=neighbor_nodes,
    ):
        with _trace_run_context(run_id, target_max_weight=target_max_weight):
            yield


def _load_candidate_diagnostics(input_dir: Path) -> pd.DataFrame:
    diagnostics_path = input_dir / "multifidelity_candidate_rank_diagnostics.csv"
    if diagnostics_path.exists():
        return pd.read_csv(diagnostics_path)
    candidate_path = input_dir / "candidate_level_rows.csv"
    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)
    return build_candidate_rank_diagnostics(pd.read_csv(candidate_path))


def _find_graph_dir(graph_dirs: list[Path], case: str) -> Path:
    for graph_dir in graph_dirs:
        if _case_name(graph_dir) == case:
            return graph_dir
    available = ", ".join(_case_name(path) for path in graph_dirs)
    raise ValueError(f"could not find graph dir for case {case!r}; available: {available}")


def _candidate_rows_for_case(
    diagnostics: pd.DataFrame,
    *,
    case_contains: str,
    seed: int,
    candidate_indices: list[int],
) -> pd.DataFrame:
    rows = diagnostics[
        diagnostics["case"].astype(str).str.contains(case_contains, regex=False)
        & diagnostics["seed"].astype(int).eq(seed)
    ].copy()
    if rows.empty:
        raise ValueError(f"no candidate rows match case_contains={case_contains!r}, seed={seed}")
    if candidate_indices:
        rows = rows[rows["candidate_index"].astype(int).isin(candidate_indices)].copy()
    if rows.empty:
        raise ValueError(f"no candidate rows left after candidate index filter {candidate_indices}")
    return rows.sort_values("candidate_index").reset_index(drop=True)


def _run_id(case: str, seed: int, candidate_index: int, replay_iterations: int) -> str:
    return f"{case}|seed={seed}|candidate={candidate_index}|p{replay_iterations}"


def replay_candidate_trajectories(args: argparse.Namespace) -> dict[str, Path]:
    input_dir = args.input_dir.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    quality_trace_path = out_dir / "candidate_trajectory_quality_trace.jsonl"
    trajectory_trace_path = out_dir / "candidate_trajectory_trace_raw.jsonl"
    phase_path = out_dir / "candidate_trajectory_phase_checkpoints.csv"
    local_move_gain_path = out_dir / "candidate_trajectory_local_move_gain_rows.csv"
    local_move_margin_summary_path = (
        out_dir / "candidate_trajectory_local_move_margin_summary.csv"
    )
    local_move_focus_events_path = (
        out_dir / "candidate_trajectory_local_move_focus_events.csv"
    )
    local_move_focus_summary_path = (
        out_dir / "candidate_trajectory_local_move_focus_summary.csv"
    )
    local_move_movement_report_path = (
        out_dir / "candidate_trajectory_local_move_movement_attribution_report.md"
    )
    perturbation_footprint_events_path = (
        out_dir / "candidate_trajectory_perturbation_footprint_events.csv"
    )
    perturbation_footprint_summary_path = (
        out_dir / "candidate_trajectory_perturbation_footprint_summary.csv"
    )
    perturbation_footprint_report_path = (
        out_dir / "candidate_trajectory_perturbation_footprint_report.md"
    )
    local_merge_parent_summary_path = (
        out_dir / "candidate_trajectory_local_merge_parent_summary.csv"
    )
    qf_points_path = out_dir / "candidate_trajectory_qf_i_k_points.csv"
    run_rows_path = out_dir / "candidate_trajectory_run_rows.csv"
    rank_summary_path = out_dir / "candidate_trajectory_rank_summary.csv"
    transition_summary_path = out_dir / "candidate_trajectory_transition_summary.csv"
    report_path = out_dir / "candidate_trajectory_report.md"
    depth_attribution_report_path = out_dir / "candidate_trajectory_depth_attribution_report.md"
    target_parent_events_path = out_dir / "candidate_trajectory_target_parent_events.csv"
    target_parent_contrast_path = out_dir / "candidate_trajectory_target_parent_contrast.csv"
    parent_causal_window_report_path = (
        out_dir / "candidate_trajectory_parent_causal_window_report.md"
    )
    summary_path = out_dir / "candidate_trajectory_summary.json"

    if not args.resume:
        for path in (
            quality_trace_path,
            trajectory_trace_path,
            phase_path,
            local_move_gain_path,
            local_move_margin_summary_path,
            local_move_focus_events_path,
            local_move_focus_summary_path,
            local_move_movement_report_path,
            perturbation_footprint_events_path,
            perturbation_footprint_summary_path,
            perturbation_footprint_report_path,
            local_merge_parent_summary_path,
            qf_points_path,
            run_rows_path,
            rank_summary_path,
            transition_summary_path,
            report_path,
            depth_attribution_report_path,
            target_parent_events_path,
            target_parent_contrast_path,
            parent_causal_window_report_path,
            summary_path,
        ):
            if path.exists():
                path.unlink()

    diagnostics = _load_candidate_diagnostics(input_dir)
    candidate_indices = _parse_int_list(args.candidate_indices)
    candidate_rows = _candidate_rows_for_case(
        diagnostics,
        case_contains=args.case_contains,
        seed=args.seed,
        candidate_indices=candidate_indices,
    )
    case = str(candidate_rows.iloc[0]["case"])
    graph_dir = _find_graph_dir(_parse_graph_dirs(args.graph_dirs), case)
    arrays = _load_graph_arrays(graph_dir)
    graph = build_leiden_graph(
        edges_src=arrays.src,
        edges_dst=arrays.dst,
        edges_weight=arrays.weight,
        n_nodes=int(arrays.node_weights.shape[0]),
        node_weights=arrays.node_weights,
    )
    replay_iterations = _parse_positive_int_list(args.replay_iterations)

    with _trace_disabled_context():
        baseline = graph.run_leiden(
            resolution=args.resolution,
            seed=args.seed,
            n_iterations=args.baseline_iterations,
            randomness=args.randomness,
        )
    baseline_membership = np.asarray(baseline.membership, dtype=np.uint64)
    run_rows: list[dict[str, Any]] = []
    footprint_event_rows: list[dict[str, Any]] = []
    footprint_summary_rows: list[dict[str, Any]] = []

    with _trace_file_context(quality_trace_path, trajectory_trace_path, resume=args.resume):
        for _, candidate in candidate_rows.iterrows():
            candidate_index = int(candidate["candidate_index"])
            group_nodes, reconstruction = _reconstruct_external_group(
                src=arrays.src,
                dst=arrays.dst,
                weight=arrays.weight,
                membership=baseline_membership,
                node_weights=arrays.node_weights,
                source_cluster=int(candidate["source_cluster"]),
                target_cluster=int(candidate["target_cluster"]),
            )
            neighbor_nodes = _one_hop_neighbor_nodes(arrays.src, arrays.dst, group_nodes)
            hop_sets = _hop_distance_sets(arrays.src, arrays.dst, group_nodes, max_hop=2)
            perturbed = baseline_membership.copy()
            perturbed[group_nodes] = np.uint64(int(candidate["target_cluster"]))
            perturbed = _compact_membership(perturbed)
            candidate_seed = args.seed + args.perturb_seed_offset + candidate_index
            for n_iterations in replay_iterations:
                run_id = _run_id(case, args.seed, candidate_index, n_iterations)
                reference_membership = baseline_membership
                reference_quality = float(baseline.quality)
                contrast_kind = "baseline"
                contrast_run_id = "baseline"
                if args.footprint_extra_contrast:
                    contrast_run_id = f"{run_id}|extra"
                    print(f"[trajectory] {run_id}: extra contrast", flush=True)
                    with _trace_disabled_context():
                        extra_result = graph.run_leiden(
                            resolution=args.resolution,
                            seed=candidate_seed,
                            n_iterations=n_iterations,
                            randomness=args.randomness,
                            initial_membership=baseline_membership,
                        )
                    reference_membership = np.asarray(
                        extra_result.membership,
                        dtype=np.uint64,
                    )
                    reference_quality = float(extra_result.quality)
                    contrast_kind = "extra"
                print(f"[trajectory] {run_id}: replay", flush=True)
                t0 = time.perf_counter()
                with _focused_trace_run_context(
                    run_id,
                    target_max_weight=args.target_max_weight,
                    target_nodes=group_nodes,
                    neighbor_nodes=neighbor_nodes,
                ):
                    result = graph.run_leiden(
                        resolution=args.resolution,
                        seed=candidate_seed,
                        n_iterations=n_iterations,
                        randomness=args.randomness,
                        initial_membership=perturbed,
                    )
                elapsed = time.perf_counter() - t0
                footprint_events, footprint_summary = build_perturbation_footprint_rows(
                    case=case,
                    seed=args.seed,
                    candidate_index=candidate_index,
                    replay_iterations=n_iterations,
                    run_id=run_id,
                    contrast_kind=contrast_kind,
                    contrast_run_id=contrast_run_id,
                    source_cluster=int(candidate["source_cluster"]),
                    target_cluster=int(candidate["target_cluster"]),
                    target_nodes=group_nodes,
                    hop_sets=hop_sets,
                    baseline_membership=baseline_membership,
                    reference_membership=reference_membership,
                    perturb_membership=np.asarray(result.membership, dtype=np.uint64),
                    baseline_quality=float(baseline.quality),
                    reference_quality=reference_quality,
                    perturb_quality=float(result.quality),
                )
                footprint_event_rows.extend(footprint_events)
                footprint_summary_rows.append(footprint_summary)
                run_rows.append(
                    {
                        "case": case,
                        "seed": args.seed,
                        "candidate_index": candidate_index,
                        "source_cluster": int(candidate["source_cluster"]),
                        "target_cluster": int(candidate["target_cluster"]),
                        "group_kind": candidate.get("group_kind", ""),
                        "group_count": int(candidate.get("group_count", 0)),
                        "p1_rank_label": _finite_float(candidate.get("p1_rank"), math.nan),
                        "p5_rank_label": _finite_float(candidate.get("p5_rank"), math.nan),
                        "p1_delta_q_label": _finite_float(candidate.get("p1_delta_q"), math.nan),
                        "p5_delta_q_label": _finite_float(candidate.get("p5_delta_q"), math.nan),
                        "is_full_p5_winner_label": bool(candidate.get("is_full_p5_winner", False)),
                        "replay_iterations": n_iterations,
                        "run_id": run_id,
                        "baseline_quality": float(baseline.quality),
                        "quality": float(result.quality),
                        "delta_q": float(result.quality - baseline.quality),
                        "elapsed_sec": elapsed,
                        "n_clusters": int(result.n_clusters),
                        "candidate_seed": candidate_seed,
                        **reconstruction,
                    }
                )

    run_frame = pd.DataFrame(run_rows)
    run_frame.to_csv(run_rows_path, index=False)
    perturbation_footprint_events = pd.DataFrame(
        footprint_event_rows,
        columns=PERTURBATION_FOOTPRINT_EVENT_COLUMNS,
    )
    perturbation_footprint_summary = pd.DataFrame(
        footprint_summary_rows,
        columns=PERTURBATION_FOOTPRINT_SUMMARY_COLUMNS,
    )
    if not perturbation_footprint_summary.empty:
        perturbation_footprint_summary = perturbation_footprint_summary.sort_values(
            ["replay_iterations", "candidate_index", "contrast_kind"],
            kind="mergesort",
        )
    if not perturbation_footprint_events.empty:
        perturbation_footprint_events = perturbation_footprint_events.sort_values(
            ["replay_iterations", "candidate_index", "hop_bucket"],
            kind="mergesort",
        )
    perturbation_footprint_events.to_csv(
        perturbation_footprint_events_path,
        index=False,
    )
    perturbation_footprint_summary.to_csv(
        perturbation_footprint_summary_path,
        index=False,
    )
    write_perturbation_footprint_report(
        perturbation_footprint_report_path,
        footprint_summary=perturbation_footprint_summary,
        extra_contrast=bool(args.footprint_extra_contrast),
    )
    phase_frame = _extract_phase_checkpoints(trajectory_trace_path, phase_path)
    local_move_gain = build_local_move_gain_rows(phase_frame, run_frame)
    local_move_gain.to_csv(local_move_gain_path, index=False)
    local_move_margin_summary, local_merge_parent_summary, attribution = (
        write_trace_margin_outputs(
            trajectory_path=trajectory_trace_path,
            run_rows=run_frame,
            local_move_gain=local_move_gain,
            local_move_margin_summary_path=local_move_margin_summary_path,
            local_merge_parent_summary_path=local_merge_parent_summary_path,
            depth_attribution_report_path=depth_attribution_report_path,
        )
    )
    local_move_focus_events, local_move_focus_summary, movement_attribution = (
        write_local_move_focus_outputs(
            trajectory_path=trajectory_trace_path,
            run_rows=run_frame,
            local_move_gain=local_move_gain,
            local_move_focus_events_path=local_move_focus_events_path,
            local_move_focus_summary_path=local_move_focus_summary_path,
            local_move_movement_report_path=local_move_movement_report_path,
        )
    )
    baseline_by_run_id = {
        str(row["run_id"]): float(row["baseline_quality"])
        for _, row in run_frame.iterrows()
    }
    qf_points = _quality_points(quality_trace_path, phase_frame, baseline_by_run_id)
    if not qf_points.empty:
        run_meta = run_frame[
            ["run_id", "case", "seed", "candidate_index", "replay_iterations"]
        ]
        qf_points = qf_points.merge(run_meta, on="run_id", how="left", suffixes=("", "_meta"))
    qf_points.to_csv(qf_points_path, index=False)
    rank_summary = build_rank_summary(run_frame, candidate_rows)
    rank_summary.to_csv(rank_summary_path, index=False)
    transition_summary = build_transition_summary(rank_summary)
    transition_summary.to_csv(transition_summary_path, index=False)
    write_report(report_path, rank_summary, transition_summary, local_move_gain)
    parent_drilldown = write_parent_drilldown_outputs(
        trajectory_path=trajectory_trace_path,
        run_rows=run_frame,
        phase_frame=phase_frame,
        local_move_gain=local_move_gain,
        target_parent_events_path=target_parent_events_path,
        target_parent_contrast_path=target_parent_contrast_path,
        parent_causal_window_report_path=parent_causal_window_report_path,
        target_parent_ids=_parse_parent_ids(args.target_parent_ids),
    )

    paths = {
        "run_rows": run_rows_path,
        "phase_checkpoints": phase_path,
        "local_move_gain_rows": local_move_gain_path,
        "local_move_margin_summary": local_move_margin_summary_path,
        "local_move_focus_events": local_move_focus_events_path,
        "local_move_focus_summary": local_move_focus_summary_path,
        "local_move_movement_attribution_report": local_move_movement_report_path,
        "perturbation_footprint_events": perturbation_footprint_events_path,
        "perturbation_footprint_summary": perturbation_footprint_summary_path,
        "perturbation_footprint_report": perturbation_footprint_report_path,
        "local_merge_parent_summary": local_merge_parent_summary_path,
        "qf_i_k_points": qf_points_path,
        "rank_summary": rank_summary_path,
        "transition_summary": transition_summary_path,
        "report": report_path,
        "depth_attribution_report": depth_attribution_report_path,
        "target_parent_events": target_parent_events_path,
        "target_parent_contrast": target_parent_contrast_path,
        "parent_causal_window_report": parent_causal_window_report_path,
        "quality_trace": quality_trace_path,
        "trajectory_trace": trajectory_trace_path,
    }
    summary = {
        "schema": "leiden_multifidelity_candidate_trajectory.v1",
        "case": case,
        "seed": args.seed,
        "candidate_indices": sorted(run_frame["candidate_index"].unique().tolist()),
        "replay_iterations": replay_iterations,
        "depth_attribution_classification": attribution["classification"],
        "local_move_movement_attribution_classification": movement_attribution[
            "classification"
        ],
        "parent_causal_window_classification": parent_drilldown["classification"],
        "local_move_margin_summary_rows": int(len(local_move_margin_summary)),
        "local_move_focus_event_rows": int(len(local_move_focus_events)),
        "local_move_focus_summary_rows": int(len(local_move_focus_summary)),
        "perturbation_footprint_event_rows": int(len(perturbation_footprint_events)),
        "perturbation_footprint_summary_rows": int(len(perturbation_footprint_summary)),
        "perturbation_footprint_extra_contrast": bool(args.footprint_extra_contrast),
        "local_merge_parent_summary_rows": int(len(local_merge_parent_summary)),
        "target_parent_event_rows": int(parent_drilldown["target_parent_event_rows"]),
        "target_parent_contrast_rows": int(parent_drilldown["target_parent_contrast_rows"]),
        "paths": {name: str(path.relative_to(REPO_ROOT)) for name, path in paths.items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    paths["summary"] = summary_path
    return paths


def _read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _run_meta_by_id(run_rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if run_rows.empty or "run_id" not in run_rows.columns:
        return {}
    meta_columns = [
        "run_id",
        "case",
        "seed",
        "candidate_index",
        "replay_iterations",
    ]
    available = [column for column in meta_columns if column in run_rows.columns]
    meta: dict[str, dict[str, Any]] = {}
    for _, row in run_rows[available].drop_duplicates("run_id").iterrows():
        run_id = str(row.get("run_id", ""))
        if not run_id:
            continue
        meta[run_id] = {
            "case": row.get("case", ""),
            "seed": _safe_int(row.get("seed"), 0),
            "candidate_index": _safe_int(row.get("candidate_index"), -1),
            "replay_iterations": _safe_int(row.get("replay_iterations"), 0),
        }
    return meta


def build_perturbation_footprint_rows(
    *,
    case: str,
    seed: int,
    candidate_index: int,
    replay_iterations: int,
    run_id: str,
    contrast_kind: str,
    contrast_run_id: str,
    source_cluster: int,
    target_cluster: int,
    target_nodes: Any,
    hop_sets: dict[int, set[int]],
    baseline_membership: Any,
    reference_membership: Any,
    perturb_membership: Any,
    baseline_quality: float,
    reference_quality: float,
    perturb_quality: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline = np.asarray(baseline_membership, dtype=np.uint64)
    reference = np.asarray(reference_membership, dtype=np.uint64)
    perturb = np.asarray(perturb_membership, dtype=np.uint64)
    if reference.shape != perturb.shape:
        raise ValueError(
            f"membership shapes differ for {run_id}: reference={reference.shape}, perturb={perturb.shape}"
        )
    if baseline.shape != perturb.shape:
        raise ValueError(
            f"baseline shape differs for {run_id}: baseline={baseline.shape}, perturb={perturb.shape}"
        )

    aligned_perturb = _align_membership_labels(reference, perturb)
    changed_nodes = np.flatnonzero(reference != aligned_perturb)
    total_nodes = int(perturb.shape[0])
    changed_total = int(changed_nodes.size)
    bucket_masks = _footprint_bucket_masks(changed_nodes, hop_sets)
    hop0_count = int(bucket_masks["0_target"].sum())
    hop1_count = int(bucket_masks["1_neighbor"].sum())
    hop2_count = int(bucket_masks["2_two_hop"].sum())
    hop3plus_count = int(bucket_masks["3plus"].sum())
    within1 = _safe_ratio(hop0_count + hop1_count, changed_total)
    within2 = _safe_ratio(hop0_count + hop1_count + hop2_count, changed_total)
    source_target_count = _source_target_changed_count(
        baseline,
        changed_nodes,
        source_cluster=source_cluster,
        target_cluster=target_cluster,
    )
    source_target_fraction = _safe_ratio(source_target_count, changed_total)
    max_hop = _max_hop_reached(
        hop0_count=hop0_count,
        hop1_count=hop1_count,
        hop2_count=hop2_count,
        hop3plus_count=hop3plus_count,
    )
    summary: dict[str, Any] = {
        "case": case,
        "seed": int(seed),
        "candidate_index": int(candidate_index),
        "replay_iterations": int(replay_iterations),
        "run_id": run_id,
        "contrast_kind": contrast_kind,
        "contrast_run_id": contrast_run_id,
        "source_cluster": int(source_cluster),
        "target_cluster": int(target_cluster),
        "target_node_count": int(len(hop_sets.get(0, set()))),
        "neighbor_node_count": int(len(hop_sets.get(1, set()))),
        "two_hop_node_count": int(len(hop_sets.get(2, set()))),
        "changed_nodes_total": changed_total,
        "changed_nodes_fraction": _safe_ratio(changed_total, total_nodes),
        "changed_hop0_count": hop0_count,
        "changed_hop1_count": hop1_count,
        "changed_hop2_count": hop2_count,
        "changed_hop3plus_count": hop3plus_count,
        "fraction_changed_within_1hop": within1,
        "fraction_changed_within_2hop": within2,
        "changed_source_or_target_initial_count": source_target_count,
        "fraction_changed_source_or_target_initial": source_target_fraction,
        "changed_reference_clusters_total": _unique_cluster_count(reference, changed_nodes),
        "changed_perturb_clusters_total": _unique_cluster_count(
            aligned_perturb,
            changed_nodes,
        ),
        "max_hop_reached": max_hop,
        "classification": "",
        "classification_confidence": "causal" if contrast_kind == "extra" else "descriptive",
        "baseline_quality": _finite_event_float(baseline_quality),
        "reference_quality": _finite_event_float(reference_quality),
        "perturb_quality": _finite_event_float(perturb_quality),
        "delta_q_vs_baseline": _finite_event_float(perturb_quality - baseline_quality),
        "delta_q_vs_reference": _finite_event_float(perturb_quality - reference_quality),
        "sample_changed_node_ids": _sample_node_ids(changed_nodes),
    }
    summary["classification"] = classify_perturbation_footprint(summary)

    event_rows: list[dict[str, Any]] = []
    for bucket in ("0_target", "1_neighbor", "2_two_hop", "3plus"):
        mask = bucket_masks[bucket]
        bucket_nodes = changed_nodes[mask]
        event_rows.append(
            {
                "case": case,
                "seed": int(seed),
                "candidate_index": int(candidate_index),
                "replay_iterations": int(replay_iterations),
                "run_id": run_id,
                "contrast_kind": contrast_kind,
                "contrast_run_id": contrast_run_id,
                "hop_bucket": bucket,
                "changed_node_count": int(bucket_nodes.size),
                "changed_node_fraction": _safe_ratio(int(bucket_nodes.size), changed_total),
                "changed_reference_clusters_count": _unique_cluster_count(reference, bucket_nodes),
                "changed_perturb_clusters_count": _unique_cluster_count(
                    aligned_perturb,
                    bucket_nodes,
                ),
                "sample_node_ids": _sample_node_ids(bucket_nodes),
            }
        )
    return event_rows, summary


def classify_perturbation_footprint(row: dict[str, Any] | pd.Series) -> str:
    changed_total = _safe_int(row.get("changed_nodes_total"), 0)
    if changed_total <= 0:
        return "unknown_insufficient_trace"
    within1 = _finite_event_float(row.get("fraction_changed_within_1hop"), 0.0)
    within2 = _finite_event_float(row.get("fraction_changed_within_2hop"), 0.0)
    source_target = _finite_event_float(
        row.get("fraction_changed_source_or_target_initial"),
        0.0,
    )
    if within1 >= 0.95:
        return "local_1hop"
    if within2 >= 0.95:
        return "local_2hop"
    if source_target >= 0.95:
        return "parent_local"
    if str(row.get("contrast_kind", "")) != "extra":
        return "unknown_insufficient_trace"
    return "diffuse_global"


def write_perturbation_footprint_report(
    path: Path,
    *,
    footprint_summary: pd.DataFrame,
    extra_contrast: bool,
) -> None:
    lines = [
        "# Perturbation Footprint Report",
        "",
        "- Scope: membership-change footprint of candidate perturbation replays.",
        f"- Contrast: {'extra replay from baseline membership' if extra_contrast else 'baseline membership only'}",
        "- Extra contrast seed: same seed as the perturbed candidate replay.",
        "- Primary use: decide whether a local or parent-local perturbation proxy is plausible before changing the algorithm.",
        "",
    ]
    if footprint_summary.empty:
        lines.append("No footprint rows were produced.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.extend(
        [
            "## Summary",
            "",
            "| candidate | replay | contrast | class | changed | within 1-hop | within 2-hop | source/target init | delta q ref | sample nodes |",
            "|---:|---:|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in footprint_summary.sort_values(
        ["replay_iterations", "candidate_index", "contrast_kind"],
        kind="mergesort",
    ).iterrows():
        lines.append(
            "| {candidate} | {replay} | {contrast} | `{classification}` | {changed} | {within1} | {within2} | {source_target} | {delta_ref} | {sample} |".format(
                candidate=_table_int(row, "candidate_index"),
                replay=_table_int(row, "replay_iterations"),
                contrast=row.get("contrast_kind", ""),
                classification=row.get("classification", ""),
                changed=_table_int(row, "changed_nodes_total"),
                within1=_format_float(row.get("fraction_changed_within_1hop"), 3),
                within2=_format_float(row.get("fraction_changed_within_2hop"), 3),
                source_target=_format_float(
                    row.get("fraction_changed_source_or_target_initial"),
                    3,
                ),
                delta_ref=_signed_format(row.get("delta_q_vs_reference")),
                sample=row.get("sample_changed_node_ids", ""),
            )
        )

    class_counts = (
        footprint_summary["classification"].astype(str).value_counts().sort_index()
    )
    lines.extend(["", "## Classification Counts", ""])
    for classification, count in class_counts.items():
        lines.append(f"- `{classification}`: {int(count)}")

    lines.extend(
        [
            "",
            "## Rule",
            "",
            "- `local_1hop`: at least 95% of changed nodes are the perturbed target group or its one-hop neighbors.",
            "- `local_2hop`: at least 95% of changed nodes are within two hops.",
            "- `parent_local`: changes are not hop-local, but at least 95% stay inside the original source/target clusters.",
            "- `diffuse_global`: extra contrast shows broad change outside both neighborhood and parent-local scopes.",
            "- `unknown_insufficient_trace`: no membership change was observed, or baseline-only contrast is too confounded to call diffuse.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _footprint_bucket_masks(
    changed_nodes: np.ndarray,
    hop_sets: dict[int, set[int]],
) -> dict[str, np.ndarray]:
    node_count = int(changed_nodes.size)
    assigned = np.zeros(node_count, dtype=bool)
    masks: dict[str, np.ndarray] = {}
    for hop, bucket in ((0, "0_target"), (1, "1_neighbor"), (2, "2_two_hop")):
        nodes = hop_sets.get(hop, set())
        if node_count and nodes:
            mask = np.isin(changed_nodes, np.asarray(sorted(nodes), dtype=np.int64))
        else:
            mask = np.zeros(node_count, dtype=bool)
        mask &= ~assigned
        masks[bucket] = mask
        assigned |= mask
    masks["3plus"] = ~assigned
    return masks


def _align_membership_labels(reference: np.ndarray, observed: np.ndarray) -> np.ndarray:
    if reference.shape != observed.shape:
        raise ValueError(
            f"membership shapes differ: reference={reference.shape}, observed={observed.shape}"
        )
    if int(observed.size) == 0:
        return observed.copy()
    mapping = _majority_label_mapping(reference, observed)
    max_label = int(np.max(observed))
    if max_label <= 10_000_000:
        remap = np.arange(max_label + 1, dtype=reference.dtype)
        for observed_label, reference_label in mapping.items():
            remap[int(observed_label)] = reference_label
        return remap[observed.astype(np.int64, copy=False)]
    return np.fromiter(
        (
            mapping.get(int(observed_label), int(observed_label))
            for observed_label in observed
        ),
        dtype=reference.dtype,
        count=int(observed.size),
    )


def _majority_label_mapping(
    reference: np.ndarray,
    observed: np.ndarray,
) -> dict[int, int]:
    if int(reference.size) == 0:
        return {}
    ref_u = reference.astype(np.uint64, copy=False)
    obs_u = observed.astype(np.uint64, copy=False)
    if (
        int(reference.size) <= 5_000_000
        and int(ref_u.max()) < (1 << 32)
        and int(obs_u.max()) < (1 << 32)
    ):
        pair_keys = (obs_u << np.uint64(32)) | ref_u
        unique_keys, counts = np.unique(pair_keys, return_counts=True)
        best: dict[int, tuple[int, int]] = {}
        for key, count in zip(unique_keys, counts, strict=False):
            observed_label = int(key >> np.uint64(32))
            reference_label = int(key & np.uint64((1 << 32) - 1))
            _update_best_label_mapping(
                best,
                observed_label=observed_label,
                reference_label=reference_label,
                count=int(count),
            )
        return {observed_label: item[0] for observed_label, item in best.items()}

    best_counts: dict[tuple[int, int], int] = {}
    for reference_label, observed_label in zip(reference, observed, strict=False):
        key = (int(observed_label), int(reference_label))
        best_counts[key] = best_counts.get(key, 0) + 1
    best: dict[int, tuple[int, int]] = {}
    for (observed_label, reference_label), count in best_counts.items():
        _update_best_label_mapping(
            best,
            observed_label=observed_label,
            reference_label=reference_label,
            count=count,
        )
    return {observed_label: item[0] for observed_label, item in best.items()}


def _update_best_label_mapping(
    best: dict[int, tuple[int, int]],
    *,
    observed_label: int,
    reference_label: int,
    count: int,
) -> None:
    current = best.get(observed_label)
    if current is None or count > current[1] or (
        count == current[1] and reference_label < current[0]
    ):
        best[observed_label] = (reference_label, count)


def _source_target_changed_count(
    baseline_membership: np.ndarray,
    changed_nodes: np.ndarray,
    *,
    source_cluster: int,
    target_cluster: int,
) -> int:
    if int(changed_nodes.size) == 0:
        return 0
    clusters = np.asarray([source_cluster, target_cluster], dtype=baseline_membership.dtype)
    return int(np.isin(baseline_membership[changed_nodes], clusters).sum())


def _unique_cluster_count(membership: np.ndarray, nodes: np.ndarray) -> int:
    if int(nodes.size) == 0:
        return 0
    return int(np.unique(membership[nodes]).size)


def _max_hop_reached(
    *,
    hop0_count: int,
    hop1_count: int,
    hop2_count: int,
    hop3plus_count: int,
) -> int:
    if hop3plus_count > 0:
        return 3
    if hop2_count > 0:
        return 2
    if hop1_count > 0:
        return 1
    if hop0_count > 0:
        return 0
    return -1


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    denom = float(denominator)
    if denom <= 0.0 or not math.isfinite(denom):
        return math.nan
    return float(numerator) / denom


def _sample_node_ids(nodes: Any, limit: int = TOP_ID_LIMIT) -> str:
    if nodes is None:
        return ""
    values = np.asarray(nodes, dtype=np.int64)
    if values.size == 0:
        return ""
    return ",".join(str(int(node)) for node in values[:limit].tolist())


def build_local_move_margin_summary(
    events: Any,
    run_rows: pd.DataFrame,
) -> pd.DataFrame:
    run_meta = _run_meta_by_id(run_rows)
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") != "local_move_margin":
            continue
        run_id = str(event.get("run_id", ""))
        meta = run_meta.get(run_id, {})
        rows.append(
            {
                "case": meta.get("case", ""),
                "seed": meta.get("seed", 0),
                "candidate_index": meta.get("candidate_index", -1),
                "replay_iterations": meta.get("replay_iterations", 0),
                "run_id": run_id,
                "iteration": _safe_int(event.get("iteration"), 0),
                "depth": _safe_int(event.get("depth"), 0),
                "rank": _safe_int(event.get("rank"), 0),
                "node": _safe_int(event.get("node"), -1),
                "margin": _finite_event_float(event.get("margin")),
                "best_increment": _finite_event_float(event.get("best_increment")),
                "second_increment": _finite_event_float(event.get("second_increment")),
                "moved": _truthy(event.get("moved")),
            }
        )
    if not rows:
        return pd.DataFrame(columns=LOCAL_MOVE_MARGIN_SUMMARY_COLUMNS)

    frame = pd.DataFrame(rows)
    out: list[dict[str, Any]] = []
    group_cols = [
        "case",
        "seed",
        "candidate_index",
        "replay_iterations",
        "run_id",
        "iteration",
        "depth",
    ]
    for key, group in frame.groupby(group_cols, dropna=False, sort=True):
        case, seed, candidate_index, replay_iterations, run_id, iteration, depth = key
        margins = _finite_values(group["margin"])
        best = _finite_values(group["best_increment"])
        second = _finite_values(group["second_increment"])
        moved = group[group["moved"].map(bool)]
        out.append(
            {
                "case": case,
                "seed": int(seed),
                "candidate_index": int(candidate_index),
                "replay_iterations": int(replay_iterations),
                "run_id": run_id,
                "iteration": int(iteration),
                "depth": int(depth),
                "event_count": int(len(group)),
                "moved_count": int(group["moved"].map(bool).sum()),
                "near_zero_margin_count": int(sum(value <= 1.0e-12 for value in margins)),
                "margin_min": _quantile(margins, 0.0),
                "margin_p10": _quantile(margins, 0.10),
                "margin_p50": _quantile(margins, 0.50),
                "best_increment_min": _quantile(best, 0.0),
                "best_increment_p50": _quantile(best, 0.50),
                "best_increment_max": _quantile(best, 1.0),
                "second_increment_min": _quantile(second, 0.0),
                "second_increment_p50": _quantile(second, 0.50),
                "second_increment_max": _quantile(second, 1.0),
                "top_low_margin_node_ids": _top_node_ids(group),
                "top_moved_low_margin_node_ids": _top_node_ids(moved),
            }
        )
    return pd.DataFrame(out, columns=LOCAL_MOVE_MARGIN_SUMMARY_COLUMNS).sort_values(
        ["replay_iterations", "candidate_index", "iteration", "depth"],
        kind="mergesort",
    )


def build_local_merge_parent_summary(
    events: Any,
    run_rows: pd.DataFrame,
) -> pd.DataFrame:
    run_meta = _run_meta_by_id(run_rows)
    aggregates: dict[tuple[Any, ...], dict[str, Any]] = {}
    for event in events:
        if event.get("event") != "local_merge_margin_summary":
            continue
        run_id = str(event.get("run_id", ""))
        meta = run_meta.get(run_id, {})
        key = (
            meta.get("case", ""),
            meta.get("seed", 0),
            meta.get("candidate_index", -1),
            meta.get("replay_iterations", 0),
            run_id,
            _safe_int(event.get("iteration"), 0),
            _safe_int(event.get("depth"), 0),
        )
        row = aggregates.setdefault(
            key,
            {
                "case": key[0],
                "seed": key[1],
                "candidate_index": key[2],
                "replay_iterations": key[3],
                "run_id": key[4],
                "iteration": key[5],
                "depth": key[6],
                "parent_row_count": 0,
                "decision_count": 0.0,
                "low_margin_count": 0.0,
                "changed_count": 0.0,
                "min_margin_min": math.nan,
                "p10_margin_min": math.nan,
                "p50_margin_min": math.nan,
                "largest_child_fraction_max": math.nan,
                "parent_totals": {},
            },
        )
        decision = _finite_event_float(event.get("decision_count"), 0.0)
        low_margin = _finite_event_float(event.get("low_margin_decision_count"), 0.0)
        changed = _finite_event_float(event.get("changed_decision_count"), 0.0)
        min_margin = _finite_event_float(event.get("min_margin"))
        p10_margin = _finite_event_float(event.get("p10_margin"))
        p50_margin = _finite_event_float(event.get("p50_margin"))
        largest_child = _finite_event_float(event.get("largest_child_fraction"))
        row["parent_row_count"] += 1
        row["decision_count"] += decision
        row["low_margin_count"] += low_margin
        row["changed_count"] += changed
        row["min_margin_min"] = _min_finite(row["min_margin_min"], min_margin)
        row["p10_margin_min"] = _min_finite(row["p10_margin_min"], p10_margin)
        row["p50_margin_min"] = _min_finite(row["p50_margin_min"], p50_margin)
        row["largest_child_fraction_max"] = _max_finite(
            row["largest_child_fraction_max"], largest_child
        )

        parent_id = str(event.get("parent_id", ""))
        parent_totals = row["parent_totals"].setdefault(
            parent_id,
            {
                "decision_count": 0.0,
                "low_margin_decision_count": 0.0,
                "changed_decision_count": 0.0,
                "min_margin": math.nan,
                "largest_child_fraction": math.nan,
            },
        )
        parent_totals["decision_count"] += decision
        parent_totals["low_margin_decision_count"] += low_margin
        parent_totals["changed_decision_count"] += changed
        parent_totals["min_margin"] = _min_finite(
            parent_totals["min_margin"], min_margin
        )
        parent_totals["largest_child_fraction"] = _max_finite(
            parent_totals["largest_child_fraction"], largest_child
        )
    if not aggregates:
        return pd.DataFrame(columns=LOCAL_MERGE_PARENT_SUMMARY_COLUMNS)

    out: list[dict[str, Any]] = []
    for key in sorted(
        aggregates,
        key=lambda item: (int(item[3]), int(item[2]), int(item[5]), int(item[6])),
    ):
        row = aggregates[key]
        parent_totals = row.pop("parent_totals")
        row["top_low_margin_parent_ids"] = _top_parent_ids_from_totals(
            parent_totals,
            metric="low_margin_decision_count",
            direction="desc",
            positive_only=True,
        )
        row["top_decision_parent_ids"] = _top_parent_ids_from_totals(
            parent_totals,
            metric="decision_count",
            direction="desc",
            positive_only=True,
        )
        row["top_changed_parent_ids"] = _top_parent_ids_from_totals(
            parent_totals,
            metric="changed_decision_count",
            direction="desc",
            positive_only=True,
        )
        row["top_small_margin_parent_ids"] = _top_parent_ids_from_totals(
            parent_totals,
            metric="min_margin",
            direction="asc",
            finite_only=True,
        )
        row["top_largest_child_fraction_parent_ids"] = _top_parent_ids_from_totals(
            parent_totals,
            metric="largest_child_fraction",
            direction="desc",
            finite_only=True,
        )
        out.append({column: row.get(column, "") for column in LOCAL_MERGE_PARENT_SUMMARY_COLUMNS})
    return pd.DataFrame(out, columns=LOCAL_MERGE_PARENT_SUMMARY_COLUMNS).sort_values(
        ["replay_iterations", "candidate_index", "iteration", "depth"],
        kind="mergesort",
    )


def write_trace_margin_outputs(
    *,
    trajectory_path: Path,
    run_rows: pd.DataFrame,
    local_move_gain: pd.DataFrame,
    local_move_margin_summary_path: Path,
    local_merge_parent_summary_path: Path,
    depth_attribution_report_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    local_move_margin_summary = build_local_move_margin_summary(
        _read_jsonl(trajectory_path),
        run_rows,
    )
    local_move_margin_summary.to_csv(local_move_margin_summary_path, index=False)
    local_merge_parent_summary = build_local_merge_parent_summary(
        _read_jsonl(trajectory_path),
        run_rows,
    )
    local_merge_parent_summary.to_csv(local_merge_parent_summary_path, index=False)
    attribution = classify_depth_attribution(
        local_move_margin_summary=local_move_margin_summary,
        local_merge_parent_summary=local_merge_parent_summary,
        local_move_gain=local_move_gain,
    )
    write_depth_attribution_report(
        depth_attribution_report_path,
        attribution=attribution,
        local_move_margin_summary=local_move_margin_summary,
        local_merge_parent_summary=local_merge_parent_summary,
    )
    return local_move_margin_summary, local_merge_parent_summary, attribution


def write_local_move_focus_outputs(
    *,
    trajectory_path: Path,
    run_rows: pd.DataFrame,
    local_move_gain: pd.DataFrame,
    local_move_focus_events_path: Path,
    local_move_focus_summary_path: Path,
    local_move_movement_report_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    local_move_focus_events = build_local_move_focus_event_rows(
        _read_jsonl(trajectory_path),
        run_rows=run_rows,
        local_move_gain=local_move_gain,
    )
    local_move_focus_events.to_csv(local_move_focus_events_path, index=False)
    local_move_focus_summary = build_local_move_focus_summary(local_move_focus_events)
    local_move_focus_summary.to_csv(local_move_focus_summary_path, index=False)
    attribution = classify_local_move_movement_attribution(
        local_move_focus_summary=local_move_focus_summary,
        local_move_gain=local_move_gain,
    )
    write_local_move_movement_report(
        local_move_movement_report_path,
        attribution=attribution,
        local_move_focus_summary=local_move_focus_summary,
    )
    return local_move_focus_events, local_move_focus_summary, attribution


def build_local_move_focus_event_rows(
    events: Any,
    *,
    run_rows: pd.DataFrame,
    local_move_gain: pd.DataFrame,
) -> pd.DataFrame:
    run_meta = _run_meta_by_id(run_rows)
    gain_lookup = _local_move_gain_lookup(local_move_gain)
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") != "local_move_focus_node":
            continue
        run_id = str(event.get("run_id", ""))
        meta = run_meta.get(run_id, {})
        iteration = _safe_int(event.get("iteration"), 0)
        depth = _safe_int(event.get("depth"), 0)
        rows.append(
            {
                "case": meta.get("case", ""),
                "seed": meta.get("seed", 0),
                "candidate_index": meta.get("candidate_index", -1),
                "replay_iterations": meta.get("replay_iterations", 0),
                "run_id": run_id,
                "iteration": iteration,
                "depth": depth,
                "node": _safe_int(event.get("node"), -1),
                "role": str(event.get("role", "")),
                "current_cluster": _safe_int(event.get("current_cluster"), -1),
                "best_cluster": _safe_int(event.get("best_cluster"), -1),
                "second_cluster": _nullable_int(event.get("second_cluster")),
                "best_increment": _finite_event_float(event.get("best_increment")),
                "second_increment": _finite_event_float(event.get("second_increment")),
                "margin": _finite_event_float(event.get("margin")),
                "moved": _truthy(event.get("moved")),
                "quality_gain_since_previous_local_move": gain_lookup.get(
                    (run_id, iteration, depth),
                    math.nan,
                ),
            }
        )
    if not rows:
        return pd.DataFrame(columns=LOCAL_MOVE_FOCUS_EVENT_COLUMNS)
    return pd.DataFrame(rows, columns=LOCAL_MOVE_FOCUS_EVENT_COLUMNS).sort_values(
        ["replay_iterations", "candidate_index", "iteration", "depth", "role", "node"],
        kind="mergesort",
    )


def build_local_move_focus_summary(local_move_focus_events: pd.DataFrame) -> pd.DataFrame:
    if local_move_focus_events.empty:
        return pd.DataFrame(columns=LOCAL_MOVE_FOCUS_SUMMARY_COLUMNS)
    rows: list[dict[str, Any]] = []
    moved_sets: dict[tuple[str, int, int], set[int]] = {}
    group_cols = [
        "case",
        "seed",
        "candidate_index",
        "replay_iterations",
        "run_id",
        "iteration",
        "depth",
    ]
    for key, group in local_move_focus_events.groupby(group_cols, dropna=False, sort=True):
        case, seed, candidate_index, replay_iterations, run_id, iteration, depth = key
        target = group[group["role"].astype(str).eq("target")]
        neighbor = group[group["role"].astype(str).eq("neighbor")]
        moved = group[group["moved"].map(bool)]
        best = _finite_values(group["best_increment"])
        second = _finite_values(group["second_increment"])
        moved_set = {
            _safe_int(node, -1)
            for node in moved["node"].tolist()
            if _safe_int(node, -1) >= 0
        }
        moved_sets[(str(run_id), int(iteration), int(depth))] = moved_set
        rows.append(
            {
                "case": case,
                "seed": int(seed),
                "candidate_index": int(candidate_index),
                "replay_iterations": int(replay_iterations),
                "run_id": run_id,
                "iteration": int(iteration),
                "depth": int(depth),
                "quality_gain_since_previous_local_move": _finite_min(
                    group["quality_gain_since_previous_local_move"]
                ),
                "target_event_count": int(len(target)),
                "target_moved_count": int(target["moved"].map(bool).sum()),
                "target_moved_node_ids": _node_ids(target[target["moved"].map(bool)]),
                "target_margin_min": _finite_min(target["margin"]) if not target.empty else math.nan,
                "target_margin_p50": _quantile(_finite_values(target["margin"]), 0.50),
                "neighbor_event_count": int(len(neighbor)),
                "neighbor_moved_count": int(neighbor["moved"].map(bool).sum()),
                "neighbor_moved_node_ids": _node_ids(neighbor[neighbor["moved"].map(bool)]),
                "neighbor_margin_min": _finite_min(neighbor["margin"]) if not neighbor.empty else math.nan,
                "neighbor_margin_p50": _quantile(_finite_values(neighbor["margin"]), 0.50),
                "moved_count": int(len(moved)),
                "moved_node_ids": _node_ids(moved),
                "moved_margin_min": _finite_min(moved["margin"]) if not moved.empty else math.nan,
                "moved_margin_p50": _quantile(_finite_values(moved["margin"]), 0.50),
                "best_increment_min": _quantile(best, 0.0),
                "best_increment_p50": _quantile(best, 0.50),
                "best_increment_max": _quantile(best, 1.0),
                "second_increment_min": _quantile(second, 0.0),
                "second_increment_p50": _quantile(second, 0.50),
                "second_increment_max": _quantile(second, 1.0),
                "moved_overlap_previous_window_count": 0,
                "moved_overlap_next_window_count": 0,
                "moved_overlap_target_window_count": 0,
            }
        )

    frame = pd.DataFrame(rows, columns=LOCAL_MOVE_FOCUS_SUMMARY_COLUMNS).sort_values(
        ["replay_iterations", "candidate_index", "iteration", "depth"],
        kind="mergesort",
    )
    _add_focus_overlap_counts(frame, moved_sets)
    return frame


def classify_local_move_movement_attribution(
    *,
    local_move_focus_summary: pd.DataFrame,
    local_move_gain: pd.DataFrame,
    target_candidate_index: int = TARGET_CANDIDATE_INDEX,
    target_replay_iterations: int = TARGET_REPLAY_ITERATIONS,
    target_iteration: int = TARGET_ITERATION,
    target_depth: int = TARGET_DEPTH,
) -> dict[str, Any]:
    target_gain = _target_gain(
        local_move_gain,
        target_candidate_index=target_candidate_index,
        target_replay_iterations=target_replay_iterations,
        target_iteration=target_iteration,
        target_depth=target_depth,
    )
    target = _target_summary_row(
        local_move_focus_summary,
        target_candidate_index=target_candidate_index,
        target_replay_iterations=target_replay_iterations,
        target_iteration=target_iteration,
        target_depth=target_depth,
    )
    peers = _peer_rows(
        local_move_focus_summary,
        target_candidate_index=target_candidate_index,
        target_replay_iterations=target_replay_iterations,
        target_iteration=target_iteration,
        target_depth=target_depth,
    )
    p1 = _candidate_depth_rows(
        local_move_focus_summary,
        candidate_index=target_candidate_index,
        replay_iterations=1,
        depth=target_depth,
    )
    followup = _candidate_window_rows(
        local_move_focus_summary,
        candidate_index=target_candidate_index,
        replay_iterations={3, 5},
        iteration=target_iteration,
        depth=target_depth,
    )
    target_target_moved = _zero_if_nan(_row_float(target, "target_moved_count"))
    target_neighbor_moved = _zero_if_nan(_row_float(target, "neighbor_moved_count"))
    peer_neighbor_moved_max = _column_max(peers, "neighbor_moved_count")
    p1_neighbor_moved_max = _column_max(p1, "neighbor_moved_count")
    followup_neighbor_moved_max = _column_max(followup, "neighbor_moved_count")

    if target_target_moved > 0.0:
        classification = "target_node_move_signal"
    elif target_neighbor_moved > 0.0 and target_neighbor_moved > max(
        peer_neighbor_moved_max,
        p1_neighbor_moved_max,
    ):
        classification = "neighbor_node_move_signal"
    else:
        classification = "no_focused_move_signal"

    return {
        "classification": classification,
        "target_candidate_index": target_candidate_index,
        "target_replay_iterations": target_replay_iterations,
        "target_iteration": target_iteration,
        "target_depth": target_depth,
        "target_local_move_gain": target_gain,
        "target_moved_count": target_target_moved,
        "target_moved_node_ids": _row_text(target, "target_moved_node_ids"),
        "target_neighbor_moved_count": target_neighbor_moved,
        "target_neighbor_moved_node_ids": _row_text(target, "neighbor_moved_node_ids"),
        "peer_neighbor_moved_count_max": peer_neighbor_moved_max,
        "candidate2_p1_neighbor_moved_count_max": p1_neighbor_moved_max,
        "candidate2_p3_p5_neighbor_moved_count_max": followup_neighbor_moved_max,
        "instrumentation_gate_open": classification == "no_focused_move_signal",
    }


def write_local_move_movement_report(
    path: Path,
    *,
    attribution: dict[str, Any],
    local_move_focus_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Candidate 2 Local-Move Movement Attribution Report",
        "",
        "- Target: candidate 2 p2 iteration 2 depth 1",
        f"- Classification: `{attribution['classification']}`",
        f"- Target local-move quality gain: {_signed_format(attribution['target_local_move_gain'])}",
        f"- Target moved nodes: `{attribution['target_moved_node_ids']}`",
        f"- Neighbor moved nodes: `{attribution['target_neighbor_moved_node_ids']}`",
        f"- Follow-up Rust instrumentation gate: {'open' if attribution['instrumentation_gate_open'] else 'closed'}",
        "",
        "## Focused Movement Contrast",
        "",
        "| contrast | candidate | replay | iteration | depth | gain | target moved | neighbor moved | moved nodes | target-window overlap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for label, row in _focus_contrast_rows(
        local_move_focus_summary,
        target_candidate_index=int(attribution["target_candidate_index"]),
        target_replay_iterations=int(attribution["target_replay_iterations"]),
        target_iteration=int(attribution["target_iteration"]),
        target_depth=int(attribution["target_depth"]),
    ):
        lines.append(
            "| {label} | {candidate} | {replay} | {iteration} | {depth} | {gain} | {target_moved} | {neighbor_moved} | {moved_nodes} | {overlap} |".format(
                label=label,
                candidate=_table_int(row, "candidate_index"),
                replay=_table_int(row, "replay_iterations"),
                iteration=_table_int(row, "iteration"),
                depth=_table_int(row, "depth"),
                gain=_signed_format(row.get("quality_gain_since_previous_local_move")),
                target_moved=_table_int(row, "target_moved_count"),
                neighbor_moved=_table_int(row, "neighbor_moved_count"),
                moved_nodes=row.get("moved_node_ids", ""),
                overlap=_table_int(row, "moved_overlap_target_window_count"),
            )
        )
    lines.extend(["", "## Interpretation", ""])
    classification = str(attribution["classification"])
    if classification == "target_node_move_signal":
        lines.append(
            "- A perturbed target-group node moves directly in the target local-move window. Treat this as the primary movement signal to inspect next."
        )
    elif classification == "neighbor_node_move_signal":
        lines.append(
            "- The target group itself does not move in the target local-move window, but its one-hop neighbors move more strongly than p1 and peer p2 contrasts."
        )
    else:
        lines.extend(
            [
                "- Target+neighbor focused local-move trace did not isolate the target-window gain source.",
                "- Broader instrumentation should only be considered after checking whether parent local-merge evidence explains the remaining gap.",
            ]
        )
    lines.extend(
        [
            "",
            "## Rule",
            "",
            "- Classification is `target_node_move_signal` if any target node moves in the target window.",
            "- Classification is `neighbor_node_move_signal` if target nodes do not move and neighbor movement exceeds p1 and peer p2 contrasts.",
            "- This report is trace-only attribution; it does not change production Leiden behavior.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def classify_depth_attribution(
    *,
    local_move_margin_summary: pd.DataFrame,
    local_merge_parent_summary: pd.DataFrame,
    local_move_gain: pd.DataFrame,
    target_candidate_index: int = TARGET_CANDIDATE_INDEX,
    target_replay_iterations: int = TARGET_REPLAY_ITERATIONS,
    target_iteration: int = TARGET_ITERATION,
    target_depth: int = TARGET_DEPTH,
) -> dict[str, Any]:
    target_gain = _target_gain(
        local_move_gain,
        target_candidate_index=target_candidate_index,
        target_replay_iterations=target_replay_iterations,
        target_iteration=target_iteration,
        target_depth=target_depth,
    )
    has_target_gain = math.isfinite(target_gain) and target_gain > 0.0
    local_move_target = _target_summary_row(
        local_move_margin_summary,
        target_candidate_index=target_candidate_index,
        target_replay_iterations=target_replay_iterations,
        target_iteration=target_iteration,
        target_depth=target_depth,
    )
    local_move_peers = _peer_rows(
        local_move_margin_summary,
        target_candidate_index=target_candidate_index,
        target_replay_iterations=target_replay_iterations,
        target_iteration=target_iteration,
        target_depth=target_depth,
    )
    local_merge_target = _target_summary_row(
        local_merge_parent_summary,
        target_candidate_index=target_candidate_index,
        target_replay_iterations=target_replay_iterations,
        target_iteration=target_iteration,
        target_depth=target_depth,
    )
    local_merge_peers = _peer_rows(
        local_merge_parent_summary,
        target_candidate_index=target_candidate_index,
        target_replay_iterations=target_replay_iterations,
        target_iteration=target_iteration,
        target_depth=target_depth,
    )
    p1_merge = _target_p1_context(
        local_merge_parent_summary,
        target_candidate_index=target_candidate_index,
        target_depth=target_depth,
    )

    target_moved = _row_float(local_move_target, "moved_count")
    peer_moved_max = _column_max(local_move_peers, "moved_count")
    target_margin_p50 = _row_float(local_move_target, "margin_p50")
    peer_margin_p50_min = _column_min(local_move_peers, "margin_p50")
    target_zero = _row_float(local_move_target, "near_zero_margin_count")
    peer_zero_max = _column_max(local_move_peers, "near_zero_margin_count")
    local_move_signal = has_target_gain and (
        (target_moved > 0.0 and target_moved > peer_moved_max)
        or (
            math.isfinite(target_margin_p50)
            and math.isfinite(peer_margin_p50_min)
            and target_margin_p50 <= peer_margin_p50_min * 0.5
            and target_zero > peer_zero_max
        )
    )

    target_low = _row_float(local_merge_target, "low_margin_count")
    peer_low_max = _column_max(local_merge_peers, "low_margin_count")
    p1_low = _row_float(p1_merge, "low_margin_count")
    low_baseline = max(peer_low_max, p1_low, 0.0)
    local_merge_signal = has_target_gain and target_low > 0.0 and (
        (low_baseline <= 0.0 and target_low >= 1.0)
        or (target_low - low_baseline >= 2.0)
        or (target_low > low_baseline and target_low / max(low_baseline, 1.0) >= 1.5)
    )

    if local_move_signal:
        classification = "local_move_margin_signal"
    elif local_merge_signal:
        classification = "local_merge_parent_signal"
    else:
        classification = "no_existing_trace_signal"

    return {
        "classification": classification,
        "target_candidate_index": target_candidate_index,
        "target_replay_iterations": target_replay_iterations,
        "target_iteration": target_iteration,
        "target_depth": target_depth,
        "target_local_move_gain": target_gain,
        "local_move_signal": bool(local_move_signal),
        "target_local_move_moved_count": target_moved,
        "peer_local_move_moved_count_max": peer_moved_max,
        "target_local_move_margin_p50": target_margin_p50,
        "peer_local_move_margin_p50_min": peer_margin_p50_min,
        "target_local_move_near_zero_count": target_zero,
        "peer_local_move_near_zero_count_max": peer_zero_max,
        "local_merge_signal": bool(local_merge_signal),
        "target_local_merge_low_margin_count": target_low,
        "peer_local_merge_low_margin_count_max": peer_low_max,
        "candidate_p1_local_merge_low_margin_count": p1_low,
        "target_local_merge_min_margin_min": _row_float(
            local_merge_target, "min_margin_min"
        ),
        "target_local_merge_largest_child_fraction_max": _row_float(
            local_merge_target, "largest_child_fraction_max"
        ),
        "target_local_merge_top_low_margin_parent_ids": _row_text(
            local_merge_target, "top_low_margin_parent_ids"
        ),
        "instrumentation_gate_open": classification == "no_existing_trace_signal",
    }


def write_depth_attribution_report(
    path: Path,
    *,
    attribution: dict[str, Any],
    local_move_margin_summary: pd.DataFrame,
    local_merge_parent_summary: pd.DataFrame,
) -> None:
    target = (
        f"candidate {attribution['target_candidate_index']} "
        f"p{attribution['target_replay_iterations']} "
        f"iteration {attribution['target_iteration']} "
        f"depth {attribution['target_depth']}"
    )
    lines = [
        "# Candidate Trajectory Depth Attribution Report",
        "",
        f"- Target: {target}",
        f"- Classification: `{attribution['classification']}`",
        f"- Target local-move quality gain: {_signed_format(attribution['target_local_move_gain'])}",
        f"- Follow-up Rust instrumentation gate: {'open' if attribution['instrumentation_gate_open'] else 'closed'}",
        "",
        "## Local-Move Margin Contrast",
        "",
        "| contrast | candidate | replay | iteration | depth | moved | near-zero | margin p50 | top low-margin nodes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label, row in _contrast_rows(
        local_move_margin_summary,
        target_candidate_index=int(attribution["target_candidate_index"]),
        target_replay_iterations=int(attribution["target_replay_iterations"]),
        target_iteration=int(attribution["target_iteration"]),
        target_depth=int(attribution["target_depth"]),
    ):
        lines.append(
            "| {label} | {candidate} | {replay} | {iteration} | {depth} | {moved} | {zero} | {p50} | {nodes} |".format(
                label=label,
                candidate=_table_int(row, "candidate_index"),
                replay=_table_int(row, "replay_iterations"),
                iteration=_table_int(row, "iteration"),
                depth=_table_int(row, "depth"),
                moved=_table_int(row, "moved_count"),
                zero=_table_int(row, "near_zero_margin_count"),
                p50=_format_float(row.get("margin_p50")),
                nodes=row.get("top_low_margin_node_ids", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Local-Merge Parent Contrast",
            "",
            "| contrast | candidate | replay | iteration | depth | parent rows | decisions | low-margin | changed | min margin | largest child max | top low-margin parents |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for label, row in _contrast_rows(
        local_merge_parent_summary,
        target_candidate_index=int(attribution["target_candidate_index"]),
        target_replay_iterations=int(attribution["target_replay_iterations"]),
        target_iteration=int(attribution["target_iteration"]),
        target_depth=int(attribution["target_depth"]),
    ):
        lines.append(
            "| {label} | {candidate} | {replay} | {iteration} | {depth} | {parents} | {decisions} | {low} | {changed} | {min_margin} | {largest} | {parent_ids} |".format(
                label=label,
                candidate=_table_int(row, "candidate_index"),
                replay=_table_int(row, "replay_iterations"),
                iteration=_table_int(row, "iteration"),
                depth=_table_int(row, "depth"),
                parents=_table_int(row, "parent_row_count"),
                decisions=_format_float(row.get("decision_count"), 0),
                low=_format_float(row.get("low_margin_count"), 0),
                changed=_format_float(row.get("changed_count"), 0),
                min_margin=_format_float(row.get("min_margin_min"), 6),
                largest=_format_float(row.get("largest_child_fraction_max"), 3),
                parent_ids=row.get("top_low_margin_parent_ids", ""),
            )
        )
    lines.extend(["", "## Interpretation", ""])
    classification = str(attribution["classification"])
    if classification == "local_move_margin_signal":
        lines.append(
            "- Existing trace points first to low-margin local moves at the target depth. "
            "The target has a stronger local-move margin pattern than the peer contrast rows."
        )
    elif classification == "local_merge_parent_signal":
        lines.append(
            "- Existing trace does not show a distinctive local-move margin pattern at the target depth. "
            "The stronger signal is parent-level local-merge low-margin concentration."
        )
        parent_ids = str(attribution["target_local_merge_top_low_margin_parent_ids"])
        if parent_ids:
            lines.append(f"- Target low-margin parent ids: `{parent_ids}`.")
    else:
        lines.extend(
            [
                "- Existing trace did not separate the cause of the target depth gain.",
                "- Minimal next instrumentation: track whether candidate 2's perturbed group nodes move in `fast_local_move`, and add parent-level selected-child change or quality-delta proxy in `local_merge`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Gate Rule",
            "",
            "- Move to Rust instrumentation only when the classification is `no_existing_trace_signal`.",
            "- This report is analysis-only and does not change production Leiden behavior.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_parent_drilldown_only(args: argparse.Namespace) -> dict[str, Path]:
    out_dir = args.output_dir.expanduser().resolve()
    trajectory_trace_path = out_dir / "candidate_trajectory_trace_raw.jsonl"
    run_rows_path = out_dir / "candidate_trajectory_run_rows.csv"
    phase_path = out_dir / "candidate_trajectory_phase_checkpoints.csv"
    local_move_gain_path = out_dir / "candidate_trajectory_local_move_gain_rows.csv"
    required = [trajectory_trace_path, run_rows_path, phase_path, local_move_gain_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "--drilldown-only requires existing trajectory outputs: "
            + ", ".join(missing)
        )

    target_parent_events_path = out_dir / "candidate_trajectory_target_parent_events.csv"
    target_parent_contrast_path = out_dir / "candidate_trajectory_target_parent_contrast.csv"
    parent_causal_window_report_path = (
        out_dir / "candidate_trajectory_parent_causal_window_report.md"
    )
    drilldown = write_parent_drilldown_outputs(
        trajectory_path=trajectory_trace_path,
        run_rows=pd.read_csv(run_rows_path),
        phase_frame=pd.read_csv(phase_path),
        local_move_gain=pd.read_csv(local_move_gain_path),
        target_parent_events_path=target_parent_events_path,
        target_parent_contrast_path=target_parent_contrast_path,
        parent_causal_window_report_path=parent_causal_window_report_path,
        target_parent_ids=_parse_parent_ids(args.target_parent_ids),
    )
    paths = {
        "target_parent_events": target_parent_events_path,
        "target_parent_contrast": target_parent_contrast_path,
        "parent_causal_window_report": parent_causal_window_report_path,
    }
    summary_path = out_dir / "candidate_trajectory_summary.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "parent_causal_window_classification": drilldown["classification"],
            "target_parent_event_rows": int(drilldown["target_parent_event_rows"]),
            "target_parent_contrast_rows": int(drilldown["target_parent_contrast_rows"]),
        }
    )
    path_summary = summary.setdefault("paths", {})
    for name, path in paths.items():
        path_summary[name] = str(path.relative_to(REPO_ROOT))
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    paths["summary"] = summary_path
    return paths


def write_parent_drilldown_outputs(
    *,
    trajectory_path: Path,
    run_rows: pd.DataFrame,
    phase_frame: pd.DataFrame,
    local_move_gain: pd.DataFrame,
    target_parent_events_path: Path,
    target_parent_contrast_path: Path,
    parent_causal_window_report_path: Path,
    target_parent_ids: list[int],
) -> dict[str, Any]:
    target_parent_events = build_target_parent_event_rows(
        _read_jsonl(trajectory_path),
        run_rows=run_rows,
        phase_frame=phase_frame,
        local_move_gain=local_move_gain,
        target_parent_ids=target_parent_ids,
    )
    target_parent_events.to_csv(target_parent_events_path, index=False)
    target_parent_contrast = build_target_parent_contrast(
        target_parent_events,
        run_rows=run_rows,
        local_move_gain=local_move_gain,
        target_parent_ids=target_parent_ids,
    )
    target_parent_contrast.to_csv(target_parent_contrast_path, index=False)
    attribution = classify_parent_causal_window(
        target_parent_events=target_parent_events,
        target_parent_contrast=target_parent_contrast,
        local_move_gain=local_move_gain,
        target_parent_ids=target_parent_ids,
    )
    write_parent_causal_window_report(
        parent_causal_window_report_path,
        attribution=attribution,
        target_parent_contrast=target_parent_contrast,
    )
    return {
        **attribution,
        "target_parent_event_rows": int(len(target_parent_events)),
        "target_parent_contrast_rows": int(len(target_parent_contrast)),
    }


def build_target_parent_event_rows(
    events: Any,
    *,
    run_rows: pd.DataFrame,
    phase_frame: pd.DataFrame,
    local_move_gain: pd.DataFrame,
    target_parent_ids: list[int],
) -> pd.DataFrame:
    run_meta = _run_meta_by_id(run_rows)
    phase_context = _phase_context_by_window(phase_frame, local_move_gain)
    parent_set = {str(parent_id) for parent_id in target_parent_ids}
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") != "local_merge_margin_summary":
            continue
        parent_id = str(event.get("parent_id", ""))
        if parent_id not in parent_set:
            continue
        run_id = str(event.get("run_id", ""))
        meta = run_meta.get(run_id, {})
        iteration = _safe_int(event.get("iteration"), 0)
        depth = _safe_int(event.get("depth"), 0)
        candidate_index = _safe_int(meta.get("candidate_index"), -1)
        replay_iterations = _safe_int(meta.get("replay_iterations"), 0)
        context = phase_context.get((run_id, iteration, depth), {})
        rows.append(
            {
                "context_role": _parent_context_role(
                    candidate_index,
                    replay_iterations,
                    iteration,
                    depth,
                ),
                "case": meta.get("case", ""),
                "seed": meta.get("seed", 0),
                "candidate_index": candidate_index,
                "replay_iterations": replay_iterations,
                "run_id": run_id,
                "iteration": iteration,
                "depth": depth,
                "parent_id": int(parent_id),
                "parent_visit_index": _safe_int(event.get("parent_visit_index"), 0),
                "source": event.get("source", ""),
                "parent_size": _safe_int(event.get("parent_size"), 0),
                "parent_weight": _finite_event_float(event.get("parent_weight")),
                "decision_count": _finite_event_float(event.get("decision_count"), 0.0),
                "low_margin_decision_count": _finite_event_float(
                    event.get("low_margin_decision_count"), 0.0
                ),
                "changed_decision_count": _finite_event_float(
                    event.get("changed_decision_count"), 0.0
                ),
                "min_margin": _finite_event_float(event.get("min_margin")),
                "p10_margin": _finite_event_float(event.get("p10_margin")),
                "p50_margin": _finite_event_float(event.get("p50_margin")),
                "selected_child_count": _safe_int(event.get("selected_child_count"), 0),
                "largest_child_fraction": _finite_event_float(
                    event.get("largest_child_fraction")
                ),
                **context,
            }
        )
    if not rows:
        return pd.DataFrame(columns=TARGET_PARENT_EVENT_COLUMNS)
    return pd.DataFrame(rows, columns=TARGET_PARENT_EVENT_COLUMNS).sort_values(
        ["replay_iterations", "candidate_index", "iteration", "depth", "parent_id"],
        kind="mergesort",
    )


def build_target_parent_contrast(
    target_parent_events: pd.DataFrame,
    *,
    run_rows: pd.DataFrame,
    local_move_gain: pd.DataFrame,
    target_parent_ids: list[int],
) -> pd.DataFrame:
    contexts = _parent_contrast_contexts(run_rows, local_move_gain)
    if not contexts:
        return pd.DataFrame(columns=TARGET_PARENT_CONTRAST_COLUMNS)
    event_lookup = _target_parent_event_lookup(target_parent_events)
    target_lookup: dict[int, pd.Series] = {}
    target_context = next(
        (context for context in contexts if context["context_role"] == "target"),
        None,
    )
    if target_context is not None:
        for parent_id in target_parent_ids:
            row = event_lookup.get((_context_key(target_context), int(parent_id)))
            if row is not None:
                target_lookup[int(parent_id)] = row

    rows: list[dict[str, Any]] = []
    for context in contexts:
        for parent_id in target_parent_ids:
            parent_id = int(parent_id)
            event_row = event_lookup.get((_context_key(context), parent_id))
            target_row = target_lookup.get(parent_id)
            rows.append(
                _contrast_row_from_context(
                    context,
                    parent_id=parent_id,
                    event_row=event_row,
                    target_row=target_row,
                )
            )
    return pd.DataFrame(rows, columns=TARGET_PARENT_CONTRAST_COLUMNS)


def classify_parent_causal_window(
    *,
    target_parent_events: pd.DataFrame,
    target_parent_contrast: pd.DataFrame,
    local_move_gain: pd.DataFrame,
    target_parent_ids: list[int] | None = None,
) -> dict[str, Any]:
    target_gain = _target_gain(
        local_move_gain,
        target_candidate_index=TARGET_CANDIDATE_INDEX,
        target_replay_iterations=TARGET_REPLAY_ITERATIONS,
        target_iteration=TARGET_ITERATION,
        target_depth=TARGET_DEPTH,
    )
    target_total_low = _sum_low_margin(target_parent_contrast, "target")
    pre_rows = target_parent_events[
        target_parent_events["context_role"].astype(str).isin(
            {"target_pre_window", "target_prior_same_depth"}
        )
    ] if not target_parent_events.empty else pd.DataFrame()
    pre_window_max = _max_group_low_margin(pre_rows)
    p1_low = _sum_low_margin(target_parent_contrast, "candidate2_p1_depth1")
    peer_low = max(
        _sum_low_margin(target_parent_contrast, "peer_candidate_0"),
        _sum_low_margin(target_parent_contrast, "peer_candidate_1"),
    )
    followup_low = max(
        _sum_low_margin(target_parent_contrast, "candidate2_p3_same_window"),
        _sum_low_margin(target_parent_contrast, "candidate2_p5_same_window"),
    )

    if not math.isfinite(target_gain) or target_gain <= 0.0 or target_total_low <= 0.0:
        classification = "ambiguous_parent_signal"
    elif pre_window_max >= target_total_low:
        classification = "pre_gain_parent_setup"
    elif target_total_low > max(pre_window_max, p1_low, peer_low):
        classification = "post_gain_parent_symptom"
    else:
        classification = "ambiguous_parent_signal"

    return {
        "classification": classification,
        "target_local_move_gain": target_gain,
        "target_total_low_margin_count": target_total_low,
        "pre_window_low_margin_max": pre_window_max,
        "candidate2_p1_low_margin_count": p1_low,
        "peer_low_margin_max": peer_low,
        "followup_low_margin_max": followup_low,
        "target_parent_ids": ",".join(
            str(parent_id) for parent_id in (target_parent_ids or TARGET_PARENT_IDS)
        ),
        "instrumentation_gate_open": classification == "ambiguous_parent_signal",
    }


def write_parent_causal_window_report(
    path: Path,
    *,
    attribution: dict[str, Any],
    target_parent_contrast: pd.DataFrame,
) -> None:
    lines = [
        "# Candidate 2 Parent Causal Window Report",
        "",
        "- Target: candidate 2 p2 iteration 2 depth 1",
        f"- Target parent ids: `{attribution['target_parent_ids']}`",
        f"- Classification: `{attribution['classification']}`",
        f"- Target local-move quality gain: {_signed_format(attribution['target_local_move_gain'])}",
        f"- Target parent low-margin count: {_format_float(attribution['target_total_low_margin_count'], 0)}",
        f"- Pre-window low-margin max: {_format_float(attribution['pre_window_low_margin_max'], 0)}",
        f"- Peer low-margin max: {_format_float(attribution['peer_low_margin_max'], 0)}",
        f"- Follow-up p3/p5 low-margin max: {_format_float(attribution['followup_low_margin_max'], 0)}",
        "",
        "## Parent Contrast",
        "",
        "| role | parent | candidate | replay | iteration | depth | low-margin | min margin | largest child | local-move gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in target_parent_contrast.iterrows():
        if str(row.get("context_role", "")) == "other":
            continue
        lines.append(
            "| {role} | {parent} | {candidate} | {replay} | {iteration} | {depth} | {low} | {min_margin} | {largest} | {gain} |".format(
                role=row.get("context_role", ""),
                parent=_format_float(row.get("parent_id"), 0),
                candidate=_format_float(row.get("candidate_index"), 0),
                replay=_format_float(row.get("replay_iterations"), 0),
                iteration=_format_float(row.get("iteration"), 0),
                depth=_format_float(row.get("depth"), 0),
                low=_format_float(row.get("low_margin_decision_count"), 0),
                min_margin=_format_float(row.get("min_margin"), 6),
                largest=_format_float(row.get("largest_child_fraction"), 3),
                gain=_signed_format(row.get("quality_gain_since_previous_local_move")),
            )
        )
    lines.extend(["", "## Interpretation", ""])
    classification = str(attribution["classification"])
    if classification == "pre_gain_parent_setup":
        lines.append(
            "- The same target parents already show a strong low-margin signal before the target local-move gain. Treat this as a plausible setup signal, not a confirmed mechanism."
        )
    elif classification == "post_gain_parent_symptom":
        lines.append(
            "- The target parent low-margin concentration is strongest in the refinement window after the target local move. Treat this as a post-gain structural symptom, not direct causal evidence."
        )
    else:
        lines.extend(
            [
                "- Existing phase ordering does not separate whether these parent rows are setup or symptom.",
                "- Minimal next instrumentation: add local-merge parent child-assignment hashes; add fast-local-move perturbed-group movement only if the parent hashes remain inconclusive.",
            ]
        )
    lines.extend(
        [
            "",
            "## Gate Rule",
            "",
            "- Move to Rust instrumentation only when the classification is `ambiguous_parent_signal`.",
            "- This report is analysis-only and does not change production Leiden behavior.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_rank_summary(run_rows: pd.DataFrame, candidate_labels: pd.DataFrame) -> pd.DataFrame:
    if run_rows.empty:
        return pd.DataFrame()
    label_by_candidate = {
        int(row["candidate_index"]): row for _, row in candidate_labels.iterrows()
    }
    full_winner_rows = [
        row for row in label_by_candidate.values() if bool(row.get("is_full_p5_winner", False))
    ]
    full_winner = (
        int(full_winner_rows[0]["candidate_index"]) if full_winner_rows else None
    )
    out: list[dict[str, Any]] = []
    group_cols = ["case", "seed", "replay_iterations"]
    for (case, seed, replay_iterations), group in run_rows.groupby(group_cols, dropna=False):
        ordered = sorted(
            group.to_dict("records"),
            key=lambda row: (-_finite_float(row.get("delta_q"), -math.inf), int(row["candidate_index"])),
        )
        for rank, row in enumerate(ordered, start=1):
            candidate_index = int(row["candidate_index"])
            label = label_by_candidate.get(candidate_index, {})
            out.append(
                {
                    "case": case,
                    "seed": int(seed),
                    "replay_iterations": int(replay_iterations),
                    "candidate_index": candidate_index,
                    "replay_rank": rank,
                    "delta_q": _finite_float(row.get("delta_q"), math.nan),
                    "quality": _finite_float(row.get("quality"), math.nan),
                    "elapsed_sec": _finite_float(row.get("elapsed_sec"), math.nan),
                    "is_replay_top1": rank == 1,
                    "is_replay_top2": rank <= 2,
                    "is_replay_top3": rank <= 3,
                    "is_full_p5_winner": full_winner == candidate_index,
                    "p1_rank_label": _finite_float(label.get("p1_rank"), math.nan),
                    "p5_rank_label": _finite_float(label.get("p5_rank"), math.nan),
                    "p1_delta_q_label": _finite_float(label.get("p1_delta_q"), math.nan),
                    "p5_delta_q_label": _finite_float(label.get("p5_delta_q"), math.nan),
                    "group_kind": label.get("group_kind", ""),
                    "group_count": label.get("group_count", ""),
                    "run_id": row.get("run_id", ""),
                }
            )
    return pd.DataFrame(out).sort_values(["replay_iterations", "replay_rank", "candidate_index"])


def build_transition_summary(rank_summary: pd.DataFrame) -> pd.DataFrame:
    if rank_summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["case", "seed"]
    for (case, seed), group in rank_summary.groupby(group_cols, dropna=False):
        winner_rows = group[group["is_full_p5_winner"]].sort_values("replay_iterations")
        if winner_rows.empty:
            continue
        first_top1 = _first_iteration_at_or_above(winner_rows, "is_replay_top1")
        first_top2 = _first_iteration_at_or_above(winner_rows, "is_replay_top2")
        first_top3 = _first_iteration_at_or_above(winner_rows, "is_replay_top3")
        final = winner_rows.iloc[-1]
        rows.append(
            {
                "case": case,
                "seed": int(seed),
                "full_p5_winner_candidate_index": int(final["candidate_index"]),
                "label_p1_rank": _finite_float(final.get("p1_rank_label"), math.nan),
                "label_p5_rank": _finite_float(final.get("p5_rank_label"), math.nan),
                "first_replay_iterations_top1": first_top1,
                "first_replay_iterations_top2": first_top2,
                "first_replay_iterations_top3": first_top3,
                "winner_rank_at_p1": _rank_at_iteration(winner_rows, 1),
                "winner_rank_at_p2": _rank_at_iteration(winner_rows, 2),
                "winner_rank_at_p3": _rank_at_iteration(winner_rows, 3),
                "winner_rank_at_p5": _rank_at_iteration(winner_rows, 5),
                "winner_delta_q_at_p1": _delta_at_iteration(winner_rows, 1),
                "winner_delta_q_at_p2": _delta_at_iteration(winner_rows, 2),
                "winner_delta_q_at_p3": _delta_at_iteration(winner_rows, 3),
                "winner_delta_q_at_p5": _delta_at_iteration(winner_rows, 5),
            }
        )
    return pd.DataFrame(rows)


def build_local_move_gain_rows(phase_frame: pd.DataFrame, run_rows: pd.DataFrame) -> pd.DataFrame:
    if phase_frame.empty or run_rows.empty:
        return pd.DataFrame()
    run_meta = run_rows[
        [
            "run_id",
            "case",
            "seed",
            "candidate_index",
            "replay_iterations",
            "baseline_quality",
        ]
    ].copy()
    rows = phase_frame[phase_frame["phase"].astype(str).eq("after_local_move")].copy()
    if rows.empty:
        return pd.DataFrame()
    rows = rows.merge(run_meta, on="run_id", how="left", suffixes=("", "_meta"))
    rows["quality_delta_vs_baseline"] = rows["quality"] - rows["baseline_quality"]
    rows = rows.sort_values(["run_id", "iteration", "depth"], kind="mergesort")
    rows["previous_local_move_quality"] = rows.groupby("run_id")["quality"].shift(1)
    rows["quality_gain_since_previous_local_move"] = (
        rows["quality"] - rows["previous_local_move_quality"]
    )
    return rows[
        [
            "case",
            "seed",
            "candidate_index",
            "replay_iterations",
            "run_id",
            "iteration",
            "depth",
            "n_clusters",
            "quality",
            "quality_delta_vs_baseline",
            "previous_local_move_quality",
            "quality_gain_since_previous_local_move",
            "membership_hash",
        ]
    ]


def _local_move_gain_lookup(local_move_gain: pd.DataFrame) -> dict[tuple[str, int, int], float]:
    lookup: dict[tuple[str, int, int], float] = {}
    if local_move_gain.empty:
        return lookup
    for _, row in local_move_gain.iterrows():
        run_id = str(row.get("run_id", ""))
        if not run_id:
            continue
        lookup[
            (
                run_id,
                _safe_int(row.get("iteration"), 0),
                _safe_int(row.get("depth"), 0),
            )
        ] = _finite_event_float(row.get("quality_gain_since_previous_local_move"))
    return lookup


def _nullable_int(value: Any) -> int | float:
    if value is None:
        return math.nan
    try:
        if pd.isna(value):
            return math.nan
        return int(value)
    except (TypeError, ValueError):
        return math.nan


def _node_ids(rows: pd.DataFrame, limit: int = TOP_ID_LIMIT) -> str:
    if rows.empty or "node" not in rows.columns:
        return ""
    nodes: list[str] = []
    for node in sorted({_safe_int(item, -1) for item in rows["node"].tolist()}):
        if node < 0:
            continue
        nodes.append(str(node))
        if len(nodes) >= limit:
            break
    return ",".join(nodes)


def _add_focus_overlap_counts(
    frame: pd.DataFrame,
    moved_sets: dict[tuple[str, int, int], set[int]],
) -> None:
    if frame.empty:
        return
    target_key = None
    target_rows = frame[
        frame["candidate_index"].astype(int).eq(TARGET_CANDIDATE_INDEX)
        & frame["replay_iterations"].astype(int).eq(TARGET_REPLAY_ITERATIONS)
        & frame["iteration"].astype(int).eq(TARGET_ITERATION)
        & frame["depth"].astype(int).eq(TARGET_DEPTH)
    ]
    if not target_rows.empty:
        target_row = target_rows.iloc[0]
        target_key = (
            str(target_row.get("run_id", "")),
            _safe_int(target_row.get("iteration"), 0),
            _safe_int(target_row.get("depth"), 0),
        )
    target_set = moved_sets.get(target_key, set()) if target_key else set()

    previous_by_key: dict[tuple[str, int, int], tuple[str, int, int] | None] = {}
    next_by_key: dict[tuple[str, int, int], tuple[str, int, int] | None] = {}
    for run_id, group in frame.groupby("run_id", dropna=False, sort=False):
        keys = [
            (str(run_id), _safe_int(row.get("iteration"), 0), _safe_int(row.get("depth"), 0))
            for _, row in group.sort_values(["iteration", "depth"], kind="mergesort").iterrows()
        ]
        for index, key in enumerate(keys):
            previous_by_key[key] = keys[index - 1] if index > 0 else None
            next_by_key[key] = keys[index + 1] if index + 1 < len(keys) else None

    for idx, row in frame.iterrows():
        key = (
            str(row.get("run_id", "")),
            _safe_int(row.get("iteration"), 0),
            _safe_int(row.get("depth"), 0),
        )
        current = moved_sets.get(key, set())
        previous_key = previous_by_key.get(key)
        next_key = next_by_key.get(key)
        previous = moved_sets.get(previous_key, set()) if previous_key else set()
        next_set = moved_sets.get(next_key, set()) if next_key else set()
        frame.at[idx, "moved_overlap_previous_window_count"] = len(current & previous)
        frame.at[idx, "moved_overlap_next_window_count"] = len(current & next_set)
        frame.at[idx, "moved_overlap_target_window_count"] = len(current & target_set)


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        if pd.isna(value):
            return fallback
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _finite_event_float(value: Any, fallback: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if math.isfinite(out) else fallback


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value) != 0.0
    return str(value).strip().lower() in {"1", "true", "yes"}


def _finite_values(values: Any) -> list[float]:
    out: list[float] = []
    for value in values:
        parsed = _finite_event_float(value)
        if math.isfinite(parsed):
            out.append(parsed)
    return out


def _quantile(values: list[float], q: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return math.nan
    if len(finite) == 1:
        return float(finite[0])
    position = (len(finite) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(finite[lower])
    weight = position - lower
    return float(finite[lower] * (1.0 - weight) + finite[upper] * weight)


def _finite_min(values: Any) -> float:
    finite = _finite_values(values)
    return min(finite) if finite else math.nan


def _finite_max(values: Any) -> float:
    finite = _finite_values(values)
    return max(finite) if finite else math.nan


def _min_finite(left: float, right: float) -> float:
    if not math.isfinite(right):
        return left
    if not math.isfinite(left):
        return right
    return min(left, right)


def _max_finite(left: float, right: float) -> float:
    if not math.isfinite(right):
        return left
    if not math.isfinite(left):
        return right
    return max(left, right)


def _parent_sort_key(parent_id: Any) -> tuple[int, int | str]:
    text = str(parent_id)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def _top_node_ids(rows: pd.DataFrame, limit: int = TOP_ID_LIMIT) -> str:
    if rows.empty:
        return ""
    ranked = rows.sort_values(["margin", "rank", "node"], kind="mergesort")
    nodes: list[str] = []
    for node in ranked["node"].tolist():
        if _safe_int(node, -1) < 0:
            continue
        node_text = str(int(node))
        if node_text not in nodes:
            nodes.append(node_text)
        if len(nodes) >= limit:
            break
    return ",".join(nodes)


def _top_parent_ids(
    rows: pd.DataFrame,
    *,
    metric: str,
    direction: str,
    positive_only: bool = False,
    finite_only: bool = False,
    limit: int = TOP_ID_LIMIT,
) -> str:
    if rows.empty:
        return ""
    parent_rows: list[dict[str, Any]] = []
    for parent_id, group in rows.groupby("parent_id", dropna=False, sort=False):
        metric_values = _finite_values(group[metric])
        if not metric_values:
            metric_value = math.nan
        elif direction == "asc":
            metric_value = min(metric_values)
        else:
            metric_value = sum(metric_values) if metric.endswith("_count") else max(metric_values)
        if positive_only and (not math.isfinite(metric_value) or metric_value <= 0.0):
            continue
        if finite_only and not math.isfinite(metric_value):
            continue
        parent_rows.append(
            {
                "parent_id": str(parent_id),
                "metric": metric_value,
                "min_margin": _finite_min(group["min_margin"])
                if "min_margin" in group
                else math.nan,
                "low_margin": float(group["low_margin_decision_count"].sum())
                if "low_margin_decision_count" in group
                else 0.0,
            }
        )
    if direction == "asc":
        ranked = sorted(
            parent_rows,
            key=lambda row: (
                row["metric"],
                -row["low_margin"],
                _parent_sort_key(row["parent_id"]),
            ),
        )
    else:
        ranked = sorted(
            parent_rows,
            key=lambda row: (
                -row["metric"],
                row["min_margin"] if math.isfinite(row["min_margin"]) else math.inf,
                _parent_sort_key(row["parent_id"]),
            ),
        )
    return ",".join(row["parent_id"] for row in ranked[:limit])


def _top_parent_ids_from_totals(
    parent_totals: dict[str, dict[str, float]],
    *,
    metric: str,
    direction: str,
    positive_only: bool = False,
    finite_only: bool = False,
    limit: int = TOP_ID_LIMIT,
) -> str:
    parent_rows: list[dict[str, Any]] = []
    for parent_id, values in parent_totals.items():
        metric_value = _finite_event_float(values.get(metric))
        if positive_only and (not math.isfinite(metric_value) or metric_value <= 0.0):
            continue
        if finite_only and not math.isfinite(metric_value):
            continue
        parent_rows.append(
            {
                "parent_id": parent_id,
                "metric": metric_value,
                "min_margin": _finite_event_float(values.get("min_margin")),
                "low_margin": _finite_event_float(
                    values.get("low_margin_decision_count"), 0.0
                ),
            }
        )
    if direction == "asc":
        ranked = sorted(
            parent_rows,
            key=lambda row: (
                row["metric"],
                -row["low_margin"],
                _parent_sort_key(row["parent_id"]),
            ),
        )
    else:
        ranked = sorted(
            parent_rows,
            key=lambda row: (
                -row["metric"],
                row["min_margin"] if math.isfinite(row["min_margin"]) else math.inf,
                _parent_sort_key(row["parent_id"]),
            ),
        )
    return ",".join(row["parent_id"] for row in ranked[:limit])


def _target_gain(
    local_move_gain: pd.DataFrame,
    *,
    target_candidate_index: int,
    target_replay_iterations: int,
    target_iteration: int,
    target_depth: int,
) -> float:
    row = _target_summary_row(
        local_move_gain,
        target_candidate_index=target_candidate_index,
        target_replay_iterations=target_replay_iterations,
        target_iteration=target_iteration,
        target_depth=target_depth,
    )
    return _row_float(row, "quality_gain_since_previous_local_move")


def _target_summary_row(
    frame: pd.DataFrame,
    *,
    target_candidate_index: int,
    target_replay_iterations: int,
    target_iteration: int,
    target_depth: int,
) -> pd.Series | None:
    if frame.empty:
        return None
    required = {"candidate_index", "replay_iterations", "iteration", "depth"}
    if not required.issubset(frame.columns):
        return None
    matched = frame[
        frame["candidate_index"].astype(int).eq(target_candidate_index)
        & frame["replay_iterations"].astype(int).eq(target_replay_iterations)
        & frame["iteration"].astype(int).eq(target_iteration)
        & frame["depth"].astype(int).eq(target_depth)
    ]
    if matched.empty:
        return None
    return matched.iloc[0]


def _peer_rows(
    frame: pd.DataFrame,
    *,
    target_candidate_index: int,
    target_replay_iterations: int,
    target_iteration: int,
    target_depth: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    required = {"candidate_index", "replay_iterations", "iteration", "depth"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    return frame[
        frame["candidate_index"].astype(int).ne(target_candidate_index)
        & frame["replay_iterations"].astype(int).eq(target_replay_iterations)
        & frame["iteration"].astype(int).eq(target_iteration)
        & frame["depth"].astype(int).eq(target_depth)
    ].copy()


def _candidate_depth_rows(
    frame: pd.DataFrame,
    *,
    candidate_index: int,
    replay_iterations: int,
    depth: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    required = {"candidate_index", "replay_iterations", "depth"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    return frame[
        frame["candidate_index"].astype(int).eq(candidate_index)
        & frame["replay_iterations"].astype(int).eq(replay_iterations)
        & frame["depth"].astype(int).eq(depth)
    ].copy()


def _candidate_window_rows(
    frame: pd.DataFrame,
    *,
    candidate_index: int,
    replay_iterations: set[int],
    iteration: int,
    depth: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    required = {"candidate_index", "replay_iterations", "iteration", "depth"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    return frame[
        frame["candidate_index"].astype(int).eq(candidate_index)
        & frame["replay_iterations"].astype(int).isin(sorted(replay_iterations))
        & frame["iteration"].astype(int).eq(iteration)
        & frame["depth"].astype(int).eq(depth)
    ].copy()


def _target_p1_context(
    frame: pd.DataFrame,
    *,
    target_candidate_index: int,
    target_depth: int,
) -> pd.Series | None:
    if frame.empty:
        return None
    required = {"candidate_index", "replay_iterations", "iteration", "depth"}
    if not required.issubset(frame.columns):
        return None
    matched = frame[
        frame["candidate_index"].astype(int).eq(target_candidate_index)
        & frame["replay_iterations"].astype(int).eq(1)
        & frame["depth"].astype(int).eq(target_depth)
    ].copy()
    if matched.empty:
        return None
    matched = matched.sort_values(["iteration", "depth"], kind="mergesort")
    return matched.iloc[-1]


def _row_float(row: pd.Series | None, column: str) -> float:
    if row is None or column not in row:
        return math.nan
    return _finite_event_float(row.get(column))


def _row_text(row: pd.Series | None, column: str) -> str:
    if row is None or column not in row:
        return ""
    value = row.get(column)
    return "" if pd.isna(value) else str(value)


def _column_max(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    values = _finite_values(frame[column])
    return max(values) if values else 0.0


def _column_min(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    values = _finite_values(frame[column])
    return min(values) if values else math.nan


def _zero_if_nan(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def _contrast_rows(
    frame: pd.DataFrame,
    *,
    target_candidate_index: int,
    target_replay_iterations: int,
    target_iteration: int,
    target_depth: int,
) -> list[tuple[str, pd.Series]]:
    if frame.empty:
        return []
    rows: list[tuple[str, pd.Series]] = []
    target = _target_summary_row(
        frame,
        target_candidate_index=target_candidate_index,
        target_replay_iterations=target_replay_iterations,
        target_iteration=target_iteration,
        target_depth=target_depth,
    )
    if target is not None:
        rows.append(("target", target))
    p1 = _target_p1_context(
        frame,
        target_candidate_index=target_candidate_index,
        target_depth=target_depth,
    )
    if p1 is not None:
        rows.append(("candidate2 p1 depth1", p1))
    peers = _peer_rows(
        frame,
        target_candidate_index=target_candidate_index,
        target_replay_iterations=target_replay_iterations,
        target_iteration=target_iteration,
        target_depth=target_depth,
    )
    for _, row in peers.sort_values(["candidate_index"], kind="mergesort").iterrows():
        rows.append((f"peer candidate {int(row['candidate_index'])}", row))
    return rows


def _focus_contrast_rows(
    frame: pd.DataFrame,
    *,
    target_candidate_index: int,
    target_replay_iterations: int,
    target_iteration: int,
    target_depth: int,
) -> list[tuple[str, pd.Series]]:
    if frame.empty:
        return []
    rows: list[tuple[str, pd.Series]] = []
    target = _target_summary_row(
        frame,
        target_candidate_index=target_candidate_index,
        target_replay_iterations=target_replay_iterations,
        target_iteration=target_iteration,
        target_depth=target_depth,
    )
    if target is not None:
        rows.append(("target", target))
    p1 = _target_p1_context(
        frame,
        target_candidate_index=target_candidate_index,
        target_depth=target_depth,
    )
    if p1 is not None:
        rows.append(("candidate2 p1 depth1", p1))
    peers = _peer_rows(
        frame,
        target_candidate_index=target_candidate_index,
        target_replay_iterations=target_replay_iterations,
        target_iteration=target_iteration,
        target_depth=target_depth,
    )
    for _, row in peers.sort_values(["candidate_index"], kind="mergesort").iterrows():
        rows.append((f"peer candidate {int(row['candidate_index'])}", row))
    followup = _candidate_window_rows(
        frame,
        candidate_index=target_candidate_index,
        replay_iterations={3, 5},
        iteration=target_iteration,
        depth=target_depth,
    )
    for _, row in followup.sort_values(["replay_iterations"], kind="mergesort").iterrows():
        rows.append((f"candidate2 p{int(row['replay_iterations'])}", row))
    return rows


def _table_int(row: pd.Series, column: str) -> str:
    if column not in row or pd.isna(row[column]):
        return ""
    return str(int(row[column]))


def _signed_format(value: Any, digits: int = 3) -> str:
    out = _finite_float(value, math.nan)
    if not math.isfinite(out):
        return ""
    return f"{out:+.{digits}f}"


def _phase_context_by_window(
    phase_frame: pd.DataFrame,
    local_move_gain: pd.DataFrame,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    context: dict[tuple[str, int, int], dict[str, Any]] = {}
    if not phase_frame.empty:
        for _, row in phase_frame.iterrows():
            run_id = str(row.get("run_id", ""))
            if not run_id:
                continue
            iteration = _safe_int(row.get("iteration"), 0)
            depth = _safe_int(row.get("depth"), 0)
            key = (run_id, iteration, depth)
            item = context.setdefault(key, _empty_phase_context())
            phase = str(row.get("phase", ""))
            quality = _finite_event_float(row.get("quality"))
            membership_hash = "" if pd.isna(row.get("membership_hash", "")) else str(row.get("membership_hash", ""))
            if phase == "after_local_move":
                item["after_local_move_quality"] = quality
                item["after_local_move_membership_hash"] = membership_hash
            elif phase == "after_refinement":
                item["after_refinement_quality"] = quality
                item["after_refinement_membership_hash"] = membership_hash
            elif phase.startswith("after_aggregation"):
                item["after_aggregation_phase"] = phase
                item["after_aggregation_quality"] = quality
                item["after_aggregation_membership_hash"] = membership_hash
    if not local_move_gain.empty:
        for _, row in local_move_gain.iterrows():
            run_id = str(row.get("run_id", ""))
            if not run_id:
                continue
            key = (
                run_id,
                _safe_int(row.get("iteration"), 0),
                _safe_int(row.get("depth"), 0),
            )
            item = context.setdefault(key, _empty_phase_context())
            item["quality_gain_since_previous_local_move"] = _finite_event_float(
                row.get("quality_gain_since_previous_local_move")
            )
    return context


def _empty_phase_context() -> dict[str, Any]:
    return {
        "after_local_move_quality": math.nan,
        "after_local_move_membership_hash": "",
        "after_refinement_quality": math.nan,
        "after_refinement_membership_hash": "",
        "after_aggregation_phase": "",
        "after_aggregation_quality": math.nan,
        "after_aggregation_membership_hash": "",
        "quality_gain_since_previous_local_move": math.nan,
    }


def _parent_context_role(
    candidate_index: int,
    replay_iterations: int,
    iteration: int,
    depth: int,
) -> str:
    window = (iteration, depth)
    target_window = (TARGET_ITERATION, TARGET_DEPTH)
    if (
        candidate_index == TARGET_CANDIDATE_INDEX
        and replay_iterations == TARGET_REPLAY_ITERATIONS
        and window == target_window
    ):
        return "target"
    if candidate_index == TARGET_CANDIDATE_INDEX and replay_iterations == TARGET_REPLAY_ITERATIONS:
        if window < target_window:
            if depth == TARGET_DEPTH:
                return "target_prior_same_depth"
            return "target_pre_window"
        return "target_post_window"
    if (
        candidate_index == TARGET_CANDIDATE_INDEX
        and replay_iterations == 1
        and depth == TARGET_DEPTH
    ):
        return "candidate2_p1_depth1"
    if (
        replay_iterations == TARGET_REPLAY_ITERATIONS
        and iteration == TARGET_ITERATION
        and depth == TARGET_DEPTH
        and candidate_index != TARGET_CANDIDATE_INDEX
    ):
        return f"peer_candidate_{candidate_index}"
    if (
        candidate_index == TARGET_CANDIDATE_INDEX
        and replay_iterations in {3, 5}
        and iteration == TARGET_ITERATION
        and depth == TARGET_DEPTH
    ):
        return f"candidate2_p{replay_iterations}_same_window"
    return "other"


def _parent_contrast_contexts(
    run_rows: pd.DataFrame,
    local_move_gain: pd.DataFrame,
) -> list[dict[str, Any]]:
    if local_move_gain.empty:
        return []
    run_meta = _run_meta_by_id(run_rows)
    gain_rows = local_move_gain.copy()
    contexts: list[dict[str, Any]] = []
    target_rows = gain_rows[
        gain_rows["candidate_index"].astype(int).eq(TARGET_CANDIDATE_INDEX)
        & gain_rows["replay_iterations"].astype(int).eq(TARGET_REPLAY_ITERATIONS)
    ].sort_values(["iteration", "depth"], kind="mergesort")
    target_window = (TARGET_ITERATION, TARGET_DEPTH)
    previous_target = target_rows[
        target_rows.apply(
            lambda row: (_safe_int(row.get("iteration")), _safe_int(row.get("depth"))) < target_window,
            axis=1,
        )
    ]
    wanted: list[tuple[str, pd.Series]] = []
    if not previous_target.empty:
        wanted.append(("target_pre_window", previous_target.iloc[-1]))
    prior_same_depth = previous_target[
        previous_target["depth"].astype(int).eq(TARGET_DEPTH)
    ]
    if not prior_same_depth.empty:
        wanted.append(("target_prior_same_depth", prior_same_depth.iloc[-1]))

    fixed = gain_rows[
        (
            gain_rows["candidate_index"].astype(int).eq(TARGET_CANDIDATE_INDEX)
            & gain_rows["replay_iterations"].astype(int).eq(TARGET_REPLAY_ITERATIONS)
            & gain_rows["iteration"].astype(int).eq(TARGET_ITERATION)
            & gain_rows["depth"].astype(int).eq(TARGET_DEPTH)
        )
        | (
            gain_rows["candidate_index"].astype(int).eq(TARGET_CANDIDATE_INDEX)
            & gain_rows["replay_iterations"].astype(int).eq(1)
            & gain_rows["depth"].astype(int).eq(TARGET_DEPTH)
        )
        | (
            gain_rows["candidate_index"].astype(int).ne(TARGET_CANDIDATE_INDEX)
            & gain_rows["replay_iterations"].astype(int).eq(TARGET_REPLAY_ITERATIONS)
            & gain_rows["iteration"].astype(int).eq(TARGET_ITERATION)
            & gain_rows["depth"].astype(int).eq(TARGET_DEPTH)
        )
        | (
            gain_rows["candidate_index"].astype(int).eq(TARGET_CANDIDATE_INDEX)
            & gain_rows["replay_iterations"].astype(int).isin([3, 5])
            & gain_rows["iteration"].astype(int).eq(TARGET_ITERATION)
            & gain_rows["depth"].astype(int).eq(TARGET_DEPTH)
        )
    ]
    for _, row in fixed.sort_values(
        ["replay_iterations", "candidate_index", "iteration", "depth"],
        kind="mergesort",
    ).iterrows():
        role = _parent_context_role(
            _safe_int(row.get("candidate_index"), -1),
            _safe_int(row.get("replay_iterations"), 0),
            _safe_int(row.get("iteration"), 0),
            _safe_int(row.get("depth"), 0),
        )
        if role != "other":
            wanted.append((role, row))

    seen: set[tuple[str, str, int, int]] = set()
    role_order = {
        "target": 0,
        "target_pre_window": 1,
        "target_prior_same_depth": 2,
        "candidate2_p1_depth1": 3,
        "peer_candidate_0": 4,
        "peer_candidate_1": 5,
        "candidate2_p3_same_window": 6,
        "candidate2_p5_same_window": 7,
    }
    for role, row in sorted(
        wanted,
        key=lambda item: (
            role_order.get(item[0], 99),
            _safe_int(item[1].get("replay_iterations"), 0),
            _safe_int(item[1].get("candidate_index"), -1),
            _safe_int(item[1].get("iteration"), 0),
            _safe_int(item[1].get("depth"), 0),
        ),
    ):
        run_id = str(row.get("run_id", ""))
        iteration = _safe_int(row.get("iteration"), 0)
        depth = _safe_int(row.get("depth"), 0)
        key = (role, run_id, iteration, depth)
        if key in seen:
            continue
        seen.add(key)
        meta = run_meta.get(run_id, {})
        contexts.append(
            {
                "context_role": role,
                "case": meta.get("case", row.get("case", "")),
                "seed": meta.get("seed", _safe_int(row.get("seed"), 0)),
                "candidate_index": _safe_int(row.get("candidate_index"), -1),
                "replay_iterations": _safe_int(row.get("replay_iterations"), 0),
                "run_id": run_id,
                "iteration": iteration,
                "depth": depth,
                "after_local_move_quality": _finite_event_float(row.get("quality")),
                "quality_gain_since_previous_local_move": _finite_event_float(
                    row.get("quality_gain_since_previous_local_move")
                ),
            }
        )
    return contexts


def _context_key(context: dict[str, Any] | pd.Series) -> tuple[str, int, int]:
    return (
        str(context.get("run_id", "")),
        _safe_int(context.get("iteration"), 0),
        _safe_int(context.get("depth"), 0),
    )


def _target_parent_event_lookup(
    target_parent_events: pd.DataFrame,
) -> dict[tuple[tuple[str, int, int], int], pd.Series]:
    if target_parent_events.empty:
        return {}
    lookup = {}
    for _, row in target_parent_events.iterrows():
        key = (
            (
                str(row.get("run_id", "")),
                _safe_int(row.get("iteration"), 0),
                _safe_int(row.get("depth"), 0),
            ),
            _safe_int(row.get("parent_id"), -1),
        )
        lookup[key] = row
    return lookup


def _contrast_row_from_context(
    context: dict[str, Any],
    *,
    parent_id: int,
    event_row: pd.Series | None,
    target_row: pd.Series | None,
) -> dict[str, Any]:
    low = _row_float(event_row, "low_margin_decision_count")
    min_margin = _row_float(event_row, "min_margin")
    target_low = _row_float(target_row, "low_margin_decision_count")
    target_min_margin = _row_float(target_row, "min_margin")
    return {
        "context_role": context.get("context_role", ""),
        "case": context.get("case", ""),
        "seed": context.get("seed", 0),
        "candidate_index": context.get("candidate_index", -1),
        "replay_iterations": context.get("replay_iterations", 0),
        "run_id": context.get("run_id", ""),
        "iteration": context.get("iteration", 0),
        "depth": context.get("depth", 0),
        "parent_id": parent_id,
        "parent_seen": event_row is not None,
        "parent_size": _row_float(event_row, "parent_size"),
        "parent_weight": _row_float(event_row, "parent_weight"),
        "decision_count": _row_float(event_row, "decision_count"),
        "low_margin_decision_count": low,
        "changed_decision_count": _row_float(event_row, "changed_decision_count"),
        "min_margin": min_margin,
        "p10_margin": _row_float(event_row, "p10_margin"),
        "p50_margin": _row_float(event_row, "p50_margin"),
        "selected_child_count": _row_float(event_row, "selected_child_count"),
        "largest_child_fraction": _row_float(event_row, "largest_child_fraction"),
        "after_local_move_quality": _row_float(event_row, "after_local_move_quality")
        if event_row is not None
        else context.get("after_local_move_quality", math.nan),
        "after_refinement_quality": _row_float(event_row, "after_refinement_quality"),
        "after_aggregation_quality": _row_float(event_row, "after_aggregation_quality"),
        "quality_gain_since_previous_local_move": _row_float(
            event_row, "quality_gain_since_previous_local_move"
        )
        if event_row is not None
        else context.get("quality_gain_since_previous_local_move", math.nan),
        "target_low_margin_delta": target_low - low
        if math.isfinite(target_low) and math.isfinite(low)
        else math.nan,
        "target_min_margin_delta": target_min_margin - min_margin
        if math.isfinite(target_min_margin) and math.isfinite(min_margin)
        else math.nan,
    }


def _sum_low_margin(frame: pd.DataFrame, context_role: str) -> float:
    if frame.empty or "context_role" not in frame.columns:
        return 0.0
    rows = frame[frame["context_role"].astype(str).eq(context_role)]
    values = _finite_values(rows.get("low_margin_decision_count", []))
    return float(sum(values))


def _max_group_low_margin(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    max_low = 0.0
    for _, group in frame.groupby(["run_id", "iteration", "depth"], dropna=False):
        max_low = max(max_low, float(sum(_finite_values(group["low_margin_decision_count"]))))
    return max_low


def _first_iteration_at_or_above(rows: pd.DataFrame, column: str) -> int | float:
    matched = rows[rows[column].map(bool)].sort_values("replay_iterations")
    if matched.empty:
        return math.nan
    return int(matched.iloc[0]["replay_iterations"])


def _rank_at_iteration(rows: pd.DataFrame, replay_iterations: int) -> int | float:
    matched = rows[rows["replay_iterations"].astype(int).eq(replay_iterations)]
    if matched.empty:
        return math.nan
    return int(matched.iloc[0]["replay_rank"])


def _delta_at_iteration(rows: pd.DataFrame, replay_iterations: int) -> float:
    matched = rows[rows["replay_iterations"].astype(int).eq(replay_iterations)]
    if matched.empty:
        return math.nan
    return _finite_float(matched.iloc[0]["delta_q"], math.nan)


def _format_float(value: Any, digits: int = 3) -> str:
    out = _finite_float(value, math.nan)
    if not math.isfinite(out):
        return ""
    return f"{out:.{digits}f}"


def write_report(
    path: Path,
    rank_summary: pd.DataFrame,
    transition_summary: pd.DataFrame,
    local_move_gain: pd.DataFrame | None = None,
) -> None:
    lines = [
        "# Leiden Multi-Fidelity Candidate Trajectory Report",
        "",
        "Focused replay of labeled candidates under traced iteration budgets.",
        "",
    ]
    if transition_summary.empty:
        lines.append("No full p5 winner transition summary was available.")
        full_winner = None
        first_top2 = math.nan
    else:
        row = transition_summary.iloc[0]
        full_winner = int(row["full_p5_winner_candidate_index"])
        first_top2 = _finite_float(row["first_replay_iterations_top2"], math.nan)
        lines.extend(
            [
                "## Full p5 Winner Transition",
                "",
                f"- Candidate: {int(row['full_p5_winner_candidate_index'])}",
                f"- Label ranks: p1={_format_float(row['label_p1_rank'], 0)}, p5={_format_float(row['label_p5_rank'], 0)}",
                f"- First replay iteration in top2: {_format_float(row['first_replay_iterations_top2'], 0)}",
                f"- Ranks by replay budget: p1={_format_float(row['winner_rank_at_p1'], 0)}, p2={_format_float(row['winner_rank_at_p2'], 0)}, p3={_format_float(row['winner_rank_at_p3'], 0)}, p5={_format_float(row['winner_rank_at_p5'], 0)}",
                f"- Delta q by replay budget: p1={_format_float(row['winner_delta_q_at_p1'])}, p2={_format_float(row['winner_delta_q_at_p2'])}, p3={_format_float(row['winner_delta_q_at_p3'])}, p5={_format_float(row['winner_delta_q_at_p5'])}",
                "",
            ]
        )
        if (
            local_move_gain is not None
            and not local_move_gain.empty
            and math.isfinite(first_top2)
        ):
            gain_rows = local_move_gain[
                local_move_gain["candidate_index"].astype(int).eq(full_winner)
                & local_move_gain["replay_iterations"].astype(int).eq(int(first_top2))
                & local_move_gain["iteration"].astype(int).eq(int(first_top2))
            ].copy()
            if not gain_rows.empty:
                lines.extend(
                    [
                        "## First Top2 Iteration Local-Move Gains",
                        "",
                        "| depth | quality delta | gain since previous local move | clusters |",
                        "|---:|---:|---:|---:|",
                    ]
                )
                for _, gain_row in gain_rows.iterrows():
                    lines.append(
                        "| {depth} | {delta} | {gain} | {clusters} |".format(
                            depth=int(gain_row["depth"]),
                            delta=_format_float(gain_row["quality_delta_vs_baseline"]),
                            gain=_format_float(
                                gain_row["quality_gain_since_previous_local_move"]
                            ),
                            clusters=int(gain_row["n_clusters"]),
                        )
                    )
                lines.append("")
    lines.extend(
        [
            "## Replay Rank Table",
            "",
            "| replay iterations | candidate | rank | delta q | full p5 winner | group | kind |",
            "|---:|---:|---:|---:|---|---:|---|",
        ]
    )
    for _, row in rank_summary.iterrows():
        lines.append(
            "| {iters} | {candidate} | {rank} | {delta} | {winner} | {group} | {kind} |".format(
                iters=int(row["replay_iterations"]),
                candidate=int(row["candidate_index"]),
                rank=int(row["replay_rank"]),
                delta=_format_float(row["delta_q"]),
                winner="yes" if bool(row["is_full_p5_winner"]) else "no",
                group=row.get("group_count", ""),
                kind=row.get("group_kind", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Replay seeds follow the multifidelity probe convention: monitor seed + perturb offset + candidate index.",
            "- Phase checkpoints and qf/k-work points are written separately for detailed inspection.",
            "- This script is analysis-only and does not change candidate selection or production Leiden behavior.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--graph-dirs", type=str, default=None)
    parser.add_argument("--case-contains", type=str, default="cc_cosine")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--candidate-indices", type=str, default="all")
    parser.add_argument("--replay-iterations", type=str, default="1,2,3,5")
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    parser.add_argument("--target-max-weight", type=float, default=1000.0)
    parser.add_argument(
        "--target-parent-ids",
        type=str,
        default=",".join(str(parent_id) for parent_id in TARGET_PARENT_IDS),
    )
    parser.add_argument(
        "--footprint-extra-contrast",
        action="store_true",
        help=(
            "run an unperturbed replay from the baseline membership for each candidate/replay "
            "and compare perturbed membership against that extra replay instead of only the baseline"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--drilldown-only",
        action="store_true",
        help="reuse existing trajectory outputs and regenerate only target parent drilldown artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.drilldown_only:
        paths = run_parent_drilldown_only(args)
    else:
        paths = replay_candidate_trajectories(args)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
