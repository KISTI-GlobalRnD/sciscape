#!/usr/bin/env python3
"""Pilot closure-frontier shrink edits for endpoint-near basin transitions.

This diagnostic starts from a recreated vanilla endpoint, applies cumulative
direct-node closure splits selected by the frontier ledger, and measures exact
quality/support costs. It is not a production Leiden or Dongdaemun policy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from analyze_leiden_basin_transition_boundaries import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_BOUNDARY_DIR,
    NODE_ROWS_FILENAME,
)
from collect_leiden_vanilla_reachability_sweep import (  # noqa: E402
    _load_graph,
    _read_candidate_rows,
    compatible_sketch_nodes,
    encode_membership_sketch,
    hash_u32_sequence,
)
from rank_leiden_basin_transition_closure_frontier import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_FRONTIER_DIR,
    FRONTIER_ROWS_FILENAME,
    MODE_LABEL_COLUMNS,
    PAIR_COLUMNS,
)
from run_leiden_basin_transition_operator_pilot import (  # noqa: E402
    DEFAULT_CANDIDATE_DIRS,
    DEFAULT_VANILLA_DIR,
    VANILLA_ROWS_FILENAME,
    CandidateMembership,
    RecreatedMembership,
    _find_candidate_row,
    _find_vanilla_row,
    _recreate_candidate,
    _run_leiden,
    _safe_int,
    changed_support_nodes,
    endpoint_distance,
    fixed_outside,
    support_distance,
)
from run_leiden_hysteresis_work_acceleration_monitor import (  # noqa: E402
    _compact_membership,
)


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "research/consensus/results/adaptive_refinement/"
    "leiden_multibasin_crossfield_budget12_support_20260519/combined_with_field30/"
    "basin_transition_closure_operator_pilot_field34_cc"
)
ROWS_FILENAME = "basin_transition_closure_operator_rows.csv"
SUMMARY_FILENAME = "basin_transition_closure_operator_summary.json"
REPORT_FILENAME = "basin_transition_closure_operator_report.md"

SUPPORTED_OPERATORS = ("closure_split_shrink_from_vanilla",)
FRESH_RAW_OPERATOR = "closure_split_shrink_from_vanilla_fresh_raw"
FRESH_POLISH_OPERATOR = "closure_split_shrink_from_vanilla_fresh_direct_polish"
NEAREST_RAW_OPERATOR = "closure_split_shrink_from_vanilla_candidate_nearest_raw"
NEAREST_POLISH_OPERATOR = (
    "closure_split_shrink_from_vanilla_candidate_nearest_direct_polish"
)
CLOSURE_OPERATOR_NAMES = (
    FRESH_RAW_OPERATOR,
    FRESH_POLISH_OPERATOR,
    NEAREST_RAW_OPERATOR,
    NEAREST_POLISH_OPERATOR,
)


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _truthy_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _pair_mask(frame: pd.DataFrame, pair: pd.Series | dict[str, Any]) -> np.ndarray:
    mask = np.ones(len(frame), dtype=np.bool_)
    for column in PAIR_COLUMNS:
        value = pair[column]
        if column == "vanilla_requested_n_iterations":
            mask &= frame[column].astype(str).eq(str(value)).to_numpy()
        elif column == "vanilla_randomness":
            mask &= np.isclose(
                pd.to_numeric(frame[column], errors="coerce"),
                float(value),
            )
        elif column in {"candidate_index", "vanilla_seed"}:
            mask &= (
                pd.to_numeric(frame[column], errors="coerce") == int(value)
            ).to_numpy()
        else:
            mask &= frame[column].astype(str).eq(str(value)).to_numpy()
    return mask


def selected_frontier_rows(
    frontier_rows: pd.DataFrame,
    *,
    closure_mode: str,
    max_pairs: int,
    max_labels_per_pair: int,
) -> pd.DataFrame:
    if frontier_rows.empty:
        return frontier_rows.copy()
    rows = frontier_rows.copy()
    rows = rows[
        _truthy_series(rows["frontier_selected"])
        & rows["closure_mode"].astype(str).eq(str(closure_mode))
    ].copy()
    if rows.empty:
        return rows
    rows["frontier_rank_in_pair"] = pd.to_numeric(
        rows["frontier_rank_in_pair"],
        errors="coerce",
    )
    rows["frontier_score"] = pd.to_numeric(rows["frontier_score"], errors="coerce")
    rows = rows.sort_values(
        [*PAIR_COLUMNS, "frontier_rank_in_pair", "frontier_score"],
        ascending=[*([True] * len(PAIR_COLUMNS)), True, False],
    )
    rows = rows.groupby(PAIR_COLUMNS, dropna=False).head(int(max_labels_per_pair))
    pair_keys = rows[PAIR_COLUMNS].drop_duplicates().head(int(max_pairs))
    if pair_keys.empty:
        return rows.iloc[0:0].copy()
    keep = np.zeros(len(rows), dtype=np.bool_)
    for _, pair in pair_keys.iterrows():
        keep |= _pair_mask(rows, pair)
    return rows[keep].reset_index(drop=True)


def direct_nodes_for_frontier_row(
    *,
    node_rows: pd.DataFrame,
    frontier_row: pd.Series,
) -> np.ndarray:
    """Return direct vanilla-extra nodes represented by a frontier label."""
    mode = str(frontier_row["closure_mode"])
    label_column = MODE_LABEL_COLUMNS.get(mode)
    if label_column is None:
        raise ValueError(f"Unsupported closure mode: {mode}")
    mask = _pair_mask(node_rows, frontier_row)
    mask &= node_rows["support_class"].astype(str).eq("vanilla_extra").to_numpy()
    mask &= (
        pd.to_numeric(node_rows[label_column], errors="coerce")
        == int(frontier_row["closure_label"])
    ).to_numpy()
    return np.asarray(sorted(set(node_rows.loc[mask, "node"].astype(int))), dtype=np.uint32)


def split_nodes_to_fresh_donor_labels(
    membership: np.ndarray,
    donor_membership: np.ndarray,
    nodes: np.ndarray,
    *,
    label_map: dict[int, int] | None = None,
    next_label: int | None = None,
) -> tuple[np.ndarray, dict[int, int], int]:
    """Split selected nodes into fresh labels that preserve donor coassignment."""
    out = np.asarray(membership, dtype=np.uint64).copy()
    donor = np.asarray(donor_membership, dtype=np.uint64)
    mapping: dict[int, int] = dict(label_map or {})
    next_out_label = (
        int(next_label)
        if next_label is not None
        else int(out.max(initial=0)) + 1
    )
    for node_raw in np.asarray(nodes, dtype=np.int64):
        node = int(node_raw)
        donor_label = int(donor[node])
        target = mapping.get(donor_label)
        if target is None:
            target = next_out_label
            mapping[donor_label] = target
            next_out_label += 1
        out[node] = np.uint64(target)
    return out, mapping, next_out_label


def assign_nodes_to_nearest_existing_donor_label(
    membership: np.ndarray,
    donor_membership: np.ndarray,
    nodes: np.ndarray,
    *,
    blocked_nodes: np.ndarray,
    next_label: int | None = None,
) -> tuple[np.ndarray, int]:
    """Move nodes to the existing label that best represents each donor label."""
    out = np.asarray(membership, dtype=np.uint64).copy()
    donor = np.asarray(donor_membership, dtype=np.uint64)
    blocked = np.asarray(blocked_nodes, dtype=np.int64)
    blocked_mask = np.zeros(out.size, dtype=np.bool_)
    if blocked.size:
        blocked_mask[blocked] = True
    next_out_label = (
        int(next_label)
        if next_label is not None
        else int(out.max(initial=0)) + 1
    )
    target_by_donor: dict[int, int] = {}
    for donor_label_raw in np.unique(donor[np.asarray(nodes, dtype=np.int64)]):
        donor_label = int(donor_label_raw)
        context_mask = (donor == np.uint64(donor_label)) & ~blocked_mask
        context_labels, counts = np.unique(out[context_mask], return_counts=True)
        if context_labels.size:
            best_index = int(np.lexsort((context_labels, -counts))[0])
            target_by_donor[donor_label] = int(context_labels[best_index])
        else:
            target_by_donor[donor_label] = next_out_label
            next_out_label += 1
    for node_raw in np.asarray(nodes, dtype=np.int64):
        node = int(node_raw)
        out[node] = np.uint64(target_by_donor[int(donor[node])])
    return out, next_out_label


def _score_membership(
    graph: Any,
    membership: np.ndarray,
    *,
    resolution: float,
) -> RecreatedMembership:
    start = time.perf_counter()
    quality = float(graph.cpm_quality(membership, resolution=float(resolution)))
    return RecreatedMembership(
        membership=np.asarray(membership, dtype=np.uint64),
        quality=quality,
        elapsed_sec=float(time.perf_counter() - start),
    )


def _evaluate_result(
    *,
    context: dict[str, Any],
    operator: str,
    result: RecreatedMembership,
    baseline: RecreatedMembership,
    candidate: CandidateMembership,
    vanilla: RecreatedMembership,
    candidate_support: np.ndarray,
    vanilla_support: np.ndarray,
    sketch_nodes: np.ndarray,
    released_stats: dict[str, Any],
) -> dict[str, Any]:
    result_support = changed_support_nodes(baseline.membership, result.membership)
    dist_candidate, inter_candidate, union_candidate = support_distance(
        result_support,
        candidate_support,
    )
    dist_vanilla, inter_vanilla, union_vanilla = support_distance(
        result_support,
        vanilla_support,
    )
    elapsed = float(result.elapsed_sec)
    delta_vs_vanilla = float(result.quality - vanilla.quality)
    return {
        **context,
        "operator": operator,
        "quality": float(result.quality),
        "baseline_quality": float(baseline.quality),
        "candidate_quality": float(candidate.recreated.quality),
        "vanilla_quality": float(vanilla.quality),
        "delta_vs_baseline": float(result.quality - baseline.quality),
        "delta_vs_candidate": float(result.quality - candidate.recreated.quality),
        "delta_vs_vanilla": delta_vs_vanilla,
        "quality_debt_vs_vanilla": float(vanilla.quality - result.quality),
        "elapsed_sec": elapsed,
        "gain_per_second_vs_vanilla": (
            delta_vs_vanilla / elapsed if elapsed > 0.0 else math.nan
        ),
        "candidate_support_size": int(candidate_support.size),
        "vanilla_support_size": int(vanilla_support.size),
        "result_support_size": int(result_support.size),
        "support_burden_reduction_vs_vanilla": int(
            vanilla_support.size - result_support.size
        ),
        "support_burden_gap_vs_candidate": int(
            result_support.size - candidate_support.size
        ),
        "result_support_distance_to_candidate": dist_candidate,
        "result_support_intersection_with_candidate": inter_candidate,
        "result_support_union_with_candidate": union_candidate,
        "result_support_distance_to_vanilla": dist_vanilla,
        "result_support_intersection_with_vanilla": inter_vanilla,
        "result_support_union_with_vanilla": union_vanilla,
        "result_endpoint_distance_to_candidate": endpoint_distance(
            result.membership,
            candidate.recreated.membership,
            sketch_nodes,
        ),
        "result_endpoint_distance_to_vanilla": endpoint_distance(
            result.membership,
            vanilla.membership,
            sketch_nodes,
        ),
        "sketch_node_hash": hash_u32_sequence(sketch_nodes),
        "sketch_membership": encode_membership_sketch(result.membership, sketch_nodes),
        **released_stats,
    }


def _zero_release_stats() -> dict[str, Any]:
    return {
        "step_index": 0,
        "released_label_count": 0,
        "latest_closure_label": "",
        "released_closure_labels": "",
        "released_direct_node_count": 0,
        "released_closure_node_count": 0,
        "released_context_extra_count": 0,
        "released_outside_support_count": 0,
        "released_direct_node_weight": 0.0,
        "released_frontier_score_sum": 0.0,
        "released_max_frontier_rank": 0,
        "mutable_node_count": 0,
        "target_donor": "",
        "fixed_outside_mutable": False,
    }


def diagnostic_label_for_row(
    row: pd.Series,
    *,
    material_delta: float = 1e-9,
    min_support_shift_from_vanilla: float = 0.1,
) -> str:
    if str(row["operator"]) not in CLOSURE_OPERATOR_NAMES:
        return "control"
    if float(row["delta_vs_vanilla"]) < -float(material_delta):
        return "quality_loss"
    if float(row["delta_vs_control_extra"]) < -float(material_delta):
        return "seed_control_dominates"
    if (
        float(row["result_support_distance_to_vanilla"])
        < float(min_support_shift_from_vanilla)
    ):
        return "quality_win_same_basin"
    return "quality_win_support_shift"


def _control_rows_for_pair(
    *,
    graph: Any,
    baseline: RecreatedMembership,
    candidate: CandidateMembership,
    vanilla: RecreatedMembership,
    candidate_support: np.ndarray,
    vanilla_support: np.ndarray,
    sketch_nodes: np.ndarray,
    context: dict[str, Any],
    resolution: float,
    randomness: float,
    transition_iterations: int,
    operator_seed: int,
) -> list[dict[str, Any]]:
    controls = {
        "baseline_recreated": baseline,
        "candidate_recreated": candidate.recreated,
        "vanilla_recreated": vanilla,
        "control_extra_from_baseline": _run_leiden(
            graph,
            resolution=resolution,
            seed=operator_seed,
            n_iterations=transition_iterations,
            randomness=randomness,
            initial_membership=baseline.membership,
        ),
    }
    release_stats = _zero_release_stats()
    return [
        _evaluate_result(
            context=context,
            operator=operator,
            result=result,
            baseline=baseline,
            candidate=candidate,
            vanilla=vanilla,
            candidate_support=candidate_support,
            vanilla_support=vanilla_support,
            sketch_nodes=sketch_nodes,
            released_stats=release_stats,
        )
        for operator, result in controls.items()
    ]


def _operator_rows_for_pair(
    *,
    graph: Any,
    baseline: RecreatedMembership,
    candidate: CandidateMembership,
    vanilla: RecreatedMembership,
    frontier_rows: pd.DataFrame,
    node_rows: pd.DataFrame,
    sketch_nodes: np.ndarray,
    context: dict[str, Any],
    resolution: float,
    randomness: float,
    local_polish_iterations: int,
    operator_seed: int,
) -> list[dict[str, Any]]:
    candidate_support = candidate.support_nodes
    vanilla_support = changed_support_nodes(baseline.membership, vanilla.membership)
    rows: list[dict[str, Any]] = []

    for strategy, raw_operator, polish_operator in [
        ("candidate_fresh", FRESH_RAW_OPERATOR, FRESH_POLISH_OPERATOR),
        (
            "candidate_nearest_existing",
            NEAREST_RAW_OPERATOR,
            NEAREST_POLISH_OPERATOR,
        ),
    ]:
        edited = np.asarray(vanilla.membership, dtype=np.uint64).copy()
        label_map: dict[int, int] = {}
        next_label = int(edited.max(initial=0)) + 1
        mutable_nodes: set[int] = set()
        released_labels: list[int] = []
        released_closure_nodes = 0
        released_context_extra = 0
        released_outside_support = 0
        released_direct_weight = 0.0
        released_score = 0.0

        for step_index, (_, frontier) in enumerate(frontier_rows.iterrows(), start=1):
            nodes = direct_nodes_for_frontier_row(
                node_rows=node_rows,
                frontier_row=frontier,
            )
            if nodes.size == 0:
                continue
            next_mutable_nodes = set(mutable_nodes)
            next_mutable_nodes.update(int(node) for node in nodes)
            mutable_array = np.asarray(sorted(next_mutable_nodes), dtype=np.uint32)
            if strategy == "candidate_fresh":
                edited, label_map, next_label = split_nodes_to_fresh_donor_labels(
                    edited,
                    candidate.recreated.membership,
                    nodes,
                    label_map=label_map,
                    next_label=next_label,
                )
                label_map = {}
            else:
                edited, next_label = assign_nodes_to_nearest_existing_donor_label(
                    edited,
                    candidate.recreated.membership,
                    nodes,
                    blocked_nodes=mutable_array,
                    next_label=next_label,
                )
            edited = _compact_membership(edited)
            next_label = int(edited.max(initial=0)) + 1
            mutable_nodes = next_mutable_nodes
            released_labels.append(int(frontier["closure_label"]))
            released_closure_nodes += int(frontier["closure_node_count"])
            released_context_extra += int(frontier["closure_context_extra_count"])
            released_outside_support += int(frontier["closure_outside_support_count"])
            released_direct_weight += _safe_float(
                frontier.get("direct_node_weight_sum"),
                0.0,
            )
            released_score += _safe_float(frontier.get("frontier_score"), 0.0)
            release_stats = {
                "step_index": int(step_index),
                "released_label_count": int(len(released_labels)),
                "latest_closure_label": int(frontier["closure_label"]),
                "released_closure_labels": ",".join(
                    str(label) for label in released_labels
                ),
                "released_direct_node_count": int(mutable_array.size),
                "released_closure_node_count": int(released_closure_nodes),
                "released_context_extra_count": int(released_context_extra),
                "released_outside_support_count": int(released_outside_support),
                "released_direct_node_weight": float(released_direct_weight),
                "released_frontier_score_sum": float(released_score),
                "released_max_frontier_rank": int(frontier["frontier_rank_in_pair"]),
                "mutable_node_count": int(mutable_array.size),
                "target_donor": strategy,
                "fixed_outside_mutable": False,
            }
            raw = _score_membership(graph, edited, resolution=resolution)
            rows.append(
                _evaluate_result(
                    context=context,
                    operator=raw_operator,
                    result=raw,
                    baseline=baseline,
                    candidate=candidate,
                    vanilla=vanilla,
                    candidate_support=candidate_support,
                    vanilla_support=vanilla_support,
                    sketch_nodes=sketch_nodes,
                    released_stats=release_stats,
                )
            )
            if int(local_polish_iterations) <= 0:
                continue
            fixed = fixed_outside(int(edited.size), mutable_array)
            polished = _run_leiden(
                graph,
                resolution=resolution,
                seed=operator_seed + step_index,
                n_iterations=local_polish_iterations,
                randomness=randomness,
                initial_membership=edited,
                fixed_nodes=fixed,
            )
            polished_stats = {**release_stats, "fixed_outside_mutable": True}
            rows.append(
                _evaluate_result(
                    context=context,
                    operator=polish_operator,
                    result=polished,
                    baseline=baseline,
                    candidate=candidate,
                    vanilla=vanilla,
                    candidate_support=candidate_support,
                    vanilla_support=vanilla_support,
                    sketch_nodes=sketch_nodes,
                    released_stats=polished_stats,
                )
            )
    return rows


def run_pilot(
    *,
    frontier_dir: Path,
    boundary_dir: Path,
    candidate_dirs: tuple[Path, ...],
    vanilla_dir: Path,
    output_dir: Path,
    closure_mode: str,
    max_pairs: int,
    max_labels_per_pair: int,
    baseline_iterations: int,
    transition_iterations: int,
    polish_iterations: int,
    local_polish_iterations: int,
    resolution: float,
    randomness: float,
    perturb_seed_offset: int,
) -> dict[str, Any]:
    if closure_mode not in MODE_LABEL_COLUMNS:
        raise ValueError(f"Unsupported closure mode: {closure_mode}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frontier_rows = pd.read_csv(frontier_dir / FRONTIER_ROWS_FILENAME)
    node_rows = pd.read_csv(boundary_dir / NODE_ROWS_FILENAME)
    selected = selected_frontier_rows(
        frontier_rows,
        closure_mode=closure_mode,
        max_pairs=max_pairs,
        max_labels_per_pair=max_labels_per_pair,
    )
    if selected.empty:
        raise ValueError("No selected closure frontier rows")
    candidates = _read_candidate_rows(candidate_dirs)
    vanilla_rows = pd.read_csv(vanilla_dir / VANILLA_ROWS_FILENAME)

    baseline_cache: dict[str, RecreatedMembership] = {}
    candidate_cache: dict[tuple[str, int], CandidateMembership] = {}
    vanilla_cache: dict[tuple[str, int, float, str], RecreatedMembership] = {}
    graph_cache: dict[str, tuple[Any, np.ndarray, Any]] = {}
    out_rows: list[dict[str, Any]] = []
    pair_rows = selected[PAIR_COLUMNS].drop_duplicates().reset_index(drop=True)

    for _, pair in pair_rows.iterrows():
        case = str(pair["case"])
        candidate_index = int(pair["candidate_index"])
        seed = int(pair["vanilla_seed"])
        vanilla_randomness = float(pair["vanilla_randomness"])
        vanilla_n = str(pair["vanilla_requested_n_iterations"])
        candidate_row = _find_candidate_row(
            candidates,
            case=case,
            candidate_index=candidate_index,
        )
        vanilla_row = _find_vanilla_row(
            vanilla_rows,
            case=case,
            seed=seed,
            randomness=vanilla_randomness,
            n_iterations=vanilla_n,
        )
        graph_dir = Path(str(vanilla_row["graph_dir"]))
        graph_key = str(graph_dir)
        if graph_key not in graph_cache:
            graph_cache[graph_key] = _load_graph(graph_dir)
        graph, node_weights, arrays = graph_cache[graph_key]
        if case not in baseline_cache:
            baseline_cache[case] = _run_leiden(
                graph,
                resolution=resolution,
                seed=int(candidate_row.get("seed", 0)),
                n_iterations=baseline_iterations,
                randomness=randomness,
            )
        baseline = baseline_cache[case]
        ckey = (case, candidate_index)
        if ckey not in candidate_cache:
            candidate_cache[ckey] = _recreate_candidate(
                graph=graph,
                arrays=arrays,
                node_weights=node_weights,
                baseline_membership=baseline.membership,
                baseline_quality=baseline.quality,
                row=candidate_row,
                resolution=resolution,
                randomness=randomness,
                perturb_seed_offset=perturb_seed_offset,
                polish_iterations=polish_iterations,
            )
        candidate = candidate_cache[ckey]
        vkey = (case, seed, vanilla_randomness, vanilla_n)
        if vkey not in vanilla_cache:
            vanilla_cache[vkey] = _run_leiden(
                graph,
                resolution=resolution,
                seed=seed,
                n_iterations=int(
                    _safe_int(vanilla_n, baseline_iterations) or baseline_iterations
                ),
                randomness=vanilla_randomness,
            )
        vanilla = vanilla_cache[vkey]
        sketch_nodes, sketch_context = compatible_sketch_nodes(
            arrays=arrays,
            baseline_membership=baseline.membership,
            node_weights=node_weights,
            candidate_rows=candidates[candidates["case"].astype(str) == case],
        )
        if not bool(sketch_context.get("sketch_context_hash_matches_candidate", False)):
            raise RuntimeError(f"sketch context mismatch for {case}")

        pair_frontier = selected[_pair_mask(selected, pair)].copy()
        pair_frontier = pair_frontier.sort_values(
            ["frontier_rank_in_pair", "frontier_score"],
            ascending=[True, False],
        )
        candidate_support = candidate.support_nodes
        vanilla_support = changed_support_nodes(baseline.membership, vanilla.membership)
        operator_seed = (
            int(candidate_row.get("seed", 0))
            + int(perturb_seed_offset)
            + int(candidate_index)
        )
        context = {
            "case": case,
            "field": pair["field"],
            "method": pair["method"],
            "candidate_index": candidate_index,
            "vanilla_seed": seed,
            "vanilla_randomness": vanilla_randomness,
            "vanilla_requested_n_iterations": vanilla_n,
            "closure_mode": closure_mode,
            "baseline_iterations": int(baseline_iterations),
            "transition_iterations": int(transition_iterations),
            "polish_iterations": int(polish_iterations),
            "local_polish_iterations": int(local_polish_iterations),
            "resolution": float(resolution),
            "randomness": float(randomness),
        }
        out_rows.extend(
            _control_rows_for_pair(
                graph=graph,
                baseline=baseline,
                candidate=candidate,
                vanilla=vanilla,
                candidate_support=candidate_support,
                vanilla_support=vanilla_support,
                sketch_nodes=sketch_nodes,
                context=context,
                resolution=resolution,
                randomness=randomness,
                transition_iterations=transition_iterations,
                operator_seed=operator_seed,
            )
        )
        out_rows.extend(
            _operator_rows_for_pair(
                graph=graph,
                baseline=baseline,
                candidate=candidate,
                vanilla=vanilla,
                frontier_rows=pair_frontier,
                node_rows=node_rows,
                sketch_nodes=sketch_nodes,
                context=context,
                resolution=resolution,
                randomness=randomness,
                local_polish_iterations=local_polish_iterations,
                operator_seed=operator_seed,
            )
        )

    rows = pd.DataFrame(out_rows)
    if not rows.empty:
        control_quality = rows[
            rows["operator"].eq("control_extra_from_baseline")
        ][[*PAIR_COLUMNS, "quality"]].rename(
            columns={"quality": "control_extra_quality"}
        )
        rows = rows.merge(control_quality, on=PAIR_COLUMNS, how="left")
        rows["delta_vs_control_extra"] = (
            rows["quality"] - rows["control_extra_quality"]
        )
        rows["diagnostic_label"] = [
            diagnostic_label_for_row(row) for _, row in rows.iterrows()
        ]
    rows.to_csv(output_dir / ROWS_FILENAME, index=False)
    selected_operator_rows = rows[rows["operator"].isin(CLOSURE_OPERATOR_NAMES)]
    summary = {
        "schema": "leiden_basin_transition_closure_operator_pilot.v1",
        "frontier_dir": str(frontier_dir),
        "boundary_dir": str(boundary_dir),
        "vanilla_dir": str(vanilla_dir),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "output_dir": str(output_dir),
        "closure_mode": closure_mode,
        "selected_pair_count": int(len(pair_rows)),
        "selected_frontier_rows": int(len(selected)),
        "operator_rows": int(len(rows)),
        "closure_operator_rows": int(len(selected_operator_rows)),
        "baseline_iterations": int(baseline_iterations),
        "transition_iterations": int(transition_iterations),
        "polish_iterations": int(polish_iterations),
        "local_polish_iterations": int(local_polish_iterations),
        "resolution": float(resolution),
        "randomness": float(randomness),
    }
    if not selected_operator_rows.empty:
        summary["best_operator_delta_vs_vanilla"] = float(
            selected_operator_rows["delta_vs_vanilla"].max()
        )
        summary["best_operator_delta_vs_control_extra"] = float(
            selected_operator_rows["delta_vs_control_extra"].max()
        )
        summary["best_operator_support_reduction_vs_vanilla"] = int(
            selected_operator_rows["support_burden_reduction_vs_vanilla"].max()
        )
        summary["best_operator_support_distance_to_candidate"] = float(
            selected_operator_rows.loc[
                selected_operator_rows["delta_vs_vanilla"].idxmax(),
                "result_support_distance_to_candidate",
            ]
        )
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(output_dir / REPORT_FILENAME, rows, summary)
    return summary


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 32) -> list[str]:
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


def write_report(path: Path, rows: pd.DataFrame, summary: dict[str, Any]) -> None:
    lines = [
        "# Closure Operator Pilot",
        "",
        "This diagnostic applies cumulative direct-node closure splits from the recreated vanilla endpoint.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in [
        "selected_pair_count",
        "selected_frontier_rows",
        "operator_rows",
        "closure_operator_rows",
        "best_operator_delta_vs_vanilla",
        "best_operator_delta_vs_control_extra",
        "best_operator_support_reduction_vs_vanilla",
        "best_operator_support_distance_to_candidate",
    ]:
        lines.append(f"| {key} | {summary.get(key, '')} |")
    lines.extend(["", "## Operator Summary", ""])
    if not rows.empty:
        operator_summary = (
            rows.groupby("operator", as_index=False)
            .agg(
                rows=("operator", "size"),
                delta_vs_vanilla_max=("delta_vs_vanilla", "max"),
                delta_vs_vanilla_median=("delta_vs_vanilla", "median"),
                delta_vs_control_extra_max=("delta_vs_control_extra", "max"),
                delta_vs_control_extra_median=("delta_vs_control_extra", "median"),
                support_reduction_vs_vanilla_max=(
                    "support_burden_reduction_vs_vanilla",
                    "max",
                ),
                support_reduction_vs_vanilla_median=(
                    "support_burden_reduction_vs_vanilla",
                    "median",
                ),
                support_distance_to_candidate_median=(
                    "result_support_distance_to_candidate",
                    "median",
                ),
                endpoint_distance_to_candidate_median=(
                    "result_endpoint_distance_to_candidate",
                    "median",
                ),
                elapsed_sec_median=("elapsed_sec", "median"),
            )
            .sort_values("operator")
        )
        lines.extend(_markdown_table(operator_summary, max_rows=40))
    lines.extend(["", "## Closure Steps", ""])
    display_cols = [
        "candidate_index",
        "vanilla_seed",
        "vanilla_randomness",
        "operator",
        "step_index",
        "latest_closure_label",
        "released_label_count",
        "released_direct_node_count",
        "released_context_extra_count",
        "delta_vs_vanilla",
        "delta_vs_control_extra",
        "diagnostic_label",
        "quality_debt_vs_vanilla",
        "support_burden_reduction_vs_vanilla",
        "result_support_distance_to_candidate",
        "result_endpoint_distance_to_candidate",
        "elapsed_sec",
    ]
    closure_rows = rows[rows["operator"].isin(CLOSURE_OPERATOR_NAMES)]
    lines.extend(_markdown_table(closure_rows[[c for c in display_cols if c in closure_rows.columns]], max_rows=60))
    lines.extend(["", "## Diagnostic Labels", ""])
    if not closure_rows.empty:
        labels = (
            closure_rows.groupby(["operator", "diagnostic_label"], as_index=False)
            .agg(rows=("operator", "size"))
            .sort_values(["operator", "diagnostic_label"])
        )
        lines.extend(_markdown_table(labels, max_rows=60))
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- A row is useful only if support burden falls without quality debt that is worse than vanilla or seed controls.",
            "- Raw direct splits diagnose the support lower bound; direct polish checks whether that split survives local optimization.",
            "- These rows remain Dongdaemun diagnostics, not a default refinement policy.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-dir", type=Path, default=DEFAULT_FRONTIER_DIR)
    parser.add_argument("--boundary-dir", type=Path, default=DEFAULT_BOUNDARY_DIR)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument("--vanilla-dir", type=Path, default=DEFAULT_VANILLA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--closure-mode", default="candidate_label")
    parser.add_argument("--max-pairs", type=int, default=5)
    parser.add_argument("--max-labels-per-pair", type=int, default=10)
    parser.add_argument("--baseline-iterations", type=int, default=10)
    parser.add_argument("--transition-iterations", type=int, default=5)
    parser.add_argument("--polish-iterations", type=int, default=5)
    parser.add_argument("--local-polish-iterations", type=int, default=3)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--randomness", type=float, default=0.01)
    parser.add_argument("--perturb-seed-offset", type=int, default=5000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_pilot(
        frontier_dir=args.frontier_dir,
        boundary_dir=args.boundary_dir,
        candidate_dirs=tuple(args.candidate_dir or DEFAULT_CANDIDATE_DIRS),
        vanilla_dir=args.vanilla_dir,
        output_dir=args.output_dir,
        closure_mode=args.closure_mode,
        max_pairs=args.max_pairs,
        max_labels_per_pair=args.max_labels_per_pair,
        baseline_iterations=args.baseline_iterations,
        transition_iterations=args.transition_iterations,
        polish_iterations=args.polish_iterations,
        local_polish_iterations=args.local_polish_iterations,
        resolution=args.resolution,
        randomness=args.randomness,
        perturb_seed_offset=args.perturb_seed_offset,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
