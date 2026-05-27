#!/usr/bin/env python3
"""Reusable diagnostics for ordered Leiden basin flip profiling.

This module provides calculation kernels for research artifact runners and
future experimental operators. It profiles raw flip frontiers and beam
trajectories; it does not accept edits or run a production operator by itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

SCORING_POLICIES = ("q_first", "progress_first", "balanced")
UNIT_TYPE_LABEL_INTERSECTION = "label_intersection_block"
BARRIER_GREEDY_VISIBLE = "greedy_visible"
BARRIER_Q_GREEDY_MISS = "q_greedy_miss"
BARRIER_PROGRESS_GREEDY_MISS = "progress_greedy_miss"
BARRIER_CLOSURE_COMPOUND_MISS = "closure_compound_miss"
BARRIER_POLISH_RECOVERY_MISS = "polish_recovery_miss"
BARRIER_FAILURE_LABELS = (
    BARRIER_Q_GREEDY_MISS,
    BARRIER_PROGRESS_GREEDY_MISS,
    BARRIER_CLOSURE_COMPOUND_MISS,
    BARRIER_POLISH_RECOVERY_MISS,
)
POLISH_RESULT_RECOVERED_SHIFT = "recovered_support_shift"
POLISH_RESULT_RECOVERED_VANILLA_NEAR = "recovered_vanilla_near"
POLISH_RESULT_QUALITY_LOSS = "quality_loss"
POLISH_RESULT_RAW_ONLY = "raw_only"


@dataclass(frozen=True)
class BeamState:
    state_id: str
    scoring_policy: str
    step_index: int
    membership: np.ndarray
    quality: float
    selected_unit_ids: tuple[str, ...]
    flipped_nodes: frozenset[int]
    raw_barrier_so_far: float
    label_map: dict[int, int]
    next_label: int


def _node_csv(nodes: np.ndarray) -> str:
    return ",".join(str(int(node)) for node in np.asarray(nodes, dtype=np.uint32))


def parse_node_ids(value: Any) -> np.ndarray:
    text = str(value)
    if not text or text.lower() == "nan":
        return np.asarray([], dtype=np.uint32)
    return np.asarray([int(part) for part in text.split(",") if part], dtype=np.uint32)


def parse_unit_ids(value: Any) -> tuple[str, ...]:
    text = str(value)
    if not text or text.lower() == "nan":
        return ()
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _best_partner_maps(
    left_membership: np.ndarray,
    right_membership: np.ndarray,
) -> tuple[dict[int, int], dict[int, int]]:
    left = np.asarray(left_membership, dtype=np.uint64)
    right = np.asarray(right_membership, dtype=np.uint64)
    if left.shape != right.shape:
        raise ValueError("memberships must have the same shape")
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


def changed_support_nodes(
    baseline: np.ndarray,
    membership: np.ndarray,
) -> np.ndarray:
    baseline = np.asarray(baseline, dtype=np.uint64)
    membership = np.asarray(membership, dtype=np.uint64)
    if baseline.shape != membership.shape:
        raise ValueError("baseline and membership must have the same shape")
    baseline_best, membership_best = _best_partner_maps(baseline, membership)
    changed: list[int] = []
    for node, (base_label, label) in enumerate(zip(baseline, membership, strict=False)):
        base = int(base_label)
        current = int(label)
        baseline_aligned = baseline_best.get(base) == current
        membership_aligned = membership_best.get(current) == base
        if not (baseline_aligned and membership_aligned):
            changed.append(node)
    return np.asarray(changed, dtype=np.uint32)


def support_distance(left: np.ndarray, right: np.ndarray) -> tuple[float, int, int]:
    left_set = set(np.asarray(left, dtype=np.uint32).tolist())
    right_set = set(np.asarray(right, dtype=np.uint32).tolist())
    union = len(left_set | right_set)
    intersection = len(left_set & right_set)
    if union == 0:
        return 0.0, 0, 0
    return 1.0 - float(intersection) / float(union), intersection, union


def _coassignment_bits(membership: np.ndarray) -> np.ndarray:
    labels = np.asarray(membership, dtype=np.uint64)
    if labels.size < 2:
        return np.asarray([], dtype=np.bool_)
    bits: list[bool] = []
    for i in range(int(labels.size) - 1):
        bits.extend((labels[i] == labels[i + 1 :]).tolist())
    return np.asarray(bits, dtype=np.bool_)


def endpoint_distance(
    left_membership: np.ndarray,
    right_membership: np.ndarray,
    sketch_nodes: np.ndarray,
) -> float:
    nodes = np.asarray(sketch_nodes, dtype=np.int64)
    if nodes.size == 0:
        return math.nan
    left = np.asarray(left_membership, dtype=np.uint64)[nodes]
    right = np.asarray(right_membership, dtype=np.uint64)[nodes]
    left_bits = _coassignment_bits(left)
    right_bits = _coassignment_bits(right)
    if left_bits.size == 0 or left_bits.size != right_bits.size:
        return math.nan
    return float(np.mean(left_bits != right_bits))


def fresh_group_transplant(
    membership: np.ndarray,
    donor_membership: np.ndarray,
    nodes: np.ndarray,
    *,
    label_map: dict[int, int] | None = None,
    next_label: int | None = None,
) -> tuple[np.ndarray, dict[int, int], int]:
    """Force donor labels into a fresh namespace for edited nodes."""
    out = np.asarray(membership, dtype=np.uint64).copy()
    donor = np.asarray(donor_membership, dtype=np.uint64)
    mapping: dict[int, int] = dict(label_map or {})
    next_out_label = (
        int(next_label)
        if next_label is not None
        else int(out.max(initial=0)) + 1
    )
    for node in np.asarray(nodes, dtype=np.int64):
        donor_label = int(donor[int(node)])
        target = mapping.get(donor_label)
        if target is None:
            target = next_out_label
            mapping[donor_label] = target
            next_out_label += 1
        out[int(node)] = np.uint64(target)
    return out, mapping, next_out_label


def compact_membership(membership: np.ndarray) -> np.ndarray:
    labels = np.asarray(membership, dtype=np.uint64)
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.astype(np.uint64, copy=False)


def fixed_outside(node_count: int, mutable_nodes: np.ndarray) -> np.ndarray:
    fixed = np.ones(int(node_count), dtype=np.bool_)
    fixed[np.asarray(mutable_nodes, dtype=np.int64)] = False
    return fixed


def apply_prefix_units(
    *,
    membership: np.ndarray,
    donor_membership: np.ndarray,
    units: pd.DataFrame,
    prefix_unit_ids: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a comma-separated prefix of unit IDs using fresh donor labels."""
    unit_ids = parse_unit_ids(prefix_unit_ids)
    if not unit_ids:
        return np.asarray(membership, dtype=np.uint64).copy(), np.asarray([], dtype=np.uint32)
    units_by_id = {str(row["unit_id"]): row for _, row in units.iterrows()}
    out = np.asarray(membership, dtype=np.uint64).copy()
    label_map: dict[int, int] = {}
    next_label = int(out.max(initial=0)) + 1
    mutable_nodes: list[int] = []
    for unit_id in unit_ids:
        if unit_id not in units_by_id:
            raise KeyError(f"Missing unit_id in units table: {unit_id}")
        unit = units_by_id[unit_id]
        nodes = parse_node_ids(unit["node_ids"])
        out, label_map, next_label = fresh_group_transplant(
            out,
            donor_membership,
            nodes,
            label_map=label_map,
            next_label=next_label,
        )
        mutable_nodes.extend(int(node) for node in nodes)
    return out, np.asarray(sorted(set(mutable_nodes)), dtype=np.uint32)


def membership_metric_row(
    *,
    membership: np.ndarray,
    quality: float,
    baseline_membership: np.ndarray,
    candidate_membership: np.ndarray,
    vanilla_membership: np.ndarray,
    sketch_nodes: np.ndarray,
    start_quality: float,
    candidate_quality: float,
    vanilla_quality: float,
    prefix: str,
) -> dict[str, Any]:
    result_support = changed_support_nodes(baseline_membership, membership)
    candidate_support = changed_support_nodes(baseline_membership, candidate_membership)
    vanilla_support = changed_support_nodes(baseline_membership, vanilla_membership)
    dist_candidate, inter_candidate, union_candidate = support_distance(
        result_support,
        candidate_support,
    )
    dist_vanilla, inter_vanilla, union_vanilla = support_distance(
        result_support,
        vanilla_support,
    )
    return {
        f"{prefix}_quality": float(quality),
        f"{prefix}_delta_q_vs_start": float(quality - start_quality),
        f"{prefix}_delta_q_vs_candidate": float(quality - candidate_quality),
        f"{prefix}_delta_q_vs_vanilla": float(quality - vanilla_quality),
        f"{prefix}_q_debt_vs_start": max(0.0, float(start_quality - quality)),
        f"{prefix}_support_size": int(result_support.size),
        f"{prefix}_support_distance_to_candidate": float(dist_candidate),
        f"{prefix}_support_intersection_with_candidate": int(inter_candidate),
        f"{prefix}_support_union_with_candidate": int(union_candidate),
        f"{prefix}_support_distance_to_vanilla": float(dist_vanilla),
        f"{prefix}_support_intersection_with_vanilla": int(inter_vanilla),
        f"{prefix}_support_union_with_vanilla": int(union_vanilla),
        f"{prefix}_endpoint_distance_to_candidate": endpoint_distance(
            membership,
            candidate_membership,
            sketch_nodes,
        ),
        f"{prefix}_endpoint_distance_to_vanilla": endpoint_distance(
            membership,
            vanilla_membership,
            sketch_nodes,
        ),
    }


def support_progress_from_vanilla(
    *,
    support_distance_to_candidate: float,
    vanilla_support_distance_to_candidate: float,
) -> float:
    return float(vanilla_support_distance_to_candidate) - float(
        support_distance_to_candidate
    )


def classify_polish_recovery(
    *,
    raw_delta_q_vs_start: float,
    polish_delta_q_vs_start: float,
    raw_progress_from_vanilla: float,
    polish_progress_from_vanilla: float,
    polish_support_distance_to_vanilla: float,
    min_retained_progress_fraction: float = 0.5,
    min_support_shift_from_vanilla: float = 0.05,
) -> str:
    if polish_delta_q_vs_start < raw_delta_q_vs_start:
        return POLISH_RESULT_QUALITY_LOSS
    if raw_progress_from_vanilla <= 0.0:
        return POLISH_RESULT_RAW_ONLY
    retained = float(polish_progress_from_vanilla) / max(
        float(raw_progress_from_vanilla),
        1e-12,
    )
    if (
        retained >= float(min_retained_progress_fraction)
        and float(polish_support_distance_to_vanilla) >= float(min_support_shift_from_vanilla)
    ):
        return POLISH_RESULT_RECOVERED_SHIFT
    return POLISH_RESULT_RECOVERED_VANILLA_NEAR


def v_only_support_nodes(
    baseline_membership: np.ndarray,
    candidate_membership: np.ndarray,
    vanilla_membership: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return candidate support, vanilla support, and E = S_V - S_C."""
    candidate_support = changed_support_nodes(baseline_membership, candidate_membership)
    vanilla_support = changed_support_nodes(baseline_membership, vanilla_membership)
    extra = np.setdiff1d(vanilla_support, candidate_support, assume_unique=False)
    return candidate_support, vanilla_support, extra.astype(np.uint32, copy=False)


def edge_summary_for_nodes(
    *,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    nodes: np.ndarray,
    node_count: int,
) -> dict[str, float]:
    nodes = np.asarray(nodes, dtype=np.int64)
    if nodes.size == 0:
        return {
            "incident_edge_count": 0,
            "internal_edge_count": 0,
            "boundary_edge_count": 0,
            "incident_edge_weight": 0.0,
            "internal_edge_weight": 0.0,
            "boundary_edge_weight": 0.0,
        }
    mask = np.zeros(int(node_count), dtype=np.bool_)
    mask[nodes] = True
    src_hit = mask[np.asarray(src, dtype=np.int64)]
    dst_hit = mask[np.asarray(dst, dtype=np.int64)]
    incident = src_hit | dst_hit
    internal = src_hit & dst_hit
    boundary = src_hit ^ dst_hit
    weights = np.asarray(weight, dtype=np.float64)
    return {
        "incident_edge_count": int(np.count_nonzero(incident)),
        "internal_edge_count": int(np.count_nonzero(internal)),
        "boundary_edge_count": int(np.count_nonzero(boundary)),
        "incident_edge_weight": float(weights[incident].sum()),
        "internal_edge_weight": float(weights[internal].sum()),
        "boundary_edge_weight": float(weights[boundary].sum()),
    }


def build_label_intersection_units(
    *,
    baseline_membership: np.ndarray,
    candidate_membership: np.ndarray,
    vanilla_membership: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    node_weights: np.ndarray,
    context: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build V->C units over E = S_V - S_C grouped by (candidate, vanilla)."""
    baseline = np.asarray(baseline_membership, dtype=np.uint64)
    candidate = np.asarray(candidate_membership, dtype=np.uint64)
    vanilla = np.asarray(vanilla_membership, dtype=np.uint64)
    candidate_support, vanilla_support, extra = v_only_support_nodes(
        baseline,
        candidate,
        vanilla,
    )
    rows: list[dict[str, Any]] = []
    if extra.size:
        unit_frame = pd.DataFrame(
            {
                "node": extra.astype(np.uint32),
                "candidate_label": candidate[extra.astype(np.int64)].astype(np.uint64),
                "vanilla_label": vanilla[extra.astype(np.int64)].astype(np.uint64),
                "baseline_label": baseline[extra.astype(np.int64)].astype(np.uint64),
            }
        )
        grouped = unit_frame.groupby(
            ["candidate_label", "vanilla_label"],
            sort=True,
            dropna=False,
        )
        candidate_label_counts = pd.Series(candidate).value_counts().to_dict()
        vanilla_label_counts = pd.Series(vanilla).value_counts().to_dict()
        for unit_index, ((candidate_label, vanilla_label), group) in enumerate(grouped):
            nodes = np.asarray(sorted(group["node"].astype(int)), dtype=np.uint32)
            candidate_closure = int(candidate_label_counts.get(int(candidate_label), 0))
            vanilla_closure = int(vanilla_label_counts.get(int(vanilla_label), 0))
            edge_summary = edge_summary_for_nodes(
                src=src,
                dst=dst,
                weight=weight,
                nodes=nodes,
                node_count=int(baseline.size),
            )
            rows.append(
                {
                    **context,
                    "direction": "vanilla_to_candidate_support",
                    "unit_type": UNIT_TYPE_LABEL_INTERSECTION,
                    "unit_id": f"lib_{unit_index:05d}",
                    "candidate_label": int(candidate_label),
                    "vanilla_label": int(vanilla_label),
                    "baseline_label_count": int(group["baseline_label"].nunique()),
                    "unit_node_count": int(nodes.size),
                    "unit_node_weight": float(
                        np.asarray(node_weights, dtype=np.float64)[
                            nodes.astype(np.int64)
                        ].sum()
                    ),
                    "candidate_label_closure_node_count": candidate_closure,
                    "candidate_label_closure_extra_count": int(
                        max(0, candidate_closure - int(nodes.size))
                    ),
                    "vanilla_label_closure_node_count": vanilla_closure,
                    "vanilla_label_closure_extra_count": int(
                        max(0, vanilla_closure - int(nodes.size))
                    ),
                    "unit_progress_fraction": (
                        float(nodes.size) / float(extra.size) if extra.size else math.nan
                    ),
                    "node_ids": _node_csv(nodes),
                    **edge_summary,
                }
            )
    units = pd.DataFrame(rows)
    summary = {
        **context,
        "direction": "vanilla_to_candidate_support",
        "unit_type": UNIT_TYPE_LABEL_INTERSECTION,
        "candidate_support_size": int(candidate_support.size),
        "vanilla_support_size": int(vanilla_support.size),
        "v_only_support_size": int(extra.size),
        "unit_count": int(len(units)),
    }
    return units, summary


def score_membership(graph: Any, membership: np.ndarray, *, resolution: float) -> float:
    return float(graph.cpm_quality(np.asarray(membership, dtype=np.uint64), resolution=float(resolution)))


def apply_fresh_forced_unit(
    *,
    membership: np.ndarray,
    donor_membership: np.ndarray,
    unit: pd.Series,
    label_map: dict[int, int] | None = None,
    next_label: int | None = None,
) -> tuple[np.ndarray, dict[int, int], int]:
    nodes = parse_node_ids(unit["node_ids"])
    return fresh_group_transplant(
        membership,
        donor_membership,
        nodes=nodes,
        label_map=label_map,
        next_label=next_label,
    )


def _balanced_score(delta_q: float, progress: float) -> float:
    if delta_q >= 0.0:
        return float(progress) + min(float(delta_q), 1.0) * 1e-6
    debt = max(abs(float(delta_q)), 1e-12)
    return float(progress) / debt


def policy_score(*, policy: str, delta_q: float, progress: float) -> float:
    if policy == "q_first":
        return float(delta_q)
    if policy == "progress_first":
        return float(progress)
    if policy == "balanced":
        return _balanced_score(delta_q, progress)
    raise ValueError(f"Unsupported scoring policy: {policy}")


def _as_numeric(series: pd.Series, *, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(float(default))


def _prefix_unit_ids(parent_ids: Any, unit_id: Any) -> str:
    parent = str(parent_ids or "")
    unit = str(unit_id)
    if not parent or parent.lower() == "nan":
        return unit
    return f"{parent},{unit}"


def _barrier_failure_label(
    *,
    q_rank: int,
    progress_rank: int,
    raw_barrier: float,
    support_progress: float,
    closure_extra_ratio: float,
    q_rank_cutoff: int,
    progress_rank_cutoff: int,
    closure_ratio_threshold: float,
) -> str:
    labels: list[str] = []
    if q_rank > int(q_rank_cutoff) and support_progress > 0.0:
        labels.append(BARRIER_Q_GREEDY_MISS)
    if progress_rank > int(progress_rank_cutoff) and support_progress > 0.0:
        labels.append(BARRIER_PROGRESS_GREEDY_MISS)
    if closure_extra_ratio >= float(closure_ratio_threshold):
        labels.append(BARRIER_CLOSURE_COMPOUND_MISS)
    if raw_barrier > 0.0 and support_progress > 0.0:
        labels.append(BARRIER_POLISH_RECOVERY_MISS)
    return ";".join(labels) if labels else BARRIER_GREEDY_VISIBLE


def annotate_barrier_aware_prefixes(
    *,
    frontier_rows: pd.DataFrame,
    beam_rows: pd.DataFrame,
    v_only_support_size: int,
    barrier_floor: float = 1.0,
    closure_penalty_weight: float = 1e-3,
    compactness_penalty_weight: float = 1e-3,
    q_rank_cutoff: int = 1,
    progress_rank_cutoff: int = 1,
    closure_ratio_threshold: float = 4.0,
) -> pd.DataFrame:
    """Annotate existing raw frontier rows with barrier-aware prefix metrics.

    This does not rerun Leiden and does not claim an accepted operator. Each row
    is one proposed next prefix from an already explored beam state.
    """
    if frontier_rows.empty:
        return frontier_rows.copy()
    rows = frontier_rows.copy()
    parent_cols = ["state_id", "selected_unit_ids", "selected_unit_count", "flipped_node_count"]
    if beam_rows.empty:
        parent = pd.DataFrame(columns=parent_cols)
    else:
        parent = beam_rows[[c for c in parent_cols if c in beam_rows.columns]].copy()
    parent = parent.rename(
        columns={
            "state_id": "parent_state_id",
            "selected_unit_ids": "parent_selected_unit_ids",
            "selected_unit_count": "parent_selected_unit_count",
            "flipped_node_count": "parent_flipped_node_count",
        }
    )
    rows = rows.merge(parent, on="parent_state_id", how="left")
    rows["parent_selected_unit_ids"] = rows["parent_selected_unit_ids"].fillna("")
    rows["parent_selected_unit_count"] = _as_numeric(
        rows.get("parent_selected_unit_count", pd.Series(0, index=rows.index)),
    ).astype(int)
    support_size = max(int(v_only_support_size), 0)
    rows["prefix_unit_ids"] = [
        _prefix_unit_ids(parent_ids, unit_id)
        for parent_ids, unit_id in zip(
            rows["parent_selected_unit_ids"],
            rows["unit_id"],
            strict=False,
        )
    ]
    rows["prefix_unit_count"] = rows["parent_selected_unit_count"] + 1
    rows["support_progress_fraction"] = _as_numeric(
        rows["candidate_progress_fraction"],
    )
    rows["prefix_flipped_node_count_estimate"] = np.rint(
        rows["support_progress_fraction"] * float(support_size)
    ).astype(int)
    rows["peak_raw_barrier"] = _as_numeric(rows["raw_barrier_if_chosen"])
    rows["raw_barrier_is_zero"] = rows["peak_raw_barrier"].le(0.0)
    rows["support_progress_per_raw_barrier"] = np.where(
        rows["peak_raw_barrier"].gt(0.0),
        rows["support_progress_fraction"] / rows["peak_raw_barrier"],
        np.inf,
    )
    denominator = np.maximum(rows["peak_raw_barrier"], float(barrier_floor))
    rows["support_progress_per_barrier_floor"] = (
        rows["support_progress_fraction"] / denominator
    )
    immediate_debt = np.maximum(-_as_numeric(rows["delta_q_immediate"]), 0.0)
    rows["incremental_progress_per_immediate_debt"] = np.where(
        immediate_debt > 0.0,
        _as_numeric(rows["incremental_progress_fraction"]) / immediate_debt,
        np.inf,
    )
    rows["closure_extra_ratio"] = _as_numeric(
        rows["candidate_label_closure_extra_count"],
    ) / np.maximum(_as_numeric(rows["unit_node_count"]), 1.0)
    rows["prefix_flipped_fraction"] = (
        rows["prefix_flipped_node_count_estimate"] / float(support_size)
        if support_size
        else math.nan
    )
    rows["barrier_aware_score"] = (
        rows["support_progress_per_barrier_floor"]
        - float(closure_penalty_weight) * rows["closure_extra_ratio"]
        - float(compactness_penalty_weight) * rows["prefix_flipped_fraction"]
    )
    group_cols = ["parent_state_id"]
    rows["q_rank_within_parent"] = (
        rows.groupby(group_cols)["q_first_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    rows["progress_rank_within_parent"] = (
        rows.groupby(group_cols)["progress_first_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    rows["balanced_rank_within_parent"] = (
        rows.groupby(group_cols)["balanced_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    rows["greedy_failure_labels"] = [
        _barrier_failure_label(
            q_rank=int(q_rank),
            progress_rank=int(progress_rank),
            raw_barrier=float(raw_barrier),
            support_progress=float(progress),
            closure_extra_ratio=float(closure_ratio),
            q_rank_cutoff=q_rank_cutoff,
            progress_rank_cutoff=progress_rank_cutoff,
            closure_ratio_threshold=closure_ratio_threshold,
        )
        for q_rank, progress_rank, raw_barrier, progress, closure_ratio in zip(
            rows["q_rank_within_parent"],
            rows["progress_rank_within_parent"],
            rows["peak_raw_barrier"],
            rows["support_progress_fraction"],
            rows["closure_extra_ratio"],
            strict=False,
        )
    ]
    return rows


def select_barrier_progress_frontier(
    prefix_rows: pd.DataFrame,
    *,
    max_rows: int = 50,
    min_support_progress: float = 0.0,
) -> pd.DataFrame:
    """Return a barrier-progress Pareto-style prefix subset.

    Rows are sorted by lower peak barrier first. A row is kept when it improves
    support progress over all lower-barrier rows seen so far. The final output
    is ordered by the diagnostic barrier-aware score.
    """
    if prefix_rows.empty:
        return prefix_rows.copy()
    rows = prefix_rows[
        prefix_rows["support_progress_fraction"].ge(float(min_support_progress))
    ].copy()
    if rows.empty:
        return rows
    rows = rows.sort_values(
        [
            "peak_raw_barrier",
            "prefix_flipped_node_count_estimate",
            "support_progress_fraction",
            "barrier_aware_score",
        ],
        ascending=[True, True, False, False],
    )
    kept: list[int] = []
    best_progress = -math.inf
    for index, row in rows.iterrows():
        progress = float(row["support_progress_fraction"])
        if progress > best_progress + 1e-12:
            kept.append(index)
            best_progress = progress
    frontier = rows.loc[kept].copy()
    return frontier.sort_values(
        [
            "barrier_aware_score",
            "support_progress_fraction",
            "peak_raw_barrier",
            "prefix_flipped_node_count_estimate",
        ],
        ascending=[False, False, True, True],
    ).head(int(max_rows))


def expand_frontier(
    *,
    graph: Any,
    state: BeamState,
    units: pd.DataFrame,
    donor_membership: np.ndarray,
    start_quality: float,
    v_only_support_size: int,
    resolution: float,
) -> list[dict[str, Any]]:
    selected = set(state.selected_unit_ids)
    rows: list[dict[str, Any]] = []
    for _, unit in units.iterrows():
        unit_id = str(unit["unit_id"])
        if unit_id in selected:
            continue
        nodes = parse_node_ids(unit["node_ids"])
        proposed, label_map, next_label = apply_fresh_forced_unit(
            membership=state.membership,
            donor_membership=donor_membership,
            unit=unit,
            label_map=state.label_map,
            next_label=state.next_label,
        )
        quality = score_membership(graph, proposed, resolution=resolution)
        delta_q = float(quality - state.quality)
        delta_vs_start = float(quality - start_quality)
        flipped_nodes = state.flipped_nodes | frozenset(int(node) for node in nodes)
        progress = (
            float(len(flipped_nodes)) / float(v_only_support_size)
            if v_only_support_size
            else math.nan
        )
        incremental_progress = (
            float(nodes.size) / float(v_only_support_size)
            if v_only_support_size
            else math.nan
        )
        raw_debt = max(0.0, float(start_quality - quality))
        rows.append(
            {
                "parent_state_id": state.state_id,
                "scoring_policy": state.scoring_policy,
                "step_index": int(state.step_index + 1),
                "unit_id": unit_id,
                "unit_type": unit["unit_type"],
                "candidate_label": int(unit["candidate_label"]),
                "vanilla_label": int(unit["vanilla_label"]),
                "unit_node_count": int(unit["unit_node_count"]),
                "unit_node_weight": float(unit["unit_node_weight"]),
                "candidate_label_closure_node_count": int(
                    unit["candidate_label_closure_node_count"]
                ),
                "candidate_label_closure_extra_count": int(
                    unit["candidate_label_closure_extra_count"]
                ),
                "boundary_edge_weight": float(unit["boundary_edge_weight"]),
                "incident_edge_weight": float(unit["incident_edge_weight"]),
                "quality_before": float(state.quality),
                "quality_after": float(quality),
                "delta_q_immediate": delta_q,
                "delta_q_vs_start": delta_vs_start,
                "raw_q_debt_vs_start": raw_debt,
                "raw_barrier_if_chosen": float(
                    max(state.raw_barrier_so_far, raw_debt)
                ),
                "candidate_progress_fraction": progress,
                "incremental_progress_fraction": incremental_progress,
                "q_first_score": policy_score(
                    policy="q_first",
                    delta_q=delta_q,
                    progress=incremental_progress,
                ),
                "progress_first_score": policy_score(
                    policy="progress_first",
                    delta_q=delta_q,
                    progress=incremental_progress,
                ),
                "balanced_score": policy_score(
                    policy="balanced",
                    delta_q=delta_q,
                    progress=incremental_progress,
                ),
                "proposed_membership": proposed,
                "proposed_label_map": label_map,
                "proposed_next_label": int(next_label),
                "proposed_flipped_nodes": flipped_nodes,
            }
        )
    return rows


def _frontier_sort_columns(policy: str) -> tuple[list[str], list[bool]]:
    score_column = f"{policy}_score"
    if policy == "q_first":
        return (
            [score_column, "candidate_progress_fraction", "raw_barrier_if_chosen", "unit_node_count"],
            [False, False, True, False],
        )
    if policy == "progress_first":
        return (
            [score_column, "delta_q_immediate", "raw_barrier_if_chosen", "unit_node_count"],
            [False, False, True, False],
        )
    if policy == "balanced":
        return (
            [score_column, "candidate_progress_fraction", "delta_q_immediate", "raw_barrier_if_chosen"],
            [False, False, False, True],
        )
    raise ValueError(f"Unsupported scoring policy: {policy}")


def _state_row(
    *,
    state: BeamState,
    baseline_membership: np.ndarray,
    candidate_membership: np.ndarray,
    vanilla_membership: np.ndarray,
    candidate_support: np.ndarray,
    vanilla_support: np.ndarray,
    sketch_nodes: np.ndarray,
    start_quality: float,
    candidate_quality: float,
    vanilla_quality: float,
    context: dict[str, Any],
) -> dict[str, Any]:
    result_support = changed_support_nodes(baseline_membership, state.membership)
    dist_candidate, inter_candidate, union_candidate = support_distance(
        result_support,
        candidate_support,
    )
    dist_vanilla, inter_vanilla, union_vanilla = support_distance(
        result_support,
        vanilla_support,
    )
    return {
        **context,
        "state_id": state.state_id,
        "scoring_policy": state.scoring_policy,
        "step_index": int(state.step_index),
        "selected_unit_count": int(len(state.selected_unit_ids)),
        "selected_unit_ids": ",".join(state.selected_unit_ids),
        "flipped_node_count": int(len(state.flipped_nodes)),
        "quality": float(state.quality),
        "delta_q_vs_start": float(state.quality - start_quality),
        "delta_q_vs_candidate": float(state.quality - candidate_quality),
        "delta_q_vs_vanilla": float(state.quality - vanilla_quality),
        "raw_q_debt_vs_start": max(0.0, float(start_quality - state.quality)),
        "raw_barrier_so_far": float(state.raw_barrier_so_far),
        "result_support_size": int(result_support.size),
        "result_support_distance_to_candidate": float(dist_candidate),
        "result_support_intersection_with_candidate": int(inter_candidate),
        "result_support_union_with_candidate": int(union_candidate),
        "result_support_distance_to_vanilla": float(dist_vanilla),
        "result_support_intersection_with_vanilla": int(inter_vanilla),
        "result_support_union_with_vanilla": int(union_vanilla),
        "endpoint_distance_to_candidate": endpoint_distance(
            state.membership,
            candidate_membership,
            sketch_nodes,
        ),
        "endpoint_distance_to_vanilla": endpoint_distance(
            state.membership,
            vanilla_membership,
            sketch_nodes,
        ),
    }


def run_ordered_flip_beam(
    *,
    graph: Any,
    units: pd.DataFrame,
    baseline_membership: np.ndarray,
    candidate_membership: np.ndarray,
    vanilla_membership: np.ndarray,
    start_quality: float,
    candidate_quality: float,
    vanilla_quality: float,
    sketch_nodes: np.ndarray,
    resolution: float,
    beam_width: int,
    max_steps: int,
    scoring_policies: tuple[str, ...],
    context: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run raw ordered flip beams over precomputed units."""
    if units.empty:
        return pd.DataFrame(), pd.DataFrame()
    candidate_support, vanilla_support, extra = v_only_support_nodes(
        baseline_membership,
        candidate_membership,
        vanilla_membership,
    )
    all_frontier_rows: list[dict[str, Any]] = []
    all_beam_rows: list[dict[str, Any]] = []
    for policy in scoring_policies:
        states = [
            BeamState(
                state_id=f"{policy}:0:root",
                scoring_policy=policy,
                step_index=0,
                membership=np.asarray(vanilla_membership, dtype=np.uint64).copy(),
                quality=float(start_quality),
                selected_unit_ids=(),
                flipped_nodes=frozenset(),
                raw_barrier_so_far=0.0,
                label_map={},
                next_label=int(np.max(vanilla_membership, initial=0)) + 1,
            )
        ]
        for step_index in range(1, int(max_steps) + 1):
            expanded: list[dict[str, Any]] = []
            for state in states:
                expanded.extend(
                    expand_frontier(
                        graph=graph,
                        state=state,
                        units=units,
                        donor_membership=candidate_membership,
                        start_quality=start_quality,
                        v_only_support_size=int(extra.size),
                        resolution=resolution,
                    )
                )
            if not expanded:
                break
            frontier = pd.DataFrame(expanded)
            public_frontier = frontier.drop(
                columns=[
                    "proposed_membership",
                    "proposed_label_map",
                    "proposed_next_label",
                    "proposed_flipped_nodes",
                ]
            )
            all_frontier_rows.extend(public_frontier.to_dict("records"))
            sort_cols, ascending = _frontier_sort_columns(policy)
            frontier = frontier.sort_values(sort_cols, ascending=ascending)
            next_states: list[BeamState] = []
            seen: set[tuple[str, ...]] = set()
            for rank, (_, row) in enumerate(frontier.iterrows(), start=1):
                parent = next(
                    state
                    for state in states
                    if state.state_id == row["parent_state_id"]
                )
                unit_ids = tuple([*parent.selected_unit_ids, str(row["unit_id"])])
                key = tuple(sorted(unit_ids))
                if key in seen:
                    continue
                seen.add(key)
                state = BeamState(
                    state_id=f"{policy}:{step_index}:{rank}",
                    scoring_policy=policy,
                    step_index=step_index,
                    membership=np.asarray(row["proposed_membership"], dtype=np.uint64),
                    quality=float(row["quality_after"]),
                    selected_unit_ids=unit_ids,
                    flipped_nodes=row["proposed_flipped_nodes"],
                    raw_barrier_so_far=float(row["raw_barrier_if_chosen"]),
                    label_map=dict(row["proposed_label_map"]),
                    next_label=int(row["proposed_next_label"]),
                )
                next_states.append(state)
                all_beam_rows.append(
                    {
                        **_state_row(
                            state=state,
                            baseline_membership=baseline_membership,
                            candidate_membership=candidate_membership,
                            vanilla_membership=vanilla_membership,
                            candidate_support=candidate_support,
                            vanilla_support=vanilla_support,
                            sketch_nodes=sketch_nodes,
                            start_quality=start_quality,
                            candidate_quality=candidate_quality,
                            vanilla_quality=vanilla_quality,
                            context=context,
                        ),
                        "chosen_unit_id": str(row["unit_id"]),
                        "chosen_delta_q_immediate": float(row["delta_q_immediate"]),
                        "chosen_incremental_progress_fraction": float(
                            row["incremental_progress_fraction"]
                        ),
                        "chosen_policy_score": float(row[f"{policy}_score"]),
                    }
                )
                if len(next_states) >= int(beam_width):
                    break
            states = next_states
            if not states:
                break
    return pd.DataFrame(all_frontier_rows), pd.DataFrame(all_beam_rows)
