#!/usr/bin/env python3
"""Audit CPM merge viability for NanoClustering symmetric-object universes.

The symmetric-object/support-neighborhood multistart pilot can fail because the
chosen free universe has no objective-positive way to merge. This script checks
that mechanism directly. It treats basin quality/cost as out of scope and asks
only whether the current CPM objective permits internal free-free merges or
external free-to-fixed attachments from a singleton baseline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from materialize_leiden_basin_nanoclustering_symmetric_object_universe_plan import (
    DEFAULT_LANDSCAPE_DIR,
    DEFAULT_OUTPUT_DIR as DEFAULT_OBJECT_UNIVERSE_DIR,
    DEFAULT_SYMMETRIC_OBJECT_DIR,
    ENDPOINT_REGISTRY_CSV,
    OBJECT_COMPONENTS_CSV,
    SYMMETRIC_ROLE_ROWS_CSV,
    _pure_seed_membership_registry,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    DEFAULT_READINESS_DIR,
    GRAPH_INPUT_ROWS_CSV,
    _compact_membership,
    _json_safe,
    _load_graph,
    _load_label_array,
    _mask_hash,
    _parse_csv_list,
    _read_csv,
    _write_csv,
)
from run_leiden_basin_nanoclustering_symmetric_object_multistart_pilot import (
    CLAIM_BOUNDARY as SYMMETRIC_OBJECT_CLAIM_BOUNDARY,
    SELECTION_POLICIES,
    _component_pattern_membership,
    _load_branch_edge_sidecars,
    _load_object_masks,
    _seed0_object_seeded_initial,
    _select_object_rows,
    _support_neighborhood_mask,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_symmetric_object_merge_viability_20260602"
)

VIABILITY_ROWS_CSV = "nanoclustering_symmetric_object_merge_viability_rows.csv"
QUALITY_ROWS_CSV = "nanoclustering_symmetric_object_merge_viability_quality_rows.csv"
SUMMARY_JSON = "nanoclustering_symmetric_object_merge_viability_summary.json"
CONFIG_JSON = "nanoclustering_symmetric_object_merge_viability_config.json"
REPORT_MD = "nanoclustering_symmetric_object_merge_viability_report.md"

RUN_STATUS = "executed_symmetric_object_merge_viability_audit"
CLAIM_BOUNDARY = (
    "NanoClustering symmetric-object CPM merge-viability audit only; checks "
    "whether selected free universes have objective-positive internal merges "
    "or external fixed-cluster attachments under the current CPM resolution. "
    "It does not promote wall/pathway claims, basin quality/cost claims, "
    "real-data method success, or algorithm novelty."
)


def _stats(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
            "p90": None,
            "p99": None,
        }
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
    }


def _prefix_stats(prefix: str, values: np.ndarray) -> dict[str, float | None]:
    return {f"{prefix}_{key}": value for key, value in _stats(values).items()}


def _singleton_membership(initial_labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    membership = np.asarray(initial_labels, dtype=np.uint64).copy()
    idx = np.flatnonzero(mask)
    if idx.size:
        next_label = int(membership.max()) + 1
        membership[idx] = np.arange(
            next_label,
            next_label + int(idx.size),
            dtype=np.uint64,
        )
    return membership


def _one_block_membership(initial_labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    membership = np.asarray(initial_labels, dtype=np.uint64).copy()
    if int(mask.sum()):
        membership[mask] = np.uint64(int(membership.max()) + 1)
    return membership


def _aggregate_internal_pairs(
    *,
    universe_mask: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    internal = universe_mask[edge_src] & universe_mask[edge_dst]
    if not bool(internal.any()):
        empty_u = np.asarray([], dtype=np.uint64)
        empty_w = np.asarray([], dtype=np.float64)
        return empty_u, empty_u, empty_w, 0, 0.0

    src = np.asarray(edge_src[internal], dtype=np.uint64)
    dst = np.asarray(edge_dst[internal], dtype=np.uint64)
    weights = np.asarray(edge_weight[internal], dtype=np.float64)
    self_loop = src == dst
    self_loop_count = int(self_loop.sum())
    self_loop_weight = float(weights[self_loop].sum()) if self_loop_count else 0.0
    src = src[~self_loop]
    dst = dst[~self_loop]
    weights = weights[~self_loop]
    if src.size == 0:
        empty_u = np.asarray([], dtype=np.uint64)
        empty_w = np.asarray([], dtype=np.float64)
        return empty_u, empty_u, empty_w, self_loop_count, self_loop_weight

    left = np.minimum(src, dst)
    right = np.maximum(src, dst)
    n_nodes = int(universe_mask.size)
    codes = left * np.uint64(n_nodes) + right
    order = np.argsort(codes, kind="mergesort")
    codes = codes[order]
    weights = weights[order]
    unique_codes, starts = np.unique(codes, return_index=True)
    pair_weights = np.add.reduceat(weights, starts)
    pair_left = unique_codes // np.uint64(n_nodes)
    pair_right = unique_codes % np.uint64(n_nodes)
    return pair_left, pair_right, pair_weights, self_loop_count, self_loop_weight


def _relation_summary(
    *,
    prefix: str,
    relation_mask: np.ndarray,
    deltas: np.ndarray,
    ratios: np.ndarray,
    pair_weights: np.ndarray,
) -> dict[str, Any]:
    rel_deltas = np.asarray(deltas[relation_mask], dtype=np.float64)
    rel_ratios = np.asarray(ratios[relation_mask], dtype=np.float64)
    rel_weights = np.asarray(pair_weights[relation_mask], dtype=np.float64)
    positive = rel_deltas > 0.0
    out: dict[str, Any] = {
        f"{prefix}_pair_count": int(rel_deltas.size),
        f"{prefix}_positive_merge_pair_count": int(positive.sum()),
        f"{prefix}_positive_merge_pair_share": (
            float(positive.mean()) if rel_deltas.size else None
        ),
        f"{prefix}_positive_merge_edge_weight_sum": float(rel_weights[positive].sum())
        if rel_deltas.size
        else 0.0,
    }
    out.update(_prefix_stats(f"{prefix}_delta_q", rel_deltas))
    out.update(_prefix_stats(f"{prefix}_edge_to_penalty_ratio", rel_ratios))
    return out


def _internal_merge_viability(
    *,
    object_mask: np.ndarray,
    support_mask: np.ndarray,
    universe_mask: np.ndarray,
    weights: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_weight: np.ndarray,
    gamma: float,
) -> dict[str, Any]:
    left, right, pair_weights, self_loop_count, self_loop_weight = _aggregate_internal_pairs(
        universe_mask=universe_mask,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_weight=edge_weight,
    )
    if pair_weights.size:
        penalties = float(gamma) * weights[left] * weights[right]
        deltas = pair_weights - penalties
        ratios = np.divide(
            pair_weights,
            penalties,
            out=np.zeros_like(pair_weights, dtype=np.float64),
            where=penalties > 0,
        )
        positive = deltas > 0.0
        near_nonnegative = deltas >= -1.0e-9
        best_idx = int(np.argmax(deltas))
        best_left = int(left[best_idx])
        best_right = int(right[best_idx])
        best_delta = float(deltas[best_idx])
        best_edge_weight = float(pair_weights[best_idx])
        best_penalty = float(penalties[best_idx])
        best_ratio = float(ratios[best_idx])
    else:
        penalties = np.asarray([], dtype=np.float64)
        deltas = np.asarray([], dtype=np.float64)
        ratios = np.asarray([], dtype=np.float64)
        positive = np.asarray([], dtype=bool)
        near_nonnegative = np.asarray([], dtype=bool)
        best_left = -1
        best_right = -1
        best_delta = None
        best_edge_weight = None
        best_penalty = None
        best_ratio = None

    left_object = object_mask[left]
    right_object = object_mask[right]
    left_support = support_mask[left]
    right_support = support_mask[right]
    object_object = left_object & right_object
    object_support = (left_object & right_support) | (left_support & right_object)
    support_support = left_support & right_support

    out: dict[str, Any] = {
        "internal_pair_count": int(pair_weights.size),
        "internal_self_loop_count": int(self_loop_count),
        "internal_self_loop_weight_sum": float(self_loop_weight),
        "internal_pair_edge_weight_sum": float(pair_weights.sum()) if pair_weights.size else 0.0,
        "positive_internal_merge_pair_count": int(positive.sum()),
        "positive_internal_merge_pair_share": (
            float(positive.mean()) if pair_weights.size else None
        ),
        "near_nonnegative_internal_merge_pair_count": int(near_nonnegative.sum()),
        "best_internal_merge_left_node_id": best_left,
        "best_internal_merge_right_node_id": best_right,
        "best_internal_merge_delta_q": best_delta,
        "best_internal_merge_edge_weight": best_edge_weight,
        "best_internal_merge_penalty": best_penalty,
        "best_internal_merge_edge_to_penalty_ratio": best_ratio,
        "internal_merge_viability_status": (
            "has_positive_internal_merge"
            if int(positive.sum())
            else "no_positive_internal_merge"
        ),
    }
    out.update(_prefix_stats("internal_pair_edge_weight", pair_weights))
    out.update(_prefix_stats("internal_pair_penalty", penalties))
    out.update(_prefix_stats("internal_pair_delta_q", deltas))
    out.update(_prefix_stats("internal_pair_edge_to_penalty_ratio", ratios))
    out.update(
        _relation_summary(
            prefix="object_object",
            relation_mask=object_object,
            deltas=deltas,
            ratios=ratios,
            pair_weights=pair_weights,
        )
    )
    out.update(
        _relation_summary(
            prefix="object_support",
            relation_mask=object_support,
            deltas=deltas,
            ratios=ratios,
            pair_weights=pair_weights,
        )
    )
    out.update(
        _relation_summary(
            prefix="support_support",
            relation_mask=support_support,
            deltas=deltas,
            ratios=ratios,
            pair_weights=pair_weights,
        )
    )
    return out


def _aggregate_external_pairs(
    *,
    universe_mask: np.ndarray,
    initial_labels: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    crossing = universe_mask[edge_src] ^ universe_mask[edge_dst]
    if not bool(crossing.any()):
        empty_u = np.asarray([], dtype=np.uint64)
        empty_w = np.asarray([], dtype=np.float64)
        return empty_u, empty_u, empty_w

    src = np.asarray(edge_src[crossing], dtype=np.uint64)
    dst = np.asarray(edge_dst[crossing], dtype=np.uint64)
    weights = np.asarray(edge_weight[crossing], dtype=np.float64)
    src_in = universe_mask[src]
    free_node = np.where(src_in, src, dst).astype(np.uint64, copy=False)
    outside_node = np.where(src_in, dst, src).astype(np.uint64, copy=False)
    outside_label = np.asarray(initial_labels[outside_node], dtype=np.uint64)
    label_stride = np.uint64(int(initial_labels.max()) + 1)
    codes = free_node * label_stride + outside_label
    order = np.argsort(codes, kind="mergesort")
    codes = codes[order]
    weights = weights[order]
    unique_codes, starts = np.unique(codes, return_index=True)
    pair_weights = np.add.reduceat(weights, starts)
    pair_free = unique_codes // label_stride
    pair_label = unique_codes % label_stride
    return pair_free, pair_label, pair_weights


def _external_attach_viability(
    *,
    universe_mask: np.ndarray,
    initial_labels: np.ndarray,
    weights: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_weight: np.ndarray,
    gamma: float,
) -> dict[str, Any]:
    free_node, outside_label, pair_weights = _aggregate_external_pairs(
        universe_mask=universe_mask,
        initial_labels=initial_labels,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_weight=edge_weight,
    )
    if pair_weights.size:
        cluster_weights = np.bincount(
            np.asarray(initial_labels, dtype=np.int64),
            weights=np.asarray(weights, dtype=np.float64),
        )
        outside_weights = cluster_weights[outside_label]
        penalties = float(gamma) * weights[free_node] * outside_weights
        deltas = pair_weights - penalties
        ratios = np.divide(
            pair_weights,
            penalties,
            out=np.zeros_like(pair_weights, dtype=np.float64),
            where=penalties > 0,
        )
        positive = deltas > 0.0
        order = np.argsort(free_node, kind="mergesort")
        ordered_free = free_node[order]
        ordered_deltas = deltas[order]
        unique_free, starts = np.unique(ordered_free, return_index=True)
        best_by_free = np.maximum.reduceat(ordered_deltas, starts)
        positive_node_count = int((best_by_free > 0.0).sum())
        best_idx = int(np.argmax(deltas))
        best_free = int(free_node[best_idx])
        best_label = int(outside_label[best_idx])
        best_delta = float(deltas[best_idx])
        best_edge_weight = float(pair_weights[best_idx])
        best_penalty = float(penalties[best_idx])
        best_ratio = float(ratios[best_idx])
    else:
        penalties = np.asarray([], dtype=np.float64)
        deltas = np.asarray([], dtype=np.float64)
        ratios = np.asarray([], dtype=np.float64)
        best_by_free = np.asarray([], dtype=np.float64)
        positive = np.asarray([], dtype=bool)
        positive_node_count = 0
        best_free = -1
        best_label = -1
        best_delta = None
        best_edge_weight = None
        best_penalty = None
        best_ratio = None

    out: dict[str, Any] = {
        "external_attach_candidate_count": int(pair_weights.size),
        "positive_external_attach_candidate_count": int(positive.sum()),
        "positive_external_attach_candidate_share": (
            float(positive.mean()) if pair_weights.size else None
        ),
        "positive_external_attach_node_count": positive_node_count,
        "positive_external_attach_node_share": (
            float(positive_node_count / int(universe_mask.sum()))
            if int(universe_mask.sum())
            else None
        ),
        "best_external_attach_free_node_id": best_free,
        "best_external_attach_cluster_id": best_label,
        "best_external_attach_delta_q": best_delta,
        "best_external_attach_edge_weight": best_edge_weight,
        "best_external_attach_penalty": best_penalty,
        "best_external_attach_edge_to_penalty_ratio": best_ratio,
        "external_attach_viability_status": (
            "has_positive_external_attach"
            if int(positive.sum())
            else "no_positive_external_attach"
        ),
    }
    out.update(_prefix_stats("external_attach_edge_weight", pair_weights))
    out.update(_prefix_stats("external_attach_penalty", penalties))
    out.update(_prefix_stats("external_attach_delta_q", deltas))
    out.update(_prefix_stats("external_attach_edge_to_penalty_ratio", ratios))
    out.update(_prefix_stats("best_external_attach_delta_q_by_node", best_by_free))
    return out


def _quality_variants(
    *,
    graph: Any,
    initial_labels: np.ndarray,
    object_mask: np.ndarray,
    seed0_mask: np.ndarray,
    support_mask: np.ndarray,
    universe_mask: np.ndarray,
    component_initial: np.ndarray,
    gamma: float,
) -> list[dict[str, Any]]:
    variants = {
        "seed0_source_state": np.asarray(initial_labels, dtype=np.uint64).copy(),
        "seed0_object_seeded": _seed0_object_seeded_initial(
            initial_labels=initial_labels,
            seed0_mask=seed0_mask,
        ),
        "object_singleton_support_seed0": _singleton_membership(initial_labels, object_mask),
        "universe_singleton_detached": _singleton_membership(initial_labels, universe_mask),
        "object_one_block_support_seed0": _one_block_membership(initial_labels, object_mask),
        "universe_one_block": _one_block_membership(initial_labels, universe_mask),
        "component_pattern_support_seed0": np.asarray(component_initial, dtype=np.uint64),
    }
    qualities = {
        name: float(graph.cpm_quality(membership, resolution=float(gamma)))
        for name, membership in variants.items()
    }
    seed0_quality = qualities["seed0_source_state"]
    singleton_quality = qualities["universe_singleton_detached"]
    rows: list[dict[str, Any]] = []
    for name, quality in qualities.items():
        membership = variants[name]
        rows.append(
            {
                "quality_variant": name,
                "quality": quality,
                "quality_delta_vs_seed0": float(quality - seed0_quality),
                "quality_delta_vs_universe_singleton": float(quality - singleton_quality),
                "object_cluster_count": int(np.unique(membership[object_mask]).size)
                if int(object_mask.sum())
                else 0,
                "support_cluster_count": int(np.unique(membership[support_mask]).size)
                if int(support_mask.sum())
                else 0,
                "universe_cluster_count": int(np.unique(membership[universe_mask]).size)
                if int(universe_mask.sum())
                else 0,
            }
        )
    return rows


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    rows: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering Symmetric-Object Merge Viability Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- universe_count: {summary['universe_count']}",
        f"- object_role_count: {summary['object_role_count']}",
        f"- support_top_ks: `{summary['support_top_ks']}`",
        f"- positive_internal_merge_universe_count: {summary['positive_internal_merge_universe_count']}",
        f"- positive_external_attach_universe_count: {summary['positive_external_attach_universe_count']}",
        f"- best_internal_merge_delta_q_max: {summary['best_internal_merge_delta_q_max']}",
        f"- best_external_attach_delta_q_max: {summary['best_external_attach_delta_q_max']}",
        f"- universe_singleton_best_quality_variant_count: {summary['universe_singleton_best_quality_variant_count']}",
        f"- elapsed_seconds: {summary['elapsed_seconds']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Universes",
    ]
    if rows.empty:
        lines.append("- no audited universes")
    else:
        order = [
            "has_positive_internal_merge",
            "has_positive_external_attach",
            "universe_node_count",
        ]
        for row in rows.sort_values(order, ascending=[False, False, False]).itertuples(
            index=False
        ):
            data = row._asdict()
            lines.append(
                "- "
                f"{data['object_role_universe_id']} / support_top_k={data['support_top_k']}: "
                f"nodes={data['universe_node_count']}, "
                f"internal_pairs={data['internal_pair_count']}, "
                f"positive_internal={data['positive_internal_merge_pair_count']}, "
                f"best_internal_delta={data['best_internal_merge_delta_q']}, "
                f"positive_external_nodes={data['positive_external_attach_node_count']}, "
                f"best_external_delta={data['best_external_attach_delta_q']}, "
                f"best_quality_variant={data['best_quality_variant']}"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "A universe with no positive internal merge candidates is not a "
                "useful free-free basin distinction target under the current CPM "
                "objective. Positive external attachments are tracked separately "
                "because they can explain singleton-looking free labels without "
                "creating a multi-node free basin."
            ),
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    readiness_dir = Path(args.readiness_dir)
    landscape_dir = Path(args.landscape_dir)
    symmetric_object_dir = Path(args.symmetric_object_dir)
    object_universe_dir = Path(args.object_universe_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_rows = _read_csv(readiness_dir / GRAPH_INPUT_ROWS_CSV)
    endpoint_registry = _read_csv(landscape_dir / ENDPOINT_REGISTRY_CSV)
    components = _read_csv(symmetric_object_dir / OBJECT_COMPONENTS_CSV)
    role_rows = _read_csv(object_universe_dir / SYMMETRIC_ROLE_ROWS_CSV)
    selected = _select_object_rows(
        role_rows,
        case_ranks=_parse_csv_list(args.case_ranks, int),
        role_sides=_parse_csv_list(args.role_sides, str),
        analysis_tiers=_parse_csv_list(args.analysis_tiers, str),
        object_status_prefixes=_parse_csv_list(args.object_status_prefixes, str),
        probe_priority_prefixes=_parse_csv_list(args.probe_priority_prefixes, str),
        strict_core_only=bool(args.strict_core_only),
        selection_policy=str(args.selection_policy),
        max_roles=int(args.max_roles),
        dedupe_symmetric_objects=bool(args.dedupe_symmetric_objects),
    )
    support_top_ks = tuple(_parse_csv_list(args.support_top_ks, int))

    graph_by_branch = {
        str(row["branch"]): row
        for _, row in graph_rows.iterrows()
        if str(row.get("runtime_graph_status", "")).startswith("ready_")
    }
    membership_registry = _pure_seed_membership_registry(endpoint_registry)
    manifest_cache: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    label_cache: dict[tuple[str, str], np.ndarray] = {}
    graph_cache: dict[str, tuple[Any, np.ndarray, float]] = {}
    edge_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    object_mask_cache: dict[tuple[str, str], dict[str, Any]] = {}
    viability_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for row in selected.itertuples(index=False):
        branch = str(row.branch)
        object_id = str(row.symmetric_object_id)
        if branch not in graph_by_branch:
            raise ValueError(f"missing ready graph input for branch: {branch}")
        if branch not in graph_cache:
            graph, weights, load_seconds = _load_graph(
                graph_by_branch[branch],
                manifest_cache,
            )
            graph_cache[branch] = graph, weights, load_seconds
        graph, weights, graph_load_seconds = graph_cache[branch]
        n_nodes = int(graph.n_nodes)

        seed0_key = (branch, 0)
        if seed0_key not in membership_registry:
            raise ValueError(f"missing pure seed0 membership for branch: {branch}")
        seed0_path, label_col = membership_registry[seed0_key]
        initial_labels = _compact_membership(_load_label_array(seed0_path, label_col))

        object_key = (branch, object_id)
        if object_key not in object_mask_cache:
            object_mask_cache[object_key] = _load_object_masks(
                branch=branch,
                symmetric_object_id=object_id,
                components=components,
                membership_registry=membership_registry,
                label_cache=label_cache,
                n_nodes=n_nodes,
                weights=weights,
            )
        masks = object_mask_cache[object_key]
        object_mask = masks["object_mask"]
        seed0_mask = masks["seed0_mask"]
        component_masks = masks["component_masks"]
        component_initial, pattern_count, unassigned_count = _component_pattern_membership(
            initial_labels=initial_labels,
            object_mask=object_mask,
            component_masks=component_masks,
        )

        if branch not in edge_cache:
            edge_cache[branch] = _load_branch_edge_sidecars(graph_by_branch[branch])
        edge_src, edge_dst, edge_weight = edge_cache[branch]

        object_role_id = f"{row.role_id}__{object_id}"
        for support_top_k in support_top_ks:
            support_stats = _support_neighborhood_mask(
                object_mask=object_mask,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_weight=edge_weight,
                top_k=int(support_top_k),
                min_weight=float(args.support_neighborhood_min_weight),
            )
            support_mask = support_stats["support_mask"]
            universe_mask = np.logical_or(object_mask, support_mask)
            internal_stats = _internal_merge_viability(
                object_mask=object_mask,
                support_mask=support_mask,
                universe_mask=universe_mask,
                weights=weights,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_weight=edge_weight,
                gamma=float(args.gamma),
            )
            external_stats = _external_attach_viability(
                universe_mask=universe_mask,
                initial_labels=initial_labels,
                weights=weights,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_weight=edge_weight,
                gamma=float(args.gamma),
            )
            if str(args.quality_mode) == "full":
                q_rows = _quality_variants(
                    graph=graph,
                    initial_labels=initial_labels,
                    object_mask=object_mask,
                    seed0_mask=seed0_mask,
                    support_mask=support_mask,
                    universe_mask=universe_mask,
                    component_initial=component_initial,
                    gamma=float(args.gamma),
                )
                best_quality = max(q_rows, key=lambda item: float(item["quality"]))
                best_quality_variant = str(best_quality["quality_variant"])
                best_quality_value = float(best_quality["quality"])
                best_quality_delta_vs_singleton = float(
                    best_quality["quality_delta_vs_universe_singleton"]
                )
            else:
                q_rows = []
                best_quality_variant = "not_evaluated"
                best_quality_value = None
                best_quality_delta_vs_singleton = None
            base = {
                "object_role_universe_id": object_role_id,
                "panel_case_id": row.panel_case_id,
                "panel_case_rank": int(row.panel_case_rank),
                "analysis_tier": row.analysis_tier,
                "strict_core_v0": bool(row.strict_core_v0),
                "role_id": row.role_id,
                "role_side": row.role_side,
                "primitive_id": row.primitive_id,
                "branch": branch,
                "symmetric_object_id": object_id,
                "probe_priority": row.probe_priority,
                "symmetric_object_route_priority_rank": int(
                    row.symmetric_object_route_priority_rank
                ),
                "object_resolution_status": row.object_resolution_status,
                "n_nodes": n_nodes,
                "n_edges": int(graph.n_edges),
                "graph_load_seconds_cached_branch": float(graph_load_seconds),
                "gamma": float(args.gamma),
                "support_top_k": int(support_top_k),
                "support_neighborhood_status": support_stats[
                    "support_neighborhood_status"
                ],
                "support_neighborhood_min_weight": float(
                    support_stats["support_neighborhood_min_weight"]
                ),
                "support_edge_weight_sum": float(support_stats["support_edge_weight_sum"]),
                "object_mask_hash": _mask_hash(object_mask),
                "support_mask_hash": _mask_hash(support_mask),
                "universe_mask_hash": _mask_hash(universe_mask),
                "object_node_count": int(object_mask.sum()),
                "support_node_count": int(support_mask.sum()),
                "universe_node_count": int(universe_mask.sum()),
                "object_doc_sum": float(weights[object_mask].sum()),
                "support_doc_sum": float(weights[support_mask].sum()),
                "universe_doc_sum": float(weights[universe_mask].sum()),
                "component_count": len(component_masks),
                "component_pattern_block_count": int(pattern_count),
                "component_pattern_unassigned_node_count": int(unassigned_count),
                "component_resolution_status_counts": masks[
                    "component_resolution_status_counts"
                ],
                "quality_mode": str(args.quality_mode),
                "best_quality_variant": best_quality_variant,
                "best_quality": best_quality_value,
                "best_quality_delta_vs_universe_singleton": (
                    best_quality_delta_vs_singleton
                ),
                "has_positive_internal_merge": bool(
                    internal_stats["positive_internal_merge_pair_count"] > 0
                ),
                "has_positive_external_attach": bool(
                    external_stats["positive_external_attach_candidate_count"] > 0
                ),
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "symmetric_object_claim_boundary": SYMMETRIC_OBJECT_CLAIM_BOUNDARY,
            }
            viability_rows.append({**base, **internal_stats, **external_stats})
            for q_row in q_rows:
                quality_rows.append({**base, **q_row})

    rows_df = pd.DataFrame(viability_rows)
    quality_df = pd.DataFrame(quality_rows)
    _write_csv(rows_df, output_dir / VIABILITY_ROWS_CSV)
    _write_csv(quality_df, output_dir / QUALITY_ROWS_CSV)

    if rows_df.empty:
        summary = {
            "schema": "nanoclustering_symmetric_object_merge_viability_summary.v1",
            "status": "no_symmetric_object_universes",
            "output_dir": str(output_dir),
            "support_top_ks": ",".join(str(k) for k in support_top_ks),
            "object_role_count": int(len(selected)),
            "universe_count": 0,
            "elapsed_seconds": float(time.perf_counter() - started),
            "claim_boundary": CLAIM_BOUNDARY,
        }
    else:
        quality_evaluated = rows_df["best_quality_variant"].astype(str).ne(
            "not_evaluated"
        )
        singleton_best = rows_df["best_quality_variant"].astype(str).eq(
            "universe_singleton_detached"
        )
        summary = {
            "schema": "nanoclustering_symmetric_object_merge_viability_summary.v1",
            "status": RUN_STATUS,
            "readiness_dir": str(readiness_dir),
            "object_universe_dir": str(object_universe_dir),
            "output_dir": str(output_dir),
            "support_top_ks": ",".join(str(k) for k in support_top_ks),
            "object_role_count": int(selected["role_id"].nunique())
            if not selected.empty
            else 0,
            "unique_symmetric_object_count": int(
                rows_df["symmetric_object_id"].nunique()
            ),
            "universe_count": int(len(rows_df)),
            "positive_internal_merge_universe_count": int(
                rows_df["has_positive_internal_merge"].sum()
            ),
            "positive_internal_merge_universe_share": float(
                rows_df["has_positive_internal_merge"].mean()
            ),
            "positive_external_attach_universe_count": int(
                rows_df["has_positive_external_attach"].sum()
            ),
            "positive_external_attach_universe_share": float(
                rows_df["has_positive_external_attach"].mean()
            ),
            "best_internal_merge_delta_q_max": (
                float(rows_df["best_internal_merge_delta_q"].max())
                if rows_df["best_internal_merge_delta_q"].notna().any()
                else None
            ),
            "best_external_attach_delta_q_max": (
                float(rows_df["best_external_attach_delta_q"].max())
                if rows_df["best_external_attach_delta_q"].notna().any()
                else None
            ),
            "positive_internal_merge_pair_count_sum": int(
                rows_df["positive_internal_merge_pair_count"].sum()
            ),
            "positive_external_attach_node_count_sum": int(
                rows_df["positive_external_attach_node_count"].sum()
            ),
            "universe_singleton_best_quality_variant_count": int(singleton_best.sum()),
            "universe_singleton_best_quality_variant_share": (
                float(singleton_best[quality_evaluated].mean())
                if bool(quality_evaluated.any())
                else None
            ),
            "quality_evaluated_universe_count": int(quality_evaluated.sum()),
            "universe_node_count_median": float(rows_df["universe_node_count"].median()),
            "universe_node_count_max": int(rows_df["universe_node_count"].max()),
            "object_node_count_median": float(rows_df["object_node_count"].median()),
            "support_node_count_median": float(rows_df["support_node_count"].median()),
            "elapsed_seconds": float(time.perf_counter() - started),
            "claim_boundary": CLAIM_BOUNDARY,
        }

    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_symmetric_object_merge_viability_config.v1",
        "readiness_dir": str(readiness_dir),
        "landscape_dir": str(landscape_dir),
        "symmetric_object_dir": str(symmetric_object_dir),
        "object_universe_dir": str(object_universe_dir),
        "output_dir": str(output_dir),
        "case_ranks": list(_parse_csv_list(args.case_ranks, int)),
        "role_sides": list(_parse_csv_list(args.role_sides, str)),
        "analysis_tiers": list(_parse_csv_list(args.analysis_tiers, str)),
        "object_status_prefixes": list(_parse_csv_list(args.object_status_prefixes, str)),
        "probe_priority_prefixes": list(_parse_csv_list(args.probe_priority_prefixes, str)),
        "strict_core_only": bool(args.strict_core_only),
        "selection_policy": str(args.selection_policy),
        "max_roles": int(args.max_roles),
        "dedupe_symmetric_objects": bool(args.dedupe_symmetric_objects),
        "support_top_ks": list(support_top_ks),
        "support_neighborhood_min_weight": float(args.support_neighborhood_min_weight),
        "quality_mode": str(args.quality_mode),
        "gamma": float(args.gamma),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_dir=output_dir, summary=summary, rows=rows_df)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-dir", type=Path, default=DEFAULT_READINESS_DIR)
    parser.add_argument("--landscape-dir", type=Path, default=DEFAULT_LANDSCAPE_DIR)
    parser.add_argument(
        "--symmetric-object-dir",
        type=Path,
        default=DEFAULT_SYMMETRIC_OBJECT_DIR,
    )
    parser.add_argument(
        "--object-universe-dir",
        type=Path,
        default=DEFAULT_OBJECT_UNIVERSE_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-ranks", default="")
    parser.add_argument("--role-sides", default="")
    parser.add_argument("--analysis-tiers", default="strict_core_v0_primary")
    parser.add_argument("--object-status-prefixes", default="ready_anchor_independent")
    parser.add_argument("--probe-priority-prefixes", default="P1_")
    parser.add_argument(
        "--strict-core-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--selection-policy",
        choices=sorted(SELECTION_POLICIES),
        default="route_priority",
    )
    parser.add_argument("--max-roles", type=int, default=6)
    parser.add_argument(
        "--dedupe-symmetric-objects",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--support-top-ks", default="0,100,1000")
    parser.add_argument("--support-neighborhood-min-weight", type=float, default=0.0)
    parser.add_argument("--quality-mode", choices=("none", "full"), default="none")
    parser.add_argument("--gamma", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
