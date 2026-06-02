#!/usr/bin/env python3
"""Audit objective-positive mechanisms for symmetric-object basin universes.

The merge-viability audit showed that selected NanoClustering free universes
have no positive CPM merge or attach candidates under the current doc-weighted
gamma. This script keeps the same universes and asks which objective mechanism
would be needed before another terminal-multiplicity pilot is meaningful.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_leiden_basin_nanoclustering_symmetric_object_merge_viability import (
    _aggregate_external_pairs,
    _aggregate_internal_pairs,
    _prefix_stats,
)
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
    _select_object_rows,
    _support_neighborhood_mask,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_symmetric_object_objective_mechanisms_20260602"
)

MECHANISM_ROWS_CSV = "nanoclustering_symmetric_object_objective_mechanism_rows.csv"
SUMMARY_JSON = "nanoclustering_symmetric_object_objective_mechanism_summary.json"
CONFIG_JSON = "nanoclustering_symmetric_object_objective_mechanism_config.json"
REPORT_MD = "nanoclustering_symmetric_object_objective_mechanism_report.md"

WEIGHT_MODES = {
    "doc_weight",
    "unit_weight",
    "sqrt_doc_weight",
    "log1p_doc_weight",
    "local_median_normalized_doc_weight",
    "local_p90_normalized_doc_weight",
}

RUN_STATUS = "executed_symmetric_object_objective_mechanism_audit"
CLAIM_BOUNDARY = (
    "NanoClustering symmetric-object objective-mechanism audit only; compares "
    "critical gamma and alternative node-weight transforms needed to create "
    "objective-positive CPM merge/attach candidates. It does not promote "
    "wall/pathway, basin-quality, cost, real-data method-success, or algorithm "
    "claims."
)


def _finite_or_none(value: float) -> float | None:
    if not math.isfinite(float(value)):
        return None
    return float(value)


def _safe_quantile(values: np.ndarray, q: float) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    return float(np.quantile(values, q))


def _weight_mode_values(
    *,
    weights: np.ndarray,
    universe_mask: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, float]:
    base = np.asarray(weights, dtype=np.float64)
    if mode == "doc_weight":
        return base, 1.0
    if mode == "unit_weight":
        return np.ones_like(base, dtype=np.float64), 1.0
    if mode == "sqrt_doc_weight":
        return np.sqrt(np.maximum(base, 0.0)), 1.0
    if mode == "log1p_doc_weight":
        return np.log1p(np.maximum(base, 0.0)), 1.0
    local = base[universe_mask]
    positive_local = local[local > 0.0]
    if positive_local.size == 0:
        scale = 1.0
    elif mode == "local_median_normalized_doc_weight":
        scale = float(np.median(positive_local))
    elif mode == "local_p90_normalized_doc_weight":
        scale = float(np.quantile(positive_local, 0.90))
    else:
        raise ValueError(f"unsupported weight mode: {mode}")
    if scale <= 0.0 or not math.isfinite(scale):
        scale = 1.0
    return base / scale, scale


def _candidate_stats(
    *,
    edge_weight: np.ndarray,
    penalty_factor: np.ndarray,
    gamma: float,
    prefix: str,
) -> dict[str, Any]:
    edge_weight = np.asarray(edge_weight, dtype=np.float64)
    penalty_factor = np.asarray(penalty_factor, dtype=np.float64)
    if edge_weight.size == 0:
        out: dict[str, Any] = {
            f"{prefix}_candidate_count": 0,
            f"{prefix}_positive_count": 0,
            f"{prefix}_positive_share": None,
            f"{prefix}_best_delta_q": None,
            f"{prefix}_best_edge_weight": None,
            f"{prefix}_best_penalty": None,
            f"{prefix}_critical_gamma_max": None,
            f"{prefix}_critical_gamma_p99": None,
            f"{prefix}_gamma_to_max_critical_ratio": None,
            f"{prefix}_edge_to_penalty_ratio_max": None,
        }
        out.update(_prefix_stats(f"{prefix}_delta_q", np.asarray([], dtype=np.float64)))
        out.update(
            _prefix_stats(f"{prefix}_critical_gamma", np.asarray([], dtype=np.float64))
        )
        return out

    penalties = float(gamma) * penalty_factor
    deltas = edge_weight - penalties
    critical_gamma = np.divide(
        edge_weight,
        penalty_factor,
        out=np.full_like(edge_weight, np.inf, dtype=np.float64),
        where=penalty_factor > 0.0,
    )
    edge_to_penalty = np.divide(
        edge_weight,
        penalties,
        out=np.zeros_like(edge_weight, dtype=np.float64),
        where=penalties > 0.0,
    )
    positive = deltas > 0.0
    best_idx = int(np.argmax(deltas))
    finite_critical = critical_gamma[np.isfinite(critical_gamma)]
    max_critical = (
        float(np.max(finite_critical)) if finite_critical.size else float("inf")
    )
    out = {
        f"{prefix}_candidate_count": int(edge_weight.size),
        f"{prefix}_positive_count": int(positive.sum()),
        f"{prefix}_positive_share": float(positive.mean()),
        f"{prefix}_best_delta_q": float(deltas[best_idx]),
        f"{prefix}_best_edge_weight": float(edge_weight[best_idx]),
        f"{prefix}_best_penalty": float(penalties[best_idx]),
        f"{prefix}_critical_gamma_max": _finite_or_none(max_critical),
        f"{prefix}_critical_gamma_p99": _safe_quantile(finite_critical, 0.99),
        f"{prefix}_gamma_to_max_critical_ratio": (
            float(gamma / max_critical)
            if max_critical > 0.0 and math.isfinite(max_critical)
            else None
        ),
        f"{prefix}_edge_to_penalty_ratio_max": float(np.max(edge_to_penalty)),
    }
    out.update(_prefix_stats(f"{prefix}_delta_q", deltas))
    out.update(_prefix_stats(f"{prefix}_critical_gamma", finite_critical))
    return out


def _relation_positive_count(
    *,
    relation_mask: np.ndarray,
    edge_weight: np.ndarray,
    penalty_factor: np.ndarray,
    gamma: float,
) -> int:
    if relation_mask.size == 0 or not bool(relation_mask.any()):
        return 0
    deltas = np.asarray(edge_weight[relation_mask], dtype=np.float64) - (
        float(gamma) * np.asarray(penalty_factor[relation_mask], dtype=np.float64)
    )
    return int((deltas > 0.0).sum())


def _mechanism_row(
    *,
    base: dict[str, Any],
    mode: str,
    mode_scale: float,
    effective_weights: np.ndarray,
    initial_labels: np.ndarray,
    object_mask: np.ndarray,
    support_mask: np.ndarray,
    universe_mask: np.ndarray,
    internal_left: np.ndarray,
    internal_right: np.ndarray,
    internal_weight: np.ndarray,
    external_free: np.ndarray,
    external_label: np.ndarray,
    external_weight: np.ndarray,
    gamma: float,
) -> dict[str, Any]:
    internal_penalty_factor = effective_weights[internal_left] * effective_weights[
        internal_right
    ]
    cluster_weights = np.bincount(
        np.asarray(initial_labels, dtype=np.int64),
        weights=np.asarray(effective_weights, dtype=np.float64),
    )
    external_penalty_factor = (
        effective_weights[external_free] * cluster_weights[external_label]
        if external_weight.size
        else np.asarray([], dtype=np.float64)
    )
    row = {
        **base,
        "weight_mode": mode,
        "weight_mode_scale": float(mode_scale),
        "weight_mode_universe_weight_sum": float(effective_weights[universe_mask].sum()),
        "weight_mode_object_weight_sum": float(effective_weights[object_mask].sum()),
        "weight_mode_support_weight_sum": float(effective_weights[support_mask].sum()),
    }
    row.update(
        _candidate_stats(
            edge_weight=internal_weight,
            penalty_factor=internal_penalty_factor,
            gamma=float(gamma),
            prefix="internal",
        )
    )
    row.update(
        _candidate_stats(
            edge_weight=external_weight,
            penalty_factor=external_penalty_factor,
            gamma=float(gamma),
            prefix="external_attach",
        )
    )
    left_object = object_mask[internal_left]
    right_object = object_mask[internal_right]
    left_support = support_mask[internal_left]
    right_support = support_mask[internal_right]
    object_object = left_object & right_object
    object_support = (left_object & right_support) | (left_support & right_object)
    support_support = left_support & right_support
    row.update(
        {
            "object_object_positive_internal_count": _relation_positive_count(
                relation_mask=object_object,
                edge_weight=internal_weight,
                penalty_factor=internal_penalty_factor,
                gamma=float(gamma),
            ),
            "object_support_positive_internal_count": _relation_positive_count(
                relation_mask=object_support,
                edge_weight=internal_weight,
                penalty_factor=internal_penalty_factor,
                gamma=float(gamma),
            ),
            "support_support_positive_internal_count": _relation_positive_count(
                relation_mask=support_support,
                edge_weight=internal_weight,
                penalty_factor=internal_penalty_factor,
                gamma=float(gamma),
            ),
        }
    )
    row["mechanism_status"] = (
        "objective_positive_at_current_gamma"
        if int(row["internal_positive_count"])
        or int(row["external_attach_positive_count"])
        else "blocked_no_positive_candidate_at_current_gamma"
    )
    row["run_status"] = RUN_STATUS
    row["claim_boundary"] = CLAIM_BOUNDARY
    return row


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    rows: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering Symmetric-Object Objective Mechanism Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- universe_count: {summary['universe_count']}",
        f"- weight_modes: `{summary['weight_modes']}`",
        f"- support_top_ks: `{summary['support_top_ks']}`",
        f"- current_doc_weight_positive_internal_universe_count: {summary['doc_weight_positive_internal_universe_count']}",
        f"- current_doc_weight_positive_external_universe_count: {summary['doc_weight_positive_external_universe_count']}",
        f"- doc_weight_internal_critical_gamma_max: {summary['doc_weight_internal_critical_gamma_max']}",
        f"- doc_weight_gamma_to_max_internal_critical_ratio_min: {summary['doc_weight_gamma_to_max_internal_critical_ratio_min']}",
        f"- objective_positive_weight_mode_count: {summary['objective_positive_weight_mode_count']}",
        f"- elapsed_seconds: {summary['elapsed_seconds']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Weight Mode Summary",
    ]
    if rows.empty:
        lines.append("- no audited rows")
    else:
        grouped = (
            rows.groupby("weight_mode", sort=True)
            .agg(
                universe_count=("object_role_universe_id", "count"),
                positive_internal=("internal_positive_count", lambda s: int((s > 0).sum())),
                positive_external=(
                    "external_attach_positive_count",
                    lambda s: int((s > 0).sum()),
                ),
                max_internal_critical_gamma=("internal_critical_gamma_max", "max"),
                max_external_critical_gamma=("external_attach_critical_gamma_max", "max"),
                best_internal_delta=("internal_best_delta_q", "max"),
                best_external_delta=("external_attach_best_delta_q", "max"),
            )
            .reset_index()
        )
        for row in grouped.itertuples(index=False):
            data = row._asdict()
            lines.append(
                "- "
                f"{data['weight_mode']}: universes={data['universe_count']}, "
                f"positive_internal={data['positive_internal']}, "
                f"positive_external={data['positive_external']}, "
                f"max_internal_critical_gamma={data['max_internal_critical_gamma']}, "
                f"max_external_critical_gamma={data['max_external_critical_gamma']}, "
                f"best_internal_delta={data['best_internal_delta']}, "
                f"best_external_delta={data['best_external_delta']}"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "This audit is a design gate. A weight mode or critical-gamma "
                "threshold opening objective-positive candidates is not a basin, "
                "wall, pathway, or method-success claim. It identifies which "
                "objective mechanism must be made explicit before rerunning "
                "terminal-multiplicity pilots."
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
    weight_modes = tuple(_parse_csv_list(args.weight_modes, str))
    unknown_modes = sorted(set(weight_modes) - WEIGHT_MODES)
    if unknown_modes:
        raise ValueError(
            f"unsupported weight modes: {unknown_modes}; expected {sorted(WEIGHT_MODES)}"
        )

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
    mechanism_rows: list[dict[str, Any]] = []
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
        component_masks = masks["component_masks"]
        _, pattern_count, unassigned_count = _component_pattern_membership(
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
            (
                internal_left,
                internal_right,
                internal_weight,
                internal_self_loop_count,
                internal_self_loop_weight,
            ) = _aggregate_internal_pairs(
                universe_mask=universe_mask,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_weight=edge_weight,
            )
            external_free, external_label, external_weight = _aggregate_external_pairs(
                universe_mask=universe_mask,
                initial_labels=initial_labels,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_weight=edge_weight,
            )
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
                "internal_self_loop_count": int(internal_self_loop_count),
                "internal_self_loop_weight_sum": float(internal_self_loop_weight),
                "symmetric_object_claim_boundary": SYMMETRIC_OBJECT_CLAIM_BOUNDARY,
            }
            for mode in weight_modes:
                effective_weights, mode_scale = _weight_mode_values(
                    weights=weights,
                    universe_mask=universe_mask,
                    mode=mode,
                )
                mechanism_rows.append(
                    _mechanism_row(
                        base=base,
                        mode=mode,
                        mode_scale=mode_scale,
                        effective_weights=effective_weights,
                        initial_labels=initial_labels,
                        object_mask=object_mask,
                        support_mask=support_mask,
                        universe_mask=universe_mask,
                        internal_left=internal_left,
                        internal_right=internal_right,
                        internal_weight=internal_weight,
                        external_free=external_free,
                        external_label=external_label,
                        external_weight=external_weight,
                        gamma=float(args.gamma),
                    )
                )

    rows_df = pd.DataFrame(mechanism_rows)
    _write_csv(rows_df, output_dir / MECHANISM_ROWS_CSV)
    elapsed = time.perf_counter() - started

    if rows_df.empty:
        summary = {
            "schema": "nanoclustering_symmetric_object_objective_mechanism_summary.v1",
            "status": "no_symmetric_object_universes",
            "output_dir": str(output_dir),
            "support_top_ks": ",".join(str(k) for k in support_top_ks),
            "weight_modes": ",".join(weight_modes),
            "universe_count": 0,
            "elapsed_seconds": float(elapsed),
            "claim_boundary": CLAIM_BOUNDARY,
        }
    else:
        doc_rows = rows_df[rows_df["weight_mode"].astype(str).eq("doc_weight")]
        objective_positive_modes = rows_df[
            (rows_df["internal_positive_count"] > 0)
            | (rows_df["external_attach_positive_count"] > 0)
        ]["weight_mode"].nunique()
        summary = {
            "schema": "nanoclustering_symmetric_object_objective_mechanism_summary.v1",
            "status": RUN_STATUS,
            "readiness_dir": str(readiness_dir),
            "object_universe_dir": str(object_universe_dir),
            "output_dir": str(output_dir),
            "support_top_ks": ",".join(str(k) for k in support_top_ks),
            "weight_modes": ",".join(weight_modes),
            "object_role_count": int(selected["role_id"].nunique())
            if not selected.empty
            else 0,
            "unique_symmetric_object_count": int(
                rows_df["symmetric_object_id"].nunique()
            ),
            "universe_count": int(
                rows_df[["object_role_universe_id", "support_top_k"]]
                .drop_duplicates()
                .shape[0]
            ),
            "row_count": int(len(rows_df)),
            "doc_weight_positive_internal_universe_count": int(
                (doc_rows["internal_positive_count"] > 0).sum()
            ),
            "doc_weight_positive_external_universe_count": int(
                (doc_rows["external_attach_positive_count"] > 0).sum()
            ),
            "doc_weight_internal_critical_gamma_max": (
                float(doc_rows["internal_critical_gamma_max"].max())
                if doc_rows["internal_critical_gamma_max"].notna().any()
                else None
            ),
            "doc_weight_external_critical_gamma_max": (
                float(doc_rows["external_attach_critical_gamma_max"].max())
                if doc_rows["external_attach_critical_gamma_max"].notna().any()
                else None
            ),
            "doc_weight_gamma_to_max_internal_critical_ratio_min": (
                float(doc_rows["internal_gamma_to_max_critical_ratio"].min())
                if doc_rows["internal_gamma_to_max_critical_ratio"].notna().any()
                else None
            ),
            "doc_weight_gamma_to_max_external_critical_ratio_min": (
                float(doc_rows["external_attach_gamma_to_max_critical_ratio"].min())
                if doc_rows["external_attach_gamma_to_max_critical_ratio"].notna().any()
                else None
            ),
            "objective_positive_weight_mode_count": int(objective_positive_modes),
            "elapsed_seconds": float(elapsed),
            "claim_boundary": CLAIM_BOUNDARY,
        }

    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_symmetric_object_objective_mechanism_config.v1",
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
        "weight_modes": list(weight_modes),
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
    parser.add_argument(
        "--weight-modes",
        default=(
            "doc_weight,unit_weight,sqrt_doc_weight,log1p_doc_weight,"
            "local_median_normalized_doc_weight,local_p90_normalized_doc_weight"
        ),
    )
    parser.add_argument("--gamma", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
