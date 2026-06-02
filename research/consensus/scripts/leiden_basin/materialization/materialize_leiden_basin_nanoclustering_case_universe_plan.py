#!/usr/bin/env python3
"""Materialize executable NanoClustering case-union basin universes.

The signature-universe multistart gate collapsed on the largest strict-core
candidate signatures. The next broader fixed-outside universe should test a
whole candidate/control case at once rather than a single role-side signature.

This script turns the design-only case rows from the basin-universe redesign
artifact into actual branch-local membership masks by OR-ing the candidate and
control signature universes. It does not run clustering or promote wall,
pathway, quality, cost, real-data method success, or algorithm claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sciscape.clustering.integer_remap import ensure_int_edge_sidecars
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    DEFAULT_READINESS_DIR,
    ENDPOINT_TARGET_ROWS_CSV,
    GRAPH_INPUT_ROWS_CSV,
    _json_safe,
    _load_manifest,
    _mask_for_row,
    _mask_hash,
    _read_csv,
    _write_csv,
)


DEFAULT_UNIVERSE_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_basin_universe_redesign_20260601"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_case_universe_plan_20260601"
)

UNIVERSE_SIGNATURE_ROWS_CSV = "nanoclustering_basin_universe_signature_rows.csv"
CASE_UNIVERSE_ROWS_CSV = "nanoclustering_case_universe_rows.csv"
CASE_SIGNATURE_ROWS_CSV = "nanoclustering_case_universe_signature_role_rows.csv"
CASE_GATE_MATRIX_CSV = "nanoclustering_case_universe_gate_matrix.csv"
CASE_CONFIG_JSON = "nanoclustering_case_universe_config.json"
CASE_SUMMARY_JSON = "nanoclustering_case_universe_summary.json"
CASE_REPORT_MD = "nanoclustering_case_universe_report.md"

CLAIM_BOUNDARY = (
    "NanoClustering case-universe materialization only; turns candidate/control "
    "signature-union designs into executable fixed-outside universe contracts. "
    "It does not run clustering, execute routes/pathways, promote walls, "
    "inspect basin quality/cost, claim real-data method success, or claim "
    "algorithm novelty."
)

START_CONTRACT = (
    "source_state;candidate_target_union_seeded;control_target_union_seeded;"
    "candidate_control_target_two_block;case_universe_singleton;random_case_blocks"
)
SCORE_CONTRACT = (
    "terminal hash over case universe; candidate target/source shares; control "
    "target/source shares; candidate-vs-control target separation; quality "
    "recorded only as diagnostic, not as success"
)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _json_dump(value: Any, path: Path) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _parse_handles(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [part for part in str(value).split(";") if part]


def _doc_sum(mask: np.ndarray, weights: np.ndarray) -> float:
    return float(weights[mask].sum())


def _load_graph_weights(graph_rows: pd.DataFrame) -> dict[str, tuple[int, np.ndarray, Path]]:
    out: dict[str, tuple[int, np.ndarray, Path]] = {}
    for _, row in graph_rows.iterrows():
        if not str(row.get("runtime_graph_status", "")).startswith("ready_"):
            continue
        branch = str(row["branch"])
        manifest_path = Path(str(row["runtime_node_manifest_path"]))
        _, weights = _load_manifest(manifest_path)
        out[branch] = (len(weights), weights, manifest_path)
    return out


def _load_branch_edges(
    graph_rows: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for _, row in graph_rows.iterrows():
        if not str(row.get("runtime_graph_status", "")).startswith("ready_"):
            continue
        branch = str(row["branch"])
        src_path, dst_path, weight_path = ensure_int_edge_sidecars(
            Path(str(row["runtime_int_edges_path"]))
        )
        out[branch] = (
            np.memmap(src_path, dtype=np.uint32, mode="r"),
            np.memmap(dst_path, dtype=np.uint32, mode="r"),
            np.memmap(weight_path, dtype=np.float64, mode="r"),
        )
    return out


def _mask_from_handles(
    *,
    endpoint_targets: pd.DataFrame,
    handles: list[str],
    label_cache: dict[tuple[str, str], np.ndarray],
    n_nodes: int,
) -> tuple[np.ndarray, list[str], list[str]]:
    mask = np.zeros(n_nodes, dtype=np.bool_)
    present: list[str] = []
    missing: list[str] = []
    for handle in handles:
        rows = endpoint_targets[
            endpoint_targets["endpoint_handle_id"].astype(str).eq(str(handle))
            & endpoint_targets["membership_path_exists"].astype(bool)
            & endpoint_targets["cluster_label_present"].astype(bool)
        ]
        if rows.empty:
            missing.append(str(handle))
            continue
        mask |= _mask_for_row(rows.iloc[0], label_cache)
        present.append(str(handle))
    return mask, present, missing


def _signature_mask_row(
    *,
    row: pd.Series,
    endpoint_targets: pd.DataFrame,
    label_cache: dict[tuple[str, str], np.ndarray],
    n_nodes: int,
    weights: np.ndarray,
) -> dict[str, Any]:
    source_mask, present_sources, missing_sources = _mask_from_handles(
        endpoint_targets=endpoint_targets,
        handles=_parse_handles(row.get("source_handles")),
        label_cache=label_cache,
        n_nodes=n_nodes,
    )
    target_mask, present_targets, missing_targets = _mask_from_handles(
        endpoint_targets=endpoint_targets,
        handles=_parse_handles(row.get("target_handles")),
        label_cache=label_cache,
        n_nodes=n_nodes,
    )
    universe_mask = np.logical_or(source_mask, target_mask)
    return {
        "signature_universe_id": row["universe_id"],
        "panel_case_id": row["panel_case_id"],
        "panel_case_rank": int(row["panel_case_rank"]),
        "analysis_tier": row["analysis_tier"],
        "strict_core_v0": bool(row["strict_core_v0"]),
        "role_id": row["role_id"],
        "role_side": row["role_side"],
        "primitive_id": row["primitive_id"],
        "branch": row["branch"],
        "endpoint_signature_id": row["endpoint_signature_id"],
        "source_mask": source_mask,
        "target_mask": target_mask,
        "signature_universe_mask": universe_mask,
        "source_node_count": int(source_mask.sum()),
        "target_node_count": int(target_mask.sum()),
        "signature_universe_node_count_actual": int(universe_mask.sum()),
        "source_doc_sum": _doc_sum(source_mask, weights),
        "target_doc_sum": _doc_sum(target_mask, weights),
        "signature_universe_doc_sum_actual": _doc_sum(universe_mask, weights),
        "source_mask_hash": _mask_hash(source_mask),
        "target_mask_hash": _mask_hash(target_mask),
        "signature_universe_mask_hash_actual": _mask_hash(universe_mask),
        "present_source_handle_count": len(present_sources),
        "missing_source_handle_count": len(missing_sources),
        "missing_source_handles": ";".join(missing_sources),
        "present_target_handle_count": len(present_targets),
        "missing_target_handle_count": len(missing_targets),
        "missing_target_handles": ";".join(missing_targets),
        "source_handles": ";".join(_parse_handles(row.get("source_handles"))),
        "target_handles": ";".join(_parse_handles(row.get("target_handles"))),
    }


def _overlap_count(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.logical_and(left, right).sum())


def _overlap_share(left: np.ndarray, right: np.ndarray) -> float:
    denom = min(int(left.sum()), int(right.sum()))
    if denom <= 0:
        return 0.0
    return float(_overlap_count(left, right) / denom)


def _case_priority(row: dict[str, Any]) -> tuple[int, int, float, int]:
    strict = 1 if bool(row["strict_core_v0"]) else 0
    ready = 1 if row["route_readiness_status"] == "ready_case_universe_route_contract" else 0
    separation = float(row["candidate_control_target_overlap_share_of_smaller"])
    nodes = int(row["case_universe_node_count"])
    return strict, ready, 1.0 - separation, nodes


def _edge_interaction_stats(
    *,
    src: np.ndarray,
    dst: np.ndarray,
    edge_weight: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
) -> dict[str, Any]:
    joint_mask = np.logical_or(left_mask, right_mask)
    internal = joint_mask[src] & joint_mask[dst]
    internal_edge_count = int(internal.sum())
    if internal_edge_count == 0:
        return {
            "internal_edge_count": 0,
            "internal_edge_weight_sum": 0.0,
            "cross_edge_count": 0,
            "cross_edge_weight_sum": 0.0,
            "left_internal_edge_count": 0,
            "left_internal_edge_weight_sum": 0.0,
            "right_internal_edge_count": 0,
            "right_internal_edge_weight_sum": 0.0,
            "cross_edge_weight_share_of_internal": 0.0,
        }

    idx = np.flatnonzero(internal)
    local_src = src[idx]
    local_dst = dst[idx]
    local_weight = edge_weight[idx]
    left_src = left_mask[local_src]
    left_dst = left_mask[local_dst]
    right_src = right_mask[local_src]
    right_dst = right_mask[local_dst]
    cross = (left_src & right_dst) | (right_src & left_dst)
    left_internal = left_src & left_dst
    right_internal = right_src & right_dst
    internal_weight_sum = float(local_weight.sum())
    cross_weight_sum = float(local_weight[cross].sum())
    return {
        "internal_edge_count": internal_edge_count,
        "internal_edge_weight_sum": internal_weight_sum,
        "cross_edge_count": int(cross.sum()),
        "cross_edge_weight_sum": cross_weight_sum,
        "left_internal_edge_count": int(left_internal.sum()),
        "left_internal_edge_weight_sum": float(local_weight[left_internal].sum()),
        "right_internal_edge_count": int(right_internal.sum()),
        "right_internal_edge_weight_sum": float(local_weight[right_internal].sum()),
        "cross_edge_weight_share_of_internal": (
            cross_weight_sum / internal_weight_sum if internal_weight_sum else 0.0
        ),
    }


def _add_edge_interactions(
    *,
    case_df: pd.DataFrame,
    case_masks: dict[str, dict[str, np.ndarray]],
    branch_edges: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    max_cases: int,
) -> pd.DataFrame:
    out = case_df.copy()
    metric_columns = [
        "case_internal_edge_count",
        "case_internal_edge_weight_sum",
        "candidate_control_cross_edge_count",
        "candidate_control_cross_edge_weight_sum",
        "candidate_control_cross_edge_weight_share_of_case_internal",
        "candidate_internal_edge_count",
        "candidate_internal_edge_weight_sum",
        "control_internal_edge_count",
        "control_internal_edge_weight_sum",
        "target_internal_edge_count",
        "target_internal_edge_weight_sum",
        "candidate_control_target_cross_edge_count",
        "candidate_control_target_cross_edge_weight_sum",
        "candidate_control_target_cross_edge_weight_share_of_target_internal",
    ]
    for column in metric_columns:
        out[column] = np.nan
    out["edge_interaction_status"] = "not_computed"
    if max_cases <= 0 or out.empty:
        return out

    selected_indexes = list(out.head(max_cases).index)
    for idx in selected_indexes:
        row = out.loc[idx]
        case_id = str(row["case_universe_id"])
        branch = str(row["branch"])
        if branch not in branch_edges:
            out.loc[idx, "edge_interaction_status"] = "blocked_missing_branch_edges"
            continue
        masks = case_masks[case_id]
        src, dst, edge_weight = branch_edges[branch]
        universe_stats = _edge_interaction_stats(
            src=src,
            dst=dst,
            edge_weight=edge_weight,
            left_mask=masks["candidate_universe"],
            right_mask=masks["control_universe"],
        )
        target_stats = _edge_interaction_stats(
            src=src,
            dst=dst,
            edge_weight=edge_weight,
            left_mask=masks["candidate_target"],
            right_mask=masks["control_target"],
        )
        out.loc[idx, "edge_interaction_status"] = "computed"
        out.loc[idx, "case_internal_edge_count"] = universe_stats["internal_edge_count"]
        out.loc[idx, "case_internal_edge_weight_sum"] = universe_stats[
            "internal_edge_weight_sum"
        ]
        out.loc[idx, "candidate_control_cross_edge_count"] = universe_stats[
            "cross_edge_count"
        ]
        out.loc[idx, "candidate_control_cross_edge_weight_sum"] = universe_stats[
            "cross_edge_weight_sum"
        ]
        out.loc[idx, "candidate_control_cross_edge_weight_share_of_case_internal"] = (
            universe_stats["cross_edge_weight_share_of_internal"]
        )
        out.loc[idx, "candidate_internal_edge_count"] = universe_stats[
            "left_internal_edge_count"
        ]
        out.loc[idx, "candidate_internal_edge_weight_sum"] = universe_stats[
            "left_internal_edge_weight_sum"
        ]
        out.loc[idx, "control_internal_edge_count"] = universe_stats[
            "right_internal_edge_count"
        ]
        out.loc[idx, "control_internal_edge_weight_sum"] = universe_stats[
            "right_internal_edge_weight_sum"
        ]
        out.loc[idx, "target_internal_edge_count"] = target_stats["internal_edge_count"]
        out.loc[idx, "target_internal_edge_weight_sum"] = target_stats[
            "internal_edge_weight_sum"
        ]
        out.loc[idx, "candidate_control_target_cross_edge_count"] = target_stats[
            "cross_edge_count"
        ]
        out.loc[idx, "candidate_control_target_cross_edge_weight_sum"] = target_stats[
            "cross_edge_weight_sum"
        ]
        out.loc[
            idx,
            "candidate_control_target_cross_edge_weight_share_of_target_internal",
        ] = target_stats["cross_edge_weight_share_of_internal"]
    computed = out["edge_interaction_status"].astype(str).eq("computed")
    out["case_interaction_pilot_priority_rank"] = np.nan
    if computed.any():
        ranked = out[computed].copy()
        ranked["_ready"] = ranked["route_readiness_status"].astype(str).eq(
            "ready_case_universe_route_contract"
        )
        ranked = ranked.sort_values(
            [
                "strict_core_v0",
                "_ready",
                "candidate_control_cross_edge_weight_share_of_case_internal",
                "candidate_control_target_cross_edge_weight_share_of_target_internal",
                "case_universe_node_count",
                "panel_case_rank",
            ],
            ascending=[False, False, False, False, False, True],
            kind="mergesort",
        )
        out.loc[ranked.index, "case_interaction_pilot_priority_rank"] = np.arange(
            1,
            len(ranked) + 1,
        )
    return out


def materialize(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    readiness_dir = Path(args.readiness_dir)
    universe_dir = Path(args.universe_dir)
    endpoint_targets = _read_csv(readiness_dir / ENDPOINT_TARGET_ROWS_CSV)
    graph_rows = _read_csv(readiness_dir / GRAPH_INPUT_ROWS_CSV)
    signature_rows = _read_csv(universe_dir / UNIVERSE_SIGNATURE_ROWS_CSV)

    graph_weights = _load_graph_weights(graph_rows)
    label_cache: dict[tuple[str, str], np.ndarray] = {}
    case_rows: list[dict[str, Any]] = []
    signature_out_rows: list[dict[str, Any]] = []
    case_masks: dict[str, dict[str, np.ndarray]] = {}

    for case_id, group in signature_rows.groupby("panel_case_id", sort=True):
        branch = str(group["branch"].iloc[0])
        if branch not in graph_weights:
            continue
        n_nodes, weights, manifest_path = graph_weights[branch]
        role_rows: dict[str, dict[str, Any]] = {}
        for _, sig_row in group.iterrows():
            mask_row = _signature_mask_row(
                row=sig_row,
                endpoint_targets=endpoint_targets,
                label_cache=label_cache,
                n_nodes=n_nodes,
                weights=weights,
            )
            role_rows[str(mask_row["role_side"])] = mask_row
            signature_out_rows.append(
                {
                    key: value
                    for key, value in mask_row.items()
                    if not isinstance(value, np.ndarray)
                }
            )

        candidate = role_rows.get("candidate")
        control = role_rows.get("control")
        if candidate is None or control is None:
            route_readiness = "blocked_missing_candidate_or_control_signature"
            case_universe_mask = np.zeros(n_nodes, dtype=np.bool_)
        else:
            route_readiness = "ready_case_universe_route_contract"
            case_universe_mask = np.logical_or(
                candidate["signature_universe_mask"],
                control["signature_universe_mask"],
            )
            if (
                int(candidate["missing_source_handle_count"])
                or int(candidate["missing_target_handle_count"])
                or int(control["missing_source_handle_count"])
                or int(control["missing_target_handle_count"])
            ):
                route_readiness = "blocked_missing_role_handles"

        candidate_universe = (
            candidate["signature_universe_mask"]
            if candidate is not None
            else np.zeros(n_nodes, dtype=np.bool_)
        )
        control_universe = (
            control["signature_universe_mask"]
            if control is not None
            else np.zeros(n_nodes, dtype=np.bool_)
        )
        candidate_source = (
            candidate["source_mask"] if candidate is not None else np.zeros(n_nodes, dtype=np.bool_)
        )
        candidate_target = (
            candidate["target_mask"] if candidate is not None else np.zeros(n_nodes, dtype=np.bool_)
        )
        control_source = (
            control["source_mask"] if control is not None else np.zeros(n_nodes, dtype=np.bool_)
        )
        control_target = (
            control["target_mask"] if control is not None else np.zeros(n_nodes, dtype=np.bool_)
        )
        node_sum_upper_bound = int(candidate_universe.sum() + control_universe.sum())
        actual_nodes = int(case_universe_mask.sum())
        case_universe_id = f"{case_id}__candidate_control_case_union"
        case_row = {
            "case_universe_id": case_universe_id,
            "universe_scope": "case_candidate_control_signature_union_materialized",
            "panel_case_id": case_id,
            "panel_case_rank": int(group["panel_case_rank"].iloc[0]),
            "analysis_tier": group["analysis_tier"].iloc[0],
            "strict_core_v0": bool(group["strict_core_v0"].iloc[0]),
            "branch": branch,
            "signature_count": int(group["endpoint_signature_id"].nunique()),
            "candidate_signature_universe_id": candidate["signature_universe_id"]
            if candidate is not None
            else "",
            "control_signature_universe_id": control["signature_universe_id"]
            if control is not None
            else "",
            "candidate_signature_universe_node_count": int(candidate_universe.sum()),
            "control_signature_universe_node_count": int(control_universe.sum()),
            "signature_universe_node_count_sum_upper_bound_actual": node_sum_upper_bound,
            "case_universe_node_count": actual_nodes,
            "case_universe_doc_sum": _doc_sum(case_universe_mask, weights),
            "case_universe_node_share": float(actual_nodes / n_nodes) if n_nodes else 0.0,
            "fixed_outside_node_count": int(n_nodes - actual_nodes),
            "case_universe_mask_hash": _mask_hash(case_universe_mask),
            "candidate_control_universe_overlap_node_count": _overlap_count(
                candidate_universe, control_universe
            ),
            "candidate_control_universe_overlap_share_of_smaller": _overlap_share(
                candidate_universe, control_universe
            ),
            "candidate_control_target_overlap_node_count": _overlap_count(
                candidate_target, control_target
            ),
            "candidate_control_target_overlap_share_of_smaller": _overlap_share(
                candidate_target, control_target
            ),
            "candidate_source_control_source_overlap_node_count": _overlap_count(
                candidate_source, control_source
            ),
            "candidate_source_control_source_overlap_share_of_smaller": _overlap_share(
                candidate_source, control_source
            ),
            "candidate_source_node_count": int(candidate_source.sum()),
            "candidate_target_node_count": int(candidate_target.sum()),
            "control_source_node_count": int(control_source.sum()),
            "control_target_node_count": int(control_target.sum()),
            "candidate_target_handle_count": int(candidate["present_target_handle_count"])
            if candidate is not None
            else 0,
            "control_target_handle_count": int(control["present_target_handle_count"])
            if control is not None
            else 0,
            "missing_handle_count": int(
                (candidate["missing_source_handle_count"] if candidate is not None else 0)
                + (candidate["missing_target_handle_count"] if candidate is not None else 0)
                + (control["missing_source_handle_count"] if control is not None else 0)
                + (control["missing_target_handle_count"] if control is not None else 0)
            ),
            "node_count_vs_sum_upper_bound_ratio": float(actual_nodes / node_sum_upper_bound)
            if node_sum_upper_bound
            else 0.0,
            "runtime_node_manifest_path": str(manifest_path),
            "route_readiness_status": route_readiness,
            "recommended_start_contract": START_CONTRACT,
            "recommended_score_contract": SCORE_CONTRACT,
            "route_execution_status": "not_executed_case_universe_materialization_only",
            "wall_promotion_status": "not_promoted_no_route_trace",
            "quality_cost_status": "excluded_case_universe_materialization_only",
            "claim_boundary": CLAIM_BOUNDARY,
        }
        case_rows.append(case_row)
        case_masks[case_universe_id] = {
            "candidate_universe": candidate_universe,
            "control_universe": control_universe,
            "candidate_target": candidate_target,
            "control_target": control_target,
        }

    case_df = pd.DataFrame(case_rows)
    if not case_df.empty:
        priority_values = [_case_priority(row) for row in case_df.to_dict("records")]
        case_df["_priority_strict"] = [value[0] for value in priority_values]
        case_df["_priority_ready"] = [value[1] for value in priority_values]
        case_df["_priority_separation"] = [value[2] for value in priority_values]
        case_df["_priority_nodes"] = [value[3] for value in priority_values]
        case_df = case_df.sort_values(
            [
                "_priority_strict",
                "_priority_ready",
                "_priority_nodes",
                "_priority_separation",
                "panel_case_rank",
            ],
            ascending=[False, False, False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        case_df["case_route_pilot_priority_rank"] = np.arange(1, len(case_df) + 1)
        case_df = case_df.drop(
            columns=[
                "_priority_strict",
                "_priority_ready",
                "_priority_separation",
                "_priority_nodes",
            ]
        )
        edge_interaction_top_k = int(args.edge_interaction_top_k)
        branch_edges = _load_branch_edges(graph_rows) if edge_interaction_top_k > 0 else {}
        case_df = _add_edge_interactions(
            case_df=case_df,
            case_masks=case_masks,
            branch_edges=branch_edges,
            max_cases=edge_interaction_top_k,
        )

    sig_df = pd.DataFrame(signature_out_rows)
    gates = _gate_matrix(case_df)
    summary = _summary(
        case_rows=case_df,
        signature_rows=sig_df,
        readiness_dir=readiness_dir,
        universe_dir=universe_dir,
        output_dir=Path(args.output_dir),
    )
    return case_df, sig_df, gates, summary


def _gate_matrix(case_rows: pd.DataFrame) -> pd.DataFrame:
    ready_count = (
        int(case_rows["route_readiness_status"].eq("ready_case_universe_route_contract").sum())
        if not case_rows.empty
        else 0
    )
    strict_ready_count = (
        int(
            (
                case_rows["route_readiness_status"].eq("ready_case_universe_route_contract")
                & case_rows["strict_core_v0"].astype(bool)
            ).sum()
        )
        if not case_rows.empty
        else 0
    )
    return pd.DataFrame(
        [
            {
                "gate_id": "C1_case_union_masks_materialized",
                "status": "pass" if len(case_rows) else "blocked",
                "evidence": f"case_universe_rows={len(case_rows)}",
                "decision": "case rows now use actual OR-union masks rather than upper-bound sums",
                "next_action": "select bounded case-universe multistart pilot rows",
                "claim_boundary": CLAIM_BOUNDARY,
            },
            {
                "gate_id": "C2_case_route_contract_ready",
                "status": "pass" if ready_count else "blocked",
                "evidence": f"ready_case_universe_rows={ready_count}",
                "decision": "ready rows can be passed to a fixed-outside multistart runner",
                "next_action": "run strict-core top cases before symmetric-object resolver",
                "claim_boundary": CLAIM_BOUNDARY,
            },
            {
                "gate_id": "C3_strict_core_pilot_slice_available",
                "status": "pass" if strict_ready_count else "blocked",
                "evidence": f"strict_ready_case_universe_rows={strict_ready_count}",
                "decision": "strict-core candidate/control cases are available for first pilot",
                "next_action": "use route_pilot_priority_rank with max_cases gate",
                "claim_boundary": CLAIM_BOUNDARY,
            },
        ]
    )


def _summary(
    *,
    case_rows: pd.DataFrame,
    signature_rows: pd.DataFrame,
    readiness_dir: Path,
    universe_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    ready = (
        case_rows["route_readiness_status"].eq("ready_case_universe_route_contract")
        if not case_rows.empty
        else pd.Series(dtype=bool)
    )
    strict = (
        case_rows["strict_core_v0"].astype(bool)
        if not case_rows.empty
        else pd.Series(dtype=bool)
    )
    strict_ready_count = int((ready & strict).sum()) if len(ready) else 0
    computed_edge_rows = (
        case_rows["edge_interaction_status"].astype(str).eq("computed")
        if "edge_interaction_status" in case_rows.columns
        else pd.Series(dtype=bool)
    )
    edge_case_rows = case_rows[computed_edge_rows] if len(computed_edge_rows) else pd.DataFrame()
    max_cross_share = (
        float(
            edge_case_rows[
                "candidate_control_cross_edge_weight_share_of_case_internal"
            ].max()
        )
        if not edge_case_rows.empty
        else None
    )
    return {
        "schema": "nanoclustering_case_universe_materialization_summary.v1",
        "status": "executed_case_universe_materialization",
        "readiness_dir": str(readiness_dir),
        "universe_dir": str(universe_dir),
        "output_dir": str(output_dir),
        "case_universe_count": int(len(case_rows)),
        "signature_role_row_count": int(len(signature_rows)),
        "ready_case_universe_count": int(ready.sum()) if len(ready) else 0,
        "strict_core_case_universe_count": int(strict.sum()) if len(strict) else 0,
        "strict_core_ready_case_universe_count": strict_ready_count,
        "case_universe_node_count_median": float(case_rows["case_universe_node_count"].median())
        if not case_rows.empty
        else None,
        "case_universe_node_count_max": int(case_rows["case_universe_node_count"].max())
        if not case_rows.empty
        else 0,
        "case_universe_node_share_median": float(case_rows["case_universe_node_share"].median())
        if not case_rows.empty
        else None,
        "candidate_control_universe_overlap_share_median": float(
            case_rows["candidate_control_universe_overlap_share_of_smaller"].median()
        )
        if not case_rows.empty
        else None,
        "candidate_control_target_overlap_share_median": float(
            case_rows["candidate_control_target_overlap_share_of_smaller"].median()
        )
        if not case_rows.empty
        else None,
        "node_count_vs_sum_upper_bound_ratio_median": float(
            case_rows["node_count_vs_sum_upper_bound_ratio"].median()
        )
        if not case_rows.empty
        else None,
        "edge_interaction_computed_case_count": int(computed_edge_rows.sum())
        if len(computed_edge_rows)
        else 0,
        "candidate_control_cross_edge_weight_share_median": float(
            edge_case_rows[
                "candidate_control_cross_edge_weight_share_of_case_internal"
            ].median()
        )
        if not edge_case_rows.empty
        else None,
        "candidate_control_cross_edge_weight_share_max": max_cross_share,
        "candidate_control_target_cross_edge_weight_share_median": float(
            edge_case_rows[
                "candidate_control_target_cross_edge_weight_share_of_target_internal"
            ].median()
        )
        if not edge_case_rows.empty
        else None,
        "recommended_first_pilot": (
            "interaction_gated_top6_by_case_interaction_pilot_priority_rank"
            if not edge_case_rows.empty
            else "strict_core_top6_by_route_pilot_priority_rank"
            if strict_ready_count
            else "blocked_no_strict_ready_case_universes"
        ),
        "case_union_interaction_read": (
            "weak_interaction_in_computed_top_slice"
            if max_cross_share is not None and max_cross_share < 0.005
            else "interaction_not_computed"
            if max_cross_share is None
            else "interaction_present_in_computed_top_slice"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(output_dir: Path, summary: dict[str, Any], case_rows: pd.DataFrame) -> None:
    lines = [
        "# NanoClustering Case-Universe Plan",
        "",
        f"- status: `{summary['status']}`",
        f"- case_universe_count: {summary['case_universe_count']}",
        f"- ready_case_universe_count: {summary['ready_case_universe_count']}",
        f"- strict_core_ready_case_universe_count: {summary['strict_core_ready_case_universe_count']}",
        f"- case_universe_node_count_median: {summary['case_universe_node_count_median']}",
        f"- case_universe_node_count_max: {summary['case_universe_node_count_max']}",
        f"- candidate_control_universe_overlap_share_median: {summary['candidate_control_universe_overlap_share_median']}",
        f"- candidate_control_target_overlap_share_median: {summary['candidate_control_target_overlap_share_median']}",
        f"- node_count_vs_sum_upper_bound_ratio_median: {summary['node_count_vs_sum_upper_bound_ratio_median']}",
        f"- edge_interaction_computed_case_count: {summary['edge_interaction_computed_case_count']}",
        f"- candidate_control_cross_edge_weight_share_median: {summary['candidate_control_cross_edge_weight_share_median']}",
        f"- candidate_control_cross_edge_weight_share_max: {summary['candidate_control_cross_edge_weight_share_max']}",
        f"- candidate_control_target_cross_edge_weight_share_median: {summary['candidate_control_target_cross_edge_weight_share_median']}",
        f"- case_union_interaction_read: `{summary['case_union_interaction_read']}`",
        f"- recommended_first_pilot: `{summary['recommended_first_pilot']}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Design Read",
        "",
        "The case-union universe is the next executable gate after the local "
        "pair-mask and signature-universe collapses. It expands the movable set "
        "from one role-side endpoint-family signature to the whole matched "
        "candidate/control case. This tests whether the candidate/control "
        "contrast itself creates a distinct optimizer-dynamics surface.",
        "",
        "A pass in the next runner means terminal multiplicity under the same "
        "case-universe fixed-outside mask. A collapse remains a negative result "
        "for this universe, not a failure of the guiding premise.",
        "",
        "## Pilot Slice",
    ]
    if case_rows.empty:
        lines.append("- no case universes")
    else:
        for row in case_rows.head(10).itertuples(index=False):
            data = row._asdict()
            lines.append(
                "- "
                f"rank {data['case_route_pilot_priority_rank']}: "
                f"{data['case_universe_id']} "
                f"interaction_rank={data.get('case_interaction_pilot_priority_rank', '')} "
                f"strict={data['strict_core_v0']} "
                f"ready={data['route_readiness_status']} "
                f"nodes={data['case_universe_node_count']} "
                f"universe_overlap={data['candidate_control_universe_overlap_share_of_smaller']} "
                f"target_overlap={data['candidate_control_target_overlap_share_of_smaller']} "
                f"edge_status={data.get('edge_interaction_status', 'not_computed')} "
                f"cross_weight_share={data.get('candidate_control_cross_edge_weight_share_of_case_internal', '')}"
            )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- `{_rel(output_dir / CASE_UNIVERSE_ROWS_CSV)}`",
            f"- `{_rel(output_dir / CASE_SIGNATURE_ROWS_CSV)}`",
            f"- `{_rel(output_dir / CASE_GATE_MATRIX_CSV)}`",
            f"- `{_rel(output_dir / CASE_SUMMARY_JSON)}`",
        ]
    )
    (output_dir / CASE_REPORT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-dir", type=Path, default=DEFAULT_READINESS_DIR)
    parser.add_argument("--universe-dir", type=Path, default=DEFAULT_UNIVERSE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--edge-interaction-top-k",
        type=int,
        default=0,
        help=(
            "Compute candidate/control edge-interaction metrics for the top K "
            "case-universe pilot rows after prioritization. Use 0 to skip."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_rows, signature_rows, gates, summary = materialize(args)
    _write_csv(case_rows, output_dir / CASE_UNIVERSE_ROWS_CSV)
    _write_csv(signature_rows, output_dir / CASE_SIGNATURE_ROWS_CSV)
    _write_csv(gates, output_dir / CASE_GATE_MATRIX_CSV)
    _json_dump(
        {
            "schema": "nanoclustering_case_universe_materialization.v1",
            "readiness_dir": str(args.readiness_dir),
            "universe_dir": str(args.universe_dir),
            "output_dir": str(output_dir),
            "edge_interaction_top_k": int(args.edge_interaction_top_k),
            "claim_boundary": CLAIM_BOUNDARY,
        },
        output_dir / CASE_CONFIG_JSON,
    )
    _json_dump(summary, output_dir / CASE_SUMMARY_JSON)
    _write_report(output_dir, summary, case_rows)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
