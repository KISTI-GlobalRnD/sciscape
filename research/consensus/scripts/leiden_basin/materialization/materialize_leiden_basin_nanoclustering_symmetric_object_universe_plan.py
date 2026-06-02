#!/usr/bin/env python3
"""Materialize NanoClustering symmetric-object basin universes.

The local pair-mask, anchor-release, common-mask, signature-universe, and
case-union design gates all indicate that loose source/target or
candidate/control unions are weak wall-facing universes. This resolver turns
anchor-independent symmetric endpoint objects into concrete branch-local masks
so the next optimizer-dynamics probe can operate on an object first, rather
than on a seed0-local handle or a loose case union.

It is a materialization/design artifact only: no clustering is run, no route is
executed, and no wall/pathway, quality/cost, real-data method-success, or
algorithm claim is promoted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    DEFAULT_READINESS_DIR,
    GRAPH_INPUT_ROWS_CSV,
    _json_safe,
    _load_label_array,
    _load_manifest,
    _mask_hash,
    _read_csv,
    _write_csv,
)


DEFAULT_LANDSCAPE_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_external_landscape_20260530"
)
DEFAULT_SYMMETRIC_OBJECT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_symmetric_endpoint_objects_20260531"
)
DEFAULT_SYMMETRIC_DECOMPOSITION_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_symmetric_object_decomposition_v1_20260531"
)
DEFAULT_UNIVERSE_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_basin_universe_redesign_20260601"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_symmetric_object_universe_plan_20260601"
)

ENDPOINT_REGISTRY_CSV = "nanoclustering_external_endpoint_registry.csv"
OBJECT_COMPONENTS_CSV = "nanoclustering_symmetric_endpoint_object_components.csv"
OBJECT_DECOMPOSITION_CSV = "nanoclustering_symmetric_object_decomposition.csv"
SYMMETRIC_RESOLVER_ROWS_CSV = "nanoclustering_basin_universe_symmetric_resolver_rows.csv"

SYMMETRIC_ROLE_ROWS_CSV = "nanoclustering_symmetric_object_universe_role_rows.csv"
SYMMETRIC_CASE_ROWS_CSV = "nanoclustering_symmetric_object_universe_case_rows.csv"
SYMMETRIC_GATE_MATRIX_CSV = "nanoclustering_symmetric_object_universe_gate_matrix.csv"
SYMMETRIC_CONFIG_JSON = "nanoclustering_symmetric_object_universe_config.json"
SYMMETRIC_SUMMARY_JSON = "nanoclustering_symmetric_object_universe_summary.json"
SYMMETRIC_REPORT_MD = "nanoclustering_symmetric_object_universe_report.md"

CLAIM_BOUNDARY = (
    "NanoClustering symmetric-object universe materialization only; resolves "
    "all-seed symmetric endpoint objects into executable fixed-outside object "
    "masks. It does not run clustering, execute routes/pathways, promote walls, "
    "inspect basin quality/cost, claim real-data method success, or claim "
    "algorithm novelty."
)
START_CONTRACT = (
    "seed0_source_state;seed0_object_seeded;object_singleton;"
    "object_seed_component_blocks;random_object_blocks"
)
SCORE_CONTRACT = (
    "terminal hash over symmetric object universe; seed0-object retention; "
    "all-seed object support retention; component-block preservation; quality "
    "recorded only as diagnostic, not as success"
)
GOOD_OBJECT_SEED_COVERAGE_MIN = 5
STRONG_OBJECT_SEED_COVERAGE_MIN = 8


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


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().eq("true")


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


def _pure_seed_membership_registry(
    endpoint_registry: pd.DataFrame,
) -> dict[tuple[str, int], tuple[Path, str]]:
    rows = endpoint_registry[_bool_series(endpoint_registry["pure_seed_ensemble"])].copy()
    rows = rows[rows["branch"].notna() & rows["seed"].notna()]
    out: dict[tuple[str, int], tuple[Path, str]] = {}
    for row in rows.itertuples(index=False):
        label_cols = [part for part in str(row.label_cols).split(";") if part]
        label_col = "candidate_micro_id" if "candidate_micro_id" in label_cols else label_cols[0]
        out[(str(row.branch), int(row.seed))] = (Path(str(row.absolute_path)), label_col)
    return out


def _component_mask(
    *,
    component: pd.Series,
    membership_registry: dict[tuple[str, int], tuple[Path, str]],
    label_cache: dict[tuple[str, str], np.ndarray],
    n_nodes: int,
) -> tuple[np.ndarray, str]:
    key = (str(component["branch"]), int(component["seed"]))
    if key not in membership_registry:
        return np.zeros(n_nodes, dtype=np.bool_), "missing_seed_membership_path"
    path, label_col = membership_registry[key]
    if not path.exists():
        return np.zeros(n_nodes, dtype=np.bool_), "missing_membership_file"
    cache_key = (str(path), label_col)
    if cache_key not in label_cache:
        label_cache[cache_key] = _load_label_array(path, label_col)
    labels = label_cache[cache_key]
    if len(labels) != n_nodes:
        return np.zeros(n_nodes, dtype=np.bool_), "membership_length_mismatch"
    return labels == int(component["cluster_id"]), "resolved"


def _resolve_object(
    *,
    branch: str,
    symmetric_object_id: str,
    components: pd.DataFrame,
    graph_weights: dict[str, tuple[int, np.ndarray, Path]],
    membership_registry: dict[tuple[str, int], tuple[Path, str]],
    label_cache: dict[tuple[str, str], np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if branch not in graph_weights:
        empty = np.zeros(0, dtype=np.bool_)
        return {
            "symmetric_object_id": symmetric_object_id,
            "branch": branch,
            "object_resolution_status": "blocked_missing_branch_graph_weights",
        }, {"object_mask": empty, "seed0_mask": empty}
    n_nodes, weights, manifest_path = graph_weights[branch]
    rows = components[
        components["branch"].astype(str).eq(branch)
        & components["symmetric_object_id"].astype(str).eq(str(symmetric_object_id))
    ].copy()
    if rows.empty:
        empty = np.zeros(n_nodes, dtype=np.bool_)
        return {
            "symmetric_object_id": symmetric_object_id,
            "branch": branch,
            "runtime_node_manifest_path": str(manifest_path),
            "object_resolution_status": "blocked_missing_symmetric_object_components",
        }, {"object_mask": empty, "seed0_mask": empty}

    object_mask = np.zeros(n_nodes, dtype=np.bool_)
    seed0_mask = np.zeros(n_nodes, dtype=np.bool_)
    component_node_sum = 0
    component_doc_sum = 0.0
    resolved_count = 0
    status_counts: dict[str, int] = {}
    seed_node_counts: dict[int, int] = {}
    for _, component in rows.iterrows():
        mask, status = _component_mask(
            component=component,
            membership_registry=membership_registry,
            label_cache=label_cache,
            n_nodes=n_nodes,
        )
        status_counts[status] = status_counts.get(status, 0) + 1
        if status != "resolved":
            continue
        resolved_count += 1
        component_node_sum += int(mask.sum())
        component_doc_sum += _doc_sum(mask, weights)
        object_mask |= mask
        seed = int(component["seed"])
        if seed == 0:
            seed0_mask |= mask
        seed_node_counts[seed] = seed_node_counts.get(seed, 0) + int(mask.sum())

    object_nodes = int(object_mask.sum())
    seed0_nodes = int(seed0_mask.sum())
    missing_count = int(len(rows) - resolved_count)
    if missing_count:
        resolution_status = "blocked_unresolved_object_components"
    elif object_nodes <= 0:
        resolution_status = "blocked_empty_symmetric_object_mask"
    elif int(rows["seed"].nunique()) >= STRONG_OBJECT_SEED_COVERAGE_MIN:
        resolution_status = "ready_anchor_independent_object_universe_contract"
    elif int(rows["seed"].nunique()) >= GOOD_OBJECT_SEED_COVERAGE_MIN:
        resolution_status = "ready_good_coverage_object_universe_contract"
    else:
        resolution_status = "ready_partial_or_anchor_local_object_contract"

    stats = {
        "symmetric_object_id": symmetric_object_id,
        "branch": branch,
        "runtime_node_manifest_path": str(manifest_path),
        "object_resolution_status": resolution_status,
        "object_component_count": int(len(rows)),
        "resolved_component_count": int(resolved_count),
        "unresolved_component_count": int(missing_count),
        "component_resolution_status_counts": ";".join(
            f"{key}:{status_counts[key]}" for key in sorted(status_counts)
        ),
        "seed_coverage_count_actual": int(rows["seed"].nunique()),
        "seed_list_actual": ";".join(str(int(seed)) for seed in sorted(rows["seed"].unique())),
        "max_components_per_seed_actual": int(rows.groupby("seed").size().max()),
        "multi_component_seed_count_actual": int((rows.groupby("seed").size() > 1).sum()),
        "component_node_count_sum_upper_bound": int(component_node_sum),
        "component_doc_sum_sum_upper_bound": float(component_doc_sum),
        "object_node_count": object_nodes,
        "object_doc_sum": _doc_sum(object_mask, weights),
        "object_node_share": float(object_nodes / n_nodes) if n_nodes else 0.0,
        "object_mask_hash": _mask_hash(object_mask),
        "seed0_object_node_count": seed0_nodes,
        "seed0_object_doc_sum": _doc_sum(seed0_mask, weights),
        "seed0_object_mask_hash": _mask_hash(seed0_mask),
        "object_node_count_vs_component_sum_ratio": (
            float(object_nodes / component_node_sum) if component_node_sum else 0.0
        ),
        "object_node_count_vs_seed0_ratio": (
            float(object_nodes / seed0_nodes) if seed0_nodes else None
        ),
        "seed_component_node_count_min": int(min(seed_node_counts.values()))
        if seed_node_counts
        else 0,
        "seed_component_node_count_median": float(np.median(list(seed_node_counts.values())))
        if seed_node_counts
        else None,
        "seed_component_node_count_max": int(max(seed_node_counts.values()))
        if seed_node_counts
        else 0,
        "recommended_start_contract": START_CONTRACT,
        "recommended_score_contract": SCORE_CONTRACT,
        "route_execution_status": "not_executed_symmetric_object_universe_materialization_only",
        "wall_promotion_status": "not_promoted_no_route_trace",
        "quality_cost_status": "excluded_symmetric_object_universe_materialization_only",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return stats, {"object_mask": object_mask, "seed0_mask": seed0_mask}


def _object_probe_rank(value: Any) -> int:
    text = str(value)
    if text.startswith("P1_"):
        return 1
    if text.startswith("P2_"):
        return 2
    if text.startswith("P3_"):
        return 3
    if text.startswith("P4_"):
        return 4
    return 5


def _route_priority(row: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    strict = 1 if bool(row.get("strict_core_v0")) else 0
    ready = 1 if str(row.get("object_resolution_status", "")).startswith("ready_") else 0
    anchor_independent = (
        1
        if str(row.get("object_resolution_status", "")).startswith(
            "ready_anchor_independent"
        )
        else 0
    )
    probe_rank = -_object_probe_rank(row.get("probe_priority", ""))
    multi_seed = int(row.get("multi_component_seed_count_actual", 0))
    nodes = int(row.get("object_node_count", 0))
    return strict, ready, anchor_independent, probe_rank, multi_seed, nodes


def _overlap_count(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.logical_and(left, right).sum())


def _overlap_share(left: np.ndarray, right: np.ndarray) -> float:
    denom = min(int(left.sum()), int(right.sum()))
    if denom <= 0:
        return 0.0
    return float(_overlap_count(left, right) / denom)


def _case_rows(
    role_rows: pd.DataFrame,
    object_masks: dict[str, dict[str, np.ndarray]],
    graph_weights: dict[str, tuple[int, np.ndarray, Path]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_id, group in role_rows.groupby("panel_case_id", sort=True):
        candidate = group[group["role_side"].astype(str).eq("candidate")]
        control = group[group["role_side"].astype(str).eq("control")]
        branch = str(group["branch"].iloc[0])
        n_nodes, weights, _ = graph_weights[branch]
        candidate_mask = np.zeros(n_nodes, dtype=np.bool_)
        control_mask = np.zeros(n_nodes, dtype=np.bool_)
        candidate_object_id = ""
        control_object_id = ""
        if not candidate.empty:
            candidate_object_id = str(candidate.iloc[0]["symmetric_object_id"])
            candidate_mask = object_masks.get(candidate_object_id, {}).get(
                "object_mask",
                candidate_mask,
            )
        if not control.empty:
            control_object_id = str(control.iloc[0]["symmetric_object_id"])
            control_mask = object_masks.get(control_object_id, {}).get(
                "object_mask",
                control_mask,
            )
        case_mask = np.logical_or(candidate_mask, control_mask)
        overlap = _overlap_count(candidate_mask, control_mask)
        if candidate_object_id and candidate_object_id == control_object_id:
            relation = "same_symmetric_object"
        elif overlap > 0:
            relation = "overlapping_symmetric_objects"
        else:
            relation = "disjoint_symmetric_objects"
        rows.append(
            {
                "case_symmetric_universe_id": f"{case_id}__candidate_control_symmetric_object_union",
                "panel_case_id": case_id,
                "panel_case_rank": int(group["panel_case_rank"].iloc[0]),
                "analysis_tier": group["analysis_tier"].iloc[0],
                "strict_core_v0": bool(group["strict_core_v0"].iloc[0]),
                "branch": branch,
                "candidate_symmetric_object_id": candidate_object_id,
                "control_symmetric_object_id": control_object_id,
                "candidate_object_node_count": int(candidate_mask.sum()),
                "control_object_node_count": int(control_mask.sum()),
                "case_symmetric_union_node_count": int(case_mask.sum()),
                "case_symmetric_union_doc_sum": _doc_sum(case_mask, weights),
                "case_symmetric_union_node_share": float(case_mask.sum() / n_nodes)
                if n_nodes
                else 0.0,
                "candidate_control_object_overlap_node_count": overlap,
                "candidate_control_object_overlap_share_of_smaller": _overlap_share(
                    candidate_mask,
                    control_mask,
                ),
                "candidate_control_symmetric_relation": relation,
                "case_universe_role": (
                    "secondary_case_relation_diagnostic_not_primary_route_universe"
                ),
                "route_execution_status": "not_executed_symmetric_object_universe_materialization_only",
                "wall_promotion_status": "not_promoted_no_route_trace",
                "quality_cost_status": "excluded_symmetric_object_universe_materialization_only",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def materialize(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    readiness_dir = Path(args.readiness_dir)
    landscape_dir = Path(args.landscape_dir)
    symmetric_object_dir = Path(args.symmetric_object_dir)
    decomposition_dir = Path(args.symmetric_decomposition_dir)
    universe_dir = Path(args.universe_dir)
    graph_rows = _read_csv(readiness_dir / GRAPH_INPUT_ROWS_CSV)
    endpoint_registry = _read_csv(landscape_dir / ENDPOINT_REGISTRY_CSV)
    resolver_rows = _read_csv(universe_dir / SYMMETRIC_RESOLVER_ROWS_CSV)
    components = _read_csv(symmetric_object_dir / OBJECT_COMPONENTS_CSV)
    decomposition = _read_csv(decomposition_dir / OBJECT_DECOMPOSITION_CSV)

    graph_weights = _load_graph_weights(graph_rows)
    membership_registry = _pure_seed_membership_registry(endpoint_registry)
    label_cache: dict[tuple[str, str], np.ndarray] = {}
    object_cache: dict[tuple[str, str], dict[str, Any]] = {}
    object_masks_by_id: dict[str, dict[str, np.ndarray]] = {}
    role_records: list[dict[str, Any]] = []

    decomp_cols = [
        "branch",
        "symmetric_object_id",
        "endpoint_node_count",
        "seed_coverage_count",
        "seed_coverage_share",
        "max_endpoint_nodes_per_seed",
        "multi_endpoint_seed_count",
        "endpoint_weight_sum_total",
        "endpoint_weight_sum_median",
        "endpoint_unit_count_median",
        "contains_seed0_endpoint",
        "seed0_source_family_count",
        "seed0_primitive_count",
        "seed0_event_count",
        "seed0_t1_family_count",
        "mapped_claim_tiers",
        "decomposition_class",
        "mechanism_probe_label",
        "probe_priority",
        "dominant_internal_relation_class",
        "dominant_internal_relation_edge_count",
    ]
    decomp_lookup = decomposition[decomp_cols].drop_duplicates(
        ["branch", "symmetric_object_id"]
    )

    for resolver in resolver_rows.itertuples(index=False):
        branch = str(resolver.branch)
        object_id = str(resolver.symmetric_object_id)
        cache_key = (branch, object_id)
        if cache_key not in object_cache:
            object_stats, masks = _resolve_object(
                branch=branch,
                symmetric_object_id=object_id,
                components=components,
                graph_weights=graph_weights,
                membership_registry=membership_registry,
                label_cache=label_cache,
            )
            object_cache[cache_key] = object_stats
            object_masks_by_id[object_id] = masks
        record = resolver._asdict()
        record.update(object_cache[cache_key])
        role_records.append(record)

    role_rows = pd.DataFrame(role_records)
    if not role_rows.empty:
        role_rows = role_rows.merge(
            decomp_lookup,
            on=["branch", "symmetric_object_id"],
            how="left",
            suffixes=("", "_object"),
        )
        object_suffix_cols: list[str] = []
        for column in decomp_cols:
            if column in {"branch", "symmetric_object_id"}:
                continue
            object_column = f"{column}_object"
            if object_column not in role_rows.columns:
                continue
            object_suffix_cols.append(object_column)
            if column in role_rows.columns:
                role_rows[column] = role_rows[column].where(
                    role_rows[column].notna(),
                    role_rows[object_column],
                )
            else:
                role_rows[column] = role_rows[object_column]
        if object_suffix_cols:
            role_rows = role_rows.drop(columns=object_suffix_cols)
        priority_values = [_route_priority(row) for row in role_rows.to_dict("records")]
        role_rows["_priority_strict"] = [value[0] for value in priority_values]
        role_rows["_priority_ready"] = [value[1] for value in priority_values]
        role_rows["_priority_anchor_independent"] = [value[2] for value in priority_values]
        role_rows["_priority_probe_rank"] = [value[3] for value in priority_values]
        role_rows["_priority_multi_seed"] = [value[4] for value in priority_values]
        role_rows["_priority_nodes"] = [value[5] for value in priority_values]
        role_rows = role_rows.sort_values(
            [
                "_priority_strict",
                "_priority_ready",
                "_priority_anchor_independent",
                "_priority_probe_rank",
                "_priority_multi_seed",
                "_priority_nodes",
                "panel_case_rank",
                "role_side",
            ],
            ascending=[False, False, False, False, False, False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        role_rows["symmetric_object_route_priority_rank"] = np.arange(
            1,
            len(role_rows) + 1,
        )
        role_rows = role_rows.drop(
            columns=[
                "_priority_strict",
                "_priority_ready",
                "_priority_anchor_independent",
                "_priority_probe_rank",
                "_priority_multi_seed",
                "_priority_nodes",
            ]
        )

    cases = _case_rows(role_rows, object_masks_by_id, graph_weights)
    gates = _gate_matrix(role_rows=role_rows, case_rows=cases, membership_registry=membership_registry)
    summary = _summary(
        role_rows=role_rows,
        case_rows=cases,
        membership_registry=membership_registry,
        output_dir=Path(args.output_dir),
    )
    return role_rows, cases, gates, summary


def _gate_matrix(
    *,
    role_rows: pd.DataFrame,
    case_rows: pd.DataFrame,
    membership_registry: dict[tuple[str, int], tuple[Path, str]],
) -> pd.DataFrame:
    ready_roles = (
        role_rows["object_resolution_status"].astype(str).str.startswith("ready_")
        if not role_rows.empty
        else pd.Series(dtype=bool)
    )
    anchor_roles = (
        role_rows["object_resolution_status"]
        .astype(str)
        .str.startswith("ready_anchor_independent")
        if not role_rows.empty
        else pd.Series(dtype=bool)
    )
    p1_roles = (
        role_rows["probe_priority"].astype(str).str.startswith("P1_")
        if "probe_priority" in role_rows.columns
        else pd.Series(dtype=bool)
    )
    return pd.DataFrame(
        [
            {
                "gate_id": "S1_pure_seed_membership_paths_resolved",
                "status": "pass" if len(membership_registry) == 20 else "blocked",
                "evidence": f"pure_seed_membership_paths={len(membership_registry)}",
                "decision": "all Java/Rust seed ensemble memberships can be used as mask sources",
                "next_action": "resolve symmetric object components to masks",
                "claim_boundary": CLAIM_BOUNDARY,
            },
            {
                "gate_id": "S2_symmetric_object_masks_materialized",
                "status": "pass" if int(ready_roles.sum()) else "blocked",
                "evidence": f"ready_role_object_universes={int(ready_roles.sum())}",
                "decision": "role-level symmetric object universes are executable contracts",
                "next_action": "select object-level multistart pilot rows",
                "claim_boundary": CLAIM_BOUNDARY,
            },
            {
                "gate_id": "S3_anchor_independent_probe_slice_available",
                "status": "pass" if int((anchor_roles & p1_roles).sum()) else "blocked",
                "evidence": (
                    f"anchor_independent_p1_roles={int((anchor_roles & p1_roles).sum())}"
                ),
                "decision": "P1 anchor-independent object probes exist for first pilot",
                "next_action": "run bounded object-level multistart before more case-union routing",
                "claim_boundary": CLAIM_BOUNDARY,
            },
            {
                "gate_id": "S4_case_relation_is_secondary",
                "status": "pass" if not case_rows.empty else "blocked",
                "evidence": f"symmetric_case_rows={len(case_rows)}",
                "decision": "candidate/control case relation is diagnostic, not the primary universe",
                "next_action": "score case relation after object-level terminal multiplicity",
                "claim_boundary": CLAIM_BOUNDARY,
            },
        ]
    )


def _summary(
    *,
    role_rows: pd.DataFrame,
    case_rows: pd.DataFrame,
    membership_registry: dict[tuple[str, int], tuple[Path, str]],
    output_dir: Path,
) -> dict[str, Any]:
    ready = (
        role_rows["object_resolution_status"].astype(str).str.startswith("ready_")
        if not role_rows.empty
        else pd.Series(dtype=bool)
    )
    anchor = (
        role_rows["object_resolution_status"]
        .astype(str)
        .str.startswith("ready_anchor_independent")
        if not role_rows.empty
        else pd.Series(dtype=bool)
    )
    strict = (
        role_rows["strict_core_v0"].astype(bool)
        if not role_rows.empty
        else pd.Series(dtype=bool)
    )
    p1 = (
        role_rows["probe_priority"].astype(str).str.startswith("P1_")
        if "probe_priority" in role_rows.columns
        else pd.Series(dtype=bool)
    )
    strict_anchor_p1_count = int((strict & anchor & p1).sum()) if len(ready) else 0
    strict_ready_count = int((strict & ready).sum()) if len(ready) else 0
    return {
        "schema": "nanoclustering_symmetric_object_universe_summary.v1",
        "status": "executed_symmetric_object_universe_materialization",
        "output_dir": str(output_dir),
        "pure_seed_membership_path_count": int(len(membership_registry)),
        "role_universe_count": int(len(role_rows)),
        "case_relation_count": int(len(case_rows)),
        "ready_role_universe_count": int(ready.sum()) if len(ready) else 0,
        "anchor_independent_ready_role_count": int(anchor.sum()) if len(anchor) else 0,
        "strict_core_ready_role_count": strict_ready_count,
        "strict_core_anchor_independent_p1_role_count": strict_anchor_p1_count,
        "unique_symmetric_object_count": int(role_rows["symmetric_object_id"].nunique())
        if not role_rows.empty
        else 0,
        "object_node_count_median": float(role_rows["object_node_count"].median())
        if not role_rows.empty
        else None,
        "object_node_count_max": int(role_rows["object_node_count"].max())
        if not role_rows.empty
        else 0,
        "object_node_count_vs_seed0_ratio_median": float(
            role_rows["object_node_count_vs_seed0_ratio"].dropna().median()
        )
        if "object_node_count_vs_seed0_ratio" in role_rows.columns
        and role_rows["object_node_count_vs_seed0_ratio"].notna().any()
        else None,
        "object_node_count_vs_component_sum_ratio_median": float(
            role_rows["object_node_count_vs_component_sum_ratio"].median()
        )
        if not role_rows.empty
        else None,
        "case_overlap_share_median": float(
            case_rows["candidate_control_object_overlap_share_of_smaller"].median()
        )
        if not case_rows.empty
        else None,
        "recommended_first_pilot": (
            "strict_core_anchor_independent_p1_top6_by_symmetric_object_route_priority_rank"
            if strict_anchor_p1_count
            else "strict_core_ready_object_top6_by_symmetric_object_route_priority_rank"
            if strict_ready_count
            else "blocked_no_ready_symmetric_object_universe"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(output_dir: Path, summary: dict[str, Any], role_rows: pd.DataFrame) -> None:
    lines = [
        "# NanoClustering Symmetric-Object Universe Plan",
        "",
        f"- status: `{summary['status']}`",
        f"- pure_seed_membership_path_count: {summary['pure_seed_membership_path_count']}",
        f"- role_universe_count: {summary['role_universe_count']}",
        f"- ready_role_universe_count: {summary['ready_role_universe_count']}",
        f"- anchor_independent_ready_role_count: {summary['anchor_independent_ready_role_count']}",
        f"- strict_core_anchor_independent_p1_role_count: {summary['strict_core_anchor_independent_p1_role_count']}",
        f"- unique_symmetric_object_count: {summary['unique_symmetric_object_count']}",
        f"- object_node_count_median: {summary['object_node_count_median']}",
        f"- object_node_count_max: {summary['object_node_count_max']}",
        f"- object_node_count_vs_seed0_ratio_median: {summary['object_node_count_vs_seed0_ratio_median']}",
        f"- object_node_count_vs_component_sum_ratio_median: {summary['object_node_count_vs_component_sum_ratio_median']}",
        f"- recommended_first_pilot: `{summary['recommended_first_pilot']}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Design Read",
        "",
        "This materializes symmetric endpoint objects as the next primary "
        "route universe. The object mask is the all-seed component union for a "
        "single symmetric object, so the route can test optimizer dynamics "
        "inside an anchor-independent object rather than inside a seed0-local "
        "pair or a loose candidate/control case union.",
        "",
        "Candidate/control case rows are retained only as secondary relation "
        "diagnostics. The first route pilot should start from role-level P1 "
        "anchor-independent objects, then evaluate case contrast only after "
        "object-level terminal multiplicity is known.",
        "",
        "## Pilot Slice",
    ]
    if role_rows.empty:
        lines.append("- no role universes")
    else:
        for row in role_rows.head(12).itertuples(index=False):
            data = row._asdict()
            lines.append(
                "- "
                f"rank {data['symmetric_object_route_priority_rank']}: "
                f"{data['role_id']} object={data['symmetric_object_id']} "
                f"role={data['role_side']} "
                f"strict={data['strict_core_v0']} "
                f"status={data['object_resolution_status']} "
                f"probe={data.get('probe_priority', '')} "
                f"nodes={data.get('object_node_count', '')} "
                f"seed_coverage={data.get('seed_coverage_count_actual', '')} "
                f"multi_seed={data.get('multi_component_seed_count_actual', '')}"
            )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- `{_rel(output_dir / SYMMETRIC_ROLE_ROWS_CSV)}`",
            f"- `{_rel(output_dir / SYMMETRIC_CASE_ROWS_CSV)}`",
            f"- `{_rel(output_dir / SYMMETRIC_GATE_MATRIX_CSV)}`",
            f"- `{_rel(output_dir / SYMMETRIC_SUMMARY_JSON)}`",
        ]
    )
    (output_dir / SYMMETRIC_REPORT_MD).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-dir", type=Path, default=DEFAULT_READINESS_DIR)
    parser.add_argument("--landscape-dir", type=Path, default=DEFAULT_LANDSCAPE_DIR)
    parser.add_argument("--symmetric-object-dir", type=Path, default=DEFAULT_SYMMETRIC_OBJECT_DIR)
    parser.add_argument(
        "--symmetric-decomposition-dir",
        type=Path,
        default=DEFAULT_SYMMETRIC_DECOMPOSITION_DIR,
    )
    parser.add_argument("--universe-dir", type=Path, default=DEFAULT_UNIVERSE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    role_rows, case_rows, gates, summary = materialize(args)
    _write_csv(role_rows, output_dir / SYMMETRIC_ROLE_ROWS_CSV)
    _write_csv(case_rows, output_dir / SYMMETRIC_CASE_ROWS_CSV)
    _write_csv(gates, output_dir / SYMMETRIC_GATE_MATRIX_CSV)
    _json_dump(
        {
            "schema": "nanoclustering_symmetric_object_universe_materialization.v1",
            "readiness_dir": str(args.readiness_dir),
            "landscape_dir": str(args.landscape_dir),
            "symmetric_object_dir": str(args.symmetric_object_dir),
            "symmetric_decomposition_dir": str(args.symmetric_decomposition_dir),
            "universe_dir": str(args.universe_dir),
            "output_dir": str(output_dir),
            "claim_boundary": CLAIM_BOUNDARY,
        },
        output_dir / SYMMETRIC_CONFIG_JSON,
    )
    _json_dump(summary, output_dir / SYMMETRIC_SUMMARY_JSON)
    _write_report(output_dir, summary, role_rows)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
