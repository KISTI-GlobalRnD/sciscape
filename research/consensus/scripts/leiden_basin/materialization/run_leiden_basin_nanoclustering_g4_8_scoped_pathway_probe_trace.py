#!/usr/bin/env python3
"""Execute the scoped G4.8 Stage 2A pathway-probe trace contract.

This runner consumes
``design_leiden_basin_nanoclustering_g4_8_scoped_pathway_probe_contract.py``.
It executes only the 30 predeclared route-plan rows for the two ready pairs,
expanding each route row into its fixed edge-weight-fraction schedule and seed
replicates on the existing local induced graph surface.

It is a tiny local route-trace diagnostic. It does not run the full
NanoClustering graph, promote basin walls, evaluate downstream quality/cost
value, or claim a method/algorithm success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from design_leiden_basin_nanoclustering_g4_8_scoped_pathway_probe_contract import (
    CONTROL_GUARD_ROWS_CSV as CONTRACT_CONTROL_GUARD_ROWS_CSV,
    DEFAULT_OUTPUT_DIR as DEFAULT_CONTRACT_DIR,
    GATE_MATRIX_CSV as CONTRACT_GATE_MATRIX_CSV,
    REQUIRED_MEASUREMENTS,
    ROUTE_PLAN_ROWS_CSV as CONTRACT_ROUTE_PLAN_ROWS_CSV,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)
from run_leiden_basin_nanoclustering_symmetric_object_variable_pair_local_ablation import (
    DEFAULT_OUTPUT_DIR as DEFAULT_LOCAL_ABLATION_DIR,
    LOCAL_GRAPH_ROWS_CSV,
    SEED_RUNS_CSV as LOCAL_ABLATION_SEED_RUNS_CSV,
    _build_igraph,
    _canonical_groups,
    _collect_induced_edges_by_branch,
    _initial_membership,
    _mechanism_read,
    _node_doc_lookup,
    _read_json,
    _signature_id,
)
from sciscape.clustering.runner import LeidenRunner


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_scoped_pathway_probe_trace_gamma1e5_20260604"
)

TRACE_ROWS_CSV = "nanoclustering_g4_8_scoped_pathway_probe_trace_rows.csv"
SEED_ROUTE_SUMMARY_CSV = (
    "nanoclustering_g4_8_scoped_pathway_probe_trace_seed_route_summary.csv"
)
ROUTE_CONTRACT_SUMMARY_CSV = (
    "nanoclustering_g4_8_scoped_pathway_probe_trace_route_contract_summary.csv"
)
GATE_MATRIX_CSV = "nanoclustering_g4_8_scoped_pathway_probe_trace_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_g4_8_scoped_pathway_probe_trace_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_scoped_pathway_probe_trace_config.json"
REPORT_MD = "nanoclustering_g4_8_scoped_pathway_probe_trace_report.md"

RUN_STATUS = "executed_nanoclustering_g4_8_scoped_pathway_probe_trace"
ROUTE_EXECUTION_STATUS = "executed_scoped_local_fractional_edge_route_trace"
WALL_PROMOTION_STATUS = "not_promoted_route_trace_audit_required"
METHOD_STATUS = "local_route_trace_diagnostic_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 scoped pathway-probe trace only; executes the tiny "
    "30-row route contract on local induced graphs with predeclared edge-weight "
    "fractions. It does not run full NanoClustering replay, promote basin walls, "
    "evaluate downstream quality/cost value, or claim method/algorithm success."
)

SCHEDULES: dict[str, tuple[dict[str, Any], ...]] = {
    "bridge_release_interpolation_probe": tuple(
        {
            "step_index": index,
            "step_label": f"bridge_fraction_{fraction:.2f}",
            "direct_edge_weight_fraction": 1.0,
            "bridge_edge_weight_fraction": float(fraction),
            "expected_final_anchor_variant": "drop_bridge_edges",
        }
        for index, fraction in enumerate((1.0, 0.75, 0.50, 0.25, 0.0), start=1)
    ),
    "direct_dependency_collapse_guard": tuple(
        {
            "step_index": index,
            "step_label": f"direct_fraction_{fraction:.2f}",
            "direct_edge_weight_fraction": float(fraction),
            "bridge_edge_weight_fraction": 1.0,
            "expected_final_anchor_variant": "drop_direct_edge",
        }
        for index, fraction in enumerate((1.0, 0.75, 0.50, 0.25, 0.0), start=1)
    ),
    "drop_both_collapse_guard": tuple(
        {
            "step_index": index,
            "step_label": f"direct_bridge_fraction_{fraction:.2f}",
            "direct_edge_weight_fraction": float(fraction),
            "bridge_edge_weight_fraction": float(fraction),
            "expected_final_anchor_variant": "drop_direct_and_bridge_edges",
        }
        for index, fraction in enumerate((1.0, 0.50, 0.0), start=1)
    ),
}

ANCHOR_VARIANT_TO_ASSIGNMENT = {
    "original": "original_source_anchor",
    "drop_bridge_edges": "drop_bridge_target_anchor",
    "drop_direct_edge": "drop_direct_guard_anchor",
    "drop_direct_and_bridge_edges": "drop_both_guard_anchor",
}
EXPECTED_FINAL_ASSIGNMENT = {
    family: ANCHOR_VARIANT_TO_ASSIGNMENT[steps[-1]["expected_final_anchor_variant"]]
    for family, steps in SCHEDULES.items()
}
GRAPH_VARIANT_ORDER = (
    "original",
    "drop_bridge_edges",
    "drop_direct_edge",
    "drop_direct_and_bridge_edges",
)


def _parse_node_ids(value: Any) -> list[int]:
    return [int(part) for part in str(value).split(";") if str(part).strip()]


def _scaled_local_edges(
    *,
    induced_edges: pd.DataFrame,
    node_ids: list[int],
    left_node: int,
    right_node: int,
    bridge_nodes: set[int],
    direct_fraction: float,
    bridge_fraction: float,
) -> pd.DataFrame:
    node_set = set(int(node_id) for node_id in node_ids)
    edges = induced_edges[
        induced_edges["source"].astype(int).isin(node_set)
        & induced_edges["target"].astype(int).isin(node_set)
    ].copy()
    if edges.empty:
        return edges

    source = edges["source"].to_numpy(dtype=np.int64)
    target = edges["target"].to_numpy(dtype=np.int64)
    direct_key = tuple(sorted((int(left_node), int(right_node))))
    is_direct = np.asarray(
        [tuple(sorted((int(src), int(dst)))) == direct_key for src, dst in zip(source, target, strict=True)],
        dtype=np.bool_,
    )
    pair_nodes = {int(left_node), int(right_node)}
    is_pair_bridge = np.asarray(
        [
            (int(src) in pair_nodes and int(dst) in bridge_nodes)
            or (int(dst) in pair_nodes and int(src) in bridge_nodes)
            for src, dst in zip(source, target, strict=True)
        ],
        dtype=np.bool_,
    )
    scale = np.ones(len(edges), dtype=np.float64)
    scale[is_direct] *= float(direct_fraction)
    scale[is_pair_bridge] *= float(bridge_fraction)
    edges["weight"] = edges["weight"].to_numpy(dtype=np.float64) * scale
    return edges[edges["weight"].astype(float).gt(0.0)].copy()


def _edge_weight_parts(
    *,
    induced_edges: pd.DataFrame,
    node_ids: list[int],
    left_node: int,
    right_node: int,
    bridge_nodes: set[int],
) -> dict[str, float]:
    original = _scaled_local_edges(
        induced_edges=induced_edges,
        node_ids=node_ids,
        left_node=left_node,
        right_node=right_node,
        bridge_nodes=bridge_nodes,
        direct_fraction=1.0,
        bridge_fraction=1.0,
    )
    if original.empty:
        return {
            "original_local_edge_weight_sum": 0.0,
            "original_direct_edge_weight": 0.0,
            "original_pair_bridge_edge_weight_sum": 0.0,
        }
    source = original["source"].to_numpy(dtype=np.int64)
    target = original["target"].to_numpy(dtype=np.int64)
    weights = original["weight"].to_numpy(dtype=np.float64)
    direct_key = tuple(sorted((int(left_node), int(right_node))))
    direct_mask = np.asarray(
        [tuple(sorted((int(src), int(dst)))) == direct_key for src, dst in zip(source, target, strict=True)],
        dtype=np.bool_,
    )
    pair_nodes = {int(left_node), int(right_node)}
    bridge_mask = np.asarray(
        [
            (int(src) in pair_nodes and int(dst) in bridge_nodes)
            or (int(dst) in pair_nodes and int(src) in bridge_nodes)
            for src, dst in zip(source, target, strict=True)
        ],
        dtype=np.bool_,
    )
    return {
        "original_local_edge_weight_sum": float(weights.sum()),
        "original_direct_edge_weight": float(weights[direct_mask].sum()),
        "original_pair_bridge_edge_weight_sum": float(weights[bridge_mask].sum()),
    }


def _groups_to_membership(node_ids: list[int], endpoint_signature: str) -> list[int]:
    groups = json.loads(str(endpoint_signature))
    index = {int(node_id): offset for offset, node_id in enumerate(node_ids)}
    labels = [-1 for _ in node_ids]
    for label, group in enumerate(groups):
        for node in group:
            node_id = int(node)
            if node_id not in index:
                raise ValueError(f"anchor endpoint contains unknown local node: {node_id}")
            labels[index[node_id]] = int(label)
    next_label = len(groups)
    for offset, label in enumerate(labels):
        if label < 0:
            labels[offset] = next_label
            next_label += 1
    return labels


def _coassignment_distance(left: list[int], right: list[int]) -> float:
    if len(left) != len(right):
        raise ValueError("membership lengths do not match")
    n = len(left)
    total = n * (n - 1) // 2
    if total == 0:
        return 0.0
    mismatch = 0
    for i in range(n):
        for j in range(i + 1, n):
            mismatch += int((left[i] == left[j]) != (right[i] == right[j]))
    return float(mismatch / total)


def _anchor_lookup(seed_runs: pd.DataFrame, route_plan: pd.DataFrame) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    keys = route_plan[["local_pair_id", "start_condition"]].drop_duplicates()
    filtered = seed_runs.merge(keys, on=["local_pair_id", "start_condition"], how="inner")
    lookup: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for row in filtered.itertuples(index=False):
        key = (
            str(row.local_pair_id),
            str(row.start_condition),
            int(row.seed),
            str(row.graph_variant),
        )
        lookup[key] = row._asdict()
    return lookup


def _anchor_match_data(
    *,
    anchors: dict[str, dict[str, Any]],
    result_signature_id: str,
    result_membership: list[int],
    expected_final_variant: str,
) -> dict[str, Any]:
    matches: list[str] = []
    distances: dict[str, float | None] = {}
    for variant in GRAPH_VARIANT_ORDER:
        anchor = anchors.get(variant)
        if anchor is None:
            distances[f"support_distance_to_{variant}"] = None
            continue
        assignment = ANCHOR_VARIANT_TO_ASSIGNMENT[variant]
        if str(anchor["endpoint_signature_id"]) == str(result_signature_id):
            matches.append(assignment)
        distances[f"support_distance_to_{variant}"] = _coassignment_distance(
            result_membership,
            anchor["membership"],
        )
    expected_assignment = ANCHOR_VARIANT_TO_ASSIGNMENT[expected_final_variant]
    if not matches:
        endpoint_assignment = "unknown_new_endpoint"
    elif len(matches) == 1:
        endpoint_assignment = matches[0]
    else:
        endpoint_assignment = "ambiguous_anchor_match:" + ";".join(sorted(matches))
    known_distances = [
        value for value in distances.values() if value is not None and math.isfinite(float(value))
    ]
    min_distance = min(known_distances) if known_distances else None
    return {
        "endpoint_assignment_by_step": endpoint_assignment,
        "matched_anchor_assignments": ";".join(sorted(matches)),
        "matches_original_anchor": "original_source_anchor" in matches,
        "matches_expected_final_anchor": expected_assignment in matches,
        "expected_final_anchor_assignment": expected_assignment,
        "support_distance_min_known_anchor": min_distance,
        "support_incompatibility_check": bool(not matches and (min_distance is None or min_distance > 0.0)),
        **distances,
    }


def _step_schedule_rows(route_plan: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for route in route_plan.itertuples(index=False):
        route_data = route._asdict()
        family = str(route_data["planned_route_family"])
        if family not in SCHEDULES:
            raise ValueError(f"route family has no schedule: {family}")
        for step in SCHEDULES[family]:
            rows.append({**route_data, **step})
    return pd.DataFrame(rows)


def _trace_rows(
    *,
    route_plan: pd.DataFrame,
    contract_dir: Path,
    local_ablation_dir: Path,
    gamma: float,
    seeds: int,
    n_iterations: int,
    edge_chunk_size: int,
) -> tuple[pd.DataFrame, int, int]:
    local_config = _read_json(
        local_ablation_dir / "nanoclustering_symmetric_object_variable_pair_local_ablation_config.json"
    )
    graph_mechanism_dir = Path(str(local_config["graph_mechanism_dir"]))
    difference_dir = Path(str(local_config["difference_dir"]))
    node_rows = _read_csv(
        difference_dir / "nanoclustering_symmetric_object_terminal_difference_node_rows.csv"
    )
    doc_lookup = _node_doc_lookup(node_rows)
    local_specs = _read_csv(local_ablation_dir / LOCAL_GRAPH_ROWS_CSV)
    candidate_pair_ids = set(route_plan["local_pair_id"].astype(str))
    specs = local_specs[local_specs["local_pair_id"].astype(str).isin(candidate_pair_ids)].copy()
    if specs["local_pair_id"].nunique() != len(candidate_pair_ids):
        missing = sorted(candidate_pair_ids - set(specs["local_pair_id"].astype(str)))
        raise ValueError(f"missing local specs for candidate pairs: {missing}")

    target_nodes_by_branch: dict[str, set[int]] = {}
    for spec in specs.itertuples(index=False):
        target_nodes_by_branch.setdefault(str(spec.branch), set()).update(
            _parse_node_ids(spec.local_node_ids)
        )
    induced_edges_by_branch = _collect_induced_edges_by_branch(
        graph_mechanism_dir=graph_mechanism_dir,
        target_nodes_by_branch=target_nodes_by_branch,
        edge_chunk_size=int(edge_chunk_size),
    )

    seed_runs = _read_csv(local_ablation_dir / LOCAL_ABLATION_SEED_RUNS_CSV)
    anchors_by_key = _anchor_lookup(seed_runs, route_plan)
    step_rows = _step_schedule_rows(route_plan)
    spec_by_pair = {
        str(row.local_pair_id): row._asdict()
        for row in specs.sort_values("local_pair_id", kind="mergesort").itertuples(index=False)
    }
    output_rows: list[dict[str, Any]] = []

    for step_row in step_rows.itertuples(index=False):
        step = step_row._asdict()
        local_pair_id = str(step["local_pair_id"])
        spec = spec_by_pair[local_pair_id]
        object_role_id = str(spec["object_role_universe_id"])
        branch = str(spec["branch"])
        left = int(spec["left_node_id"])
        right = int(spec["right_node_id"])
        node_ids = _parse_node_ids(spec["local_node_ids"])
        bridge_nodes = set(_parse_node_ids(spec["selected_bridge_node_ids"]))
        node_sizes = [
            int(doc_lookup.get((object_role_id, int(node_id)), 1))
            for node_id in node_ids
        ]
        induced_edges = induced_edges_by_branch.get(
            branch,
            pd.DataFrame(columns=["source", "target", "weight"]),
        )
        local_edges = _scaled_local_edges(
            induced_edges=induced_edges,
            node_ids=node_ids,
            left_node=left,
            right_node=right,
            bridge_nodes=bridge_nodes,
            direct_fraction=float(step["direct_edge_weight_fraction"]),
            bridge_fraction=float(step["bridge_edge_weight_fraction"]),
        )
        edge_parts = _edge_weight_parts(
            induced_edges=induced_edges,
            node_ids=node_ids,
            left_node=left,
            right_node=right,
            bridge_nodes=bridge_nodes,
        )
        graph = _build_igraph(node_ids, local_edges)
        runner = LeidenRunner(graph, objective="cpm", default_iterations=int(n_iterations))
        initial_membership = _initial_membership(
            start_condition=str(step["start_condition"]),
            node_ids=node_ids,
            left_node=left,
            right_node=right,
            bridge_nodes=bridge_nodes,
        )
        initial_signature = _signature_id(_canonical_groups(node_ids, initial_membership))
        for seed in range(int(seeds)):
            anchors: dict[str, dict[str, Any]] = {}
            for variant in GRAPH_VARIANT_ORDER:
                anchor = anchors_by_key.get(
                    (local_pair_id, str(step["start_condition"]), int(seed), variant)
                )
                if anchor is None:
                    continue
                anchors[variant] = {
                    **anchor,
                    "membership": _groups_to_membership(
                        node_ids,
                        str(anchor["endpoint_signature"]),
                    ),
                }
            result = runner.run(
                float(gamma),
                seed=int(seed),
                initial_membership=initial_membership,
                node_sizes=node_sizes,
            )
            membership = list(map(int, result.membership))
            groups = _canonical_groups(node_ids, membership)
            result_signature_id = _signature_id(groups)
            read = _mechanism_read(
                membership=membership,
                node_ids=node_ids,
                left_node=left,
                right_node=right,
                bridge_nodes=bridge_nodes,
            )
            anchor_data = _anchor_match_data(
                anchors=anchors,
                result_signature_id=result_signature_id,
                result_membership=membership,
                expected_final_variant=str(step["expected_final_anchor_variant"]),
            )
            support_distance = anchor_data["support_distance_min_known_anchor"]
            polish_reversion = bool(anchor_data["matches_original_anchor"])
            output_rows.append(
                {
                    "route_trace_row_id": (
                        f"{step['route_contract_id']}__seed{seed:02d}"
                        f"__step{int(step['step_index']):02d}"
                    ),
                    "route_contract_id": str(step["route_contract_id"]),
                    "validation_unit_id": str(step["validation_unit_id"]),
                    "local_pair_id": local_pair_id,
                    "branch": branch,
                    "left_node_id": left,
                    "right_node_id": right,
                    "object_role_universe_id": object_role_id,
                    "start_condition": str(step["start_condition"]),
                    "planned_route_family": str(step["planned_route_family"]),
                    "route_family_role": str(step["route_family_role"]),
                    "step_index": int(step["step_index"]),
                    "step_label": str(step["step_label"]),
                    "seed": int(seed),
                    "gamma": float(gamma),
                    "n_iterations": int(n_iterations),
                    "direct_edge_weight_fraction": float(step["direct_edge_weight_fraction"]),
                    "bridge_edge_weight_fraction": float(step["bridge_edge_weight_fraction"]),
                    "expected_final_anchor_variant": str(step["expected_final_anchor_variant"]),
                    "local_node_count": int(len(node_ids)),
                    "selected_bridge_count": int(len(bridge_nodes)),
                    "local_edge_count": int(graph.ecount()),
                    "local_edge_weight_sum": float(sum(graph.es["weight"])) if graph.ecount() else 0.0,
                    "active_direct_edge_weight": float(
                        edge_parts["original_direct_edge_weight"]
                        * float(step["direct_edge_weight_fraction"])
                    ),
                    "active_pair_bridge_edge_weight_sum": float(
                        edge_parts["original_pair_bridge_edge_weight_sum"]
                        * float(step["bridge_edge_weight_fraction"])
                    ),
                    **edge_parts,
                    "initial_endpoint_signature_id": initial_signature,
                    "result_endpoint_signature_id": result_signature_id,
                    "result_endpoint_signature": json.dumps(groups, sort_keys=True),
                    "objective_value_by_step": float(result.quality),
                    "cluster_count": int(result.cluster_count),
                    **read,
                    **anchor_data,
                    "support_distance_by_step": support_distance,
                    "polish_changed_from_initial": result_signature_id != initial_signature,
                    "polish_reverted_to_original_anchor": polish_reversion,
                    "polish_reversion_check": polish_reversion,
                    "post_route_endpoint_assignment_available": True,
                    "route_execution_status": ROUTE_EXECUTION_STATUS,
                    "wall_promotion_status": WALL_PROMOTION_STATUS,
                    "method_status": METHOD_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                    "run_status": RUN_STATUS,
                    "contract_dir": str(contract_dir),
                    "local_ablation_dir": str(local_ablation_dir),
                }
            )
    rows = pd.DataFrame(output_rows)
    if rows.empty:
        return rows, int(len(step_rows)), int(specs["local_pair_id"].nunique())
    rows = rows.sort_values(
        ["route_contract_id", "seed", "step_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    group_cols = ["route_contract_id", "seed"]
    rows["objective_start_value"] = rows.groupby(group_cols, sort=False)[
        "objective_value_by_step"
    ].transform("first")
    rows["objective_delta_from_start"] = (
        rows["objective_value_by_step"] - rows["objective_start_value"]
    )
    rows["objective_debt_from_start"] = np.maximum(
        0.0,
        rows["objective_start_value"] - rows["objective_value_by_step"],
    )
    rows["objective_min_so_far"] = rows.groupby(group_cols, sort=False)[
        "objective_value_by_step"
    ].cummin()
    rows["objective_recovery_from_min"] = (
        rows["objective_value_by_step"] - rows["objective_min_so_far"]
    )
    return rows, int(len(step_rows)), int(specs["local_pair_id"].nunique())


def _seed_route_summary(trace_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "route_contract_id",
        "validation_unit_id",
        "local_pair_id",
        "start_condition",
        "planned_route_family",
        "seed",
    ]
    for keys, group in trace_rows.groupby(group_cols, sort=False):
        key_data = dict(zip(group_cols, keys, strict=True))
        ordered = group.sort_values("step_index", kind="mergesort")
        first = ordered.iloc[0]
        last = ordered.iloc[-1]
        expected_assignment = str(first["expected_final_anchor_assignment"])
        assignments = list(map(str, ordered["endpoint_assignment_by_step"]))
        signature_count = int(ordered["result_endpoint_signature_id"].nunique())
        endpoint_transition_observed = signature_count > 1
        source_start = bool(first["matches_original_anchor"])
        expected_final = bool(last["matches_expected_final_anchor"])
        unknown_count = int(ordered["endpoint_assignment_by_step"].eq("unknown_new_endpoint").sum())
        if endpoint_transition_observed and source_start and expected_final:
            trace_class = "source_to_expected_anchor_transition"
        elif not endpoint_transition_observed:
            trace_class = "no_endpoint_transition"
        elif expected_final:
            trace_class = "expected_final_anchor_reached_without_source_start"
        elif unknown_count:
            trace_class = "contains_unknown_endpoint"
        else:
            trace_class = "other_anchor_transition"
        rows.append(
            {
                **key_data,
                "route_step_count": int(len(ordered)),
                "endpoint_transition_observed": bool(endpoint_transition_observed),
                "source_start_anchor_matched": bool(source_start),
                "expected_final_anchor_reached": bool(expected_final),
                "expected_final_anchor_assignment": expected_assignment,
                "first_endpoint_assignment": str(first["endpoint_assignment_by_step"]),
                "final_endpoint_assignment": str(last["endpoint_assignment_by_step"]),
                "endpoint_assignment_sequence": " -> ".join(assignments),
                "distinct_result_endpoint_count": signature_count,
                "unknown_endpoint_step_count": unknown_count,
                "max_objective_debt_from_start": float(
                    ordered["objective_debt_from_start"].max()
                ),
                "max_objective_recovery_from_min": float(
                    ordered["objective_recovery_from_min"].max()
                ),
                "min_support_distance_known_anchor": float(
                    ordered["support_distance_min_known_anchor"].min()
                ),
                "max_support_distance_known_anchor": float(
                    ordered["support_distance_min_known_anchor"].max()
                ),
                "support_incompatibility_step_count": int(
                    ordered["support_incompatibility_check"].astype(bool).sum()
                ),
                "polish_reverted_to_original_step_count": int(
                    ordered["polish_reverted_to_original_anchor"].astype(bool).sum()
                ),
                "route_trace_class": trace_class,
                "wall_claim_ready_after_trace": False,
                "wall_claim_block_reason": (
                    "route traces are materialized, but distinct basin-pair relation, "
                    "direct-path audit, and support-incompatibility audit are not yet "
                    "sufficient for wall promotion"
                ),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows)


def _route_contract_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "route_contract_id",
        "validation_unit_id",
        "local_pair_id",
        "start_condition",
        "planned_route_family",
    ]
    for keys, group in seed_summary.groupby(group_cols, sort=False):
        key_data = dict(zip(group_cols, keys, strict=True))
        seed_count = int(group["seed"].nunique())
        expected_count = int(
            group["route_trace_class"].eq("source_to_expected_anchor_transition").sum()
        )
        no_transition_count = int(group["route_trace_class"].eq("no_endpoint_transition").sum())
        unknown_count = int(group["unknown_endpoint_step_count"].gt(0).sum())
        if expected_count == seed_count and seed_count > 0:
            status = "all_seeds_source_to_expected_anchor_transition"
        elif expected_count > 0:
            status = "partial_source_to_expected_anchor_transition"
        elif no_transition_count == seed_count and seed_count > 0:
            status = "no_seed_endpoint_transition"
        elif unknown_count:
            status = "contains_unknown_endpoint_trace"
        else:
            status = "mixed_nonexpected_anchor_trace"
        rows.append(
            {
                **key_data,
                "seed_count": seed_count,
                "expected_transition_seed_count": expected_count,
                "no_transition_seed_count": no_transition_count,
                "unknown_endpoint_seed_count": unknown_count,
                "route_contract_trace_status": status,
                "max_objective_debt_from_start": float(
                    group["max_objective_debt_from_start"].max()
                ),
                "max_objective_recovery_from_min": float(
                    group["max_objective_recovery_from_min"].max()
                ),
                "support_incompatibility_seed_count": int(
                    group["support_incompatibility_step_count"].gt(0).sum()
                ),
                "wall_claim_ready_after_trace": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows)


def _gate_row(
    gate_id: str,
    question: str,
    observed: Any,
    minimum_or_rule: str,
    passed: bool,
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "question": question,
        "observed": observed,
        "minimum_or_rule": minimum_or_rule,
        "gate_status": "pass" if bool(passed) else "fail",
    }


def _gate_matrix(
    *,
    contract_gates: pd.DataFrame,
    route_plan: pd.DataFrame,
    control_guards: pd.DataFrame,
    trace_rows: pd.DataFrame,
    step_config_count: int,
    contract_summary: pd.DataFrame,
) -> pd.DataFrame:
    required_trace_columns = {
        "route_trace_row_id",
        "objective_value_by_step",
        "objective_debt_from_start",
        "objective_recovery_from_min",
        "endpoint_assignment_by_step",
        "support_distance_by_step",
        "support_distance_min_known_anchor",
        "polish_reversion_check",
        "polish_reverted_to_original_anchor",
        "support_incompatibility_check",
        "post_route_endpoint_assignment_available",
    }
    start_rows = trace_rows[trace_rows["step_index"].astype(int).eq(1)]
    final_rows = (
        trace_rows.sort_values("step_index", kind="mergesort")
        .groupby(["route_contract_id", "seed"], sort=False)
        .tail(1)
    )
    expected_status_counts = (
        contract_summary["route_contract_trace_status"].value_counts().to_dict()
        if not contract_summary.empty
        else {}
    )
    rows = [
        _gate_row(
            "G1_contract_gates_pass",
            "Did every upstream scoped pathway-probe contract gate pass?",
            contract_gates["gate_status"].value_counts().to_dict(),
            "all contract gates pass",
            bool(contract_gates["gate_status"].astype(str).eq("pass").all()),
        ),
        _gate_row(
            "G2_exact_route_contract_scope",
            "Was execution restricted to the 30 predeclared route-plan rows?",
            f"route_plan_rows={len(route_plan)} executed_route_contracts={trace_rows['route_contract_id'].nunique()}",
            "30 route-plan rows, no extra route contracts",
            len(route_plan) == 30
            and trace_rows["route_contract_id"].nunique() == 30
            and set(trace_rows["route_contract_id"]) == set(route_plan["route_contract_id"]),
        ),
        _gate_row(
            "G3_controls_retained_not_executed",
            "Were noncandidate controls retained but excluded from route execution?",
            f"control_guards={len(control_guards)} trace_control_rows=0",
            "65 control guards and zero executed control rows",
            len(control_guards) == 65
            and not set(control_guards.get("validation_unit_id", pd.Series(dtype=str)).astype(str))
            & set(trace_rows["validation_unit_id"].astype(str)),
        ),
        _gate_row(
            "G4_schedule_steps_predeclared",
            "Were route rows expanded only into their predeclared fraction schedules?",
            f"route_step_configs={step_config_count} trace_rows={len(trace_rows)}",
            "130 route-step configs: 10*(5+5+3), with seed replicates only",
            step_config_count == 130 and len(trace_rows) >= step_config_count,
        ),
        _gate_row(
            "G5_required_measurements_materialized",
            "Did trace rows include the required objective, endpoint, polish, and support fields?",
            sorted(required_trace_columns & set(trace_rows.columns)),
            "all required trace-measurement columns present",
            required_trace_columns.issubset(set(trace_rows.columns)),
        ),
        _gate_row(
            "G6_start_step_reconciles_original_anchor",
            "Does every fraction-1 start step match the same-seed original anchor?",
            f"start_rows={len(start_rows)} original_matches={int(start_rows['matches_original_anchor'].astype(bool).sum())}",
            "all start rows match original anchor",
            not start_rows.empty and bool(start_rows["matches_original_anchor"].astype(bool).all()),
        ),
        _gate_row(
            "G7_final_step_reconciles_expected_anchor",
            "Does every final fraction step match the predeclared expected anchor?",
            f"final_rows={len(final_rows)} expected_matches={int(final_rows['matches_expected_final_anchor'].astype(bool).sum())}",
            "all final rows match expected route-family anchor",
            not final_rows.empty
            and bool(final_rows["matches_expected_final_anchor"].astype(bool).all()),
        ),
        _gate_row(
            "G8_trace_classification_no_wall_promotion",
            "Are route traces classified while wall claims remain closed?",
            expected_status_counts,
            "trace classification exists and wall_claim_ready_after_trace is false",
            not contract_summary.empty
            and bool(contract_summary["wall_claim_ready_after_trace"].eq(False).all()),
        ),
        _gate_row(
            "G9_no_method_quality_or_full_replay_claim",
            "Are method, quality/cost, full replay, and algorithm claims closed?",
            CLAIM_BOUNDARY,
            "claim boundary explicitly closed",
            True,
        ),
    ]
    return pd.DataFrame(rows)


def _summary(
    *,
    contract_dir: Path,
    local_ablation_dir: Path,
    output_dir: Path,
    route_plan: pd.DataFrame,
    control_guards: pd.DataFrame,
    trace_rows: pd.DataFrame,
    seed_summary: pd.DataFrame,
    contract_summary: pd.DataFrame,
    gates: pd.DataFrame,
    step_config_count: int,
    candidate_pair_count: int,
) -> dict[str, Any]:
    return {
        "schema": "nanoclustering_g4_8_scoped_pathway_probe_trace_summary.v1",
        "status": RUN_STATUS,
        "contract_dir": str(contract_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "candidate_pair_count": int(candidate_pair_count),
        "route_plan_row_count": int(len(route_plan)),
        "route_step_config_count": int(step_config_count),
        "trace_row_count": int(len(trace_rows)),
        "seed_route_summary_count": int(len(seed_summary)),
        "route_contract_summary_count": int(len(contract_summary)),
        "control_guard_row_count": int(len(control_guards)),
        "route_family_counts": route_plan["planned_route_family"].value_counts().to_dict(),
        "route_contract_trace_status_counts": contract_summary[
            "route_contract_trace_status"
        ].value_counts().to_dict()
        if not contract_summary.empty
        else {},
        "expected_transition_contract_count": int(
            contract_summary["route_contract_trace_status"]
            .astype(str)
            .str.contains("source_to_expected_anchor_transition")
            .sum()
        )
        if not contract_summary.empty
        else 0,
        "unknown_endpoint_trace_contract_count": int(
            contract_summary["route_contract_trace_status"]
            .astype(str)
            .eq("contains_unknown_endpoint_trace")
            .sum()
        )
        if not contract_summary.empty
        else 0,
        "intermediate_unknown_endpoint_contract_count": int(
            contract_summary["unknown_endpoint_seed_count"].gt(0).sum()
        )
        if not contract_summary.empty
        else 0,
        "intermediate_unknown_endpoint_seed_route_count": int(
            seed_summary["unknown_endpoint_step_count"].gt(0).sum()
        )
        if not seed_summary.empty
        else 0,
        "gate_status_counts": gates["gate_status"].value_counts().to_dict(),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "required_measurements": list(REQUIRED_MEASUREMENTS),
        "interpretation": (
            "The 30 scoped route-plan rows were executed as local fractional "
            "edge-weight traces. This materializes route traces and objective/"
            "endpoint/support/polish fields, but wall and method claims remain "
            "closed pending a separate wall-evidence audit."
        ),
        "recommended_next_gate": (
            "Audit the trace classifications for distinct basin-pair relation, "
            "direct-path availability, objective debt/recovery shape, polish "
            "reversion, and support incompatibility before any wall language."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    cols = [col for col in columns if col in frame.columns]
    if not cols:
        return "No columns."
    visible = frame[cols].head(int(max_rows))
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    rows: list[str] = []
    for row in visible.itertuples(index=False):
        values: list[str] = []
        for value in row:
            if isinstance(value, (dict, list, tuple, set)):
                values.append(json.dumps(_json_safe(value), sort_keys=True))
            elif pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value).replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    contract_summary: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 Scoped Pathway-Probe Trace",
        "",
        f"- status: `{summary['status']}`",
        f"- candidate_pair_count: {summary['candidate_pair_count']}",
        f"- route_plan_row_count: {summary['route_plan_row_count']}",
        f"- route_step_config_count: {summary['route_step_config_count']}",
        f"- trace_row_count: {summary['trace_row_count']}",
        f"- control_guard_row_count: {summary['control_guard_row_count']}",
        f"- route_family_counts: {summary['route_family_counts']}",
        f"- route_contract_trace_status_counts: {summary['route_contract_trace_status_counts']}",
        f"- intermediate_unknown_endpoint_contract_count: {summary['intermediate_unknown_endpoint_contract_count']}",
        f"- intermediate_unknown_endpoint_seed_route_count: {summary['intermediate_unknown_endpoint_seed_route_count']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Route Contract Summary",
        "",
        _markdown_table(
            contract_summary.sort_values(
                ["local_pair_id", "start_condition", "planned_route_family"],
                kind="mergesort",
            ),
            [
                "local_pair_id",
                "start_condition",
                "planned_route_family",
                "seed_count",
                "expected_transition_seed_count",
                "no_transition_seed_count",
                "unknown_endpoint_seed_count",
                "route_contract_trace_status",
                "max_objective_debt_from_start",
                "max_objective_recovery_from_min",
            ],
            max_rows=40,
        ),
        "",
        "## Gate Matrix",
        "",
        _markdown_table(
            gates,
            ["gate_id", "gate_status", "observed", "minimum_or_rule", "question"],
            max_rows=20,
        ),
        "",
        "## Boundary",
        "",
        (
            "This is the first execution of the scoped Stage 2A route contract. "
            "It materializes route traces, not wall evidence. Wall language stays "
            "closed until a separate audit accepts distinct basin-pair relations, "
            "direct-path availability, objective debt/recovery behavior, polish "
            "reversion, and support incompatibility."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract_dir = Path(args.contract_dir)
    local_ablation_dir = Path(args.local_ablation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    route_plan = _read_csv(contract_dir / CONTRACT_ROUTE_PLAN_ROWS_CSV)
    contract_gates = _read_csv(contract_dir / CONTRACT_GATE_MATRIX_CSV)
    control_guards = _read_csv(contract_dir / CONTRACT_CONTROL_GUARD_ROWS_CSV)
    trace_rows, step_config_count, candidate_pair_count = _trace_rows(
        route_plan=route_plan,
        contract_dir=contract_dir,
        local_ablation_dir=local_ablation_dir,
        gamma=float(args.gamma),
        seeds=int(args.seeds),
        n_iterations=int(args.n_iterations),
        edge_chunk_size=int(args.edge_chunk_size),
    )
    seed_summary = _seed_route_summary(trace_rows)
    contract_summary = _route_contract_summary(seed_summary)
    gates = _gate_matrix(
        contract_gates=contract_gates,
        route_plan=route_plan,
        control_guards=control_guards,
        trace_rows=trace_rows,
        step_config_count=step_config_count,
        contract_summary=contract_summary,
    )
    summary = _summary(
        contract_dir=contract_dir,
        local_ablation_dir=local_ablation_dir,
        output_dir=output_dir,
        route_plan=route_plan,
        control_guards=control_guards,
        trace_rows=trace_rows,
        seed_summary=seed_summary,
        contract_summary=contract_summary,
        gates=gates,
        step_config_count=step_config_count,
        candidate_pair_count=candidate_pair_count,
    )
    _write_csv(trace_rows, output_dir / TRACE_ROWS_CSV)
    _write_csv(seed_summary, output_dir / SEED_ROUTE_SUMMARY_CSV)
    _write_csv(contract_summary, output_dir / ROUTE_CONTRACT_SUMMARY_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_scoped_pathway_probe_trace_config.v1",
        "contract_dir": str(contract_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "gamma": float(args.gamma),
        "seeds": int(args.seeds),
        "n_iterations": int(args.n_iterations),
        "edge_chunk_size": int(args.edge_chunk_size),
        "route_schedules": SCHEDULES,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_dir=output_dir, summary=summary, contract_summary=contract_summary, gates=gates)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT_DIR)
    parser.add_argument("--local-ablation-dir", type=Path, default=DEFAULT_LOCAL_ABLATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gamma", type=float, default=1.0e-5)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--n-iterations", type=int, default=2)
    parser.add_argument("--edge-chunk-size", type=int, default=5_000_000)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
