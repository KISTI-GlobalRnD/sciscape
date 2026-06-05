#!/usr/bin/env python3
"""Execute the first-pass local_pair_014 wall-localization contract.

This runner consumes
``design_leiden_basin_nanoclustering_g4_8_first_pass_014_wall_localization_contract.py``.
It executes exactly the 16 predeclared route rows and 192 fraction steps:
``local_pair_014`` descent/ascent scans and retained ``local_pair_005``
boundary guards.

The readout is endpoint-object localization, not single-anchor target
reconstruction. It records fraction-level object states, transition intervals,
and G4.9A-style boundary classes. It does not promote wall generality, evaluate
quality/cost value, replay full NanoClustering, or claim method success.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from design_leiden_basin_nanoclustering_g4_8_first_pass_014_wall_localization_contract import (
    BOUNDARY_VOCAB_ROWS_CSV as CONTRACT_BOUNDARY_VOCAB_ROWS_CSV,
    DEFAULT_OUTPUT_DIR as DEFAULT_CONTRACT_DIR,
    FRACTION_STEP_ROWS_CSV as CONTRACT_FRACTION_STEP_ROWS_CSV,
    GATE_MATRIX_CSV as CONTRACT_GATE_MATRIX_CSV,
    PAIR_ROWS_CSV as CONTRACT_PAIR_ROWS_CSV,
    READOUT_RULE_ROWS_CSV as CONTRACT_READOUT_RULE_ROWS_CSV,
    ROUTE_PLAN_ROWS_CSV as CONTRACT_ROUTE_PLAN_ROWS_CSV,
)
from run_leiden_basin_nanoclustering_g4_8_scoped_pathway_probe_trace import (
    ANCHOR_VARIANT_TO_ASSIGNMENT,
    GRAPH_VARIANT_ORDER,
    _anchor_lookup,
    _coassignment_distance,
    _edge_weight_parts,
    _groups_to_membership,
    _parse_node_ids,
    _scaled_local_edges,
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
    / "leiden_basin_nanoclustering_g4_8_first_pass_014_wall_localization_trace_gamma1e5_20260605"
)

ROUTE_EXECUTION_PLAN_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_wall_localization_route_execution_plan_rows.csv"
)
TRACE_ROWS_CSV = "nanoclustering_g4_8_first_pass_014_wall_localization_trace_rows.csv"
ROUTE_SCAN_RESULT_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_wall_localization_route_scan_result_rows.csv"
)
SEED_LOCALIZATION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_wall_localization_seed_rows.csv"
)
PAIR_LOCALIZATION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_wall_localization_pair_rows.csv"
)
BOUNDARY_GUARD_RESULT_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_wall_localization_boundary_guard_rows.csv"
)
GATE_MATRIX_CSV = "nanoclustering_g4_8_first_pass_014_wall_localization_trace_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_014_wall_localization_trace_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_014_wall_localization_trace_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_014_wall_localization_trace_report.md"

POSITIVE_PAIR_ID = "local_pair_014"
BOUNDARY_PAIR_ID = "local_pair_005"

RUN_STATUS = "executed_nanoclustering_g4_8_first_pass_014_wall_localization_trace"
ROUTE_EXECUTION_STATUS = "executed_first_pass_014_wall_localization_fraction_scan"
WALL_PROMOTION_STATUS = "not_promoted_wall_localization_trace_only"
METHOD_STATUS = "local_wall_localization_trace_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass local_pair_014 wall-localization trace only; "
    "executes the 16 predeclared localization route rows and reads fraction-level "
    "endpoint-object transitions. It does not promote wall generality, evaluate "
    "quality/cost value, replay full NanoClustering, or claim method success."
)

SOURCE_OBJECTS = {"source_endpoint_object", "source_like_endpoint_object"}
TARGET_OBJECT = "exclusive_target_endpoint_object"
BOUNDARY_TARGET_OBJECT = "boundary_target_endpoint_object_not_positive"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _sequence(values: pd.Series | list[Any]) -> str:
    if isinstance(values, pd.Series):
        items = values.astype(str).tolist()
    else:
        items = [str(value) for value in values]
    return " -> ".join(items)


def _source_like(value: Any) -> bool:
    return str(value) in SOURCE_OBJECTS


def _target_like(value: Any, *, positive_only: bool) -> bool:
    if str(value) == TARGET_OBJECT:
        return True
    return bool(not positive_only and str(value) == BOUNDARY_TARGET_OBJECT)


def _endpoint_object_assignment(row: pd.Series) -> str:
    endpoint = str(row["endpoint_assignment_by_step"])
    pair_id = str(row["local_pair_id"])
    if endpoint == "original_source_anchor":
        return "source_endpoint_object"
    if endpoint == "drop_bridge_target_anchor":
        if pair_id == POSITIVE_PAIR_ID:
            return TARGET_OBJECT
        return BOUNDARY_TARGET_OBJECT
    if endpoint == "drop_direct_guard_anchor":
        return "direct_drop_guard_endpoint_object"
    if endpoint == "drop_both_guard_anchor":
        return "drop_both_guard_endpoint_object"
    if endpoint == "unknown_new_endpoint":
        return "unknown_endpoint_object"
    if endpoint.startswith("ambiguous_anchor_match"):
        anchor_names = {
            value
            for value in endpoint.split(":", 1)[-1].split(";")
            if value
        }
        if anchor_names and anchor_names.issubset(
            {"original_source_anchor", "drop_direct_guard_anchor"}
        ):
            return "source_like_endpoint_object"
        if anchor_names and anchor_names.issubset({"drop_bridge_target_anchor"}):
            if pair_id == POSITIVE_PAIR_ID:
                return TARGET_OBJECT
            return BOUNDARY_TARGET_OBJECT
        return "ambiguous_endpoint_object"
    return "other_known_endpoint_object"


def _anchor_match_data(
    *,
    anchors: dict[str, dict[str, Any]],
    result_signature_id: str,
    result_membership: list[int],
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
        "matches_drop_bridge_target_anchor": "drop_bridge_target_anchor" in matches,
        "matches_drop_direct_guard_anchor": "drop_direct_guard_anchor" in matches,
        "matches_drop_both_guard_anchor": "drop_both_guard_anchor" in matches,
        "support_distance_min_known_anchor": min_distance,
        "support_incompatibility_check": bool(not matches and (min_distance is None or min_distance > 0.0)),
        **distances,
    }


def _execution_plan(route_plan: pd.DataFrame) -> pd.DataFrame:
    rows = route_plan.copy()
    rows["validation_unit_id"] = rows["route_contract_id"].astype(str)
    rows["runner_support_status_after_execution"] = "implemented_in_wall_localization_runner"
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.reset_index(drop=True)


def _trace_rows(
    *,
    execution_plan: pd.DataFrame,
    fraction_steps: pd.DataFrame,
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
    candidate_pair_ids = set(execution_plan["local_pair_id"].astype(str))
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
    anchors_by_key = _anchor_lookup(seed_runs, execution_plan)
    metadata_cols = [
        "route_contract_id",
        "validation_unit_id",
        "contract_pair_role",
        "counts_as_positive_if_accepted",
        "runner_support_status_after_execution",
    ]
    step_rows = fraction_steps.merge(
        execution_plan[metadata_cols],
        on="route_contract_id",
        how="left",
        validate="many_to_one",
    )
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
            )
            support_distance = anchor_data["support_distance_min_known_anchor"]
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
                    "contract_pair_role": str(step["contract_pair_role"]),
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
    rows["endpoint_object_assignment_by_step"] = rows.apply(
        _endpoint_object_assignment,
        axis=1,
    )
    rows["direct_edge_retained_by_step"] = rows["active_direct_edge_weight"].astype(float).gt(0.0)
    rows["bridge_support_suppressed_by_step"] = (
        rows["active_pair_bridge_edge_weight_sum"].astype(float).eq(0.0)
    )
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


def _first_step(ordered: pd.DataFrame, mask: pd.Series) -> int | None:
    matches = ordered.loc[mask.astype(bool), "step_index"]
    if matches.empty:
        return None
    return int(matches.iloc[0])


def _first_fraction(ordered: pd.DataFrame, mask: pd.Series) -> float | None:
    matches = ordered.loc[mask.astype(bool), "bridge_edge_weight_fraction"]
    if matches.empty:
        return None
    return float(matches.iloc[0])


def _tail_all(ordered: pd.DataFrame, *, start_step: int | None, mask: pd.Series) -> bool:
    if start_step is None:
        return False
    tail = ordered.loc[ordered["step_index"].astype(int).ge(int(start_step))]
    if tail.empty:
        return False
    aligned = mask.loc[tail.index]
    return bool(aligned.astype(bool).all())


def _route_scan_results(trace_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "route_contract_id",
        "validation_unit_id",
        "local_pair_id",
        "branch",
        "start_condition",
        "planned_route_family",
        "route_family_role",
        "contract_pair_role",
        "seed",
    ]
    for keys, group in trace_rows.groupby(group_cols, sort=False):
        key_data = dict(zip(group_cols, keys, strict=True))
        ordered = group.sort_values("step_index", kind="mergesort")
        endpoint_objects = ordered["endpoint_object_assignment_by_step"].astype(str)
        endpoints = ordered["endpoint_assignment_by_step"].astype(str)
        fractions = ordered["bridge_edge_weight_fraction"].astype(float)
        is_positive_pair = str(key_data["local_pair_id"]) == POSITIVE_PAIR_ID
        positive_target_mask = endpoint_objects.eq(TARGET_OBJECT)
        target_mask = positive_target_mask | endpoint_objects.eq(BOUNDARY_TARGET_OBJECT)
        source_mask = endpoint_objects.map(_source_like)
        unknown_step_count = int(endpoint_objects.eq("unknown_endpoint_object").sum())
        ambiguous_step_count = int(endpoint_objects.eq("ambiguous_endpoint_object").sum())
        raw_anchor_ambiguous_step_count = int(
            endpoints.str.startswith("ambiguous_anchor_match").sum()
        )
        support_incompatibility_step_count = int(
            ordered["support_incompatibility_check"].map(_as_bool).sum()
        )
        interpretable = unknown_step_count == 0 and ambiguous_step_count == 0
        first_positive_target_step = _first_step(ordered, positive_target_mask)
        first_any_target_step = _first_step(ordered, target_mask)
        first_source_step = _first_step(ordered, source_mask)
        first_positive_target_fraction = _first_fraction(ordered, positive_target_mask)
        first_any_target_fraction = _first_fraction(ordered, target_mask)
        first_source_fraction = _first_fraction(ordered, source_mask)
        positive_target_tail = _tail_all(
            ordered,
            start_step=first_positive_target_step,
            mask=positive_target_mask,
        )
        any_target_tail = _tail_all(
            ordered,
            start_step=first_any_target_step,
            mask=target_mask,
        )
        source_tail = _tail_all(ordered, start_step=first_source_step, mask=source_mask)
        max_debt = float(ordered["objective_debt_from_start"].astype(float).max())
        max_recovery = float(ordered["objective_recovery_from_min"].astype(float).max())
        min_objective_idx = ordered["objective_value_by_step"].astype(float).idxmin()
        min_objective_step = int(ordered.loc[min_objective_idx, "step_index"])
        recovery_after_min_rows = ordered[
            ordered["step_index"].astype(int).gt(min_objective_step)
            & ordered["objective_recovery_from_min"].astype(float).gt(0.0)
        ]
        family = str(key_data["planned_route_family"])
        is_descent = "descent" in family
        is_ascent = "ascent" in family
        if is_descent:
            route_transition_pass = bool(
                _source_like(endpoint_objects.iloc[0])
                and first_positive_target_step is not None
                and positive_target_tail
                and interpretable
                and support_incompatibility_step_count == 0
            )
        elif is_ascent:
            route_transition_pass = bool(
                endpoint_objects.iloc[0] == TARGET_OBJECT
                and first_source_step is not None
                and first_source_step > 1
                and source_tail
                and interpretable
                and support_incompatibility_step_count == 0
            )
        else:
            route_transition_pass = False
        if is_positive_pair and route_transition_pass:
            route_scan_status = "positive_transition_scan_pass"
        elif is_positive_pair and first_positive_target_step is None:
            route_scan_status = "target_absent_or_source_locked_scan"
        elif is_positive_pair and not _source_like(endpoint_objects.iloc[0]) and is_descent:
            route_scan_status = "target_saturated_scan"
        elif is_positive_pair:
            route_scan_status = "nonrobust_or_mixed_transition_scan"
        elif route_transition_pass:
            route_scan_status = "boundary_positive_pattern_observed_not_counted"
        else:
            route_scan_status = "boundary_scan_closed"
        rows.append(
            {
                **key_data,
                "route_step_count": int(len(ordered)),
                "bridge_fraction_sequence": ";".join(f"{float(value):.3f}" for value in fractions),
                "endpoint_assignment_sequence": _sequence(endpoints),
                "endpoint_object_assignment_sequence": _sequence(endpoint_objects),
                "first_object": str(endpoint_objects.iloc[0]),
                "final_object": str(endpoint_objects.iloc[-1]),
                "source_like_step_count": int(source_mask.sum()),
                "positive_target_step_count": int(positive_target_mask.sum()),
                "any_target_step_count": int(target_mask.sum()),
                "first_positive_target_step": first_positive_target_step,
                "first_positive_target_fraction": first_positive_target_fraction,
                "first_any_target_step": first_any_target_step,
                "first_any_target_fraction": first_any_target_fraction,
                "first_source_step": first_source_step,
                "first_source_fraction": first_source_fraction,
                "positive_target_tail_after_first": bool(positive_target_tail),
                "any_target_tail_after_first": bool(any_target_tail),
                "source_tail_after_first": bool(source_tail),
                "endpoint_objects_interpretable_all_steps": bool(interpretable),
                "unknown_step_count": unknown_step_count,
                "ambiguous_step_count": ambiguous_step_count,
                "raw_anchor_ambiguous_step_count": raw_anchor_ambiguous_step_count,
                "support_incompatibility_step_count": support_incompatibility_step_count,
                "min_objective_step": min_objective_step,
                "max_objective_debt_from_start": max_debt,
                "max_objective_recovery_from_min": max_recovery,
                "accepted_recovery_after_min": bool(max_debt > 0.0 and not recovery_after_min_rows.empty),
                "route_transition_pass": bool(route_transition_pass),
                "route_scan_status": route_scan_status,
                "wall_generality_claim_allowed_after_scan": False,
                "method_claim_allowed_after_scan": False,
                "quality_cost_claim_allowed_after_scan": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _seed_localization_rows(route_results: pd.DataFrame) -> pd.DataFrame:
    descent = route_results[
        route_results["planned_route_family"].astype(str).str.contains("descent")
    ].copy()
    ascent = route_results[
        route_results["planned_route_family"].astype(str).str.contains("ascent")
    ].copy()
    key_cols = [
        "local_pair_id",
        "branch",
        "start_condition",
        "contract_pair_role",
        "seed",
    ]
    merged = descent.merge(
        ascent,
        on=key_cols,
        suffixes=("_descent", "_ascent"),
        how="outer",
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for row in merged.sort_values(["local_pair_id", "start_condition", "seed"], kind="mergesort").itertuples(index=False):
        data = row._asdict()
        pair_id = str(data.get("local_pair_id", ""))
        is_positive = pair_id == POSITIVE_PAIR_ID
        descent_pass = _as_bool(data.get("route_transition_pass_descent", False))
        ascent_pass = _as_bool(data.get("route_transition_pass_ascent", False))
        target_seen = bool(
            int(data.get("positive_target_step_count_descent", 0)) > 0
            or int(data.get("positive_target_step_count_ascent", 0)) > 0
        )
        any_target_seen = bool(
            int(data.get("any_target_step_count_descent", 0)) > 0
            or int(data.get("any_target_step_count_ascent", 0)) > 0
        )
        support_bad = bool(
            int(data.get("support_incompatibility_step_count_descent", 0)) > 0
            or int(data.get("support_incompatibility_step_count_ascent", 0)) > 0
        )
        unknown_bad = bool(
            int(data.get("unknown_step_count_descent", 0)) > 0
            or int(data.get("unknown_step_count_ascent", 0)) > 0
            or int(data.get("ambiguous_step_count_descent", 0)) > 0
            or int(data.get("ambiguous_step_count_ascent", 0)) > 0
        )
        wall_ready = bool(is_positive and descent_pass and ascent_pass)
        boundary_leak = bool(not is_positive and descent_pass and ascent_pass)
        if wall_ready:
            code = "W"
            status = "full_primitive_wall_seed_localized"
        elif is_positive and not target_seen:
            code = "N"
            status = "target_absent_or_source_locked_seed"
        elif is_positive and str(data.get("first_object_descent", "")) == TARGET_OBJECT:
            code = "T"
            status = "target_saturated_seed"
        elif is_positive and (support_bad or unknown_bad or target_seen):
            code = "P"
            status = "nonrobust_or_mixed_boundary_seed"
        elif boundary_leak:
            code = "P"
            status = "boundary_positive_pattern_observed_not_counted"
        elif not any_target_seen:
            code = "N"
            status = "boundary_target_absent_or_source_locked_closed"
        else:
            code = "P"
            status = "boundary_nonrobust_or_mixed_closed"
        descent_fraction = data.get("first_positive_target_fraction_descent", None)
        ascent_fraction = data.get("first_source_fraction_ascent", None)
        interval_width = None
        if descent_fraction is not None and not pd.isna(descent_fraction) and ascent_fraction is not None and not pd.isna(ascent_fraction):
            interval_width = abs(float(ascent_fraction) - float(descent_fraction))
        rows.append(
            {
                "localization_seed_id": (
                    f"{pair_id}__{data.get('start_condition', '')}"
                    f"__seed{int(data.get('seed', -1)):02d}"
                ),
                "local_pair_id": pair_id,
                "branch": str(data.get("branch", "")),
                "start_condition": str(data.get("start_condition", "")),
                "contract_pair_role": str(data.get("contract_pair_role", "")),
                "seed": int(data.get("seed", -1)),
                "descent_route_contract_id": str(data.get("route_contract_id_descent", "")),
                "ascent_route_contract_id": str(data.get("route_contract_id_ascent", "")),
                "descent_transition_pass": bool(descent_pass),
                "ascent_transition_pass": bool(ascent_pass),
                "wall_seed_localized": bool(wall_ready),
                "boundary_positive_pattern_observed": bool(boundary_leak),
                "g4_9a_vocab_code": code,
                "localization_seed_status": status,
                "descent_first_positive_target_fraction": (
                    None if pd.isna(descent_fraction) else float(descent_fraction)
                ),
                "ascent_first_source_recovery_fraction": (
                    None if pd.isna(ascent_fraction) else float(ascent_fraction)
                ),
                "transition_interval_width": interval_width,
                "descent_endpoint_object_sequence": str(
                    data.get("endpoint_object_assignment_sequence_descent", "")
                ),
                "ascent_endpoint_object_sequence": str(
                    data.get("endpoint_object_assignment_sequence_ascent", "")
                ),
                "descent_max_objective_debt_from_start": float(
                    data.get("max_objective_debt_from_start_descent", 0.0)
                ),
                "ascent_max_objective_recovery_from_min": float(
                    data.get("max_objective_recovery_from_min_ascent", 0.0)
                ),
                "support_incompatibility_observed": bool(support_bad),
                "unknown_or_ambiguous_object_observed": bool(unknown_bad),
                "wall_generality_claim_allowed_after_scan": False,
                "method_claim_allowed_after_scan": False,
                "quality_cost_claim_allowed_after_scan": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _pair_localization_rows(pair_rows: pd.DataFrame, seed_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair in pair_rows.itertuples(index=False):
        pair_id = str(pair.local_pair_id)
        group = seed_rows[seed_rows["local_pair_id"].astype(str).eq(pair_id)]
        seed_count = int(len(group))
        ready_count = int(group["wall_seed_localized"].map(_as_bool).sum()) if not group.empty else 0
        boundary_leaks = int(group["boundary_positive_pattern_observed"].map(_as_bool).sum()) if not group.empty else 0
        if pair_id == POSITIVE_PAIR_ID and seed_count > 0 and ready_count == seed_count:
            pair_status = "W_full_localized_primitive_wall"
            pair_code = "W"
        elif pair_id == POSITIVE_PAIR_ID and ready_count > 0:
            pair_status = "w_partial_or_fragile_localized_wall"
            pair_code = "w"
        elif pair_id == POSITIVE_PAIR_ID and seed_count > 0:
            dominant = str(group["g4_9a_vocab_code"].mode().iloc[0])
            pair_status = f"{dominant}_positive_wall_localization_closed"
            pair_code = dominant
        elif boundary_leaks:
            pair_status = "boundary_positive_pattern_observed_not_counted"
            pair_code = "P"
        else:
            pair_status = "boundary_guard_closed"
            pair_code = "closed"
        rows.append(
            {
                "local_pair_id": pair_id,
                "contract_pair_role": str(pair.contract_pair_role),
                "seed_localization_row_count": seed_count,
                "wall_seed_localized_count": ready_count,
                "boundary_positive_pattern_seed_count": boundary_leaks,
                "g4_9a_vocab_code_counts": _count_dict(group["g4_9a_vocab_code"]) if not group.empty else {},
                "localization_seed_status_counts": _count_dict(group["localization_seed_status"]) if not group.empty else {},
                "transition_interval_width_min": (
                    float(group["transition_interval_width"].dropna().min())
                    if not group.empty and not group["transition_interval_width"].dropna().empty
                    else None
                ),
                "transition_interval_width_median": (
                    float(group["transition_interval_width"].dropna().median())
                    if not group.empty and not group["transition_interval_width"].dropna().empty
                    else None
                ),
                "transition_interval_width_max": (
                    float(group["transition_interval_width"].dropna().max())
                    if not group.empty and not group["transition_interval_width"].dropna().empty
                    else None
                ),
                "pair_localization_code": pair_code,
                "pair_localization_status": pair_status,
                "wall_generality_claim_allowed_after_scan": False,
                "method_claim_allowed_after_scan": False,
                "quality_cost_claim_allowed_after_scan": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _boundary_guard_rows(seed_rows: pd.DataFrame) -> pd.DataFrame:
    boundary = seed_rows[seed_rows["local_pair_id"].astype(str).eq(BOUNDARY_PAIR_ID)].copy()
    if boundary.empty:
        return pd.DataFrame()
    rows = (
        boundary.groupby(["local_pair_id", "branch", "start_condition"], sort=False)
        .agg(
            seed_count=("seed", "nunique"),
            boundary_positive_pattern_seed_count=("boundary_positive_pattern_observed", "sum"),
            status_counts=("localization_seed_status", _count_dict),
        )
        .reset_index()
    )
    rows["boundary_guard_closed"] = rows["boundary_positive_pattern_seed_count"].astype(int).eq(0)
    rows["boundary_guard_status"] = rows["boundary_guard_closed"].map(
        lambda value: "closed" if bool(value) else "positive_pattern_observed"
    )
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


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
    trace_rows: pd.DataFrame,
    route_results: pd.DataFrame,
    seed_rows: pd.DataFrame,
    pair_results: pd.DataFrame,
    boundary_results: pd.DataFrame,
    expected_trace_rows: int,
) -> pd.DataFrame:
    positive_seed_rows = seed_rows[seed_rows["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)]
    positive_pair = pair_results[pair_results["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)]
    boundary_pair = pair_results[pair_results["local_pair_id"].astype(str).eq(BOUNDARY_PAIR_ID)]
    rows = [
        _gate_row(
            "G1_contract_gates_pass",
            "Did the upstream wall-localization contract gates pass?",
            _count_dict(contract_gates["gate_status"]),
            "all contract gates pass",
            bool(contract_gates["gate_status"].astype(str).eq("pass").all()),
        ),
        _gate_row(
            "G2_trace_rows_complete",
            "Did every route/fraction/seed row execute?",
            f"trace_rows={len(trace_rows)} expected={expected_trace_rows}",
            "16 route rows * 12 fraction steps * 8 seeds",
            len(trace_rows) == expected_trace_rows,
        ),
        _gate_row(
            "G3_no_single_final_anchor_required",
            "Was localization read without a single expected final anchor?",
            sorted(trace_rows["expected_final_anchor_variant"].astype(str).unique().tolist()),
            "localization_scan_no_single_expected_anchor only",
            set(trace_rows["expected_final_anchor_variant"].astype(str))
            == {"localization_scan_no_single_expected_anchor"},
        ),
        _gate_row(
            "G4_seed_transition_rows_materialized",
            "Were paired descent/ascent localization rows materialized?",
            {
                "seed_rows": int(len(seed_rows)),
                "positive_seed_rows": int(len(positive_seed_rows)),
            },
            "64 seed rows total, 32 positive rows",
            len(seed_rows) == 64 and len(positive_seed_rows) == 32,
        ),
        _gate_row(
            "G5_boundary_guard_no_positive_pattern",
            "Did the 005 boundary guard avoid positive W-like patterns?",
            boundary_pair[
                ["local_pair_id", "boundary_positive_pattern_seed_count", "pair_localization_status"]
            ].to_dict("records"),
            "boundary positive pattern count is zero",
            not boundary_pair.empty
            and int(boundary_pair.iloc[0]["boundary_positive_pattern_seed_count"]) == 0
            and bool(boundary_results["boundary_guard_closed"].map(_as_bool).all()),
        ),
        _gate_row(
            "G6_pair_localization_classified",
            "Was the positive pair classified with G4.9A vocabulary?",
            positive_pair[
                ["local_pair_id", "pair_localization_code", "pair_localization_status"]
            ].to_dict("records"),
            "positive pair has a localization code",
            not positive_pair.empty
            and str(positive_pair.iloc[0]["pair_localization_code"]) in {"W", "w", "T", "N", "P"},
        ),
        _gate_row(
            "G7_claims_closed",
            "Are method, quality/cost, and wall generality claims closed?",
            CLAIM_BOUNDARY,
            "all claim flags false",
            bool(route_results["wall_generality_claim_allowed_after_scan"].eq(False).all())
            and bool(pair_results["wall_generality_claim_allowed_after_scan"].eq(False).all()),
        ),
    ]
    gates = pd.DataFrame(rows)
    gates["run_status"] = RUN_STATUS
    gates["route_execution_status"] = ROUTE_EXECUTION_STATUS
    gates["wall_promotion_status"] = WALL_PROMOTION_STATUS
    gates["method_status"] = METHOD_STATUS
    gates["claim_boundary"] = CLAIM_BOUNDARY
    return gates


def _summary(
    *,
    contract_dir: Path,
    local_ablation_dir: Path,
    output_dir: Path,
    trace_rows: pd.DataFrame,
    route_results: pd.DataFrame,
    seed_rows: pd.DataFrame,
    pair_results: pd.DataFrame,
    boundary_results: pd.DataFrame,
    gates: pd.DataFrame,
    step_count: int,
    pair_count: int,
) -> dict[str, Any]:
    positive_pair = pair_results[pair_results["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)]
    boundary_pair = pair_results[pair_results["local_pair_id"].astype(str).eq(BOUNDARY_PAIR_ID)]
    return {
        "schema": "nanoclustering_g4_8_first_pass_014_wall_localization_trace_summary.v1",
        "status": RUN_STATUS,
        "contract_dir": str(contract_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "local_pair_count": int(pair_count),
        "route_step_config_count": int(step_count),
        "trace_row_count": int(len(trace_rows)),
        "route_scan_result_row_count": int(len(route_results)),
        "seed_localization_row_count": int(len(seed_rows)),
        "pair_localization_row_count": int(len(pair_results)),
        "boundary_guard_row_count": int(len(boundary_results)),
        "positive_pair_localization_status": (
            str(positive_pair.iloc[0]["pair_localization_status"])
            if not positive_pair.empty
            else ""
        ),
        "positive_pair_localization_code": (
            str(positive_pair.iloc[0]["pair_localization_code"])
            if not positive_pair.empty
            else ""
        ),
        "positive_wall_seed_localized_count": (
            int(positive_pair.iloc[0]["wall_seed_localized_count"])
            if not positive_pair.empty
            else 0
        ),
        "boundary_positive_pattern_seed_count": (
            int(boundary_pair.iloc[0]["boundary_positive_pattern_seed_count"])
            if not boundary_pair.empty
            else 0
        ),
        "seed_vocab_code_counts": _count_dict(seed_rows["g4_9a_vocab_code"]),
        "localization_seed_status_counts": _count_dict(seed_rows["localization_seed_status"]),
        "route_scan_status_counts": _count_dict(route_results["route_scan_status"]),
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"), "gate_id"
        ].tolist(),
        "interpretation": (
            "The runner executed the fixed wall-localization scan and classified "
            "the 014/005 seed-start units with the G4.9A boundary vocabulary. "
            "The result is wall-localization readout only; wall generality, "
            "method, quality/cost, and full replay claims remain closed."
        ),
        "recommended_next_gate": (
            "Audit the positive transition intervals and boundary object "
            "patterns before deciding whether a narrower wall-location statement "
            "is warranted."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 60) -> str:
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
    seed_rows: pd.DataFrame,
    pair_results: pd.DataFrame,
    boundary_results: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass 014 Wall-Localization Trace",
        "",
        f"- status: `{summary['status']}`",
        f"- trace_row_count: {summary['trace_row_count']}",
        f"- route_scan_result_row_count: {summary['route_scan_result_row_count']}",
        f"- seed_localization_row_count: {summary['seed_localization_row_count']}",
        f"- positive_pair_localization_status: `{summary['positive_pair_localization_status']}`",
        f"- positive_pair_localization_code: `{summary['positive_pair_localization_code']}`",
        f"- positive_wall_seed_localized_count: {summary['positive_wall_seed_localized_count']}",
        f"- boundary_positive_pattern_seed_count: {summary['boundary_positive_pattern_seed_count']}",
        f"- seed_vocab_code_counts: {summary['seed_vocab_code_counts']}",
        f"- localization_seed_status_counts: {summary['localization_seed_status_counts']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Pair Localization",
        "",
        _markdown_table(
            pair_results,
            [
                "local_pair_id",
                "contract_pair_role",
                "seed_localization_row_count",
                "wall_seed_localized_count",
                "boundary_positive_pattern_seed_count",
                "g4_9a_vocab_code_counts",
                "pair_localization_code",
                "pair_localization_status",
                "transition_interval_width_min",
                "transition_interval_width_median",
                "transition_interval_width_max",
            ],
            max_rows=10,
        ),
        "",
        "## Seed Localization",
        "",
        _markdown_table(
            seed_rows,
            [
                "local_pair_id",
                "start_condition",
                "seed",
                "g4_9a_vocab_code",
                "localization_seed_status",
                "descent_first_positive_target_fraction",
                "ascent_first_source_recovery_fraction",
                "transition_interval_width",
            ],
            max_rows=80,
        ),
        "",
        "## Boundary Guards",
        "",
        _markdown_table(
            boundary_results,
            [
                "local_pair_id",
                "start_condition",
                "seed_count",
                "boundary_positive_pattern_seed_count",
                "boundary_guard_status",
            ],
            max_rows=20,
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
            "This trace executes a local diagnostic scan. It does not promote "
            "wall generality, method success, full replay, or quality/cost value."
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
    fraction_steps = _read_csv(contract_dir / CONTRACT_FRACTION_STEP_ROWS_CSV)
    pair_rows = _read_csv(contract_dir / CONTRACT_PAIR_ROWS_CSV)
    boundary_vocab = _read_csv(contract_dir / CONTRACT_BOUNDARY_VOCAB_ROWS_CSV)
    readout_rules = _read_csv(contract_dir / CONTRACT_READOUT_RULE_ROWS_CSV)
    contract_gates = _read_csv(contract_dir / CONTRACT_GATE_MATRIX_CSV)
    execution_plan = _execution_plan(route_plan)
    trace_rows, step_count, pair_count = _trace_rows(
        execution_plan=execution_plan,
        fraction_steps=fraction_steps,
        contract_dir=contract_dir,
        local_ablation_dir=local_ablation_dir,
        gamma=float(args.gamma),
        seeds=int(args.seeds),
        n_iterations=int(args.n_iterations),
        edge_chunk_size=int(args.edge_chunk_size),
    )
    route_results = _route_scan_results(trace_rows)
    seed_rows = _seed_localization_rows(route_results)
    pair_results = _pair_localization_rows(pair_rows, seed_rows)
    boundary_results = _boundary_guard_rows(seed_rows)
    expected_trace_rows = int(len(fraction_steps) * int(args.seeds))
    gates = _gate_matrix(
        contract_gates=contract_gates,
        trace_rows=trace_rows,
        route_results=route_results,
        seed_rows=seed_rows,
        pair_results=pair_results,
        boundary_results=boundary_results,
        expected_trace_rows=expected_trace_rows,
    )
    summary = _summary(
        contract_dir=contract_dir,
        local_ablation_dir=local_ablation_dir,
        output_dir=output_dir,
        trace_rows=trace_rows,
        route_results=route_results,
        seed_rows=seed_rows,
        pair_results=pair_results,
        boundary_results=boundary_results,
        gates=gates,
        step_count=step_count,
        pair_count=pair_count,
    )

    _write_csv(execution_plan, output_dir / ROUTE_EXECUTION_PLAN_ROWS_CSV)
    _write_csv(trace_rows, output_dir / TRACE_ROWS_CSV)
    _write_csv(route_results, output_dir / ROUTE_SCAN_RESULT_ROWS_CSV)
    _write_csv(seed_rows, output_dir / SEED_LOCALIZATION_ROWS_CSV)
    _write_csv(pair_results, output_dir / PAIR_LOCALIZATION_ROWS_CSV)
    _write_csv(boundary_results, output_dir / BOUNDARY_GUARD_RESULT_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_014_wall_localization_trace_config.v1",
        "contract_dir": str(contract_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "gamma": float(args.gamma),
        "seeds": int(args.seeds),
        "n_iterations": int(args.n_iterations),
        "edge_chunk_size": int(args.edge_chunk_size),
        "boundary_vocab_codes": sorted(boundary_vocab["vocab_code"].astype(str).unique().tolist()),
        "readout_rule_count": int(len(readout_rules)),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        seed_rows=seed_rows,
        pair_results=pair_results,
        boundary_results=boundary_results,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT_DIR)
    parser.add_argument("--local-ablation-dir", type=Path, default=DEFAULT_LOCAL_ABLATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gamma", type=float, default=1.0e-5)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--n-iterations", type=int, default=2)
    parser.add_argument("--edge-chunk-size", type=int, default=2_000_000)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
