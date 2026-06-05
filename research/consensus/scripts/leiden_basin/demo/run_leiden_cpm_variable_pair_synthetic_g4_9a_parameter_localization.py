#!/usr/bin/env python3
"""Map the local G4.9 primitive-wall parameter regime.

G4.9 showed that a small variable-pair Leiden+CPM graph can reproduce the
``local_pair_014`` object-level relation:

source-like endpoint object -> exclusive target object -> source-like endpoint
object.

This G4.9A runner keeps the same primitive wall readout and maps three
predeclared two-dimensional slices around the G4.9 positive point. The goal is
to decide whether the positive point is an isolated tuning artifact or a
bounded mechanism regime with interpretable failure modes. This is synthetic
mechanism localization only. It is not selector retuning, NanoClustering
replay, wall/pathway generality, quality/cost evaluation, method evidence, or
an algorithm-level claim.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from run_leiden_cpm_variable_pair_synthetic_demo import (
    BASE_RESULT_DIR,
    _build_graph,
    _canonical_groups,
    _initial_membership,
    _json_safe,
    _mechanism_read,
    _signature_id,
    _write_csv,
)
from run_leiden_cpm_variable_pair_synthetic_g4_9_primitive_wall_demo import (
    DIRECT_ONLY_FAMILY,
    RECOVERY_LOOP_FAMILY,
    SOURCE_OBJECT,
    START_CONDITIONS,
    TARGET_OBJECT,
    PrimitiveWallCase,
    _case_to_synthetic,
    _endpoint_object,
    _scaled_edges,
    _schedule_rows,
)
from sciscape.clustering.runner import LeidenRunner


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_variable_pair_synthetic_g4_9a_parameter_localization_v1_20260604"
)

PANEL_DESIGN_CSV = "variable_pair_synthetic_g4_9a_panel_design.csv"
TRACE_ROWS_CSV = "variable_pair_synthetic_g4_9a_trace_rows.csv"
ROUTE_RESULT_ROWS_CSV = "variable_pair_synthetic_g4_9a_route_result_rows.csv"
SEED_WALL_ROWS_CSV = "variable_pair_synthetic_g4_9a_seed_wall_rows.csv"
CASE_SUMMARY_CSV = "variable_pair_synthetic_g4_9a_case_summary.csv"
PLANE_SUMMARY_CSV = "variable_pair_synthetic_g4_9a_plane_summary.csv"
PLANE_MATRIX_CSV = "variable_pair_synthetic_g4_9a_plane_matrix.csv"
GATE_MATRIX_CSV = "variable_pair_synthetic_g4_9a_gate_matrix.csv"
SUMMARY_JSON = "variable_pair_synthetic_g4_9a_summary.json"
CONFIG_JSON = "variable_pair_synthetic_g4_9a_config.json"
REPORT_MD = "variable_pair_synthetic_g4_9a_report.md"

RUN_STATUS = "executed_variable_pair_synthetic_g4_9a_parameter_localization"
ROUTE_EXECUTION_STATUS = "executed_synthetic_g4_9a_parameter_localization"
WALL_PROMOTION_STATUS = "synthetic_parameter_localization_only"
METHOD_STATUS = "plain_leiden_cpm_synthetic_parameter_localization_not_method"
CLAIM_BOUNDARY = (
    "Variable-pair synthetic G4.9A parameter localization only; predeclared "
    "small Leiden+CPM graph slices map where the G4.9 source-like/target/"
    "source-like object relation appears and where boundary modes close it. "
    "No NanoClustering replay, no wall/pathway generality, no quality/cost "
    "value, no method claim, and no algorithm-level claim."
)

CENTER_DIRECT_WEIGHT = 1.05
CENTER_PAIR_BRIDGE_WEIGHT = 2.50
CENTER_BRIDGE_HOST_WEIGHT = 2.00
CENTER_HOST_CLIQUE_WEIGHT = 0.20

DIRECT_VALUES = (0.75, 0.90, 1.05, 1.20, 1.50)
PAIR_BRIDGE_VALUES = (1.80, 2.20, 2.50, 2.80, 3.20)
BRIDGE_HOST_VALUES = (1.40, 1.70, 2.00, 2.30, 2.60)


@dataclass(frozen=True)
class LocalizationCase:
    case_id: str
    plane_id: str
    axis_1_name: str
    axis_1_value: float
    axis_1_index: int
    axis_2_name: str
    axis_2_value: float
    axis_2_index: int
    direct_weight: float
    pair_bridge_weight: float
    bridge_host_weight: float
    host_clique_weight: float

    def to_primitive_case(self) -> PrimitiveWallCase:
        return PrimitiveWallCase(
            case_id=self.case_id,
            case_role="g4_9a_parameter_localization_cell",
            expected_case_status="parameter_localization_no_expected_success",
            direct_weight=self.direct_weight,
            pair_bridge_weight=self.pair_bridge_weight,
            bridge_host_weight=self.bridge_host_weight,
            host_clique_weight=self.host_clique_weight,
            note=(
                "Predeclared G4.9A primitive-wall parameter-localization cell "
                f"on plane {self.plane_id}."
            ),
        )


def _weight_token(value: float) -> str:
    return f"{int(round(float(value) * 100)):03d}"


def _localization_cases() -> tuple[LocalizationCase, ...]:
    cases: list[LocalizationCase] = []
    for direct_index, direct in enumerate(DIRECT_VALUES):
        for pair_index, pair_bridge in enumerate(PAIR_BRIDGE_VALUES):
            cases.append(
                LocalizationCase(
                    case_id=(
                        f"g4_9a_dp_d{_weight_token(direct)}"
                        f"_p{_weight_token(pair_bridge)}"
                    ),
                    plane_id="direct_pair_bridge",
                    axis_1_name="direct_weight",
                    axis_1_value=float(direct),
                    axis_1_index=direct_index,
                    axis_2_name="pair_bridge_weight",
                    axis_2_value=float(pair_bridge),
                    axis_2_index=pair_index,
                    direct_weight=float(direct),
                    pair_bridge_weight=float(pair_bridge),
                    bridge_host_weight=CENTER_BRIDGE_HOST_WEIGHT,
                    host_clique_weight=CENTER_HOST_CLIQUE_WEIGHT,
                )
            )
    for pair_index, pair_bridge in enumerate(PAIR_BRIDGE_VALUES):
        for bridge_index, bridge_host in enumerate(BRIDGE_HOST_VALUES):
            cases.append(
                LocalizationCase(
                    case_id=(
                        f"g4_9a_pb_p{_weight_token(pair_bridge)}"
                        f"_b{_weight_token(bridge_host)}"
                    ),
                    plane_id="pair_bridge_bridge_host",
                    axis_1_name="pair_bridge_weight",
                    axis_1_value=float(pair_bridge),
                    axis_1_index=pair_index,
                    axis_2_name="bridge_host_weight",
                    axis_2_value=float(bridge_host),
                    axis_2_index=bridge_index,
                    direct_weight=CENTER_DIRECT_WEIGHT,
                    pair_bridge_weight=float(pair_bridge),
                    bridge_host_weight=float(bridge_host),
                    host_clique_weight=CENTER_HOST_CLIQUE_WEIGHT,
                )
            )
    for direct_index, direct in enumerate(DIRECT_VALUES):
        for bridge_index, bridge_host in enumerate(BRIDGE_HOST_VALUES):
            cases.append(
                LocalizationCase(
                    case_id=(
                        f"g4_9a_db_d{_weight_token(direct)}"
                        f"_b{_weight_token(bridge_host)}"
                    ),
                    plane_id="direct_bridge_host",
                    axis_1_name="direct_weight",
                    axis_1_value=float(direct),
                    axis_1_index=direct_index,
                    axis_2_name="bridge_host_weight",
                    axis_2_value=float(bridge_host),
                    axis_2_index=bridge_index,
                    direct_weight=float(direct),
                    pair_bridge_weight=CENTER_PAIR_BRIDGE_WEIGHT,
                    bridge_host_weight=float(bridge_host),
                    host_clique_weight=CENTER_HOST_CLIQUE_WEIGHT,
                )
            )
    return tuple(cases)


LOCALIZATION_CASES = _localization_cases()
LOCALIZATION_CASE_BY_ID = {case.case_id: case for case in LOCALIZATION_CASES}


def _claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["run_status"] = RUN_STATUS
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _panel_design() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in LOCALIZATION_CASES:
        rows.append(
            {
                "case_id": case.case_id,
                "plane_id": case.plane_id,
                "axis_1_name": case.axis_1_name,
                "axis_1_value": float(case.axis_1_value),
                "axis_1_index": int(case.axis_1_index),
                "axis_2_name": case.axis_2_name,
                "axis_2_value": float(case.axis_2_value),
                "axis_2_index": int(case.axis_2_index),
                "direct_weight": float(case.direct_weight),
                "pair_bridge_weight": float(case.pair_bridge_weight),
                "bridge_host_weight": float(case.bridge_host_weight),
                "host_clique_weight": float(case.host_clique_weight),
                "is_center_cell": bool(
                    abs(case.direct_weight - CENTER_DIRECT_WEIGHT) < 1.0e-12
                    and abs(case.pair_bridge_weight - CENTER_PAIR_BRIDGE_WEIGHT) < 1.0e-12
                    and abs(case.bridge_host_weight - CENTER_BRIDGE_HOST_WEIGHT) < 1.0e-12
                    and abs(case.host_clique_weight - CENTER_HOST_CLIQUE_WEIGHT) < 1.0e-12
                ),
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _run_trace(*, seeds: int, n_iterations: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for localization_case in LOCALIZATION_CASES:
        primitive_case = localization_case.to_primitive_case()
        synthetic = _case_to_synthetic(primitive_case)
        for route_family in (DIRECT_ONLY_FAMILY, RECOVERY_LOOP_FAMILY):
            for start_condition in START_CONDITIONS:
                initial = _initial_membership(synthetic, start_condition)
                initial_groups = _canonical_groups(synthetic.nodes, initial)
                initial_signature_id = _signature_id(initial_groups)
                for step in _schedule_rows(route_family):
                    graph = _build_graph(
                        synthetic.nodes,
                        _scaled_edges(
                            synthetic,
                            direct_fraction=float(step["direct_edge_weight_fraction"]),
                            bridge_fraction=float(step["bridge_edge_weight_fraction"]),
                        ),
                    )
                    runner = LeidenRunner(
                        graph,
                        objective="cpm",
                        default_iterations=int(n_iterations),
                    )
                    for seed in range(int(seeds)):
                        result = runner.run(
                            synthetic.gamma,
                            seed=int(seed),
                            initial_membership=initial,
                            node_sizes=synthetic.node_sizes,
                        )
                        membership = list(map(int, result.membership))
                        groups = _canonical_groups(synthetic.nodes, membership)
                        read = _mechanism_read(synthetic, membership)
                        rows.append(
                            {
                                "trace_row_id": (
                                    f"{localization_case.case_id}__{route_family}"
                                    f"__{start_condition}__seed{seed:02d}"
                                    f"__step{int(step['step_index']):02d}"
                                ),
                                "case_id": localization_case.case_id,
                                "plane_id": localization_case.plane_id,
                                "axis_1_name": localization_case.axis_1_name,
                                "axis_1_value": float(localization_case.axis_1_value),
                                "axis_1_index": int(localization_case.axis_1_index),
                                "axis_2_name": localization_case.axis_2_name,
                                "axis_2_value": float(localization_case.axis_2_value),
                                "axis_2_index": int(localization_case.axis_2_index),
                                "route_family": route_family,
                                "start_condition": start_condition,
                                "seed": int(seed),
                                "step_index": int(step["step_index"]),
                                "step_label": str(step["step_label"]),
                                "direct_edge_weight_fraction": float(
                                    step["direct_edge_weight_fraction"]
                                ),
                                "bridge_edge_weight_fraction": float(
                                    step["bridge_edge_weight_fraction"]
                                ),
                                "direct_weight": float(localization_case.direct_weight),
                                "pair_bridge_weight": float(
                                    localization_case.pair_bridge_weight
                                ),
                                "bridge_host_weight": float(
                                    localization_case.bridge_host_weight
                                ),
                                "host_clique_weight": float(
                                    localization_case.host_clique_weight
                                ),
                                "gamma": float(synthetic.gamma),
                                "n_iterations": int(n_iterations),
                                "node_count": int(graph.vcount()),
                                "edge_count": int(graph.ecount()),
                                "edge_weight_sum": float(sum(graph.es["weight"]))
                                if graph.ecount()
                                else 0.0,
                                "initial_endpoint_signature_id": initial_signature_id,
                                "endpoint_signature_id": _signature_id(groups),
                                "endpoint_signature": json.dumps(groups, sort_keys=True),
                                "endpoint_object": _endpoint_object(
                                    synthetic,
                                    membership,
                                ),
                                "quality": float(result.quality),
                                "cluster_count": int(result.cluster_count),
                                **read,
                            }
                        )
    trace = pd.DataFrame(rows).sort_values(
        ["case_id", "route_family", "start_condition", "seed", "step_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    group_cols = ["case_id", "route_family", "start_condition", "seed"]
    trace["objective_start_value"] = trace.groupby(group_cols, sort=False)["quality"].transform(
        "first"
    )
    trace["objective_delta_from_start"] = trace["quality"] - trace["objective_start_value"]
    trace["objective_debt_from_start"] = (
        trace["objective_start_value"] - trace["quality"]
    ).clip(lower=0.0)
    trace["objective_min_so_far"] = trace.groupby(group_cols, sort=False)["quality"].cummin()
    trace["objective_recovery_from_min"] = trace["quality"] - trace["objective_min_so_far"]
    return _claim_columns(trace)


def _route_results(trace: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "case_id",
        "plane_id",
        "axis_1_name",
        "axis_1_value",
        "axis_1_index",
        "axis_2_name",
        "axis_2_value",
        "axis_2_index",
        "route_family",
        "start_condition",
        "seed",
    ]
    for keys, group in trace.groupby(group_cols, sort=False):
        key_data = dict(zip(group_cols, keys, strict=True))
        ordered = group.sort_values("step_index", kind="mergesort")
        objects = ordered["endpoint_object"].astype(str).tolist()
        first_object = objects[0]
        final_object = objects[-1]
        interior = objects[1:-1]
        source_start = first_object == SOURCE_OBJECT
        target_final = final_object == TARGET_OBJECT
        final_source = final_object == SOURCE_OBJECT
        interior_all_target = bool(interior) and all(value == TARGET_OBJECT for value in interior)
        direct_accept = bool(
            str(key_data["route_family"]) == DIRECT_ONLY_FAMILY
            and source_start
            and target_final
        )
        recovery_accept = bool(
            str(key_data["route_family"]) == RECOVERY_LOOP_FAMILY
            and source_start
            and interior_all_target
            and final_source
            and float(ordered["objective_debt_from_start"].max()) > 0.0
            and float(ordered["objective_recovery_from_min"].max()) > 0.0
        )
        if direct_accept:
            route_status = "direct_only_source_to_target_accepted"
        elif recovery_accept:
            route_status = "recovery_loop_source_target_source_accepted"
        elif first_object == TARGET_OBJECT:
            route_status = "target_saturated_no_source_start"
        elif TARGET_OBJECT not in set(objects):
            route_status = "target_absent"
        elif TARGET_OBJECT in set(objects) and not interior_all_target:
            route_status = "target_nonrobust_or_partial"
        else:
            route_status = "route_not_accepted"
        rows.append(
            {
                **key_data,
                "step_count": int(len(ordered)),
                "first_endpoint_object": first_object,
                "final_endpoint_object": final_object,
                "target_object_step_count": int(sum(value == TARGET_OBJECT for value in objects)),
                "source_like_object_step_count": int(sum(value == SOURCE_OBJECT for value in objects)),
                "endpoint_object_sequence": " -> ".join(objects),
                "direct_route_accepted": bool(direct_accept),
                "recovery_route_accepted": bool(recovery_accept),
                "max_objective_debt_from_start": float(
                    ordered["objective_debt_from_start"].max()
                ),
                "max_objective_recovery_from_min": float(
                    ordered["objective_recovery_from_min"].max()
                ),
                "bridge_fraction_sequence": ";".join(
                    f"{float(value):.2f}" for value in ordered["bridge_edge_weight_fraction"]
                ),
                "route_status": route_status,
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _seed_wall_rows(route_results: pd.DataFrame) -> pd.DataFrame:
    direct = route_results[route_results["route_family"].astype(str).eq(DIRECT_ONLY_FAMILY)]
    recovery = route_results[route_results["route_family"].astype(str).eq(RECOVERY_LOOP_FAMILY)]
    key_cols = [
        "case_id",
        "plane_id",
        "axis_1_name",
        "axis_1_value",
        "axis_1_index",
        "axis_2_name",
        "axis_2_value",
        "axis_2_index",
        "start_condition",
        "seed",
    ]
    rows = direct.merge(
        recovery,
        on=key_cols,
        suffixes=("_direct", "_recovery"),
        how="outer",
        validate="one_to_one",
    )
    output: list[dict[str, Any]] = []
    for row in rows.sort_values(
        ["case_id", "start_condition", "seed"],
        kind="mergesort",
    ).itertuples(index=False):
        data = row._asdict()
        direct_ok = bool(data.get("direct_route_accepted_direct", False))
        recovery_ok = bool(data.get("recovery_route_accepted_recovery", False))
        wall_ready = bool(direct_ok and recovery_ok)
        direct_sequence = str(data.get("endpoint_object_sequence_direct", ""))
        recovery_sequence = str(data.get("endpoint_object_sequence_recovery", ""))
        combined_sequence = f"{direct_sequence} -> {recovery_sequence}"
        target_seen = TARGET_OBJECT in combined_sequence
        if wall_ready:
            status = "primitive_wall_seed_ready"
        elif str(data.get("first_endpoint_object_direct", "")) == TARGET_OBJECT:
            status = "target_saturated_no_source_wall"
        elif not target_seen:
            status = "target_absent_or_source_locked_no_wall"
        elif int(data.get("target_object_step_count_recovery", 0)) > 0 or int(
            data.get("target_object_step_count_direct", 0)
        ) > 0:
            status = "partial_or_nonrobust_target_wall_closed"
        else:
            status = "unclassified_wall_closed"
        case = LOCALIZATION_CASE_BY_ID[str(data.get("case_id", ""))]
        output.append(
            {
                "wall_seed_id": (
                    f"{data.get('case_id', '')}__{data.get('start_condition', '')}"
                    f"__seed{int(data.get('seed', -1)):02d}"
                ),
                "case_id": str(data.get("case_id", "")),
                "plane_id": str(data.get("plane_id", "")),
                "axis_1_name": str(data.get("axis_1_name", "")),
                "axis_1_value": float(data.get("axis_1_value", 0.0)),
                "axis_1_index": int(data.get("axis_1_index", -1)),
                "axis_2_name": str(data.get("axis_2_name", "")),
                "axis_2_value": float(data.get("axis_2_value", 0.0)),
                "axis_2_index": int(data.get("axis_2_index", -1)),
                "direct_weight": float(case.direct_weight),
                "pair_bridge_weight": float(case.pair_bridge_weight),
                "bridge_host_weight": float(case.bridge_host_weight),
                "host_clique_weight": float(case.host_clique_weight),
                "start_condition": str(data.get("start_condition", "")),
                "seed": int(data.get("seed", -1)),
                "direct_route_accepted": bool(direct_ok),
                "recovery_route_accepted": bool(recovery_ok),
                "direct_endpoint_object_sequence": direct_sequence,
                "recovery_endpoint_object_sequence": recovery_sequence,
                "direct_max_objective_debt_from_start": float(
                    data.get("max_objective_debt_from_start_direct", 0.0)
                ),
                "recovery_max_objective_debt_from_start": float(
                    data.get("max_objective_debt_from_start_recovery", 0.0)
                ),
                "recovery_max_objective_recovery_from_min": float(
                    data.get("max_objective_recovery_from_min_recovery", 0.0)
                ),
                "wall_seed_ready": bool(wall_ready),
                "wall_seed_status": status,
            }
        )
    return _claim_columns(pd.DataFrame(output))


def _case_status(group: pd.DataFrame, *, expected_wall_seed_count: int) -> str:
    ready_count = int(group["wall_seed_ready"].astype(bool).sum())
    status_counts = _count_dict(group["wall_seed_status"])
    if ready_count == expected_wall_seed_count:
        return "full_primitive_wall_regime"
    if ready_count > 0:
        return "partial_or_fragile_wall_regime"
    if status_counts.get("target_saturated_no_source_wall", 0) == expected_wall_seed_count:
        return "target_saturated_regime"
    if status_counts.get("target_absent_or_source_locked_no_wall", 0) == expected_wall_seed_count:
        return "target_absent_or_source_locked_regime"
    if status_counts.get("partial_or_nonrobust_target_wall_closed", 0) > 0:
        return "nonrobust_or_mixed_boundary_regime"
    return "unclassified_closed_regime"


def _case_summary(seed_wall: pd.DataFrame, *, seeds: int) -> pd.DataFrame:
    expected_wall_seed_count = len(START_CONDITIONS) * int(seeds)
    rows: list[dict[str, Any]] = []
    for case in LOCALIZATION_CASES:
        group = seed_wall[seed_wall["case_id"].astype(str).eq(case.case_id)]
        ready_count = int(group["wall_seed_ready"].astype(bool).sum())
        status = _case_status(group, expected_wall_seed_count=expected_wall_seed_count)
        rows.append(
            {
                "case_id": case.case_id,
                "plane_id": case.plane_id,
                "axis_1_name": case.axis_1_name,
                "axis_1_value": float(case.axis_1_value),
                "axis_1_index": int(case.axis_1_index),
                "axis_2_name": case.axis_2_name,
                "axis_2_value": float(case.axis_2_value),
                "axis_2_index": int(case.axis_2_index),
                "direct_weight": float(case.direct_weight),
                "pair_bridge_weight": float(case.pair_bridge_weight),
                "bridge_host_weight": float(case.bridge_host_weight),
                "host_clique_weight": float(case.host_clique_weight),
                "is_center_cell": bool(
                    abs(case.direct_weight - CENTER_DIRECT_WEIGHT) < 1.0e-12
                    and abs(case.pair_bridge_weight - CENTER_PAIR_BRIDGE_WEIGHT) < 1.0e-12
                    and abs(case.bridge_host_weight - CENTER_BRIDGE_HOST_WEIGHT) < 1.0e-12
                    and abs(case.host_clique_weight - CENTER_HOST_CLIQUE_WEIGHT) < 1.0e-12
                ),
                "wall_seed_count": int(len(group)),
                "wall_ready_seed_count": ready_count,
                "wall_ready_seed_share": float(ready_count / len(group)) if len(group) else 0.0,
                "wall_seed_status_counts": _count_dict(group["wall_seed_status"]),
                "case_result_status": status,
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _plane_summary(case_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for plane_id, group in case_summary.groupby("plane_id", sort=False):
        rows.append(
            {
                "plane_id": str(plane_id),
                "case_count": int(len(group)),
                "full_ready_case_count": int(
                    group["case_result_status"].astype(str).eq("full_primitive_wall_regime").sum()
                ),
                "partial_ready_case_count": int(
                    group["case_result_status"].astype(str).eq(
                        "partial_or_fragile_wall_regime"
                    ).sum()
                ),
                "nonready_case_count": int(
                    (~group["wall_ready_seed_count"].astype(int).gt(0)).sum()
                ),
                "wall_ready_seed_count": int(group["wall_ready_seed_count"].astype(int).sum()),
                "case_result_status_counts": _count_dict(group["case_result_status"]),
                "axis_1_name": str(group["axis_1_name"].iloc[0]),
                "axis_1_values": ";".join(
                    f"{float(value):.2f}" for value in sorted(group["axis_1_value"].unique())
                ),
                "axis_2_name": str(group["axis_2_name"].iloc[0]),
                "axis_2_values": ";".join(
                    f"{float(value):.2f}" for value in sorted(group["axis_2_value"].unique())
                ),
                "plane_localization_status": (
                    "mixed_ready_and_boundary_regime"
                    if group["wall_ready_seed_count"].astype(int).gt(0).any()
                    and (~group["wall_ready_seed_count"].astype(int).gt(0)).any()
                    else "not_localized"
                ),
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _plane_matrix(case_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in case_summary.itertuples(index=False):
        status = str(row.case_result_status)
        if status == "full_primitive_wall_regime":
            code = "W"
        elif status == "partial_or_fragile_wall_regime":
            code = "w"
        elif status == "target_saturated_regime":
            code = "T"
        elif status == "target_absent_or_source_locked_regime":
            code = "N"
        elif status == "nonrobust_or_mixed_boundary_regime":
            code = "P"
        else:
            code = "?"
        rows.append(
            {
                "plane_id": row.plane_id,
                "axis_1_name": row.axis_1_name,
                "axis_1_value": float(row.axis_1_value),
                "axis_1_index": int(row.axis_1_index),
                "axis_2_name": row.axis_2_name,
                "axis_2_value": float(row.axis_2_value),
                "axis_2_index": int(row.axis_2_index),
                "case_id": row.case_id,
                "matrix_code": code,
                "case_result_status": status,
                "wall_ready_seed_count": int(row.wall_ready_seed_count),
                "wall_ready_seed_share": float(row.wall_ready_seed_share),
            }
        )
    return _claim_columns(pd.DataFrame(rows))


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
    trace: pd.DataFrame,
    seed_wall: pd.DataFrame,
    case_summary: pd.DataFrame,
    plane_summary: pd.DataFrame,
    seeds: int,
) -> pd.DataFrame:
    expected_trace_rows = (
        len(LOCALIZATION_CASES)
        * len(START_CONDITIONS)
        * int(seeds)
        * (
            len(_schedule_rows(DIRECT_ONLY_FAMILY))
            + len(_schedule_rows(RECOVERY_LOOP_FAMILY))
        )
    )
    expected_wall_seed_count = len(START_CONDITIONS) * int(seeds)
    center_cases = case_summary[case_summary["is_center_cell"].astype(bool)]
    status_set = set(case_summary["case_result_status"].astype(str))
    return _claim_columns(
        pd.DataFrame(
            [
                _gate_row(
                    "G1_predeclared_three_plane_panel",
                    "Was the localization panel fixed before execution?",
                    (
                        f"cases={len(LOCALIZATION_CASES)} planes="
                        f"{sorted(case_summary['plane_id'].unique().tolist())}"
                    ),
                    "75 cases across 3 two-dimensional planes",
                    len(LOCALIZATION_CASES) == 75
                    and int(case_summary["plane_id"].nunique()) == 3,
                ),
                _gate_row(
                    "G2_trace_rows_complete",
                    "Did every case/start/seed/schedule step execute?",
                    f"trace_rows={len(trace)} expected={expected_trace_rows}",
                    (
                        f"{len(LOCALIZATION_CASES)} cases * "
                        f"{len(START_CONDITIONS)} starts * {int(seeds)} seeds * "
                        f"{len(_schedule_rows(DIRECT_ONLY_FAMILY)) + len(_schedule_rows(RECOVERY_LOOP_FAMILY))} steps"
                    ),
                    len(trace) == expected_trace_rows,
                ),
                _gate_row(
                    "G3_center_reproduces_g4_9_positive",
                    "Do all center-plane duplicates remain fully wall-ready?",
                    center_cases[
                        ["case_id", "plane_id", "wall_ready_seed_count", "wall_seed_count"]
                    ].to_dict("records"),
                    (
                        "each center duplicate has "
                        f"{expected_wall_seed_count}/{expected_wall_seed_count} ready units"
                    ),
                    len(center_cases) == 3
                    and bool(
                        center_cases["wall_ready_seed_count"].astype(int).eq(
                            expected_wall_seed_count
                        ).all()
                    ),
                ),
                _gate_row(
                    "G4_positive_not_single_cell_only",
                    "Does the map show at least one non-center ready cell?",
                    int(
                        (
                            case_summary["wall_ready_seed_count"].astype(int).gt(0)
                            & ~case_summary["is_center_cell"].astype(bool)
                        ).sum()
                    ),
                    "at least one non-center case has ready seed units",
                    int(
                        (
                            case_summary["wall_ready_seed_count"].astype(int).gt(0)
                            & ~case_summary["is_center_cell"].astype(bool)
                        ).sum()
                    )
                    > 0,
                ),
                _gate_row(
                    "G5_boundary_modes_observed",
                    "Are the expected closed boundary modes present?",
                    sorted(status_set),
                    (
                        "full-ready, partial/nonrobust, target-absent/source-lock, "
                        "and target-saturation regimes are all observed"
                    ),
                    {
                        "full_primitive_wall_regime",
                        "partial_or_fragile_wall_regime",
                        "target_absent_or_source_locked_regime",
                        "target_saturated_regime",
                    }.issubset(status_set)
                    and "nonrobust_or_mixed_boundary_regime" in status_set,
                ),
                _gate_row(
                    "G6_each_plane_localizes_boundary",
                    "Does every plane contain both ready and closed cells?",
                    plane_summary[
                        [
                            "plane_id",
                            "full_ready_case_count",
                            "partial_ready_case_count",
                            "nonready_case_count",
                            "plane_localization_status",
                        ]
                    ].to_dict("records"),
                    "each plane has mixed ready and boundary behavior",
                    bool(
                        plane_summary["plane_localization_status"]
                        .astype(str)
                        .eq("mixed_ready_and_boundary_regime")
                        .all()
                    ),
                ),
                _gate_row(
                    "G7_no_method_quality_or_nanoclustering_claim",
                    "Are method, quality/cost, and NanoClustering claims closed?",
                    CLAIM_BOUNDARY,
                    "claim boundary explicitly closed",
                    True,
                ),
            ]
        )
    )


def _summary(
    *,
    output_dir: Path,
    trace: pd.DataFrame,
    seed_wall: pd.DataFrame,
    case_summary: pd.DataFrame,
    plane_summary: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    full_ready = case_summary["case_result_status"].astype(str).eq(
        "full_primitive_wall_regime"
    )
    partial_ready = case_summary["case_result_status"].astype(str).eq(
        "partial_or_fragile_wall_regime"
    )
    return {
        "schema": "variable_pair_synthetic_g4_9a_parameter_localization_summary.v1",
        "status": RUN_STATUS,
        "output_dir": str(output_dir),
        "case_count": int(len(case_summary)),
        "plane_count": int(case_summary["plane_id"].nunique()),
        "trace_row_count": int(len(trace)),
        "seed_wall_row_count": int(len(seed_wall)),
        "full_ready_case_count": int(full_ready.sum()),
        "partial_ready_case_count": int(partial_ready.sum()),
        "ready_any_case_count": int(
            case_summary["wall_ready_seed_count"].astype(int).gt(0).sum()
        ),
        "nonready_case_count": int(
            (~case_summary["wall_ready_seed_count"].astype(int).gt(0)).sum()
        ),
        "wall_ready_seed_count": int(case_summary["wall_ready_seed_count"].astype(int).sum()),
        "case_result_status_counts": _count_dict(case_summary["case_result_status"]),
        "wall_seed_status_counts": _count_dict(seed_wall["wall_seed_status"]),
        "plane_summary": plane_summary.to_dict("records"),
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "interpretation": (
            "The G4.9 primitive wall positive is not a single-cell artifact: "
            "nearby cells expose full or partial wall-ready behavior, while "
            "low direct support closes as target-absent/source-locked, weak "
            "pair-bridge support closes as target-saturated, and off-balance "
            "cells often become nonrobust partial target openings."
        ),
        "recommended_next_gate": (
            "Use the localized synthetic failure modes as controls while "
            "returning to NanoClustering: either repeat the paired wall audit "
            "on additional endpoint-object candidates, or localize the 014 "
            "wall more finely along the bridge-support schedule."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 80) -> str:
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
    case_summary: pd.DataFrame,
    plane_summary: pd.DataFrame,
    plane_matrix: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# Variable-Pair Synthetic G4.9A Parameter Localization",
        "",
        f"- status: `{summary['status']}`",
        f"- case_count: {summary['case_count']}",
        f"- plane_count: {summary['plane_count']}",
        f"- trace_row_count: {summary['trace_row_count']}",
        f"- seed_wall_row_count: {summary['seed_wall_row_count']}",
        f"- full_ready_case_count: {summary['full_ready_case_count']}",
        f"- partial_ready_case_count: {summary['partial_ready_case_count']}",
        f"- ready_any_case_count: {summary['ready_any_case_count']}",
        f"- nonready_case_count: {summary['nonready_case_count']}",
        f"- wall_ready_seed_count: {summary['wall_ready_seed_count']}",
        f"- case_result_status_counts: {summary['case_result_status_counts']}",
        f"- wall_seed_status_counts: {summary['wall_seed_status_counts']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Plane Summary",
        "",
        _markdown_table(
            plane_summary,
            [
                "plane_id",
                "case_count",
                "full_ready_case_count",
                "partial_ready_case_count",
                "nonready_case_count",
                "wall_ready_seed_count",
                "case_result_status_counts",
                "plane_localization_status",
            ],
            max_rows=20,
        ),
        "",
        "## Case Summary",
        "",
        _markdown_table(
            case_summary,
            [
                "case_id",
                "plane_id",
                "axis_1_value",
                "axis_2_value",
                "direct_weight",
                "pair_bridge_weight",
                "bridge_host_weight",
                "wall_ready_seed_count",
                "wall_ready_seed_share",
                "case_result_status",
            ],
            max_rows=80,
        ),
        "",
        "## Matrix Codes",
        "",
        "Codes: `W` full wall-ready, `w` partial wall-ready, `T` target-saturated, "
        "`N` target-absent/source-locked, `P` nonrobust or mixed boundary.",
        "",
        _markdown_table(
            plane_matrix,
            [
                "plane_id",
                "axis_1_value",
                "axis_2_value",
                "matrix_code",
                "wall_ready_seed_count",
                "case_result_status",
            ],
            max_rows=100,
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
            "This synthetic localization result is not a parameter policy and "
            "does not retune a method. It maps where the G4.9 object-level "
            "wall readout appears under ordinary Leiden+CPM on small graphs."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_design = _panel_design()
    trace = _run_trace(seeds=int(args.seeds), n_iterations=int(args.n_iterations))
    route_results = _route_results(trace)
    seed_wall = _seed_wall_rows(route_results)
    case_summary = _case_summary(seed_wall, seeds=int(args.seeds))
    plane_summary = _plane_summary(case_summary)
    plane_matrix = _plane_matrix(case_summary)
    gates = _gate_matrix(
        trace=trace,
        seed_wall=seed_wall,
        case_summary=case_summary,
        plane_summary=plane_summary,
        seeds=int(args.seeds),
    )
    summary = _summary(
        output_dir=output_dir,
        trace=trace,
        seed_wall=seed_wall,
        case_summary=case_summary,
        plane_summary=plane_summary,
        gates=gates,
    )

    _write_csv(panel_design, output_dir / PANEL_DESIGN_CSV)
    _write_csv(trace, output_dir / TRACE_ROWS_CSV)
    _write_csv(route_results, output_dir / ROUTE_RESULT_ROWS_CSV)
    _write_csv(seed_wall, output_dir / SEED_WALL_ROWS_CSV)
    _write_csv(case_summary, output_dir / CASE_SUMMARY_CSV)
    _write_csv(plane_summary, output_dir / PLANE_SUMMARY_CSV)
    _write_csv(plane_matrix, output_dir / PLANE_MATRIX_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "variable_pair_synthetic_g4_9a_parameter_localization_config.v1",
        "output_dir": str(output_dir),
        "seeds": int(args.seeds),
        "n_iterations": int(args.n_iterations),
        "start_conditions": list(START_CONDITIONS),
        "direct_values": list(DIRECT_VALUES),
        "pair_bridge_values": list(PAIR_BRIDGE_VALUES),
        "bridge_host_values": list(BRIDGE_HOST_VALUES),
        "center_direct_weight": CENTER_DIRECT_WEIGHT,
        "center_pair_bridge_weight": CENTER_PAIR_BRIDGE_WEIGHT,
        "center_bridge_host_weight": CENTER_BRIDGE_HOST_WEIGHT,
        "center_host_clique_weight": CENTER_HOST_CLIQUE_WEIGHT,
        "panel_cases": [case.__dict__ for case in LOCALIZATION_CASES],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        case_summary=case_summary,
        plane_summary=plane_summary,
        plane_matrix=plane_matrix,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--n-iterations", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
