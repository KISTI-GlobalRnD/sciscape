#!/usr/bin/env python3
"""Run a synthetic G4.9 primitive wall demo inspired by NanoClustering 014.

This runner creates a small, controlled Leiden+CPM variable-pair graph surface
whose positive case is designed to reproduce the object-level relation observed
in ``local_pair_014``:

source-like endpoint object -> exclusive target object -> source-like endpoint
object.

The positive case and boundary controls are predeclared. The audit unit pairs a
direct-only route and a recovery-loop route for the same start condition and
seed. This is a synthetic mechanism demo only. It does not replay
NanoClustering, measure quality/cost value, promote a method, or make an
algorithm-level claim.
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
    EdgeSpec,
    _build_graph,
    _canonical_groups,
    _competition_case,
    _initial_membership,
    _json_safe,
    _mechanism_read,
    _signature_id,
    _write_csv,
)
from sciscape.clustering.runner import LeidenRunner


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_variable_pair_synthetic_g4_9_primitive_wall_demo_v1_20260604"
)

PANEL_DESIGN_CSV = "variable_pair_synthetic_g4_9_panel_design.csv"
GRAPH_EDGES_CSV = "variable_pair_synthetic_g4_9_graph_edges.csv"
TRACE_ROWS_CSV = "variable_pair_synthetic_g4_9_trace_rows.csv"
ROUTE_RESULT_ROWS_CSV = "variable_pair_synthetic_g4_9_route_result_rows.csv"
SEED_WALL_ROWS_CSV = "variable_pair_synthetic_g4_9_seed_wall_rows.csv"
CASE_SUMMARY_CSV = "variable_pair_synthetic_g4_9_case_summary.csv"
GATE_MATRIX_CSV = "variable_pair_synthetic_g4_9_gate_matrix.csv"
SUMMARY_JSON = "variable_pair_synthetic_g4_9_summary.json"
CONFIG_JSON = "variable_pair_synthetic_g4_9_config.json"
REPORT_MD = "variable_pair_synthetic_g4_9_report.md"

START_CONDITIONS = (
    "all_local_together",
    "bridges_to_left",
    "bridges_to_right",
    "pair_together",
)
RECOVERY_BRIDGE_FRACTIONS = (1.0, 0.75, 0.50, 0.25, 0.0, 0.25, 0.50, 0.75, 1.0)
DIRECT_ONLY_BRIDGE_FRACTIONS = (1.0, 0.0)
DIRECT_ONLY_FAMILY = "direct_only_target_availability"
RECOVERY_LOOP_FAMILY = "bridge_recovery_loop"
TARGET_OBJECT = "exclusive_target_endpoint_object"
SOURCE_OBJECT = "source_like_endpoint_object"

RUN_STATUS = "executed_variable_pair_synthetic_g4_9_primitive_wall_demo"
ROUTE_EXECUTION_STATUS = "executed_synthetic_g4_9_direct_recovery_wall_probe"
WALL_PROMOTION_STATUS = "synthetic_primitive_wall_demo_only"
METHOD_STATUS = "plain_leiden_cpm_synthetic_wall_mechanism_demo_not_method"
CLAIM_BOUNDARY = (
    "Variable-pair synthetic G4.9 primitive wall demo only; predeclared small "
    "Leiden+CPM graphs test source-like/target/source-like object-level wall "
    "mechanisms and boundary controls. No NanoClustering replay, no "
    "quality/cost value, no method claim, and no algorithm-level claim."
)


@dataclass(frozen=True)
class PrimitiveWallCase:
    case_id: str
    case_role: str
    expected_case_status: str
    direct_weight: float
    pair_bridge_weight: float
    bridge_host_weight: float
    host_clique_weight: float
    note: str
    pair_node_size: int = 1
    gamma: float = 1.0


PANEL_CASES: tuple[PrimitiveWallCase, ...] = (
    PrimitiveWallCase(
        case_id="g4_9_wall_ready_center",
        case_role="positive_014_like_wall",
        expected_case_status="primitive_wall_ready",
        direct_weight=1.05,
        pair_bridge_weight=2.50,
        bridge_host_weight=2.00,
        host_clique_weight=0.20,
        note=(
            "014-like balance: full bridge support gives source-like endpoints, "
            "bridge suppression exposes the exclusive target, and bridge "
            "restoration recovers source-like endpoints."
        ),
    ),
    PrimitiveWallCase(
        case_id="g4_9_target_saturated_high_direct",
        case_role="boundary_control",
        expected_case_status="target_saturated_no_source_wall",
        direct_weight=2.00,
        pair_bridge_weight=1.00,
        bridge_host_weight=3.00,
        host_clique_weight=0.80,
        note="Direct support dominates: the source-like endpoint is absent.",
    ),
    PrimitiveWallCase(
        case_id="g4_9_target_absent_low_direct",
        case_role="boundary_control",
        expected_case_status="target_absent_no_wall",
        direct_weight=0.50,
        pair_bridge_weight=3.00,
        bridge_host_weight=3.00,
        host_clique_weight=0.80,
        note="Direct support is too weak: bridge suppression cannot open target.",
    ),
    PrimitiveWallCase(
        case_id="g4_9_nonrobust_low_context",
        case_role="boundary_control",
        expected_case_status="nonrobust_partial_wall",
        direct_weight=1.00,
        pair_bridge_weight=3.00,
        bridge_host_weight=0.80,
        host_clique_weight=0.80,
        note=(
            "Target opens only in the middle of the schedule; the full "
            "source-like/target/source-like wall relation is not stable."
        ),
    ),
    PrimitiveWallCase(
        case_id="g4_9_source_locked_strong_bridge",
        case_role="boundary_control",
        expected_case_status="source_locked_no_target_wall",
        direct_weight=0.80,
        pair_bridge_weight=4.00,
        bridge_host_weight=3.00,
        host_clique_weight=0.80,
        note="Source context dominates and bridge suppression does not open target.",
    ),
)


def _claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["run_status"] = RUN_STATUS
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _case_to_synthetic(case: PrimitiveWallCase):
    return _competition_case(
        design_family=case.case_id,
        synthetic_demo_role=case.case_role,
        expected_signature=case.expected_case_status,
        direct_weight=case.direct_weight,
        pair_bridge_weight=case.pair_bridge_weight,
        bridge_host_weight=case.bridge_host_weight,
        host_clique_weight=case.host_clique_weight,
        gamma=case.gamma,
        pair_node_size=case.pair_node_size,
    )


def _scaled_edges(synthetic, *, direct_fraction: float, bridge_fraction: float) -> tuple[EdgeSpec, ...]:
    edges: list[EdgeSpec] = []
    for edge in synthetic.edges:
        scale = 1.0
        if edge.edge_type == "direct_pair":
            scale *= float(direct_fraction)
        if edge.edge_type == "pair_bridge":
            scale *= float(bridge_fraction)
        weight = float(edge.weight) * scale
        if weight > 0.0:
            edges.append(
                EdgeSpec(
                    left=edge.left,
                    right=edge.right,
                    weight=weight,
                    edge_type=edge.edge_type,
                )
            )
    return tuple(edges)


def _endpoint_object(synthetic, membership: list[int]) -> str:
    read = _mechanism_read(synthetic, membership)
    if bool(read["pair_coassigned"]):
        bridge_count = int(read["left_bridge_same_cluster_count"]) + int(
            read["right_bridge_same_cluster_count"]
        )
        if bridge_count == 0:
            return TARGET_OBJECT
        return "target_with_bridge_context_object"
    if int(read["left_bridge_same_cluster_count"]) or int(read["right_bridge_same_cluster_count"]):
        return SOURCE_OBJECT
    return "separated_no_bridge_endpoint_object"


def _panel_design_and_edges() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    for case in PANEL_CASES:
        synthetic = _case_to_synthetic(case)
        panel_rows.append(
            {
                "case_id": case.case_id,
                "case_role": case.case_role,
                "expected_case_status": case.expected_case_status,
                "direct_weight": float(case.direct_weight),
                "pair_bridge_weight": float(case.pair_bridge_weight),
                "bridge_host_weight": float(case.bridge_host_weight),
                "host_clique_weight": float(case.host_clique_weight),
                "pair_node_size": int(case.pair_node_size),
                "gamma": float(case.gamma),
                "node_count": int(len(synthetic.nodes)),
                "bridge_nodes": ";".join(synthetic.bridge_nodes),
                "note": case.note,
            }
        )
        for edge in synthetic.edges:
            edge_rows.append(
                {
                    "case_id": case.case_id,
                    "source": edge.left,
                    "target": edge.right,
                    "weight": float(edge.weight),
                    "edge_type": edge.edge_type,
                }
            )
    return _claim_columns(pd.DataFrame(panel_rows)), _claim_columns(pd.DataFrame(edge_rows))


def _schedule_rows(route_family: str) -> tuple[dict[str, Any], ...]:
    if route_family == DIRECT_ONLY_FAMILY:
        return tuple(
            {
                "step_index": index,
                "step_label": (
                    "baseline_source_support" if fraction == 1.0 else "direct_only_bridge_suppressed"
                ),
                "direct_edge_weight_fraction": 1.0,
                "bridge_edge_weight_fraction": float(fraction),
            }
            for index, fraction in enumerate(DIRECT_ONLY_BRIDGE_FRACTIONS, start=1)
        )
    if route_family == RECOVERY_LOOP_FAMILY:
        return tuple(
            {
                "step_index": index,
                "step_label": f"recovery_bridge_fraction_{fraction:.2f}",
                "direct_edge_weight_fraction": 1.0,
                "bridge_edge_weight_fraction": float(fraction),
            }
            for index, fraction in enumerate(RECOVERY_BRIDGE_FRACTIONS, start=1)
        )
    raise ValueError(f"unknown route family: {route_family}")


def _run_trace(*, seeds: int, n_iterations: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cases_by_id = {case.case_id: case for case in PANEL_CASES}
    for case in PANEL_CASES:
        synthetic = _case_to_synthetic(case)
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
                                    f"{case.case_id}__{route_family}__{start_condition}"
                                    f"__seed{seed:02d}__step{int(step['step_index']):02d}"
                                ),
                                "case_id": case.case_id,
                                "case_role": case.case_role,
                                "expected_case_status": case.expected_case_status,
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
                                "gamma": float(synthetic.gamma),
                                "n_iterations": int(n_iterations),
                                "direct_weight": float(cases_by_id[case.case_id].direct_weight),
                                "pair_bridge_weight": float(
                                    cases_by_id[case.case_id].pair_bridge_weight
                                ),
                                "bridge_host_weight": float(
                                    cases_by_id[case.case_id].bridge_host_weight
                                ),
                                "host_clique_weight": float(
                                    cases_by_id[case.case_id].host_clique_weight
                                ),
                                "node_count": int(graph.vcount()),
                                "edge_count": int(graph.ecount()),
                                "edge_weight_sum": float(sum(graph.es["weight"]))
                                if graph.ecount()
                                else 0.0,
                                "initial_endpoint_signature_id": initial_signature_id,
                                "endpoint_signature_id": _signature_id(groups),
                                "endpoint_signature": json.dumps(groups, sort_keys=True),
                                "endpoint_object": _endpoint_object(synthetic, membership),
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


def _sequence(values: pd.Series) -> str:
    return " -> ".join(str(value) for value in values.astype(str).tolist())


def _route_results(trace: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["case_id", "case_role", "expected_case_status", "route_family", "start_condition", "seed"]
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
    key_cols = ["case_id", "case_role", "expected_case_status", "start_condition", "seed"]
    rows = direct.merge(
        recovery,
        on=key_cols,
        suffixes=("_direct", "_recovery"),
        how="outer",
        validate="one_to_one",
    )
    output: list[dict[str, Any]] = []
    for row in rows.sort_values(["case_id", "start_condition", "seed"], kind="mergesort").itertuples(index=False):
        data = row._asdict()
        direct_ok = bool(data.get("direct_route_accepted_direct", False))
        recovery_ok = bool(data.get("recovery_route_accepted_recovery", False))
        wall_ready = bool(direct_ok and recovery_ok)
        if wall_ready:
            status = "primitive_wall_seed_ready"
        elif str(data.get("first_endpoint_object_direct", "")) == TARGET_OBJECT:
            status = "target_saturated_no_source_wall"
        elif int(data.get("target_object_step_count_direct", 0)) == 0 and int(
            data.get("target_object_step_count_recovery", 0)
        ) == 0:
            status = "target_absent_no_wall"
        elif int(data.get("target_object_step_count_recovery", 0)) > 0:
            status = "partial_or_nonrobust_target_wall_closed"
        else:
            status = "source_locked_or_unclassified_wall_closed"
        output.append(
            {
                "wall_seed_id": (
                    f"{data.get('case_id', '')}__{data.get('start_condition', '')}"
                    f"__seed{int(data.get('seed', -1)):02d}"
                ),
                "case_id": str(data.get("case_id", "")),
                "case_role": str(data.get("case_role", "")),
                "expected_case_status": str(data.get("expected_case_status", "")),
                "start_condition": str(data.get("start_condition", "")),
                "seed": int(data.get("seed", -1)),
                "direct_route_accepted": bool(direct_ok),
                "recovery_route_accepted": bool(recovery_ok),
                "direct_endpoint_object_sequence": str(
                    data.get("endpoint_object_sequence_direct", "")
                ),
                "recovery_endpoint_object_sequence": str(
                    data.get("endpoint_object_sequence_recovery", "")
                ),
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


def _case_summary(seed_wall: pd.DataFrame, *, seeds: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected_wall_seed_count = len(START_CONDITIONS) * int(seeds)
    for case in PANEL_CASES:
        group = seed_wall[seed_wall["case_id"].astype(str).eq(case.case_id)]
        wall_ready_count = int(group["wall_seed_ready"].astype(bool).sum())
        control_leak = bool(case.case_role != "positive_014_like_wall" and wall_ready_count > 0)
        expected_positive = case.case_role == "positive_014_like_wall"
        expected_pass = bool(
            (
                expected_positive
                and wall_ready_count == len(group)
                and len(group) == expected_wall_seed_count
            )
            or (
                not expected_positive
                and wall_ready_count == 0
                and len(group) == expected_wall_seed_count
            )
        )
        if expected_positive and expected_pass:
            status = "positive_primitive_wall_reproduced"
        elif control_leak:
            status = "boundary_control_wall_leak"
        elif not expected_positive and expected_pass:
            status = "boundary_control_closed"
        else:
            status = "case_unexpected_result"
        rows.append(
            {
                "case_id": case.case_id,
                "case_role": case.case_role,
                "expected_case_status": case.expected_case_status,
                "wall_seed_count": int(len(group)),
                "wall_ready_seed_count": wall_ready_count,
                "control_wall_leak_observed": bool(control_leak),
                "wall_seed_status_counts": _count_dict(group["wall_seed_status"]),
                "expected_case_behavior_pass": bool(expected_pass),
                "case_result_status": status,
                "direct_weight": float(case.direct_weight),
                "pair_bridge_weight": float(case.pair_bridge_weight),
                "bridge_host_weight": float(case.bridge_host_weight),
                "host_clique_weight": float(case.host_clique_weight),
                "mechanism_interpretation": case.note,
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _gate_row(gate_id: str, question: str, observed: Any, minimum_or_rule: str, passed: bool) -> dict[str, Any]:
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
    seeds: int,
) -> pd.DataFrame:
    positive = case_summary[case_summary["case_role"].astype(str).eq("positive_014_like_wall")]
    controls = case_summary[case_summary["case_role"].astype(str).ne("positive_014_like_wall")]
    expected_wall_seed_count = len(START_CONDITIONS) * int(seeds)
    expected_trace_rows = len(PANEL_CASES) * 2 * len(START_CONDITIONS) * int(seeds) * (
        len(DIRECT_ONLY_BRIDGE_FRACTIONS) + len(RECOVERY_BRIDGE_FRACTIONS)
    ) // 2
    return _claim_columns(
        pd.DataFrame(
            [
                _gate_row(
                    "G1_predeclared_panel_size",
                    "Were the positive and boundary synthetic cases predeclared?",
                    f"case_count={len(PANEL_CASES)} positive={len(positive)} controls={len(controls)}",
                    "1 positive and 4 controls",
                    len(positive) == 1 and len(controls) == 4,
                ),
                _gate_row(
                    "G2_trace_rows_complete",
                    "Did every case/start/seed/schedule step execute?",
                    f"trace_rows={len(trace)} expected={expected_trace_rows}",
                    (
                        f"{len(PANEL_CASES)} cases * {len(START_CONDITIONS)} starts * "
                        f"{int(seeds)} seeds * "
                        f"({len(DIRECT_ONLY_BRIDGE_FRACTIONS)}+"
                        f"{len(RECOVERY_BRIDGE_FRACTIONS)} route steps)"
                    ),
                    len(trace) == expected_trace_rows,
                ),
                _gate_row(
                    "G3_positive_wall_seed_units_ready",
                    "Does the 014-like synthetic positive reproduce primitive wall evidence?",
                    positive[["case_id", "wall_ready_seed_count", "wall_seed_count"]].to_dict("records"),
                    (
                        "positive has "
                        f"{expected_wall_seed_count} of {expected_wall_seed_count} "
                        "wall seed units ready"
                    ),
                    not positive.empty
                    and int(positive.iloc[0]["wall_seed_count"]) == expected_wall_seed_count
                    and int(positive.iloc[0]["wall_ready_seed_count"]) == expected_wall_seed_count,
                ),
                _gate_row(
                    "G4_boundary_controls_no_wall_leak",
                    "Do all boundary controls stay non-positive?",
                    controls[["case_id", "wall_ready_seed_count", "case_result_status"]].to_dict("records"),
                    "all controls have zero wall-ready seed units",
                    len(controls) == 4
                    and bool(controls["wall_ready_seed_count"].astype(int).eq(0).all()),
                ),
                _gate_row(
                    "G5_expected_case_behaviors_pass",
                    "Does every case match its predeclared role?",
                    _count_dict(case_summary["case_result_status"]),
                    "all five case behaviors pass",
                    bool(case_summary["expected_case_behavior_pass"].astype(bool).all()),
                ),
                _gate_row(
                    "G6_object_level_not_exact_label_claim",
                    "Is the synthetic result read at endpoint-object level?",
                    sorted(seed_wall["wall_seed_status"].astype(str).unique().tolist()),
                    "wall statuses are endpoint-object mechanism statuses",
                    "primitive_wall_seed_ready" in set(seed_wall["wall_seed_status"].astype(str)),
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
    gates: pd.DataFrame,
) -> dict[str, Any]:
    positive = case_summary[case_summary["case_role"].astype(str).eq("positive_014_like_wall")]
    controls = case_summary[case_summary["case_role"].astype(str).ne("positive_014_like_wall")]
    return {
        "schema": "variable_pair_synthetic_g4_9_primitive_wall_demo_summary.v1",
        "status": RUN_STATUS,
        "output_dir": str(output_dir),
        "case_count": int(len(case_summary)),
        "trace_row_count": int(len(trace)),
        "seed_wall_row_count": int(len(seed_wall)),
        "positive_wall_ready_seed_count": int(positive.iloc[0]["wall_ready_seed_count"])
        if not positive.empty
        else 0,
        "positive_wall_seed_count": int(positive.iloc[0]["wall_seed_count"])
        if not positive.empty
        else 0,
        "control_wall_leak_case_count": int(
            controls["control_wall_leak_observed"].astype(bool).sum()
        ),
        "case_result_status_counts": _count_dict(case_summary["case_result_status"]),
        "wall_seed_status_counts": _count_dict(seed_wall["wall_seed_status"]),
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "interpretation": (
            "A small synthetic Leiden+CPM graph can reproduce the 014-like "
            "primitive wall relation when direct support, pair-bridge support, "
            "and bridge-host context are balanced. The controls show three "
            "failure modes: target saturation, target absence/source lock, and "
            "nonrobust partial target opening."
        ),
        "recommended_next_gate": (
            "Use this G4.9 demo as the synthetic explanation scaffold, then "
            "either map the local parameter regime around the positive point or "
            "apply the paired wall audit to additional NanoClustering candidates."
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
    case_summary: pd.DataFrame,
    seed_wall: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# Variable-Pair Synthetic G4.9 Primitive Wall Demo",
        "",
        f"- status: `{summary['status']}`",
        f"- case_count: {summary['case_count']}",
        f"- trace_row_count: {summary['trace_row_count']}",
        f"- seed_wall_row_count: {summary['seed_wall_row_count']}",
        f"- positive_wall_ready_seed_count: {summary['positive_wall_ready_seed_count']}/{summary['positive_wall_seed_count']}",
        f"- control_wall_leak_case_count: {summary['control_wall_leak_case_count']}",
        f"- case_result_status_counts: {summary['case_result_status_counts']}",
        f"- wall_seed_status_counts: {summary['wall_seed_status_counts']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Case Summary",
        "",
        _markdown_table(
            case_summary,
            [
                "case_id",
                "case_role",
                "expected_case_status",
                "wall_seed_count",
                "wall_ready_seed_count",
                "control_wall_leak_observed",
                "case_result_status",
                "direct_weight",
                "pair_bridge_weight",
                "bridge_host_weight",
                "host_clique_weight",
            ],
            max_rows=20,
        ),
        "",
        "## Seed Wall Rows",
        "",
        _markdown_table(
            seed_wall,
            [
                "case_id",
                "start_condition",
                "seed",
                "direct_route_accepted",
                "recovery_route_accepted",
                "wall_seed_ready",
                "wall_seed_status",
                "direct_endpoint_object_sequence",
                "recovery_endpoint_object_sequence",
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
            "This synthetic result explains a mechanism surface. It is not "
            "NanoClustering replay, not a quality/cost comparison, and not a "
            "method or algorithm claim."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_design, graph_edges = _panel_design_and_edges()
    trace = _run_trace(seeds=int(args.seeds), n_iterations=int(args.n_iterations))
    route_results = _route_results(trace)
    seed_wall = _seed_wall_rows(route_results)
    case_summary = _case_summary(seed_wall, seeds=int(args.seeds))
    gates = _gate_matrix(
        trace=trace,
        seed_wall=seed_wall,
        case_summary=case_summary,
        seeds=int(args.seeds),
    )
    summary = _summary(
        output_dir=output_dir,
        trace=trace,
        seed_wall=seed_wall,
        case_summary=case_summary,
        gates=gates,
    )

    _write_csv(panel_design, output_dir / PANEL_DESIGN_CSV)
    _write_csv(graph_edges, output_dir / GRAPH_EDGES_CSV)
    _write_csv(trace, output_dir / TRACE_ROWS_CSV)
    _write_csv(route_results, output_dir / ROUTE_RESULT_ROWS_CSV)
    _write_csv(seed_wall, output_dir / SEED_WALL_ROWS_CSV)
    _write_csv(case_summary, output_dir / CASE_SUMMARY_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "variable_pair_synthetic_g4_9_primitive_wall_demo_config.v1",
        "output_dir": str(output_dir),
        "seeds": int(args.seeds),
        "n_iterations": int(args.n_iterations),
        "start_conditions": list(START_CONDITIONS),
        "direct_only_bridge_fractions": list(DIRECT_ONLY_BRIDGE_FRACTIONS),
        "recovery_bridge_fractions": list(RECOVERY_BRIDGE_FRACTIONS),
        "panel_cases": [case.__dict__ for case in PANEL_CASES],
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
        seed_wall=seed_wall,
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
