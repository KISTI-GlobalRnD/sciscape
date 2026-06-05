#!/usr/bin/env python3
"""Execute the first-pass local_pair_014 pathway-probe trace contract.

This runner consumes
``design_leiden_basin_nanoclustering_g4_8_first_pass_014_pathway_probe_contract.py``.
It executes exactly the 16 predeclared route rows: eight positive
``local_pair_014`` rows and eight ``local_pair_005`` boundary-control rows.

The execution kernel is the existing local fractional-edge trace runner, but
the readout is specific to the 014 contract: independent direct-path
availability, recovery after an objective-debt minimum, and boundary-control
leak status are reported separately. This does not promote wall, method, or
quality/cost claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from design_leiden_basin_nanoclustering_g4_8_first_pass_014_pathway_probe_contract import (
    CONTROL_GUARD_ROWS_CSV as CONTRACT_CONTROL_GUARD_ROWS_CSV,
    DEFAULT_OUTPUT_DIR as DEFAULT_CONTRACT_DIR,
    GATE_MATRIX_CSV as CONTRACT_GATE_MATRIX_CSV,
    PAIR_ROWS_CSV as CONTRACT_PAIR_ROWS_CSV,
    ROUTE_PLAN_ROWS_CSV as CONTRACT_ROUTE_PLAN_ROWS_CSV,
)
from run_leiden_basin_nanoclustering_g4_8_scoped_pathway_probe_trace import (
    CLAIM_BOUNDARY as SCOPED_TRACE_CLAIM_BOUNDARY,
    SCHEDULES,
    _route_contract_summary,
    _seed_route_summary,
    _trace_rows,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)
from run_leiden_basin_nanoclustering_symmetric_object_variable_pair_local_ablation import (
    DEFAULT_OUTPUT_DIR as DEFAULT_LOCAL_ABLATION_DIR,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_014_pathway_probe_trace_gamma1e5_20260604"
)

ROUTE_EXECUTION_PLAN_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_pathway_probe_route_execution_plan_rows.csv"
)
TRACE_ROWS_CSV = "nanoclustering_g4_8_first_pass_014_pathway_probe_trace_rows.csv"
SEED_ROUTE_SUMMARY_CSV = (
    "nanoclustering_g4_8_first_pass_014_pathway_probe_trace_seed_route_summary.csv"
)
ROUTE_CONTRACT_SUMMARY_CSV = (
    "nanoclustering_g4_8_first_pass_014_pathway_probe_trace_route_contract_summary.csv"
)
ROUTE_PROBE_RESULT_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_pathway_probe_route_result_rows.csv"
)
ROUTE_PROBE_SUMMARY_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_pathway_probe_route_summary_rows.csv"
)
PAIR_PROBE_RESULT_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_pathway_probe_pair_result_rows.csv"
)
CONTROL_GUARD_RESULT_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_pathway_probe_control_guard_result_rows.csv"
)
GATE_MATRIX_CSV = "nanoclustering_g4_8_first_pass_014_pathway_probe_trace_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_014_pathway_probe_trace_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_014_pathway_probe_trace_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_014_pathway_probe_trace_report.md"

POSITIVE_PAIR_ID = "local_pair_014"
BOUNDARY_PAIR_ID = "local_pair_005"

RUN_STATUS = "executed_nanoclustering_g4_8_first_pass_014_pathway_probe_trace"
ROUTE_EXECUTION_STATUS = "executed_first_pass_014_pathway_probe_local_route_trace"
WALL_PROMOTION_STATUS = "not_promoted_pathway_probe_trace_only"
METHOD_STATUS = "local_pathway_probe_trace_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass local_pair_014 pathway-probe trace only; "
    "executes the 16 predeclared local route rows and reads independent "
    "direct-path availability, recovery-loop shape, and 005 boundary-control "
    "leaks. It does not promote basin walls, evaluate quality/cost value, "
    "replay full NanoClustering, or claim method success."
)

RECOVERY_LOOP_FAMILIES = {
    "first_pass_014_recovery_loop_probe",
    "first_pass_005_boundary_recovery_loop_guard",
}
DIRECT_ONLY_FAMILIES = {
    "first_pass_014_direct_only_target_availability_probe",
    "first_pass_005_boundary_direct_only_guard",
}
SUPPORTED_FAMILIES = RECOVERY_LOOP_FAMILIES | DIRECT_ONLY_FAMILIES

RECOVERY_BRIDGE_FRACTIONS = (1.0, 0.75, 0.50, 0.25, 0.0, 0.25, 0.50, 0.75, 1.0)


def _register_first_pass_014_schedules() -> dict[str, tuple[dict[str, Any], ...]]:
    recovery_schedule = tuple(
        {
            "step_index": index,
            "step_label": f"recovery_loop_bridge_fraction_{fraction:.2f}",
            "direct_edge_weight_fraction": 1.0,
            "bridge_edge_weight_fraction": float(fraction),
            "expected_final_anchor_variant": (
                "original" if float(fraction) == 1.0 else "drop_bridge_edges"
            ),
        }
        for index, fraction in enumerate(RECOVERY_BRIDGE_FRACTIONS, start=1)
    )
    direct_only_schedule = (
        {
            "step_index": 1,
            "step_label": "baseline_source_anchor",
            "direct_edge_weight_fraction": 1.0,
            "bridge_edge_weight_fraction": 1.0,
            "expected_final_anchor_variant": "original",
        },
        {
            "step_index": 2,
            "step_label": "direct_only_bridge_suppressed",
            "direct_edge_weight_fraction": 1.0,
            "bridge_edge_weight_fraction": 0.0,
            "expected_final_anchor_variant": "drop_bridge_edges",
        },
    )
    schedules = {
        "first_pass_014_recovery_loop_probe": recovery_schedule,
        "first_pass_005_boundary_recovery_loop_guard": recovery_schedule,
        "first_pass_014_direct_only_target_availability_probe": direct_only_schedule,
        "first_pass_005_boundary_direct_only_guard": direct_only_schedule,
    }
    SCHEDULES.update(schedules)
    return schedules


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


def _execution_plan(route_plan: pd.DataFrame) -> pd.DataFrame:
    rows = route_plan[route_plan["new_route_execution_required"].map(_as_bool)].copy()
    if rows.empty:
        raise ValueError("No route rows are marked new_route_execution_required.")
    bad_families = sorted(set(rows["planned_route_family"].astype(str)) - SUPPORTED_FAMILIES)
    if bad_families:
        raise ValueError(f"Unsupported planned_route_family values: {bad_families}")
    rows["validation_unit_id"] = rows["route_contract_id"].astype(str)
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["runner_support_status_after_execution"] = "implemented_in_first_pass_014_pathway_probe_runner"
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.reset_index(drop=True)


def _endpoint_object_assignment(row: pd.Series) -> str:
    endpoint = str(row["endpoint_assignment_by_step"])
    pair_id = str(row["local_pair_id"])
    if endpoint == "original_source_anchor":
        return "source_endpoint_object"
    if endpoint == "drop_bridge_target_anchor":
        if pair_id == POSITIVE_PAIR_ID:
            return "exclusive_target_endpoint_object"
        return "boundary_target_endpoint_object_not_positive"
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
                return "exclusive_target_endpoint_object"
            return "boundary_target_endpoint_object_not_positive"
        return "ambiguous_endpoint_object"
    return "other_known_endpoint_object"


def _enrich_trace_rows(trace_rows: pd.DataFrame, execution_plan: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = [
        "route_contract_id",
        "contract_pair_role",
        "counts_as_positive_if_accepted",
        "runner_support_status_after_execution",
    ]
    metadata = execution_plan[metadata_cols].drop_duplicates("route_contract_id")
    rows = trace_rows.merge(metadata, on="route_contract_id", how="left")
    rows["endpoint_object_assignment_by_step"] = rows.apply(_endpoint_object_assignment, axis=1)
    rows["direct_edge_retained_by_step"] = rows["active_direct_edge_weight"].astype(float).gt(0.0)
    rows["bridge_support_suppressed_by_step"] = (
        rows["active_pair_bridge_edge_weight_sum"].astype(float).eq(0.0)
    )
    for column, value in [
        ("route_execution_status", ROUTE_EXECUTION_STATUS),
        ("wall_promotion_status", WALL_PROMOTION_STATUS),
        ("method_status", METHOD_STATUS),
        ("claim_boundary", CLAIM_BOUNDARY),
        ("run_status", RUN_STATUS),
    ]:
        rows[column] = value
    return rows


def _bridge_sequence_matches(ordered: pd.DataFrame, expected: tuple[float, ...]) -> bool:
    observed = tuple(round(float(value), 2) for value in ordered["bridge_edge_weight_fraction"])
    return observed == tuple(round(float(value), 2) for value in expected)


def _first_step(ordered: pd.DataFrame, mask: pd.Series) -> int | None:
    matches = ordered.loc[mask.astype(bool), "step_index"]
    if matches.empty:
        return None
    return int(matches.iloc[0])


def _sequence(values: pd.Series) -> str:
    return " -> ".join(str(value) for value in values.astype(str).tolist())


def _route_probe_results(trace_rows: pd.DataFrame) -> pd.DataFrame:
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
        first = ordered.iloc[0]
        final = ordered.iloc[-1]
        pair_id = str(key_data["local_pair_id"])
        family = str(key_data["planned_route_family"])
        is_positive_pair = pair_id == POSITIVE_PAIR_ID
        is_boundary_pair = pair_id == BOUNDARY_PAIR_ID
        is_direct_only = family in DIRECT_ONLY_FAMILIES
        is_recovery_loop = family in RECOVERY_LOOP_FAMILIES

        endpoint_objects = ordered["endpoint_object_assignment_by_step"].astype(str)
        endpoints = ordered["endpoint_assignment_by_step"].astype(str)
        source_baseline_pass = (
            bool(first["matches_original_anchor"])
            and float(first["bridge_edge_weight_fraction"]) == 1.0
            and float(first["direct_edge_weight_fraction"]) == 1.0
        )
        direct_edge_retained_all_steps = bool(
            ordered["active_direct_edge_weight"].astype(float).gt(0.0).all()
        )
        final_bridge_suppressed = bool(
            float(final["bridge_edge_weight_fraction"]) == 0.0
            and float(final["active_pair_bridge_edge_weight_sum"]) == 0.0
        )
        final_exclusive_target_object = str(
            final["endpoint_object_assignment_by_step"]
        ) == "exclusive_target_endpoint_object"
        final_target_anchor = str(final["endpoint_assignment_by_step"]) == "drop_bridge_target_anchor"
        raw_anchor_ambiguous_step_count = int(
            endpoints.str.startswith("ambiguous_anchor_match").sum()
        )
        unknown_step_count = int(endpoint_objects.eq("unknown_endpoint_object").sum())
        ambiguous_step_count = int(endpoint_objects.eq("ambiguous_endpoint_object").sum())
        support_incompatibility_step_count = int(
            ordered["support_incompatibility_check"].map(_as_bool).sum()
        )
        endpoint_objects_interpretable_all_steps = (
            unknown_step_count == 0 and ambiguous_step_count == 0
        )
        first_exclusive_target_step = _first_step(
            ordered,
            endpoint_objects.eq("exclusive_target_endpoint_object"),
        )
        first_target_anchor_step = _first_step(ordered, endpoints.eq("drop_bridge_target_anchor"))
        max_debt = float(ordered["objective_debt_from_start"].astype(float).max())
        max_recovery = float(ordered["objective_recovery_from_min"].astype(float).max())
        min_objective_idx = ordered["objective_value_by_step"].astype(float).idxmin()
        min_objective_step = int(ordered.loc[min_objective_idx, "step_index"])
        recovery_after_min_rows = ordered[
            ordered["step_index"].astype(int).gt(min_objective_step)
            & ordered["objective_recovery_from_min"].astype(float).gt(0.0)
        ]
        accepted_recovery_after_min = bool(max_debt > 0.0 and not recovery_after_min_rows.empty)
        recovery_loop_schedule_pass = _bridge_sequence_matches(
            ordered,
            RECOVERY_BRIDGE_FRACTIONS,
        )
        direct_only_schedule_pass = _bridge_sequence_matches(ordered, (1.0, 0.0))

        direct_path_accepted = bool(
            is_positive_pair
            and is_direct_only
            and source_baseline_pass
            and direct_edge_retained_all_steps
            and direct_only_schedule_pass
            and final_bridge_suppressed
            and final_target_anchor
            and final_exclusive_target_object
            and endpoint_objects_interpretable_all_steps
            and support_incompatibility_step_count == 0
        )
        recovery_accepted = bool(
            is_positive_pair
            and is_recovery_loop
            and source_baseline_pass
            and direct_edge_retained_all_steps
            and recovery_loop_schedule_pass
            and first_exclusive_target_step is not None
            and accepted_recovery_after_min
            and endpoint_objects_interpretable_all_steps
            and support_incompatibility_step_count == 0
        )
        boundary_positive_leak_observed = bool(
            is_boundary_pair
            and (
                (
                    is_direct_only
                    and source_baseline_pass
                    and direct_edge_retained_all_steps
                    and direct_only_schedule_pass
                    and final_bridge_suppressed
                    and final_exclusive_target_object
                    and support_incompatibility_step_count == 0
                )
                or (
                    is_recovery_loop
                    and source_baseline_pass
                    and direct_edge_retained_all_steps
                    and recovery_loop_schedule_pass
                    and first_exclusive_target_step is not None
                    and accepted_recovery_after_min
                    and support_incompatibility_step_count == 0
                )
            )
        )

        if direct_path_accepted:
            route_probe_outcome_class = "positive_direct_path_seed_accepted"
        elif recovery_accepted:
            route_probe_outcome_class = "positive_recovery_seed_accepted"
        elif boundary_positive_leak_observed:
            route_probe_outcome_class = "boundary_control_positive_leak"
        elif is_boundary_pair and first_target_anchor_step is not None:
            route_probe_outcome_class = "boundary_structural_signal_not_positive"
        elif is_boundary_pair:
            route_probe_outcome_class = "boundary_guard_closed_no_positive_signal"
        elif is_direct_only and final_target_anchor and support_incompatibility_step_count > 0:
            route_probe_outcome_class = "direct_target_reached_with_support_incompatibility"
        elif is_direct_only and final_target_anchor:
            route_probe_outcome_class = "direct_target_reached_but_not_accepted"
        elif is_direct_only:
            route_probe_outcome_class = "direct_target_not_reached"
        elif is_recovery_loop and first_exclusive_target_step is not None and not accepted_recovery_after_min:
            route_probe_outcome_class = "target_reached_without_accepted_recovery"
        elif is_recovery_loop:
            route_probe_outcome_class = "recovery_loop_not_accepted"
        else:
            route_probe_outcome_class = "unsupported_route_probe_outcome"

        rows.append(
            {
                **key_data,
                "route_step_count": int(len(ordered)),
                "source_baseline_pass": bool(source_baseline_pass),
                "direct_edge_retained_all_steps": bool(direct_edge_retained_all_steps),
                "direct_only_schedule_pass": bool(direct_only_schedule_pass),
                "recovery_loop_schedule_pass": bool(recovery_loop_schedule_pass),
                "final_bridge_suppressed": bool(final_bridge_suppressed),
                "final_target_anchor": bool(final_target_anchor),
                "final_exclusive_target_object": bool(final_exclusive_target_object),
                "first_target_anchor_step": first_target_anchor_step,
                "first_exclusive_target_step": first_exclusive_target_step,
                "endpoint_objects_interpretable_all_steps": bool(
                    endpoint_objects_interpretable_all_steps
                ),
                "unknown_step_count": unknown_step_count,
                "ambiguous_step_count": ambiguous_step_count,
                "raw_anchor_ambiguous_step_count": raw_anchor_ambiguous_step_count,
                "support_incompatibility_step_count": support_incompatibility_step_count,
                "min_objective_step": min_objective_step,
                "max_objective_debt_from_start": max_debt,
                "max_objective_recovery_from_min": max_recovery,
                "accepted_recovery_after_min": bool(accepted_recovery_after_min),
                "objective_recovery_step_count_after_min": int(len(recovery_after_min_rows)),
                "bridge_fraction_sequence": ";".join(
                    f"{float(value):.2f}" for value in ordered["bridge_edge_weight_fraction"]
                ),
                "endpoint_assignment_sequence": _sequence(endpoints),
                "endpoint_object_assignment_sequence": _sequence(endpoint_objects),
                "direct_path_accepted_seed": bool(direct_path_accepted),
                "recovery_accepted_seed": bool(recovery_accepted),
                "boundary_positive_leak_observed": bool(boundary_positive_leak_observed),
                "boundary_control_leak_status": (
                    "leak_observed" if boundary_positive_leak_observed else "closed"
                )
                if is_boundary_pair
                else "not_boundary_control",
                "route_probe_outcome_class": route_probe_outcome_class,
                "wall_claim_allowed_after_probe": False,
                "method_claim_allowed_after_probe": False,
                "quality_cost_claim_allowed_after_probe": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _route_probe_summary(route_results: pd.DataFrame) -> pd.DataFrame:
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
    ]
    for keys, group in route_results.groupby(group_cols, sort=False):
        key_data = dict(zip(group_cols, keys, strict=True))
        seed_count = int(group["seed"].nunique())
        direct_accepted = int(group["direct_path_accepted_seed"].map(_as_bool).sum())
        recovery_accepted = int(group["recovery_accepted_seed"].map(_as_bool).sum())
        boundary_leaks = int(group["boundary_positive_leak_observed"].map(_as_bool).sum())
        family = str(key_data["planned_route_family"])
        pair_id = str(key_data["local_pair_id"])
        if pair_id == POSITIVE_PAIR_ID and family in DIRECT_ONLY_FAMILIES:
            if direct_accepted == seed_count and seed_count > 0:
                route_probe_status = "all_seeds_direct_path_accepted"
            elif direct_accepted:
                route_probe_status = "partial_direct_path_accepted"
            else:
                route_probe_status = "no_seed_direct_path_accepted"
        elif pair_id == POSITIVE_PAIR_ID and family in RECOVERY_LOOP_FAMILIES:
            if recovery_accepted == seed_count and seed_count > 0:
                route_probe_status = "all_seeds_recovery_accepted"
            elif recovery_accepted:
                route_probe_status = "partial_recovery_accepted"
            else:
                route_probe_status = "no_seed_recovery_accepted"
        elif boundary_leaks:
            route_probe_status = "boundary_control_leak_observed"
        else:
            route_probe_status = "boundary_control_closed"
        rows.append(
            {
                **key_data,
                "seed_count": seed_count,
                "direct_path_accepted_seed_count": direct_accepted,
                "recovery_accepted_seed_count": recovery_accepted,
                "boundary_positive_leak_seed_count": boundary_leaks,
                "route_probe_outcome_class_counts": _count_dict(
                    group["route_probe_outcome_class"]
                ),
                "route_probe_status": route_probe_status,
                "max_objective_debt_from_start": float(
                    group["max_objective_debt_from_start"].max()
                ),
                "max_objective_recovery_from_min": float(
                    group["max_objective_recovery_from_min"].max()
                ),
                "wall_claim_allowed_after_probe": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _pair_probe_results(pair_rows: pd.DataFrame, route_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair in pair_rows.itertuples(index=False):
        pair_id = str(pair.local_pair_id)
        group = route_results[route_results["local_pair_id"].astype(str).eq(pair_id)]
        direct_group = group[group["planned_route_family"].astype(str).isin(DIRECT_ONLY_FAMILIES)]
        recovery_group = group[group["planned_route_family"].astype(str).isin(RECOVERY_LOOP_FAMILIES)]
        direct_count = int(direct_group["direct_path_accepted_seed"].map(_as_bool).sum())
        recovery_count = int(recovery_group["recovery_accepted_seed"].map(_as_bool).sum())
        boundary_leak_count = int(group["boundary_positive_leak_observed"].map(_as_bool).sum())
        direct_expected = int(len(direct_group))
        recovery_expected = int(len(recovery_group))
        if pair_id == POSITIVE_PAIR_ID:
            if (
                direct_expected > 0
                and recovery_expected > 0
                and direct_count == direct_expected
                and recovery_count == recovery_expected
            ):
                pair_probe_status = "all_direct_and_recovery_seed_routes_accepted_wall_claim_closed"
            elif direct_count == direct_expected and recovery_count < recovery_expected:
                pair_probe_status = "direct_path_accepted_recovery_not_fully_accepted_wall_claim_closed"
            elif direct_count < direct_expected and recovery_count == recovery_expected:
                pair_probe_status = "recovery_accepted_direct_path_not_fully_accepted_wall_claim_closed"
            elif direct_count or recovery_count:
                pair_probe_status = "partial_pathway_probe_acceptance_wall_claim_closed"
            else:
                pair_probe_status = "no_pathway_probe_acceptance_wall_claim_closed"
        elif boundary_leak_count:
            pair_probe_status = "boundary_control_leaked"
        else:
            pair_probe_status = "boundary_control_closed"
        rows.append(
            {
                "local_pair_id": pair_id,
                "branch": str(getattr(pair, "branch", "")),
                "contract_pair_role": str(pair.contract_pair_role),
                "seed_route_result_count": int(len(group)),
                "direct_path_seed_route_count": direct_expected,
                "direct_path_accepted_seed_route_count": direct_count,
                "recovery_seed_route_count": recovery_expected,
                "recovery_accepted_seed_route_count": recovery_count,
                "boundary_positive_leak_seed_route_count": boundary_leak_count,
                "route_probe_outcome_class_counts": _count_dict(
                    group["route_probe_outcome_class"]
                )
                if not group.empty
                else {},
                "pair_probe_status": pair_probe_status,
                "wall_claim_allowed_after_probe": False,
                "method_claim_allowed_after_probe": False,
                "quality_cost_claim_allowed_after_probe": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _control_guard_results(
    control_guards: pd.DataFrame,
    route_results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for control in control_guards.itertuples(index=False):
        route_contract_id = str(control.route_contract_id)
        group = route_results[
            route_results["route_contract_id"].astype(str).eq(route_contract_id)
        ]
        leak_count = int(group["boundary_positive_leak_observed"].map(_as_bool).sum())
        rows.append(
            {
                "control_guard_id": str(control.control_guard_id),
                "route_contract_id": route_contract_id,
                "local_pair_id": str(control.local_pair_id),
                "start_condition": str(control.start_condition),
                "planned_route_family": str(control.planned_route_family),
                "seed_route_result_count": int(len(group)),
                "boundary_positive_leak_seed_count": leak_count,
                "control_guard_result": "leak_observed" if leak_count else "closed",
                "wall_claim_allowed_after_probe": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
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
    execution_plan: pd.DataFrame,
    schedules: dict[str, tuple[dict[str, Any], ...]],
    trace_rows: pd.DataFrame,
    route_results: pd.DataFrame,
    pair_results: pd.DataFrame,
    control_results: pd.DataFrame,
    step_config_count: int,
    seeds: int,
) -> pd.DataFrame:
    required_trace_columns = {
        "route_trace_row_id",
        "route_contract_id",
        "step_index",
        "seed",
        "endpoint_assignment_by_step",
        "endpoint_object_assignment_by_step",
        "active_direct_edge_weight",
        "active_pair_bridge_edge_weight_sum",
        "objective_debt_from_start",
        "objective_recovery_from_min",
        "support_incompatibility_check",
        "direct_edge_retained_by_step",
        "bridge_support_suppressed_by_step",
    }
    expected_step_configs = int(
        sum(len(schedules[str(row.planned_route_family)]) for row in execution_plan.itertuples())
    )
    expected_trace_rows = expected_step_configs * int(seeds)
    positive_pair = pair_results[
        pair_results["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)
    ]
    boundary_pair = pair_results[
        pair_results["local_pair_id"].astype(str).eq(BOUNDARY_PAIR_ID)
    ]
    if positive_pair.empty:
        direct_accepted = recovery_accepted = direct_expected = recovery_expected = 0
    else:
        positive = positive_pair.iloc[0]
        direct_accepted = int(positive["direct_path_accepted_seed_route_count"])
        recovery_accepted = int(positive["recovery_accepted_seed_route_count"])
        direct_expected = int(positive["direct_path_seed_route_count"])
        recovery_expected = int(positive["recovery_seed_route_count"])
    boundary_leaks = int(
        boundary_pair["boundary_positive_leak_seed_route_count"].sum()
    ) if not boundary_pair.empty else -1
    rows = [
        _gate_row(
            "G1_upstream_contract_gates_pass",
            "Did every upstream 014 pathway-probe contract gate pass?",
            _count_dict(contract_gates["gate_status"]),
            "all upstream contract gates pass",
            bool(contract_gates["gate_status"].astype(str).eq("pass").all()),
        ),
        _gate_row(
            "G2_exact_16_route_scope",
            "Was execution restricted to the 16 predeclared route-plan rows?",
            f"execution_plan_rows={len(execution_plan)} executed_route_contracts={trace_rows['route_contract_id'].nunique()}",
            "16 route rows and no extra route contracts",
            len(execution_plan) == 16
            and trace_rows["route_contract_id"].nunique() == 16
            and set(trace_rows["route_contract_id"]) == set(execution_plan["route_contract_id"]),
        ),
        _gate_row(
            "G3_predeclared_schedule_expansion",
            "Were route rows expanded only into the predeclared recovery-loop and direct-only schedules?",
            f"route_step_configs={step_config_count} expected={expected_step_configs}",
            "8 recovery routes * 9 steps plus 8 direct-only routes * 2 steps",
            step_config_count == expected_step_configs,
        ),
        _gate_row(
            "G4_seed_replicates_complete",
            "Did every route-step config run the requested same-seed replicates?",
            f"trace_rows={len(trace_rows)} expected={expected_trace_rows}",
            "route-step configs * requested seeds",
            len(trace_rows) == expected_trace_rows,
        ),
        _gate_row(
            "G5_required_measurements_materialized",
            "Did trace rows include required direct-path, endpoint-object, objective, and support fields?",
            sorted(required_trace_columns & set(trace_rows.columns)),
            "all required trace columns present",
            required_trace_columns.issubset(set(trace_rows.columns)),
        ),
        _gate_row(
            "G6_positive_direct_path_acceptance",
            "Were all 014 direct-only seed-routes accepted?",
            f"accepted={direct_accepted} expected={direct_expected}",
            "all 014 direct-only seed-routes accepted",
            direct_expected > 0 and direct_accepted == direct_expected,
        ),
        _gate_row(
            "G7_positive_recovery_acceptance",
            "Were all 014 recovery-loop seed-routes accepted?",
            f"accepted={recovery_accepted} expected={recovery_expected}",
            "all 014 recovery-loop seed-routes accepted",
            recovery_expected > 0 and recovery_accepted == recovery_expected,
        ),
        _gate_row(
            "G8_boundary_controls_no_positive_leak",
            "Did the 005 boundary controls remain non-positive?",
            {
                "boundary_leak_seed_routes": boundary_leaks,
                "control_guard_results": _count_dict(control_results["control_guard_result"]),
            },
            "zero 005 positive leaks",
            boundary_leaks == 0
            and not control_results.empty
            and bool(control_results["control_guard_result"].astype(str).eq("closed").all()),
        ),
        _gate_row(
            "G9_wall_method_quality_claims_closed",
            "Are wall, method, quality/cost, and full-replay claims explicitly closed?",
            CLAIM_BOUNDARY,
            "all promotion flags false",
            bool(route_results["wall_claim_allowed_after_probe"].eq(False).all())
            and bool(pair_results["wall_claim_allowed_after_probe"].eq(False).all())
            and bool(route_results["method_claim_allowed_after_probe"].eq(False).all())
            and bool(route_results["quality_cost_claim_allowed_after_probe"].eq(False).all()),
        ),
    ]
    return pd.DataFrame(rows)


def _summary(
    *,
    contract_dir: Path,
    local_ablation_dir: Path,
    output_dir: Path,
    execution_plan: pd.DataFrame,
    trace_rows: pd.DataFrame,
    seed_summary: pd.DataFrame,
    route_contract_summary: pd.DataFrame,
    route_results: pd.DataFrame,
    route_probe_summary: pd.DataFrame,
    pair_results: pd.DataFrame,
    control_results: pd.DataFrame,
    gates: pd.DataFrame,
    step_config_count: int,
    candidate_pair_count: int,
    seeds: int,
) -> dict[str, Any]:
    positive_pair = pair_results[
        pair_results["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)
    ]
    boundary_pair = pair_results[
        pair_results["local_pair_id"].astype(str).eq(BOUNDARY_PAIR_ID)
    ]
    positive_status = (
        str(positive_pair.iloc[0]["pair_probe_status"])
        if not positive_pair.empty
        else "missing_positive_pair_result"
    )
    boundary_status = (
        str(boundary_pair.iloc[0]["pair_probe_status"])
        if not boundary_pair.empty
        else "missing_boundary_pair_result"
    )
    failed_gates = gates.loc[
        ~gates["gate_status"].astype(str).eq("pass"),
        "gate_id",
    ].tolist()
    if {"G6_positive_direct_path_acceptance", "G7_positive_recovery_acceptance"} & set(
        failed_gates
    ):
        recommended_next_gate = (
            "Inspect the rejected direct-only or recovery-loop seed routes before "
            "promoting any stronger wall/pathway audit."
        )
    else:
        recommended_next_gate = (
            "Run a separate wall-evidence audit on accepted 014 direct/recovery "
            "routes while keeping 005 as the boundary false-positive guard."
        )
    return {
        "schema": "nanoclustering_g4_8_first_pass_014_pathway_probe_trace_summary.v1",
        "status": RUN_STATUS,
        "contract_dir": str(contract_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "requested_seeds": int(seeds),
        "candidate_pair_count_from_trace": int(candidate_pair_count),
        "route_execution_plan_row_count": int(len(execution_plan)),
        "route_step_config_count": int(step_config_count),
        "trace_row_count": int(len(trace_rows)),
        "seed_route_summary_count": int(len(seed_summary)),
        "route_contract_summary_count": int(len(route_contract_summary)),
        "route_probe_result_count": int(len(route_results)),
        "route_probe_summary_count": int(len(route_probe_summary)),
        "pair_probe_result_count": int(len(pair_results)),
        "control_guard_result_count": int(len(control_results)),
        "execution_contract_pair_role_counts": _count_dict(execution_plan["contract_pair_role"]),
        "execution_route_family_counts": _count_dict(execution_plan["planned_route_family"]),
        "route_probe_outcome_class_counts": _count_dict(route_results["route_probe_outcome_class"]),
        "route_probe_status_counts": _count_dict(route_probe_summary["route_probe_status"]),
        "pair_probe_status_counts": _count_dict(pair_results["pair_probe_status"]),
        "control_guard_result_counts": _count_dict(control_results["control_guard_result"]),
        "positive_pair_probe_status": positive_status,
        "boundary_pair_probe_status": boundary_status,
        "positive_pair_results": positive_pair.to_dict("records"),
        "boundary_pair_results": boundary_pair.to_dict("records"),
        "wall_claim_ready_pairs": [],
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": failed_gates,
        "interpretation": (
            "The 16-row 014 pathway-probe contract was executed as local "
            "fractional-edge traces. Direct-path availability, recovery-loop "
            "acceptance, and 005 boundary leaks are now materialized, but wall "
            "and method claims remain closed."
        ),
        "recommended_next_gate": recommended_next_gate,
        "claim_boundary": CLAIM_BOUNDARY,
        "scoped_trace_claim_boundary_used_for_execution_kernel": SCOPED_TRACE_CLAIM_BOUNDARY,
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
    pair_results: pd.DataFrame,
    route_probe_summary: pd.DataFrame,
    route_results: pd.DataFrame,
    control_results: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass 014 Pathway-Probe Trace",
        "",
        f"- status: `{summary['status']}`",
        f"- route_execution_plan_row_count: {summary['route_execution_plan_row_count']}",
        f"- route_step_config_count: {summary['route_step_config_count']}",
        f"- trace_row_count: {summary['trace_row_count']}",
        f"- route_probe_outcome_class_counts: {summary['route_probe_outcome_class_counts']}",
        f"- route_probe_status_counts: {summary['route_probe_status_counts']}",
        f"- pair_probe_status_counts: {summary['pair_probe_status_counts']}",
        f"- control_guard_result_counts: {summary['control_guard_result_counts']}",
        f"- positive_pair_probe_status: `{summary['positive_pair_probe_status']}`",
        f"- boundary_pair_probe_status: `{summary['boundary_pair_probe_status']}`",
        f"- wall_claim_ready_pairs: {summary['wall_claim_ready_pairs']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Pair Results",
        "",
        _markdown_table(
            pair_results.sort_values(["contract_pair_role", "local_pair_id"], kind="mergesort"),
            [
                "local_pair_id",
                "contract_pair_role",
                "seed_route_result_count",
                "direct_path_accepted_seed_route_count",
                "recovery_accepted_seed_route_count",
                "boundary_positive_leak_seed_route_count",
                "pair_probe_status",
            ],
            max_rows=10,
        ),
        "",
        "## Route Summary",
        "",
        _markdown_table(
            route_probe_summary.sort_values(
                ["local_pair_id", "start_condition", "planned_route_family"],
                kind="mergesort",
            ),
            [
                "local_pair_id",
                "start_condition",
                "planned_route_family",
                "seed_count",
                "direct_path_accepted_seed_count",
                "recovery_accepted_seed_count",
                "boundary_positive_leak_seed_count",
                "route_probe_status",
                "max_objective_debt_from_start",
                "max_objective_recovery_from_min",
            ],
            max_rows=40,
        ),
        "",
        "## Control Guards",
        "",
        _markdown_table(
            control_results.sort_values(
                ["local_pair_id", "start_condition", "planned_route_family"],
                kind="mergesort",
            ),
            [
                "local_pair_id",
                "start_condition",
                "planned_route_family",
                "seed_route_result_count",
                "boundary_positive_leak_seed_count",
                "control_guard_result",
            ],
            max_rows=20,
        ),
        "",
        "## Seed Route Results",
        "",
        _markdown_table(
            route_results.sort_values(
                ["local_pair_id", "planned_route_family", "start_condition", "seed"],
                kind="mergesort",
            ),
            [
                "local_pair_id",
                "planned_route_family",
                "start_condition",
                "seed",
                "source_baseline_pass",
                "direct_edge_retained_all_steps",
                "final_exclusive_target_object",
                "accepted_recovery_after_min",
                "direct_path_accepted_seed",
                "recovery_accepted_seed",
                "boundary_positive_leak_observed",
                "route_probe_outcome_class",
            ],
            max_rows=80,
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
            "This run executes the predeclared pathway-probe contract only. It "
            "is not wall evidence and not method evidence. Any wall language must "
            "wait for a separate wall-evidence audit over the accepted direct and "
            "recovery routes, with 005 retained as the boundary guard."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    schedules = _register_first_pass_014_schedules()
    contract_dir = Path(args.contract_dir)
    local_ablation_dir = Path(args.local_ablation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contract_route_plan = _read_csv(contract_dir / CONTRACT_ROUTE_PLAN_ROWS_CSV)
    contract_pair_rows = _read_csv(contract_dir / CONTRACT_PAIR_ROWS_CSV)
    contract_control_guards = _read_csv(contract_dir / CONTRACT_CONTROL_GUARD_ROWS_CSV)
    contract_gates = _read_csv(contract_dir / CONTRACT_GATE_MATRIX_CSV)

    execution_plan = _execution_plan(contract_route_plan)
    trace_rows, step_config_count, candidate_pair_count = _trace_rows(
        route_plan=execution_plan,
        contract_dir=contract_dir,
        local_ablation_dir=local_ablation_dir,
        gamma=float(args.gamma),
        seeds=int(args.seeds),
        n_iterations=int(args.n_iterations),
        edge_chunk_size=int(args.edge_chunk_size),
    )
    trace_rows = _enrich_trace_rows(trace_rows, execution_plan)
    seed_summary = _seed_route_summary(trace_rows)
    route_contract_summary = _route_contract_summary(seed_summary)
    for frame in (seed_summary, route_contract_summary):
        frame["route_execution_status"] = ROUTE_EXECUTION_STATUS
        frame["wall_promotion_status"] = WALL_PROMOTION_STATUS
        frame["method_status"] = METHOD_STATUS
        frame["claim_boundary"] = CLAIM_BOUNDARY
        frame["run_status"] = RUN_STATUS

    route_results = _route_probe_results(trace_rows)
    route_probe_summary = _route_probe_summary(route_results)
    pair_results = _pair_probe_results(contract_pair_rows, route_results)
    control_results = _control_guard_results(contract_control_guards, route_results)
    gates = _gate_matrix(
        contract_gates=contract_gates,
        execution_plan=execution_plan,
        schedules=schedules,
        trace_rows=trace_rows,
        route_results=route_results,
        pair_results=pair_results,
        control_results=control_results,
        step_config_count=step_config_count,
        seeds=int(args.seeds),
    )
    summary = _summary(
        contract_dir=contract_dir,
        local_ablation_dir=local_ablation_dir,
        output_dir=output_dir,
        execution_plan=execution_plan,
        trace_rows=trace_rows,
        seed_summary=seed_summary,
        route_contract_summary=route_contract_summary,
        route_results=route_results,
        route_probe_summary=route_probe_summary,
        pair_results=pair_results,
        control_results=control_results,
        gates=gates,
        step_config_count=step_config_count,
        candidate_pair_count=candidate_pair_count,
        seeds=int(args.seeds),
    )

    _write_csv(execution_plan, output_dir / ROUTE_EXECUTION_PLAN_ROWS_CSV)
    _write_csv(trace_rows, output_dir / TRACE_ROWS_CSV)
    _write_csv(seed_summary, output_dir / SEED_ROUTE_SUMMARY_CSV)
    _write_csv(route_contract_summary, output_dir / ROUTE_CONTRACT_SUMMARY_CSV)
    _write_csv(route_results, output_dir / ROUTE_PROBE_RESULT_ROWS_CSV)
    _write_csv(route_probe_summary, output_dir / ROUTE_PROBE_SUMMARY_ROWS_CSV)
    _write_csv(pair_results, output_dir / PAIR_PROBE_RESULT_ROWS_CSV)
    _write_csv(control_results, output_dir / CONTROL_GUARD_RESULT_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_014_pathway_probe_trace_config.v1",
        "contract_dir": str(contract_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "gamma": float(args.gamma),
        "seeds": int(args.seeds),
        "n_iterations": int(args.n_iterations),
        "edge_chunk_size": int(args.edge_chunk_size),
        "route_schedules": schedules,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        pair_results=pair_results,
        route_probe_summary=route_probe_summary,
        route_results=route_results,
        control_results=control_results,
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
    parser.add_argument("--edge-chunk-size", type=int, default=5_000_000)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
