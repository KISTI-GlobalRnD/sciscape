#!/usr/bin/env python3
"""Execute the first-pass local_pair_016 object-wall transfer contract.

This runner consumes
``design_leiden_basin_nanoclustering_g4_8_first_pass_016_object_wall_transfer_contract.py``.
It executes exactly the 14 predeclared route rows: six ``local_pair_016``
positive-transfer rows and eight ``local_pair_005`` boundary-control rows.

The execution kernel is the existing local fractional-edge trace runner. The
readout is specific to the transfer contract: it records direct-only target
availability, recovery-loop shape, typed transient assignments, object-identity
transfer status, and 005 boundary-control leak status. This does not promote a
pathway label, basin wall, method, full replay, or quality/cost claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from design_leiden_basin_nanoclustering_g4_8_first_pass_016_object_wall_transfer_contract import (
    BOUNDARY_GUARD_ROWS_CSV as CONTRACT_BOUNDARY_GUARD_ROWS_CSV,
    DEFAULT_OUTPUT_DIR as DEFAULT_CONTRACT_DIR,
    GATE_MATRIX_CSV as CONTRACT_GATE_MATRIX_CSV,
    PAIR_ROWS_CSV as CONTRACT_PAIR_ROWS_CSV,
    ROUTE_PLAN_ROWS_CSV as CONTRACT_ROUTE_PLAN_ROWS_CSV,
)
from design_leiden_basin_nanoclustering_g4_8_first_pass_016_transient_persistence_contract import (
    TARGET_SIGNATURE_ID,
    TRANSIENT_SIGNATURE_ID,
)
from run_leiden_basin_nanoclustering_g4_8_scoped_pathway_probe_trace import (
    ANCHOR_VARIANT_TO_ASSIGNMENT,
    CLAIM_BOUNDARY as SCOPED_TRACE_CLAIM_BOUNDARY,
    EXPECTED_FINAL_ASSIGNMENT,
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
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace_gamma1e5_20260607"
)

ROUTE_EXECUTION_PLAN_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_route_execution_plan_rows.csv"
)
TRACE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace_rows.csv"
)
SEED_ROUTE_SUMMARY_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace_seed_route_summary.csv"
)
ROUTE_CONTRACT_SUMMARY_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace_route_contract_summary.csv"
)
ROUTE_TRANSFER_RESULT_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_route_result_rows.csv"
)
ROUTE_TRANSFER_SUMMARY_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_route_summary_rows.csv"
)
PAIR_TRANSFER_RESULT_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_pair_result_rows.csv"
)
BOUNDARY_GUARD_RESULT_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_boundary_guard_result_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace_gate_matrix.csv"
)
SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace_summary.json"
)
CONFIG_JSON = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace_config.json"
)
REPORT_MD = "nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace_report.md"

POSITIVE_PAIR_ID = "local_pair_016"
BOUNDARY_PAIR_ID = "local_pair_005"

RUN_STATUS = "executed_nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace"
ROUTE_EXECUTION_STATUS = "executed_first_pass_016_object_wall_transfer_local_route_trace"
WALL_PROMOTION_STATUS = "not_promoted_016_object_wall_transfer_trace_only"
METHOD_STATUS = "object_wall_transfer_trace_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass local_pair_016 object-wall transfer trace "
    "only; executes the 14 predeclared local route rows and records direct-only "
    "target availability, recovery-loop shape, typed transient assignments, "
    "object-identity transfer status, and 005 boundary-control leaks. It does "
    "not promote pathway labels, basin walls, evaluate quality/cost value, "
    "replay full NanoClustering, or claim method success."
)

RECOVERY_LOOP_FAMILIES = {
    "first_pass_016_recovery_loop_probe",
    "first_pass_005_boundary_recovery_loop_guard",
}
DIRECT_ONLY_FAMILIES = {
    "first_pass_016_direct_only_target_availability_probe",
    "first_pass_005_boundary_direct_only_guard",
}
SUPPORTED_FAMILIES = RECOVERY_LOOP_FAMILIES | DIRECT_ONLY_FAMILIES
RECOVERY_BRIDGE_FRACTIONS = (1.0, 0.75, 0.50, 0.25, 0.0, 0.25, 0.50, 0.75, 1.0)


def _register_first_pass_016_transfer_schedules() -> dict[str, tuple[dict[str, Any], ...]]:
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
        "first_pass_016_recovery_loop_probe": recovery_schedule,
        "first_pass_005_boundary_recovery_loop_guard": recovery_schedule,
        "first_pass_016_direct_only_target_availability_probe": direct_only_schedule,
        "first_pass_005_boundary_direct_only_guard": direct_only_schedule,
    }
    SCHEDULES.update(schedules)
    for family, steps in schedules.items():
        final_variant = str(steps[-1]["expected_final_anchor_variant"])
        EXPECTED_FINAL_ASSIGNMENT[family] = ANCHOR_VARIANT_TO_ASSIGNMENT[final_variant]
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
    rows["runner_support_status_after_execution"] = (
        "implemented_in_first_pass_016_object_wall_transfer_runner"
    )
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values(
        ["contract_pair_role", "local_pair_id", "start_condition", "route_family_order"],
        kind="mergesort",
    ).reset_index(drop=True)


def _typed_transient_assignment(row: pd.Series) -> str:
    signature_id = str(row["result_endpoint_signature_id"])
    endpoint = str(row["endpoint_assignment_by_step"])
    if signature_id == TRANSIENT_SIGNATURE_ID:
        return "pathway_intermediate"
    if endpoint == "unknown_new_endpoint":
        return "object_identity_blocker" if _as_bool(row.get("support_incompatibility_check", False)) else "unknown"
    if endpoint.startswith("ambiguous_anchor_match"):
        return "object_identity_blocker"
    return "not_transient_known_anchor"


def _object_identity_transfer_status(row: pd.Series) -> str:
    typed = str(row["typed_transient_assignment_by_step"])
    endpoint = str(row["endpoint_assignment_by_step"])
    pair_id = str(row["local_pair_id"])
    signature_id = str(row["result_endpoint_signature_id"])
    if typed == "pathway_intermediate":
        return "typed_pathway_intermediate_object_identity_unresolved"
    if typed in {"object_identity_blocker", "unknown"}:
        return f"{typed}_object_identity_unresolved"
    if pair_id == BOUNDARY_PAIR_ID and typed == "boundary_leak":
        return "boundary_target_anchor_not_positive"
    if endpoint == "original_source_anchor":
        return "source_anchor_proxy"
    if endpoint == "drop_bridge_target_anchor" or signature_id == TARGET_SIGNATURE_ID:
        if pair_id == POSITIVE_PAIR_ID:
            return "target_anchor_proxy_object_identity_unresolved"
        return "boundary_target_anchor_not_positive"
    if endpoint == "drop_direct_guard_anchor":
        return "drop_direct_guard_proxy"
    if endpoint == "drop_both_guard_anchor":
        return "drop_both_guard_proxy"
    return "known_anchor_proxy_object_identity_unresolved"


def _endpoint_object_assignment(row: pd.Series) -> str:
    status = str(row["object_identity_transfer_status"])
    if status == "source_anchor_proxy":
        return "source_endpoint_object_proxy"
    if status == "target_anchor_proxy_object_identity_unresolved":
        return "target_endpoint_object_proxy_without_endpoint_identity"
    if status == "boundary_target_anchor_not_positive":
        return "boundary_target_endpoint_object_not_positive"
    if status == "typed_pathway_intermediate_object_identity_unresolved":
        return "typed_transient_pathway_intermediate"
    if "object_identity_unresolved" in status:
        return "object_identity_unresolved_endpoint_object"
    if status.endswith("_guard_proxy"):
        return status.replace("_proxy", "_endpoint_object_proxy")
    return "known_endpoint_object_proxy"


def _enrich_trace_rows(trace_rows: pd.DataFrame, execution_plan: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = [
        "route_contract_id",
        "contract_pair_role",
        "counts_as_positive_if_accepted",
        "runner_support_status_after_execution",
    ]
    metadata = execution_plan[metadata_cols].drop_duplicates("route_contract_id")
    rows = trace_rows.merge(metadata, on="route_contract_id", how="left")
    rows["typed_transient_assignment_by_step"] = rows.apply(
        _typed_transient_assignment,
        axis=1,
    )
    rows["object_identity_transfer_status"] = rows.apply(
        _object_identity_transfer_status,
        axis=1,
    )
    rows["endpoint_object_assignment_by_step"] = rows.apply(
        _endpoint_object_assignment,
        axis=1,
    )
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


def _route_transfer_results(trace_rows: pd.DataFrame) -> pd.DataFrame:
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

        endpoints = ordered["endpoint_assignment_by_step"].astype(str)
        endpoint_objects = ordered["endpoint_object_assignment_by_step"].astype(str)
        typed_assignments = ordered["typed_transient_assignment_by_step"].astype(str)
        identity_status = ordered["object_identity_transfer_status"].astype(str)
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
        direct_only_schedule_pass = _bridge_sequence_matches(ordered, (1.0, 0.0))
        recovery_loop_schedule_pass = _bridge_sequence_matches(
            ordered,
            RECOVERY_BRIDGE_FRACTIONS,
        )
        final_target_anchor_proxy = str(
            final["object_identity_transfer_status"]
        ) == "target_anchor_proxy_object_identity_unresolved"
        final_boundary_target = str(
            final["object_identity_transfer_status"]
        ) == "boundary_target_anchor_not_positive"
        first_target_anchor_proxy_step = _first_step(
            ordered,
            identity_status.eq("target_anchor_proxy_object_identity_unresolved"),
        )
        first_typed_transient_step = _first_step(
            ordered,
            typed_assignments.eq("pathway_intermediate"),
        )
        typed_transient_step_count = int(typed_assignments.eq("pathway_intermediate").sum())
        object_identity_blocker_step_count = int(
            typed_assignments.isin(["object_identity_blocker", "unknown"]).sum()
        )
        boundary_leak_step_count = int(typed_assignments.eq("boundary_leak").sum())
        support_incompatibility_step_count = int(
            ordered["support_incompatibility_check"].map(_as_bool).sum()
        )
        untyped_transient_step_count = int(
            ordered["endpoint_assignment_by_step"].astype(str).eq("unknown_new_endpoint").sum()
            - typed_assignments.isin(["pathway_intermediate", "object_identity_blocker", "unknown"]).sum()
        )
        max_debt = float(ordered["objective_debt_from_start"].astype(float).max())
        max_recovery = float(ordered["objective_recovery_from_min"].astype(float).max())
        min_objective_idx = ordered["objective_value_by_step"].astype(float).idxmin()
        min_objective_step = int(ordered.loc[min_objective_idx, "step_index"])
        recovery_after_min_rows = ordered[
            ordered["step_index"].astype(int).gt(min_objective_step)
            & ordered["objective_recovery_from_min"].astype(float).gt(0.0)
        ]
        accepted_recovery_after_min = bool(max_debt > 0.0 and not recovery_after_min_rows.empty)
        direct_target_available_seed = bool(
            is_positive_pair
            and is_direct_only
            and source_baseline_pass
            and direct_edge_retained_all_steps
            and direct_only_schedule_pass
            and final_bridge_suppressed
            and final_target_anchor_proxy
            and support_incompatibility_step_count == 0
        )
        direct_object_identity_block_seed = bool(
            is_positive_pair
            and is_direct_only
            and not direct_target_available_seed
            and object_identity_blocker_step_count > 0
        )
        recovery_target_with_recovery_seed = bool(
            is_positive_pair
            and is_recovery_loop
            and source_baseline_pass
            and direct_edge_retained_all_steps
            and recovery_loop_schedule_pass
            and first_target_anchor_proxy_step is not None
            and accepted_recovery_after_min
            and support_incompatibility_step_count == 0
        )
        recovery_typed_transient_block_seed = bool(
            is_positive_pair
            and is_recovery_loop
            and first_typed_transient_step is not None
            and not recovery_target_with_recovery_seed
        )
        boundary_positive_leak_observed = bool(
            is_boundary_pair
            and (
                boundary_leak_step_count > 0
            )
        )

        if direct_target_available_seed:
            route_transfer_outcome_class = "direct_target_anchor_available_identity_unresolved"
        elif direct_object_identity_block_seed:
            route_transfer_outcome_class = "direct_object_identity_blocker"
        elif recovery_target_with_recovery_seed:
            route_transfer_outcome_class = "recovery_target_anchor_with_recovery_identity_unresolved"
        elif recovery_typed_transient_block_seed:
            route_transfer_outcome_class = "recovery_typed_transient_block"
        elif boundary_positive_leak_observed:
            route_transfer_outcome_class = "boundary_control_positive_leak"
        elif is_boundary_pair and final_boundary_target:
            route_transfer_outcome_class = "boundary_structural_target_not_positive"
        elif is_boundary_pair:
            route_transfer_outcome_class = "boundary_guard_closed_no_positive_signal"
        elif object_identity_blocker_step_count:
            route_transfer_outcome_class = "object_identity_blocked"
        elif is_direct_only:
            route_transfer_outcome_class = "direct_transfer_not_observed"
        elif is_recovery_loop:
            route_transfer_outcome_class = "recovery_transfer_not_observed"
        else:
            route_transfer_outcome_class = "unsupported_route_transfer_outcome"

        rows.append(
            {
                **key_data,
                "route_step_count": int(len(ordered)),
                "source_baseline_pass": bool(source_baseline_pass),
                "direct_edge_retained_all_steps": bool(direct_edge_retained_all_steps),
                "direct_only_schedule_pass": bool(direct_only_schedule_pass),
                "recovery_loop_schedule_pass": bool(recovery_loop_schedule_pass),
                "final_bridge_suppressed": bool(final_bridge_suppressed),
                "final_target_anchor_proxy": bool(final_target_anchor_proxy),
                "final_boundary_target_anchor": bool(final_boundary_target),
                "first_target_anchor_proxy_step": first_target_anchor_proxy_step,
                "first_typed_transient_step": first_typed_transient_step,
                "typed_transient_step_count": typed_transient_step_count,
                "object_identity_blocker_step_count": object_identity_blocker_step_count,
                "boundary_leak_step_count": boundary_leak_step_count,
                "untyped_transient_step_count": max(0, untyped_transient_step_count),
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
                "typed_transient_assignment_sequence": _sequence(typed_assignments),
                "object_identity_transfer_status_sequence": _sequence(identity_status),
                "direct_target_available_seed": bool(direct_target_available_seed),
                "direct_object_identity_block_seed": bool(direct_object_identity_block_seed),
                "recovery_target_with_recovery_seed": bool(recovery_target_with_recovery_seed),
                "recovery_typed_transient_block_seed": bool(recovery_typed_transient_block_seed),
                "boundary_positive_leak_observed": bool(boundary_positive_leak_observed),
                "boundary_control_leak_status": (
                    "leak_observed" if boundary_positive_leak_observed else "closed"
                )
                if is_boundary_pair
                else "not_boundary_control",
                "route_transfer_outcome_class": route_transfer_outcome_class,
                "wall_claim_allowed_after_trace": False,
                "pathway_claim_allowed_after_trace": False,
                "method_claim_allowed_after_trace": False,
                "quality_cost_claim_allowed_after_trace": False,
                "full_replay_claim_allowed_after_trace": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _route_transfer_summary(route_results: pd.DataFrame) -> pd.DataFrame:
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
        direct_available = int(group["direct_target_available_seed"].map(_as_bool).sum())
        direct_block = int(group["direct_object_identity_block_seed"].map(_as_bool).sum())
        recovery_available = int(
            group["recovery_target_with_recovery_seed"].map(_as_bool).sum()
        )
        recovery_block = int(
            group["recovery_typed_transient_block_seed"].map(_as_bool).sum()
        )
        boundary_leaks = int(group["boundary_positive_leak_observed"].map(_as_bool).sum())
        pair_id = str(key_data["local_pair_id"])
        family = str(key_data["planned_route_family"])
        if pair_id == POSITIVE_PAIR_ID and family in DIRECT_ONLY_FAMILIES:
            if direct_available == seed_count and seed_count > 0:
                route_transfer_status = "all_seeds_direct_target_available_identity_unresolved"
            elif direct_block == seed_count and seed_count > 0:
                route_transfer_status = "all_seeds_direct_object_identity_blocked"
            elif direct_available or direct_block:
                route_transfer_status = "mixed_direct_transfer_readout"
            else:
                route_transfer_status = "direct_transfer_not_observed"
        elif pair_id == POSITIVE_PAIR_ID and family in RECOVERY_LOOP_FAMILIES:
            if recovery_available == seed_count and seed_count > 0:
                route_transfer_status = "all_seeds_recovery_target_with_recovery_identity_unresolved"
            elif recovery_block == seed_count and seed_count > 0:
                route_transfer_status = "all_seeds_recovery_typed_transient_block"
            elif recovery_available or recovery_block:
                route_transfer_status = "mixed_recovery_transfer_readout"
            else:
                route_transfer_status = "recovery_transfer_not_observed"
        elif boundary_leaks:
            route_transfer_status = "boundary_control_leak_observed"
        else:
            route_transfer_status = "boundary_control_closed"
        rows.append(
            {
                **key_data,
                "seed_count": seed_count,
                "direct_target_available_seed_count": direct_available,
                "direct_object_identity_block_seed_count": direct_block,
                "recovery_target_with_recovery_seed_count": recovery_available,
                "recovery_typed_transient_block_seed_count": recovery_block,
                "boundary_positive_leak_seed_count": boundary_leaks,
                "route_transfer_outcome_class_counts": _count_dict(
                    group["route_transfer_outcome_class"]
                ),
                "route_transfer_status": route_transfer_status,
                "typed_transient_seed_route_count": int(
                    group["typed_transient_step_count"].astype(int).gt(0).sum()
                ),
                "untyped_transient_seed_route_count": int(
                    group["untyped_transient_step_count"].astype(int).gt(0).sum()
                ),
                "max_objective_debt_from_start": float(
                    group["max_objective_debt_from_start"].max()
                ),
                "max_objective_recovery_from_min": float(
                    group["max_objective_recovery_from_min"].max()
                ),
                "wall_claim_allowed_after_trace": False,
                "pathway_claim_allowed_after_trace": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _pair_transfer_results(pair_rows: pd.DataFrame, route_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair in pair_rows.itertuples(index=False):
        pair_id = str(pair.local_pair_id)
        group = route_results[route_results["local_pair_id"].astype(str).eq(pair_id)]
        direct_group = group[group["planned_route_family"].astype(str).isin(DIRECT_ONLY_FAMILIES)]
        recovery_group = group[group["planned_route_family"].astype(str).isin(RECOVERY_LOOP_FAMILIES)]
        direct_available = int(direct_group["direct_target_available_seed"].map(_as_bool).sum())
        direct_block = int(direct_group["direct_object_identity_block_seed"].map(_as_bool).sum())
        recovery_available = int(
            recovery_group["recovery_target_with_recovery_seed"].map(_as_bool).sum()
        )
        recovery_block = int(
            recovery_group["recovery_typed_transient_block_seed"].map(_as_bool).sum()
        )
        boundary_leak_count = int(group["boundary_positive_leak_observed"].map(_as_bool).sum())
        direct_expected = int(len(direct_group))
        recovery_expected = int(len(recovery_group))
        untyped_count = int(group["untyped_transient_step_count"].astype(int).gt(0).sum())
        if pair_id == POSITIVE_PAIR_ID:
            if untyped_count:
                pair_transfer_status = "untyped_transfer_states_block_wall_claim"
            elif (
                direct_available == direct_expected
                and recovery_available == recovery_expected
                and direct_expected > 0
                and recovery_expected > 0
            ):
                pair_transfer_status = (
                    "all_direct_and_recovery_target_shape_observed_identity_unresolved_wall_claim_closed"
                )
            elif direct_available == direct_expected and recovery_block:
                pair_transfer_status = "direct_target_available_recovery_typed_block_wall_claim_closed"
            elif direct_block and recovery_available == recovery_expected:
                pair_transfer_status = "direct_identity_block_recovery_target_available_wall_claim_closed"
            elif direct_available or recovery_available or direct_block or recovery_block:
                pair_transfer_status = "partial_transfer_readout_wall_claim_closed"
            else:
                pair_transfer_status = "no_transfer_readout_wall_claim_closed"
        elif boundary_leak_count:
            pair_transfer_status = "boundary_control_leaked"
        elif pair_id == BOUNDARY_PAIR_ID:
            pair_transfer_status = "boundary_control_closed"
        else:
            pair_transfer_status = "source_vocabulary_context_not_executed"
        rows.append(
            {
                "local_pair_id": pair_id,
                "branch": str(getattr(pair, "branch", "")),
                "contract_pair_role": str(pair.contract_pair_role),
                "seed_route_result_count": int(len(group)),
                "direct_seed_route_count": direct_expected,
                "direct_target_available_seed_route_count": direct_available,
                "direct_object_identity_block_seed_route_count": direct_block,
                "recovery_seed_route_count": recovery_expected,
                "recovery_target_with_recovery_seed_route_count": recovery_available,
                "recovery_typed_transient_block_seed_route_count": recovery_block,
                "boundary_positive_leak_seed_route_count": boundary_leak_count,
                "untyped_transfer_seed_route_count": untyped_count,
                "route_transfer_outcome_class_counts": _count_dict(
                    group["route_transfer_outcome_class"]
                )
                if not group.empty
                else {},
                "pair_transfer_status": pair_transfer_status,
                "wall_claim_allowed_after_trace": False,
                "pathway_claim_allowed_after_trace": False,
                "method_claim_allowed_after_trace": False,
                "quality_cost_claim_allowed_after_trace": False,
                "full_replay_claim_allowed_after_trace": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _boundary_guard_results(
    boundary_guards: pd.DataFrame,
    route_results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for guard in boundary_guards.itertuples(index=False):
        route_contract_id = str(guard.route_contract_id)
        group = route_results[
            route_results["route_contract_id"].astype(str).eq(route_contract_id)
        ]
        leak_count = int(group["boundary_positive_leak_observed"].map(_as_bool).sum())
        rows.append(
            {
                "boundary_guard_id": str(guard.boundary_guard_id),
                "route_contract_id": route_contract_id,
                "local_pair_id": str(guard.local_pair_id),
                "start_condition": str(guard.start_condition),
                "planned_route_family": str(guard.planned_route_family),
                "seed_route_result_count": int(len(group)),
                "boundary_positive_leak_seed_count": leak_count,
                "boundary_guard_result": "leak_observed" if leak_count else "closed",
                "wall_claim_allowed_after_trace": False,
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
    boundary_results: pd.DataFrame,
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
        "typed_transient_assignment_by_step",
        "object_identity_transfer_status",
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
    positive_routes = route_results[
        route_results["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)
    ]
    direct_positive = positive_routes[
        positive_routes["planned_route_family"].astype(str).isin(DIRECT_ONLY_FAMILIES)
    ]
    recovery_positive = positive_routes[
        positive_routes["planned_route_family"].astype(str).isin(RECOVERY_LOOP_FAMILIES)
    ]
    typed_classes = set(trace_rows["typed_transient_assignment_by_step"].astype(str))
    boundary_leaks = int(
        pair_results.loc[
            pair_results["local_pair_id"].astype(str).eq(BOUNDARY_PAIR_ID),
            "boundary_positive_leak_seed_route_count",
        ].sum()
    )
    rows = [
        _gate_row(
            "G1_upstream_contract_gates_pass",
            "Did every upstream 016 object-wall transfer contract gate pass?",
            _count_dict(contract_gates["gate_status"]),
            "all upstream contract gates pass",
            bool(contract_gates["gate_status"].astype(str).eq("pass").all()),
        ),
        _gate_row(
            "G2_exact_14_route_scope",
            "Was execution restricted to the 14 predeclared route-plan rows?",
            f"execution_plan_rows={len(execution_plan)} executed_route_contracts={trace_rows['route_contract_id'].nunique()}",
            "14 route rows and no extra route contracts",
            len(execution_plan) == 14
            and trace_rows["route_contract_id"].nunique() == 14
            and set(trace_rows["route_contract_id"]) == set(execution_plan["route_contract_id"]),
        ),
        _gate_row(
            "G3_predeclared_schedule_expansion",
            "Were route rows expanded only into the transferred recovery-loop and direct-only schedules?",
            f"route_step_configs={step_config_count} expected={expected_step_configs}",
            "7 recovery routes * 9 steps plus 7 direct-only routes * 2 steps",
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
            "G5_required_transfer_measurements_materialized",
            "Did trace rows include required endpoint-object, typed-transient, object-identity, objective, and support fields?",
            sorted(required_trace_columns & set(trace_rows.columns)),
            "all required trace columns present",
            required_trace_columns.issubset(set(trace_rows.columns)),
        ),
        _gate_row(
            "G6_016_direct_only_readout_typed",
            "Are all 016 direct-only seed routes classified as target-available or object-identity-blocked?",
            {
                "direct_seed_routes": int(len(direct_positive)),
                "direct_target_available": int(
                    direct_positive["direct_target_available_seed"].map(_as_bool).sum()
                ),
                "direct_object_identity_block": int(
                    direct_positive["direct_object_identity_block_seed"].map(_as_bool).sum()
                ),
                "direct_untyped": int(
                    direct_positive["untyped_transient_step_count"].astype(int).gt(0).sum()
                ),
            },
            "all 016 direct-only routes have typed readout and zero untyped states",
            not direct_positive.empty
            and int(direct_positive["untyped_transient_step_count"].astype(int).gt(0).sum()) == 0
            and (
                int(direct_positive["direct_target_available_seed"].map(_as_bool).sum())
                + int(direct_positive["direct_object_identity_block_seed"].map(_as_bool).sum())
                == len(direct_positive)
            ),
        ),
        _gate_row(
            "G7_016_recovery_readout_typed",
            "Are all 016 recovery-loop seed routes classified as target-with-recovery or typed-transient/object-identity readout?",
            {
                "recovery_seed_routes": int(len(recovery_positive)),
                "recovery_target_with_recovery": int(
                    recovery_positive["recovery_target_with_recovery_seed"].map(_as_bool).sum()
                ),
                "recovery_typed_transient_block": int(
                    recovery_positive["recovery_typed_transient_block_seed"].map(_as_bool).sum()
                ),
                "recovery_object_identity_block_routes": int(
                    recovery_positive["object_identity_blocker_step_count"].astype(int).gt(0).sum()
                ),
                "recovery_untyped": int(
                    recovery_positive["untyped_transient_step_count"].astype(int).gt(0).sum()
                ),
            },
            "all 016 recovery routes have typed readout and zero untyped states",
            not recovery_positive.empty
            and int(recovery_positive["untyped_transient_step_count"].astype(int).gt(0).sum()) == 0
            and bool(
                (
                    recovery_positive["recovery_target_with_recovery_seed"].map(_as_bool)
                    | recovery_positive["recovery_typed_transient_block_seed"].map(_as_bool)
                    | recovery_positive["object_identity_blocker_step_count"].astype(int).gt(0)
                ).all()
            ),
        ),
        _gate_row(
            "G8_005_boundary_controls_no_positive_leak",
            "Did the 005 boundary controls remain non-positive?",
            {
                "boundary_leak_seed_routes": boundary_leaks,
                "boundary_guard_results": _count_dict(boundary_results["boundary_guard_result"]),
            },
            "zero 005 positive leaks",
            boundary_leaks == 0
            and not boundary_results.empty
            and bool(boundary_results["boundary_guard_result"].astype(str).eq("closed").all()),
        ),
        _gate_row(
            "G9_typed_transient_vocabulary_available",
            "Does the trace use explicit typed-transient vocabulary?",
            sorted(typed_classes),
            "pathway_intermediate/object_identity_blocker/boundary_leak/unknown vocabulary allowed, no untyped bucket required",
            typed_classes.issubset(
                {
                    "pathway_intermediate",
                    "object_identity_blocker",
                    "boundary_leak",
                    "unknown",
                    "not_transient_known_anchor",
                }
            ),
        ),
        _gate_row(
            "G10_wall_method_quality_claims_closed",
            "Are pathway, wall, method, quality/cost, and full-replay claims explicitly closed?",
            CLAIM_BOUNDARY,
            "all promotion flags false",
            bool(route_results["wall_claim_allowed_after_trace"].eq(False).all())
            and bool(route_results["pathway_claim_allowed_after_trace"].eq(False).all())
            and bool(pair_results["wall_claim_allowed_after_trace"].eq(False).all())
            and bool(pair_results["pathway_claim_allowed_after_trace"].eq(False).all())
            and bool(route_results["method_claim_allowed_after_trace"].eq(False).all())
            and bool(route_results["quality_cost_claim_allowed_after_trace"].eq(False).all())
            and bool(route_results["full_replay_claim_allowed_after_trace"].eq(False).all()),
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
    route_transfer_summary: pd.DataFrame,
    pair_results: pd.DataFrame,
    boundary_results: pd.DataFrame,
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
        str(positive_pair.iloc[0]["pair_transfer_status"])
        if not positive_pair.empty
        else "missing_positive_pair_result"
    )
    boundary_status = (
        str(boundary_pair.iloc[0]["pair_transfer_status"])
        if not boundary_pair.empty
        else "missing_boundary_pair_result"
    )
    failed_gates = gates.loc[
        ~gates["gate_status"].astype(str).eq("pass"),
        "gate_id",
    ].tolist()
    if failed_gates:
        recommended_next_gate = (
            "Inspect failed execution/readout gates before interpreting 016 transfer outcomes."
        )
    else:
        recommended_next_gate = (
            "Audit the executed 016 transfer trace: decide whether target-anchor "
            "shape plus typed-transient/object-identity status is enough for a "
            "local object-wall evidence audit, while keeping labels closed."
        )
    return {
        "schema": "nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace_summary.v1",
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
        "route_transfer_result_count": int(len(route_results)),
        "route_transfer_summary_count": int(len(route_transfer_summary)),
        "pair_transfer_result_count": int(len(pair_results)),
        "boundary_guard_result_count": int(len(boundary_results)),
        "execution_contract_pair_role_counts": _count_dict(execution_plan["contract_pair_role"]),
        "execution_route_family_counts": _count_dict(execution_plan["planned_route_family"]),
        "route_transfer_outcome_class_counts": _count_dict(route_results["route_transfer_outcome_class"]),
        "route_transfer_status_counts": _count_dict(route_transfer_summary["route_transfer_status"]),
        "pair_transfer_status_counts": _count_dict(pair_results["pair_transfer_status"]),
        "boundary_guard_result_counts": _count_dict(boundary_results["boundary_guard_result"]),
        "typed_transient_assignment_counts": _count_dict(
            trace_rows["typed_transient_assignment_by_step"]
        ),
        "object_identity_transfer_status_counts": _count_dict(
            trace_rows["object_identity_transfer_status"]
        ),
        "positive_pair_transfer_status": positive_status,
        "boundary_pair_transfer_status": boundary_status,
        "positive_pair_results": positive_pair.to_dict("records"),
        "boundary_pair_results": boundary_pair.to_dict("records"),
        "wall_claim_ready_pairs": [],
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": failed_gates,
        "interpretation": (
            "The 14-row 016 object-wall transfer contract was executed as local "
            "fractional-edge traces. The trace materializes target-anchor shape, "
            "typed transient assignments, object-identity transfer status, and "
            "005 boundary guard status, but pathway, wall, method, quality/cost, "
            "and full-replay claims remain closed."
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
    route_transfer_summary: pd.DataFrame,
    route_results: pd.DataFrame,
    boundary_results: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass 016 Object-Wall Transfer Trace",
        "",
        f"- status: `{summary['status']}`",
        f"- route_execution_plan_row_count: {summary['route_execution_plan_row_count']}",
        f"- route_step_config_count: {summary['route_step_config_count']}",
        f"- trace_row_count: {summary['trace_row_count']}",
        f"- route_transfer_outcome_class_counts: {summary['route_transfer_outcome_class_counts']}",
        f"- route_transfer_status_counts: {summary['route_transfer_status_counts']}",
        f"- pair_transfer_status_counts: {summary['pair_transfer_status_counts']}",
        f"- boundary_guard_result_counts: {summary['boundary_guard_result_counts']}",
        f"- typed_transient_assignment_counts: {summary['typed_transient_assignment_counts']}",
        f"- object_identity_transfer_status_counts: {summary['object_identity_transfer_status_counts']}",
        f"- positive_pair_transfer_status: `{summary['positive_pair_transfer_status']}`",
        f"- boundary_pair_transfer_status: `{summary['boundary_pair_transfer_status']}`",
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
                "direct_target_available_seed_route_count",
                "direct_object_identity_block_seed_route_count",
                "recovery_target_with_recovery_seed_route_count",
                "recovery_typed_transient_block_seed_route_count",
                "boundary_positive_leak_seed_route_count",
                "untyped_transfer_seed_route_count",
                "pair_transfer_status",
            ],
            max_rows=10,
        ),
        "",
        "## Route Summary",
        "",
        _markdown_table(
            route_transfer_summary.sort_values(
                ["local_pair_id", "start_condition", "planned_route_family"],
                kind="mergesort",
            ),
            [
                "local_pair_id",
                "start_condition",
                "planned_route_family",
                "seed_count",
                "direct_target_available_seed_count",
                "direct_object_identity_block_seed_count",
                "recovery_target_with_recovery_seed_count",
                "recovery_typed_transient_block_seed_count",
                "boundary_positive_leak_seed_count",
                "route_transfer_status",
                "max_objective_debt_from_start",
                "max_objective_recovery_from_min",
            ],
            max_rows=40,
        ),
        "",
        "## Boundary Guards",
        "",
        _markdown_table(
            boundary_results.sort_values(
                ["local_pair_id", "start_condition", "planned_route_family"],
                kind="mergesort",
            ),
            [
                "local_pair_id",
                "start_condition",
                "planned_route_family",
                "seed_route_result_count",
                "boundary_positive_leak_seed_count",
                "boundary_guard_result",
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
                "final_target_anchor_proxy",
                "first_typed_transient_step",
                "accepted_recovery_after_min",
                "direct_target_available_seed",
                "recovery_target_with_recovery_seed",
                "boundary_positive_leak_observed",
                "route_transfer_outcome_class",
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
            "This run executes the predeclared transfer contract only. It is not "
            "pathway evidence, wall evidence, or method evidence. Any object-wall "
            "language must wait for a separate audit over the materialized typed "
            "trace, with 005 retained as the boundary guard."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    schedules = _register_first_pass_016_transfer_schedules()
    contract_dir = Path(args.contract_dir)
    local_ablation_dir = Path(args.local_ablation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contract_route_plan = _read_csv(contract_dir / CONTRACT_ROUTE_PLAN_ROWS_CSV)
    contract_pair_rows = _read_csv(contract_dir / CONTRACT_PAIR_ROWS_CSV)
    contract_boundary_guards = _read_csv(contract_dir / CONTRACT_BOUNDARY_GUARD_ROWS_CSV)
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

    route_results = _route_transfer_results(trace_rows)
    route_transfer_summary = _route_transfer_summary(route_results)
    pair_results = _pair_transfer_results(contract_pair_rows, route_results)
    boundary_results = _boundary_guard_results(contract_boundary_guards, route_results)
    gates = _gate_matrix(
        contract_gates=contract_gates,
        execution_plan=execution_plan,
        schedules=schedules,
        trace_rows=trace_rows,
        route_results=route_results,
        pair_results=pair_results,
        boundary_results=boundary_results,
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
        route_transfer_summary=route_transfer_summary,
        pair_results=pair_results,
        boundary_results=boundary_results,
        gates=gates,
        step_config_count=step_config_count,
        candidate_pair_count=candidate_pair_count,
        seeds=int(args.seeds),
    )

    _write_csv(execution_plan, output_dir / ROUTE_EXECUTION_PLAN_ROWS_CSV)
    _write_csv(trace_rows, output_dir / TRACE_ROWS_CSV)
    _write_csv(seed_summary, output_dir / SEED_ROUTE_SUMMARY_CSV)
    _write_csv(route_contract_summary, output_dir / ROUTE_CONTRACT_SUMMARY_CSV)
    _write_csv(route_results, output_dir / ROUTE_TRANSFER_RESULT_ROWS_CSV)
    _write_csv(route_transfer_summary, output_dir / ROUTE_TRANSFER_SUMMARY_ROWS_CSV)
    _write_csv(pair_results, output_dir / PAIR_TRANSFER_RESULT_ROWS_CSV)
    _write_csv(boundary_results, output_dir / BOUNDARY_GUARD_RESULT_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace_config.v1",
        "contract_dir": str(contract_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "gamma": float(args.gamma),
        "seeds": int(args.seeds),
        "n_iterations": int(args.n_iterations),
        "edge_chunk_size": int(args.edge_chunk_size),
        "route_schedules": schedules,
        "target_signature_id": TARGET_SIGNATURE_ID,
        "transient_signature_id": TRANSIENT_SIGNATURE_ID,
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
        route_transfer_summary=route_transfer_summary,
        route_results=route_results,
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
    parser.add_argument("--edge-chunk-size", type=int, default=5_000_000)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
