#!/usr/bin/env python3
"""Design the G4.8 fresh Axis B panel contract.

The previous Axis B source-start support contract clarified that source-start
support and post-start endpoint continuity must be measured separately. This
script freezes the next fresh panel around that split. It keeps the two already
executed route pairs as calibration rows, promotes not-yet-routed ready-like
pairs into fresh candidates, and predeclares control guards.

It does not run Leiden, execute route/pathway traces, promote walls, evaluate
quality/cost value, replay full NanoClustering, or claim method success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)


DEFAULT_LOCAL_PANEL_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_local_analog_validation_panel_gamma1e5_20260604"
)
DEFAULT_EXECUTION_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_local_validation_execution_contract_gamma1e5_20260604"
)
DEFAULT_READINESS_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_pathway_wall_readiness_audit_gamma1e5_20260604"
)
DEFAULT_SOURCE_START_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_axis_b_source_start_support_contract_gamma1e5_20260604"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_fresh_axis_b_panel_contract_gamma1e5_20260604"
)

LOCAL_PANEL_ROWS_CSV = "nanoclustering_g4_8_local_analog_validation_panel_rows.csv"
EXECUTION_PAIR_ROWS_CSV = (
    "nanoclustering_g4_8_local_validation_execution_contract_pair_rows.csv"
)
EXECUTION_UNIT_ROWS_CSV = (
    "nanoclustering_g4_8_local_validation_execution_contract_unit_rows.csv"
)
EXECUTION_GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_local_validation_execution_contract_gate_matrix.csv"
)
READINESS_PAIR_ROWS_CSV = "nanoclustering_g4_8_pathway_wall_readiness_audit_pair_rows.csv"
READINESS_UNIT_ROWS_CSV = "nanoclustering_g4_8_pathway_wall_readiness_audit_unit_rows.csv"
READINESS_GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_pathway_wall_readiness_audit_gate_matrix.csv"
)
SOURCE_START_CONTRACT_ROWS_CSV = (
    "nanoclustering_g4_8_axis_b_source_start_support_contract_rows.csv"
)
SOURCE_START_GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_axis_b_source_start_support_gate_matrix.csv"
)

FIELD_CONTRACT_ROWS_CSV = (
    "nanoclustering_g4_8_fresh_axis_b_panel_field_contract_rows.csv"
)
PANEL_PAIR_ROWS_CSV = "nanoclustering_g4_8_fresh_axis_b_panel_pair_rows.csv"
PANEL_UNIT_ROWS_CSV = "nanoclustering_g4_8_fresh_axis_b_panel_unit_rows.csv"
PANEL_ROUTE_PLAN_ROWS_CSV = (
    "nanoclustering_g4_8_fresh_axis_b_panel_route_plan_rows.csv"
)
PANEL_BATCH_PLAN_ROWS_CSV = (
    "nanoclustering_g4_8_fresh_axis_b_panel_batch_plan_rows.csv"
)
GATE_MATRIX_CSV = "nanoclustering_g4_8_fresh_axis_b_panel_gate_matrix.csv"
CONFIG_JSON = "nanoclustering_g4_8_fresh_axis_b_panel_config.json"
SUMMARY_JSON = "nanoclustering_g4_8_fresh_axis_b_panel_summary.json"
REPORT_MD = "nanoclustering_g4_8_fresh_axis_b_panel_report.md"

RUN_STATUS = "designed_nanoclustering_g4_8_fresh_axis_b_panel_contract"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 fresh Axis B panel contract design only; reads the "
    "local validation, pathway-readiness, and source-start split artifacts to "
    "predeclare calibration, fresh ready-like, and control rows. It does not run "
    "Leiden, execute route/pathway traces, promote walls, evaluate quality/cost "
    "value, replay full NanoClustering, or claim method success."
)

START_CONDITIONS = (
    "singleton",
    "pair_together",
    "bridges_to_left",
    "bridges_to_right",
    "all_local_together",
)
CALIBRATION_PAIR_IDS = ("local_pair_009", "local_pair_012")
CONTROL_AXIS_ORDER = (
    "target_saturated_no_handle",
    "latent_release_without_original_coassigned_source",
    "hard_no_release_control",
    "coupled_direct_bridge_failure",
)

FIELD_CONTRACTS = (
    {
        "field_id": "F1_source_start_support",
        "field_family": "source_start",
        "required_for_route_plan": True,
        "field_question": "Does step 1 have source-start support under the specified rotation vocabulary?",
        "fresh_panel_requirement": "record separately; never repair source-start caveats with interior endpoint evidence",
        "preexisting_calibration_status": "available for local_pair_009 and local_pair_012 only",
    },
    {
        "field_id": "F2_post_start_endpoint_continuity",
        "field_family": "interior_endpoint",
        "required_for_route_plan": True,
        "field_question": "Are post-start endpoint signatures pair-level known, with no true-novel endpoint?",
        "fresh_panel_requirement": "record after step 1 only",
        "preexisting_calibration_status": "passes for 80 of 80 calibration routes in every rotation mode",
    },
    {
        "field_id": "F3_target_final_continuity",
        "field_family": "interior_endpoint",
        "required_for_route_plan": True,
        "field_question": "Does the route end at the intended pair-level target?",
        "fresh_panel_requirement": "record independently from source-start support",
        "preexisting_calibration_status": "available for calibration routes only",
    },
    {
        "field_id": "F4_direct_edge_retention",
        "field_family": "physical_direct_path",
        "required_for_route_plan": True,
        "field_question": "Is the direct pair edge retained while Axis B continuity is tested?",
        "fresh_panel_requirement": "record as a physical direct-path guard",
        "preexisting_calibration_status": "available for calibration routes only",
    },
    {
        "field_id": "F5_same_seed_unknown_reclassification",
        "field_family": "endpoint_atlas",
        "required_for_route_plan": True,
        "field_question": "Can same-seed unknown endpoints be reclassified against a pair-level atlas?",
        "fresh_panel_requirement": "record as a diagnostic; it cannot repair source-start support",
        "preexisting_calibration_status": "same-seed unknown endpoint routes are pair-level known under rotation",
    },
    {
        "field_id": "F6_objective_debt_and_recovery",
        "field_family": "pathway_shape",
        "required_for_route_plan": False,
        "field_question": "Does the route incur and recover objective debt?",
        "fresh_panel_requirement": "record only as pathway-shape evidence, not as a wall claim",
        "preexisting_calibration_status": "available only for executed scoped route traces",
    },
    {
        "field_id": "F7_wall_claim_closed",
        "field_family": "claim_boundary",
        "required_for_route_plan": True,
        "field_question": "Are wall claims still closed for this panel?",
        "fresh_panel_requirement": "wall_claim_allowed must remain false for every route-plan row",
        "preexisting_calibration_status": "wall readiness remains zero",
    },
    {
        "field_id": "F8_method_quality_cost_claims_closed",
        "field_family": "claim_boundary",
        "required_for_route_plan": True,
        "field_question": "Are method, quality/cost, and full-replay claims closed?",
        "fresh_panel_requirement": "claim fields remain false unless a later execution artifact explicitly opens them",
        "preexisting_calibration_status": "closed",
    },
)


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _clean_text(value: Any, default: str = "") -> str:
    if pd.isna(value):
        return default
    text = str(value)
    if text.lower() == "nan":
        return default
    return text


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 50) -> str:
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


def _field_contract_rows() -> pd.DataFrame:
    rows = pd.DataFrame(list(FIELD_CONTRACTS))
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _limitation_axis(row: pd.Series) -> str:
    existing = _clean_text(row.get("limitation_axis", ""))
    if existing:
        return existing
    stratum = _clean_text(row.get("validation_stratum", ""))
    mapping = {
        "strict_ready": "ready_conditional_or_boundary",
        "rare_ready": "ready_conditional_or_boundary",
        "target_saturated_no_handle": "target_saturated_no_handle",
        "latent_release_no_source_control": "latent_release_without_original_coassigned_source",
        "no_release_control": "hard_no_release_control",
        "coupled_direct_bridge_failure_control": "coupled_direct_bridge_failure",
    }
    return mapping.get(stratum, stratum or "unclassified")


def _first_pass_control_ids(pair_rows: pd.DataFrame) -> set[str]:
    selected: set[str] = set()
    stable_controls = pair_rows[
        pair_rows["execution_lane"].astype(str).eq("stable_lane")
        & ~pair_rows["local_pair_id"].astype(str).isin(CALIBRATION_PAIR_IDS)
    ].copy()
    stable_controls["limitation_axis_resolved"] = stable_controls.apply(
        _limitation_axis, axis=1
    )
    for axis in CONTROL_AXIS_ORDER:
        matches = stable_controls[
            stable_controls["limitation_axis_resolved"].astype(str).eq(axis)
        ].sort_values("local_pair_id", kind="mergesort")
        if not matches.empty:
            selected.add(str(matches.iloc[0]["local_pair_id"]))
    return selected


def _source_start_pair_summary(source_contract_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair_id, group in source_contract_rows.groupby("local_pair_id", sort=True):
        source_pass = group["source_start_contract_pass"].map(_as_bool)
        interior_pass = group["post_start_interior_contract_pass"].map(_as_bool)
        caveat_rows = group[~source_pass]
        rows.append(
            {
                "local_pair_id": str(pair_id),
                "existing_source_start_contract_row_count": int(len(group)),
                "existing_source_start_contract_pass_count": int(source_pass.sum()),
                "existing_post_start_interior_contract_pass_count": int(
                    interior_pass.sum()
                ),
                "existing_source_start_caveat_contract_count": int(len(caveat_rows)),
                "existing_source_start_status_counts": json.dumps(
                    _count_dict(group["source_start_support_contract_status"]),
                    sort_keys=True,
                ),
                "existing_source_start_caveat_starts": ";".join(
                    sorted(caveat_rows["start_condition"].astype(str).unique())
                ),
                "existing_source_start_split_status": (
                    "source_start_split_measured_with_caveat"
                    if len(caveat_rows)
                    else "source_start_split_measured_pass"
                ),
            }
        )
    return pd.DataFrame(rows)


def _panel_role(row: pd.Series, first_pass_control_ids: set[str]) -> str:
    pair_id = str(row["local_pair_id"])
    lane = _clean_text(row.get("execution_lane", ""))
    family = _clean_text(row.get("validation_family", ""))
    if pair_id in CALIBRATION_PAIR_IDS:
        return "carryover_axis_b_calibration_pair"
    if family == "ready" and lane == "conditional_lane":
        return "fresh_core_ready_conditional_pair"
    if family == "ready" and lane == "boundary_lane":
        return "fresh_diagnostic_ready_boundary_pair"
    if pair_id in first_pass_control_ids:
        return "fresh_first_pass_control_pair"
    if family in {"target_saturated", "nonready_control", "failure_control"}:
        if lane == "boundary_lane":
            return "diagnostic_control_boundary_pair"
        return "reserve_control_pair"
    return "unselected_or_review_pair"


def _role_priority(role: str) -> int:
    if role == "carryover_axis_b_calibration_pair":
        return 0
    if role in {"fresh_core_ready_conditional_pair", "fresh_first_pass_control_pair"}:
        return 1
    if role in {
        "fresh_diagnostic_ready_boundary_pair",
        "diagnostic_control_boundary_pair",
    }:
        return 2
    if role == "reserve_control_pair":
        return 3
    return 9


def _role_action(role: str) -> str:
    if role == "carryover_axis_b_calibration_pair":
        return "read_existing_axis_b_source_start_split_outputs"
    if role == "fresh_core_ready_conditional_pair":
        return "execute_fresh_axis_b_ready_like_allowed_starts_if_next_gate_opens"
    if role == "fresh_first_pass_control_pair":
        return "execute_fresh_axis_b_control_guard_allowed_starts_if_next_gate_opens"
    if role == "fresh_diagnostic_ready_boundary_pair":
        return "hold_for_diagnostic_ready_boundary_execution_after_first_pass"
    if role == "diagnostic_control_boundary_pair":
        return "hold_for_diagnostic_control_boundary_execution"
    if role == "reserve_control_pair":
        return "hold_as_control_reserve"
    return "hold_pending_review"


def _pair_selection_reason(row: pd.Series) -> str:
    role = str(row["panel_role"])
    if role == "carryover_axis_b_calibration_pair":
        return "existing scoped route pair; retained only as calibration for split Axis B fields"
    if role == "fresh_core_ready_conditional_pair":
        return "not-yet-routed ready-like pair with allowed conditional starts"
    if role == "fresh_diagnostic_ready_boundary_pair":
        return "not-yet-routed ready-like boundary pair; diagnostic only"
    if role == "fresh_first_pass_control_pair":
        return "lowest-id stable control selected to cover a blocked/control limitation axis"
    if role == "diagnostic_control_boundary_pair":
        return "boundary control retained as diagnostic, not first-pass evidence"
    if role == "reserve_control_pair":
        return "additional stable control retained as reserve guard"
    return "not selected for the fresh Axis B panel"


def _build_pair_rows(
    *,
    local_rows: pd.DataFrame,
    execution_pair_rows: pd.DataFrame,
    readiness_pair_rows: pd.DataFrame,
    source_start_summary: pd.DataFrame,
) -> pd.DataFrame:
    local_keep = [
        "local_pair_id",
        "analog_macro_role",
        "analog_source_condition",
        "bridge_release_lift_proxy",
        "direct_dependency_proxy",
        "original_distinct_endpoint_count",
        "original_source_endpoint_signature_proxy_count",
    ]
    local_extra = local_rows[[col for col in local_keep if col in local_rows.columns]].copy()
    rows = execution_pair_rows.merge(
        local_extra,
        on="local_pair_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_local"),
    )
    readiness_keep = [
        "local_pair_id",
        "limitation_axis",
        "pathway_readiness_status",
        "pathway_probe_candidate_pair",
        "wall_claim_ready_pair",
        "pathway_probe_block_reasons",
        "missing_for_wall_claim",
    ]
    rows = rows.merge(
        readiness_pair_rows[[col for col in readiness_keep if col in readiness_pair_rows.columns]],
        on="local_pair_id",
        how="left",
        validate="one_to_one",
    )
    rows = rows.merge(
        source_start_summary,
        on="local_pair_id",
        how="left",
        validate="one_to_one",
    )
    for col in [
        "existing_source_start_contract_row_count",
        "existing_source_start_contract_pass_count",
        "existing_post_start_interior_contract_pass_count",
        "existing_source_start_caveat_contract_count",
    ]:
        rows[col] = rows[col].fillna(0).astype(int)
    rows["existing_source_start_status_counts"] = rows[
        "existing_source_start_status_counts"
    ].fillna("{}")
    rows["existing_source_start_caveat_starts"] = rows[
        "existing_source_start_caveat_starts"
    ].fillna("")
    rows["existing_source_start_split_status"] = rows[
        "existing_source_start_split_status"
    ].fillna("not_yet_measured")

    rows["limitation_axis_resolved"] = rows.apply(_limitation_axis, axis=1)
    first_pass_controls = _first_pass_control_ids(rows)
    rows["panel_role"] = [
        _panel_role(row, first_pass_controls) for _, row in rows.iterrows()
    ]
    rows["route_execution_priority"] = rows["panel_role"].map(_role_priority).astype(int)
    rows["route_plan_action"] = rows["panel_role"].map(_role_action)
    rows["panel_selection_reason"] = rows.apply(_pair_selection_reason, axis=1)
    rows["is_carryover_calibration_pair"] = rows["panel_role"].eq(
        "carryover_axis_b_calibration_pair"
    )
    rows["is_fresh_axis_b_pair"] = rows["panel_role"].isin(
        [
            "fresh_core_ready_conditional_pair",
            "fresh_diagnostic_ready_boundary_pair",
            "fresh_first_pass_control_pair",
            "diagnostic_control_boundary_pair",
            "reserve_control_pair",
        ]
    )
    rows["include_in_first_pass_axis_b_panel"] = rows["panel_role"].isin(
        ["fresh_core_ready_conditional_pair", "fresh_first_pass_control_pair"]
    )
    rows["include_as_diagnostic_axis_b_panel"] = rows["panel_role"].isin(
        ["fresh_diagnostic_ready_boundary_pair", "diagnostic_control_boundary_pair"]
    )
    rows["include_as_axis_b_control_reserve"] = rows["panel_role"].eq(
        "reserve_control_pair"
    )
    rows["fresh_stable_positive_pair"] = (
        rows["validation_family"].astype(str).eq("ready")
        & rows["execution_lane"].astype(str).eq("stable_lane")
        & ~rows["local_pair_id"].astype(str).isin(CALIBRATION_PAIR_IDS)
    )
    rows["source_start_support_required"] = True
    rows["post_start_endpoint_continuity_required"] = True
    rows["target_final_continuity_required"] = True
    rows["direct_edge_retention_required"] = True
    rows["wall_claim_allowed"] = False
    rows["method_claim_allowed"] = False
    rows["quality_cost_claim_allowed"] = False
    rows["no_new_leiden_execution"] = True
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    sort_cols = ["route_execution_priority", "panel_role", "local_pair_id"]
    return rows.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _build_unit_rows(execution_unit_rows: pd.DataFrame, panel_pair_rows: pd.DataFrame) -> pd.DataFrame:
    pair_keep = [
        "local_pair_id",
        "panel_role",
        "route_execution_priority",
        "route_plan_action",
        "panel_selection_reason",
        "limitation_axis_resolved",
        "is_carryover_calibration_pair",
        "is_fresh_axis_b_pair",
        "include_in_first_pass_axis_b_panel",
        "include_as_diagnostic_axis_b_panel",
        "include_as_axis_b_control_reserve",
        "source_start_support_required",
        "post_start_endpoint_continuity_required",
        "target_final_continuity_required",
        "direct_edge_retention_required",
        "wall_claim_allowed",
        "method_claim_allowed",
        "quality_cost_claim_allowed",
    ]
    rows = execution_unit_rows.merge(
        panel_pair_rows[pair_keep],
        on="local_pair_id",
        how="left",
        validate="many_to_one",
    )
    rows["axis_b_panel_unit_id"] = (
        rows["local_pair_id"].astype(str) + "__" + rows["start_condition"].astype(str)
    )
    rows["include_in_axis_b_route_inventory"] = True
    rows["new_route_execution_required"] = rows[
        "include_in_first_pass_axis_b_panel"
    ].astype(bool)
    rows["diagnostic_route_execution_optional"] = rows[
        "include_as_diagnostic_axis_b_panel"
    ].astype(bool)
    rows["reserve_route_execution_optional"] = rows[
        "include_as_axis_b_control_reserve"
    ].astype(bool)
    rows["calibration_uses_existing_route_output"] = rows[
        "is_carryover_calibration_pair"
    ].astype(bool)
    rows["required_route_family"] = "primary_bridge_release_axis_b_split"
    rows["fresh_axis_b_measurement_contract"] = (
        "record source-start support, post-start endpoint continuity, "
        "target-final continuity, and direct-edge retention separately"
    )
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    sort_cols = [
        "route_execution_priority",
        "panel_role",
        "local_pair_id",
        "start_condition",
    ]
    return rows.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _build_route_plan_rows(panel_unit_rows: pd.DataFrame) -> pd.DataFrame:
    rows = panel_unit_rows.copy()
    rows["route_plan_id"] = (
        rows["axis_b_panel_unit_id"].astype(str)
        + "__"
        + rows["required_route_family"].astype(str)
    )
    rows["route_plan_status"] = rows["panel_role"].map(
        {
            "carryover_axis_b_calibration_pair": "existing_output_calibration",
            "fresh_core_ready_conditional_pair": "fresh_first_pass_ready_like",
            "fresh_first_pass_control_pair": "fresh_first_pass_control_guard",
            "fresh_diagnostic_ready_boundary_pair": "diagnostic_ready_boundary_hold",
            "diagnostic_control_boundary_pair": "diagnostic_control_boundary_hold",
            "reserve_control_pair": "control_reserve_hold",
        }
    ).fillna("hold_pending_review")
    rows["source_start_support_measurement_required"] = True
    rows["post_start_endpoint_continuity_measurement_required"] = True
    rows["target_final_continuity_measurement_required"] = True
    rows["direct_edge_retention_measurement_required"] = True
    rows["same_seed_unknown_reclassification_measurement_required"] = True
    rows["objective_debt_recovery_measurement_required"] = rows["panel_role"].isin(
        [
            "carryover_axis_b_calibration_pair",
            "fresh_core_ready_conditional_pair",
            "fresh_diagnostic_ready_boundary_pair",
        ]
    )
    rows["wall_claim_allowed"] = False
    rows["method_claim_allowed"] = False
    rows["quality_cost_claim_allowed"] = False
    rows["no_new_leiden_execution_in_this_contract"] = True
    keep = [
        "route_plan_id",
        "axis_b_panel_unit_id",
        "validation_unit_id",
        "local_pair_id",
        "start_condition",
        "branch",
        "execution_lane",
        "validation_stratum",
        "validation_family",
        "panel_role",
        "route_execution_priority",
        "route_plan_status",
        "route_plan_action",
        "limitation_axis_resolved",
        "required_route_family",
        "new_route_execution_required",
        "diagnostic_route_execution_optional",
        "reserve_route_execution_optional",
        "calibration_uses_existing_route_output",
        "source_start_support_measurement_required",
        "post_start_endpoint_continuity_measurement_required",
        "target_final_continuity_measurement_required",
        "direct_edge_retention_measurement_required",
        "same_seed_unknown_reclassification_measurement_required",
        "objective_debt_recovery_measurement_required",
        "wall_claim_allowed",
        "method_claim_allowed",
        "quality_cost_claim_allowed",
        "no_new_leiden_execution_in_this_contract",
        "fresh_axis_b_measurement_contract",
        "run_status",
        "claim_boundary",
    ]
    return rows[[col for col in keep if col in rows.columns]].sort_values(
        ["route_execution_priority", "panel_role", "local_pair_id", "start_condition"],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_batch_plan_rows(route_plan_rows: pd.DataFrame) -> pd.DataFrame:
    specs = [
        (
            "B0_carryover_calibration",
            "carryover_axis_b_calibration_pair",
            "calibration",
            "Read existing Axis B split outputs only; do not count as fresh evidence.",
        ),
        (
            "B1_fresh_core_ready_conditional",
            "fresh_core_ready_conditional_pair",
            "fresh_primary_ready_like",
            "First fresh ready-like Axis B execution if the next gate opens.",
        ),
        (
            "B2_fresh_first_pass_controls",
            "fresh_first_pass_control_pair",
            "fresh_primary_control_guard",
            "One stable control per blocked/control limitation axis.",
        ),
        (
            "B3_diagnostic_ready_boundary",
            "fresh_diagnostic_ready_boundary_pair",
            "diagnostic_ready_like",
            "Boundary ready-like rows; hold until first-pass behavior is understood.",
        ),
        (
            "B4_diagnostic_control_boundary",
            "diagnostic_control_boundary_pair",
            "diagnostic_control",
            "Boundary control rows; diagnostic only.",
        ),
        (
            "B5_reserve_controls",
            "reserve_control_pair",
            "control_reserve",
            "Additional controls retained for later stress tests.",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for order, (batch_id, role, batch_role, instruction) in enumerate(specs):
        group = route_plan_rows[route_plan_rows["panel_role"].astype(str).eq(role)]
        rows.append(
            {
                "batch_order": order,
                "batch_id": batch_id,
                "panel_role": role,
                "batch_role": batch_role,
                "pair_count": int(group["local_pair_id"].nunique()),
                "route_plan_row_count": int(len(group)),
                "new_route_execution_row_count": int(
                    group["new_route_execution_required"].map(_as_bool).sum()
                )
                if not group.empty
                else 0,
                "start_condition_counts": json.dumps(
                    _count_dict(group["start_condition"]) if not group.empty else {},
                    sort_keys=True,
                ),
                "local_pair_ids": ";".join(
                    sorted(group["local_pair_id"].astype(str).unique())
                ),
                "instruction": instruction,
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
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _gate_matrix(
    *,
    execution_gates: pd.DataFrame,
    readiness_gates: pd.DataFrame,
    source_start_gates: pd.DataFrame,
    field_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    unit_rows: pd.DataFrame,
    route_plan_rows: pd.DataFrame,
    batch_rows: pd.DataFrame,
) -> pd.DataFrame:
    first_pass = route_plan_rows[
        route_plan_rows["new_route_execution_required"].map(_as_bool)
    ]
    fresh_ready = pair_rows[
        pair_rows["panel_role"].astype(str).isin(
            [
                "fresh_core_ready_conditional_pair",
                "fresh_diagnostic_ready_boundary_pair",
            ]
        )
    ]
    first_pass_controls = pair_rows[
        pair_rows["panel_role"].astype(str).eq("fresh_first_pass_control_pair")
    ]
    calibration = pair_rows[
        pair_rows["panel_role"].astype(str).eq("carryover_axis_b_calibration_pair")
    ]
    return pd.DataFrame(
        [
            _gate_row(
                "G1_upstream_design_gates_pass",
                "Do upstream local execution, readiness, and source-start gates pass?",
                {
                    "execution": _count_dict(execution_gates["gate_status"]),
                    "readiness": _count_dict(readiness_gates["gate_status"]),
                    "source_start": _count_dict(source_start_gates["gate_status"]),
                },
                "all upstream gates pass",
                bool(execution_gates["gate_status"].astype(str).eq("pass").all())
                and bool(readiness_gates["gate_status"].astype(str).eq("pass").all())
                and bool(source_start_gates["gate_status"].astype(str).eq("pass").all()),
            ),
            _gate_row(
                "G2_split_fields_materialized",
                "Are source-start, post-start, target-final, direct-edge, and boundary fields explicit?",
                f"field_count={len(field_rows)} field_ids={';'.join(field_rows['field_id'].astype(str))}",
                "8 field contracts with source-start and interior fields split",
                len(field_rows) == 8
                and bool(field_rows["required_for_route_plan"].map(_as_bool).any()),
            ),
            _gate_row(
                "G3_calibration_separated_from_fresh_panel",
                "Are the two executed pairs retained only as calibration?",
                {
                    "calibration_pair_count": int(len(calibration)),
                    "calibration_new_route_rows": int(
                        route_plan_rows[
                            route_plan_rows["panel_role"].astype(str).eq(
                                "carryover_axis_b_calibration_pair"
                            )
                            & route_plan_rows["new_route_execution_required"].map(
                                _as_bool
                            )
                        ].shape[0]
                    ),
                },
                "2 calibration pairs and zero new calibration route rows",
                int(len(calibration)) == 2
                and not bool(
                    route_plan_rows[
                        route_plan_rows["panel_role"].astype(str).eq(
                            "carryover_axis_b_calibration_pair"
                        )
                    ]["new_route_execution_required"].map(_as_bool).any()
                ),
            ),
            _gate_row(
                "G4_fresh_ready_like_panel_exists",
                "Does the panel contain not-yet-routed ready-like pairs?",
                {
                    "fresh_ready_pair_count": int(len(fresh_ready)),
                    "fresh_ready_roles": _count_dict(fresh_ready["panel_role"]),
                },
                "at least 5 conditional ready-like pairs plus diagnostic boundary ready-like rows",
                int(
                    pair_rows["panel_role"]
                    .astype(str)
                    .eq("fresh_core_ready_conditional_pair")
                    .sum()
                )
                >= 5
                and int(
                    pair_rows["panel_role"]
                    .astype(str)
                    .eq("fresh_diagnostic_ready_boundary_pair")
                    .sum()
                )
                >= 2,
            ),
            _gate_row(
                "G5_first_pass_is_bounded_and_controlled",
                "Is the first fresh execution slice bounded and paired with controls?",
                {
                    "first_pass_route_rows": int(len(first_pass)),
                    "first_pass_pair_count": int(first_pass["local_pair_id"].nunique()),
                    "first_pass_control_pair_count": int(len(first_pass_controls)),
                },
                "first pass has 20-45 route rows and at least 4 control pairs",
                20 <= int(len(first_pass)) <= 45 and int(len(first_pass_controls)) >= 4,
            ),
            _gate_row(
                "G6_first_pass_controls_cover_limitation_axes",
                "Do first-pass controls cover target saturation, latent source absence, no-release, and coupled failure?",
                sorted(first_pass_controls["limitation_axis_resolved"].astype(str).unique()),
                "all four blocked/control limitation axes represented",
                set(CONTROL_AXIS_ORDER).issubset(
                    set(first_pass_controls["limitation_axis_resolved"].astype(str))
                ),
            ),
            _gate_row(
                "G7_no_fresh_stable_positive_overclaim",
                "Is the lack of new stable positive ready pairs explicitly recorded?",
                int(pair_rows["fresh_stable_positive_pair"].map(_as_bool).sum()),
                "zero fresh stable positive pairs outside calibration",
                int(pair_rows["fresh_stable_positive_pair"].map(_as_bool).sum()) == 0,
            ),
            _gate_row(
                "G8_route_rows_require_split_measurements",
                "Does every route-plan row require the split Axis B fields?",
                {
                    "source_start": int(
                        route_plan_rows[
                            "source_start_support_measurement_required"
                        ].map(_as_bool).sum()
                    ),
                    "post_start": int(
                        route_plan_rows[
                            "post_start_endpoint_continuity_measurement_required"
                        ].map(_as_bool).sum()
                    ),
                    "target_final": int(
                        route_plan_rows[
                            "target_final_continuity_measurement_required"
                        ].map(_as_bool).sum()
                    ),
                    "direct_edge": int(
                        route_plan_rows[
                            "direct_edge_retention_measurement_required"
                        ].map(_as_bool).sum()
                    ),
                    "route_plan_rows": int(len(route_plan_rows)),
                },
                "all route-plan rows require source-start, post-start, target-final, and direct-edge fields",
                all(
                    bool(route_plan_rows[col].map(_as_bool).all())
                    for col in [
                        "source_start_support_measurement_required",
                        "post_start_endpoint_continuity_measurement_required",
                        "target_final_continuity_measurement_required",
                        "direct_edge_retention_measurement_required",
                    ]
                ),
            ),
            _gate_row(
                "G9_wall_method_quality_claims_closed",
                "Are wall, method, and quality/cost claims closed in every route-plan row?",
                {
                    "wall_claim_allowed_count": int(
                        route_plan_rows["wall_claim_allowed"].map(_as_bool).sum()
                    ),
                    "method_claim_allowed_count": int(
                        route_plan_rows["method_claim_allowed"].map(_as_bool).sum()
                    ),
                    "quality_cost_claim_allowed_count": int(
                        route_plan_rows["quality_cost_claim_allowed"].map(_as_bool).sum()
                    ),
                },
                "zero allowed wall/method/quality-cost claims",
                not bool(route_plan_rows["wall_claim_allowed"].map(_as_bool).any())
                and not bool(route_plan_rows["method_claim_allowed"].map(_as_bool).any())
                and not bool(
                    route_plan_rows["quality_cost_claim_allowed"].map(_as_bool).any()
                ),
            ),
            _gate_row(
                "G10_batch_plan_covers_route_inventory",
                "Does the batch plan cover every route inventory row exactly once by role?",
                {
                    "batch_route_rows": int(batch_rows["route_plan_row_count"].sum()),
                    "route_plan_rows": int(len(route_plan_rows)),
                    "unit_rows": int(len(unit_rows)),
                },
                "batch route rows equal route-plan rows and unit rows",
                int(batch_rows["route_plan_row_count"].sum()) == int(len(route_plan_rows))
                == int(len(unit_rows)),
            ),
            _gate_row(
                "G11_no_new_leiden_execution",
                "Is this artifact a design contract rather than an execution run?",
                RUN_STATUS,
                "design/materialization only",
                True,
            ),
        ]
    )


def _status(gates: pd.DataFrame) -> str:
    if not bool(gates["gate_status"].astype(str).eq("pass").all()):
        return "fresh_axis_b_panel_contract_gate_failed"
    return "fresh_axis_b_panel_contract_ready_no_new_stable_positive_limit_recorded"


def _summary(
    *,
    output_dir: Path,
    local_panel_dir: Path,
    execution_dir: Path,
    readiness_dir: Path,
    source_start_dir: Path,
    field_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    unit_rows: pd.DataFrame,
    route_plan_rows: pd.DataFrame,
    batch_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    first_pass = route_plan_rows[
        route_plan_rows["new_route_execution_required"].map(_as_bool)
    ]
    fresh_ready = pair_rows[
        pair_rows["panel_role"].astype(str).isin(
            [
                "fresh_core_ready_conditional_pair",
                "fresh_diagnostic_ready_boundary_pair",
            ]
        )
    ]
    calibration = pair_rows[
        pair_rows["panel_role"].astype(str).eq("carryover_axis_b_calibration_pair")
    ]
    first_pass_controls = pair_rows[
        pair_rows["panel_role"].astype(str).eq("fresh_first_pass_control_pair")
    ]
    return {
        "schema": "nanoclustering_g4_8_fresh_axis_b_panel_summary.v1",
        "status": _status(gates),
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "local_panel_dir": str(local_panel_dir),
        "execution_dir": str(execution_dir),
        "readiness_dir": str(readiness_dir),
        "source_start_dir": str(source_start_dir),
        "output_dir": str(output_dir),
        "field_contract_count": int(len(field_rows)),
        "pair_count": int(len(pair_rows)),
        "unit_row_count": int(len(unit_rows)),
        "route_plan_row_count": int(len(route_plan_rows)),
        "batch_plan_row_count": int(len(batch_rows)),
        "calibration_pair_count": int(len(calibration)),
        "fresh_ready_like_pair_count": int(len(fresh_ready)),
        "fresh_core_ready_conditional_pair_count": int(
            pair_rows["panel_role"]
            .astype(str)
            .eq("fresh_core_ready_conditional_pair")
            .sum()
        ),
        "fresh_diagnostic_ready_boundary_pair_count": int(
            pair_rows["panel_role"]
            .astype(str)
            .eq("fresh_diagnostic_ready_boundary_pair")
            .sum()
        ),
        "fresh_first_pass_control_pair_count": int(len(first_pass_controls)),
        "fresh_first_pass_route_row_count": int(len(first_pass)),
        "fresh_stable_positive_pair_count": int(
            pair_rows["fresh_stable_positive_pair"].map(_as_bool).sum()
        ),
        "panel_role_counts": _count_dict(pair_rows["panel_role"]),
        "route_plan_status_counts": _count_dict(route_plan_rows["route_plan_status"]),
        "limitation_axis_counts": _count_dict(pair_rows["limitation_axis_resolved"]),
        "first_pass_limitation_axis_counts": _count_dict(
            first_pass["limitation_axis_resolved"]
        ),
        "calibration_source_start_status_counts": _count_dict(
            calibration["existing_source_start_split_status"]
        ),
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "interpretation": (
            "The next panel should not be a rerun of local_pair_009 and "
            "local_pair_012. Those two pairs are retained only to calibrate the "
            "split Axis B fields. Fresh evidence should come from not-yet-routed "
            "ready-like conditional and boundary pairs, while the first pass is "
            "bounded by one control per blocked/control limitation axis. No new "
            "stable positive pair exists outside the calibration rows in the "
            "current contract surface, so that limitation is explicit."
        ),
        "recommended_next_gate": (
            "If execution is opened, run only the first-pass fresh rows first: "
            "fresh_core_ready_conditional_pair plus fresh_first_pass_control_pair. "
            "For every route, record source-start support, post-start endpoint "
            "continuity, target-final continuity, and direct-edge retention "
            "separately. Keep boundary ready rows and reserve controls out of the "
            "main evidence until the first pass is inspected."
        ),
        "written_artifacts": [
            FIELD_CONTRACT_ROWS_CSV,
            PANEL_PAIR_ROWS_CSV,
            PANEL_UNIT_ROWS_CSV,
            PANEL_ROUTE_PLAN_ROWS_CSV,
            PANEL_BATCH_PLAN_ROWS_CSV,
            GATE_MATRIX_CSV,
            CONFIG_JSON,
            SUMMARY_JSON,
            REPORT_MD,
        ],
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    field_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    route_plan_rows: pd.DataFrame,
    batch_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 Fresh Axis B Panel Contract",
        "",
        f"- status: `{summary['status']}`",
        f"- pair_count: {summary['pair_count']}",
        f"- route_plan_row_count: {summary['route_plan_row_count']}",
        f"- calibration_pair_count: {summary['calibration_pair_count']}",
        f"- fresh_ready_like_pair_count: {summary['fresh_ready_like_pair_count']}",
        f"- fresh_core_ready_conditional_pair_count: {summary['fresh_core_ready_conditional_pair_count']}",
        f"- fresh_diagnostic_ready_boundary_pair_count: {summary['fresh_diagnostic_ready_boundary_pair_count']}",
        f"- fresh_first_pass_control_pair_count: {summary['fresh_first_pass_control_pair_count']}",
        f"- fresh_first_pass_route_row_count: {summary['fresh_first_pass_route_row_count']}",
        f"- fresh_stable_positive_pair_count: {summary['fresh_stable_positive_pair_count']}",
        f"- panel_role_counts: {summary['panel_role_counts']}",
        f"- first_pass_limitation_axis_counts: {summary['first_pass_limitation_axis_counts']}",
        f"- calibration_source_start_status_counts: {summary['calibration_source_start_status_counts']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Field Contract",
        "",
        _markdown_table(
            field_rows,
            [
                "field_id",
                "field_family",
                "required_for_route_plan",
                "field_question",
                "fresh_panel_requirement",
                "preexisting_calibration_status",
            ],
            max_rows=20,
        ),
        "",
        "## Pair Roles",
        "",
        _markdown_table(
            pair_rows,
            [
                "local_pair_id",
                "branch",
                "validation_stratum",
                "execution_lane",
                "limitation_axis_resolved",
                "panel_role",
                "route_execution_priority",
                "allowed_execution_unit_count",
                "existing_source_start_split_status",
                "panel_selection_reason",
            ],
            max_rows=40,
        ),
        "",
        "## Batch Plan",
        "",
        _markdown_table(
            batch_rows,
            [
                "batch_order",
                "batch_id",
                "batch_role",
                "pair_count",
                "route_plan_row_count",
                "new_route_execution_row_count",
                "local_pair_ids",
                "instruction",
            ],
            max_rows=20,
        ),
        "",
        "## First-Pass Route Rows",
        "",
        _markdown_table(
            route_plan_rows[
                route_plan_rows["new_route_execution_required"].map(_as_bool)
            ],
            [
                "route_plan_id",
                "local_pair_id",
                "start_condition",
                "panel_role",
                "limitation_axis_resolved",
                "route_plan_status",
                "source_start_support_measurement_required",
                "post_start_endpoint_continuity_measurement_required",
                "target_final_continuity_measurement_required",
                "direct_edge_retention_measurement_required",
            ],
            max_rows=60,
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
            "This artifact is a panel contract. It opens no execution and makes no "
            "wall, quality/cost, full-replay, or method claim."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    local_panel_dir: Path,
    execution_dir: Path,
    readiness_dir: Path,
    source_start_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    local_rows = _read_csv(local_panel_dir / LOCAL_PANEL_ROWS_CSV)
    execution_pair_rows = _read_csv(execution_dir / EXECUTION_PAIR_ROWS_CSV)
    execution_unit_rows = _read_csv(execution_dir / EXECUTION_UNIT_ROWS_CSV)
    execution_gates = _read_csv(execution_dir / EXECUTION_GATE_MATRIX_CSV)
    readiness_pair_rows = _read_csv(readiness_dir / READINESS_PAIR_ROWS_CSV)
    _read_csv(readiness_dir / READINESS_UNIT_ROWS_CSV)
    readiness_gates = _read_csv(readiness_dir / READINESS_GATE_MATRIX_CSV)
    source_start_contract_rows = _read_csv(
        source_start_dir / SOURCE_START_CONTRACT_ROWS_CSV
    )
    source_start_gates = _read_csv(source_start_dir / SOURCE_START_GATE_MATRIX_CSV)

    field_rows = _field_contract_rows()
    source_start_summary = _source_start_pair_summary(source_start_contract_rows)
    pair_rows = _build_pair_rows(
        local_rows=local_rows,
        execution_pair_rows=execution_pair_rows,
        readiness_pair_rows=readiness_pair_rows,
        source_start_summary=source_start_summary,
    )
    unit_rows = _build_unit_rows(execution_unit_rows, pair_rows)
    route_plan_rows = _build_route_plan_rows(unit_rows)
    batch_rows = _build_batch_plan_rows(route_plan_rows)
    gates = _gate_matrix(
        execution_gates=execution_gates,
        readiness_gates=readiness_gates,
        source_start_gates=source_start_gates,
        field_rows=field_rows,
        pair_rows=pair_rows,
        unit_rows=unit_rows,
        route_plan_rows=route_plan_rows,
        batch_rows=batch_rows,
    )
    summary = _summary(
        output_dir=output_dir,
        local_panel_dir=local_panel_dir,
        execution_dir=execution_dir,
        readiness_dir=readiness_dir,
        source_start_dir=source_start_dir,
        field_rows=field_rows,
        pair_rows=pair_rows,
        unit_rows=unit_rows,
        route_plan_rows=route_plan_rows,
        batch_rows=batch_rows,
        gates=gates,
    )
    config = {
        "schema": "nanoclustering_g4_8_fresh_axis_b_panel_config.v1",
        "local_panel_dir": str(local_panel_dir),
        "execution_dir": str(execution_dir),
        "readiness_dir": str(readiness_dir),
        "source_start_dir": str(source_start_dir),
        "output_dir": str(output_dir),
        "calibration_pair_ids": list(CALIBRATION_PAIR_IDS),
        "control_axis_order": list(CONTROL_AXIS_ORDER),
        "start_conditions": list(START_CONDITIONS),
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }

    _write_csv(field_rows, output_dir / FIELD_CONTRACT_ROWS_CSV)
    _write_csv(pair_rows, output_dir / PANEL_PAIR_ROWS_CSV)
    _write_csv(unit_rows, output_dir / PANEL_UNIT_ROWS_CSV)
    _write_csv(route_plan_rows, output_dir / PANEL_ROUTE_PLAN_ROWS_CSV)
    _write_csv(batch_rows, output_dir / PANEL_BATCH_PLAN_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        field_rows=field_rows,
        pair_rows=pair_rows,
        route_plan_rows=route_plan_rows,
        batch_rows=batch_rows,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-panel-dir", type=Path, default=DEFAULT_LOCAL_PANEL_DIR)
    parser.add_argument("--execution-dir", type=Path, default=DEFAULT_EXECUTION_DIR)
    parser.add_argument("--readiness-dir", type=Path, default=DEFAULT_READINESS_DIR)
    parser.add_argument(
        "--source-start-dir", type=Path, default=DEFAULT_SOURCE_START_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        local_panel_dir=args.local_panel_dir,
        execution_dir=args.execution_dir,
        readiness_dir=args.readiness_dir,
        source_start_dir=args.source_start_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
