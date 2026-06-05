#!/usr/bin/env python3
"""Execute and read the G4.8 fresh Axis B first-pass route rows.

This runner consumes the 36-row first-pass readout contract and executes only
the route rows that were predeclared there. It maps the contract's
``primary_bridge_release_axis_b_split`` family onto the existing local
bridge-release interpolation schedule, then reads the result through the
first-pass evidence ladder.

It is a route-local first-pass screen. It does not promote basin walls, replay
full NanoClustering, evaluate downstream quality/cost value, or claim method
success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from design_leiden_basin_nanoclustering_g4_8_fresh_axis_b_first_pass_readout_contract import (
    CONTROL_TRAP_ROWS_CSV as CONTRACT_CONTROL_TRAP_ROWS_CSV,
    DEFAULT_OUTPUT_DIR as DEFAULT_READOUT_CONTRACT_DIR,
    GATE_MATRIX_CSV as CONTRACT_GATE_MATRIX_CSV,
    PAIR_READOUT_ROWS_CSV as CONTRACT_PAIR_READOUT_ROWS_CSV,
    ROUTE_READOUT_ROWS_CSV as CONTRACT_ROUTE_READOUT_ROWS_CSV,
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
    / "leiden_basin_nanoclustering_g4_8_fresh_axis_b_first_pass_trace_gamma1e5_20260604"
)

TRACE_ROWS_CSV = "nanoclustering_g4_8_fresh_axis_b_first_pass_trace_rows.csv"
ROUTE_EXECUTION_PLAN_ROWS_CSV = (
    "nanoclustering_g4_8_fresh_axis_b_first_pass_route_execution_plan_rows.csv"
)
SEED_ROUTE_SUMMARY_CSV = (
    "nanoclustering_g4_8_fresh_axis_b_first_pass_trace_seed_route_summary.csv"
)
ROUTE_CONTRACT_SUMMARY_CSV = (
    "nanoclustering_g4_8_fresh_axis_b_first_pass_trace_route_contract_summary.csv"
)
ROUTE_READOUT_RESULT_ROWS_CSV = (
    "nanoclustering_g4_8_fresh_axis_b_first_pass_route_readout_result_rows.csv"
)
PAIR_READOUT_RESULT_ROWS_CSV = (
    "nanoclustering_g4_8_fresh_axis_b_first_pass_pair_readout_result_rows.csv"
)
CONTROL_TRAP_RESULT_ROWS_CSV = (
    "nanoclustering_g4_8_fresh_axis_b_first_pass_control_trap_result_rows.csv"
)
GATE_MATRIX_CSV = "nanoclustering_g4_8_fresh_axis_b_first_pass_trace_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_g4_8_fresh_axis_b_first_pass_trace_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_fresh_axis_b_first_pass_trace_config.json"
REPORT_MD = "nanoclustering_g4_8_fresh_axis_b_first_pass_trace_report.md"

RUN_STATUS = "executed_nanoclustering_g4_8_fresh_axis_b_first_pass_trace"
ROUTE_EXECUTION_STATUS = "executed_fresh_axis_b_first_pass_local_route_trace"
WALL_PROMOTION_STATUS = "not_promoted_first_pass_screen_only"
METHOD_STATUS = "local_first_pass_route_screen_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 fresh Axis B first-pass route-local screen only; "
    "executes the 36 predeclared readout rows using the bridge-release "
    "interpolation schedule and reads source-start support, post-start "
    "endpoint continuity, target-final continuity, direct-edge retention, "
    "and control traps. It does not promote basin walls, evaluate "
    "quality/cost value, replay full NanoClustering, or claim method success."
)

CONTRACT_ROUTE_FAMILY = "primary_bridge_release_axis_b_split"
EXECUTED_ROUTE_FAMILY = "bridge_release_interpolation_probe"
READY_EVIDENCE_ROLE = "conditional_ready_like_test"
CONTROL_EVIDENCE_ROLE = "control_false_positive_guard"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


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


def _execution_plan(contract_routes: pd.DataFrame) -> pd.DataFrame:
    required = contract_routes[
        contract_routes["new_route_execution_required"].map(_as_bool)
    ].copy()
    if required.empty:
        raise ValueError("No route rows are marked new_route_execution_required.")
    bad_families = sorted(
        set(required["required_route_family"].astype(str)) - {CONTRACT_ROUTE_FAMILY}
    )
    if bad_families:
        raise ValueError(f"Unsupported required_route_family values: {bad_families}")
    rows = required.copy()
    rows["route_contract_id"] = rows["readout_plan_id"].astype(str)
    rows["planned_route_family"] = EXECUTED_ROUTE_FAMILY
    rows["route_family_role"] = rows["evidence_role"].astype(str).map(
        {
            READY_EVIDENCE_ROLE: "fresh_axis_b_ready_like",
            CONTROL_EVIDENCE_ROLE: "fresh_axis_b_control_guard",
        }
    )
    rows.loc[rows["route_family_role"].isna(), "route_family_role"] = "fresh_axis_b_other"
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _route_readout_results(
    *,
    contract_routes: pd.DataFrame,
    seed_summary: pd.DataFrame,
    trace_rows: pd.DataFrame,
) -> pd.DataFrame:
    contract_lookup = contract_routes.set_index("readout_plan_id").to_dict("index")
    first_rows = trace_rows[trace_rows["step_index"].astype(int).eq(1)].copy()
    final_rows = (
        trace_rows.sort_values("step_index", kind="mergesort")
        .groupby(["route_contract_id", "seed"], sort=False)
        .tail(1)
        .copy()
    )
    post_rows = trace_rows[trace_rows["step_index"].astype(int).gt(1)].copy()
    first_lookup = first_rows.set_index(["route_contract_id", "seed"]).to_dict("index")
    final_lookup = final_rows.set_index(["route_contract_id", "seed"]).to_dict("index")
    post_group = post_rows.groupby(["route_contract_id", "seed"], sort=False)

    rows: list[dict[str, Any]] = []
    for summary in seed_summary.itertuples(index=False):
        route_contract_id = str(summary.route_contract_id)
        seed = int(summary.seed)
        contract = contract_lookup.get(route_contract_id, {})
        evidence_role = str(contract.get("evidence_role", "unknown"))
        first = first_lookup.get((route_contract_id, seed), {})
        final = final_lookup.get((route_contract_id, seed), {})
        try:
            post = post_group.get_group((route_contract_id, seed))
        except KeyError:
            post = pd.DataFrame()

        post_assignments = (
            post["endpoint_assignment_by_step"].astype(str).tolist()
            if not post.empty
            else []
        )
        final_endpoint_assignment = str(
            getattr(summary, "final_endpoint_assignment", "")
        )
        source_start_support_pass = bool(first.get("matches_original_anchor", False))
        post_start_endpoint_continuity_pass = (
            not post.empty
            and bool(post["post_route_endpoint_assignment_available"].map(_as_bool).all())
            and not any(assignment == "unknown_new_endpoint" for assignment in post_assignments)
            and not any(assignment.startswith("ambiguous_anchor_match") for assignment in post_assignments)
        )
        target_final_continuity_pass = bool(final.get("matches_expected_final_anchor", False))
        target_final_bridge_exclusive_pass = final_endpoint_assignment == "drop_bridge_target_anchor"
        direct_edge_retention_pass = (
            not post.empty
            and bool(post["active_direct_edge_weight"].astype(float).gt(0.0).all())
            and bool(first.get("active_direct_edge_weight", 0.0) > 0.0)
        )
        same_seed_unknown_reclassification_pass = int(
            getattr(summary, "unknown_endpoint_step_count", 0)
        ) == 0
        objective_debt_recovery_observed = bool(
            getattr(summary, "max_objective_debt_from_start", 0.0) >= 0.0
        ) and bool(getattr(summary, "max_objective_recovery_from_min", 0.0) >= 0.0)

        all_positive_requirements_pass = all(
            [
                source_start_support_pass,
                post_start_endpoint_continuity_pass,
                target_final_bridge_exclusive_pass,
                direct_edge_retention_pass,
            ]
        )
        if evidence_role == CONTROL_EVIDENCE_ROLE:
            control_trap_leak_observed = all_positive_requirements_pass
            route_outcome_class = (
                "control_false_positive_leak"
                if control_trap_leak_observed
                else "control_trap_closed"
            )
        elif all_positive_requirements_pass:
            control_trap_leak_observed = False
            route_outcome_class = "conditional_ready_like_screen_pass"
        elif target_final_continuity_pass and not target_final_bridge_exclusive_pass:
            control_trap_leak_observed = False
            route_outcome_class = "nonexclusive_target_anchor_failure"
        elif (
            post_start_endpoint_continuity_pass
            and target_final_continuity_pass
            and direct_edge_retention_pass
        ):
            control_trap_leak_observed = False
            route_outcome_class = "interior_only_pass_source_start_weak"
        elif not target_final_continuity_pass:
            control_trap_leak_observed = False
            route_outcome_class = "target_final_continuity_failure"
        elif not post_start_endpoint_continuity_pass:
            control_trap_leak_observed = False
            route_outcome_class = "post_start_endpoint_continuity_failure"
        else:
            control_trap_leak_observed = False
            route_outcome_class = "partial_first_pass_signal"

        rows.append(
            {
                "route_contract_id": route_contract_id,
                "readout_plan_id": route_contract_id,
                "validation_unit_id": str(summary.validation_unit_id),
                "local_pair_id": str(summary.local_pair_id),
                "branch": str(contract.get("branch", "")),
                "start_condition": str(summary.start_condition),
                "seed": seed,
                "panel_role": str(contract.get("panel_role", "")),
                "evidence_role": evidence_role,
                "validation_stratum": str(contract.get("validation_stratum", "")),
                "limitation_axis_resolved": str(contract.get("limitation_axis_resolved", "")),
                "planned_route_family": str(summary.planned_route_family),
                "source_start_support_pass": source_start_support_pass,
                "post_start_endpoint_continuity_pass": post_start_endpoint_continuity_pass,
                "target_final_continuity_pass": target_final_continuity_pass,
                "target_final_bridge_exclusive_pass": target_final_bridge_exclusive_pass,
                "direct_edge_retention_pass": direct_edge_retention_pass,
                "same_seed_unknown_reclassification_pass": same_seed_unknown_reclassification_pass,
                "objective_debt_recovery_observed": objective_debt_recovery_observed,
                "all_positive_requirements_pass": all_positive_requirements_pass,
                "control_trap_leak_observed": control_trap_leak_observed,
                "route_outcome_class": route_outcome_class,
                "route_trace_class": str(summary.route_trace_class),
                "unknown_endpoint_step_count": int(
                    getattr(summary, "unknown_endpoint_step_count", 0)
                ),
                "max_objective_debt_from_start": float(
                    getattr(summary, "max_objective_debt_from_start", 0.0)
                ),
                "max_objective_recovery_from_min": float(
                    getattr(summary, "max_objective_recovery_from_min", 0.0)
                ),
                "first_endpoint_assignment": str(
                    getattr(summary, "first_endpoint_assignment", "")
                ),
                "final_endpoint_assignment": final_endpoint_assignment,
                "first_active_direct_edge_weight": float(
                    first.get("active_direct_edge_weight", 0.0)
                ),
                "final_active_direct_edge_weight": float(
                    final.get("active_direct_edge_weight", 0.0)
                ),
                "wall_claim_allowed_after_readout": False,
                "method_claim_allowed_after_readout": False,
                "quality_cost_claim_allowed_after_readout": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _pair_readout_results(
    *,
    contract_pairs: pd.DataFrame,
    route_results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair in contract_pairs.itertuples(index=False):
        local_pair_id = str(pair.local_pair_id)
        group = route_results[route_results["local_pair_id"].astype(str).eq(local_pair_id)]
        seed_route_count = int(len(group))
        ready_like_seed_route_pass_count = int(
            group["route_outcome_class"].astype(str).eq("conditional_ready_like_screen_pass").sum()
        )
        source_start_pass_count = int(group["source_start_support_pass"].map(_as_bool).sum())
        target_final_pass_count = int(group["target_final_continuity_pass"].map(_as_bool).sum())
        target_final_bridge_exclusive_pass_count = int(
            group["target_final_bridge_exclusive_pass"].map(_as_bool).sum()
        )
        control_leak_count = int(group["control_trap_leak_observed"].map(_as_bool).sum())
        if str(pair.evidence_role) == CONTROL_EVIDENCE_ROLE:
            pair_first_pass_result = (
                "control_trap_leaked" if control_leak_count else "control_trap_closed"
            )
            escalation_candidate = False
        elif seed_route_count > 0 and ready_like_seed_route_pass_count == seed_route_count:
            pair_first_pass_result = "all_seed_routes_conditional_ready_like_pass"
            escalation_candidate = True
        elif ready_like_seed_route_pass_count > 0:
            pair_first_pass_result = "partial_conditional_ready_like_pass"
            escalation_candidate = True
        else:
            pair_first_pass_result = "no_conditional_ready_like_pass"
            escalation_candidate = False
        rows.append(
            {
                "local_pair_id": local_pair_id,
                "branch": str(pair.branch),
                "panel_role": str(pair.panel_role),
                "evidence_role": str(pair.evidence_role),
                "validation_stratum": str(pair.validation_stratum),
                "limitation_axis_resolved": str(pair.limitation_axis_resolved),
                "route_readout_row_count": int(pair.route_readout_row_count),
                "seed_route_result_count": seed_route_count,
                "source_start_pass_count": source_start_pass_count,
                "target_final_pass_count": target_final_pass_count,
                "target_final_bridge_exclusive_pass_count": target_final_bridge_exclusive_pass_count,
                "ready_like_seed_route_pass_count": ready_like_seed_route_pass_count,
                "control_trap_leak_seed_route_count": control_leak_count,
                "route_outcome_class_counts": group["route_outcome_class"].value_counts().to_dict()
                if not group.empty
                else {},
                "pair_first_pass_result": pair_first_pass_result,
                "post_first_pass_escalation_candidate": escalation_candidate,
                "wall_claim_allowed_after_readout": False,
                "method_claim_allowed_after_readout": False,
                "quality_cost_claim_allowed_after_readout": False,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _control_trap_results(
    *,
    contract_controls: pd.DataFrame,
    route_results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for control in contract_controls.itertuples(index=False):
        local_pair_id = str(control.local_pair_id)
        group = route_results[route_results["local_pair_id"].astype(str).eq(local_pair_id)]
        leak_count = int(group["control_trap_leak_observed"].map(_as_bool).sum())
        rows.append(
            {
                "local_pair_id": local_pair_id,
                "branch": str(control.branch),
                "validation_stratum": str(control.validation_stratum),
                "limitation_axis_resolved": str(control.limitation_axis_resolved),
                "control_trap_id": str(control.control_trap_id),
                "control_trap_family": str(control.control_trap_family),
                "trap_question": str(control.trap_question),
                "positive_leak_signal": str(control.positive_leak_signal),
                "expected_guard_outcome": str(control.expected_guard_outcome),
                "seed_route_result_count": int(len(group)),
                "control_trap_leak_seed_route_count": leak_count,
                "control_trap_result": "leak_observed" if leak_count else "closed",
                "wall_claim_allowed_after_readout": False,
                "method_claim_allowed_after_readout": False,
                "quality_cost_claim_allowed_after_readout": False,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _gate_matrix(
    *,
    contract_gates: pd.DataFrame,
    execution_plan: pd.DataFrame,
    trace_rows: pd.DataFrame,
    route_results: pd.DataFrame,
    pair_results: pd.DataFrame,
    control_results: pd.DataFrame,
    step_config_count: int,
    seeds: int,
) -> pd.DataFrame:
    required_columns = {
        "route_trace_row_id",
        "route_contract_id",
        "step_index",
        "seed",
        "endpoint_assignment_by_step",
        "matches_original_anchor",
        "matches_expected_final_anchor",
        "active_direct_edge_weight",
        "objective_debt_from_start",
        "objective_recovery_from_min",
        "post_route_endpoint_assignment_available",
    }
    expected_step_configs = len(execution_plan) * len(SCHEDULES[EXECUTED_ROUTE_FAMILY])
    expected_trace_rows = expected_step_configs * int(seeds)
    upstream_status_counts = contract_gates["gate_status"].value_counts().to_dict()
    route_status_counts = route_results["route_outcome_class"].value_counts().to_dict()
    pair_status_counts = pair_results["pair_first_pass_result"].value_counts().to_dict()
    control_status_counts = control_results["control_trap_result"].value_counts().to_dict()
    rows = [
        _gate_row(
            "G1_upstream_readout_contract_gates_pass",
            "Did every upstream first-pass readout-contract gate pass?",
            upstream_status_counts,
            "all upstream readout-contract gates pass",
            bool(contract_gates["gate_status"].astype(str).eq("pass").all()),
        ),
        _gate_row(
            "G2_exact_36_route_scope",
            "Was execution restricted to the 36 predeclared fresh Axis B route rows?",
            f"execution_plan_rows={len(execution_plan)} executed_route_contracts={trace_rows['route_contract_id'].nunique()}",
            "36 route rows and no extra route contracts",
            len(execution_plan) == 36
            and trace_rows["route_contract_id"].nunique() == 36
            and set(trace_rows["route_contract_id"]) == set(execution_plan["route_contract_id"]),
        ),
        _gate_row(
            "G3_bridge_release_schedule_only",
            "Were all route rows expanded only into the bridge-release schedule?",
            f"route_step_configs={step_config_count} expected={expected_step_configs}",
            "36 * 5 route-step configs",
            step_config_count == expected_step_configs
            and execution_plan["planned_route_family"].astype(str).eq(EXECUTED_ROUTE_FAMILY).all(),
        ),
        _gate_row(
            "G4_seed_replicates_complete",
            "Did every route-step config run the requested same-seed replicates?",
            f"trace_rows={len(trace_rows)} expected={expected_trace_rows}",
            "route rows * 5 steps * requested seeds",
            len(trace_rows) == expected_trace_rows,
        ),
        _gate_row(
            "G5_required_trace_fields_materialized",
            "Did trace rows include required route-local readout fields?",
            sorted(required_columns & set(trace_rows.columns)),
            "all required trace columns present",
            required_columns.issubset(set(trace_rows.columns)),
        ),
        _gate_row(
            "G6_controls_first_readout_materialized",
            "Were control-trap rows read before ready-like promotion?",
            control_status_counts,
            "4 control traps materialized and no control leak",
            len(control_results) == 4
            and bool(control_results["control_trap_result"].astype(str).eq("closed").all()),
        ),
        _gate_row(
            "G7_pair_readout_materialized_without_wall_promotion",
            "Were pair-level readouts materialized while wall claims stayed closed?",
            pair_status_counts,
            "9 pair readouts and all wall claims false",
            len(pair_results) == 9
            and bool(pair_results["wall_claim_allowed_after_readout"].eq(False).all()),
        ),
        _gate_row(
            "G8_route_result_taxonomy_nonempty",
            "Did route-level first-pass taxonomy produce interpretable outcomes?",
            route_status_counts,
            "at least one route result and no missing route outcome class",
            not route_results.empty
            and not bool(route_results["route_outcome_class"].isna().any()),
        ),
        _gate_row(
            "G9_no_wall_method_quality_claim",
            "Are wall, method, and quality/cost claims still explicitly closed?",
            CLAIM_BOUNDARY,
            "all promotion flags remain false",
            bool(route_results["wall_claim_allowed_after_readout"].eq(False).all())
            and bool(route_results["method_claim_allowed_after_readout"].eq(False).all())
            and bool(route_results["quality_cost_claim_allowed_after_readout"].eq(False).all()),
        ),
    ]
    return pd.DataFrame(rows)


def _summary(
    *,
    readout_contract_dir: Path,
    local_ablation_dir: Path,
    output_dir: Path,
    execution_plan: pd.DataFrame,
    trace_rows: pd.DataFrame,
    seed_summary: pd.DataFrame,
    route_contract_summary: pd.DataFrame,
    route_results: pd.DataFrame,
    pair_results: pd.DataFrame,
    control_results: pd.DataFrame,
    gates: pd.DataFrame,
    step_config_count: int,
    candidate_pair_count: int,
    seeds: int,
) -> dict[str, Any]:
    ready_pairs = pair_results[pair_results["evidence_role"].astype(str).eq(READY_EVIDENCE_ROLE)]
    control_pairs = pair_results[
        pair_results["evidence_role"].astype(str).eq(CONTROL_EVIDENCE_ROLE)
    ]
    return {
        "schema": "nanoclustering_g4_8_fresh_axis_b_first_pass_trace_summary.v1",
        "status": RUN_STATUS,
        "readout_contract_dir": str(readout_contract_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "requested_seeds": int(seeds),
        "route_execution_plan_row_count": int(len(execution_plan)),
        "route_step_config_count": int(step_config_count),
        "trace_row_count": int(len(trace_rows)),
        "seed_route_summary_count": int(len(seed_summary)),
        "route_contract_summary_count": int(len(route_contract_summary)),
        "route_readout_result_count": int(len(route_results)),
        "pair_readout_result_count": int(len(pair_results)),
        "control_trap_result_count": int(len(control_results)),
        "candidate_pair_count_from_trace": int(candidate_pair_count),
        "execution_panel_role_counts": execution_plan["panel_role"].value_counts().to_dict(),
        "execution_evidence_role_counts": execution_plan["evidence_role"].value_counts().to_dict(),
        "route_outcome_class_counts": route_results["route_outcome_class"].value_counts().to_dict(),
        "pair_first_pass_result_counts": pair_results[
            "pair_first_pass_result"
        ].value_counts().to_dict(),
        "control_trap_result_counts": control_results["control_trap_result"].value_counts().to_dict(),
        "ready_pair_results": ready_pairs[
            ["local_pair_id", "pair_first_pass_result", "ready_like_seed_route_pass_count"]
        ].to_dict("records"),
        "control_pair_results": control_pairs[
            ["local_pair_id", "pair_first_pass_result", "control_trap_leak_seed_route_count"]
        ].to_dict("records"),
        "gate_status_counts": gates["gate_status"].value_counts().to_dict(),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"), "gate_id"
        ].tolist(),
        "interpretation": (
            "The fresh Axis B first-pass route rows were executed as local "
            "bridge-release interpolation traces and read through source-start, "
            "post-start, target-final, direct-edge-retention, and control-trap "
            "fields. This is still a route-local screen, not wall evidence."
        ),
        "recommended_next_gate": (
            "Inspect control leaks and ready-like seed-route pass patterns before "
            "deciding whether any pair deserves a stronger wall/pathway audit."
        ),
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
    control_results: pd.DataFrame,
    route_results: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 Fresh Axis B First-Pass Trace",
        "",
        f"- status: `{summary['status']}`",
        f"- route_execution_plan_row_count: {summary['route_execution_plan_row_count']}",
        f"- route_step_config_count: {summary['route_step_config_count']}",
        f"- trace_row_count: {summary['trace_row_count']}",
        f"- route_outcome_class_counts: {summary['route_outcome_class_counts']}",
        f"- pair_first_pass_result_counts: {summary['pair_first_pass_result_counts']}",
        f"- control_trap_result_counts: {summary['control_trap_result_counts']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Pair Results",
        "",
        _markdown_table(
            pair_results.sort_values(["evidence_role", "local_pair_id"], kind="mergesort"),
            [
                "local_pair_id",
                "branch",
                "evidence_role",
                "validation_stratum",
                "seed_route_result_count",
                "ready_like_seed_route_pass_count",
                "target_final_bridge_exclusive_pass_count",
                "control_trap_leak_seed_route_count",
                "pair_first_pass_result",
                "post_first_pass_escalation_candidate",
            ],
            max_rows=20,
        ),
        "",
        "## Control Trap Results",
        "",
        _markdown_table(
            control_results.sort_values("local_pair_id", kind="mergesort"),
            [
                "local_pair_id",
                "control_trap_family",
                "seed_route_result_count",
                "control_trap_leak_seed_route_count",
                "control_trap_result",
            ],
            max_rows=10,
        ),
        "",
        "## Route Results",
        "",
        _markdown_table(
            route_results.sort_values(
                ["evidence_role", "local_pair_id", "start_condition", "seed"],
                kind="mergesort",
            ),
            [
                "local_pair_id",
                "start_condition",
                "seed",
                "evidence_role",
                "source_start_support_pass",
                "post_start_endpoint_continuity_pass",
                "target_final_continuity_pass",
                "target_final_bridge_exclusive_pass",
                "direct_edge_retention_pass",
                "route_outcome_class",
                "route_trace_class",
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
            "This run executes the first-pass evidence screen. It is not a wall "
            "claim and it is not a method claim. Any escalation must separately "
            "audit distinct basin relations, direct-path availability, objective "
            "shape, support compatibility, and reproducibility."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    readout_contract_dir = Path(args.readout_contract_dir)
    local_ablation_dir = Path(args.local_ablation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contract_routes = _read_csv(readout_contract_dir / CONTRACT_ROUTE_READOUT_ROWS_CSV)
    contract_pairs = _read_csv(readout_contract_dir / CONTRACT_PAIR_READOUT_ROWS_CSV)
    contract_controls = _read_csv(readout_contract_dir / CONTRACT_CONTROL_TRAP_ROWS_CSV)
    contract_gates = _read_csv(readout_contract_dir / CONTRACT_GATE_MATRIX_CSV)

    execution_plan = _execution_plan(contract_routes)
    trace_rows, step_config_count, candidate_pair_count = _trace_rows(
        route_plan=execution_plan,
        contract_dir=readout_contract_dir,
        local_ablation_dir=local_ablation_dir,
        gamma=float(args.gamma),
        seeds=int(args.seeds),
        n_iterations=int(args.n_iterations),
        edge_chunk_size=int(args.edge_chunk_size),
    )
    for column, value in [
        ("route_execution_status", ROUTE_EXECUTION_STATUS),
        ("wall_promotion_status", WALL_PROMOTION_STATUS),
        ("method_status", METHOD_STATUS),
        ("claim_boundary", CLAIM_BOUNDARY),
        ("run_status", RUN_STATUS),
    ]:
        trace_rows[column] = value
    seed_summary = _seed_route_summary(trace_rows)
    route_contract_summary = _route_contract_summary(seed_summary)
    for frame in (seed_summary, route_contract_summary):
        frame["route_execution_status"] = ROUTE_EXECUTION_STATUS
        frame["wall_promotion_status"] = WALL_PROMOTION_STATUS
        frame["method_status"] = METHOD_STATUS
        frame["claim_boundary"] = CLAIM_BOUNDARY
        frame["run_status"] = RUN_STATUS

    route_results = _route_readout_results(
        contract_routes=contract_routes,
        seed_summary=seed_summary,
        trace_rows=trace_rows,
    )
    pair_results = _pair_readout_results(
        contract_pairs=contract_pairs,
        route_results=route_results,
    )
    control_results = _control_trap_results(
        contract_controls=contract_controls,
        route_results=route_results,
    )
    gates = _gate_matrix(
        contract_gates=contract_gates,
        execution_plan=execution_plan,
        trace_rows=trace_rows,
        route_results=route_results,
        pair_results=pair_results,
        control_results=control_results,
        step_config_count=step_config_count,
        seeds=int(args.seeds),
    )
    summary = _summary(
        readout_contract_dir=readout_contract_dir,
        local_ablation_dir=local_ablation_dir,
        output_dir=output_dir,
        execution_plan=execution_plan,
        trace_rows=trace_rows,
        seed_summary=seed_summary,
        route_contract_summary=route_contract_summary,
        route_results=route_results,
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
    _write_csv(route_results, output_dir / ROUTE_READOUT_RESULT_ROWS_CSV)
    _write_csv(pair_results, output_dir / PAIR_READOUT_RESULT_ROWS_CSV)
    _write_csv(control_results, output_dir / CONTROL_TRAP_RESULT_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_fresh_axis_b_first_pass_trace_config.v1",
        "readout_contract_dir": str(readout_contract_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "gamma": float(args.gamma),
        "seeds": int(args.seeds),
        "n_iterations": int(args.n_iterations),
        "edge_chunk_size": int(args.edge_chunk_size),
        "contract_route_family": CONTRACT_ROUTE_FAMILY,
        "executed_route_family": EXECUTED_ROUTE_FAMILY,
        "route_schedule": SCHEDULES[EXECUTED_ROUTE_FAMILY],
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
        control_results=control_results,
        route_results=route_results,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout-contract-dir", type=Path, default=DEFAULT_READOUT_CONTRACT_DIR)
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
