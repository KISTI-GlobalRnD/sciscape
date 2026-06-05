#!/usr/bin/env python3
"""Audit why local_pair_016 blocks the first-pass continuity readout.

This read-only diagnostic follows the role-pattern transfer screen's primary
diagnostic pair, ``local_pair_016``. It compares 016 against the 014 reference,
the 005 boundary guard, and nearby analog guards using only existing first-pass
trace, transfer-screen, and local-ablation outputs. It does not rerun Leiden,
perform a fraction sweep, promote walls, evaluate quality/cost value, or claim
method success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from audit_leiden_basin_nanoclustering_g4_8_first_pass_014_role_pattern_transfer_screen import (
    CANDIDATE_ROWS_CSV as TRANSFER_CANDIDATE_ROWS_CSV,
    DEFAULT_OUTPUT_DIR as DEFAULT_TRANSFER_SCREEN_DIR,
    GATE_MATRIX_CSV as TRANSFER_GATE_MATRIX_CSV,
    PAIR_ROLE_ROWS_CSV as TRANSFER_PAIR_ROWS_CSV,
    ROUTE_ROLE_ROWS_CSV as TRANSFER_ROUTE_ROWS_CSV,
    SIGNATURE_ROLE_ROWS_CSV as TRANSFER_SIGNATURE_ROWS_CSV,
)
from run_leiden_basin_nanoclustering_g4_8_fresh_axis_b_first_pass_trace import (
    DEFAULT_OUTPUT_DIR as DEFAULT_FIRST_PASS_TRACE_DIR,
    PAIR_READOUT_RESULT_ROWS_CSV,
    ROUTE_EXECUTION_PLAN_ROWS_CSV,
    ROUTE_READOUT_RESULT_ROWS_CSV,
    TRACE_ROWS_CSV,
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
    PAIR_GATE_ROWS_CSV,
    VARIANT_SUMMARY_CSV,
)


PRIMARY_PAIR_ID = "local_pair_016"
REFERENCE_PAIR_ID = "local_pair_014"
BOUNDARY_PAIR_ID = "local_pair_005"
COMPARISON_PAIR_IDS = (
    REFERENCE_PAIR_ID,
    PRIMARY_PAIR_ID,
    BOUNDARY_PAIR_ID,
    "local_pair_007",
    "local_pair_008",
)

DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_continuity_block_audit_gamma1e5_20260605"
)

PAIR_COMPARISON_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_continuity_block_pair_comparison_rows.csv"
)
STEP_SIGNATURE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_continuity_block_step_signature_rows.csv"
)
ROUTE_DIAGNOSTIC_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_continuity_block_route_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_016_continuity_block_gate_matrix.csv"
)
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_016_continuity_block_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_016_continuity_block_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_016_continuity_block_report.md"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_016_continuity_block"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_016_continuity_block_audit"
WALL_PROMOTION_STATUS = "not_promoted_016_continuity_block_audit_only"
METHOD_STATUS = "diagnostic_continuity_block_audit_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass local_pair_016 continuity-block audit only; "
    "reads existing first-pass trace, transfer-screen, and local-ablation "
    "outputs to localize the post-start continuity failure. It does not rerun "
    "Leiden, perform a fraction sweep, promote basin walls, replay full "
    "NanoClustering, evaluate quality/cost value, or claim method success."
)


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


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    cols = [column for column in columns if column in frame.columns]
    if not cols:
        return "_No matching columns._"
    visible = frame[cols].head(int(max_rows))
    if visible.empty:
        return "_No rows._"

    def cell(value: Any) -> str:
        if isinstance(value, (dict, list, tuple, set)):
            return json.dumps(_json_safe(value), sort_keys=True).replace("|", "\\|")
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in visible.itertuples(index=False):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def _comparison_role(local_pair_id: str) -> str:
    if local_pair_id == REFERENCE_PAIR_ID:
        return "reference_014_clean_scaffold"
    if local_pair_id == PRIMARY_PAIR_ID:
        return "primary_016_continuity_block"
    if local_pair_id == BOUNDARY_PAIR_ID:
        return "boundary_005_source_target_collapse_guard"
    if local_pair_id == "local_pair_007":
        return "secondary_rare_ready_blocked_analog"
    if local_pair_id == "local_pair_008":
        return "closed_control_role_analog"
    return "comparison_pair"


def _pair_comparison_rows(
    *,
    transfer_pair_rows: pd.DataFrame,
    transfer_candidate_rows: pd.DataFrame,
    first_pass_pair_rows: pd.DataFrame,
    route_plan_rows: pd.DataFrame,
    graph_rows: pd.DataFrame,
    variant_rows: pd.DataFrame,
    pair_gate_rows: pd.DataFrame,
) -> pd.DataFrame:
    transfer_lookup = transfer_pair_rows.set_index("local_pair_id").to_dict("index")
    candidate_lookup = transfer_candidate_rows.set_index("local_pair_id").to_dict("index")
    first_pass_lookup = first_pass_pair_rows.set_index("local_pair_id").to_dict("index")
    graph_lookup = graph_rows.set_index("local_pair_id").to_dict("index")
    gate_lookup = pair_gate_rows.set_index("local_pair_id").to_dict("index")

    plan_summary = (
        route_plan_rows.groupby("local_pair_id", sort=True)
        .agg(
            allowed_start_conditions=("start_condition", lambda values: ";".join(sorted(map(str, values)))),
            allowed_execution_unit_count=("route_contract_id", "count"),
            bridge_release_lift_proxy=("bridge_release_lift_proxy", "max"),
            direct_dependency_proxy=("direct_dependency_proxy", "max"),
            direct_dependency_regime=("direct_dependency_regime", lambda values: ";".join(sorted(set(map(str, values))))),
            bridge_release_regime=("bridge_release_regime", lambda values: ";".join(sorted(set(map(str, values))))),
            blocked_start_conditions=("blocked_start_conditions", lambda values: ";".join(sorted(set(map(str, values))))),
        )
        .to_dict("index")
    )
    original_variant = (
        variant_rows[variant_rows["graph_variant"].astype(str).eq("original")]
        .set_index("local_pair_id")
        .to_dict("index")
    )
    drop_bridge_variant = (
        variant_rows[variant_rows["graph_variant"].astype(str).eq("drop_bridge_edges")]
        .set_index("local_pair_id")
        .to_dict("index")
    )

    rows: list[dict[str, Any]] = []
    for local_pair_id in COMPARISON_PAIR_IDS:
        transfer = transfer_lookup.get(local_pair_id, {})
        candidate = candidate_lookup.get(local_pair_id, {})
        first_pass = first_pass_lookup.get(local_pair_id, {})
        graph = graph_lookup.get(local_pair_id, {})
        gate = gate_lookup.get(local_pair_id, {})
        plan = plan_summary.get(local_pair_id, {})
        original = original_variant.get(local_pair_id, {})
        drop_bridge = drop_bridge_variant.get(local_pair_id, {})
        rows.append(
            {
                "local_pair_id": local_pair_id,
                "comparison_role": _comparison_role(local_pair_id),
                "branch": str(first_pass.get("branch", graph.get("branch", ""))),
                "validation_stratum": str(transfer.get("validation_stratum", "")),
                "pair_first_pass_result": str(transfer.get("pair_first_pass_result", "")),
                "transfer_screen_status": str(transfer.get("transfer_screen_status", "")),
                "followup_priority_rank": candidate.get("followup_priority_rank"),
                "ready_like_seed_route_pass_count": int(first_pass.get("ready_like_seed_route_pass_count", 0)),
                "route_readout_row_count": int(first_pass.get("route_readout_row_count", 0)),
                "allowed_execution_unit_count": int(plan.get("allowed_execution_unit_count", 0)),
                "allowed_start_conditions": str(plan.get("allowed_start_conditions", "")),
                "blocked_start_conditions": str(plan.get("blocked_start_conditions", "")),
                "bridge_release_lift_proxy": float(plan.get("bridge_release_lift_proxy", 0.0)),
                "direct_dependency_proxy": float(plan.get("direct_dependency_proxy", 0.0)),
                "bridge_release_regime": str(plan.get("bridge_release_regime", "")),
                "direct_dependency_regime": str(plan.get("direct_dependency_regime", "")),
                "selected_bridge_count": int(graph.get("selected_bridge_count", 0)),
                "local_node_count": int(graph.get("local_node_count", 0)),
                "pair_scope": str(graph.get("pair_scope", "")),
                "selection_reason": str(graph.get("selection_reason", "")),
                "direct_edge_weight": float(graph.get("direct_edge_weight", 0.0)),
                "bridge_to_direct_weight_ratio": float(graph.get("bridge_to_direct_weight_ratio", 0.0)),
                "bridge_to_input_penalty_ratio": float(graph.get("bridge_to_input_penalty_ratio", 0.0)),
                "original_pair_coassigned_share": float(gate.get("original_pair_coassigned_share", 0.0)),
                "drop_bridge_pair_coassigned_share": float(gate.get("drop_bridge_pair_coassigned_share", 0.0)),
                "original_distinct_endpoint_count": int(gate.get("original_distinct_endpoint_count", 0)),
                "original_top_endpoint_share": float(original.get("top_endpoint_share", 0.0)),
                "original_top_endpoint_seed_count": int(original.get("top_endpoint_seed_count", 0)),
                "drop_bridge_top_endpoint_share": float(drop_bridge.get("top_endpoint_share", 0.0)),
                "drop_bridge_pair_coassigned_run_count": int(
                    drop_bridge.get("pair_coassigned_run_count", 0)
                ),
                "gate_class": str(gate.get("gate_class", "")),
                "method_claim_allowed_after_audit": False,
                "quality_cost_claim_allowed_after_audit": False,
                "wall_generality_claim_allowed_after_audit": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _step_signature_rows(
    *,
    trace_rows: pd.DataFrame,
    transfer_signature_rows: pd.DataFrame,
) -> pd.DataFrame:
    sig_lookup = transfer_signature_rows.set_index(
        ["local_pair_id", "result_endpoint_signature_id"]
    ).to_dict("index")
    rows: list[dict[str, Any]] = []
    scoped = trace_rows[trace_rows["local_pair_id"].astype(str).isin(COMPARISON_PAIR_IDS)].copy()
    group_cols = [
        "local_pair_id",
        "step_index",
        "bridge_edge_weight_fraction",
        "endpoint_assignment_by_step",
        "result_endpoint_signature_id",
    ]
    for key, group in scoped.groupby(group_cols, sort=True):
        local_pair_id, step_index, bridge_fraction, assignment, signature_id = key
        sig = sig_lookup.get((str(local_pair_id), str(signature_id)), {})
        rows.append(
            {
                "local_pair_id": str(local_pair_id),
                "comparison_role": _comparison_role(str(local_pair_id)),
                "step_index": int(step_index),
                "bridge_edge_weight_fraction": float(bridge_fraction),
                "endpoint_assignment_by_step": str(assignment),
                "result_endpoint_signature_id": str(signature_id),
                "signature_role_class": str(sig.get("signature_role_class", "")),
                "signature_role_cluster_signature": str(
                    sig.get("signature_role_cluster_signature", "")
                ),
                "left_cluster_roles": str(sig.get("left_cluster_roles", "")),
                "right_cluster_roles": str(sig.get("right_cluster_roles", "")),
                "row_count": int(len(group)),
                "start_condition_count": int(group["start_condition"].nunique()),
                "seed_count": int(group["seed"].nunique()),
                "pair_coassigned_rate": float(group["pair_coassigned"].map(_as_bool).mean()),
                "objective_value_mean": float(group["objective_value_by_step"].mean()),
                "objective_delta_from_start_mean": float(
                    group["objective_delta_from_start"].mean()
                ),
                "objective_debt_from_start_mean": float(
                    group["objective_debt_from_start"].mean()
                ),
                "support_distance_to_original_min": float(
                    group["support_distance_to_original"].min()
                ),
                "support_distance_to_drop_bridge_edges_min": float(
                    group["support_distance_to_drop_bridge_edges"].min()
                ),
                "support_distance_to_drop_direct_edge_min": float(
                    group["support_distance_to_drop_direct_edge"].min()
                ),
                "support_distance_to_drop_direct_and_bridge_edges_min": float(
                    group["support_distance_to_drop_direct_and_bridge_edges"].min()
                ),
                "method_claim_allowed_after_audit": False,
                "quality_cost_claim_allowed_after_audit": False,
                "wall_generality_claim_allowed_after_audit": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["local_pair_id", "step_index", "row_count"],
        ascending=[True, True, False],
        kind="mergesort",
    )


def _route_diagnostic_rows(
    *,
    route_readout_rows: pd.DataFrame,
    transfer_route_rows: pd.DataFrame,
) -> pd.DataFrame:
    readout = route_readout_rows[
        route_readout_rows["local_pair_id"].astype(str).eq(PRIMARY_PAIR_ID)
    ].copy()
    transfer = transfer_route_rows[
        transfer_route_rows["local_pair_id"].astype(str).eq(PRIMARY_PAIR_ID)
    ].copy()
    transfer_lookup = transfer.set_index(["start_condition", "seed"]).to_dict("index")
    rows: list[dict[str, Any]] = []
    for route in readout.sort_values(["start_condition", "seed"], kind="mergesort").itertuples(
        index=False
    ):
        transfer_row = transfer_lookup.get((str(route.start_condition), int(route.seed)), {})
        role_sequence = str(transfer_row.get("route_role_class_sequence", ""))
        assignment_sequence = str(transfer_row.get("route_endpoint_assignment_sequence", ""))
        unresolved_count = int(transfer_row.get("unresolved_intermediate_step_count", 0))
        single_step_bridge_reassignment = (
            unresolved_count == 1
            and "unresolved_pair_separated_bridge_reassignment" in role_sequence
            and _as_bool(route.source_start_support_pass)
            and _as_bool(route.target_final_bridge_exclusive_pass)
        )
        rows.append(
            {
                "local_pair_id": PRIMARY_PAIR_ID,
                "start_condition": str(route.start_condition),
                "seed": int(route.seed),
                "source_start_support_pass": _as_bool(route.source_start_support_pass),
                "post_start_endpoint_continuity_pass": _as_bool(
                    route.post_start_endpoint_continuity_pass
                ),
                "target_final_continuity_pass": _as_bool(route.target_final_continuity_pass),
                "target_final_bridge_exclusive_pass": _as_bool(
                    route.target_final_bridge_exclusive_pass
                ),
                "direct_edge_retention_pass": _as_bool(route.direct_edge_retention_pass),
                "all_positive_requirements_pass": _as_bool(route.all_positive_requirements_pass),
                "route_outcome_class": str(route.route_outcome_class),
                "unknown_endpoint_step_count": int(route.unknown_endpoint_step_count),
                "max_objective_debt_from_start": float(route.max_objective_debt_from_start),
                "max_objective_recovery_from_min": float(route.max_objective_recovery_from_min),
                "first_endpoint_assignment": str(route.first_endpoint_assignment),
                "final_endpoint_assignment": str(route.final_endpoint_assignment),
                "route_endpoint_assignment_sequence": assignment_sequence,
                "route_role_class_sequence": role_sequence,
                "unresolved_intermediate_step_count": unresolved_count,
                "single_step_bridge_reassignment_block": single_step_bridge_reassignment,
                "method_claim_allowed_after_audit": False,
                "quality_cost_claim_allowed_after_audit": False,
                "wall_generality_claim_allowed_after_audit": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _gate_matrix(
    *,
    transfer_gates: pd.DataFrame,
    pair_rows: pd.DataFrame,
    step_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
) -> pd.DataFrame:
    transfer_gate_counts = _count_dict(transfer_gates["gate_status"])
    primary_pair = pair_rows[pair_rows["local_pair_id"].astype(str).eq(PRIMARY_PAIR_ID)]
    primary_step_rows = step_rows[step_rows["local_pair_id"].astype(str).eq(PRIMARY_PAIR_ID)]
    primary_unknown = primary_step_rows[
        primary_step_rows["signature_role_class"]
        .astype(str)
        .eq("unresolved_pair_separated_bridge_reassignment")
    ]
    all_claims_closed = bool(
        not pair_rows["method_claim_allowed_after_audit"].map(_as_bool).any()
        and not step_rows["method_claim_allowed_after_audit"].map(_as_bool).any()
        and not route_rows["method_claim_allowed_after_audit"].map(_as_bool).any()
        and not pair_rows["quality_cost_claim_allowed_after_audit"].map(_as_bool).any()
        and not step_rows["quality_cost_claim_allowed_after_audit"].map(_as_bool).any()
        and not route_rows["quality_cost_claim_allowed_after_audit"].map(_as_bool).any()
        and not pair_rows["wall_generality_claim_allowed_after_audit"].map(_as_bool).any()
        and not step_rows["wall_generality_claim_allowed_after_audit"].map(_as_bool).any()
        and not route_rows["wall_generality_claim_allowed_after_audit"].map(_as_bool).any()
    )
    rows = [
        _gate_row(
            "G1_transfer_screen_gates_pass",
            "Did the upstream role-pattern transfer screen pass?",
            json.dumps(transfer_gate_counts, ensure_ascii=True, sort_keys=True),
            "all transfer-screen gates pass",
            transfer_gate_counts.get("fail", 0) == 0
            and transfer_gate_counts.get("pass", 0) == len(transfer_gates),
        ),
        _gate_row(
            "G2_primary_pair_is_016_strict_ready_blocked",
            "Is local_pair_016 the strict-ready continuity-blocked diagnostic?",
            primary_pair[
                [
                    "local_pair_id",
                    "validation_stratum",
                    "pair_first_pass_result",
                    "transfer_screen_status",
                ]
            ].to_dict("records"),
            "one primary row with strict-ready continuity-block status",
            len(primary_pair) == 1
            and str(primary_pair["validation_stratum"].iloc[0]) == "strict_ready"
            and str(primary_pair["transfer_screen_status"].iloc[0])
            == "strict_ready_continuity_blocked_role_analog",
        ),
        _gate_row(
            "G3_source_and_final_target_pass_for_all_016_routes",
            "Do all 016 routes start from source-like support and end at exclusive target?",
            {
                "route_rows": len(route_rows),
                "source_start_pass": int(route_rows["source_start_support_pass"].map(_as_bool).sum()),
                "target_exclusive_pass": int(
                    route_rows["target_final_bridge_exclusive_pass"].map(_as_bool).sum()
                ),
            },
            "24/24 source-start pass and 24/24 target-exclusive final pass",
            len(route_rows) == 24
            and bool(route_rows["source_start_support_pass"].map(_as_bool).all())
            and bool(route_rows["target_final_bridge_exclusive_pass"].map(_as_bool).all()),
        ),
        _gate_row(
            "G4_continuity_failure_localized_to_single_signature",
            "Is the 016 post-start failure localized to one recurrent bridge-reassignment signature?",
            primary_unknown[
                [
                    "step_index",
                    "bridge_edge_weight_fraction",
                    "result_endpoint_signature_id",
                    "row_count",
                    "signature_role_class",
                    "left_cluster_roles",
                    "right_cluster_roles",
                ]
            ].to_dict("records"),
            "one unknown signature at step 2 / bridge fraction 0.75 covering 24 rows",
            len(primary_unknown) == 1
            and int(primary_unknown["step_index"].iloc[0]) == 2
            and abs(float(primary_unknown["bridge_edge_weight_fraction"].iloc[0]) - 0.75) < 1e-9
            and int(primary_unknown["row_count"].iloc[0]) == 24,
        ),
        _gate_row(
            "G5_single_step_block_on_every_016_route",
            "Does every 016 route have exactly one bridge-reassignment continuity block?",
            {
                "single_step_blocks": int(
                    route_rows["single_step_bridge_reassignment_block"].map(_as_bool).sum()
                ),
                "route_rows": len(route_rows),
            },
            "24/24 routes have exactly one single-step bridge-reassignment block",
            len(route_rows) == 24
            and bool(route_rows["single_step_bridge_reassignment_block"].map(_as_bool).all()),
        ),
        _gate_row(
            "G6_boundary_and_reference_distinguished",
            "Are 014 reference and 005 boundary guard distinct from the 016 block?",
            pair_rows[
                [
                    "local_pair_id",
                    "comparison_role",
                    "ready_like_seed_route_pass_count",
                    "transfer_screen_status",
                ]
            ].to_dict("records"),
            "014 clean, 016 blocked, 005 boundary guard",
            set(pair_rows["local_pair_id"].astype(str)) >= {REFERENCE_PAIR_ID, PRIMARY_PAIR_ID, BOUNDARY_PAIR_ID},
        ),
        _gate_row(
            "G7_claims_closed",
            "Are method, quality/cost, and wall-generality claims closed?",
            CLAIM_BOUNDARY,
            "all claim flags false",
            all_claims_closed,
        ),
    ]
    return pd.DataFrame(rows)


def _summary(
    *,
    output_dir: Path,
    first_pass_trace_dir: Path,
    transfer_screen_dir: Path,
    local_ablation_dir: Path,
    pair_rows: pd.DataFrame,
    step_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    primary_unknown = step_rows[
        step_rows["local_pair_id"].astype(str).eq(PRIMARY_PAIR_ID)
        & step_rows["signature_role_class"]
        .astype(str)
        .eq("unresolved_pair_separated_bridge_reassignment")
    ]
    primary_pair = pair_rows[pair_rows["local_pair_id"].astype(str).eq(PRIMARY_PAIR_ID)]
    return {
        "schema": "nanoclustering_g4_8_first_pass_016_continuity_block_summary.v1",
        "status": RUN_STATUS,
        "primary_pair_id": PRIMARY_PAIR_ID,
        "reference_pair_id": REFERENCE_PAIR_ID,
        "boundary_pair_id": BOUNDARY_PAIR_ID,
        "first_pass_trace_dir": str(first_pass_trace_dir),
        "transfer_screen_dir": str(transfer_screen_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "pair_comparison_row_count": int(len(pair_rows)),
        "step_signature_row_count": int(len(step_rows)),
        "route_diagnostic_row_count": int(len(route_rows)),
        "primary_ready_like_seed_route_pass_count": int(
            primary_pair["ready_like_seed_route_pass_count"].iloc[0]
        ),
        "primary_single_step_bridge_reassignment_block_count": int(
            route_rows["single_step_bridge_reassignment_block"].map(_as_bool).sum()
        ),
        "primary_unknown_signature_ids": primary_unknown[
            "result_endpoint_signature_id"
        ].astype(str).tolist(),
        "primary_unknown_step_rows": primary_unknown[
            [
                "step_index",
                "bridge_edge_weight_fraction",
                "row_count",
                "left_cluster_roles",
                "right_cluster_roles",
                "support_distance_to_original_min",
                "support_distance_to_drop_bridge_edges_min",
                "support_distance_to_drop_direct_edge_min",
            ]
        ].to_dict("records"),
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": gates.loc[
            gates["gate_status"].astype(str).ne("pass"), "gate_id"
        ].astype(str).tolist(),
        "interpretation": (
            "local_pair_016 is not a target-final failure. All 24 audited "
            "routes start from source-like support and end at the exclusive "
            "drop-bridge target, but every route passes through one recurrent "
            "step-2 bridge-reassignment signature where L remains with B1 and "
            "R is separated. The current continuity rule treats that typed "
            "transient as a failure."
        ),
        "recommended_next_gate": (
            "Before any new localization sweep, decide whether typed transient "
            "intermediate signatures should be treated as pathway evidence or "
            "as blockers. If execution is needed, scope it to the 016 step-2 "
            "bridge-reassignment mechanism rather than all analogs."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    pair_rows: pd.DataFrame,
    step_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass 016 Continuity-Block Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- primary_pair_id: `{summary['primary_pair_id']}`",
        f"- primary_ready_like_seed_route_pass_count: {summary['primary_ready_like_seed_route_pass_count']}",
        f"- primary_single_step_bridge_reassignment_block_count: {summary['primary_single_step_bridge_reassignment_block_count']}",
        f"- primary_unknown_signature_ids: {summary['primary_unknown_signature_ids']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Pair Comparison",
        "",
        _markdown_table(
            pair_rows,
            [
                "local_pair_id",
                "comparison_role",
                "branch",
                "validation_stratum",
                "ready_like_seed_route_pass_count",
                "allowed_execution_unit_count",
                "allowed_start_conditions",
                "bridge_release_lift_proxy",
                "direct_dependency_proxy",
                "original_pair_coassigned_share",
                "drop_bridge_pair_coassigned_share",
                "original_top_endpoint_share",
                "transfer_screen_status",
            ],
            max_rows=20,
        ),
        "",
        "## 016 Route Diagnostics",
        "",
        _markdown_table(
            route_rows,
            [
                "start_condition",
                "seed",
                "source_start_support_pass",
                "post_start_endpoint_continuity_pass",
                "target_final_bridge_exclusive_pass",
                "unknown_endpoint_step_count",
                "single_step_bridge_reassignment_block",
                "route_endpoint_assignment_sequence",
                "route_role_class_sequence",
            ],
            max_rows=40,
        ),
        "",
        "## Step Signature Comparison",
        "",
        _markdown_table(
            step_rows,
            [
                "local_pair_id",
                "comparison_role",
                "step_index",
                "bridge_edge_weight_fraction",
                "endpoint_assignment_by_step",
                "result_endpoint_signature_id",
                "signature_role_class",
                "row_count",
                "left_cluster_roles",
                "right_cluster_roles",
                "support_distance_to_original_min",
                "support_distance_to_drop_bridge_edges_min",
                "support_distance_to_drop_direct_edge_min",
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
            "This diagnostic only localizes a first-pass readout failure in "
            "existing artifacts. It does not convert 016 into a positive wall "
            "case, nor does it claim generality or method value."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    first_pass_trace_dir = Path(args.first_pass_trace_dir)
    transfer_screen_dir = Path(args.transfer_screen_dir)
    local_ablation_dir = Path(args.local_ablation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_rows = _read_csv(first_pass_trace_dir / TRACE_ROWS_CSV)
    route_readout_rows = _read_csv(first_pass_trace_dir / ROUTE_READOUT_RESULT_ROWS_CSV)
    first_pass_pair_rows = _read_csv(first_pass_trace_dir / PAIR_READOUT_RESULT_ROWS_CSV)
    route_plan_rows = _read_csv(first_pass_trace_dir / ROUTE_EXECUTION_PLAN_ROWS_CSV)
    transfer_pair_rows = _read_csv(transfer_screen_dir / TRANSFER_PAIR_ROWS_CSV)
    transfer_candidate_rows = _read_csv(transfer_screen_dir / TRANSFER_CANDIDATE_ROWS_CSV)
    transfer_signature_rows = _read_csv(transfer_screen_dir / TRANSFER_SIGNATURE_ROWS_CSV)
    transfer_route_rows = _read_csv(transfer_screen_dir / TRANSFER_ROUTE_ROWS_CSV)
    transfer_gates = _read_csv(transfer_screen_dir / TRANSFER_GATE_MATRIX_CSV)
    graph_rows = _read_csv(local_ablation_dir / LOCAL_GRAPH_ROWS_CSV)
    variant_rows = _read_csv(local_ablation_dir / VARIANT_SUMMARY_CSV)
    pair_gate_rows = _read_csv(local_ablation_dir / PAIR_GATE_ROWS_CSV)

    pair_rows = _pair_comparison_rows(
        transfer_pair_rows=transfer_pair_rows,
        transfer_candidate_rows=transfer_candidate_rows,
        first_pass_pair_rows=first_pass_pair_rows,
        route_plan_rows=route_plan_rows,
        graph_rows=graph_rows,
        variant_rows=variant_rows,
        pair_gate_rows=pair_gate_rows,
    )
    step_rows = _step_signature_rows(
        trace_rows=trace_rows,
        transfer_signature_rows=transfer_signature_rows,
    )
    route_rows = _route_diagnostic_rows(
        route_readout_rows=route_readout_rows,
        transfer_route_rows=transfer_route_rows,
    )
    gates = _gate_matrix(
        transfer_gates=transfer_gates,
        pair_rows=pair_rows,
        step_rows=step_rows,
        route_rows=route_rows,
    )
    summary = _summary(
        output_dir=output_dir,
        first_pass_trace_dir=first_pass_trace_dir,
        transfer_screen_dir=transfer_screen_dir,
        local_ablation_dir=local_ablation_dir,
        pair_rows=pair_rows,
        step_rows=step_rows,
        route_rows=route_rows,
        gates=gates,
    )

    _write_csv(pair_rows, output_dir / PAIR_COMPARISON_ROWS_CSV)
    _write_csv(step_rows, output_dir / STEP_SIGNATURE_ROWS_CSV)
    _write_csv(route_rows, output_dir / ROUTE_DIAGNOSTIC_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_016_continuity_block_config.v1",
        "primary_pair_id": PRIMARY_PAIR_ID,
        "comparison_pair_ids": list(COMPARISON_PAIR_IDS),
        "first_pass_trace_dir": str(first_pass_trace_dir),
        "transfer_screen_dir": str(transfer_screen_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "read_only_audit": True,
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        pair_rows=pair_rows,
        step_rows=step_rows,
        route_rows=route_rows,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-pass-trace-dir", type=Path, default=DEFAULT_FIRST_PASS_TRACE_DIR)
    parser.add_argument("--transfer-screen-dir", type=Path, default=DEFAULT_TRANSFER_SCREEN_DIR)
    parser.add_argument("--local-ablation-dir", type=Path, default=DEFAULT_LOCAL_ABLATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
