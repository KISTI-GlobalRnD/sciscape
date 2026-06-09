#!/usr/bin/env python3
"""Design the first-pass local_pair_016 object-wall transfer contract.

This consumes the 014/016 reconciliation, the 014 pathway-probe contract, the
016 continuity-block audit, and the basin-state assignment surface. It freezes
the next narrow transfer probe: move the 014 direct-only/recovery-loop
vocabulary onto local_pair_016 only where 016 has allowed starts, while retaining
local_pair_005 as a boundary guard.

This is a contract design only. It does not run Leiden, execute routes, promote
walls, evaluate quality/cost value, replay full NanoClustering, or claim method
success.
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


DEFAULT_RECONCILIATION_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_014_016_surface_reconciliation_gamma1e5_20260607"
)
DEFAULT_014_CONTRACT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_014_pathway_probe_contract_gamma1e5_20260604"
)
DEFAULT_016_CONTINUITY_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_continuity_block_audit_gamma1e5_20260605"
)
DEFAULT_ASSIGNMENT_SURFACE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_basin_state_assignment_surface_gamma1e5_20260606"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_object_wall_transfer_contract_gamma1e5_20260607"
)

RECONCILIATION_SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_014_016_surface_reconciliation_summary.json"
)
RECONCILIATION_GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_014_016_surface_reconciliation_gate_matrix.csv"
)
RECONCILIATION_PAIR_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_016_surface_reconciliation_pair_rows.csv"
)
CONTRACT_014_ROUTE_PLAN_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_pathway_probe_contract_route_plan_rows.csv"
)
CONTRACT_014_RULE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_pathway_probe_contract_rule_rows.csv"
)
CONTRACT_014_SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_014_pathway_probe_contract_summary.json"
)
CONTINUITY_PAIR_COMPARISON_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_continuity_block_pair_comparison_rows.csv"
)
CONTINUITY_SUMMARY_JSON = "nanoclustering_g4_8_first_pass_016_continuity_block_summary.json"
ASSIGNMENT_PAIR_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_basin_state_assignment_surface_pair_rows.csv"
)
ASSIGNMENT_SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_basin_state_assignment_surface_summary.json"
)

RULE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_contract_rule_rows.csv"
)
PAIR_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_contract_pair_rows.csv"
)
ROUTE_PLAN_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_contract_route_plan_rows.csv"
)
BOUNDARY_GUARD_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_contract_boundary_guard_rows.csv"
)
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_contract_decision_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_contract_gate_matrix.csv"
)
SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_contract_summary.json"
)
CONFIG_JSON = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_contract_config.json"
)
REPORT_MD = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_contract_report.md"
)

PAIR_014 = "local_pair_014"
PAIR_016 = "local_pair_016"
BOUNDARY_PAIR = "local_pair_005"
FOCUS_PAIR_IDS = (PAIR_014, PAIR_016, BOUNDARY_PAIR)

POSITIVE_ALLOWED_STARTS = ("bridges_to_left", "pair_together", "singleton")
POSITIVE_BLOCKED_STARTS = ("all_local_together", "bridges_to_right")
BOUNDARY_GUARD_STARTS = (
    "all_local_together",
    "bridges_to_right",
    "pair_together",
    "singleton",
)

RUN_STATUS = "designed_nanoclustering_g4_8_first_pass_016_object_wall_transfer_contract"
ROUTE_EXECUTION_STATUS = "design_only_not_executed"
WALL_PROMOTION_STATUS = "not_promoted_object_wall_transfer_contract_only"
METHOD_STATUS = "object_wall_transfer_contract_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass local_pair_016 object-wall transfer contract "
    "design only; reads the 014/016 reconciliation, 014 pathway-probe contract, "
    "016 continuity-block audit, and basin-state assignment surface. It does not "
    "run Leiden, execute routes, promote walls, evaluate quality/cost value, "
    "replay full NanoClustering, or claim method success."
)

REQUIRED_MEASUREMENTS = (
    "endpoint_assignment_by_step",
    "endpoint_object_assignment_by_step",
    "typed_transient_assignment_by_step",
    "object_identity_transfer_status",
    "direct_edge_retained_all_steps",
    "bridge_fraction_by_step",
    "first_exclusive_target_step",
    "objective_debt_from_start",
    "objective_recovery_from_min",
    "accepted_recovery_after_min",
    "support_incompatibility_by_step",
    "boundary_control_leak_status",
)

ACCEPTANCE_RULES: tuple[dict[str, str], ...] = (
    {
        "rule_id": "T1",
        "rule_group": "scope",
        "rule_question": "Is source endpoint identity available at baseline?",
        "acceptance_requirement": (
            "source endpoint identity/readout is available before interpreting "
            "any 016 transfer route"
        ),
        "claim_effect": "required_entry_condition",
    },
    {
        "rule_id": "T2",
        "rule_group": "object_readout",
        "rule_question": "Is endpoint-object assignment available at every step?",
        "acceptance_requirement": (
            "endpoint_object_assignment_by_step has no missing endpoint-object rows "
            "for a positive 016 route"
        ),
        "claim_effect": "blocks_route_positive_without_object_identity",
    },
    {
        "rule_id": "T3",
        "rule_group": "boundary_control",
        "rule_question": "Does the 005 boundary remain non-positive/no-leak?",
        "acceptance_requirement": (
            "local_pair_005 boundary routes must not satisfy positive 016 object-wall "
            "acceptance under the transferred probe families"
        ),
        "claim_effect": "false_positive_guard",
    },
    {
        "rule_id": "T4",
        "rule_group": "direct_only",
        "rule_question": "Is target-object availability shown without bridge support?",
        "acceptance_requirement": (
            "direct-only route keeps the direct edge, suppresses bridge support, and "
            "reaches an exclusive target object or an explicitly typed object-identity block"
        ),
        "claim_effect": "required_before_direct_path_language",
    },
    {
        "rule_id": "T5",
        "rule_group": "recovery_loop",
        "rule_question": "Does a recovery-loop route show typed transition/recovery?",
        "acceptance_requirement": (
            "recovery-loop route reports source-to-target object transition and "
            "accepted objective recovery after the debt minimum, or an explicitly "
            "typed transient block"
        ),
        "claim_effect": "required_before_wall_language",
    },
    {
        "rule_id": "T6",
        "rule_group": "boundary_control",
        "rule_question": "Is boundary guard status closed?",
        "acceptance_requirement": "boundary_guard_status == closed_for_all_005_guard_rows",
        "claim_effect": "required_false_positive_guard",
    },
    {
        "rule_id": "T7",
        "rule_group": "claim_boundary",
        "rule_question": "Are method, quality, full-replay, pathway, and wall claims closed?",
        "acceptance_requirement": (
            "contract alone cannot promote method, quality/cost, full replay, pathway, "
            "or wall labels"
        ),
        "claim_effect": "promotion_blocked_until_execution_and_acceptance",
    },
    {
        "rule_id": "T8",
        "rule_group": "typed_transient",
        "rule_question": "Is every transient explicitly typed before any positive wall label?",
        "acceptance_requirement": (
            "typed_transient_assignment_by_step must classify transient states as "
            "pathway_intermediate, object_identity_blocker, boundary_leak, or unknown"
        ),
        "claim_effect": "prevents_untyped_morphology_promotion",
    },
)

POSITIVE_TRANSFER_FAMILIES: tuple[dict[str, str], ...] = (
    {
        "planned_route_family": "first_pass_016_recovery_loop_probe",
        "source_planned_route_family": "first_pass_014_recovery_loop_probe",
        "route_family_role": "primary_recovery_probe",
        "expected_endpoint_pattern": (
            "source_object_to_exclusive_target_object_with_recovery_or_explicit_typed_transient_block"
        ),
        "acceptance_rule_ids": "T1;T2;T5;T7;T8",
        "probe_question": (
            "Can 016 reproduce the 014 recovery-loop vocabulary as a typed object transition, "
            "or does it expose a typed object-identity/transient block?"
        ),
    },
    {
        "planned_route_family": "first_pass_016_direct_only_target_availability_probe",
        "source_planned_route_family": (
            "first_pass_014_direct_only_target_availability_probe"
        ),
        "route_family_role": "independent_direct_path_probe",
        "expected_endpoint_pattern": (
            "exclusive_target_object_available_without_bridge_support_or_explicit_object_identity_block"
        ),
        "acceptance_rule_ids": "T1;T2;T4;T7;T8",
        "probe_question": (
            "Is 016 target-object availability visible when bridge support is suppressed "
            "and the direct edge is retained?"
        ),
    },
)

BOUNDARY_TRANSFER_FAMILIES: tuple[dict[str, str], ...] = (
    {
        "planned_route_family": "first_pass_005_boundary_recovery_loop_guard",
        "source_planned_route_family": "first_pass_005_boundary_recovery_loop_guard",
        "route_family_role": "boundary_recovery_control",
        "expected_endpoint_pattern": "source_target_collapse_or_mixed_boundary_not_positive",
        "acceptance_rule_ids": "T3;T6;T7;T8",
        "probe_question": (
            "Does the 005 boundary remain non-positive under the transferred recovery-loop schedule?"
        ),
    },
    {
        "planned_route_family": "first_pass_005_boundary_direct_only_guard",
        "source_planned_route_family": "first_pass_005_boundary_direct_only_guard",
        "route_family_role": "boundary_direct_path_control",
        "expected_endpoint_pattern": "boundary_target_availability_must_not_promote_positive",
        "acceptance_rule_ids": "T3;T6;T7;T8",
        "probe_question": (
            "Does the 005 boundary avoid becoming positive under the transferred direct-only schedule?"
        ),
    },
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _split_semicolon(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _safe_scalar(row: pd.Series, column: str, default: Any = "") -> Any:
    if column not in row or pd.isna(row[column]):
        return default
    value = row[column]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _row_for_pair(rows: pd.DataFrame, pair_id: str) -> pd.Series:
    scoped = rows[rows["local_pair_id"].astype(str).eq(pair_id)]
    if scoped.empty:
        raise ValueError(f"Missing row for {pair_id}")
    return scoped.iloc[0]


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
        "observed": json.dumps(_json_safe(observed), sort_keys=True),
        "minimum_or_rule": minimum_or_rule,
        "gate_status": "pass" if bool(passed) else "fail",
    }


def _source_family_lookup(contract_014_route_plan: pd.DataFrame) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for family in (
        "first_pass_014_recovery_loop_probe",
        "first_pass_014_direct_only_target_availability_probe",
        "first_pass_005_boundary_recovery_loop_guard",
        "first_pass_005_boundary_direct_only_guard",
    ):
        scoped = contract_014_route_plan[
            contract_014_route_plan["planned_route_family"].astype(str).eq(family)
        ]
        if scoped.empty:
            raise ValueError(f"Missing 014 contract route family: {family}")
        row = scoped.iloc[0]
        lookup[family] = {
            "planned_intervention_schedule": row["planned_intervention_schedule"],
            "source_runner_support_status": row["runner_support_status"],
            "source_expected_endpoint_pattern": row["expected_endpoint_pattern"],
            "source_acceptance_rule_ids": row["acceptance_rule_ids"],
        }
    return lookup


def _rule_rows() -> pd.DataFrame:
    rows = pd.DataFrame(list(ACCEPTANCE_RULES))
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _pair_rows(
    *,
    reconciliation_pair_rows: pd.DataFrame,
    continuity_pair_rows: pd.DataFrame,
    assignment_pair_rows: pd.DataFrame,
) -> pd.DataFrame:
    assignment_lookup = {
        str(row["local_pair_id"]): row for _, row in assignment_pair_rows.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for pair_id in FOCUS_PAIR_IDS:
        continuity = _row_for_pair(continuity_pair_rows, pair_id)
        reconciliation_scoped = reconciliation_pair_rows[
            reconciliation_pair_rows["local_pair_id"].astype(str).eq(pair_id)
        ]
        reconciliation = reconciliation_scoped.iloc[0] if not reconciliation_scoped.empty else pd.Series()
        assignment = assignment_lookup.get(pair_id, pd.Series())
        rows.append(
            {
                "local_pair_id": pair_id,
                "contract_pair_role": {
                    PAIR_014: "source_vocabulary_pair",
                    PAIR_016: "positive_object_wall_transfer_candidate",
                    BOUNDARY_PAIR: "boundary_collapse_control",
                }[pair_id],
                "comparison_role": _safe_scalar(continuity, "comparison_role"),
                "branch": _safe_scalar(continuity, "branch"),
                "pair_scope": _safe_scalar(continuity, "pair_scope"),
                "selection_reason": _safe_scalar(continuity, "selection_reason"),
                "gate_class": _safe_scalar(continuity, "gate_class"),
                "allowed_start_conditions": _safe_scalar(
                    continuity, "allowed_start_conditions"
                ),
                "blocked_start_conditions": _safe_scalar(
                    continuity, "blocked_start_conditions"
                ),
                "direct_edge_weight": _safe_scalar(continuity, "direct_edge_weight"),
                "bridge_to_direct_weight_ratio": _safe_scalar(
                    continuity, "bridge_to_direct_weight_ratio"
                ),
                "original_pair_coassigned_share": _safe_scalar(
                    continuity, "original_pair_coassigned_share"
                ),
                "drop_bridge_pair_coassigned_share": _safe_scalar(
                    continuity, "drop_bridge_pair_coassigned_share"
                ),
                "drop_bridge_pair_coassigned_run_count": _safe_scalar(
                    continuity, "drop_bridge_pair_coassigned_run_count"
                ),
                "reconciliation_surface_class": _safe_scalar(
                    reconciliation, "surface_reconciliation_class"
                ),
                "reconciliation_interpretation": _safe_scalar(
                    reconciliation, "interpretation"
                ),
                "assignment_surface_basin_state_status": _safe_scalar(
                    assignment, "basin_state_assignment_class"
                ),
                "assignment_surface_wall_evidence_status": _safe_scalar(
                    assignment, "wall_evidence_status"
                ),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "wall_claim_allowed_after_contract": False,
                "method_claim_allowed_after_contract": False,
                "quality_cost_claim_allowed_after_contract": False,
                "full_replay_claim_allowed_after_contract": False,
                "required_measurements": ";".join(REQUIRED_MEASUREMENTS),
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _route_plan_rows(
    *,
    continuity_pair_rows: pd.DataFrame,
    contract_014_route_plan: pd.DataFrame,
) -> pd.DataFrame:
    source_lookup = _source_family_lookup(contract_014_route_plan)
    pair_016 = _row_for_pair(continuity_pair_rows, PAIR_016)
    boundary = _row_for_pair(continuity_pair_rows, BOUNDARY_PAIR)
    rows: list[dict[str, Any]] = []

    for start_condition in POSITIVE_ALLOWED_STARTS:
        for order, family in enumerate(POSITIVE_TRANSFER_FAMILIES, start=1):
            source_family = source_lookup[family["source_planned_route_family"]]
            route_contract_id = (
                f"{PAIR_016}__{start_condition}__{family['planned_route_family']}"
            )
            rows.append(
                {
                    "route_contract_id": route_contract_id,
                    "local_pair_id": PAIR_016,
                    "branch": _safe_scalar(pair_016, "branch"),
                    "start_condition": start_condition,
                    "contract_pair_role": "positive_object_wall_transfer_candidate",
                    "route_family_order": order,
                    "source_vocabulary_pair_id": PAIR_014,
                    "boundary_guard_pair_id": BOUNDARY_PAIR,
                    **family,
                    "planned_intervention_schedule": source_family[
                        "planned_intervention_schedule"
                    ],
                    "source_runner_support_status": source_family[
                        "source_runner_support_status"
                    ],
                    "runner_support_status": ROUTE_EXECUTION_STATUS,
                    "current_allowed_start_conditions": _safe_scalar(
                        pair_016, "allowed_start_conditions"
                    ),
                    "current_blocked_start_conditions": _safe_scalar(
                        pair_016, "blocked_start_conditions"
                    ),
                    "current_direct_edge_weight": _safe_scalar(
                        pair_016, "direct_edge_weight"
                    ),
                    "current_bridge_to_direct_weight_ratio": _safe_scalar(
                        pair_016, "bridge_to_direct_weight_ratio"
                    ),
                    "current_original_pair_coassigned_share": _safe_scalar(
                        pair_016, "original_pair_coassigned_share"
                    ),
                    "current_drop_bridge_pair_coassigned_share": _safe_scalar(
                        pair_016, "drop_bridge_pair_coassigned_share"
                    ),
                    "new_route_execution_required": True,
                    "counts_as_positive_if_accepted": True,
                    "wall_claim_allowed_after_contract": False,
                    "pathway_claim_allowed_after_contract": False,
                    "method_claim_allowed_after_contract": False,
                    "quality_cost_claim_allowed_after_contract": False,
                    "full_replay_claim_allowed_after_contract": False,
                    "required_measurements": ";".join(REQUIRED_MEASUREMENTS),
                    "route_execution_status": ROUTE_EXECUTION_STATUS,
                    "wall_promotion_status": WALL_PROMOTION_STATUS,
                    "method_status": METHOD_STATUS,
                    "run_status": RUN_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )

    for start_condition in BOUNDARY_GUARD_STARTS:
        for order, family in enumerate(BOUNDARY_TRANSFER_FAMILIES, start=1):
            source_family = source_lookup[family["source_planned_route_family"]]
            route_contract_id = (
                f"{BOUNDARY_PAIR}__{start_condition}__{family['planned_route_family']}"
            )
            rows.append(
                {
                    "route_contract_id": route_contract_id,
                    "local_pair_id": BOUNDARY_PAIR,
                    "branch": _safe_scalar(boundary, "branch"),
                    "start_condition": start_condition,
                    "contract_pair_role": "boundary_collapse_control",
                    "route_family_order": order,
                    "source_vocabulary_pair_id": PAIR_014,
                    "boundary_guard_pair_id": BOUNDARY_PAIR,
                    **family,
                    "planned_intervention_schedule": source_family[
                        "planned_intervention_schedule"
                    ],
                    "source_runner_support_status": source_family[
                        "source_runner_support_status"
                    ],
                    "runner_support_status": ROUTE_EXECUTION_STATUS,
                    "current_allowed_start_conditions": _safe_scalar(
                        boundary, "allowed_start_conditions"
                    ),
                    "current_blocked_start_conditions": _safe_scalar(
                        boundary, "blocked_start_conditions"
                    ),
                    "current_direct_edge_weight": _safe_scalar(
                        boundary, "direct_edge_weight"
                    ),
                    "current_bridge_to_direct_weight_ratio": _safe_scalar(
                        boundary, "bridge_to_direct_weight_ratio"
                    ),
                    "current_original_pair_coassigned_share": _safe_scalar(
                        boundary, "original_pair_coassigned_share"
                    ),
                    "current_drop_bridge_pair_coassigned_share": _safe_scalar(
                        boundary, "drop_bridge_pair_coassigned_share"
                    ),
                    "new_route_execution_required": True,
                    "counts_as_positive_if_accepted": False,
                    "wall_claim_allowed_after_contract": False,
                    "pathway_claim_allowed_after_contract": False,
                    "method_claim_allowed_after_contract": False,
                    "quality_cost_claim_allowed_after_contract": False,
                    "full_replay_claim_allowed_after_contract": False,
                    "required_measurements": ";".join(REQUIRED_MEASUREMENTS),
                    "route_execution_status": ROUTE_EXECUTION_STATUS,
                    "wall_promotion_status": WALL_PROMOTION_STATUS,
                    "method_status": METHOD_STATUS,
                    "run_status": RUN_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["contract_pair_role", "local_pair_id", "start_condition", "route_family_order"],
        kind="mergesort",
    ).reset_index(drop=True)


def _boundary_guard_rows(route_plan: pd.DataFrame) -> pd.DataFrame:
    rows = route_plan[
        route_plan["contract_pair_role"].astype(str).eq("boundary_collapse_control")
    ].copy()
    rows["boundary_guard_id"] = rows["route_contract_id"].astype(str)
    rows["boundary_guard_family"] = "source_target_collapse_boundary"
    rows["positive_leak_signal"] = (
        "005 route satisfies 016 positive direct-only or recovery-loop acceptance"
    )
    rows["expected_guard_outcome"] = "must_not_count_as_positive_object_wall_evidence"
    rows["boundary_guard_status"] = "predeclared_closed_until_execution"
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.reset_index(drop=True)


def _decision_rows() -> pd.DataFrame:
    rows = [
        {
            "decision_id": "D1",
            "decision": "design_only_no_execution",
            "rationale": (
                "The contract fixes rows and gates only; it does not execute transferred routes."
            ),
        },
        {
            "decision_id": "D2",
            "decision": "016_positive_rows_use_allowed_starts_only",
            "rationale": (
                "016 continuity evidence allows bridges_to_left, pair_together, and singleton; "
                "blocked starts are excluded from positive transfer rows."
            ),
        },
        {
            "decision_id": "D3",
            "decision": "005_boundary_guard_retained",
            "rationale": (
                "The same transferred schedules must be checked against the 005 source/target "
                "collapse guard before any 016 object-wall label can be considered."
            ),
        },
        {
            "decision_id": "D4",
            "decision": "typed_transient_classification_mandatory",
            "rationale": (
                "016 may fail by object-identity or transient-state ambiguity; every transient "
                "must be typed before positive wall language is allowed."
            ),
        },
        {
            "decision_id": "D5",
            "decision": "no_pathway_wall_method_promotion",
            "rationale": (
                "The contract can only authorize a narrow next execution gate, not a pathway, "
                "wall, method, quality, or full-replay claim."
            ),
        },
    ]
    frame = pd.DataFrame(rows)
    frame["run_status"] = RUN_STATUS
    frame["claim_boundary"] = CLAIM_BOUNDARY
    return frame


def _gate_matrix(
    *,
    reconciliation_summary: dict[str, Any],
    reconciliation_gates: pd.DataFrame,
    rule_rows: pd.DataFrame,
    route_plan: pd.DataFrame,
    boundary_guards: pd.DataFrame,
    continuity_pair_rows: pd.DataFrame,
) -> pd.DataFrame:
    positive_routes = route_plan[
        route_plan["contract_pair_role"].astype(str).eq(
            "positive_object_wall_transfer_candidate"
        )
    ]
    boundary_routes = route_plan[
        route_plan["contract_pair_role"].astype(str).eq("boundary_collapse_control")
    ]
    pair_016 = _row_for_pair(continuity_pair_rows, PAIR_016)
    allowed_016 = _split_semicolon(pair_016["allowed_start_conditions"])
    blocked_016 = _split_semicolon(pair_016["blocked_start_conditions"])
    positive_starts = sorted(positive_routes["start_condition"].astype(str).unique())
    required_transient_classes = {
        "pathway_intermediate",
        "object_identity_blocker",
        "boundary_leak",
        "unknown",
    }
    t8_requirement = str(
        rule_rows.loc[rule_rows["rule_id"].astype(str).eq("T8"), "acceptance_requirement"]
        .iloc[0]
    )
    rows = [
        _gate_row(
            "G1_sources_readable_and_reconciliation_clean",
            "Are source artifacts readable and reconciliation gates clean?",
            {
                "reconciliation_failed_gates": reconciliation_summary.get("failed_gates", []),
                "reconciliation_gate_status_counts": _count_dict(
                    reconciliation_gates["gate_status"]
                ),
            },
            "reconciliation failed_gates empty and all reconciliation gates pass",
            len(reconciliation_summary.get("failed_gates", [])) == 0
            and bool(reconciliation_gates["gate_status"].astype(str).eq("pass").all()),
        ),
        _gate_row(
            "G2_016_focus_and_014_vocabulary_transfer_explicit",
            "Is the contract scoped to 016 while transferring 014 vocabulary?",
            {
                "positive_pair_ids": sorted(
                    positive_routes["local_pair_id"].astype(str).unique().tolist()
                ),
                "source_vocabulary_pair_ids": sorted(
                    positive_routes["source_vocabulary_pair_id"].astype(str).unique().tolist()
                ),
            },
            "positive rows are 016 and source_vocabulary_pair_id is 014",
            sorted(positive_routes["local_pair_id"].astype(str).unique().tolist())
            == [PAIR_016]
            and sorted(
                positive_routes["source_vocabulary_pair_id"].astype(str).unique().tolist()
            )
            == [PAIR_014],
        ),
        _gate_row(
            "G3_016_positive_rows_allowed_starts_only",
            "Do 016 positive route rows contain exactly allowed starts and no blocked starts?",
            {
                "positive_route_count": int(len(positive_routes)),
                "positive_starts": positive_starts,
                "allowed_016": allowed_016,
                "blocked_016": blocked_016,
            },
            "6 positive rows, starts bridges_to_left/pair_together/singleton only",
            len(positive_routes) == 6
            and set(positive_starts) == set(POSITIVE_ALLOWED_STARTS)
            and set(positive_starts).issubset(set(allowed_016))
            and set(positive_starts).isdisjoint(set(blocked_016))
            and set(POSITIVE_BLOCKED_STARTS).issubset(set(blocked_016)),
        ),
        _gate_row(
            "G4_005_boundary_guard_retained",
            "Are 005 boundary guard rows retained under both transferred families?",
            {
                "boundary_route_count": int(len(boundary_routes)),
                "boundary_guard_count": int(len(boundary_guards)),
                "boundary_starts": sorted(
                    boundary_routes["start_condition"].astype(str).unique().tolist()
                ),
            },
            "8 boundary guard rows over 4 starts and 2 families",
            len(boundary_routes) == 8
            and len(boundary_guards) == 8
            and set(boundary_routes["start_condition"].astype(str).unique())
            == set(BOUNDARY_GUARD_STARTS)
            and bool(boundary_routes["counts_as_positive_if_accepted"].eq(False).all()),
        ),
        _gate_row(
            "G5_typed_transient_classification_required",
            "Does the contract require typed transient classification?",
            {
                "rule_ids": rule_rows["rule_id"].astype(str).tolist(),
                "T8_acceptance_requirement": t8_requirement,
            },
            "T8 present with four required transient classes",
            "T8" in set(rule_rows["rule_id"].astype(str))
            and all(required in t8_requirement for required in required_transient_classes),
        ),
        _gate_row(
            "G6_no_execution_or_promotion_claims",
            "Are execution and promotion claims closed?",
            {
                "route_execution_status_counts": _count_dict(
                    route_plan["route_execution_status"]
                ),
                "wall_claim_flags": _count_dict(
                    route_plan["wall_claim_allowed_after_contract"]
                ),
                "pathway_claim_flags": _count_dict(
                    route_plan["pathway_claim_allowed_after_contract"]
                ),
            },
            "all rows design-only and all promotion flags false",
            bool(route_plan["route_execution_status"].astype(str).eq(ROUTE_EXECUTION_STATUS).all())
            and bool(route_plan["wall_claim_allowed_after_contract"].eq(False).all())
            and bool(route_plan["pathway_claim_allowed_after_contract"].eq(False).all())
            and bool(route_plan["method_claim_allowed_after_contract"].eq(False).all())
            and bool(route_plan["quality_cost_claim_allowed_after_contract"].eq(False).all())
            and bool(route_plan["full_replay_claim_allowed_after_contract"].eq(False).all()),
        ),
        _gate_row(
            "G7_next_execution_gate_is_narrow",
            "Is the next executable plan narrow and predeclared?",
            {
                "route_plan_row_count": int(len(route_plan)),
                "positive_route_count": int(len(positive_routes)),
                "boundary_route_count": int(len(boundary_routes)),
            },
            "14 total rows: 6 positive 016 rows plus 8 boundary 005 rows",
            len(route_plan) == 14
            and len(positive_routes) == 6
            and len(boundary_routes) == 8,
        ),
    ]
    return pd.DataFrame(rows)


def _summary(
    *,
    reconciliation_dir: Path,
    contract_014_dir: Path,
    continuity_016_dir: Path,
    assignment_surface_dir: Path,
    output_dir: Path,
    rule_rows: pd.DataFrame,
    route_plan: pd.DataFrame,
    boundary_guards: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    positive_routes = route_plan[
        route_plan["contract_pair_role"].astype(str).eq(
            "positive_object_wall_transfer_candidate"
        )
    ]
    boundary_routes = route_plan[
        route_plan["contract_pair_role"].astype(str).eq("boundary_collapse_control")
    ]
    return {
        "schema": "nanoclustering_g4_8_first_pass_016_object_wall_transfer_contract_summary.v1",
        "status": RUN_STATUS,
        "contract_status": "designed_016_object_wall_transfer_contract_not_executed",
        "reconciliation_dir": str(reconciliation_dir),
        "contract_014_dir": str(contract_014_dir),
        "continuity_016_dir": str(continuity_016_dir),
        "assignment_surface_dir": str(assignment_surface_dir),
        "output_dir": str(output_dir),
        "source_vocabulary_pair_id": PAIR_014,
        "positive_pair_id": PAIR_016,
        "boundary_guard_pair_id": BOUNDARY_PAIR,
        "allowed_start_conditions_for_016": list(POSITIVE_ALLOWED_STARTS),
        "blocked_start_conditions_for_016": list(POSITIVE_BLOCKED_STARTS),
        "boundary_guard_start_conditions": list(BOUNDARY_GUARD_STARTS),
        "route_plan_row_count": int(len(route_plan)),
        "positive_route_plan_row_count": int(len(positive_routes)),
        "boundary_guard_row_count": int(len(boundary_guards)),
        "boundary_route_plan_row_count": int(len(boundary_routes)),
        "planned_route_family_counts": _count_dict(route_plan["planned_route_family"]),
        "runner_support_status_counts": _count_dict(route_plan["runner_support_status"]),
        "acceptance_rule_count": int(len(rule_rows)),
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"), "gate_id"
        ].tolist(),
        "interpretation": (
            "The contract transfers the 014 direct-only/recovery-loop vocabulary to "
            "016 only over 016 allowed starts, while retaining 005 as a boundary "
            "collapse guard. It is a design-only artifact and does not establish "
            "a wall, pathway, method, quality, or full-replay claim."
        ),
        "recommended_next_gate": (
            "Review this design-only contract; only after review, execute the 14 "
            "predeclared route rows if runner support exists. Do not broaden "
            "candidates or promote labels."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 50) -> str:
    cols = [column for column in columns if column in frame.columns]
    if not cols:
        return "_No matching columns._"
    visible = frame[cols].head(max_rows)
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


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    rule_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    route_plan: pd.DataFrame,
    boundary_guards: pd.DataFrame,
    decisions: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass 016 Object-Wall Transfer Contract",
        "",
        f"- status: `{summary['status']}`",
        f"- contract_status: `{summary['contract_status']}`",
        f"- source_vocabulary_pair_id: `{PAIR_014}`",
        f"- positive_pair_id: `{PAIR_016}`",
        f"- boundary_guard_pair_id: `{BOUNDARY_PAIR}`",
        f"- route_plan_row_count: {summary['route_plan_row_count']}",
        f"- positive_route_plan_row_count: {summary['positive_route_plan_row_count']}",
        f"- boundary_guard_row_count: {summary['boundary_guard_row_count']}",
        f"- allowed_start_conditions_for_016: {summary['allowed_start_conditions_for_016']}",
        f"- blocked_start_conditions_for_016: {summary['blocked_start_conditions_for_016']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Acceptance Rules",
        "",
        _markdown_table(
            rule_rows,
            [
                "rule_id",
                "rule_group",
                "rule_question",
                "acceptance_requirement",
                "claim_effect",
            ],
            max_rows=20,
        ),
        "",
        "## Pair Rows",
        "",
        _markdown_table(
            pair_rows,
            [
                "local_pair_id",
                "contract_pair_role",
                "branch",
                "pair_scope",
                "allowed_start_conditions",
                "blocked_start_conditions",
                "direct_edge_weight",
                "bridge_to_direct_weight_ratio",
                "gate_class",
                "reconciliation_surface_class",
            ],
            max_rows=10,
        ),
        "",
        "## Route Plan",
        "",
        _markdown_table(
            route_plan,
            [
                "route_contract_id",
                "local_pair_id",
                "start_condition",
                "contract_pair_role",
                "planned_route_family",
                "source_planned_route_family",
                "route_family_role",
                "planned_intervention_schedule",
                "expected_endpoint_pattern",
                "acceptance_rule_ids",
                "route_execution_status",
                "counts_as_positive_if_accepted",
            ],
            max_rows=30,
        ),
        "",
        "## Boundary Guards",
        "",
        _markdown_table(
            boundary_guards,
            [
                "boundary_guard_id",
                "start_condition",
                "planned_route_family",
                "expected_guard_outcome",
                "positive_leak_signal",
                "boundary_guard_status",
            ],
            max_rows=20,
        ),
        "",
        "## Decisions",
        "",
        _markdown_table(decisions, ["decision_id", "decision", "rationale"], max_rows=10),
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
            "This is a transfer contract. It must not be read as proof that 016 has "
            "an accepted pathway or wall. It only fixes the next narrow execution "
            "surface and the evidence that would be needed after execution."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    reconciliation_dir = Path(args.reconciliation_dir)
    contract_014_dir = Path(args.contract_014_dir)
    continuity_016_dir = Path(args.continuity_016_dir)
    assignment_surface_dir = Path(args.assignment_surface_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reconciliation_summary = _read_json(reconciliation_dir / RECONCILIATION_SUMMARY_JSON)
    reconciliation_gates = _read_csv(reconciliation_dir / RECONCILIATION_GATE_MATRIX_CSV)
    reconciliation_pair_rows = _read_csv(reconciliation_dir / RECONCILIATION_PAIR_ROWS_CSV)
    contract_014_route_plan = _read_csv(
        contract_014_dir / CONTRACT_014_ROUTE_PLAN_ROWS_CSV
    )
    contract_014_rule_rows = _read_csv(contract_014_dir / CONTRACT_014_RULE_ROWS_CSV)
    contract_014_summary = _read_json(contract_014_dir / CONTRACT_014_SUMMARY_JSON)
    continuity_pair_rows = _read_csv(
        continuity_016_dir / CONTINUITY_PAIR_COMPARISON_ROWS_CSV
    )
    continuity_summary = _read_json(continuity_016_dir / CONTINUITY_SUMMARY_JSON)
    assignment_pair_rows = _read_csv(assignment_surface_dir / ASSIGNMENT_PAIR_ROWS_CSV)
    assignment_summary = _read_json(assignment_surface_dir / ASSIGNMENT_SUMMARY_JSON)

    rule_rows = _rule_rows()
    pair_rows = _pair_rows(
        reconciliation_pair_rows=reconciliation_pair_rows,
        continuity_pair_rows=continuity_pair_rows,
        assignment_pair_rows=assignment_pair_rows,
    )
    route_plan = _route_plan_rows(
        continuity_pair_rows=continuity_pair_rows,
        contract_014_route_plan=contract_014_route_plan,
    )
    boundary_guards = _boundary_guard_rows(route_plan)
    decisions = _decision_rows()
    gates = _gate_matrix(
        reconciliation_summary=reconciliation_summary,
        reconciliation_gates=reconciliation_gates,
        rule_rows=rule_rows,
        route_plan=route_plan,
        boundary_guards=boundary_guards,
        continuity_pair_rows=continuity_pair_rows,
    )
    summary = _summary(
        reconciliation_dir=reconciliation_dir,
        contract_014_dir=contract_014_dir,
        continuity_016_dir=continuity_016_dir,
        assignment_surface_dir=assignment_surface_dir,
        output_dir=output_dir,
        rule_rows=rule_rows,
        route_plan=route_plan,
        boundary_guards=boundary_guards,
        gates=gates,
    )

    _write_csv(rule_rows, output_dir / RULE_ROWS_CSV)
    _write_csv(pair_rows, output_dir / PAIR_ROWS_CSV)
    _write_csv(route_plan, output_dir / ROUTE_PLAN_ROWS_CSV)
    _write_csv(boundary_guards, output_dir / BOUNDARY_GUARD_ROWS_CSV)
    _write_csv(decisions, output_dir / DECISION_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_016_object_wall_transfer_contract_config.v1",
        "reconciliation_dir": str(reconciliation_dir),
        "contract_014_dir": str(contract_014_dir),
        "continuity_016_dir": str(continuity_016_dir),
        "assignment_surface_dir": str(assignment_surface_dir),
        "output_dir": str(output_dir),
        "contract_014_status": contract_014_summary.get("status"),
        "continuity_016_status": continuity_summary.get("status"),
        "assignment_surface_status": assignment_summary.get("status"),
        "source_014_rule_ids": contract_014_rule_rows["rule_id"].astype(str).tolist(),
        "required_measurements": list(REQUIRED_MEASUREMENTS),
        "positive_allowed_starts": list(POSITIVE_ALLOWED_STARTS),
        "positive_blocked_starts": list(POSITIVE_BLOCKED_STARTS),
        "boundary_guard_starts": list(BOUNDARY_GUARD_STARTS),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        rule_rows=rule_rows,
        pair_rows=pair_rows,
        route_plan=route_plan,
        boundary_guards=boundary_guards,
        decisions=decisions,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconciliation-dir", type=Path, default=DEFAULT_RECONCILIATION_DIR)
    parser.add_argument("--contract-014-dir", type=Path, default=DEFAULT_014_CONTRACT_DIR)
    parser.add_argument("--continuity-016-dir", type=Path, default=DEFAULT_016_CONTINUITY_DIR)
    parser.add_argument(
        "--assignment-surface-dir", type=Path, default=DEFAULT_ASSIGNMENT_SURFACE_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
