#!/usr/bin/env python3
"""Apply the first-pass typed-ladder relation-rule contract.

This read-only audit applies the predeclared typed-ladder relation rule to the
current eight-row scoreable surface. It updates only the wording status for
016: the row may now carry diagnostic typed-ladder relation wording. The same
application keeps all controls and claim boundaries fixed.

It does not rerun Leiden, execute routes, reopen screened gaps, promote wall or
pathway claims, evaluate quality/cost, replay full NanoClustering, or claim
method success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from audit_leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace import (
    DEFAULT_OUTPUT_DIR as DEFAULT_LOW_FRACTION_AUDIT_DIR,
    GATE_MATRIX_CSV as LOW_FRACTION_GATE_MATRIX_CSV,
    PAIR_SURFACE_ROWS_CSV as LOW_FRACTION_PAIR_SURFACE_ROWS_CSV,
    SUMMARY_JSON as LOW_FRACTION_SUMMARY_JSON,
)
from design_leiden_basin_nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_contract import (
    CASE_ROWS_CSV as CONTRACT_CASE_ROWS_CSV,
    CONTROL_ROWS_CSV as CONTRACT_CONTROL_ROWS_CSV,
    DEFAULT_OUTPUT_DIR as DEFAULT_TYPED_LADDER_CONTRACT_DIR,
    GATE_MATRIX_CSV as CONTRACT_GATE_MATRIX_CSV,
    RULE_ROWS_CSV as CONTRACT_RULE_ROWS_CSV,
    SUMMARY_JSON as CONTRACT_SUMMARY_JSON,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)
from surface_claim_schema_adapter import (
    SCHEMA_ADAPTER_VERSION,
    surface_claim_count_dict as _count_dict,
    surface_claim_gate_row as _gate_row,
    surface_claim_json_dump as _json_dump,
    validate_surface_claim_rows,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_application_gamma1e5_20260609"
)

PAIR_SURFACE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_application_pair_surface_rows.csv"
)
SCOREABLE_APPLICATION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_application_scoreable_rows.csv"
)
CONTROL_APPLICATION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_application_control_rows.csv"
)
EVIDENCE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_application_evidence_rows.csv"
)
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_application_decision_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_application_gate_matrix.csv"
)
SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_application_summary.json"
)
CONFIG_JSON = (
    "nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_application_config.json"
)
REPORT_MD = (
    "nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_application_report.md"
)

RUN_STATUS = (
    "audited_nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_application"
)
ROUTE_EXECUTION_STATUS = "read_only_typed_ladder_relation_rule_application"
RELATION_RULE_STATUS = "typed_ladder_relation_rule_applied_diagnostic_only"
WALL_PROMOTION_STATUS = "not_promoted_endpoint_object_membership_unresolved"
PATHWAY_PROMOTION_STATUS = "not_promoted_relation_vocabulary_only"
METHOD_STATUS = "typed_ladder_relation_rule_application_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass typed-ladder relation-rule application "
    "audit only; applies the predeclared rule to the eight-row scoreable "
    "surface. It may move 016 from blocked relation wording to diagnostic-only "
    "typed-ladder relation wording, but it keeps endpoint-object wall, pathway, "
    "panel-generality, method, quality/cost, full-replay, route execution, and "
    "screened-gap expansion claims closed."
)

REFERENCE_PAIR_ID = "local_pair_016"
TARGET_COLLAPSE_CONTROL_IDS = {"local_pair_001", "local_pair_007"}
STRICT_ANALOG_CONTROL_IDS = {"local_pair_009", "local_pair_012", "local_pair_020"}
BOUNDARY_CONTROL_ID = "local_pair_005"
CROSS_SURFACE_GUARD_ID = "local_pair_014"
EXPECTED_SCOREABLE_IDS = {
    REFERENCE_PAIR_ID,
    CROSS_SURFACE_GUARD_ID,
    *TARGET_COLLAPSE_CONTROL_IDS,
    *STRICT_ANALOG_CONTROL_IDS,
    BOUNDARY_CONTROL_ID,
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 80) -> str:
    cols = [column for column in columns if column in frame.columns]
    if not cols:
        return "_No matching columns._"
    visible = frame[cols].head(max_rows)
    if visible.empty:
        return "_No rows._"

    def cell(value: Any) -> str:
        if isinstance(value, (dict, list, tuple, set)):
            return _json_dump(value).replace("|", "\\|")
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


def _load_context(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "contract_summary": _read_json(args.typed_ladder_contract_dir / CONTRACT_SUMMARY_JSON),
        "contract_gates": _read_csv(args.typed_ladder_contract_dir / CONTRACT_GATE_MATRIX_CSV),
        "contract_case_rows": _read_csv(args.typed_ladder_contract_dir / CONTRACT_CASE_ROWS_CSV),
        "contract_control_rows": _read_csv(
            args.typed_ladder_contract_dir / CONTRACT_CONTROL_ROWS_CSV
        ),
        "contract_rule_rows": _read_csv(args.typed_ladder_contract_dir / CONTRACT_RULE_ROWS_CSV),
        "low_fraction_summary": _read_json(
            args.low_fraction_audit_dir / LOW_FRACTION_SUMMARY_JSON
        ),
        "low_fraction_gates": _read_csv(
            args.low_fraction_audit_dir / LOW_FRACTION_GATE_MATRIX_CSV
        ),
        "low_fraction_pair_rows": _read_csv(
            args.low_fraction_audit_dir / LOW_FRACTION_PAIR_SURFACE_ROWS_CSV
        ),
    }


def _all_gates_pass(frame: pd.DataFrame) -> bool:
    if "gate_status" not in frame.columns:
        return True
    return bool(frame["gate_status"].astype(str).eq("pass").all())


def _application_class(pair_id: str, contract_case: dict[str, Any] | None) -> dict[str, str]:
    if pair_id == REFERENCE_PAIR_ID:
        return {
            "surface_level": "relation",
            "object_status": "split",
            "relation_status": "ladder",
            "claim_status": "diagnostic_only",
            "application_role": "positive_typed_ladder_relation_reference",
            "application_decision": "accept_diagnostic_typed_ladder_relation",
            "application_claim_delta": "blocked_to_diagnostic_only_relation_wording",
            "allowed_wording_after_application": (
                "016 has a diagnostic typed-ladder relation over local "
                "signature-object states."
            ),
            "blocked_wording_after_application": (
                "016 has endpoint-object wall, pathway, method, panel-generality, "
                "quality/cost, or full-replay evidence."
            ),
        }
    if pair_id in TARGET_COLLAPSE_CONTROL_IDS:
        return {
            "surface_level": "state",
            "object_status": "unknown",
            "relation_status": "collapse",
            "claim_status": "blocked",
            "application_role": "target_collapse_control",
            "application_decision": "reject_target_endpoint_only_relation",
            "application_claim_delta": "blocked_remains_blocked",
            "allowed_wording_after_application": "late target-collapse control",
            "blocked_wording_after_application": "typed-ladder relation or wall evidence",
        }
    if pair_id in STRICT_ANALOG_CONTROL_IDS:
        return {
            "surface_level": "state",
            "object_status": "unknown",
            "relation_status": "unresolved",
            "claim_status": "blocked",
            "application_role": "strict_analog_control",
            "application_decision": "reject_nonfinite_or_abrupt_relation",
            "application_claim_delta": "blocked_remains_blocked",
            "allowed_wording_after_application": "strict analog guard",
            "blocked_wording_after_application": "finite recurrent typed-ladder relation",
        }
    if pair_id == BOUNDARY_CONTROL_ID:
        return {
            "surface_level": "endpoint_object",
            "object_status": "collapse",
            "relation_status": "collapse",
            "claim_status": "closed",
            "application_role": "boundary_control",
            "application_decision": "reject_boundary_collapse_relation",
            "application_claim_delta": "closed_remains_closed",
            "allowed_wording_after_application": "closed boundary/collapse guard",
            "blocked_wording_after_application": "typed-ladder relation or wall evidence",
        }
    if pair_id == CROSS_SURFACE_GUARD_ID:
        return {
            "surface_level": "endpoint_object",
            "object_status": "certified",
            "relation_status": "clean",
            "claim_status": "diagnostic_only",
            "application_role": "cross_surface_guard",
            "application_decision": "keep_separate_object_wall_surface",
            "application_claim_delta": "diagnostic_only_remains_separate_surface",
            "allowed_wording_after_application": (
                "014 remains separate local object-wall evidence."
            ),
            "blocked_wording_after_application": "014 wall wording does not transfer to 016.",
        }
    if contract_case is not None:
        raise ValueError(f"Unhandled contract case for {pair_id}: {contract_case}")
    return {
        "application_role": "screened_gap_or_non_scoreable_row",
        "application_decision": "not_in_typed_ladder_application_surface",
        "application_claim_delta": "unchanged_not_scoreable_or_closed",
        "allowed_wording_after_application": "not-scoreable screened row",
        "blocked_wording_after_application": "typed-ladder relation, wall, or pathway evidence",
    }


def _apply_rule(context: dict[str, Any]) -> pd.DataFrame:
    prior_rows = context["low_fraction_pair_rows"].copy()
    case_lookup = (
        context["contract_case_rows"].set_index("local_pair_id").to_dict("index")
    )
    prior_rows["pre_application_surface_level"] = prior_rows["surface_level"]
    prior_rows["pre_application_object_status"] = prior_rows["object_status"]
    prior_rows["pre_application_relation_status"] = prior_rows["relation_status"]
    prior_rows["pre_application_claim_status"] = prior_rows["claim_status"]
    prior_rows["pre_application_surface_rule_class"] = prior_rows["surface_rule_class"]
    for index, row in prior_rows.iterrows():
        pair_id = str(row["local_pair_id"])
        contract_case = case_lookup.get(pair_id)
        decision = _application_class(pair_id, contract_case)
        for column, value in decision.items():
            prior_rows.at[index, column] = value
        if pair_id in EXPECTED_SCOREABLE_IDS:
            prior_rows.at[index, "scoreability_status"] = "scoreable_core"
            prior_rows.at[index, "relation_rule_status"] = RELATION_RULE_STATUS
            prior_rows.at[index, "route_execution_status"] = ROUTE_EXECUTION_STATUS
            prior_rows.at[index, "wall_promotion_status"] = WALL_PROMOTION_STATUS
            prior_rows.at[index, "pathway_promotion_status"] = PATHWAY_PROMOTION_STATUS
            prior_rows.at[index, "method_status"] = METHOD_STATUS
            prior_rows.at[index, "schema_adapter_version"] = SCHEMA_ADAPTER_VERSION
    prior_rows["run_status"] = RUN_STATUS
    prior_rows["claim_boundary"] = CLAIM_BOUNDARY
    for column, value in [
        ("relation_rule_status", RELATION_RULE_STATUS),
        ("route_execution_status", ROUTE_EXECUTION_STATUS),
        ("wall_promotion_status", WALL_PROMOTION_STATUS),
        ("pathway_promotion_status", PATHWAY_PROMOTION_STATUS),
        ("method_status", METHOD_STATUS),
    ]:
        prior_rows[column] = prior_rows[column].fillna(value)
    return prior_rows.sort_values("local_pair_id", kind="mergesort").reset_index(drop=True)


def _control_application_rows(pair_rows: pd.DataFrame) -> pd.DataFrame:
    rows = pair_rows[
        pair_rows["local_pair_id"]
        .astype(str)
        .isin(EXPECTED_SCOREABLE_IDS - {REFERENCE_PAIR_ID})
    ].copy()
    return rows[
        [
            "local_pair_id",
            "application_role",
            "application_decision",
            "application_claim_delta",
            "surface_level",
            "object_status",
            "relation_status",
            "claim_status",
            "allowed_wording_after_application",
            "blocked_wording_after_application",
            "relation_rule_status",
            "wall_promotion_status",
            "pathway_promotion_status",
            "method_status",
            "run_status",
            "claim_boundary",
        ]
    ].reset_index(drop=True)


def _evidence_rows(context: dict[str, Any], pair_rows: pd.DataFrame) -> pd.DataFrame:
    contract_summary = context["contract_summary"]
    low_fraction_summary = context["low_fraction_summary"]
    scoreable_rows = pair_rows[
        pair_rows["scoreability_status"].astype(str).eq("scoreable_core")
    ]
    rows = [
        {
            "evidence_id": "E1_typed_ladder_contract",
            "evidence_type": "predeclared_contract",
            "evidence_summary": {
                "contract_failed_gates": contract_summary.get("failed_gates"),
                "relation_rule_status": contract_summary.get("relation_rule_status"),
                "typed_ladder_reference_pair_ids": contract_summary.get(
                    "typed_ladder_reference_pair_ids"
                ),
                "negative_control_pair_ids": contract_summary.get(
                    "negative_control_pair_ids"
                ),
                "cross_surface_guard_pair_ids": contract_summary.get(
                    "cross_surface_guard_pair_ids"
                ),
            },
        },
        {
            "evidence_id": "E2_eight_row_surface",
            "evidence_type": "upstream_surface",
            "evidence_summary": {
                "low_fraction_failed_gates": low_fraction_summary.get("failed_gates"),
                "scoreable_pair_count": low_fraction_summary.get("scoreable_pair_count"),
                "not_scoreable_pair_count": low_fraction_summary.get(
                    "not_scoreable_pair_count"
                ),
                "observed_scoreable_ids": scoreable_rows["local_pair_id"]
                .astype(str)
                .tolist(),
            },
        },
        {
            "evidence_id": "E3_application_result",
            "evidence_type": "application_rows",
            "evidence_summary": {
                "application_role_counts": _count_dict(pair_rows["application_role"]),
                "claim_status_counts": _count_dict(pair_rows["claim_status"]),
                "relation_status_counts": _count_dict(pair_rows["relation_status"]),
            },
        },
    ]
    frame = pd.DataFrame(rows)
    frame["run_status"] = RUN_STATUS
    frame["claim_boundary"] = CLAIM_BOUNDARY
    frame["evidence_summary"] = frame["evidence_summary"].map(_json_dump)
    return frame


def _decision_rows() -> pd.DataFrame:
    rows = pd.DataFrame(
        [
            {
                "decision_id": "D1",
                "decision": "apply_typed_ladder_relation_wording_to_016",
                "rationale": (
                    "The predeclared contract passed and isolates 016 as the only "
                    "finite recurrent typed-ladder reference."
                ),
            },
            {
                "decision_id": "D2",
                "decision": "retain_all_controls",
                "rationale": (
                    "001/007, 009/012/020, 005, and 014 remain false-positive or "
                    "surface-separation controls."
                ),
            },
            {
                "decision_id": "D3",
                "decision": "keep_wall_pathway_and_method_claims_closed",
                "rationale": (
                    "The application changes relation wording only; endpoint-object "
                    "identity remains unresolved."
                ),
            },
            {
                "decision_id": "D4",
                "decision": "do_not_expand_screened_gaps",
                "rationale": (
                    "The 15 screened gaps are not part of this relation-rule "
                    "application surface."
                ),
            },
        ]
    )
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _gate_matrix(
    *,
    context: dict[str, Any],
    pair_rows: pd.DataFrame,
    scoreable_rows: pd.DataFrame,
    schema_validation: dict[str, Any],
) -> pd.DataFrame:
    contract_summary = context["contract_summary"]
    low_fraction_summary = context["low_fraction_summary"]
    scoreable_ids = set(scoreable_rows["local_pair_id"].astype(str))
    row_by_pair = pair_rows.set_index("local_pair_id").to_dict("index")
    reference = row_by_pair[REFERENCE_PAIR_ID]
    controls = {
        pair_id: row_by_pair[pair_id]
        for pair_id in sorted(EXPECTED_SCOREABLE_IDS - {REFERENCE_PAIR_ID})
    }
    non_scoreable_rows = pair_rows[
        pair_rows["scoreability_status"].astype(str).ne("scoreable_core")
    ]
    no_open_claims = bool(pair_rows["claim_status"].astype(str).ne("open").all())
    promotion_closed = (
        bool(pair_rows["wall_promotion_status"].astype(str).eq(WALL_PROMOTION_STATUS).all())
        and bool(
            pair_rows["pathway_promotion_status"]
            .astype(str)
            .eq(PATHWAY_PROMOTION_STATUS)
            .all()
        )
        and bool(pair_rows["method_status"].astype(str).eq(METHOD_STATUS).all())
    )
    return pd.DataFrame(
        [
            _gate_row(
                "G1_upstream_contract_and_surface_pass",
                "Did the contract and eight-row surface pass upstream gates?",
                {
                    "contract_failed_gates": contract_summary.get("failed_gates"),
                    "low_fraction_failed_gates": low_fraction_summary.get("failed_gates"),
                    "contract_gates_pass": _all_gates_pass(context["contract_gates"]),
                    "low_fraction_gates_pass": _all_gates_pass(
                        context["low_fraction_gates"]
                    ),
                },
                "contract and low-fraction surface have no failed gates",
                contract_summary.get("failed_gates") == []
                and low_fraction_summary.get("failed_gates") == []
                and _all_gates_pass(context["contract_gates"])
                and _all_gates_pass(context["low_fraction_gates"]),
            ),
            _gate_row(
                "G2_eight_row_scoreable_surface_preserved",
                "Is the application limited to the expected eight scoreable rows?",
                {
                    "expected_scoreable_ids": sorted(EXPECTED_SCOREABLE_IDS),
                    "observed_scoreable_ids": sorted(scoreable_ids),
                    "scoreable_count": int(len(scoreable_rows)),
                    "non_scoreable_count": int(len(non_scoreable_rows)),
                },
                "eight scoreable rows and 15 not-scoreable rows",
                scoreable_ids == EXPECTED_SCOREABLE_IDS
                and int(len(scoreable_rows)) == 8
                and int(len(non_scoreable_rows)) == 15,
            ),
            _gate_row(
                "G3_016_relation_wording_opened_diagnostic_only",
                "Was 016 moved to diagnostic typed-ladder relation wording only?",
                {
                    "surface_level": reference.get("surface_level"),
                    "object_status": reference.get("object_status"),
                    "relation_status": reference.get("relation_status"),
                    "claim_status": reference.get("claim_status"),
                    "application_decision": reference.get("application_decision"),
                    "pre_application_claim_status": reference.get(
                        "pre_application_claim_status"
                    ),
                },
                "016 relation/ladder/diagnostic_only and no wall/pathway promotion",
                reference.get("surface_level") == "relation"
                and reference.get("object_status") == "split"
                and reference.get("relation_status") == "ladder"
                and reference.get("claim_status") == "diagnostic_only"
                and reference.get("application_decision")
                == "accept_diagnostic_typed_ladder_relation"
                and reference.get("wall_promotion_status") == WALL_PROMOTION_STATUS
                and reference.get("pathway_promotion_status") == PATHWAY_PROMOTION_STATUS,
            ),
            _gate_row(
                "G4_controls_remain_blocked_or_separate",
                "Do controls remain blocked, closed, or separate surfaces?",
                {
                    pair_id: {
                        "application_role": row.get("application_role"),
                        "relation_status": row.get("relation_status"),
                        "claim_status": row.get("claim_status"),
                    }
                    for pair_id, row in controls.items()
                },
                "001/007 and 009/012/020 blocked, 005 closed, 014 diagnostic separate",
                all(controls[pair_id]["claim_status"] == "blocked" for pair_id in TARGET_COLLAPSE_CONTROL_IDS)
                and all(controls[pair_id]["claim_status"] == "blocked" for pair_id in STRICT_ANALOG_CONTROL_IDS)
                and controls[BOUNDARY_CONTROL_ID]["claim_status"] == "closed"
                and controls[CROSS_SURFACE_GUARD_ID]["claim_status"] == "diagnostic_only"
                and controls[CROSS_SURFACE_GUARD_ID]["application_decision"]
                == "keep_separate_object_wall_surface",
            ),
            _gate_row(
                "G5_surface_schema_valid_after_application",
                "Are required surface-claim schema columns and values valid?",
                {
                    "required_columns_present": schema_validation.get(
                        "required_columns_present"
                    ),
                    "required_values_valid": schema_validation.get(
                        "required_values_valid"
                    ),
                    "invalid_values_by_column": schema_validation.get(
                        "invalid_values_by_column"
                    ),
                },
                "required columns present and values valid",
                bool(schema_validation.get("required_columns_present"))
                and bool(schema_validation.get("required_values_valid")),
            ),
            _gate_row(
                "G6_no_promotion_beyond_relation_vocabulary",
                "Are wall, pathway, method, quality, replay, and screened-gap claims closed?",
                {
                    "claim_status_counts": _count_dict(pair_rows["claim_status"]),
                    "wall_promotion_status": _count_dict(pair_rows["wall_promotion_status"]),
                    "pathway_promotion_status": _count_dict(
                        pair_rows["pathway_promotion_status"]
                    ),
                    "method_status": _count_dict(pair_rows["method_status"]),
                    "no_open_claims": no_open_claims,
                },
                "no open claims and no wall/pathway/method promotion",
                no_open_claims and promotion_closed,
            ),
        ]
    )


def _summary(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    pair_rows: pd.DataFrame,
    scoreable_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
    schema_validation: dict[str, Any],
) -> dict[str, Any]:
    failed_gates = list(
        gate_matrix.loc[gate_matrix["gate_status"].ne("pass"), "gate_id"].astype(str)
    )
    reference_row = (
        scoreable_rows[scoreable_rows["local_pair_id"].astype(str).eq(REFERENCE_PAIR_ID)]
        .iloc[0]
        .to_dict()
    )
    return {
        "schema": "nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_application_summary.v1",
        "status": RUN_STATUS,
        "output_dir": str(output_dir),
        "typed_ladder_contract_dir": str(args.typed_ladder_contract_dir),
        "low_fraction_audit_dir": str(args.low_fraction_audit_dir),
        "relation_rule_status": RELATION_RULE_STATUS,
        "pair_row_count": int(len(pair_rows)),
        "scoreable_pair_count": int(len(scoreable_rows)),
        "not_scoreable_pair_count": int(
            pair_rows["scoreability_status"].astype(str).ne("scoreable_core").sum()
        ),
        "scoreable_pair_ids": scoreable_rows["local_pair_id"].astype(str).tolist(),
        "typed_ladder_relation_ready_pair_ids": [REFERENCE_PAIR_ID],
        "diagnostic_claim_pair_ids": pair_rows.loc[
            pair_rows["claim_status"].astype(str).eq("diagnostic_only"),
            "local_pair_id",
        ].astype(str).tolist(),
        "diagnostic_typed_ladder_relation_pair_ids": [REFERENCE_PAIR_ID],
        "separate_object_wall_diagnostic_pair_ids": [CROSS_SURFACE_GUARD_ID],
        "blocked_control_pair_ids": scoreable_rows.loc[
            scoreable_rows["claim_status"].astype(str).eq("blocked"),
            "local_pair_id",
        ].astype(str).tolist(),
        "closed_control_pair_ids": scoreable_rows.loc[
            scoreable_rows["claim_status"].astype(str).eq("closed"),
            "local_pair_id",
        ].astype(str).tolist(),
        "application_role_counts": _count_dict(pair_rows["application_role"]),
        "claim_status_counts": _count_dict(pair_rows["claim_status"]),
        "scoreable_claim_status_counts": _count_dict(scoreable_rows["claim_status"]),
        "relation_status_counts": _count_dict(pair_rows["relation_status"]),
        "gate_status_counts": _count_dict(gate_matrix["gate_status"]),
        "failed_gates": failed_gates,
        "required_columns_present": schema_validation.get("required_columns_present"),
        "required_values_valid": schema_validation.get("required_values_valid"),
        "invalid_values_by_column": schema_validation.get("invalid_values_by_column"),
        "reference_pair_post_application": {
            "local_pair_id": reference_row.get("local_pair_id"),
            "pre_application_claim_status": reference_row.get(
                "pre_application_claim_status"
            ),
            "surface_level": reference_row.get("surface_level"),
            "object_status": reference_row.get("object_status"),
            "relation_status": reference_row.get("relation_status"),
            "claim_status": reference_row.get("claim_status"),
            "application_decision": reference_row.get("application_decision"),
        },
        "recommended_next_gate": "endpoint_object_membership_audit_if_wall_wording_is_required",
        "alternative_next_gate": "small_demo_leiden_cpm_reproduction_design_if_method_goal",
        "blocked_next_gates": [
            "screened_gap_expansion",
            "method_or_quality_comparison",
            "full_replay_claim",
        ],
        "route_execution_opened": False,
        "screened_gap_expansion_opened": False,
        "wall_claim_ready": False,
        "pathway_claim_ready": False,
        "panel_generality_claim_ready": False,
        "method_claim_ready": False,
        "quality_claim_ready": False,
        "full_replay_claim_ready": False,
        "interpretation": (
            "The typed-ladder relation rule now works as a controlled wording "
            "application: 016 moves from blocked relation wording to diagnostic-only "
            "typed-ladder relation wording, while all controls remain blocked, "
            "closed, or separate. This is still not wall, pathway, method, quality, "
            "or generality evidence."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _report(
    *,
    summary: dict[str, Any],
    scoreable_rows: pd.DataFrame,
    control_rows: pd.DataFrame,
    evidence_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# NanoClustering G4.8 First-Pass Typed-Ladder Relation-Rule Application",
            "",
            f"- status: `{summary['status']}`",
            f"- relation_rule_status: `{summary['relation_rule_status']}`",
            f"- scoreable_pair_count: {summary['scoreable_pair_count']}",
            f"- typed_ladder_relation_ready_pair_ids: {summary['typed_ladder_relation_ready_pair_ids']}",
            f"- diagnostic_claim_pair_ids: {summary['diagnostic_claim_pair_ids']}",
            f"- diagnostic_typed_ladder_relation_pair_ids: {summary['diagnostic_typed_ladder_relation_pair_ids']}",
            f"- separate_object_wall_diagnostic_pair_ids: {summary['separate_object_wall_diagnostic_pair_ids']}",
            f"- blocked_control_pair_ids: {summary['blocked_control_pair_ids']}",
            f"- closed_control_pair_ids: {summary['closed_control_pair_ids']}",
            f"- reference_pair_post_application: {summary['reference_pair_post_application']}",
            f"- recommended_next_gate: `{summary['recommended_next_gate']}`",
            f"- alternative_next_gate: `{summary['alternative_next_gate']}`",
            f"- blocked_next_gates: {summary['blocked_next_gates']}",
            f"- failed_gates: {summary['failed_gates']}",
            f"- interpretation: {summary['interpretation']}",
            f"- claim_boundary: {CLAIM_BOUNDARY}",
            "",
            "## Scoreable Application Rows",
            "",
            _markdown_table(
                scoreable_rows,
                [
                    "local_pair_id",
                    "application_role",
                    "application_decision",
                    "pre_application_claim_status",
                    "surface_level",
                    "object_status",
                    "relation_status",
                    "claim_status",
                    "application_claim_delta",
                ],
            ),
            "",
            "## Control Rows",
            "",
            _markdown_table(
                control_rows,
                [
                    "local_pair_id",
                    "application_role",
                    "application_decision",
                    "claim_status",
                    "allowed_wording_after_application",
                    "blocked_wording_after_application",
                ],
            ),
            "",
            "## Evidence",
            "",
            _markdown_table(evidence_rows, ["evidence_id", "evidence_type", "evidence_summary"]),
            "",
            "## Decisions",
            "",
            _markdown_table(decision_rows, ["decision_id", "decision", "rationale"]),
            "",
            "## Gate Matrix",
            "",
            _markdown_table(
                gate_matrix,
                ["gate_id", "gate_status", "observed", "minimum_or_rule", "question"],
            ),
            "",
        ]
    )


def _write_outputs(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    context = _load_context(args)
    pair_rows = _apply_rule(context)
    scoreable_rows = pair_rows[
        pair_rows["scoreability_status"].astype(str).eq("scoreable_core")
    ].copy()
    control_rows = _control_application_rows(pair_rows)
    evidence_rows = _evidence_rows(context, pair_rows)
    decision_rows = _decision_rows()
    schema_validation = validate_surface_claim_rows(pair_rows)
    gate_matrix = _gate_matrix(
        context=context,
        pair_rows=pair_rows,
        scoreable_rows=scoreable_rows,
        schema_validation=schema_validation,
    )
    summary = _summary(
        output_dir=args.output_dir,
        args=args,
        pair_rows=pair_rows,
        scoreable_rows=scoreable_rows,
        gate_matrix=gate_matrix,
        schema_validation=schema_validation,
    )
    config = {
        "typed_ladder_contract_dir": str(args.typed_ladder_contract_dir),
        "low_fraction_audit_dir": str(args.low_fraction_audit_dir),
        "output_dir": str(args.output_dir),
        "schema_adapter_version": SCHEMA_ADAPTER_VERSION,
        "relation_rule_status": RELATION_RULE_STATUS,
        "route_execution_status": ROUTE_EXECUTION_STATUS,
        "wall_promotion_status": WALL_PROMOTION_STATUS,
        "pathway_promotion_status": PATHWAY_PROMOTION_STATUS,
        "method_status": METHOD_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }

    _write_csv(pair_rows, args.output_dir / PAIR_SURFACE_ROWS_CSV)
    _write_csv(scoreable_rows, args.output_dir / SCOREABLE_APPLICATION_ROWS_CSV)
    _write_csv(control_rows, args.output_dir / CONTROL_APPLICATION_ROWS_CSV)
    _write_csv(evidence_rows, args.output_dir / EVIDENCE_ROWS_CSV)
    _write_csv(decision_rows, args.output_dir / DECISION_ROWS_CSV)
    _write_csv(gate_matrix, args.output_dir / GATE_MATRIX_CSV)
    (args.output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / REPORT_MD).write_text(
        _report(
            summary=summary,
            scoreable_rows=scoreable_rows,
            control_rows=control_rows,
            evidence_rows=evidence_rows,
            decision_rows=decision_rows,
            gate_matrix=gate_matrix,
        ),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--typed-ladder-contract-dir",
        type=Path,
        default=DEFAULT_TYPED_LADDER_CONTRACT_DIR,
    )
    parser.add_argument(
        "--low-fraction-audit-dir",
        type=Path,
        default=DEFAULT_LOW_FRACTION_AUDIT_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = _write_outputs(args)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
