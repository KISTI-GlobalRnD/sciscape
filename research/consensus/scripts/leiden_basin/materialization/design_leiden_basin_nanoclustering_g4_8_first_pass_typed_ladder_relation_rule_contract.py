#!/usr/bin/env python3
"""Design the first-pass typed-ladder relation-rule contract.

This design reads the transition-type panel, object-surface rule decision,
016 object-identity certificate, and 016 signature-identity resolution. It
predeclares the wording boundary for stronger 016 relation language:

- a recurrent typed ladder can be accepted as diagnostic relation vocabulary;
- target-like endpoint/collapse controls are not ladder evidence;
- endpoint-object wall/pathway wording remains blocked until a separate object
  membership gate passes.

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

from audit_leiden_basin_nanoclustering_g4_8_first_pass_016_object_identity_certificate import (
    DEFAULT_OUTPUT_DIR as DEFAULT_016_OBJECT_IDENTITY_DIR,
)
from audit_leiden_basin_nanoclustering_g4_8_first_pass_016_object_signature_identity_resolution import (
    DEFAULT_OUTPUT_DIR as DEFAULT_016_SIGNATURE_IDENTITY_DIR,
)
from audit_leiden_basin_nanoclustering_g4_8_first_pass_object_surface_rule_decision import (
    DEFAULT_OUTPUT_DIR as DEFAULT_OBJECT_SURFACE_RULE_DIR,
)
from design_leiden_basin_nanoclustering_g4_8_first_pass_transition_type_panel_contract import (
    DEFAULT_OUTPUT_DIR as DEFAULT_TRANSITION_TYPE_PANEL_DIR,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)
from surface_claim_schema_adapter import (
    surface_claim_count_dict as _count_dict,
    surface_claim_gate_row as _gate_row,
    surface_claim_json_dump as _json_dump,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_contract_gamma1e5_20260609"
)

TRANSITION_SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_transition_type_panel_contract_summary.json"
)
TRANSITION_CASE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_transition_type_panel_contract_case_rows.csv"
)
TRANSITION_GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_transition_type_panel_contract_gate_matrix.csv"
)
OBJECT_SURFACE_SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_object_surface_rule_decision_summary.json"
)
OBJECT_SURFACE_RULE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_object_surface_rule_decision_rule_rows.csv"
)
OBJECT_SURFACE_CASE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_object_surface_rule_decision_case_surface_rows.csv"
)
OBJECT_IDENTITY_SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_016_object_identity_certificate_summary.json"
)
OBJECT_IDENTITY_RELATION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_identity_certificate_relation_rows.csv"
)
OBJECT_IDENTITY_LOCAL_OBJECT_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_identity_certificate_local_object_rows.csv"
)
OBJECT_IDENTITY_GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_identity_certificate_gate_matrix.csv"
)
SIGNATURE_IDENTITY_SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_016_object_signature_identity_resolution_summary.json"
)
SIGNATURE_IDENTITY_SIGNATURE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_signature_identity_resolution_signature_rows.csv"
)
SIGNATURE_IDENTITY_GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_signature_identity_resolution_gate_matrix.csv"
)

RULE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_contract_rule_rows.csv"
)
CASE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_contract_case_rows.csv"
)
CONTROL_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_contract_control_rows.csv"
)
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_contract_decision_rows.csv"
)
NEXT_GATE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_contract_next_gate_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_contract_gate_matrix.csv"
)
SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_contract_summary.json"
)
CONFIG_JSON = (
    "nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_contract_config.json"
)
REPORT_MD = (
    "nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_contract_report.md"
)

RUN_STATUS = (
    "designed_nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_contract"
)
ROUTE_EXECUTION_STATUS = "design_only_no_new_route_execution"
RELATION_RULE_STATUS = "typed_ladder_relation_rule_predeclared_diagnostic_only"
WALL_PROMOTION_STATUS = "not_promoted_endpoint_object_membership_unresolved"
PATHWAY_PROMOTION_STATUS = "not_promoted_relation_vocabulary_only"
METHOD_STATUS = "typed_ladder_relation_contract_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass typed-ladder relation-rule contract only; "
    "reads existing transition-type, object-surface, object-identity, and "
    "signature-identity artifacts. It predeclares diagnostic relation wording "
    "for 016 typed-ladder evidence, while keeping endpoint-object wall, "
    "pathway, panel-generality, method, quality/cost, full-replay, route "
    "execution, and screened-gap expansion claims closed."
)

REFERENCE_PAIR_ID = "local_pair_016"
TARGET_COLLAPSE_CONTROL_IDS = ("local_pair_001", "local_pair_007")
STRICT_ANALOG_CONTROL_IDS = ("local_pair_009", "local_pair_012", "local_pair_020")
BOUNDARY_CONTROL_ID = "local_pair_005"
CROSS_SURFACE_GUARD_ID = "local_pair_014"
TRANSIENT_SIGNATURE_ID = "aeb59ab537e6"
TARGET_SIGNATURE_ID = "3c9b8a190753"


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
        "transition_summary": _read_json(
            args.transition_type_panel_dir / TRANSITION_SUMMARY_JSON
        ),
        "transition_case_rows": _read_csv(
            args.transition_type_panel_dir / TRANSITION_CASE_ROWS_CSV
        ),
        "transition_gates": _read_csv(
            args.transition_type_panel_dir / TRANSITION_GATE_MATRIX_CSV
        ),
        "object_surface_summary": _read_json(
            args.object_surface_rule_dir / OBJECT_SURFACE_SUMMARY_JSON
        ),
        "object_surface_rule_rows": _read_csv(
            args.object_surface_rule_dir / OBJECT_SURFACE_RULE_ROWS_CSV
        ),
        "object_surface_case_rows": _read_csv(
            args.object_surface_rule_dir / OBJECT_SURFACE_CASE_ROWS_CSV
        ),
        "object_identity_summary": _read_json(
            args.object_identity_dir / OBJECT_IDENTITY_SUMMARY_JSON
        ),
        "object_identity_relation_rows": _read_csv(
            args.object_identity_dir / OBJECT_IDENTITY_RELATION_ROWS_CSV
        ),
        "object_identity_local_object_rows": _read_csv(
            args.object_identity_dir / OBJECT_IDENTITY_LOCAL_OBJECT_ROWS_CSV
        ),
        "object_identity_gates": _read_csv(
            args.object_identity_dir / OBJECT_IDENTITY_GATE_MATRIX_CSV
        ),
        "signature_identity_summary": _read_json(
            args.signature_identity_dir / SIGNATURE_IDENTITY_SUMMARY_JSON
        ),
        "signature_identity_signature_rows": _read_csv(
            args.signature_identity_dir / SIGNATURE_IDENTITY_SIGNATURE_ROWS_CSV
        ),
        "signature_identity_gates": _read_csv(
            args.signature_identity_dir / SIGNATURE_IDENTITY_GATE_MATRIX_CSV
        ),
    }


def _rule_rows() -> pd.DataFrame:
    rows = pd.DataFrame(
        [
            {
                "rule_id": "TLR1",
                "rule_group": "scope_lock",
                "rule_status": "opened",
                "rule": (
                    "Typed-ladder relation is a wording contract over existing "
                    "artifacts, not a new route or screened-gap expansion."
                ),
                "positive_requirement": (
                    "read transition-type, object-surface, object-identity, and "
                    "signature-identity evidence with all upstream gates passing"
                ),
                "blocker_or_control": "no new route execution or threshold sweep",
            },
            {
                "rule_id": "TLR2",
                "rule_group": "positive_typed_ladder",
                "rule_status": "opened_diagnostic_only",
                "rule": (
                    "016-like typed ladder requires a recurrent finite typed "
                    "transient between source-family and target signature states."
                ),
                "positive_requirement": (
                    "finite recurrent transition-band reference plus stable "
                    "typed-transient and target signatures"
                ),
                "blocker_or_control": (
                    "without endpoint-object membership this supports relation "
                    "vocabulary only, not wall wording"
                ),
            },
            {
                "rule_id": "TLR3",
                "rule_group": "target_not_sufficient",
                "rule_status": "closed_for_false_positive",
                "rule": (
                    "A target-like endpoint or low-fraction target collapse is not "
                    "sufficient to call a typed ladder."
                ),
                "positive_requirement": "target state plus finite typed transient band",
                "blocker_or_control": "001/007 late target-collapse controls and 005 boundary guard",
            },
            {
                "rule_id": "TLR4",
                "rule_group": "finite_band_specificity",
                "rule_status": "closed_for_false_positive",
                "rule": (
                    "Strict analogs that are abrupt, fragmented, or point-only are "
                    "not accepted as typed-ladder relations."
                ),
                "positive_requirement": "adjacent recurrent typed transient fractions",
                "blocker_or_control": "009/012/020 strict analog guards",
            },
            {
                "rule_id": "TLR5",
                "rule_group": "surface_separation",
                "rule_status": "closed_for_wall_transfer",
                "rule": (
                    "Typed ladder and endpoint-object wall surfaces stay separate."
                ),
                "positive_requirement": (
                    "014 remains a cross-surface object-wall guard; 016 remains "
                    "signature-object relation evidence"
                ),
                "blocker_or_control": "do not transfer 014 wall wording onto 016",
            },
            {
                "rule_id": "TLR6",
                "rule_group": "wall_blocker",
                "rule_status": "closed_for_promotion",
                "rule": (
                    "Endpoint-object wall/pathway wording is blocked until "
                    "symmetric endpoint-object membership is resolved."
                ),
                "positive_requirement": "separate endpoint-object membership audit",
                "blocker_or_control": "016 object_identity_resolved=false",
            },
        ]
    )
    rows["relation_rule_status"] = RELATION_RULE_STATUS
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["pathway_promotion_status"] = PATHWAY_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _case_rows(context: dict[str, Any]) -> pd.DataFrame:
    transition_cases = context["transition_case_rows"]
    signature_rows = context["signature_identity_signature_rows"]
    object_summary = context["object_identity_summary"]
    transition_lookup = transition_cases.set_index("local_pair_id").to_dict("index")
    signature_lookup = signature_rows.set_index("signature_id").to_dict("index")
    transient = signature_lookup.get(TRANSIENT_SIGNATURE_ID, {})
    target = signature_lookup.get(TARGET_SIGNATURE_ID, {})
    object_ready = bool(object_summary.get("local_object_wall_evidence_audit_ready"))
    object_resolved = bool(object_summary.get("object_identity_resolved"))

    rows: list[dict[str, Any]] = [
        {
            "local_pair_id": REFERENCE_PAIR_ID,
            "case_role": "positive_reference",
            "typed_ladder_case_class": "typed_ladder_relation_reference",
            "transition_type_class": transition_lookup[REFERENCE_PAIR_ID][
                "transition_type_class"
            ],
            "relation_decision": "accept_typed_ladder_relation_diagnostic_only",
            "relation_claim_status": "diagnostic_only",
            "surface_level": "signature_object",
            "object_status": "local_signature_object_certificate_available",
            "relation_status": "ladder",
            "typed_transient_signature_id": TRANSIENT_SIGNATURE_ID,
            "target_signature_id": TARGET_SIGNATURE_ID,
            "transient_trace_rows": transient.get("trace_row_count"),
            "transient_seed_count": transient.get("seed_count"),
            "transient_bridge_fractions": transient.get("bridge_fractions"),
            "target_local_object_certified": object_summary.get(
                "target_local_object_certified"
            ),
            "object_identity_resolved": object_resolved,
            "local_object_wall_evidence_audit_ready": object_ready,
            "allowed_wording": (
                "016 has a diagnostic typed-ladder relation over local "
                "signature-object states."
            ),
            "blocked_wording": (
                "016 has a certified object wall, pathway, method improvement, "
                "panel-level generality, or full-replay result."
            ),
            "next_use": "relation_vocabulary_for_016_only",
        },
        {
            "local_pair_id": CROSS_SURFACE_GUARD_ID,
            "case_role": "cross_surface_guard",
            "typed_ladder_case_class": "object_wall_surface_separate_from_typed_ladder",
            "transition_type_class": transition_lookup[CROSS_SURFACE_GUARD_ID][
                "transition_type_class"
            ],
            "relation_decision": "keep_as_separate_object_wall_guard",
            "relation_claim_status": "diagnostic_only",
            "surface_level": "endpoint_object",
            "object_status": "certified",
            "relation_status": "clean",
            "typed_transient_signature_id": "",
            "target_signature_id": "",
            "transient_trace_rows": "",
            "transient_seed_count": "",
            "transient_bridge_fractions": "",
            "target_local_object_certified": "",
            "object_identity_resolved": "",
            "local_object_wall_evidence_audit_ready": "",
            "allowed_wording": "014 remains separate local object-wall evidence.",
            "blocked_wording": "014 does not transfer wall wording to 016.",
            "next_use": "surface_separation_guard",
        },
    ]

    for pair_id in TARGET_COLLAPSE_CONTROL_IDS:
        transition_case = transition_lookup[pair_id]
        rows.append(
            {
                "local_pair_id": pair_id,
                "case_role": "target_collapse_control",
                "typed_ladder_case_class": "target_collapse_not_typed_ladder",
                "transition_type_class": transition_case["transition_type_class"],
                "relation_decision": "reject_target_endpoint_only_relation",
                "relation_claim_status": "blocked",
                "surface_level": "signature_object",
                "object_status": "not_endpoint_object_evidence",
                "relation_status": "collapse",
                "typed_transient_signature_id": "",
                "target_signature_id": "",
                "transient_trace_rows": "",
                "transient_seed_count": "",
                "transient_bridge_fractions": "",
                "target_local_object_certified": "",
                "object_identity_resolved": "",
                "local_object_wall_evidence_audit_ready": "",
                "allowed_wording": "late target-collapse control",
                "blocked_wording": "typed-ladder relation or wall evidence",
                "next_use": "target_not_sufficient_control",
            }
        )
    for pair_id in STRICT_ANALOG_CONTROL_IDS:
        transition_case = transition_lookup[pair_id]
        rows.append(
            {
                "local_pair_id": pair_id,
                "case_role": "strict_analog_control",
                "typed_ladder_case_class": "strict_analog_not_typed_ladder",
                "transition_type_class": transition_case["transition_type_class"],
                "relation_decision": "reject_nonfinite_or_abrupt_relation",
                "relation_claim_status": "blocked",
                "surface_level": "signature_object",
                "object_status": "not_endpoint_object_evidence",
                "relation_status": "unresolved",
                "typed_transient_signature_id": "",
                "target_signature_id": "",
                "transient_trace_rows": "",
                "transient_seed_count": "",
                "transient_bridge_fractions": "",
                "target_local_object_certified": "",
                "object_identity_resolved": "",
                "local_object_wall_evidence_audit_ready": "",
                "allowed_wording": "strict analog guard",
                "blocked_wording": "finite recurrent typed-ladder relation",
                "next_use": "finite_band_specificity_control",
            }
        )
    transition_case = transition_lookup[BOUNDARY_CONTROL_ID]
    rows.append(
        {
            "local_pair_id": BOUNDARY_CONTROL_ID,
            "case_role": "boundary_control",
            "typed_ladder_case_class": "boundary_collapse_not_typed_ladder",
            "transition_type_class": transition_case["transition_type_class"],
            "relation_decision": "reject_boundary_collapse_relation",
            "relation_claim_status": "closed",
            "surface_level": "endpoint_object",
            "object_status": "collapse",
            "relation_status": "collapse",
            "typed_transient_signature_id": "",
            "target_signature_id": "",
            "transient_trace_rows": "",
            "transient_seed_count": "",
            "transient_bridge_fractions": "",
            "target_local_object_certified": "",
            "object_identity_resolved": "",
            "local_object_wall_evidence_audit_ready": "",
            "allowed_wording": "closed boundary/collapse guard",
            "blocked_wording": "typed-ladder relation or wall evidence",
            "next_use": "boundary_false_positive_control",
        }
    )
    frame = pd.DataFrame(rows)
    frame["relation_rule_status"] = RELATION_RULE_STATUS
    frame["route_execution_status"] = ROUTE_EXECUTION_STATUS
    frame["wall_promotion_status"] = WALL_PROMOTION_STATUS
    frame["pathway_promotion_status"] = PATHWAY_PROMOTION_STATUS
    frame["method_status"] = METHOD_STATUS
    frame["run_status"] = RUN_STATUS
    frame["claim_boundary"] = CLAIM_BOUNDARY
    return frame


def _control_rows(context: dict[str, Any]) -> pd.DataFrame:
    transition_cases = context["transition_case_rows"].set_index("local_pair_id")
    controls = [
        (
            "C1",
            list(TARGET_COLLAPSE_CONTROL_IDS),
            "target_endpoint_not_sufficient",
            "001/007 become target-like at low fractions but have no finite band.",
        ),
        (
            "C2",
            list(STRICT_ANALOG_CONTROL_IDS),
            "finite_band_specificity",
            "009/012/020 are strict analog guards, not finite typed ladders.",
        ),
        (
            "C3",
            [BOUNDARY_CONTROL_ID],
            "collapse_false_positive_guard",
            "005 remains a boundary/collapse guard.",
        ),
        (
            "C4",
            [CROSS_SURFACE_GUARD_ID],
            "surface_separation_guard",
            "014 remains separate object-wall evidence and cannot transfer to 016.",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for control_id, pair_ids, control_role, control_logic in controls:
        for pair_id in pair_ids:
            rows.append(
                {
                    "control_id": control_id,
                    "local_pair_id": pair_id,
                    "control_role": control_role,
                    "transition_type_class": transition_cases.loc[
                        pair_id, "transition_type_class"
                    ],
                    "control_logic": control_logic,
                    "control_decision": "blocks_false_positive_typed_ladder_widening",
                }
            )
    frame = pd.DataFrame(rows)
    frame["relation_rule_status"] = RELATION_RULE_STATUS
    frame["route_execution_status"] = ROUTE_EXECUTION_STATUS
    frame["wall_promotion_status"] = WALL_PROMOTION_STATUS
    frame["pathway_promotion_status"] = PATHWAY_PROMOTION_STATUS
    frame["method_status"] = METHOD_STATUS
    frame["run_status"] = RUN_STATUS
    frame["claim_boundary"] = CLAIM_BOUNDARY
    return frame


def _decision_rows() -> pd.DataFrame:
    rows = pd.DataFrame(
        [
            {
                "decision_id": "D1",
                "decision": "open_typed_ladder_relation_diagnostic_only",
                "rationale": (
                    "016 has the only finite recurrent transition band and a stable "
                    "typed transient signature, so relation vocabulary is useful."
                ),
            },
            {
                "decision_id": "D2",
                "decision": "keep_wall_and_pathway_claims_blocked",
                "rationale": (
                    "The 016 object-identity certificate still reports "
                    "object_identity_resolved=false and local object-wall evidence "
                    "not ready."
                ),
            },
            {
                "decision_id": "D3",
                "decision": "use_controls_before_widening_definition",
                "rationale": (
                    "001/007, 009/012/020, 005, and 014 prevent typed ladder from "
                    "collapsing into any target endpoint, strict analog, boundary, "
                    "or object-wall surface."
                ),
            },
            {
                "decision_id": "D4",
                "decision": "do_not_expand_screened_gaps",
                "rationale": (
                    "The current mechanism question is wording and relation type, "
                    "not discovering more weak rows among the 15 screened gaps."
                ),
            },
        ]
    )
    rows["relation_rule_status"] = RELATION_RULE_STATUS
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["pathway_promotion_status"] = PATHWAY_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _next_gate_rows() -> pd.DataFrame:
    rows = pd.DataFrame(
        [
            {
                "next_gate_id": "NG1",
                "next_gate": "typed_ladder_relation_rule_application",
                "priority": 1,
                "status": "recommended_now",
                "rationale": (
                    "Apply the predeclared rule to the eight-row scoreable surface "
                    "so 016 relation wording and controls are auditable in one row set."
                ),
                "execution_type": "read_only_audit",
            },
            {
                "next_gate_id": "NG2",
                "next_gate": "endpoint_object_membership_audit",
                "priority": 2,
                "status": "alternative_if_promoting_wall_or_object_wording",
                "rationale": (
                    "Wall/object wording needs endpoint-object membership, not "
                    "typed-ladder route morphology."
                ),
                "execution_type": "separate_membership_audit",
            },
            {
                "next_gate_id": "NG3",
                "next_gate": "screened_gap_expansion",
                "priority": 3,
                "status": "blocked_for_now",
                "rationale": (
                    "The typed-ladder rule must be applied and audited before "
                    "opening the 15 screened gaps."
                ),
                "execution_type": "do_not_execute",
            },
            {
                "next_gate_id": "NG4",
                "next_gate": "method_or_quality_comparison",
                "priority": 4,
                "status": "blocked_until_relation_and_wall_gates_settle",
                "rationale": (
                    "Quality/cost comparison is premature until relation wording and "
                    "wall eligibility are fixed."
                ),
                "execution_type": "do_not_execute",
            },
        ]
    )
    rows["relation_rule_status"] = RELATION_RULE_STATUS
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["pathway_promotion_status"] = PATHWAY_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _all_gates_pass(frame: pd.DataFrame) -> bool:
    if "gate_status" not in frame.columns:
        return True
    return bool(frame["gate_status"].astype(str).eq("pass").all())


def _gate_matrix(
    *,
    context: dict[str, Any],
    rule_rows: pd.DataFrame,
    case_rows: pd.DataFrame,
    control_rows: pd.DataFrame,
    next_gate_rows: pd.DataFrame,
) -> pd.DataFrame:
    transition_summary = context["transition_summary"]
    object_surface_summary = context["object_surface_summary"]
    object_identity_summary = context["object_identity_summary"]
    signature_identity_summary = context["signature_identity_summary"]
    transition_cases = context["transition_case_rows"].set_index("local_pair_id")
    signature_rows = context["signature_identity_signature_rows"].set_index(
        "signature_id"
    )

    transient_present = TRANSIENT_SIGNATURE_ID in signature_rows.index
    target_present = TARGET_SIGNATURE_ID in signature_rows.index
    transient_row = (
        signature_rows.loc[TRANSIENT_SIGNATURE_ID].to_dict()
        if transient_present
        else {}
    )
    required_controls = {
        *TARGET_COLLAPSE_CONTROL_IDS,
        *STRICT_ANALOG_CONTROL_IDS,
        BOUNDARY_CONTROL_ID,
        CROSS_SURFACE_GUARD_ID,
    }
    observed_controls = set(control_rows["local_pair_id"].astype(str))

    return pd.DataFrame(
        [
            _gate_row(
                "G1_upstream_artifacts_pass",
                "Did the upstream evidence pass without failed gates?",
                {
                    "transition_failed_gates": transition_summary.get("failed_gates"),
                    "object_surface_failed_gates": object_surface_summary.get(
                        "failed_gates"
                    ),
                    "object_identity_failed_gates": object_identity_summary.get(
                        "failed_gates"
                    ),
                    "signature_identity_failed_gates": signature_identity_summary.get(
                        "failed_gates"
                    ),
                    "transition_gates_pass": _all_gates_pass(context["transition_gates"]),
                    "object_identity_gates_pass": _all_gates_pass(
                        context["object_identity_gates"]
                    ),
                    "signature_identity_gates_pass": _all_gates_pass(
                        context["signature_identity_gates"]
                    ),
                },
                "all upstream failed_gates lists empty and gate matrices pass",
                transition_summary.get("failed_gates") == []
                and object_surface_summary.get("failed_gates") == []
                and object_identity_summary.get("failed_gates") == []
                and signature_identity_summary.get("failed_gates") == []
                and _all_gates_pass(context["transition_gates"])
                and _all_gates_pass(context["object_identity_gates"])
                and _all_gates_pass(context["signature_identity_gates"]),
            ),
            _gate_row(
                "G2_016_positive_typed_ladder_evidence_present",
                "Does 016 satisfy the positive diagnostic typed-ladder rule?",
                {
                    "transition_type_class": transition_cases.loc[
                        REFERENCE_PAIR_ID, "transition_type_class"
                    ],
                    "typed_transient_signature_present": transient_present,
                    "target_signature_present": target_present,
                    "transient_trace_rows": transient_row.get("trace_row_count"),
                    "transient_seed_count": transient_row.get("seed_count"),
                    "transient_bridge_fractions": transient_row.get("bridge_fractions"),
                },
                "016 finite transition-band plus recurrent typed transient and target signatures",
                str(
                    transition_cases.loc[REFERENCE_PAIR_ID, "transition_type_class"]
                )
                == "finite_recurrent_transition_band_reference"
                and transient_present
                and target_present
                and int(transient_row.get("trace_row_count", 0)) >= 48
                and int(transient_row.get("seed_count", 0)) >= 8,
            ),
            _gate_row(
                "G3_false_positive_controls_present",
                "Are target-collapse, strict-analog, boundary, and cross-surface controls present?",
                {
                    "required_controls": sorted(required_controls),
                    "observed_controls": sorted(observed_controls),
                    "control_roles": _count_dict(control_rows["control_role"]),
                },
                "all seven controls materialized",
                required_controls == observed_controls and int(len(control_rows)) == 7,
            ),
            _gate_row(
                "G4_object_wall_promotion_blocked",
                "Does object evidence keep wall/pathway wording closed?",
                {
                    "object_identity_resolved": object_identity_summary.get(
                        "object_identity_resolved"
                    ),
                    "local_object_wall_evidence_audit_ready": object_identity_summary.get(
                        "local_object_wall_evidence_audit_ready"
                    ),
                    "target_local_object_certified": object_identity_summary.get(
                        "target_local_object_certified"
                    ),
                    "object_surface_rule_status": object_surface_summary.get(
                        "object_surface_rule_status"
                    ),
                },
                "target local object may be certified but endpoint object identity unresolved",
                bool(object_identity_summary.get("target_local_object_certified"))
                and not bool(object_identity_summary.get("object_identity_resolved"))
                and not bool(
                    object_identity_summary.get("local_object_wall_evidence_audit_ready")
                ),
            ),
            _gate_row(
                "G5_rules_and_claim_boundaries_predeclared",
                "Are relation rules and claim boundaries explicit?",
                {
                    "rule_count": int(len(rule_rows)),
                    "case_count": int(len(case_rows)),
                    "next_gate_count": int(len(next_gate_rows)),
                    "relation_rule_status": RELATION_RULE_STATUS,
                    "wall_promotion_status": _count_dict(
                        case_rows["wall_promotion_status"]
                    ),
                    "pathway_promotion_status": _count_dict(
                        case_rows["pathway_promotion_status"]
                    ),
                },
                "six rules, eight cases, four next gates, no wall/pathway promotion",
                int(len(rule_rows)) >= 6
                and int(len(case_rows)) == 8
                and int(len(next_gate_rows)) >= 4
                and bool(case_rows["wall_promotion_status"].eq(WALL_PROMOTION_STATUS).all())
                and bool(
                    case_rows["pathway_promotion_status"]
                    .eq(PATHWAY_PROMOTION_STATUS)
                    .all()
                ),
            ),
            _gate_row(
                "G6_no_route_or_method_promotion",
                "Are route execution, quality, full replay, and method claims closed?",
                {
                    "route_execution_status": _count_dict(
                        case_rows["route_execution_status"]
                    ),
                    "method_status": _count_dict(case_rows["method_status"]),
                    "blocked_next_gates": next_gate_rows.loc[
                        next_gate_rows["status"].astype(str).str.contains("blocked"),
                        "next_gate",
                    ].astype(str).tolist(),
                },
                "design-only contract and screened-gap/method gates blocked",
                bool(
                    case_rows["route_execution_status"]
                    .eq(ROUTE_EXECUTION_STATUS)
                    .all()
                )
                and bool(case_rows["method_status"].eq(METHOD_STATUS).all())
                and "screened_gap_expansion"
                in set(
                    next_gate_rows.loc[
                        next_gate_rows["status"].astype(str).str.contains("blocked"),
                        "next_gate",
                    ].astype(str)
                )
                and "method_or_quality_comparison"
                in set(
                    next_gate_rows.loc[
                        next_gate_rows["status"].astype(str).str.contains("blocked"),
                        "next_gate",
                    ].astype(str)
                ),
            ),
        ]
    )


def _summary(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    rule_rows: pd.DataFrame,
    case_rows: pd.DataFrame,
    control_rows: pd.DataFrame,
    next_gate_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> dict[str, Any]:
    failed_gates = list(
        gate_matrix.loc[gate_matrix["gate_status"].ne("pass"), "gate_id"].astype(str)
    )
    return {
        "schema": "nanoclustering_g4_8_first_pass_typed_ladder_relation_rule_contract_summary.v1",
        "status": RUN_STATUS,
        "output_dir": str(output_dir),
        "transition_type_panel_dir": str(args.transition_type_panel_dir),
        "object_surface_rule_dir": str(args.object_surface_rule_dir),
        "object_identity_dir": str(args.object_identity_dir),
        "signature_identity_dir": str(args.signature_identity_dir),
        "relation_rule_status": RELATION_RULE_STATUS,
        "case_row_count": int(len(case_rows)),
        "control_row_count": int(len(control_rows)),
        "rule_row_count": int(len(rule_rows)),
        "typed_ladder_reference_pair_ids": case_rows.loc[
            case_rows["typed_ladder_case_class"].astype(str).eq(
                "typed_ladder_relation_reference"
            ),
            "local_pair_id",
        ].astype(str).tolist(),
        "negative_control_pair_ids": sorted(
            {
                *TARGET_COLLAPSE_CONTROL_IDS,
                *STRICT_ANALOG_CONTROL_IDS,
                BOUNDARY_CONTROL_ID,
            }
        ),
        "cross_surface_guard_pair_ids": [CROSS_SURFACE_GUARD_ID],
        "typed_ladder_case_class_counts": _count_dict(
            case_rows["typed_ladder_case_class"]
        ),
        "relation_decision_counts": _count_dict(case_rows["relation_decision"]),
        "relation_claim_status_counts": _count_dict(
            case_rows["relation_claim_status"]
        ),
        "control_role_counts": _count_dict(control_rows["control_role"]),
        "next_gate_status_counts": _count_dict(next_gate_rows["status"]),
        "recommended_next_gate": "typed_ladder_relation_rule_application",
        "alternative_next_gate": "endpoint_object_membership_audit",
        "blocked_next_gates": [
            "screened_gap_expansion",
            "method_or_quality_comparison",
        ],
        "gate_status_counts": _count_dict(gate_matrix["gate_status"]),
        "failed_gates": failed_gates,
        "route_execution_opened": False,
        "screened_gap_expansion_opened": False,
        "wall_claim_ready": False,
        "pathway_claim_ready": False,
        "panel_generality_claim_ready": False,
        "method_claim_ready": False,
        "quality_claim_ready": False,
        "full_replay_claim_ready": False,
        "interpretation": (
            "The typed-ladder rule can now be used as diagnostic relation "
            "vocabulary for 016 only: recurrent finite typed-transient evidence "
            "separates it from target-collapse, strict-analog, boundary, and "
            "cross-surface controls. Object-wall and pathway wording remain "
            "blocked because endpoint-object identity is still unresolved."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _report(
    *,
    summary: dict[str, Any],
    rule_rows: pd.DataFrame,
    case_rows: pd.DataFrame,
    control_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    next_gate_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# NanoClustering G4.8 First-Pass Typed-Ladder Relation-Rule Contract",
            "",
            f"- status: `{summary['status']}`",
            f"- relation_rule_status: `{summary['relation_rule_status']}`",
            f"- typed_ladder_reference_pair_ids: {summary['typed_ladder_reference_pair_ids']}",
            f"- negative_control_pair_ids: {summary['negative_control_pair_ids']}",
            f"- cross_surface_guard_pair_ids: {summary['cross_surface_guard_pair_ids']}",
            f"- relation_claim_status_counts: {summary['relation_claim_status_counts']}",
            f"- recommended_next_gate: `{summary['recommended_next_gate']}`",
            f"- alternative_next_gate: `{summary['alternative_next_gate']}`",
            f"- blocked_next_gates: {summary['blocked_next_gates']}",
            f"- failed_gates: {summary['failed_gates']}",
            f"- interpretation: {summary['interpretation']}",
            f"- claim_boundary: {CLAIM_BOUNDARY}",
            "",
            "## Relation Rules",
            "",
            _markdown_table(
                rule_rows,
                [
                    "rule_id",
                    "rule_group",
                    "rule_status",
                    "rule",
                    "positive_requirement",
                    "blocker_or_control",
                ],
            ),
            "",
            "## Case Rows",
            "",
            _markdown_table(
                case_rows,
                [
                    "local_pair_id",
                    "case_role",
                    "typed_ladder_case_class",
                    "relation_decision",
                    "relation_claim_status",
                    "allowed_wording",
                    "blocked_wording",
                    "next_use",
                ],
            ),
            "",
            "## Controls",
            "",
            _markdown_table(
                control_rows,
                [
                    "control_id",
                    "local_pair_id",
                    "control_role",
                    "transition_type_class",
                    "control_logic",
                    "control_decision",
                ],
            ),
            "",
            "## Decisions",
            "",
            _markdown_table(decision_rows, ["decision_id", "decision", "rationale"]),
            "",
            "## Next Gates",
            "",
            _markdown_table(
                next_gate_rows,
                ["next_gate_id", "next_gate", "priority", "status", "rationale"],
            ),
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
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    context = _load_context(args)
    rule_rows = _rule_rows()
    case_rows = _case_rows(context)
    control_rows = _control_rows(context)
    decision_rows = _decision_rows()
    next_gate_rows = _next_gate_rows()
    gate_matrix = _gate_matrix(
        context=context,
        rule_rows=rule_rows,
        case_rows=case_rows,
        control_rows=control_rows,
        next_gate_rows=next_gate_rows,
    )
    summary = _summary(
        output_dir=output_dir,
        args=args,
        rule_rows=rule_rows,
        case_rows=case_rows,
        control_rows=control_rows,
        next_gate_rows=next_gate_rows,
        gate_matrix=gate_matrix,
    )
    config = {
        "transition_type_panel_dir": str(args.transition_type_panel_dir),
        "object_surface_rule_dir": str(args.object_surface_rule_dir),
        "object_identity_dir": str(args.object_identity_dir),
        "signature_identity_dir": str(args.signature_identity_dir),
        "output_dir": str(output_dir),
        "route_execution_status": ROUTE_EXECUTION_STATUS,
        "relation_rule_status": RELATION_RULE_STATUS,
        "wall_promotion_status": WALL_PROMOTION_STATUS,
        "pathway_promotion_status": PATHWAY_PROMOTION_STATUS,
        "method_status": METHOD_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }

    _write_csv(rule_rows, output_dir / RULE_ROWS_CSV)
    _write_csv(case_rows, output_dir / CASE_ROWS_CSV)
    _write_csv(control_rows, output_dir / CONTROL_ROWS_CSV)
    _write_csv(decision_rows, output_dir / DECISION_ROWS_CSV)
    _write_csv(next_gate_rows, output_dir / NEXT_GATE_ROWS_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / REPORT_MD).write_text(
        _report(
            summary=summary,
            rule_rows=rule_rows,
            case_rows=case_rows,
            control_rows=control_rows,
            decision_rows=decision_rows,
            next_gate_rows=next_gate_rows,
            gate_matrix=gate_matrix,
        ),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transition-type-panel-dir",
        type=Path,
        default=DEFAULT_TRANSITION_TYPE_PANEL_DIR,
    )
    parser.add_argument(
        "--object-surface-rule-dir",
        type=Path,
        default=DEFAULT_OBJECT_SURFACE_RULE_DIR,
    )
    parser.add_argument(
        "--object-identity-dir",
        type=Path,
        default=DEFAULT_016_OBJECT_IDENTITY_DIR,
    )
    parser.add_argument(
        "--signature-identity-dir",
        type=Path,
        default=DEFAULT_016_SIGNATURE_IDENTITY_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = _write_outputs(args)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
