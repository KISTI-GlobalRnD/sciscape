#!/usr/bin/env python3
"""Audit the object-surface rule decision under the surface-claim schema.

This read-only audit consumes the current surface-claim schema application,
the 016 object-identity certificate, the accepted 014/005 primitive wall
evidence audit, and the G4.9/G4.9A synthetic control summaries. It fixes the
next object-surface rule:

- local signature-objects may be used as diagnostic basin-state surfaces;
- endpoint-object membership remains required for object-wall wording;
- typed ladder evidence remains blocked until a separate rule and controls are
  predeclared.

It does not rerun Leiden, execute routes, promote pathway labels or walls,
evaluate quality/cost value, replay full NanoClustering, or claim method
success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from audit_leiden_basin_nanoclustering_g4_8_first_pass_014_wall_evidence import (
    DEFAULT_OUTPUT_DIR as DEFAULT_014_WALL_EVIDENCE_DIR,
    GATE_MATRIX_CSV as WALL_EVIDENCE_GATE_MATRIX_CSV,
    SUMMARY_JSON as WALL_EVIDENCE_SUMMARY_JSON,
)
from audit_leiden_basin_nanoclustering_g4_8_first_pass_016_object_identity_certificate import (
    DEFAULT_OUTPUT_DIR as DEFAULT_016_OBJECT_IDENTITY_CERTIFICATE_DIR,
    GATE_MATRIX_CSV as OBJECT_IDENTITY_GATE_MATRIX_CSV,
    SUMMARY_JSON as OBJECT_IDENTITY_SUMMARY_JSON,
)
from audit_leiden_basin_nanoclustering_g4_8_first_pass_surface_claim_schema_application import (
    CASE_ROWS_CSV as SCHEMA_APPLICATION_CASE_ROWS_CSV,
    DEFAULT_OUTPUT_DIR as DEFAULT_SCHEMA_APPLICATION_DIR,
    GATE_MATRIX_CSV as SCHEMA_APPLICATION_GATE_MATRIX_CSV,
    SUMMARY_JSON as SCHEMA_APPLICATION_SUMMARY_JSON,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)
from surface_claim_schema_adapter import (
    REQUIRED_COLUMNS,
    SCHEMA_ADAPTER_VERSION,
    surface_claim_count_dict as _count_dict,
    surface_claim_gate_row as _gate_row,
    surface_claim_json_dump as _json_dump,
    surface_claim_mapping_by_case,
    validate_surface_claim_rows,
)


G4_9_DIR = (
    BASE_RESULT_DIR / "leiden_basin_variable_pair_synthetic_g4_9_primitive_wall_demo_v1_20260604"
)
G4_9A_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_variable_pair_synthetic_g4_9a_parameter_localization_v1_20260604"
)
G4_9_SUMMARY_JSON = "variable_pair_synthetic_g4_9_summary.json"
G4_9_GATE_MATRIX_CSV = "variable_pair_synthetic_g4_9_gate_matrix.csv"
G4_9A_SUMMARY_JSON = "variable_pair_synthetic_g4_9a_summary.json"
G4_9A_GATE_MATRIX_CSV = "variable_pair_synthetic_g4_9a_gate_matrix.csv"

DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_object_surface_rule_decision_gamma1e5_20260608"
)

CASE_SURFACE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_object_surface_rule_decision_case_surface_rows.csv"
)
RULE_ROWS_CSV = "nanoclustering_g4_8_first_pass_object_surface_rule_decision_rule_rows.csv"
EVIDENCE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_object_surface_rule_decision_evidence_rows.csv"
)
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_object_surface_rule_decision_decision_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_object_surface_rule_decision_gate_matrix.csv"
)
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_object_surface_rule_decision_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_object_surface_rule_decision_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_object_surface_rule_decision_report.md"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_object_surface_rule_decision"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass object-surface rule-decision audit only; "
    "reads the surface-claim schema application, 016 object-identity "
    "certificate, 014/005 primitive wall-evidence audit, and G4.9/G4.9A "
    "synthetic control summaries. It may accept local signature-objects as a "
    "diagnostic basin-state surface, but it does not promote pathway labels, "
    "object-wall claims, method claims, quality/cost claims, full replay, or "
    "new route execution."
)
SCHEMA_ADAPTER_PATH = Path(__file__).resolve().parent / "surface_claim_schema_adapter.py"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


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
        "schema_application_summary": _read_json(
            args.schema_application_dir / SCHEMA_APPLICATION_SUMMARY_JSON
        ),
        "schema_application_gates": _read_csv(
            args.schema_application_dir / SCHEMA_APPLICATION_GATE_MATRIX_CSV
        ),
        "schema_application_case_rows": _read_csv(
            args.schema_application_dir / SCHEMA_APPLICATION_CASE_ROWS_CSV
        ),
        "object_identity_summary": _read_json(
            args.object_identity_certificate_dir / OBJECT_IDENTITY_SUMMARY_JSON
        ),
        "object_identity_gates": _read_csv(
            args.object_identity_certificate_dir / OBJECT_IDENTITY_GATE_MATRIX_CSV
        ),
        "wall_evidence_summary": _read_json(
            args.wall_evidence_dir / WALL_EVIDENCE_SUMMARY_JSON
        ),
        "wall_evidence_gates": _read_csv(
            args.wall_evidence_dir / WALL_EVIDENCE_GATE_MATRIX_CSV
        ),
        "g4_9_summary": _read_json(args.g4_9_dir / G4_9_SUMMARY_JSON),
        "g4_9_gates": _read_csv(args.g4_9_dir / G4_9_GATE_MATRIX_CSV),
        "g4_9a_summary": _read_json(args.g4_9a_dir / G4_9A_SUMMARY_JSON),
        "g4_9a_gates": _read_csv(args.g4_9a_dir / G4_9A_GATE_MATRIX_CSV),
    }


def _case_surface_rows(context: dict[str, Any]) -> pd.DataFrame:
    frame = context["schema_application_case_rows"].copy()
    decisions = {
        "local_pair_014": {
            "object_surface_rule": "endpoint_object_membership_evidence_available",
            "object_surface_decision": "retain_object_wall_diagnostic_only",
            "rule_effect": "accepted local object-level primitive wall evidence remains diagnostic only",
        },
        "local_pair_016": {
            "object_surface_rule": "signature_object_surface_accepts_diagnostic_only",
            "object_surface_decision": "accept_signature_object_surface_wall_blocked",
            "rule_effect": "local signature-object transition band may be described diagnostically only",
        },
        "local_pair_005": {
            "object_surface_rule": "boundary_collapse_control_retained",
            "object_surface_decision": "closed_false_positive_guard",
            "rule_effect": "collapse control stays closed and guards wall broadening",
        },
    }
    for column in ["object_surface_rule", "object_surface_decision", "rule_effect"]:
        frame[column] = frame["case_id"].astype(str).map(
            {case_id: row[column] for case_id, row in decisions.items()}
        )
    frame["schema_adapter_version"] = SCHEMA_ADAPTER_VERSION
    frame["run_status"] = RUN_STATUS
    frame["claim_boundary"] = CLAIM_BOUNDARY
    return frame


def _rule_rows(
    *,
    case_rows: pd.DataFrame,
    context: dict[str, Any],
) -> pd.DataFrame:
    mapping = surface_claim_mapping_by_case(case_rows)
    object_summary = context["object_identity_summary"]
    wall_summary = context["wall_evidence_summary"]
    g4_9_summary = context["g4_9_summary"]
    g4_9a_summary = context["g4_9a_summary"]
    rows = [
        {
            "rule_id": "R1_signature_object_surface_accepts_diagnostic_only",
            "rule_status": "accepted_diagnostic_surface_only",
            "evidence_case_ids": "local_pair_016",
            "observed": {
                "case_mapping": mapping.get("local_pair_016"),
                "local_signature_object_certificate_available": object_summary.get(
                    "local_signature_object_certificate_available"
                ),
                "target_local_object_certified": object_summary.get(
                    "target_local_object_certified"
                ),
                "object_identity_resolved": object_summary.get("object_identity_resolved"),
            },
            "allowed_wording": (
                "016 is a signature-object transition-band diagnostic surface with "
                "a certified target local object"
            ),
            "blocked_wording": "endpoint basin; object wall; pathway label; method claim",
            "claim_effect": "opens diagnostic surface wording only",
        },
        {
            "rule_id": "R2_endpoint_object_membership_required_for_wall",
            "rule_status": "retained_for_wall_claims",
            "evidence_case_ids": "local_pair_014;local_pair_016;local_pair_005",
            "observed": {
                "014_mapping": mapping.get("local_pair_014"),
                "016_mapping": mapping.get("local_pair_016"),
                "005_mapping": mapping.get("local_pair_005"),
                "016_local_object_wall_evidence_audit_ready": object_summary.get(
                    "local_object_wall_evidence_audit_ready"
                ),
                "016_wall_claim_ready_pairs": object_summary.get("wall_claim_ready_pairs"),
            },
            "allowed_wording": (
                "object-wall wording requires endpoint-object membership or an "
                "explicitly predeclared substitute relation rule"
            ),
            "blocked_wording": "016 object-wall wording from local signature objects alone",
            "claim_effect": "keeps 016 wall and pathway claims blocked",
        },
        {
            "rule_id": "R3_typed_ladder_wall_rule_predeclared",
            "rule_status": "not_opened_requires_separate_contract",
            "evidence_case_ids": "local_pair_016",
            "observed": {
                "016_relation_status": mapping.get("local_pair_016", {}).get(
                    "relation_status"
                ),
                "016_claim_status": mapping.get("local_pair_016", {}).get("claim_status"),
                "source_family_object_unified": object_summary.get(
                    "source_family_object_unified"
                ),
                "transient_endpoint_object_certified": object_summary.get(
                    "transient_endpoint_object_certified"
                ),
            },
            "allowed_wording": "typed ladder relation evidence is a blocked diagnostic surface",
            "blocked_wording": "typed ladder wall until a separate rule and guards pass",
            "claim_effect": "prevents ladder evidence from becoming wall evidence",
        },
        {
            "rule_id": "R4_g4_9_boundary_control_vocabulary_retained",
            "rule_status": "retained_as_false_positive_guard",
            "evidence_case_ids": "local_pair_014;local_pair_005;synthetic_g4_9;synthetic_g4_9a",
            "observed": {
                "primitive_wall_evidence_ready_pairs": wall_summary.get(
                    "primitive_wall_evidence_ready_pairs"
                ),
                "boundary_guard_closed_seed_count": wall_summary.get(
                    "boundary_guard_closed_seed_count"
                ),
                "g4_9_control_wall_leak_case_count": g4_9_summary.get(
                    "control_wall_leak_case_count"
                ),
                "g4_9a_nonready_case_count": g4_9a_summary.get("nonready_case_count"),
                "g4_9a_partial_ready_case_count": g4_9a_summary.get(
                    "partial_ready_case_count"
                ),
            },
            "allowed_wording": (
                "G4.9/G4.9A W/w/T/N/P-style vocabulary may classify boundary "
                "and partial regimes"
            ),
            "blocked_wording": "collapsing boundary controls into positive wall evidence",
            "claim_effect": "keeps false-positive guards in the object-surface rule",
        },
        {
            "rule_id": "R5_route_method_quality_claims_closed",
            "rule_status": "closed",
            "evidence_case_ids": "all",
            "observed": {
                "route_execution_opened": False,
                "method_claim_ready": False,
                "quality_claim_ready": False,
                "new_wall_claim_ready_pairs": [],
            },
            "allowed_wording": "read-only object-surface rule decision",
            "blocked_wording": "route execution; method claim; quality/cost claim; full replay",
            "claim_effect": "preserves the current claim boundary",
        },
    ]
    frame = pd.DataFrame(rows)
    frame["observed"] = frame["observed"].map(_json_dump)
    frame["schema_adapter_version"] = SCHEMA_ADAPTER_VERSION
    frame["run_status"] = RUN_STATUS
    frame["claim_boundary"] = CLAIM_BOUNDARY
    return frame


def _evidence_rows(context: dict[str, Any], validation: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "evidence_id": "E1_surface_schema_application_ready",
            "evidence_status": "ready",
            "observed": {
                "failed_gates": context["schema_application_summary"].get("failed_gates"),
                "adapter_validation": {
                    "required_columns_present": validation["required_columns_present"],
                    "required_values_valid": validation["required_values_valid"],
                    "missing_required_columns": validation["missing_required_columns"],
                    "invalid_values_by_column": validation["invalid_values_by_column"],
                },
                "gate_status_counts": _count_dict(
                    context["schema_application_gates"]["gate_status"]
                ),
            },
            "claim_effect": "permits rule decision on validated surface rows",
        },
        {
            "evidence_id": "E2_016_local_signature_objects_ready",
            "evidence_status": "diagnostic_surface_available",
            "observed": {
                "failed_gates": context["object_identity_summary"].get("failed_gates"),
                "local_signature_object_certificate_available": context[
                    "object_identity_summary"
                ].get("local_signature_object_certificate_available"),
                "target_local_object_certified": context["object_identity_summary"].get(
                    "target_local_object_certified"
                ),
                "object_identity_resolved": context["object_identity_summary"].get(
                    "object_identity_resolved"
                ),
                "local_object_wall_evidence_audit_ready": context[
                    "object_identity_summary"
                ].get("local_object_wall_evidence_audit_ready"),
            },
            "claim_effect": "allows 016 diagnostic surface and blocks wall wording",
        },
        {
            "evidence_id": "E3_014_005_wall_boundary_controls_ready",
            "evidence_status": "controls_ready",
            "observed": {
                "failed_gates": context["wall_evidence_summary"].get("failed_gates"),
                "primitive_wall_evidence_ready_pairs": context[
                    "wall_evidence_summary"
                ].get("primitive_wall_evidence_ready_pairs"),
                "boundary_guard_closed_seed_count": context[
                    "wall_evidence_summary"
                ].get("boundary_guard_closed_seed_count"),
                "gate_status_counts": _count_dict(
                    context["wall_evidence_gates"]["gate_status"]
                ),
            },
            "claim_effect": "retains 014 as diagnostic object-wall surface and 005 as guard",
        },
        {
            "evidence_id": "E4_g4_9_control_vocabulary_ready",
            "evidence_status": "control_vocabulary_ready",
            "observed": {
                "g4_9_failed_gates": context["g4_9_summary"].get("failed_gates"),
                "g4_9a_failed_gates": context["g4_9a_summary"].get("failed_gates"),
                "g4_9_control_wall_leak_case_count": context["g4_9_summary"].get(
                    "control_wall_leak_case_count"
                ),
                "g4_9a_full_ready_case_count": context["g4_9a_summary"].get(
                    "full_ready_case_count"
                ),
                "g4_9a_partial_ready_case_count": context["g4_9a_summary"].get(
                    "partial_ready_case_count"
                ),
                "g4_9a_nonready_case_count": context["g4_9a_summary"].get(
                    "nonready_case_count"
                ),
            },
            "claim_effect": "keeps real-data rule decision tied to boundary controls",
        },
    ]
    frame = pd.DataFrame(rows)
    frame["observed"] = frame["observed"].map(_json_dump)
    frame["run_status"] = RUN_STATUS
    frame["claim_boundary"] = CLAIM_BOUNDARY
    return frame


def _decision_rows() -> pd.DataFrame:
    rows = [
        {
            "decision_id": "D1",
            "decision": "accept_signature_object_surface_diagnostic_only",
            "rationale": (
                "016 has certified local signature-object evidence and a certified "
                "target local object, but object identity and wall evidence remain "
                "unresolved."
            ),
        },
        {
            "decision_id": "D2",
            "decision": "retain_endpoint_object_requirement_for_wall",
            "rationale": (
                "Endpoint-object membership or an explicit substitute relation rule "
                "is still required before any object-wall wording."
            ),
        },
        {
            "decision_id": "D3",
            "decision": "keep_typed_ladder_wall_rule_closed",
            "rationale": (
                "016 ladder evidence is informative but remains blocked until a "
                "separate typed-ladder rule and negative controls are predeclared."
            ),
        },
        {
            "decision_id": "D4",
            "decision": "retain_boundary_control_vocabulary",
            "rationale": (
                "014/005 and G4.9/G4.9A controls keep partial opening, target absence, "
                "source lock, and target saturation separate."
            ),
        },
        {
            "decision_id": "D5",
            "decision": "no_route_or_claim_promotion",
            "rationale": (
                "This is a definition-surface decision over existing artifacts, not "
                "route execution, method validation, quality/cost evaluation, or full replay."
            ),
        },
    ]
    frame = pd.DataFrame(rows)
    frame["run_status"] = RUN_STATUS
    frame["claim_boundary"] = CLAIM_BOUNDARY
    return frame


def _gate_matrix(
    *,
    case_rows: pd.DataFrame,
    rule_rows: pd.DataFrame,
    context: dict[str, Any],
    validation: dict[str, Any],
) -> pd.DataFrame:
    mapping = surface_claim_mapping_by_case(case_rows)
    object_summary = context["object_identity_summary"]
    wall_summary = context["wall_evidence_summary"]
    g4_9_summary = context["g4_9_summary"]
    g4_9a_summary = context["g4_9a_summary"]

    schema_application_ready = (
        not context["schema_application_summary"].get("failed_gates")
        and validation["required_columns_present"]
        and validation["required_values_valid"]
    )
    upstream_gates_ready = (
        not context["object_identity_summary"].get("failed_gates")
        and not context["wall_evidence_summary"].get("failed_gates")
        and not context["g4_9_summary"].get("failed_gates")
        and not context["g4_9a_summary"].get("failed_gates")
    )
    signature_surface_ready = (
        mapping.get("local_pair_016", {}).get("surface_level") == "signature_object"
        and mapping.get("local_pair_016", {}).get("claim_status") == "blocked"
        and _as_bool(object_summary.get("local_signature_object_certificate_available"))
        and _as_bool(object_summary.get("target_local_object_certified"))
    )
    endpoint_wall_requirement_retained = (
        not _as_bool(object_summary.get("object_identity_resolved"))
        and not _as_bool(object_summary.get("local_object_wall_evidence_audit_ready"))
        and not object_summary.get("wall_claim_ready_pairs")
    )
    typed_ladder_blocked = (
        mapping.get("local_pair_016", {}).get("relation_status") == "ladder"
        and mapping.get("local_pair_016", {}).get("claim_status") == "blocked"
        and rule_rows.loc[
            rule_rows["rule_id"].astype(str).eq("R3_typed_ladder_wall_rule_predeclared"),
            "rule_status",
        ].astype(str).eq("not_opened_requires_separate_contract").all()
    )
    boundary_controls_retained = (
        wall_summary.get("primitive_wall_evidence_ready_pairs") == ["local_pair_014"]
        and int(wall_summary.get("boundary_guard_closed_seed_count", 0)) > 0
        and int(g4_9_summary.get("control_wall_leak_case_count", -1)) == 0
        and int(g4_9a_summary.get("nonready_case_count", 0)) > 0
    )
    no_claim_promotion = (
        rule_rows["rule_status"].astype(str).ne("promoted").all()
        and bool(case_rows["wall_promotion_status"].astype(str).str.contains("not_promoted").all())
        and bool(case_rows["method_status"].astype(str).str.contains("not_method").all())
    )

    rows = [
        _gate_row(
            "G1_schema_application_and_adapter_ready",
            "Are the surface rows validated by the shared schema adapter?",
            {
                "schema_application_failed_gates": context[
                    "schema_application_summary"
                ].get("failed_gates"),
                "adapter_validation": validation,
            },
            "schema application has no failed gates and adapter validation passes",
            schema_application_ready,
        ),
        _gate_row(
            "G2_upstream_controls_ready",
            "Do object identity, wall evidence, and G4.9/G4.9A controls pass?",
            {
                "object_identity_failed_gates": object_summary.get("failed_gates"),
                "wall_evidence_failed_gates": wall_summary.get("failed_gates"),
                "g4_9_failed_gates": g4_9_summary.get("failed_gates"),
                "g4_9a_failed_gates": g4_9a_summary.get("failed_gates"),
            },
            "all upstream summaries report no failed gates",
            upstream_gates_ready,
        ),
        _gate_row(
            "G3_signature_object_surface_diagnostic_ready",
            "Can local signature-objects be accepted as diagnostic-only surface evidence?",
            {
                "016_mapping": mapping.get("local_pair_016"),
                "local_signature_object_certificate_available": object_summary.get(
                    "local_signature_object_certificate_available"
                ),
                "target_local_object_certified": object_summary.get(
                    "target_local_object_certified"
                ),
            },
            "016 is a blocked signature-object surface with local object certificate evidence",
            signature_surface_ready,
        ),
        _gate_row(
            "G4_endpoint_object_wall_requirement_retained",
            "Is endpoint-object identity still required before wall wording?",
            {
                "object_identity_resolved": object_summary.get("object_identity_resolved"),
                "local_object_wall_evidence_audit_ready": object_summary.get(
                    "local_object_wall_evidence_audit_ready"
                ),
                "wall_claim_ready_pairs": object_summary.get("wall_claim_ready_pairs"),
            },
            "016 object identity is unresolved and no wall-ready pair is opened",
            endpoint_wall_requirement_retained,
        ),
        _gate_row(
            "G5_typed_ladder_wall_rule_not_opened",
            "Is typed ladder evidence kept separate from wall evidence?",
            {
                "016_mapping": mapping.get("local_pair_016"),
                "typed_ladder_rule_status": rule_rows.loc[
                    rule_rows["rule_id"].astype(str).eq(
                        "R3_typed_ladder_wall_rule_predeclared"
                    ),
                    "rule_status",
                ].astype(str).tolist(),
            },
            "016 ladder relation remains blocked pending separate rule and guards",
            typed_ladder_blocked,
        ),
        _gate_row(
            "G6_boundary_controls_retained",
            "Do 005 and synthetic controls remain false-positive guards?",
            {
                "primitive_wall_evidence_ready_pairs": wall_summary.get(
                    "primitive_wall_evidence_ready_pairs"
                ),
                "boundary_guard_closed_seed_count": wall_summary.get(
                    "boundary_guard_closed_seed_count"
                ),
                "g4_9_control_wall_leak_case_count": g4_9_summary.get(
                    "control_wall_leak_case_count"
                ),
                "g4_9a_nonready_case_count": g4_9a_summary.get("nonready_case_count"),
                "g4_9a_partial_ready_case_count": g4_9a_summary.get(
                    "partial_ready_case_count"
                ),
            },
            "014 is the only wall-ready evidence pair and controls remain non-positive",
            boundary_controls_retained,
        ),
        _gate_row(
            "G7_no_claim_promotion",
            "Does the audit keep pathway, wall, method, quality, replay, and route claims closed?",
            {
                "rule_status_counts": _count_dict(rule_rows["rule_status"]),
                "case_claim_status_counts": validation["claim_status_counts"],
                "wall_promotion_status_counts": _count_dict(
                    case_rows["wall_promotion_status"]
                ),
                "method_status_counts": _count_dict(case_rows["method_status"]),
            },
            "only diagnostic surface wording is opened; no wall/pathway/method/quality/replay/route claim",
            no_claim_promotion,
        ),
    ]
    return pd.DataFrame(rows)


def _report(
    *,
    summary: dict[str, Any],
    case_rows: pd.DataFrame,
    rule_rows: pd.DataFrame,
    evidence_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# NanoClustering G4.8 First-Pass Object Surface Rule Decision",
            "",
            f"- status: `{summary['status']}`",
            f"- case_row_count: {summary['case_row_count']}",
            f"- rule_row_count: {summary['rule_row_count']}",
            f"- required_columns_present: {summary['required_columns_present']}",
            f"- required_values_valid: {summary['required_values_valid']}",
            f"- object_surface_rule_decision: {summary['object_surface_rule_decision']}",
            f"- gate_status_counts: {summary['gate_status_counts']}",
            f"- failed_gates: {summary['failed_gates']}",
            f"- interpretation: {summary['interpretation']}",
            f"- recommended_next_gate: {summary['recommended_next_gate']}",
            f"- claim_boundary: {summary['claim_boundary']}",
            "",
            "## Case Surface Rows",
            "",
            _markdown_table(
                case_rows,
                [
                    "case_id",
                    "surface_level",
                    "object_status",
                    "relation_status",
                    "claim_status",
                    "object_surface_rule",
                    "object_surface_decision",
                    "rule_effect",
                ],
            ),
            "",
            "## Rule Rows",
            "",
            _markdown_table(
                rule_rows,
                [
                    "rule_id",
                    "rule_status",
                    "evidence_case_ids",
                    "claim_effect",
                    "allowed_wording",
                    "blocked_wording",
                ],
            ),
            "",
            "## Evidence Rows",
            "",
            _markdown_table(
                evidence_rows,
                ["evidence_id", "evidence_status", "claim_effect", "observed"],
            ),
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
            "## Boundary",
            "",
            "This audit accepts signature-object evidence only as a diagnostic",
            "surface. Endpoint-object or separately predeclared typed-relation",
            "evidence is still required before wall/pathway wording.",
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    context = _load_context(args)
    case_rows = _case_surface_rows(context)
    validation = validate_surface_claim_rows(case_rows)
    rule_rows = _rule_rows(case_rows=case_rows, context=context)
    evidence_rows = _evidence_rows(context, validation)
    decision_rows = _decision_rows()
    gate_matrix = _gate_matrix(
        case_rows=case_rows,
        rule_rows=rule_rows,
        context=context,
        validation=validation,
    )
    failed_gates = gate_matrix.loc[
        gate_matrix["gate_status"].astype(str).eq("fail"), "gate_id"
    ].astype(str).tolist()
    summary = {
        "schema": "nanoclustering_g4_8_first_pass_object_surface_rule_decision_summary.v1",
        "status": RUN_STATUS,
        "schema_adapter": str(SCHEMA_ADAPTER_PATH.resolve()),
        "schema_adapter_version": SCHEMA_ADAPTER_VERSION,
        "schema_application_dir": str(args.schema_application_dir.resolve()),
        "object_identity_certificate_dir": str(
            args.object_identity_certificate_dir.resolve()
        ),
        "wall_evidence_dir": str(args.wall_evidence_dir.resolve()),
        "g4_9_dir": str(args.g4_9_dir.resolve()),
        "g4_9a_dir": str(args.g4_9a_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "case_row_count": int(len(case_rows)),
        "rule_row_count": int(len(rule_rows)),
        "evidence_row_count": int(len(evidence_rows)),
        "decision_row_count": int(len(decision_rows)),
        "required_columns": REQUIRED_COLUMNS,
        "required_columns_present": validation["required_columns_present"],
        "required_values_valid": validation["required_values_valid"],
        "missing_required_columns": validation["missing_required_columns"],
        "invalid_values_by_column": validation["invalid_values_by_column"],
        "case_claim_status_counts": validation["claim_status_counts"],
        "case_surface_level_counts": validation["surface_level_counts"],
        "rule_status_counts": _count_dict(rule_rows["rule_status"]),
        "gate_status_counts": _count_dict(gate_matrix["gate_status"]),
        "failed_gates": failed_gates,
        "object_surface_rule_decision": (
            "local_signature_objects_accepted_as_diagnostic_surface_only;"
            "endpoint_object_membership_required_for_wall;"
            "typed_ladder_wall_rule_not_opened"
        ),
        "diagnostic_surface_ready_pairs": ["local_pair_016"],
        "existing_local_primitive_wall_evidence_ready_pairs": context[
            "wall_evidence_summary"
        ].get("primitive_wall_evidence_ready_pairs"),
        "new_wall_claim_ready_pairs": [],
        "route_execution_opened": False,
        "method_claim_ready": False,
        "quality_claim_ready": False,
        "interpretation": (
            "The object-surface decision accepts 016 local signature-objects as "
            "diagnostic basin-state evidence only. 014 remains local object-level "
            "primitive wall evidence under the existing direct/recovery surface; "
            "005 remains a closed collapse guard. 016 wall/pathway wording stays "
            "blocked because endpoint-object identity is unresolved and no typed "
            "ladder wall rule has been predeclared."
        ),
        "recommended_next_gate": (
            "For any 016 promotion attempt, predeclare either an endpoint-object "
            "membership audit or a typed-ladder relation rule with negative controls; "
            "otherwise apply this adapter-backed object-surface rule to the next "
            "candidate panel without route expansion."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(case_rows, args.output_dir / CASE_SURFACE_ROWS_CSV)
    _write_csv(rule_rows, args.output_dir / RULE_ROWS_CSV)
    _write_csv(evidence_rows, args.output_dir / EVIDENCE_ROWS_CSV)
    _write_csv(decision_rows, args.output_dir / DECISION_ROWS_CSV)
    _write_csv(gate_matrix, args.output_dir / GATE_MATRIX_CSV)
    (args.output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_object_surface_rule_decision_config.v1",
        "schema_adapter": str(SCHEMA_ADAPTER_PATH.resolve()),
        "schema_adapter_version": SCHEMA_ADAPTER_VERSION,
        "schema_application_dir": str(args.schema_application_dir.resolve()),
        "object_identity_certificate_dir": str(
            args.object_identity_certificate_dir.resolve()
        ),
        "wall_evidence_dir": str(args.wall_evidence_dir.resolve()),
        "g4_9_dir": str(args.g4_9_dir.resolve()),
        "g4_9a_dir": str(args.g4_9a_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "required_columns": REQUIRED_COLUMNS,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (args.output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / REPORT_MD).write_text(
        _report(
            summary=summary,
            case_rows=case_rows,
            rule_rows=rule_rows,
            evidence_rows=evidence_rows,
            decision_rows=decision_rows,
            gate_matrix=gate_matrix,
        ),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schema-application-dir",
        type=Path,
        default=DEFAULT_SCHEMA_APPLICATION_DIR,
        help="Directory containing the surface-claim schema application audit.",
    )
    parser.add_argument(
        "--object-identity-certificate-dir",
        type=Path,
        default=DEFAULT_016_OBJECT_IDENTITY_CERTIFICATE_DIR,
        help="Directory containing the 016 object-identity certificate audit.",
    )
    parser.add_argument(
        "--wall-evidence-dir",
        type=Path,
        default=DEFAULT_014_WALL_EVIDENCE_DIR,
        help="Directory containing the 014/005 primitive wall-evidence audit.",
    )
    parser.add_argument(
        "--g4-9-dir",
        type=Path,
        default=G4_9_DIR,
        help="Directory containing the G4.9 synthetic primitive-wall demo.",
    )
    parser.add_argument(
        "--g4-9a-dir",
        type=Path,
        default=G4_9A_DIR,
        help="Directory containing the G4.9A synthetic parameter-localization map.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for this object-surface rule decision audit.",
    )
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
