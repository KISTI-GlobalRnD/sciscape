#!/usr/bin/env python3
"""Apply the surface-qualified basin claim schema to 014/016/005.

This read-only audit consumes the current endpoint-object and 016
object-identity certificate artifacts, then normalizes the three active cases
onto the surface-qualified claim schema:

``surface_level, object_status, relation_status, claim_status``.

It does not rerun Leiden, expand route rows, promote pathway labels or walls,
evaluate quality/cost value, replay full NanoClustering, or claim method
success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from audit_leiden_basin_nanoclustering_g4_8_first_pass_016_object_identity_certificate import (
    DEFAULT_OUTPUT_DIR as DEFAULT_OBJECT_IDENTITY_CERTIFICATE_DIR,
    GATE_MATRIX_CSV as OBJECT_IDENTITY_GATE_MATRIX_CSV,
    LOCAL_OBJECT_ROWS_CSV as OBJECT_IDENTITY_LOCAL_OBJECT_ROWS_CSV,
    RELATION_ROWS_CSV as OBJECT_IDENTITY_RELATION_ROWS_CSV,
    SUMMARY_JSON as OBJECT_IDENTITY_SUMMARY_JSON,
)
from audit_leiden_basin_nanoclustering_g4_8_first_pass_symmetric_endpoint_objects import (
    DEFAULT_OUTPUT_DIR as DEFAULT_SYMMETRIC_ENDPOINT_OBJECT_DIR,
    GATE_MATRIX_CSV as SYMMETRIC_GATE_MATRIX_CSV,
    PAIR_SUMMARY_ROWS_CSV as SYMMETRIC_PAIR_SUMMARY_ROWS_CSV,
    SUMMARY_JSON as SYMMETRIC_SUMMARY_JSON,
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
    surface_claim_case_row,
    surface_claim_count_dict as _count_dict,
    surface_claim_gate_row as _gate_row,
    surface_claim_json_dump as _json_dump,
    surface_claim_mapping_by_case,
    surface_claim_required_columns_named,
    validate_surface_claim_rows,
)


DEFAULT_SCHEMA_DOC = (
    Path(__file__).resolve().parents[5]
    / "docs"
    / "research"
    / "leiden_basin"
    / "core"
    / "leiden_basin_surface_claim_schema.md"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_surface_claim_schema_application_gamma1e5_20260608"
)
SCHEMA_ADAPTER_PATH = Path(__file__).resolve().parent / "surface_claim_schema_adapter.py"

CASE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_claim_schema_application_case_rows.csv"
)
EVIDENCE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_claim_schema_application_evidence_rows.csv"
)
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_claim_schema_application_decision_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_surface_claim_schema_application_gate_matrix.csv"
)
SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_surface_claim_schema_application_summary.json"
)
CONFIG_JSON = (
    "nanoclustering_g4_8_first_pass_surface_claim_schema_application_config.json"
)
REPORT_MD = (
    "nanoclustering_g4_8_first_pass_surface_claim_schema_application_report.md"
)

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_surface_claim_schema_application"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_surface_claim_schema_application"
WALL_PROMOTION_STATUS = "not_promoted_schema_application_only"
METHOD_STATUS = "surface_claim_schema_application_audit_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass surface-claim schema application audit only; "
    "reads current 014/005 symmetric endpoint-object evidence and the 016 "
    "object-identity certificate. It normalizes cases to surface_level, "
    "object_status, relation_status, and claim_status. It does not rerun Leiden, "
    "expand routes, promote pathway labels or walls, evaluate quality/cost value, "
    "replay full NanoClustering, or claim method success."
)

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


def _as_int(value: Any, default: int = 0) -> int:
    if pd.isna(value):
        return default
    return int(float(value))


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
        "schema_doc_text": args.schema_doc.read_text(encoding="utf-8"),
        "symmetric_summary": _read_json(
            args.symmetric_endpoint_object_dir / SYMMETRIC_SUMMARY_JSON
        ),
        "symmetric_gates": _read_csv(
            args.symmetric_endpoint_object_dir / SYMMETRIC_GATE_MATRIX_CSV
        ),
        "symmetric_pair_summary": _read_csv(
            args.symmetric_endpoint_object_dir / SYMMETRIC_PAIR_SUMMARY_ROWS_CSV
        ),
        "object_identity_summary": _read_json(
            args.object_identity_certificate_dir / OBJECT_IDENTITY_SUMMARY_JSON
        ),
        "object_identity_gates": _read_csv(
            args.object_identity_certificate_dir / OBJECT_IDENTITY_GATE_MATRIX_CSV
        ),
        "object_identity_local_object_rows": _read_csv(
            args.object_identity_certificate_dir / OBJECT_IDENTITY_LOCAL_OBJECT_ROWS_CSV
        ),
        "object_identity_relation_rows": _read_csv(
            args.object_identity_certificate_dir / OBJECT_IDENTITY_RELATION_ROWS_CSV
        ),
    }


def _symmetric_pair_row(pair_summary: pd.DataFrame, pair_id: str) -> dict[str, Any]:
    rows = pair_summary[pair_summary["local_pair_id"].astype(str).eq(pair_id)]
    if rows.empty:
        raise ValueError(f"missing symmetric pair row: {pair_id}")
    return rows.iloc[0].to_dict()


def _case_rows(context: dict[str, Any]) -> pd.DataFrame:
    pair_summary = context["symmetric_pair_summary"]
    object_identity_summary = context["object_identity_summary"]
    local_object_rows = context["object_identity_local_object_rows"]
    relation_rows = context["object_identity_relation_rows"]

    pair_014 = _symmetric_pair_row(pair_summary, "local_pair_014")
    pair_005 = _symmetric_pair_row(pair_summary, "local_pair_005")
    source_rows = local_object_rows[
        local_object_rows["signature_id"].astype(str).isin(
            {"5536308f50fc", "c475d13ca500"}
        )
    ]
    transient_rows = local_object_rows[
        local_object_rows["signature_id"].astype(str).eq("aeb59ab537e6")
    ]
    target_rows = local_object_rows[
        local_object_rows["signature_id"].astype(str).eq("3c9b8a190753")
    ]

    rows = [
        surface_claim_case_row(
            case_id="local_pair_014",
            case_role="endpoint_object_primitive_wall_candidate",
            surface_level="endpoint_object",
            object_status="certified",
            relation_status="clean",
            claim_status="diagnostic_only",
            promotion_level="L5_typed_relation",
            relation_evidence_level="L5_typed_relation",
            promotion_blocker="wall_promotion_not_opened;quality_not_tested;method_not_tested",
            allowed_wording=(
                "endpoint-object primitive wall candidate under the "
                "direct-only/recovery-loop surface"
            ),
            blocked_wording=(
                "general wall;method claim;quality claim;full replay claim"
            ),
            object_status_detail="exclusive target object certified",
            relation_status_detail="32 clean source-to-exclusive-target object relations",
            source_artifacts=context["symmetric_summary"]["output_dir"],
            evidence_counts={
                "route_count": _as_int(pair_014["route_count"]),
                "exclusive_target_object_count": _as_int(
                    pair_014["exclusive_target_object_count"]
                ),
                "clean_relation_count": _as_int(pair_014["clean_relation_count"]),
                "source_target_collapse_relation_count": _as_int(
                    pair_014["source_target_collapse_relation_count"]
                ),
            },
            claim_boundary=CLAIM_BOUNDARY,
            route_execution_status=ROUTE_EXECUTION_STATUS,
            wall_promotion_status=WALL_PROMOTION_STATUS,
            method_status=METHOD_STATUS,
            run_status=RUN_STATUS,
        ),
        surface_claim_case_row(
            case_id="local_pair_016",
            case_role="signature_object_transition_band_case",
            surface_level="signature_object",
            object_status="split",
            relation_status="ladder",
            claim_status="blocked",
            promotion_level="L3_local_signature_object",
            relation_evidence_level="L5_typed_relation_observed_but_blocked",
            promotion_blocker=(
                "external_membership_absent;source_family_split;"
                "transient_nonendpoint;wall_relation_not_clean"
            ),
            allowed_wording=(
                "signature-object transition-band case with certified target "
                "local object"
            ),
            blocked_wording="endpoint basin;object wall;pathway label;method claim",
            object_status_detail=(
                "target certified; source strict/guard split; transient nonendpoint"
            ),
            relation_status_detail=(
                "24 direct source-to-target signature relations and 24 recovery "
                "ladder relations"
            ),
            source_artifacts=object_identity_summary["output_dir"],
            evidence_counts={
                "local_object_row_count": _as_int(
                    object_identity_summary["local_object_row_count"]
                ),
                "direct_relation_count": int(
                    relation_rows["relation_class"]
                    .astype(str)
                    .eq("direct_source_component_to_target_signature")
                    .sum()
                ),
                "ladder_relation_count": int(
                    relation_rows["relation_class"]
                    .astype(str)
                    .eq("recovery_ladder_source_target_transient_return")
                    .sum()
                ),
                "source_signature_count": int(len(source_rows)),
                "target_local_object_certified": bool(
                    target_rows["target_local_object_certified"].map(_as_bool).all()
                ),
                "transient_endpoint_object_certified": bool(
                    transient_rows["transient_endpoint_object_certified"]
                    .map(_as_bool)
                    .all()
                ),
            },
            claim_boundary=CLAIM_BOUNDARY,
            route_execution_status=ROUTE_EXECUTION_STATUS,
            wall_promotion_status=WALL_PROMOTION_STATUS,
            method_status=METHOD_STATUS,
            run_status=RUN_STATUS,
        ),
        surface_claim_case_row(
            case_id="local_pair_005",
            case_role="boundary_collapse_control",
            surface_level="endpoint_object",
            object_status="collapse",
            relation_status="collapse",
            claim_status="closed",
            promotion_level="L5_typed_relation",
            relation_evidence_level="L5_collapse_relation_control",
            promotion_blocker="source_target_collapse;negative_boundary_control",
            allowed_wording="boundary/collapse control",
            blocked_wording="positive wall;endpoint-object candidate;method claim",
            object_status_detail="partial source-target collapse boundary",
            relation_status_detail="24 clean relations plus 8 collapse relations",
            source_artifacts=context["symmetric_summary"]["output_dir"],
            evidence_counts={
                "route_count": _as_int(pair_005["route_count"]),
                "exclusive_target_object_count": _as_int(
                    pair_005["exclusive_target_object_count"]
                ),
                "clean_relation_count": _as_int(pair_005["clean_relation_count"]),
                "source_target_collapse_relation_count": _as_int(
                    pair_005["source_target_collapse_relation_count"]
                ),
            },
            claim_boundary=CLAIM_BOUNDARY,
            route_execution_status=ROUTE_EXECUTION_STATUS,
            wall_promotion_status=WALL_PROMOTION_STATUS,
            method_status=METHOD_STATUS,
            run_status=RUN_STATUS,
        ),
    ]
    return pd.DataFrame(rows)


def _evidence_rows(
    *,
    case_rows: pd.DataFrame,
    context: dict[str, Any],
) -> pd.DataFrame:
    schema_text = context["schema_doc_text"]
    return pd.DataFrame(
        [
            {
                "evidence_id": "E1_schema_doc_available",
                "evidence_question": "Is the surface-qualified schema available and does it define required columns?",
                "observed": {
                    "has_core_definition": "specified partition-state surface"
                    in schema_text,
                    "required_columns": REQUIRED_COLUMNS,
                    "all_required_columns_named": surface_claim_required_columns_named(
                        schema_text
                    ),
                },
                "evidence_status": "schema_available",
                "claim_effect": "permits_schema_application_only",
            },
            {
                "evidence_id": "E2_source_artifacts_ready",
                "evidence_question": "Do source audits pass before schema application?",
                "observed": {
                    "symmetric_failed_gates": context["symmetric_summary"].get(
                        "failed_gates"
                    ),
                    "object_identity_failed_gates": context[
                        "object_identity_summary"
                    ].get("failed_gates"),
                    "symmetric_gate_status_counts": _count_dict(
                        context["symmetric_gates"]["gate_status"]
                    ),
                    "object_identity_gate_status_counts": _count_dict(
                        context["object_identity_gates"]["gate_status"]
                    ),
                },
                "evidence_status": "source_artifacts_ready",
                "claim_effect": "supports_read_only_mapping",
            },
            {
                "evidence_id": "E3_required_columns_materialized",
                "evidence_question": "Are required schema columns present for all cases?",
                "observed": {
                    "case_count": int(len(case_rows)),
                    "required_columns": REQUIRED_COLUMNS,
                    "case_ids": case_rows["case_id"].astype(str).tolist(),
                },
                "evidence_status": "required_columns_materialized",
                "claim_effect": "standardizes_case_comparison",
            },
            {
                "evidence_id": "E4_claims_remain_bounded",
                "evidence_question": "Does the application avoid method, quality, replay, and route claims?",
                "observed": {
                    "claim_status_counts": _count_dict(case_rows["claim_status"]),
                    "wall_promotion_status_counts": _count_dict(
                        case_rows["wall_promotion_status"]
                    ),
                    "method_status_counts": _count_dict(case_rows["method_status"]),
                },
                "evidence_status": "claims_bounded",
                "claim_effect": "prevents_surface_success_from_becoming_method_claim",
            },
        ]
    )


def _decision_rows() -> pd.DataFrame:
    rows = [
        {
            "decision_id": "D1",
            "decision": "schema_application_ready",
            "rationale": "The 014/016/005 cases can be represented with the required surface_level, object_status, relation_status, and claim_status columns.",
        },
        {
            "decision_id": "D2",
            "decision": "014_endpoint_object_diagnostic_only",
            "rationale": "014 has certified endpoint-object and clean relation evidence, but method, quality, replay, and general-wall claims remain closed.",
        },
        {
            "decision_id": "D3",
            "decision": "016_signature_object_wall_blocked",
            "rationale": "016 has local signature-object evidence and typed direct/ladder relations, but source split and nonendpoint transient blockers prevent object-wall wording.",
        },
        {
            "decision_id": "D4",
            "decision": "005_boundary_collapse_control",
            "rationale": "005 is retained as a collapse control, not a positive wall or endpoint-object candidate.",
        },
        {
            "decision_id": "D5",
            "decision": "schema_adapter_contract_available",
            "rationale": "Future basin-surface scripts should import the reusable surface_claim_schema_adapter before route execution or label promotion.",
        },
    ]
    frame = pd.DataFrame(rows)
    frame["run_status"] = RUN_STATUS
    frame["claim_boundary"] = CLAIM_BOUNDARY
    return frame


def _gate_matrix(
    *,
    case_rows: pd.DataFrame,
    context: dict[str, Any],
) -> pd.DataFrame:
    validation = validate_surface_claim_rows(case_rows)
    mapping_by_case = surface_claim_mapping_by_case(case_rows)
    required_columns_named = surface_claim_required_columns_named(
        context["schema_doc_text"]
    )
    rows = [
        _gate_row(
            "G1_schema_source_available",
            "Is the surface-qualified schema available and used as the audit contract?",
            {
                "required_columns": REQUIRED_COLUMNS,
                "schema_mentions_required_columns": required_columns_named,
            },
            "schema document exists and names all required columns",
            required_columns_named,
        ),
        _gate_row(
            "G2_source_artifacts_pass",
            "Do upstream object/source artifacts have no failed gates?",
            {
                "symmetric_failed_gates": context["symmetric_summary"].get(
                    "failed_gates"
                ),
                "object_identity_failed_gates": context["object_identity_summary"].get(
                    "failed_gates"
                ),
            },
            "symmetric endpoint-object and 016 object-identity audits have no failed gates",
            not context["symmetric_summary"].get("failed_gates")
            and not context["object_identity_summary"].get("failed_gates"),
        ),
        _gate_row(
            "G3_required_columns_present",
            "Are all required schema columns present?",
            {
                "required_columns": REQUIRED_COLUMNS,
                "observed_columns": validation["observed_columns"],
                "missing_required_columns": validation["missing_required_columns"],
            },
            "surface_level, object_status, relation_status, claim_status present",
            validation["required_columns_present"],
        ),
        _gate_row(
            "G4_required_values_valid",
            "Do required columns use the allowed schema values?",
            {
                "surface_levels": validation["surface_level_counts"],
                "object_statuses": validation["object_status_counts"],
                "relation_statuses": validation["relation_status_counts"],
                "claim_statuses": validation["claim_status_counts"],
                "invalid_values_by_column": validation["invalid_values_by_column"],
            },
            "all values are within the surface-claim schema vocabulary",
            validation["required_values_valid"],
        ),
        _gate_row(
            "G5_case_mapping_matches_current_schema",
            "Does the mapping preserve the current 014/016/005 interpretation?",
            mapping_by_case,
            "014 endpoint-object diagnostic, 016 signature-object blocked, 005 collapse closed",
            mapping_by_case.get("local_pair_014", {}).get("surface_level")
            == "endpoint_object"
            and mapping_by_case.get("local_pair_014", {}).get("claim_status")
            == "diagnostic_only"
            and mapping_by_case.get("local_pair_016", {}).get("surface_level")
            == "signature_object"
            and mapping_by_case.get("local_pair_016", {}).get("claim_status")
            == "blocked"
            and mapping_by_case.get("local_pair_005", {}).get("object_status")
            == "collapse"
            and mapping_by_case.get("local_pair_005", {}).get("claim_status")
            == "closed",
        ),
        _gate_row(
            "G6_no_claim_promotion",
            "Are pathway, wall, method, quality/cost, replay, and route claims closed?",
            {
                "claim_status_counts": _count_dict(case_rows["claim_status"]),
                "wall_promotion_status_counts": _count_dict(
                    case_rows["wall_promotion_status"]
                ),
                "method_status_counts": _count_dict(case_rows["method_status"]),
            },
            "no case opens wall/method/quality/replay or route execution claims",
            bool(case_rows["wall_promotion_status"].astype(str).eq(WALL_PROMOTION_STATUS).all())
            and bool(case_rows["method_status"].astype(str).eq(METHOD_STATUS).all()),
        ),
    ]
    return pd.DataFrame(rows)


def _report(
    *,
    summary: dict[str, Any],
    case_rows: pd.DataFrame,
    evidence_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# NanoClustering G4.8 First-Pass Surface Claim Schema Application",
            "",
            f"- status: `{summary['status']}`",
            f"- case_row_count: {summary['case_row_count']}",
            f"- required_columns_present: {summary['required_columns_present']}",
            f"- required_values_valid: {summary['required_values_valid']}",
            f"- gate_status_counts: {summary['gate_status_counts']}",
            f"- failed_gates: {summary['failed_gates']}",
            f"- interpretation: {summary['interpretation']}",
            f"- recommended_next_gate: {summary['recommended_next_gate']}",
            f"- claim_boundary: {summary['claim_boundary']}",
            "",
            "## Case Rows",
            "",
            _markdown_table(
                case_rows,
                [
                    "case_id",
                    "case_role",
                    "surface_level",
                    "object_status",
                    "relation_status",
                    "claim_status",
                    "promotion_level",
                    "relation_evidence_level",
                    "promotion_blocker",
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
            "This audit applies the surface-qualified schema. It does not change the",
            "underlying evidence: 014 remains diagnostic-only, 016 remains wall-blocked,",
            "and 005 remains a closed collapse control.",
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    context = _load_context(args)
    case_rows = _case_rows(context)
    evidence_rows = _evidence_rows(case_rows=case_rows, context=context)
    decision_rows = _decision_rows()
    gate_matrix = _gate_matrix(case_rows=case_rows, context=context)
    validation = validate_surface_claim_rows(case_rows)

    failed_gates = gate_matrix.loc[
        gate_matrix["gate_status"].astype(str).eq("fail"), "gate_id"
    ].astype(str).tolist()
    summary = {
        "schema": "nanoclustering_g4_8_first_pass_surface_claim_schema_application_summary.v1",
        "status": RUN_STATUS,
        "schema_doc": str(args.schema_doc.resolve()),
        "schema_adapter": str(SCHEMA_ADAPTER_PATH.resolve()),
        "schema_adapter_version": SCHEMA_ADAPTER_VERSION,
        "symmetric_endpoint_object_dir": str(
            args.symmetric_endpoint_object_dir.resolve()
        ),
        "object_identity_certificate_dir": str(
            args.object_identity_certificate_dir.resolve()
        ),
        "output_dir": str(args.output_dir.resolve()),
        "case_row_count": int(len(case_rows)),
        "evidence_row_count": int(len(evidence_rows)),
        "decision_row_count": int(len(decision_rows)),
        "required_columns": REQUIRED_COLUMNS,
        "required_columns_present": validation["required_columns_present"],
        "required_values_valid": validation["required_values_valid"],
        "missing_required_columns": validation["missing_required_columns"],
        "invalid_values_by_column": validation["invalid_values_by_column"],
        "case_claim_status_counts": validation["claim_status_counts"],
        "case_surface_level_counts": validation["surface_level_counts"],
        "case_object_status_counts": validation["object_status_counts"],
        "case_relation_status_counts": validation["relation_status_counts"],
        "gate_status_counts": _count_dict(gate_matrix["gate_status"]),
        "failed_gates": failed_gates,
        "wall_claim_ready_pairs": [],
        "method_claim_ready": False,
        "quality_claim_ready": False,
        "route_execution_opened": False,
        "interpretation": (
            "The surface-qualified schema now represents the current 014/016/005 "
            "evidence with one shared vocabulary: 014 is an endpoint-object "
            "diagnostic primitive wall candidate, 016 is a signature-object "
            "transition-band case whose wall claim is blocked, and 005 is a "
            "closed collapse control."
        ),
        "recommended_next_gate": (
            "Use surface_claim_schema_adapter.py in the next basin-surface audit so "
            "every new case reports surface_level, object_status, relation_status, "
            "and claim_status before route execution, label promotion, or "
            "method/quality language."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(case_rows, args.output_dir / CASE_ROWS_CSV)
    _write_csv(evidence_rows, args.output_dir / EVIDENCE_ROWS_CSV)
    _write_csv(decision_rows, args.output_dir / DECISION_ROWS_CSV)
    _write_csv(gate_matrix, args.output_dir / GATE_MATRIX_CSV)
    (args.output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_surface_claim_schema_application_config.v1",
        "schema_doc": str(args.schema_doc.resolve()),
        "schema_adapter": str(SCHEMA_ADAPTER_PATH.resolve()),
        "schema_adapter_version": SCHEMA_ADAPTER_VERSION,
        "symmetric_endpoint_object_dir": str(
            args.symmetric_endpoint_object_dir.resolve()
        ),
        "object_identity_certificate_dir": str(
            args.object_identity_certificate_dir.resolve()
        ),
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
        "--schema-doc",
        type=Path,
        default=DEFAULT_SCHEMA_DOC,
        help="Surface-qualified basin claim schema document.",
    )
    parser.add_argument(
        "--symmetric-endpoint-object-dir",
        type=Path,
        default=DEFAULT_SYMMETRIC_ENDPOINT_OBJECT_DIR,
        help="Directory containing the first-pass symmetric endpoint-object audit.",
    )
    parser.add_argument(
        "--object-identity-certificate-dir",
        type=Path,
        default=DEFAULT_OBJECT_IDENTITY_CERTIFICATE_DIR,
        help="Directory containing the 016 object-identity certificate audit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for this schema application audit.",
    )
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
