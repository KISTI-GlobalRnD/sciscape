#!/usr/bin/env python3
"""Reusable helpers for surface-qualified Leiden basin claim audits."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from run_leiden_basin_nanoclustering_role_local_route_pilot import _json_safe


SCHEMA_ADAPTER_VERSION = "surface_claim_schema_adapter.v1"
REQUIRED_COLUMNS = [
    "surface_level",
    "object_status",
    "relation_status",
    "claim_status",
]
ALLOWED_SURFACE_LEVELS = {
    "state",
    "signature_object",
    "endpoint_object",
    "relation",
    "quality",
}
ALLOWED_OBJECT_STATUSES = {
    "certified",
    "split",
    "nonendpoint",
    "collapse",
    "unknown",
    "not_applicable",
}
ALLOWED_RELATION_STATUSES = {
    "clean",
    "ladder",
    "collapse",
    "direct_only",
    "unresolved",
    "not_applicable",
}
ALLOWED_CLAIM_STATUSES = {"open", "diagnostic_only", "blocked", "closed"}
ALLOWED_VALUES_BY_COLUMN = {
    "surface_level": ALLOWED_SURFACE_LEVELS,
    "object_status": ALLOWED_OBJECT_STATUSES,
    "relation_status": ALLOWED_RELATION_STATUSES,
    "claim_status": ALLOWED_CLAIM_STATUSES,
}


def surface_claim_json_dump(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True)


def surface_claim_count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {
        str(key): int(value)
        for key, value in series.value_counts(dropna=False).items()
    }


def surface_claim_required_columns_named(schema_doc_text: str) -> bool:
    return all(column in schema_doc_text for column in REQUIRED_COLUMNS)


def surface_claim_gate_row(
    gate_id: str,
    question: str,
    observed: Any,
    minimum_or_rule: str,
    passed: bool,
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "question": question,
        "observed": surface_claim_json_dump(observed),
        "minimum_or_rule": minimum_or_rule,
        "gate_status": "pass" if bool(passed) else "fail",
    }


def surface_claim_case_row(
    *,
    case_id: str,
    case_role: str,
    surface_level: str,
    object_status: str,
    relation_status: str,
    claim_status: str,
    promotion_level: str,
    relation_evidence_level: str,
    promotion_blocker: str,
    allowed_wording: str,
    blocked_wording: str,
    object_status_detail: str,
    relation_status_detail: str,
    source_artifacts: str,
    evidence_counts: dict[str, Any],
    claim_boundary: str,
    route_execution_status: str,
    wall_promotion_status: str,
    method_status: str,
    run_status: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "case_role": case_role,
        "surface_level": surface_level,
        "object_status": object_status,
        "relation_status": relation_status,
        "claim_status": claim_status,
        "promotion_level": promotion_level,
        "relation_evidence_level": relation_evidence_level,
        "promotion_blocker": promotion_blocker,
        "allowed_wording": allowed_wording,
        "blocked_wording": blocked_wording,
        "object_status_detail": object_status_detail,
        "relation_status_detail": relation_status_detail,
        "source_artifacts": source_artifacts,
        "evidence_counts": surface_claim_json_dump(evidence_counts),
        "claim_boundary": claim_boundary,
        "route_execution_status": route_execution_status,
        "wall_promotion_status": wall_promotion_status,
        "method_status": method_status,
        "run_status": run_status,
    }


def _invalid_values(case_rows: pd.DataFrame, column: str) -> list[str]:
    if column not in case_rows.columns:
        return []
    allowed = ALLOWED_VALUES_BY_COLUMN[column]
    return sorted(
        {
            str(value)
            for value in case_rows[column].astype(str).unique()
            if str(value) not in allowed
        }
    )


def _column_counts(case_rows: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in case_rows.columns:
        return {}
    return surface_claim_count_dict(case_rows[column])


def validate_surface_claim_rows(case_rows: pd.DataFrame) -> dict[str, Any]:
    observed_columns = sorted(str(column) for column in case_rows.columns)
    missing_required_columns = [
        column for column in REQUIRED_COLUMNS if column not in case_rows.columns
    ]
    invalid_values_by_column = {
        column: invalid
        for column in REQUIRED_COLUMNS
        if (invalid := _invalid_values(case_rows, column))
    }
    required_columns_present = not missing_required_columns
    required_values_valid = required_columns_present and not invalid_values_by_column
    return {
        "required_columns": list(REQUIRED_COLUMNS),
        "observed_columns": observed_columns,
        "missing_required_columns": missing_required_columns,
        "invalid_values_by_column": invalid_values_by_column,
        "required_columns_present": bool(required_columns_present),
        "required_values_valid": bool(required_values_valid),
        "claim_status_counts": _column_counts(case_rows, "claim_status"),
        "surface_level_counts": _column_counts(case_rows, "surface_level"),
        "object_status_counts": _column_counts(case_rows, "object_status"),
        "relation_status_counts": _column_counts(case_rows, "relation_status"),
    }


def surface_claim_mapping_by_case(case_rows: pd.DataFrame) -> dict[str, dict[str, str]]:
    mapping_columns = [
        column
        for column in [*REQUIRED_COLUMNS, "promotion_level"]
        if column in case_rows.columns
    ]
    case_column = "case_id" if "case_id" in case_rows.columns else None
    mapping: dict[str, dict[str, str]] = {}
    for index, row in case_rows.iterrows():
        case_id = str(row[case_column]) if case_column else str(index)
        mapping[case_id] = {column: str(row[column]) for column in mapping_columns}
    return mapping
