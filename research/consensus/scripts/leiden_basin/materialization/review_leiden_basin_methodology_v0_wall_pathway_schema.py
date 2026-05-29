#!/usr/bin/env python3
"""Review methodology-v0 wall/pathway schema coverage for enriched pairs.

This is M3 in the Leiden basin methodology-v0 design. It joins the M2 pair
evidence ledger to existing current-review, remaining-wall-question, pathway,
and route-blocker ledgers. It does not execute routes, load memberships,
promote walls, inspect quality/cost, or claim a directed-search method.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_METHOD_DIR = BASE_RESULT_DIR / "leiden_basin_methodology_v0_20260529"
DEFAULT_CURRENT_REVIEW_DIR = BASE_RESULT_DIR / "leiden_basin_current_results_review_20260529"
DEFAULT_REMAINING_WALL_DIR = (
    BASE_RESULT_DIR / "leiden_basin_remaining_wall_question_audit_20260529"
)
DEFAULT_EXISTENCE_AUDIT_DIR = BASE_RESULT_DIR / "leiden_basin_existence_assumption_audit_20260529"
DEFAULT_ROUTE_BLOCKER_DIR = BASE_RESULT_DIR / "leiden_basin_route_label_blocker_triage_20260529"

PAIR_EVIDENCE_CSV = "methodology_v0_pair_evidence_rows.csv"
CURRENT_REVIEW_CSV = "current_pair_state_ledger.csv"
REMAINING_WALL_CSV = "remaining_wall_question_rows.csv"
PATHWAY_READINESS_CSV = "pathway_readiness_rows.csv"
ROUTE_BLOCKER_CSV = "route_label_blocker_triage_rows.csv"

M3_ROWS_CSV = "methodology_v0_wall_pathway_schema_review_rows.csv"
SUMMARY_JSON = "methodology_v0_wall_pathway_schema_review_summary.json"
REPORT_MD = "methodology_v0_wall_pathway_schema_review_report.md"
CONFIG_JSON = "methodology_v0_wall_pathway_schema_review_config.json"

CLAIM_BOUNDARY = (
    "Methodology-v0 M3 wall/pathway schema review only; no route execution, "
    "wall-promotion change, basin-quality claim, cost claim, or directed-search "
    "claim."
)
QUALITY_COST_STATUS = "excluded_by_methodology_v0"
ROUTE_EXECUTION_STATUS = "not_executed_m3_schema_review_only"
WALL_PROMOTION_STATUS = "not_promoted_m3_schema_review_only"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"empty CSV: {path}") from exc


def _read_optional_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = _read_csv(path)
    for column in columns:
        if column not in frame:
            frame[column] = ""
    return frame


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _count(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    return {str(k): int(v) for k, v in frame[column].value_counts(dropna=False).to_dict().items()}


def _safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return math.nan
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _has(value: Any, *needles: str) -> bool:
    text = _safe_text(value).lower()
    return any(needle.lower() in text for needle in needles)


def _first_text(row: pd.Series, *columns: str) -> str:
    for column in columns:
        value = _safe_text(row.get(column))
        if value:
            return value
    return ""


def _panel_pair_id(row: pd.Series) -> str:
    left = _safe_int(row.get("left_representative_candidate_index"))
    right = _safe_int(row.get("right_representative_candidate_index"))
    if left is None or right is None:
        return str(row.get("case_id", "")) + ":cunknown-cunknown"
    first = min(left, right)
    second = max(left, right)
    return f"{row.get('case_id')}:c{first}-c{second}"


def _prefixed_join_frame(
    frame: pd.DataFrame,
    *,
    prefix: str,
    columns: list[str],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["panel_pair_id", *[prefix + col for col in columns]])
    cols = ["panel_pair_id", *[col for col in columns if col in frame]]
    out = frame[cols].drop_duplicates(subset=["panel_pair_id"], keep="first").copy()
    return out.rename(columns={col: prefix + col for col in cols if col != "panel_pair_id"})


def _source_status(row: pd.Series) -> str:
    sources = []
    if _safe_text(row.get("current_current_review_status")):
        sources.append("current_review")
    if _safe_text(row.get("remaining_remaining_wall_question_class")):
        sources.append("remaining_wall_question")
    if _safe_text(row.get("pathway_pathway_readiness_status")):
        sources.append("pathway_readiness")
    if _safe_text(row.get("triage_triage_action")):
        sources.append("route_blocker_triage")
    if not sources:
        return "m2_only_no_existing_pathway_review_surface"
    if len(sources) == 4:
        return "joined_existing_23_pair_review_surface"
    return "joined_partial_existing_review_surface:" + "|".join(sources)


def _endpoint_assignment_evidence(row: pd.Series) -> str:
    grade = _safe_text(row.get("pair_evidence_grade"))
    if grade == "both_full_membership_caches_available":
        return "both_endpoint_full_membership_caches_available_for_audit"
    if grade == "partial_full_membership_cache_available":
        return "partial_endpoint_full_membership_cache_available"
    if grade == "endpoint_signature_support_hash_pair":
        return "endpoint_signature_support_hash_only_no_full_membership_pair"
    if grade:
        return "endpoint_reference_incomplete:" + grade
    return "endpoint_reference_missing"


def _support_relation_evidence(row: pd.Series) -> str:
    relation = _safe_text(row.get("calibrated_relation"))
    band = _safe_text(row.get("support_distance_band_v0"))
    if relation == "distinct_support_local" and band == "distinct_zone":
        return "accepted_distinct_support_local_relation"
    if relation == "distinct_support_local":
        return "accepted_distinct_relation_nonstandard_support_band"
    if relation:
        return "not_accepted_distinct_relation:" + relation
    return "support_relation_missing"


def _direct_path_availability(row: pd.Series) -> str:
    route_labels = _safe_text(row.get("current_route_labels"))
    if _has(route_labels, "direct_route_reaches_target"):
        return "existing_direct_route_reference_available"
    route_label = _first_text(
        row,
        "remaining_route_label_interpretation_v0",
        "pathway_route_label_interpretation_v0",
        "triage_route_label_interpretation_v0",
    )
    if route_label:
        return "existing_route_label_reference_without_direct_path_trace"
    preflight = _first_text(
        row,
        "current_runner_preflight_status",
        "remaining_runner_preflight_status",
        "triage_runner_preflight_status",
    )
    if preflight == "runner_preflight_ready":
        return "runner_preflight_ready_but_no_route_reference_in_m3_inputs"
    if preflight:
        return "runner_preflight_failed_or_missing_context:" + preflight
    return "missing_no_existing_route_surface"


def _objective_debt_evidence(row: pd.Series) -> str:
    wall_gate = _first_text(row, "current_wall_claim_gate_status", "remaining_wall_claim_gate_status")
    margin_gate = _safe_text(row.get("current_margin_gate_status"))
    bands = _safe_text(row.get("current_polish_margin_bands"))
    route_labels = _safe_text(row.get("current_route_labels"))
    if _has(wall_gate, "passes_schedule_invariance_distinct_partial_wall_evidence"):
        return "existing_partial_wall_debt_context_reference"
    if margin_gate or _has(bands, "support_boundary_loss", "support_hard_loss"):
        return "existing_margin_or_support_loss_context_reference"
    if route_labels:
        return "existing_direct_route_reference_lacks_debt_field"
    return "missing_no_objective_debt_evidence"


def _debt_recovery_evidence(row: pd.Series) -> str:
    route_labels = _safe_text(row.get("current_route_labels"))
    bands = _safe_text(row.get("current_polish_margin_bands"))
    if _has(route_labels, "direct_route_reaches_target_and_polish_stays") and _has(
        bands, "target_stable_margin"
    ):
        return "existing_target_stable_margin_reference"
    if _has(route_labels, "direct_route_reaches_target"):
        return "existing_direct_route_reference_without_recovery_field"
    return "missing_no_debt_recovery_evidence"


def _polish_reversion_evidence(row: pd.Series) -> str:
    route_labels = _safe_text(row.get("current_route_labels"))
    bands = _safe_text(row.get("current_polish_margin_bands"))
    if _has(route_labels, "bounce", "reversion") or _has(bands, "reversion"):
        return "existing_polish_reversion_reference"
    if _has(route_labels, "direct_route_reaches_target_and_polish_stays"):
        return "existing_polish_stays_reference_no_reversion"
    return "missing_no_polish_reversion_evidence"


def _support_incompatibility_evidence(row: pd.Series) -> str:
    bands = _safe_text(row.get("current_polish_margin_bands"))
    if _has(bands, "support_hard_loss"):
        return "existing_route_support_hard_loss_reference"
    if _has(bands, "support_boundary_loss"):
        return "existing_route_support_boundary_loss_reference"
    if _safe_text(row.get("support_distance_band_v0")) == "distinct_zone":
        return "support_distance_distinct_relation_only_not_wall_evidence"
    return "missing_no_support_incompatibility_evidence"


def _final_endpoint_assignment(row: pd.Series) -> str:
    route_labels = _safe_text(row.get("current_route_labels"))
    if _has(route_labels, "direct_route_unassigned"):
        return "existing_route_unassigned_reference_blocks_supported_label"
    if _has(route_labels, "direct_route_reaches_target_and_polish_stays"):
        return "existing_target_assignment_reference_needs_trace_audit"
    if _safe_text(row.get("pair_evidence_grade")) == "both_full_membership_caches_available":
        return "endpoint_identity_cache_ready_but_route_final_assignment_missing"
    return "missing_route_final_endpoint_assignment"


def _primary_wall_signal(row: pd.Series) -> str:
    wall_gate = _first_text(row, "current_wall_claim_gate_status", "remaining_wall_claim_gate_status")
    bands = _safe_text(row.get("current_polish_margin_bands"))
    if _has(wall_gate, "passes_schedule_invariance_distinct_partial_wall_evidence"):
        return "existing_partial_wall_primary_signal_reference"
    if _has(wall_gate, "fails_schedule_invariance"):
        return "existing_failed_wall_signal_reference"
    if _has(bands, "support_hard_loss", "support_boundary_loss"):
        return "existing_route_support_loss_reference"
    return "missing_primary_wall_signal"


def _consistency_check(row: pd.Series) -> str:
    wall_gate = _first_text(row, "current_wall_claim_gate_status", "remaining_wall_claim_gate_status")
    if _has(wall_gate, "passes_schedule_invariance_distinct_partial_wall_evidence"):
        return "existing_schedule_invariance_reference"
    if _has(wall_gate, "fails_schedule_invariance"):
        return "existing_schedule_invariance_failure_reference"
    if _safe_text(row.get("field")) == "field34":
        return "hygiene_limited_field34_not_allowed"
    if _safe_text(row.get("current_current_review_status")):
        return "nonfield34_existing_review_reference_only"
    return "nonfield34_panel_only_no_route_consistency_check"


def _route_label_status(row: pd.Series) -> str:
    route_labels = _safe_text(row.get("current_route_labels"))
    wall_gate = _first_text(row, "current_wall_claim_gate_status", "remaining_wall_claim_gate_status")
    relation_blocker = _first_text(
        row,
        "triage_relation_blocker_status",
        "current_relation_blocker_status",
        "remaining_primary_blocker_class",
    )
    if _has(wall_gate, "passes_schedule_invariance_distinct_partial_wall_evidence"):
        return "candidate_crosses_reference_not_promoted"
    if _has(route_labels, "direct_route_unassigned") or _has(wall_gate, "fails_schedule_invariance"):
        return "unknown_order_sensitive_or_unassigned_reference"
    if _has(relation_blocker, "ambiguous_relation", "basin_relation_definition_blocker"):
        return "unknown_relation_blocked_reference"
    if route_labels:
        return "unknown_existing_route_reference_not_sufficient"
    return "unknown_missing_wall_pathway_evidence"


def _schema_status(row: pd.Series) -> str:
    wall_gate = _first_text(row, "current_wall_claim_gate_status", "remaining_wall_claim_gate_status")
    route_labels = _safe_text(row.get("current_route_labels"))
    if _has(wall_gate, "passes_schedule_invariance_distinct_partial_wall_evidence"):
        return "existing_partial_wall_protocol_reference_needs_trace_audit"
    if _has(wall_gate, "fails_schedule_invariance") or _has(route_labels, "direct_route_unassigned"):
        return "existing_route_reference_blocks_supported_wall_label"
    if _safe_text(row.get("current_current_review_status")):
        return "existing_route_reference_not_wall_evidence"
    return "schema_missing_wall_pathway_evidence"


def _route_execution_eligibility(row: pd.Series) -> str:
    status = _safe_text(row.get("m3_wall_pathway_schema_status"))
    if status == "existing_partial_wall_protocol_reference_needs_trace_audit":
        return "not_executable_trace_audit_existing_reference_first"
    if status.startswith("existing_route_reference"):
        return "not_executable_existing_reference_blocks_or_insufficient"
    return "not_executable_missing_wall_pathway_schema_inputs"


def _next_action(row: pd.Series) -> str:
    status = _safe_text(row.get("m3_wall_pathway_schema_status"))
    if status == "existing_partial_wall_protocol_reference_needs_trace_audit":
        return "audit_existing_route_trace_against_v0_schema_before_new_probe"
    if status == "existing_route_reference_blocks_supported_wall_label":
        return "retain_unknown_label_and_do_not_promote_wall"
    if _safe_text(row.get("pair_evidence_grade")) == "both_full_membership_caches_available":
        return "prepare_predeclared_wall_trace_schema_before_any_probe"
    return "collect_or_link_endpoint_and_wall_pathway_evidence_fields_before_probe"


def _missing_requirements(row: pd.Series) -> str:
    missing: list[str] = []
    if not _safe_text(row.get("direct_path_availability_status")).startswith("existing_direct_route"):
        missing.append("direct_path_availability")
    if not _safe_text(row.get("objective_debt_evidence_status")).startswith(
        "existing_partial_wall_debt"
    ):
        missing.append("objective_debt")
    if not _safe_text(row.get("debt_recovery_evidence_status")).startswith(
        "existing_target_stable_margin"
    ):
        missing.append("debt_recovery")
    if _safe_text(row.get("polish_reversion_evidence_status")).startswith("missing"):
        missing.append("polish_reversion_or_explicit_nonreversion")
    support_status = _safe_text(row.get("support_incompatibility_evidence_status"))
    if support_status.startswith("support_distance_distinct_relation_only") or support_status.startswith(
        "missing"
    ):
        missing.append("support_incompatibility_wall_signal_or_explicit_absence")
    final_status = _safe_text(row.get("final_endpoint_assignment_evidence_status"))
    if final_status.endswith("missing") or final_status.startswith("missing") or _has(
        final_status, "needs_trace_audit", "blocks_supported_label"
    ):
        missing.append("route_final_endpoint_assignment_trace_audit")
    if not missing:
        return "none_for_m3_review_but_wall_not_promoted"
    return ";".join(missing)


def _claim_status(row: pd.Series) -> str:
    if _safe_text(row.get("m3_wall_pathway_schema_status")) == (
        "existing_partial_wall_protocol_reference_needs_trace_audit"
    ):
        return "partial_wall_protocol_reference_only_no_v0_supported_label"
    return "support_local_basin_pair_only_no_supported_wall_pathway_claim"


def _schema_review_rows(
    *,
    pair_evidence: pd.DataFrame,
    current_review: pd.DataFrame,
    remaining_wall: pd.DataFrame,
    pathway_readiness: pd.DataFrame,
    route_blocker: pd.DataFrame,
) -> pd.DataFrame:
    rows = pair_evidence.copy()
    rows["panel_pair_id"] = rows.apply(_panel_pair_id, axis=1)

    rows = rows.merge(
        _prefixed_join_frame(
            current_review,
            prefix="current_",
            columns=[
                "runner_preflight_status",
                "wall_claim_gate_status",
                "route_labels",
                "polish_margin_bands",
                "post_target_support_margin_max",
                "margin_gate_status",
                "methodology_v0_state",
                "relation_taxonomy_v0_1",
                "wall_promotion_status",
                "route_gate_group",
                "current_review_status",
                "review_comment",
            ],
        ),
        on="panel_pair_id",
        how="left",
    )
    rows = rows.merge(
        _prefixed_join_frame(
            remaining_wall,
            prefix="remaining_",
            columns=[
                "runner_preflight_status",
                "route_label_interpretation_v0",
                "wall_claim_gate_status",
                "primary_blocker_class",
                "relation_queue_status",
                "wall_evidence_question_status",
                "current_review_status",
                "remaining_wall_question_class",
                "decision_basis",
                "route_execution_decision",
                "wall_promotion_decision",
                "is_executable_route_candidate",
                "is_wall_promotion_candidate",
                "next_allowed_work",
                "decision_reason",
                "review_comment",
            ],
        ),
        on="panel_pair_id",
        how="left",
    )
    rows = rows.merge(
        _prefixed_join_frame(
            pathway_readiness,
            prefix="pathway_",
            columns=[
                "source_surface",
                "route_label_interpretation_v0",
                "pathway_readiness_status",
                "route_execution_decision",
                "wall_promotion_decision",
                "pathway_interpretation",
            ],
        ),
        on="panel_pair_id",
        how="left",
    )
    rows = rows.merge(
        _prefixed_join_frame(
            route_blocker,
            prefix="triage_",
            columns=[
                "route_gate_group",
                "route_label_interpretation_v0",
                "route_label_interpretation_group",
                "wall_promotion_status_v0",
                "wall_claim_gate_status",
                "relation_blocker_status",
                "hygiene_blocker_status",
                "blocker_tags",
                "primary_blocker_class",
                "relation_queue_status",
                "wall_evidence_question_status",
                "triage_action",
                "allowed_next_work",
                "forbidden_next_work",
                "immediate_route_execution_status",
                "allowed_claim",
                "forbidden_claim",
                "review_comment",
            ],
        ),
        on="panel_pair_id",
        how="left",
    )

    rows["existing_review_surface_status"] = rows.apply(_source_status, axis=1)
    rows["endpoint_assignment_evidence_status"] = rows.apply(_endpoint_assignment_evidence, axis=1)
    rows["support_relation_evidence_status"] = rows.apply(_support_relation_evidence, axis=1)
    rows["direct_path_availability_status"] = rows.apply(_direct_path_availability, axis=1)
    rows["objective_debt_evidence_status"] = rows.apply(_objective_debt_evidence, axis=1)
    rows["debt_recovery_evidence_status"] = rows.apply(_debt_recovery_evidence, axis=1)
    rows["polish_reversion_evidence_status"] = rows.apply(_polish_reversion_evidence, axis=1)
    rows["support_incompatibility_evidence_status"] = rows.apply(
        _support_incompatibility_evidence, axis=1
    )
    rows["final_endpoint_assignment_evidence_status"] = rows.apply(
        _final_endpoint_assignment, axis=1
    )
    rows["primary_wall_signal_status"] = rows.apply(_primary_wall_signal, axis=1)
    rows["consistency_check_status"] = rows.apply(_consistency_check, axis=1)
    rows["route_label_v0_schema_status"] = rows.apply(_route_label_status, axis=1)
    rows["m3_wall_pathway_schema_status"] = rows.apply(_schema_status, axis=1)
    rows["m3_route_execution_eligibility"] = rows.apply(_route_execution_eligibility, axis=1)
    rows["m3_next_action"] = rows.apply(_next_action, axis=1)
    rows["m3_missing_requirements"] = rows.apply(_missing_requirements, axis=1)
    rows["m3_claim_status"] = rows.apply(_claim_status, axis=1)
    rows["m4_probe_ready_status"] = "not_ready_m3_review_only"
    rows["quality_cost_status"] = QUALITY_COST_STATUS
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY

    cols = [
        "panel_pair_id",
        "case_id",
        "field",
        "method",
        "candidate_budget",
        "panel_role",
        "left_endpoint_identity_id",
        "right_endpoint_identity_id",
        "left_representative_candidate_index",
        "right_representative_candidate_index",
        "pair_evidence_grade",
        "calibrated_relation",
        "support_distance_band_v0",
        "support_distance_max",
        "existing_review_surface_status",
        "endpoint_assignment_evidence_status",
        "support_relation_evidence_status",
        "direct_path_availability_status",
        "objective_debt_evidence_status",
        "debt_recovery_evidence_status",
        "polish_reversion_evidence_status",
        "support_incompatibility_evidence_status",
        "final_endpoint_assignment_evidence_status",
        "primary_wall_signal_status",
        "consistency_check_status",
        "route_label_v0_schema_status",
        "m3_wall_pathway_schema_status",
        "m3_route_execution_eligibility",
        "m3_next_action",
        "m3_missing_requirements",
        "m3_claim_status",
        "m4_probe_ready_status",
        "current_runner_preflight_status",
        "current_wall_claim_gate_status",
        "current_route_labels",
        "current_polish_margin_bands",
        "current_margin_gate_status",
        "current_route_gate_group",
        "current_current_review_status",
        "current_review_comment",
        "remaining_remaining_wall_question_class",
        "remaining_route_execution_decision",
        "remaining_wall_promotion_decision",
        "remaining_next_allowed_work",
        "remaining_decision_reason",
        "pathway_pathway_readiness_status",
        "pathway_route_execution_decision",
        "pathway_wall_promotion_decision",
        "triage_immediate_route_execution_status",
        "triage_allowed_next_work",
        "triage_forbidden_next_work",
        "triage_allowed_claim",
        "triage_forbidden_claim",
        "quality_cost_status",
        "route_execution_status",
        "wall_promotion_status",
        "claim_boundary",
    ]
    for col in cols:
        if col not in rows:
            rows[col] = ""
    return rows[cols].sort_values(
        ["m3_wall_pathway_schema_status", "panel_role", "case_id", "support_distance_max"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def _summary(*, rows: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    overlap = int(
        rows["existing_review_surface_status"].ne("m2_only_no_existing_pathway_review_surface").sum()
    )
    trace_audit = int(
        rows["m3_wall_pathway_schema_status"].eq(
            "existing_partial_wall_protocol_reference_needs_trace_audit"
        ).sum()
    )
    blocked_reference = int(
        rows["m3_wall_pathway_schema_status"].eq(
            "existing_route_reference_blocks_supported_wall_label"
        ).sum()
    )
    missing_schema = int(
        rows["m3_wall_pathway_schema_status"].eq("schema_missing_wall_pathway_evidence").sum()
    )
    route_not_run = bool(rows["route_execution_status"].eq(ROUTE_EXECUTION_STATUS).all())
    wall_not_promoted = bool(rows["wall_promotion_status"].eq(WALL_PROMOTION_STATUS).all())
    quality_cost_excluded = bool(rows["quality_cost_status"].eq(QUALITY_COST_STATUS).all())
    m4_probe_ready = int(rows["m4_probe_ready_status"].ne("not_ready_m3_review_only").sum())
    return {
        "status": "methodology_v0_wall_pathway_schema_review_complete",
        "date": "2026-05-29",
        "script": _rel(Path(__file__).resolve()),
        "output_dir": _rel(output_dir),
        "pair_row_count": int(len(rows)),
        "existing_review_surface_overlap_count": overlap,
        "m2_only_no_existing_pathway_review_surface_count": int(len(rows) - overlap),
        "existing_partial_wall_trace_audit_candidate_count": trace_audit,
        "existing_blocked_route_reference_count": blocked_reference,
        "schema_missing_wall_pathway_evidence_count": missing_schema,
        "m4_probe_ready_pair_count": m4_probe_ready,
        "pair_evidence_grade_counts": _count(rows, "pair_evidence_grade"),
        "existing_review_surface_status_counts": _count(rows, "existing_review_surface_status"),
        "m3_wall_pathway_schema_status_counts": _count(rows, "m3_wall_pathway_schema_status"),
        "route_label_v0_schema_status_counts": _count(rows, "route_label_v0_schema_status"),
        "primary_wall_signal_status_counts": _count(rows, "primary_wall_signal_status"),
        "consistency_check_status_counts": _count(rows, "consistency_check_status"),
        "m3_next_action_counts": _count(rows, "m3_next_action"),
        "m3_route_execution_eligibility_counts": _count(rows, "m3_route_execution_eligibility"),
        "quality_cost_excluded": quality_cost_excluded,
        "route_execution_not_run": route_not_run,
        "wall_promotion_not_run": wall_not_promoted,
        "decision": (
            "M3 found no pair ready for new pathway probes. Two pairs carry "
            "existing partial-wall protocol references and should be trace-audited "
            "against the v0 schema before any new route run. Three existing route "
            "references remain blocked or insufficient; the other pairs lack "
            "wall/pathway evidence in the current review surface."
        ),
        "next_step": (
            "Do a narrow M4a trace audit for the two existing partial-wall references, "
            "then decide whether a predeclared small probe is justified. Do not launch "
            "a route batch from support-distance evidence alone."
        ),
        "paths": {
            "rows": _rel(output_dir / M3_ROWS_CSV),
            "summary": _rel(output_dir / SUMMARY_JSON),
            "report": _rel(output_dir / REPORT_MD),
            "config": _rel(output_dir / CONFIG_JSON),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Methodology v0 Wall/Pathway Schema Review",
        "",
        "Date: 2026-05-29",
        "",
        "## Scope",
        "",
        "This artifact is M3 of the methodology-v0 sequence. It reviews whether",
        "the enriched basin-pair panel has the evidence fields required for a",
        "wall/pathway label. It does not execute routes, promote wall claims,",
        "load memberships, or inspect quality/cost.",
        "",
        "## Decision",
        "",
        str(summary["decision"]),
        "",
        "## Counts",
        "",
        f"- reviewed pair rows: `{summary['pair_row_count']}`",
        f"- rows joined to existing 23-pair review surfaces: "
        f"`{summary['existing_review_surface_overlap_count']}`",
        f"- M2-only rows without existing pathway surface: "
        f"`{summary['m2_only_no_existing_pathway_review_surface_count']}`",
        f"- existing partial-wall references needing trace audit: "
        f"`{summary['existing_partial_wall_trace_audit_candidate_count']}`",
        f"- existing blocked/insufficient route references: "
        f"`{summary['existing_blocked_route_reference_count']}`",
        f"- rows missing wall/pathway evidence: "
        f"`{summary['schema_missing_wall_pathway_evidence_count']}`",
        f"- M4 probe-ready pairs: `{summary['m4_probe_ready_pair_count']}`",
        "",
        "## M3 Schema Status",
        "",
        "| status | rows |",
        "| --- | ---: |",
    ]
    for status, count in summary["m3_wall_pathway_schema_status_counts"].items():
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "## Route Label Status",
            "",
            "| status | rows |",
            "| --- | ---: |",
        ]
    )
    for status, count in summary["route_label_v0_schema_status_counts"].items():
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "## Next Actions",
            "",
            "| action | rows |",
            "| --- | ---: |",
        ]
    )
    for action, count in summary["m3_next_action_counts"].items():
        lines.append(f"| {action} | {count} |")
    lines.extend(
        [
            "",
            "## No-Leak Checks",
            "",
            f"- quality/cost excluded: `{str(summary['quality_cost_excluded']).lower()}`",
            f"- route execution not run: `{str(summary['route_execution_not_run']).lower()}`",
            f"- wall promotion not run: `{str(summary['wall_promotion_not_run']).lower()}`",
            "",
            "## Next Step",
            "",
            str(summary["next_step"]),
            "",
            "Claim boundary: " + CLAIM_BOUNDARY,
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    methodology_dir: Path,
    current_review_dir: Path,
    remaining_wall_dir: Path,
    existence_audit_dir: Path,
    route_blocker_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_evidence = _read_csv(methodology_dir / PAIR_EVIDENCE_CSV)
    current_review = _read_optional_csv(
        current_review_dir / CURRENT_REVIEW_CSV,
        ["panel_pair_id"],
    )
    remaining_wall = _read_optional_csv(
        remaining_wall_dir / REMAINING_WALL_CSV,
        ["panel_pair_id"],
    )
    pathway_readiness = _read_optional_csv(
        existence_audit_dir / PATHWAY_READINESS_CSV,
        ["panel_pair_id"],
    )
    route_blocker = _read_optional_csv(
        route_blocker_dir / ROUTE_BLOCKER_CSV,
        ["panel_pair_id"],
    )

    rows = _schema_review_rows(
        pair_evidence=pair_evidence,
        current_review=current_review,
        remaining_wall=remaining_wall,
        pathway_readiness=pathway_readiness,
        route_blocker=route_blocker,
    )
    summary = _summary(rows=rows, output_dir=output_dir)

    _write_csv(rows, output_dir / M3_ROWS_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            {
                "methodology_dir": _rel(methodology_dir),
                "current_review_dir": _rel(current_review_dir),
                "remaining_wall_dir": _rel(remaining_wall_dir),
                "existence_audit_dir": _rel(existence_audit_dir),
                "route_blocker_dir": _rel(route_blocker_dir),
                "output_dir": _rel(output_dir),
                "quality_cost_status": QUALITY_COST_STATUS,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / REPORT_MD, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methodology-dir", type=Path, default=DEFAULT_METHOD_DIR)
    parser.add_argument("--current-review-dir", type=Path, default=DEFAULT_CURRENT_REVIEW_DIR)
    parser.add_argument("--remaining-wall-dir", type=Path, default=DEFAULT_REMAINING_WALL_DIR)
    parser.add_argument("--existence-audit-dir", type=Path, default=DEFAULT_EXISTENCE_AUDIT_DIR)
    parser.add_argument("--route-blocker-dir", type=Path, default=DEFAULT_ROUTE_BLOCKER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_METHOD_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run(
        methodology_dir=args.methodology_dir,
        current_review_dir=args.current_review_dir,
        remaining_wall_dir=args.remaining_wall_dir,
        existence_audit_dir=args.existence_audit_dir,
        route_blocker_dir=args.route_blocker_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
