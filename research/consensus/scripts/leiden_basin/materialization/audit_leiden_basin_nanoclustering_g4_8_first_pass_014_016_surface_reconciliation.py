#!/usr/bin/env python3
"""Reconcile 014 object-wall evidence with 016 positive route morphology.

This read-only audit compares two evidence surfaces that currently diverge:
local_pair_014 has local object-level basin/wall evidence but is a current
fixed-predicate morphology guard, while local_pair_016 has the positive
stable-plateau route morphology but lacks endpoint-object identity and wall
evidence. The audit does not execute Leiden or promote pathway labels.
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


DEFAULT_ASSIGNMENT_SURFACE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_basin_state_assignment_surface_gamma1e5_20260606"
)
DEFAULT_MORPHOLOGY_TAXONOMY_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy_gamma1e5_20260606"
)
DEFAULT_PATHWAY_TRACE_014_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_014_pathway_probe_trace_gamma1e5_20260604"
)
DEFAULT_WALL_EVIDENCE_014_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_014_wall_evidence_audit_gamma1e5_20260604"
)
DEFAULT_ENDPOINT_OBJECT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_symmetric_endpoint_objects_audit_gamma1e5_20260604"
)
DEFAULT_CONTINUITY_016_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_continuity_block_audit_gamma1e5_20260605"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_014_016_surface_reconciliation_gamma1e5_20260607"
)

ASSIGNMENT_PAIR_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_basin_state_assignment_surface_pair_rows.csv"
)
ASSIGNMENT_SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_basin_state_assignment_surface_summary.json"
)
MORPHOLOGY_PAIR_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy_pair_rows.csv"
)
MORPHOLOGY_SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy_summary.json"
)
PATHWAY_PAIR_RESULT_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_pathway_probe_pair_result_rows.csv"
)
PATHWAY_ROUTE_SUMMARY_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_pathway_probe_route_summary_rows.csv"
)
PATHWAY_SUMMARY_JSON = "nanoclustering_g4_8_first_pass_014_pathway_probe_trace_summary.json"
WALL_PAIR_ROWS_CSV = "nanoclustering_g4_8_first_pass_014_wall_evidence_pair_rows.csv"
WALL_SUMMARY_JSON = "nanoclustering_g4_8_first_pass_014_wall_evidence_summary.json"
ENDPOINT_OBJECT_PAIR_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_symmetric_endpoint_object_pair_summary_rows.csv"
)
CONTINUITY_PAIR_COMPARISON_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_continuity_block_pair_comparison_rows.csv"
)
CONTINUITY_ROUTE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_continuity_block_route_rows.csv"
)
CONTINUITY_SUMMARY_JSON = "nanoclustering_g4_8_first_pass_016_continuity_block_summary.json"

PAIR_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_016_surface_reconciliation_pair_rows.csv"
)
AXIS_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_016_surface_reconciliation_axis_rows.csv"
)
SCHEDULE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_016_surface_reconciliation_schedule_rows.csv"
)
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_016_surface_reconciliation_decision_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_014_016_surface_reconciliation_gate_matrix.csv"
)
SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_014_016_surface_reconciliation_summary.json"
)
CONFIG_JSON = "nanoclustering_g4_8_first_pass_014_016_surface_reconciliation_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_014_016_surface_reconciliation_report.md"

PAIR_014 = "local_pair_014"
PAIR_016 = "local_pair_016"
BOUNDARY_PAIR = "local_pair_005"
FOCUS_PAIR_IDS = (PAIR_014, PAIR_016)
CONTEXT_PAIR_IDS = (PAIR_014, PAIR_016, BOUNDARY_PAIR)

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_014_016_surface_reconciliation"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_014_016_surface_reconciliation"
WALL_PROMOTION_STATUS = "not_promoted_surface_reconciliation_only"
METHOD_STATUS = "surface_reconciliation_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass 014/016 surface reconciliation audit only; "
    "reads existing assignment, morphology, 014 pathway/wall, endpoint-object, "
    "and 016 continuity artifacts. It does not execute Leiden, promote pathway "
    "labels, promote a general wall, replay full NanoClustering, evaluate "
    "quality/cost value, or claim method success."
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


def _as_float(value: Any, default: float | None = None) -> float | None:
    if pd.isna(value):
        return default
    return float(value)


def _safe_str(value: Any, default: str = "") -> str:
    if pd.isna(value):
        return default
    return str(value)


def _index_by_pair(frame: pd.DataFrame) -> dict[str, pd.Series]:
    if frame.empty or "local_pair_id" not in frame.columns:
        return {}
    return {str(row["local_pair_id"]): row for _, row in frame.iterrows()}


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
        "gate_status": "pass" if passed else "fail",
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 80) -> str:
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


def _load_context(
    *,
    assignment_surface_dir: Path,
    morphology_taxonomy_dir: Path,
    pathway_trace_014_dir: Path,
    wall_evidence_014_dir: Path,
    endpoint_object_dir: Path,
    continuity_016_dir: Path,
) -> dict[str, Any]:
    return {
        "paths": {
            "assignment_surface_dir": assignment_surface_dir,
            "morphology_taxonomy_dir": morphology_taxonomy_dir,
            "pathway_trace_014_dir": pathway_trace_014_dir,
            "wall_evidence_014_dir": wall_evidence_014_dir,
            "endpoint_object_dir": endpoint_object_dir,
            "continuity_016_dir": continuity_016_dir,
        },
        "assignment_pair_rows": _read_csv(assignment_surface_dir / ASSIGNMENT_PAIR_ROWS_CSV),
        "assignment_summary": _read_json(assignment_surface_dir / ASSIGNMENT_SUMMARY_JSON),
        "morphology_pair_rows": _read_csv(morphology_taxonomy_dir / MORPHOLOGY_PAIR_ROWS_CSV),
        "morphology_summary": _read_json(morphology_taxonomy_dir / MORPHOLOGY_SUMMARY_JSON),
        "pathway_pair_result_rows": _read_csv(
            pathway_trace_014_dir / PATHWAY_PAIR_RESULT_ROWS_CSV
        ),
        "pathway_route_summary_rows": _read_csv(
            pathway_trace_014_dir / PATHWAY_ROUTE_SUMMARY_ROWS_CSV
        ),
        "pathway_summary": _read_json(pathway_trace_014_dir / PATHWAY_SUMMARY_JSON),
        "wall_pair_rows": _read_csv(wall_evidence_014_dir / WALL_PAIR_ROWS_CSV),
        "wall_summary": _read_json(wall_evidence_014_dir / WALL_SUMMARY_JSON),
        "endpoint_object_pair_rows": _read_csv(
            endpoint_object_dir / ENDPOINT_OBJECT_PAIR_ROWS_CSV
        ),
        "continuity_pair_comparison_rows": _read_csv(
            continuity_016_dir / CONTINUITY_PAIR_COMPARISON_ROWS_CSV
        ),
        "continuity_route_rows": _read_csv(continuity_016_dir / CONTINUITY_ROUTE_ROWS_CSV),
        "continuity_summary": _read_json(continuity_016_dir / CONTINUITY_SUMMARY_JSON),
    }


def _surface_class(pair_id: str) -> str:
    if pair_id == PAIR_014:
        return "object_wall_positive_fixed_fraction_morphology_guard"
    if pair_id == PAIR_016:
        return "fixed_fraction_positive_object_wall_missing"
    return "context_or_boundary_control"


def _pair_interpretation(pair_id: str) -> str:
    if pair_id == PAIR_014:
        return (
            "014 has accepted local object-level basin-state and wall evidence, "
            "but the current fixed-fraction morphology treats it as a fragmented "
            "single-side guard. This is a route-family mismatch, not a pathway "
            "promotion."
        )
    if pair_id == PAIR_016:
        return (
            "016 is the positive stable-plateau morphology reference with complete "
            "source/target route anchors, but endpoint-object identity and wall "
            "evidence are missing."
        )
    return "Boundary/control context."


def _build_pair_rows(context: dict[str, Any]) -> pd.DataFrame:
    assignment_by_pair = _index_by_pair(context["assignment_pair_rows"])
    morphology_by_pair = _index_by_pair(context["morphology_pair_rows"])
    pathway_by_pair = _index_by_pair(context["pathway_pair_result_rows"])
    wall_by_pair = _index_by_pair(context["wall_pair_rows"])
    endpoint_by_pair = _index_by_pair(context["endpoint_object_pair_rows"])
    continuity_by_pair = _index_by_pair(context["continuity_pair_comparison_rows"])

    rows: list[dict[str, Any]] = []
    for pair_id in FOCUS_PAIR_IDS:
        assignment = assignment_by_pair[pair_id]
        morphology = morphology_by_pair[pair_id]
        pathway = pathway_by_pair.get(pair_id)
        wall = wall_by_pair.get(pair_id)
        endpoint = endpoint_by_pair.get(pair_id)
        continuity = continuity_by_pair.get(pair_id)
        rows.append(
            {
                "local_pair_id": pair_id,
                "surface_reconciliation_class": _surface_class(pair_id),
                "route_state_morphology_class": morphology["route_state_morphology_class"],
                "route_state_sequence": morphology["route_state_sequence"],
                "seed_start_stable_finite_plateau": _as_bool(
                    morphology["seed_start_stable_finite_plateau"]
                ),
                "all_route_single_side_fraction_count": _as_int(
                    morphology["all_route_single_side_fraction_count"]
                ),
                "any_single_side_fraction_count": _as_int(
                    morphology["any_single_side_fraction_count"]
                ),
                "accepted_local_object_basin_pair": _as_bool(
                    assignment["accepted_local_object_basin_pair"]
                ),
                "object_evidence_status": assignment["object_evidence_status"],
                "object_audit_class": assignment["object_audit_class"],
                "wall_evidence_status": assignment["wall_evidence_status"],
                "wall_evidence_ready_local_only": _as_bool(
                    assignment["wall_evidence_ready_local_only"]
                ),
                "pathway_probe_pair_status": _safe_str(
                    pathway.get("pair_probe_status") if pathway is not None else None,
                    "not_executed_on_this_surface",
                ),
                "direct_path_accepted_seed_route_count": _as_int(
                    pathway.get("direct_path_accepted_seed_route_count")
                    if pathway is not None
                    else None
                ),
                "recovery_accepted_seed_route_count": _as_int(
                    pathway.get("recovery_accepted_seed_route_count")
                    if pathway is not None
                    else None
                ),
                "boundary_positive_leak_seed_route_count": _as_int(
                    pathway.get("boundary_positive_leak_seed_route_count")
                    if pathway is not None
                    else None
                ),
                "wall_pair_status": _safe_str(
                    wall.get("primitive_wall_evidence_status") if wall is not None else None,
                    "not_tested_on_this_surface",
                ),
                "endpoint_object_audit_class": _safe_str(
                    endpoint.get("object_audit_class") if endpoint is not None else None,
                    "not_tested_on_this_surface",
                ),
                "continuity_pair_result": _safe_str(
                    continuity.get("pair_first_pass_result") if continuity is not None else None,
                    "not_compared_on_this_surface",
                ),
                "ready_like_seed_route_pass_count": _as_int(
                    continuity.get("ready_like_seed_route_pass_count")
                    if continuity is not None
                    else None
                ),
                "allowed_start_conditions": _safe_str(
                    continuity.get("allowed_start_conditions")
                    if continuity is not None
                    else None
                ),
                "blocked_start_conditions": _safe_str(
                    continuity.get("blocked_start_conditions")
                    if continuity is not None
                    else None
                ),
                "pathway_label_eligibility": "blocked_surface_mismatch",
                "recommended_next_action": _recommended_next_action(pair_id),
                "interpretation": _pair_interpretation(pair_id),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows)


def _recommended_next_action(pair_id: str) -> str:
    if pair_id == PAIR_014:
        return (
            "Use 014 as the local object-wall reference, but do not treat it as a "
            "fixed-fraction positive morphology."
        )
    if pair_id == PAIR_016:
        return (
            "Design a 016 object-wall transfer contract using the 014 direct-only "
            "and recovery-loop evidence vocabulary before executing any route."
        )
    return "Keep as context only."


def _build_axis_rows(pair_rows: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    by_pair = _index_by_pair(pair_rows)
    row014 = by_pair[PAIR_014]
    row016 = by_pair[PAIR_016]
    boundary_status = context["pathway_summary"].get("boundary_pair_probe_status")
    axes = [
        {
            "axis_id": "A1_fixed_fraction_route_morphology",
            "axis_question": "Which pair is positive under the fixed-fraction P1-P6 morphology surface?",
            "local_pair_014_readout": row014["route_state_morphology_class"],
            "local_pair_016_readout": row016["route_state_morphology_class"],
            "axis_reconciliation_status": "split_016_positive_014_guard",
            "interpretation": "016 is positive; 014 is a fragmented/point single-side guard.",
        },
        {
            "axis_id": "A2_endpoint_object_identity",
            "axis_question": "Which pair has accepted endpoint-object basin identity?",
            "local_pair_014_readout": row014["object_evidence_status"],
            "local_pair_016_readout": row016["object_evidence_status"],
            "axis_reconciliation_status": "split_014_accepted_016_missing",
            "interpretation": "014 has clean local object endpoint identity; 016 does not.",
        },
        {
            "axis_id": "A3_local_wall_evidence",
            "axis_question": "Which pair has local primitive wall evidence?",
            "local_pair_014_readout": row014["wall_evidence_status"],
            "local_pair_016_readout": row016["wall_evidence_status"],
            "axis_reconciliation_status": "split_014_ready_016_not_tested",
            "interpretation": "014 has local-only primitive wall evidence; 016 has not been tested on that surface.",
        },
        {
            "axis_id": "A4_route_family_schedule",
            "axis_question": "Are the two positive signals measured on the same route family?",
            "local_pair_014_readout": "direct-only/recovery-loop pathway probe",
            "local_pair_016_readout": "fixed-fraction plateau/continuity audit",
            "axis_reconciliation_status": "not_same_route_family",
            "interpretation": "The current positive signals are measured on different route families.",
        },
        {
            "axis_id": "A5_boundary_guard",
            "axis_question": "Does the 005 boundary guard stay closed?",
            "local_pair_014_readout": str(boundary_status),
            "local_pair_016_readout": "not_applicable",
            "axis_reconciliation_status": "boundary_guard_closed",
            "interpretation": "005 remains the false-positive guard for the 014 object-wall surface.",
        },
        {
            "axis_id": "A6_pathway_label",
            "axis_question": "Can any current pathway label be accepted?",
            "local_pair_014_readout": row014["pathway_label_eligibility"],
            "local_pair_016_readout": row016["pathway_label_eligibility"],
            "axis_reconciliation_status": "promotion_blocked",
            "interpretation": "No pathway label is promoted on the reconciled surface.",
        },
    ]
    for row in axes:
        row["claim_boundary"] = CLAIM_BOUNDARY
        row["run_status"] = RUN_STATUS
    return pd.DataFrame(axes)


def _build_schedule_rows(context: dict[str, Any]) -> pd.DataFrame:
    pathway_route_rows = context["pathway_route_summary_rows"].copy()
    rows: list[dict[str, Any]] = []
    for _, row in pathway_route_rows.iterrows():
        if str(row["local_pair_id"]) not in {PAIR_014, BOUNDARY_PAIR}:
            continue
        rows.append(
            {
                "schedule_surface": "014_pathway_probe_direct_recovery",
                "local_pair_id": row["local_pair_id"],
                "start_condition": row["start_condition"],
                "planned_route_family": row["planned_route_family"],
                "route_family_role": row["route_family_role"],
                "evidence_unit_count": _as_int(row["seed_count"]),
                "evidence_unit_count_source": "seed_count",
                "direct_path_accepted_seed_count": _as_int(
                    row["direct_path_accepted_seed_count"]
                ),
                "recovery_accepted_seed_count": _as_int(
                    row["recovery_accepted_seed_count"]
                ),
                "route_probe_status": row["route_probe_status"],
                "reconciliation_role": (
                    "object_wall_reference_schedule"
                    if str(row["local_pair_id"]) == PAIR_014
                    else "boundary_guard_schedule"
                ),
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    continuity_by_pair = _index_by_pair(context["continuity_pair_comparison_rows"])
    for pair_id in (PAIR_014, PAIR_016):
        row = continuity_by_pair[pair_id]
        rows.append(
            {
                "schedule_surface": "016_continuity_fixed_fraction_comparison",
                "local_pair_id": pair_id,
                "start_condition": row["allowed_start_conditions"],
                "planned_route_family": "fixed_fraction_plateau_continuity",
                "route_family_role": row["comparison_role"],
                "evidence_unit_count": _as_int(row["allowed_execution_unit_count"]),
                "evidence_unit_count_source": "allowed_execution_unit_count",
                "direct_path_accepted_seed_count": None,
                "recovery_accepted_seed_count": None,
                "route_probe_status": row["pair_first_pass_result"],
                "reconciliation_role": (
                    "positive_morphology_reference_schedule"
                    if pair_id == PAIR_016
                    else "clean_scaffold_reference_schedule"
                ),
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows)


def _build_decision_rows() -> pd.DataFrame:
    rows = [
        {
            "decision_id": "D1_no_contradiction_different_surfaces",
            "decision": (
                "014 and 016 do not contradict each other; their positive signals "
                "are measured on different route families and evidence axes."
            ),
            "evidence": "014 is object-wall positive under direct/recovery probes; 016 is fixed-fraction morphology positive.",
            "decision_status": "accepted_reconciliation",
        },
        {
            "decision_id": "D2_no_pathway_label_promotion",
            "decision": "No current pathway label can be accepted from this surface.",
            "evidence": "014 is a current morphology guard; 016 lacks object identity and wall evidence.",
            "decision_status": "accepted_blocker",
        },
        {
            "decision_id": "D3_next_gate_design_only_016_transfer_contract",
            "decision": (
                "The next executable design should be a 016 object-wall transfer "
                "contract using the 014 direct-only/recovery-loop vocabulary."
            ),
            "evidence": "Use 014 as the local object-wall reference and 005 as boundary guard.",
            "decision_status": "next_gate",
        },
        {
            "decision_id": "D4_no_candidate_expansion_or_policy_sweep",
            "decision": (
                "Do not broaden candidates, rerun threshold/policy sweeps, or test "
                "009/012/020 until the 014/016 surface mismatch is resolved."
            ),
            "evidence": "Failed-direction guardrails require a mechanism question before new sweeps.",
            "decision_status": "accepted_guardrail",
        },
    ]
    for row in rows:
        row["claim_boundary"] = CLAIM_BOUNDARY
        row["run_status"] = RUN_STATUS
    return pd.DataFrame(rows)


def _build_gate_matrix(
    *,
    context: dict[str, Any],
    pair_rows: pd.DataFrame,
    axis_rows: pd.DataFrame,
    schedule_rows: pd.DataFrame,
) -> pd.DataFrame:
    by_pair = _index_by_pair(pair_rows)
    row014 = by_pair[PAIR_014]
    row016 = by_pair[PAIR_016]
    gates = [
        _gate_row(
            "G1_sources_readable",
            "Were assignment, morphology, 014 pathway/wall, endpoint-object, and 016 continuity sources readable?",
            {
                "assignment_pair_rows": int(len(context["assignment_pair_rows"])),
                "morphology_pair_rows": int(len(context["morphology_pair_rows"])),
                "pathway_pair_rows": int(len(context["pathway_pair_result_rows"])),
                "wall_pair_rows": int(len(context["wall_pair_rows"])),
                "endpoint_object_pair_rows": int(len(context["endpoint_object_pair_rows"])),
                "continuity_pair_rows": int(len(context["continuity_pair_comparison_rows"])),
            },
            "all required source surfaces have rows",
            all(
                len(context[key]) > 0
                for key in (
                    "assignment_pair_rows",
                    "morphology_pair_rows",
                    "pathway_pair_result_rows",
                    "wall_pair_rows",
                    "endpoint_object_pair_rows",
                    "continuity_pair_comparison_rows",
                )
            ),
        ),
        _gate_row(
            "G2_focus_pairs_present",
            "Are 014 and 016 both materialized on the reconciliation surface?",
            list(pair_rows["local_pair_id"].astype(str)),
            "exactly 014 and 016 focus rows",
            tuple(pair_rows["local_pair_id"].astype(str)) == FOCUS_PAIR_IDS,
        ),
        _gate_row(
            "G3_014_object_wall_positive_morphology_guard",
            "Is 014 object-wall positive but fixed-fraction morphology-negative?",
            {
                "object": row014["object_evidence_status"],
                "wall": row014["wall_evidence_status"],
                "morphology": row014["route_state_morphology_class"],
                "direct": int(row014["direct_path_accepted_seed_route_count"]),
                "recovery": int(row014["recovery_accepted_seed_route_count"]),
            },
            "014 accepted object pair, local wall ready, direct/recovery 32/32, morphology guard",
            _as_bool(row014["accepted_local_object_basin_pair"])
            and _as_bool(row014["wall_evidence_ready_local_only"])
            and int(row014["direct_path_accepted_seed_route_count"]) == 32
            and int(row014["recovery_accepted_seed_route_count"]) == 32
            and str(row014["route_state_morphology_class"])
            == "fragmented_or_point_single_side_negative",
        ),
        _gate_row(
            "G4_016_positive_morphology_object_wall_missing",
            "Is 016 positive under route morphology but missing object/wall evidence?",
            {
                "object": row016["object_evidence_status"],
                "wall": row016["wall_evidence_status"],
                "morphology": row016["route_state_morphology_class"],
                "stable": bool(row016["seed_start_stable_finite_plateau"]),
            },
            "016 stable plateau positive; endpoint-object and wall evidence missing",
            str(row016["route_state_morphology_class"])
            == "stable_finite_single_side_plateau_reference"
            and _as_bool(row016["seed_start_stable_finite_plateau"])
            and str(row016["object_evidence_status"])
            == "missing_endpoint_object_identity_evidence"
            and str(row016["wall_evidence_status"])
            == "not_tested_no_object_endpoint_pair",
        ),
        _gate_row(
            "G5_route_family_mismatch_explicit",
            "Does the audit make schedule non-comparability explicit?",
            {
                "axis_statuses": list(axis_rows["axis_reconciliation_status"].astype(str)),
                "schedule_surfaces": sorted(set(schedule_rows["schedule_surface"].astype(str))),
            },
            "014 direct/recovery and 016 fixed-fraction surfaces are both recorded",
            "not_same_route_family"
            in set(axis_rows["axis_reconciliation_status"].astype(str))
            and {
                "014_pathway_probe_direct_recovery",
                "016_continuity_fixed_fraction_comparison",
            }.issubset(set(schedule_rows["schedule_surface"].astype(str))),
        ),
        _gate_row(
            "G6_005_boundary_guard_preserved",
            "Is 005 retained as a closed boundary guard on the 014 object-wall surface?",
            {
                "boundary_pair_probe_status": context["pathway_summary"].get(
                    "boundary_pair_probe_status"
                ),
                "boundary_closed_seed_count": context["wall_summary"].get(
                    "boundary_guard_closed_seed_count"
                ),
            },
            "005 boundary control is closed with 32 closed guard seeds",
            context["pathway_summary"].get("boundary_pair_probe_status")
            == "boundary_control_closed"
            and context["wall_summary"].get("boundary_guard_closed_seed_count") == 32,
        ),
        _gate_row(
            "G7_pathway_promotion_blocked",
            "Are pathway labels still blocked for both focus pairs?",
            dict(
                zip(
                    pair_rows["local_pair_id"].astype(str),
                    pair_rows["pathway_label_eligibility"].astype(str),
                    strict=False,
                )
            ),
            "both focus rows are blocked_surface_mismatch",
            set(pair_rows["pathway_label_eligibility"].astype(str))
            == {"blocked_surface_mismatch"},
        ),
        _gate_row(
            "G8_claim_boundaries_closed",
            "Are method, general wall, pathway-label, quality/cost, and full-replay claims closed?",
            CLAIM_BOUNDARY,
            "read-only reconciliation only",
            all(
                pair_rows[column].astype(str).eq(expected).all()
                for column, expected in {
                    "route_execution_status": ROUTE_EXECUTION_STATUS,
                    "wall_promotion_status": WALL_PROMOTION_STATUS,
                    "method_status": METHOD_STATUS,
                }.items()
            ),
        ),
    ]
    return pd.DataFrame(gates)


def _write_report(
    *,
    output_dir: Path,
    pair_rows: pd.DataFrame,
    axis_rows: pd.DataFrame,
    schedule_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    report = f"""# NanoClustering G4.8 First-Pass 014/016 Surface Reconciliation

## Status

- Reconciliation status: `{summary["reconciliation_status"]}`
- Focus pairs: `{", ".join(summary["focus_pair_ids"])}`
- Accepted pathway labels: `{", ".join(summary["accepted_pathway_label_ids"]) or "none"}`
- Recommended next gate: {summary["recommended_next_gate"]}

## Pair Rows

{_markdown_table(
    pair_rows,
    [
        "local_pair_id",
        "surface_reconciliation_class",
        "route_state_morphology_class",
        "object_evidence_status",
        "wall_evidence_status",
        "pathway_probe_pair_status",
        "pathway_label_eligibility",
        "recommended_next_action",
    ],
)}

## Axis Rows

{_markdown_table(
    axis_rows,
    [
        "axis_id",
        "axis_reconciliation_status",
        "local_pair_014_readout",
        "local_pair_016_readout",
        "interpretation",
    ],
)}

## Schedule Rows

{_markdown_table(
    schedule_rows,
    [
        "schedule_surface",
        "local_pair_id",
        "start_condition",
        "planned_route_family",
        "evidence_unit_count",
        "evidence_unit_count_source",
        "route_probe_status",
        "reconciliation_role",
    ],
)}

## Decisions

{_markdown_table(
    decision_rows,
    [
        "decision_id",
        "decision_status",
        "decision",
        "evidence",
    ],
)}

## Gates

{_markdown_table(
    gate_matrix,
    [
        "gate_id",
        "gate_status",
        "question",
        "minimum_or_rule",
    ],
)}

## Claim Boundary

{CLAIM_BOUNDARY}
"""
    (output_dir / REPORT_MD).write_text(report, encoding="utf-8")


def run(
    *,
    assignment_surface_dir: Path,
    morphology_taxonomy_dir: Path,
    pathway_trace_014_dir: Path,
    wall_evidence_014_dir: Path,
    endpoint_object_dir: Path,
    continuity_016_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    context = _load_context(
        assignment_surface_dir=assignment_surface_dir,
        morphology_taxonomy_dir=morphology_taxonomy_dir,
        pathway_trace_014_dir=pathway_trace_014_dir,
        wall_evidence_014_dir=wall_evidence_014_dir,
        endpoint_object_dir=endpoint_object_dir,
        continuity_016_dir=continuity_016_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows = _build_pair_rows(context)
    axis_rows = _build_axis_rows(pair_rows, context)
    schedule_rows = _build_schedule_rows(context)
    decision_rows = _build_decision_rows()
    gate_matrix = _build_gate_matrix(
        context=context,
        pair_rows=pair_rows,
        axis_rows=axis_rows,
        schedule_rows=schedule_rows,
    )
    failed_gates = list(
        gate_matrix.loc[gate_matrix["gate_status"].astype(str).eq("fail"), "gate_id"].astype(str)
    )
    recommended_next_gate = (
        "Design a 016 object-wall transfer contract using the 014 direct-only "
        "and recovery-loop vocabulary, with 005 retained as the boundary guard; "
        "do not execute routes, broaden candidates, or promote pathway labels "
        "until that contract is reviewed."
    )
    summary = {
        "reconciliation_status": "014_object_wall_positive_016_morphology_positive_surface_split",
        "output_dir": str(output_dir),
        "focus_pair_ids": list(FOCUS_PAIR_IDS),
        "context_pair_ids": list(CONTEXT_PAIR_IDS),
        "accepted_pathway_label_ids": [],
        "surface_split": {
            PAIR_014: "object_wall_positive_fixed_fraction_morphology_guard",
            PAIR_016: "fixed_fraction_positive_object_wall_missing",
        },
        "failed_gates": failed_gates,
        "recommended_next_gate": recommended_next_gate,
        "claim_boundary": CLAIM_BOUNDARY,
        "run_status": RUN_STATUS,
    }
    config = {
        "assignment_surface_dir": str(assignment_surface_dir),
        "morphology_taxonomy_dir": str(morphology_taxonomy_dir),
        "pathway_trace_014_dir": str(pathway_trace_014_dir),
        "wall_evidence_014_dir": str(wall_evidence_014_dir),
        "endpoint_object_dir": str(endpoint_object_dir),
        "continuity_016_dir": str(continuity_016_dir),
        "output_dir": str(output_dir),
        "focus_pair_ids": list(FOCUS_PAIR_IDS),
        "context_pair_ids": list(CONTEXT_PAIR_IDS),
        "route_execution_status": ROUTE_EXECUTION_STATUS,
        "wall_promotion_status": WALL_PROMOTION_STATUS,
        "method_status": METHOD_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "run_status": RUN_STATUS,
    }
    _write_csv(pair_rows, output_dir / PAIR_ROWS_CSV)
    _write_csv(axis_rows, output_dir / AXIS_ROWS_CSV)
    _write_csv(schedule_rows, output_dir / SCHEDULE_ROWS_CSV)
    _write_csv(decision_rows, output_dir / DECISION_ROWS_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        pair_rows=pair_rows,
        axis_rows=axis_rows,
        schedule_rows=schedule_rows,
        decision_rows=decision_rows,
        gate_matrix=gate_matrix,
        summary=summary,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconcile 014 object-wall evidence with 016 route morphology."
    )
    parser.add_argument(
        "--assignment-surface-dir",
        type=Path,
        default=DEFAULT_ASSIGNMENT_SURFACE_DIR,
    )
    parser.add_argument(
        "--morphology-taxonomy-dir",
        type=Path,
        default=DEFAULT_MORPHOLOGY_TAXONOMY_DIR,
    )
    parser.add_argument(
        "--pathway-trace-014-dir",
        type=Path,
        default=DEFAULT_PATHWAY_TRACE_014_DIR,
    )
    parser.add_argument(
        "--wall-evidence-014-dir",
        type=Path,
        default=DEFAULT_WALL_EVIDENCE_014_DIR,
    )
    parser.add_argument(
        "--endpoint-object-dir",
        type=Path,
        default=DEFAULT_ENDPOINT_OBJECT_DIR,
    )
    parser.add_argument(
        "--continuity-016-dir",
        type=Path,
        default=DEFAULT_CONTINUITY_016_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = run(
        assignment_surface_dir=args.assignment_surface_dir,
        morphology_taxonomy_dir=args.morphology_taxonomy_dir,
        pathway_trace_014_dir=args.pathway_trace_014_dir,
        wall_evidence_014_dir=args.wall_evidence_014_dir,
        endpoint_object_dir=args.endpoint_object_dir,
        continuity_016_dir=args.continuity_016_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
