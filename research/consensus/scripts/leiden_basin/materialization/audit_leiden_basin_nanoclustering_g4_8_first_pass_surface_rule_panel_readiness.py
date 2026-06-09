#!/usr/bin/env python3
"""Audit first-pass panel readiness under the surface-rule schema.

This read-only audit applies the adapter-backed object-surface rule to the
current 23-pair first-pass panel. It separates:

- scoreable core rows with enough route/basin-state evidence;
- diagnostic but not scoreable rows;
- screened rows that must not receive a surface claim yet.

The audit is a readiness surface, not a generalization claim. It does not rerun
Leiden, execute route/fraction traces, broaden candidates, promote pathway or
wall labels, evaluate quality/cost value, replay full NanoClustering, or claim
method success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from audit_leiden_basin_nanoclustering_g4_8_first_pass_object_surface_rule_decision import (
    DEFAULT_OUTPUT_DIR as DEFAULT_OBJECT_SURFACE_RULE_DECISION_DIR,
    GATE_MATRIX_CSV as OBJECT_SURFACE_RULE_GATE_MATRIX_CSV,
    SUMMARY_JSON as OBJECT_SURFACE_RULE_SUMMARY_JSON,
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
    validate_surface_claim_rows,
)


MECHANISM_SCREEN_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_mechanism_generalization_screen_gamma1e5_20260605"
)
PLATEAU_APPLICATION_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_plateau_stability_gate_application_gamma1e5_20260606"
)
ROUTE_MORPHOLOGY_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy_gamma1e5_20260606"
)
BASIN_ASSIGNMENT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_basin_state_assignment_surface_gamma1e5_20260606"
)

MECHANISM_PAIR_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_mechanism_generalization_pair_rows.csv"
)
MECHANISM_CLASS_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_mechanism_generalization_class_rows.csv"
)
MECHANISM_GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_mechanism_generalization_gate_matrix.csv"
)
MECHANISM_SUMMARY_JSON = "nanoclustering_g4_8_first_pass_mechanism_generalization_summary.json"
PLATEAU_PAIR_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_plateau_stability_gate_application_pair_rows.csv"
)
PLATEAU_CLASS_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_plateau_stability_gate_application_class_rows.csv"
)
PLATEAU_GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_plateau_stability_gate_application_gate_matrix.csv"
)
PLATEAU_SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_plateau_stability_gate_application_summary.json"
)
MORPHOLOGY_PAIR_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy_pair_rows.csv"
)
MORPHOLOGY_CLASS_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy_class_rows.csv"
)
MORPHOLOGY_GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy_gate_matrix.csv"
)
MORPHOLOGY_SUMMARY_JSON = "nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy_summary.json"
BASIN_PAIR_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_basin_state_assignment_surface_pair_rows.csv"
)
BASIN_CLASS_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_basin_state_assignment_surface_class_rows.csv"
)
BASIN_GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_basin_state_assignment_surface_gate_matrix.csv"
)
BASIN_SUMMARY_JSON = "nanoclustering_g4_8_first_pass_basin_state_assignment_surface_summary.json"

DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_panel_readiness_gamma1e5_20260609"
)

PAIR_SURFACE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_panel_readiness_pair_surface_rows.csv"
)
CORE_READINESS_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_panel_readiness_core_readiness_rows.csv"
)
NON_SCOREABLE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_panel_readiness_non_scoreable_rows.csv"
)
CLASS_ROWS_CSV = "nanoclustering_g4_8_first_pass_surface_rule_panel_readiness_class_rows.csv"
EVIDENCE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_panel_readiness_evidence_rows.csv"
)
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_panel_readiness_decision_rows.csv"
)
GATE_MATRIX_CSV = "nanoclustering_g4_8_first_pass_surface_rule_panel_readiness_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_surface_rule_panel_readiness_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_surface_rule_panel_readiness_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_surface_rule_panel_readiness_report.md"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_surface_rule_panel_readiness"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass surface-rule panel-readiness audit only; "
    "reads existing mechanism screen, plateau application, route morphology, "
    "basin-state assignment, and object-surface rule artifacts. It separates "
    "scoreable core rows from not-scoreable surface gaps. It does not rerun "
    "Leiden, execute route/fraction traces, broaden candidates, promote pathway "
    "or wall labels, evaluate quality/cost value, replay full NanoClustering, "
    "or claim method success."
)
SCHEMA_ADAPTER_PATH = Path(__file__).resolve().parent / "surface_claim_schema_adapter.py"
CORE_SCOREABLE_IDS = {
    "local_pair_016",
    "local_pair_014",
    "local_pair_009",
    "local_pair_012",
    "local_pair_020",
    "local_pair_005",
}
EXPECTED_PANEL_COUNT = 23
EXPECTED_MECHANISM_FAILED_GATES = ["G4_route_level_generality_not_yet_established"]


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
        "object_surface_summary": _read_json(
            args.object_surface_rule_decision_dir / OBJECT_SURFACE_RULE_SUMMARY_JSON
        ),
        "object_surface_gates": _read_csv(
            args.object_surface_rule_decision_dir / OBJECT_SURFACE_RULE_GATE_MATRIX_CSV
        ),
        "mechanism_summary": _read_json(args.mechanism_screen_dir / MECHANISM_SUMMARY_JSON),
        "mechanism_gates": _read_csv(args.mechanism_screen_dir / MECHANISM_GATE_MATRIX_CSV),
        "mechanism_pair_rows": _read_csv(
            args.mechanism_screen_dir / MECHANISM_PAIR_ROWS_CSV
        ),
        "mechanism_class_rows": _read_csv(
            args.mechanism_screen_dir / MECHANISM_CLASS_ROWS_CSV
        ),
        "plateau_summary": _read_json(args.plateau_application_dir / PLATEAU_SUMMARY_JSON),
        "plateau_gates": _read_csv(args.plateau_application_dir / PLATEAU_GATE_MATRIX_CSV),
        "plateau_pair_rows": _read_csv(
            args.plateau_application_dir / PLATEAU_PAIR_ROWS_CSV
        ),
        "plateau_class_rows": _read_csv(
            args.plateau_application_dir / PLATEAU_CLASS_ROWS_CSV
        ),
        "morphology_summary": _read_json(args.route_morphology_dir / MORPHOLOGY_SUMMARY_JSON),
        "morphology_gates": _read_csv(args.route_morphology_dir / MORPHOLOGY_GATE_MATRIX_CSV),
        "morphology_pair_rows": _read_csv(args.route_morphology_dir / MORPHOLOGY_PAIR_ROWS_CSV),
        "morphology_class_rows": _read_csv(
            args.route_morphology_dir / MORPHOLOGY_CLASS_ROWS_CSV
        ),
        "basin_summary": _read_json(args.basin_assignment_dir / BASIN_SUMMARY_JSON),
        "basin_gates": _read_csv(args.basin_assignment_dir / BASIN_GATE_MATRIX_CSV),
        "basin_pair_rows": _read_csv(args.basin_assignment_dir / BASIN_PAIR_ROWS_CSV),
        "basin_class_rows": _read_csv(args.basin_assignment_dir / BASIN_CLASS_ROWS_CSV),
    }


def _merge_panel_rows(context: dict[str, Any]) -> pd.DataFrame:
    mechanism = context["mechanism_pair_rows"].copy()
    plateau = context["plateau_pair_rows"].copy()
    morphology = context["morphology_pair_rows"].copy()
    basin = context["basin_pair_rows"].copy()

    keep_mechanism = [
        column
        for column in [
            "local_pair_id",
            "mechanism_generalization_class",
            "is_primary_typed_transient_pair",
            "is_reference_pair",
            "is_boundary_guard_pair",
            "is_control_pair",
            "direct_positive_weak_pair",
            "original_pair_coassigned_share",
        ]
        if column in mechanism.columns
    ]
    keep_plateau = [
        column
        for column in [
            "local_pair_id",
            "validation_stratum",
            "guard_family",
            "contract_feature_scoreable",
            "contract_application_class",
            "contract_accepts_p1_p6",
            "p1_status",
            "p2_status",
            "p3_status",
            "p4_status",
            "p5_status",
            "p6_status",
            "pair_explanation_class",
        ]
        if column in plateau.columns
    ]
    keep_morphology = [
        column
        for column in [
            "local_pair_id",
            "route_trace_pair_present",
            "route_negative_pair_present",
            "route_state_morphology_class",
            "route_state_morphology_role",
            "basin_interpretation",
            "recommended_next_action",
            "claim_status",
            "route_count",
            "full_fixed_016_route_predicate_count",
            "finite_single_side_band_count",
            "all_route_single_side_fraction_count",
            "single_side_latch_signature",
            "seed_start_stable_finite_plateau",
        ]
        if column in morphology.columns
    ]
    keep_basin = [
        column
        for column in [
            "local_pair_id",
            "basin_state_assignment_class",
            "source_basin_assignment_status",
            "target_basin_assignment_status",
            "accepted_source_basin_candidate",
            "accepted_target_basin_candidate",
            "accepted_local_object_basin_pair",
            "object_evidence_status",
            "object_audit_class",
            "wall_evidence_status",
            "wall_evidence_ready_local_only",
            "pathway_label_promotion_status",
            "accepted_pathway_label",
            "demo_readiness_status",
            "local_gate_status",
        ]
        if column in basin.columns
    ]

    rows = mechanism[keep_mechanism].merge(
        plateau[keep_plateau],
        on="local_pair_id",
        how="outer",
        validate="one_to_one",
    )
    rows = rows.merge(
        morphology[keep_morphology],
        on="local_pair_id",
        how="outer",
        validate="one_to_one",
    )
    rows = rows.merge(
        basin[keep_basin],
        on="local_pair_id",
        how="left",
        validate="one_to_one",
    )
    if "claim_status" in rows.columns:
        rows = rows.rename(columns={"claim_status": "source_claim_status"})
    return rows.sort_values("local_pair_id", kind="mergesort").reset_index(drop=True)


def _classify_pair(row: pd.Series) -> dict[str, str]:
    pair_id = str(row["local_pair_id"])
    morphology_class = str(row.get("route_state_morphology_class", ""))
    contract_class = str(row.get("contract_application_class", ""))
    mechanism_class = str(row.get("mechanism_generalization_class", ""))

    if pair_id == "local_pair_016":
        return {
            "scoreability_status": "scoreable_core",
            "surface_level": "signature_object",
            "object_status": "split",
            "relation_status": "ladder",
            "claim_status": "blocked",
            "surface_rule_class": "diagnostic_transition_band_surface_reference",
            "generalization_role": "single_positive_reference",
            "generalization_status": "single_reference_not_panel_generality",
            "promotion_status": "diagnostic_only_wall_pathway_blocked",
            "readiness_gap": "needs_independent_scoreable_recurrence_or_typed_ladder_contract",
            "readiness_decision": "retain_as_reference_only",
        }
    if pair_id == "local_pair_014":
        return {
            "scoreability_status": "scoreable_core",
            "surface_level": "endpoint_object",
            "object_status": "certified",
            "relation_status": "clean",
            "claim_status": "diagnostic_only",
            "surface_rule_class": "object_wall_diagnostic_morphology_mismatch_guard",
            "generalization_role": "different_surface_guard",
            "generalization_status": "different_surface_not_016_like",
            "promotion_status": "diagnostic_only_no_pathway_promotion",
            "readiness_gap": "object_wall_surface_and_016_morphology_surface_remain_separate",
            "readiness_decision": "retain_as_cross_surface_guard",
        }
    if pair_id in {"local_pair_009", "local_pair_012", "local_pair_020"}:
        guard_class = (
            "strict_analog_abrupt_switch_negative_guard"
            if "abrupt" in morphology_class
            else "strict_analog_fragmented_or_point_negative_guard"
        )
        return {
            "scoreability_status": "scoreable_core",
            "surface_level": "state",
            "object_status": "unknown",
            "relation_status": "unresolved",
            "claim_status": "blocked",
            "surface_rule_class": guard_class,
            "generalization_role": "scoreable_negative_guard",
            "generalization_status": "strict_analog_not_016_like",
            "promotion_status": "blocked_negative_guard",
            "readiness_gap": "route_morphology_negative_and_object_wall_evidence_missing",
            "readiness_decision": "retain_as_specificity_guard",
        }
    if pair_id == "local_pair_005":
        return {
            "scoreability_status": "scoreable_core",
            "surface_level": "endpoint_object",
            "object_status": "collapse",
            "relation_status": "collapse",
            "claim_status": "closed",
            "surface_rule_class": "boundary_collapse_guard",
            "generalization_role": "boundary_false_positive_guard",
            "generalization_status": "closed_boundary_guard",
            "promotion_status": "closed_no_promotion",
            "readiness_gap": "source_target_collapse_boundary",
            "readiness_decision": "retain_as_boundary_guard",
        }
    if contract_class == "non_strict_local_signature_diagnostic_not_scoreable":
        return {
            "scoreability_status": "diagnostic_not_scoreable",
            "surface_level": "state",
            "object_status": "unknown",
            "relation_status": "not_applicable",
            "claim_status": "diagnostic_only",
            "surface_rule_class": "non_strict_local_signature_diagnostic_not_scoreable",
            "generalization_role": "unrouted_diagnostic_gap",
            "generalization_status": "not_scoreable_surface_gap",
            "promotion_status": "closed_no_promotion",
            "readiness_gap": "non_strict_signature_without_route_fraction_readout",
            "readiness_decision": "record_gap_do_not_score",
        }
    if "not_scoreable_without_route_fraction_readout" in contract_class:
        return {
            "scoreability_status": "screened_not_scoreable",
            "surface_level": "state",
            "object_status": "not_applicable",
            "relation_status": "not_applicable",
            "claim_status": "closed",
            "surface_rule_class": "screened_panel_context_not_scoreable",
            "generalization_role": "screened_gap_or_control",
            "generalization_status": "not_scoreable_surface_gap",
            "promotion_status": "closed_no_promotion",
            "readiness_gap": "route_fraction_readout_absent",
            "readiness_decision": "record_gap_do_not_score",
        }
    if mechanism_class in {"nonanalog", "closed_control_rejected", "guard_or_control_nonanalog"}:
        return {
            "scoreability_status": "screened_not_scoreable",
            "surface_level": "state",
            "object_status": "not_applicable",
            "relation_status": "not_applicable",
            "claim_status": "closed",
            "surface_rule_class": "screened_panel_context_not_scoreable",
            "generalization_role": "screened_gap_or_control",
            "generalization_status": "not_scoreable_surface_gap",
            "promotion_status": "closed_no_promotion",
            "readiness_gap": "screen_rejected_or_nonanalog",
            "readiness_decision": "record_gap_do_not_score",
        }
    return {
        "scoreability_status": "screened_not_scoreable",
        "surface_level": "state",
        "object_status": "unknown",
        "relation_status": "unresolved",
        "claim_status": "blocked",
        "surface_rule_class": "unresolved_panel_context",
        "generalization_role": "unresolved_gap",
        "generalization_status": "not_scoreable_surface_gap",
        "promotion_status": "closed_no_promotion",
        "readiness_gap": "unresolved_panel_context",
        "readiness_decision": "record_gap_do_not_score",
    }


def _pair_surface_rows(context: dict[str, Any]) -> pd.DataFrame:
    rows = _merge_panel_rows(context)
    classifications = rows.apply(_classify_pair, axis=1).apply(pd.Series)
    rows = pd.concat([rows, classifications], axis=1)
    rows["schema_adapter_version"] = SCHEMA_ADAPTER_VERSION
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    output_columns = [
        "local_pair_id",
        "scoreability_status",
        "surface_level",
        "object_status",
        "relation_status",
        "claim_status",
        "surface_rule_class",
        "generalization_role",
        "generalization_status",
        "promotion_status",
        "readiness_gap",
        "readiness_decision",
        "mechanism_generalization_class",
        "contract_application_class",
        "route_state_morphology_class",
        "source_claim_status",
        "basin_state_assignment_class",
        "route_trace_pair_present",
        "route_negative_pair_present",
        "contract_feature_scoreable",
        "accepted_local_object_basin_pair",
        "wall_evidence_ready_local_only",
        "pathway_label_promotion_status",
        "schema_adapter_version",
        "run_status",
        "claim_boundary",
    ]
    return rows[[column for column in output_columns if column in rows.columns]]


def _class_rows(pair_rows: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        pair_rows.groupby(
            [
                "scoreability_status",
                "surface_rule_class",
                "generalization_status",
                "promotion_status",
            ],
            dropna=False,
        )
        .agg(
            pair_count=("local_pair_id", "size"),
            local_pair_ids=("local_pair_id", lambda values: ";".join(map(str, values))),
        )
        .reset_index()
    )
    grouped["run_status"] = RUN_STATUS
    grouped["claim_boundary"] = CLAIM_BOUNDARY
    return grouped


def _evidence_rows(context: dict[str, Any], pair_rows: pd.DataFrame) -> pd.DataFrame:
    mechanism_expected_gap = context["mechanism_summary"].get(
        "failed_gates"
    ) == EXPECTED_MECHANISM_FAILED_GATES
    rows = [
        {
            "evidence_id": "E1_object_surface_rule_available",
            "evidence_status": "ready",
            "observed": {
                "failed_gates": context["object_surface_summary"].get("failed_gates"),
                "object_surface_rule_decision": context["object_surface_summary"].get(
                    "object_surface_rule_decision"
                ),
                "gate_status_counts": _count_dict(
                    context["object_surface_gates"]["gate_status"]
                ),
            },
            "claim_effect": "provides adapter-backed object-surface rule",
        },
        {
            "evidence_id": "E2_mechanism_screen_gap_acknowledged",
            "evidence_status": "expected_gap_acknowledged"
            if mechanism_expected_gap
            else "unexpected_gap_state",
            "observed": {
                "failed_gates": context["mechanism_summary"].get("failed_gates"),
                "strict_nonboundary_route_gap_pairs": context["mechanism_summary"].get(
                    "strict_nonboundary_route_gap_pairs"
                ),
                "p1_route_predicate_accept_pairs": context["mechanism_summary"].get(
                    "p1_route_predicate_accept_pairs"
                ),
            },
            "claim_effect": "prevents old screen failure from becoming positive evidence",
        },
        {
            "evidence_id": "E3_latest_route_and_basin_surfaces_available",
            "evidence_status": "ready",
            "observed": {
                "plateau_failed_gates": context["plateau_summary"].get("failed_gates"),
                "morphology_failed_gates": context["morphology_summary"].get(
                    "failed_gates"
                ),
                "basin_failed_gates": context["basin_summary"].get("failed_gates"),
                "scoreability_counts": _count_dict(pair_rows["scoreability_status"]),
            },
            "claim_effect": "uses post-screen route morphology and basin-state surfaces",
        },
        {
            "evidence_id": "E4_panel_readiness_counts",
            "evidence_status": "readiness_materialized",
            "observed": {
                "pair_count": int(len(pair_rows)),
                "surface_rule_class_counts": _count_dict(pair_rows["surface_rule_class"]),
                "generalization_status_counts": _count_dict(
                    pair_rows["generalization_status"]
                ),
                "promotion_status_counts": _count_dict(pair_rows["promotion_status"]),
            },
            "claim_effect": "separates single reference, guards, and unscoreable gaps",
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
            "decision": "panel_readiness_not_generality",
            "rationale": (
                "The panel is ready for surface-rule accounting, but current evidence "
                "does not establish panel-level recurrence of the 016-like surface."
            ),
        },
        {
            "decision_id": "D2",
            "decision": "score_only_core_six",
            "rationale": (
                "Only 016, 014, 009, 012, 020, and 005 have enough current route or "
                "basin-state evidence to receive scoreable surface-rule roles."
            ),
        },
        {
            "decision_id": "D3",
            "decision": "retain_016_single_reference",
            "rationale": (
                "016 remains the only diagnostic transition-band surface reference; "
                "this is not yet generality."
            ),
        },
        {
            "decision_id": "D4",
            "decision": "retain_guards_and_gaps",
            "rationale": (
                "014, 009, 012, 020, and 005 remain specificity guards, while 001/007 "
                "and screened rows remain not-scoreable gaps."
            ),
        },
        {
            "decision_id": "D5",
            "decision": "no_claim_promotion",
            "rationale": (
                "The audit opens no wall, pathway, method, quality/cost, full-replay, "
                "or route-execution claim."
            ),
        },
    ]
    frame = pd.DataFrame(rows)
    frame["run_status"] = RUN_STATUS
    frame["claim_boundary"] = CLAIM_BOUNDARY
    return frame


def _gate_matrix(
    *,
    pair_rows: pd.DataFrame,
    core_rows: pd.DataFrame,
    non_scoreable_rows: pd.DataFrame,
    context: dict[str, Any],
    validation: dict[str, Any],
) -> pd.DataFrame:
    pair_ids = set(pair_rows["local_pair_id"].astype(str))
    core_ids = set(core_rows["local_pair_id"].astype(str))
    non_scoreable_ids = set(non_scoreable_rows["local_pair_id"].astype(str))
    source_artifacts_ready = (
        not context["object_surface_summary"].get("failed_gates")
        and not context["plateau_summary"].get("failed_gates")
        and not context["morphology_summary"].get("failed_gates")
        and not context["basin_summary"].get("failed_gates")
    )
    mechanism_gap_acknowledged = context["mechanism_summary"].get(
        "failed_gates"
    ) == EXPECTED_MECHANISM_FAILED_GATES
    panel_complete = len(pair_rows) == EXPECTED_PANEL_COUNT and len(pair_ids) == EXPECTED_PANEL_COUNT
    core_isolated = core_ids == CORE_SCOREABLE_IDS and len(core_rows) == len(CORE_SCOREABLE_IDS)
    reference_pairs = pair_rows.loc[
        pair_rows["surface_rule_class"].astype(str).eq(
            "diagnostic_transition_band_surface_reference"
        ),
        "local_pair_id",
    ].astype(str).tolist()
    single_reference_only = reference_pairs == ["local_pair_016"]
    guards = {
        pair_id: pair_rows.loc[
            pair_rows["local_pair_id"].astype(str).eq(pair_id), "generalization_status"
        ].astype(str).tolist()
        for pair_id in ["local_pair_014", "local_pair_009", "local_pair_012", "local_pair_020", "local_pair_005"]
    }
    guards_retained = (
        guards.get("local_pair_014") == ["different_surface_not_016_like"]
        and guards.get("local_pair_009") == ["strict_analog_not_016_like"]
        and guards.get("local_pair_012") == ["strict_analog_not_016_like"]
        and guards.get("local_pair_020") == ["strict_analog_not_016_like"]
        and guards.get("local_pair_005") == ["closed_boundary_guard"]
    )
    non_scoreable_not_promoted = (
        len(non_scoreable_rows) == EXPECTED_PANEL_COUNT - len(CORE_SCOREABLE_IDS)
        and non_scoreable_rows["generalization_status"].astype(str).eq(
            "not_scoreable_surface_gap"
        ).all()
        and non_scoreable_rows["promotion_status"].astype(str).eq("closed_no_promotion").all()
    )
    no_claim_promotion = (
        pair_rows["claim_status"].astype(str).ne("open").all()
        and pair_rows["promotion_status"].astype(str).str.contains("blocked|closed|diagnostic").all()
    )
    readiness_not_generality = (
        single_reference_only
        and int(pair_rows["generalization_status"].astype(str).eq("single_reference_not_panel_generality").sum())
        == 1
        and int(pair_rows["generalization_status"].astype(str).eq("not_scoreable_surface_gap").sum())
        == len(non_scoreable_rows)
    )

    rows = [
        _gate_row(
            "G1_source_artifacts_ready_with_expected_screen_gap",
            "Are post-screen source artifacts ready and is the old route-level gap acknowledged?",
            {
                "object_surface_failed_gates": context["object_surface_summary"].get(
                    "failed_gates"
                ),
                "plateau_failed_gates": context["plateau_summary"].get("failed_gates"),
                "morphology_failed_gates": context["morphology_summary"].get(
                    "failed_gates"
                ),
                "basin_failed_gates": context["basin_summary"].get("failed_gates"),
                "mechanism_screen_failed_gates": context["mechanism_summary"].get(
                    "failed_gates"
                ),
            },
            "post-screen artifacts pass and mechanism screen retains only the expected route-generality gap",
            source_artifacts_ready and mechanism_gap_acknowledged,
        ),
        _gate_row(
            "G2_adapter_validation_passes_for_panel",
            "Do all panel rows satisfy the shared surface-claim schema?",
            validation,
            "required columns are present and values are valid",
            validation["required_columns_present"] and validation["required_values_valid"],
        ),
        _gate_row(
            "G3_panel_complete",
            "Are all 23 first-pass pairs represented?",
            {
                "pair_count": int(len(pair_rows)),
                "unique_pair_count": int(len(pair_ids)),
                "expected_pair_count": EXPECTED_PANEL_COUNT,
            },
            "23 unique panel rows",
            panel_complete,
        ),
        _gate_row(
            "G4_scoreable_core_isolated",
            "Are scoreable core rows separated from not-scoreable gaps?",
            {
                "core_pair_ids": sorted(core_ids),
                "non_scoreable_pair_count": int(len(non_scoreable_rows)),
                "scoreability_status_counts": _count_dict(pair_rows["scoreability_status"]),
            },
            "only 016, 014, 009, 012, 020, and 005 are scoreable core rows",
            core_isolated,
        ),
        _gate_row(
            "G5_single_016_reference_only",
            "Is 016 the only diagnostic transition-band surface reference?",
            {
                "diagnostic_transition_band_reference_pairs": reference_pairs,
                "generalization_status_counts": _count_dict(
                    pair_rows["generalization_status"]
                ),
            },
            "016 is the only reference and panel generality is not claimed",
            single_reference_only,
        ),
        _gate_row(
            "G6_specificity_guards_retained",
            "Are 014, 009, 012, 020, and 005 retained as guards rather than positives?",
            guards,
            "014 is a different surface, 009/012/020 are strict analog negatives, 005 is closed",
            guards_retained,
        ),
        _gate_row(
            "G7_non_scoreable_rows_not_promoted",
            "Are unscoreable rows recorded as gaps rather than positives?",
            {
                "non_scoreable_pair_count": int(len(non_scoreable_rows)),
                "non_scoreable_pair_ids": sorted(non_scoreable_ids),
                "non_scoreable_status_counts": _count_dict(
                    non_scoreable_rows["scoreability_status"]
                ),
            },
            "17 rows are explicitly not-scoreable and closed",
            non_scoreable_not_promoted,
        ),
        _gate_row(
            "G8_no_claim_promotion",
            "Are wall, pathway, method, quality, replay, and route-execution claims closed?",
            {
                "claim_status_counts": _count_dict(pair_rows["claim_status"]),
                "promotion_status_counts": _count_dict(pair_rows["promotion_status"]),
            },
            "no row opens a claim beyond diagnostic/readiness wording",
            no_claim_promotion,
        ),
        _gate_row(
            "G9_readiness_not_generality",
            "Does the panel materialize readiness without claiming generality?",
            {
                "single_reference_only": single_reference_only,
                "not_scoreable_gap_count": int(
                    pair_rows["generalization_status"].astype(str).eq(
                        "not_scoreable_surface_gap"
                    ).sum()
                ),
                "panel_generality_claimed": False,
            },
            "readiness surface is materialized and panel-level generality remains closed",
            readiness_not_generality,
        ),
    ]
    return pd.DataFrame(rows)


def _report(
    *,
    summary: dict[str, Any],
    pair_rows: pd.DataFrame,
    core_rows: pd.DataFrame,
    non_scoreable_rows: pd.DataFrame,
    class_rows: pd.DataFrame,
    evidence_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# NanoClustering G4.8 First-Pass Surface Rule Panel Readiness",
            "",
            f"- status: `{summary['status']}`",
            f"- pair_row_count: {summary['pair_row_count']}",
            f"- scoreable_core_pair_count: {summary['scoreable_core_pair_count']}",
            f"- non_scoreable_pair_count: {summary['non_scoreable_pair_count']}",
            f"- diagnostic_transition_band_reference_pairs: {summary['diagnostic_transition_band_reference_pairs']}",
            f"- panel_generality_established: {summary['panel_generality_established']}",
            f"- required_columns_present: {summary['required_columns_present']}",
            f"- required_values_valid: {summary['required_values_valid']}",
            f"- gate_status_counts: {summary['gate_status_counts']}",
            f"- failed_gates: {summary['failed_gates']}",
            f"- interpretation: {summary['interpretation']}",
            f"- recommended_next_gate: {summary['recommended_next_gate']}",
            f"- claim_boundary: {summary['claim_boundary']}",
            "",
            "## Core Readiness Rows",
            "",
            _markdown_table(
                core_rows,
                [
                    "local_pair_id",
                    "surface_rule_class",
                    "generalization_status",
                    "promotion_status",
                    "readiness_gap",
                    "readiness_decision",
                ],
            ),
            "",
            "## Class Rows",
            "",
            _markdown_table(
                class_rows,
                [
                    "scoreability_status",
                    "surface_rule_class",
                    "generalization_status",
                    "promotion_status",
                    "pair_count",
                    "local_pair_ids",
                ],
            ),
            "",
            "## Non-Scoreable Rows",
            "",
            _markdown_table(
                non_scoreable_rows,
                [
                    "local_pair_id",
                    "scoreability_status",
                    "surface_rule_class",
                    "generalization_status",
                    "readiness_gap",
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
            "This audit materializes panel readiness only. It does not convert the",
            "single 016 reference into panel generality, and it does not promote",
            "not-scoreable rows into positive evidence.",
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    context = _load_context(args)
    pair_rows = _pair_surface_rows(context)
    validation = validate_surface_claim_rows(pair_rows)
    core_rows = pair_rows[pair_rows["scoreability_status"].astype(str).eq("scoreable_core")].copy()
    non_scoreable_rows = pair_rows[
        pair_rows["scoreability_status"].astype(str).ne("scoreable_core")
    ].copy()
    class_rows = _class_rows(pair_rows)
    evidence_rows = _evidence_rows(context, pair_rows)
    decision_rows = _decision_rows()
    gate_matrix = _gate_matrix(
        pair_rows=pair_rows,
        core_rows=core_rows,
        non_scoreable_rows=non_scoreable_rows,
        context=context,
        validation=validation,
    )
    failed_gates = gate_matrix.loc[
        gate_matrix["gate_status"].astype(str).eq("fail"), "gate_id"
    ].astype(str).tolist()
    reference_pairs = pair_rows.loc[
        pair_rows["surface_rule_class"].astype(str).eq(
            "diagnostic_transition_band_surface_reference"
        ),
        "local_pair_id",
    ].astype(str).tolist()
    summary = {
        "schema": "nanoclustering_g4_8_first_pass_surface_rule_panel_readiness_summary.v1",
        "status": RUN_STATUS,
        "schema_adapter": str(SCHEMA_ADAPTER_PATH.resolve()),
        "schema_adapter_version": SCHEMA_ADAPTER_VERSION,
        "object_surface_rule_decision_dir": str(
            args.object_surface_rule_decision_dir.resolve()
        ),
        "mechanism_screen_dir": str(args.mechanism_screen_dir.resolve()),
        "plateau_application_dir": str(args.plateau_application_dir.resolve()),
        "route_morphology_dir": str(args.route_morphology_dir.resolve()),
        "basin_assignment_dir": str(args.basin_assignment_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "pair_row_count": int(len(pair_rows)),
        "scoreable_core_pair_count": int(len(core_rows)),
        "non_scoreable_pair_count": int(len(non_scoreable_rows)),
        "class_row_count": int(len(class_rows)),
        "evidence_row_count": int(len(evidence_rows)),
        "decision_row_count": int(len(decision_rows)),
        "required_columns": REQUIRED_COLUMNS,
        "required_columns_present": validation["required_columns_present"],
        "required_values_valid": validation["required_values_valid"],
        "missing_required_columns": validation["missing_required_columns"],
        "invalid_values_by_column": validation["invalid_values_by_column"],
        "scoreability_status_counts": _count_dict(pair_rows["scoreability_status"]),
        "surface_rule_class_counts": _count_dict(pair_rows["surface_rule_class"]),
        "generalization_status_counts": _count_dict(pair_rows["generalization_status"]),
        "promotion_status_counts": _count_dict(pair_rows["promotion_status"]),
        "diagnostic_transition_band_reference_pairs": reference_pairs,
        "scoreable_core_pair_ids": core_rows["local_pair_id"].astype(str).tolist(),
        "non_scoreable_pair_ids": non_scoreable_rows["local_pair_id"].astype(str).tolist(),
        "panel_generality_established": False,
        "route_execution_opened": False,
        "method_claim_ready": False,
        "quality_claim_ready": False,
        "new_wall_claim_ready_pairs": [],
        "gate_status_counts": _count_dict(gate_matrix["gate_status"]),
        "failed_gates": failed_gates,
        "interpretation": (
            "The 23-pair panel is now organized as a surface-rule readiness "
            "surface, not a generalization result. Only 016 is a diagnostic "
            "transition-band reference. The scoreable guards 014, 009, 012, 020, "
            "and 005 remain non-positive or cross-surface guards, while 17 rows "
            "remain not-scoreable gaps."
        ),
        "recommended_next_gate": (
            "Do not claim panel generality. Either fill the route/fraction or "
            "object-surface evidence gaps for a predeclared subset, or use the "
            "six-row scoreable core as the fixed guard set for any typed-ladder "
            "or endpoint-object membership contract."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(pair_rows, args.output_dir / PAIR_SURFACE_ROWS_CSV)
    _write_csv(core_rows, args.output_dir / CORE_READINESS_ROWS_CSV)
    _write_csv(non_scoreable_rows, args.output_dir / NON_SCOREABLE_ROWS_CSV)
    _write_csv(class_rows, args.output_dir / CLASS_ROWS_CSV)
    _write_csv(evidence_rows, args.output_dir / EVIDENCE_ROWS_CSV)
    _write_csv(decision_rows, args.output_dir / DECISION_ROWS_CSV)
    _write_csv(gate_matrix, args.output_dir / GATE_MATRIX_CSV)
    (args.output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_surface_rule_panel_readiness_config.v1",
        "schema_adapter": str(SCHEMA_ADAPTER_PATH.resolve()),
        "schema_adapter_version": SCHEMA_ADAPTER_VERSION,
        "object_surface_rule_decision_dir": str(
            args.object_surface_rule_decision_dir.resolve()
        ),
        "mechanism_screen_dir": str(args.mechanism_screen_dir.resolve()),
        "plateau_application_dir": str(args.plateau_application_dir.resolve()),
        "route_morphology_dir": str(args.route_morphology_dir.resolve()),
        "basin_assignment_dir": str(args.basin_assignment_dir.resolve()),
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
            pair_rows=pair_rows,
            core_rows=core_rows,
            non_scoreable_rows=non_scoreable_rows,
            class_rows=class_rows,
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
        "--object-surface-rule-decision-dir",
        type=Path,
        default=DEFAULT_OBJECT_SURFACE_RULE_DECISION_DIR,
        help="Directory containing the object-surface rule-decision audit.",
    )
    parser.add_argument(
        "--mechanism-screen-dir",
        type=Path,
        default=MECHANISM_SCREEN_DIR,
        help="Directory containing the mechanism-generalization screen.",
    )
    parser.add_argument(
        "--plateau-application-dir",
        type=Path,
        default=PLATEAU_APPLICATION_DIR,
        help="Directory containing the plateau-stability gate application.",
    )
    parser.add_argument(
        "--route-morphology-dir",
        type=Path,
        default=ROUTE_MORPHOLOGY_DIR,
        help="Directory containing the route-state morphology taxonomy.",
    )
    parser.add_argument(
        "--basin-assignment-dir",
        type=Path,
        default=BASIN_ASSIGNMENT_DIR,
        help="Directory containing the basin-state assignment surface.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for this surface-rule panel-readiness audit.",
    )
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
