#!/usr/bin/env python3
"""Design the first-pass transition-type panel contract.

This design reads the 016 transition-band evidence, the 001/007 low-fraction
boundary audit, and the current surface-rule panel. It freezes the next
direction as transition-type discrimination, not more gap expansion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from audit_leiden_basin_nanoclustering_g4_8_first_pass_016_pathway_shape import (
    DEFAULT_OUTPUT_DIR as DEFAULT_016_PATHWAY_DIR,
)
from audit_leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace import (
    DEFAULT_OUTPUT_DIR as DEFAULT_LOW_FRACTION_AUDIT_DIR,
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


DEFAULT_SURFACE_PANEL_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_panel_readiness_gamma1e5_20260609"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_transition_type_panel_contract_gamma1e5_20260609"
)

PATHWAY_SUMMARY_JSON = "nanoclustering_g4_8_first_pass_016_pathway_shape_summary.json"
PATHWAY_FRACTION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_pathway_shape_fraction_rows.csv"
)
LOW_FRACTION_AUDIT_SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_audit_summary.json"
)
LOW_FRACTION_AUDIT_PAIR_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_audit_pair_surface_rows.csv"
)
SURFACE_PANEL_SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_surface_rule_panel_readiness_summary.json"
)

CASE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_transition_type_panel_contract_case_rows.csv"
)
DISCRIMINANT_RULE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_transition_type_panel_contract_discriminant_rule_rows.csv"
)
NEXT_GATE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_transition_type_panel_contract_next_gate_rows.csv"
)
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_transition_type_panel_contract_decision_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_transition_type_panel_contract_gate_matrix.csv"
)
SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_transition_type_panel_contract_summary.json"
)
CONFIG_JSON = "nanoclustering_g4_8_first_pass_transition_type_panel_contract_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_transition_type_panel_contract_report.md"

RUN_STATUS = "designed_nanoclustering_g4_8_first_pass_transition_type_panel_contract"
ROUTE_EXECUTION_STATUS = "design_only_no_new_route_execution"
WALL_PROMOTION_STATUS = "not_promoted_transition_type_panel_contract_only"
METHOD_STATUS = "transition_type_panel_contract_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass transition-type panel contract only; reads "
    "existing 016 transition-band, 001/007 late-collapse, and surface-panel "
    "artifacts. It freezes the next direction as transition-type discrimination "
    "and does not execute routes, reopen screened gaps, promote walls/pathways, "
    "evaluate quality/cost, replay full NanoClustering, or claim method success."
)

REFERENCE_PAIR_ID = "local_pair_016"
LATE_COLLAPSE_GUARD_IDS = ("local_pair_001", "local_pair_007")
STRICT_ANALOG_GUARD_IDS = ("local_pair_009", "local_pair_012", "local_pair_020")
CROSS_SURFACE_GUARD_ID = "local_pair_014"
BOUNDARY_GUARD_ID = "local_pair_005"


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


def _fraction_set(rows: pd.DataFrame, column: str) -> str:
    values = sorted(rows.loc[rows[column].astype(int).gt(0), "bridge_edge_weight_fraction"])
    return ";".join(f"{float(value):.5g}" for value in values)


def _case_rows(
    *,
    pathway_fraction_rows: pd.DataFrame,
    low_fraction_pair_rows: pd.DataFrame,
) -> pd.DataFrame:
    source_fractions = _fraction_set(pathway_fraction_rows, "forward_source_family_count")
    transient_fractions = _fraction_set(
        pathway_fraction_rows, "forward_transient_signature_count"
    )
    target_fractions = _fraction_set(pathway_fraction_rows, "forward_target_anchor_count")
    cases: list[dict[str, Any]] = [
        {
            "local_pair_id": REFERENCE_PAIR_ID,
            "transition_type_role": "positive_reference",
            "transition_type_class": "finite_recurrent_transition_band_reference",
            "primary_evidence": (
                "source-family upper fractions, six adjacent transient fractions, "
                "and target anchor at 0.5"
            ),
            "source_fraction_readout": source_fractions,
            "transition_band_fraction_readout": transient_fractions,
            "target_fraction_readout": target_fractions,
            "single_side_or_typed_transient_present": True,
            "target_anchor_present": True,
            "claim_status_after_contract": "diagnostic_only",
            "next_use": "reference_for_transition_type_discrimination",
        }
    ]
    low_lookup = low_fraction_pair_rows.set_index("local_pair_id").to_dict("index")
    for pair_id in LATE_COLLAPSE_GUARD_IDS:
        row = low_lookup[pair_id]
        cases.append(
            {
                "local_pair_id": pair_id,
                "transition_type_role": "late_collapse_guard",
                "transition_type_class": "late_target_collapse_without_transition_band",
                "primary_evidence": (
                    "target-like collapse below 0.5 with zero single-side band and "
                    "zero diagnostic recurrence"
                ),
                "source_fraction_readout": "0.5",
                "transition_band_fraction_readout": "",
                "target_fraction_readout": "low_fraction_target_like",
                "single_side_or_typed_transient_present": False,
                "target_anchor_present": bool(
                    int(row["low_fraction_target_like_fraction_total"]) > 0
                ),
                "claim_status_after_contract": "blocked",
                "next_use": "negative_control_for_target_anchor_not_sufficient",
            }
        )
    for pair_id in STRICT_ANALOG_GUARD_IDS:
        cases.append(
            {
                "local_pair_id": pair_id,
                "transition_type_role": "strict_analog_guard",
                "transition_type_class": "strict_analog_not_transition_band",
                "primary_evidence": "prior strict analog route morphology is abrupt, fragmented, or point-only",
                "source_fraction_readout": "prior_surface_panel",
                "transition_band_fraction_readout": "absent_or_nonfinite",
                "target_fraction_readout": "prior_surface_panel",
                "single_side_or_typed_transient_present": False,
                "target_anchor_present": "mixed",
                "claim_status_after_contract": "blocked",
                "next_use": "specificity_guard_for_transition_band_rule",
            }
        )
    cases.extend(
        [
            {
                "local_pair_id": CROSS_SURFACE_GUARD_ID,
                "transition_type_role": "cross_surface_guard",
                "transition_type_class": "object_wall_surface_not_016_transition_band",
                "primary_evidence": "existing object-wall diagnostic morphology differs from 016",
                "source_fraction_readout": "different_surface",
                "transition_band_fraction_readout": "not_016_like",
                "target_fraction_readout": "different_surface",
                "single_side_or_typed_transient_present": "not_applicable",
                "target_anchor_present": "not_applicable",
                "claim_status_after_contract": "diagnostic_only",
                "next_use": "keep object-wall evidence separate from transition-band reference",
            },
            {
                "local_pair_id": BOUNDARY_GUARD_ID,
                "transition_type_role": "boundary_guard",
                "transition_type_class": "boundary_collapse_not_transition_band",
                "primary_evidence": "closed collapse guard",
                "source_fraction_readout": "collapse_surface",
                "transition_band_fraction_readout": "",
                "target_fraction_readout": "collapse_surface",
                "single_side_or_typed_transient_present": False,
                "target_anchor_present": True,
                "claim_status_after_contract": "closed",
                "next_use": "boundary_false_positive_guard",
            },
        ]
    )
    frame = pd.DataFrame(cases)
    frame["route_execution_status"] = ROUTE_EXECUTION_STATUS
    frame["wall_promotion_status"] = WALL_PROMOTION_STATUS
    frame["method_status"] = METHOD_STATUS
    frame["run_status"] = RUN_STATUS
    frame["claim_boundary"] = CLAIM_BOUNDARY
    return frame


def _discriminant_rule_rows() -> pd.DataFrame:
    rows = pd.DataFrame(
        [
            {
                "rule_id": "TT1",
                "rule_group": "target_not_sufficient",
                "rule": "target-like endpoint or collapse is not sufficient for 016-like basin-transition evidence",
                "positive_requirement": "target anchor must be paired with a recurrent transition band",
                "negative_control": "001/007 late target collapse; 005 boundary collapse",
            },
            {
                "rule_id": "TT2",
                "rule_group": "finite_transition_band",
                "rule": "016-like reference requires a finite adjacent intermediate band",
                "positive_requirement": "same typed transient/single-side state across adjacent fractions and seeds",
                "negative_control": "009/012/020 abrupt, fragmented, or point-only strict analog guards",
            },
            {
                "rule_id": "TT3",
                "rule_group": "bidirectional_readout",
                "rule": "transition-band evidence must be read under same source-family vocabulary in both directions when available",
                "positive_requirement": "source, transient, and target fractions align in forward and reverse readouts",
                "negative_control": "reverse anchor mismatch is a caveat, not wall evidence",
            },
            {
                "rule_id": "TT4",
                "rule_group": "surface_separation",
                "rule": "route morphology and object-surface promotion are separate gates",
                "positive_requirement": "signature-object diagnostic surface can support state vocabulary only",
                "negative_control": "endpoint-object wall wording stays blocked until a separate object rule passes",
            },
            {
                "rule_id": "TT5",
                "rule_group": "claim_boundary",
                "rule": "the transition-type panel is a direction contract, not a method result",
                "positive_requirement": "no new route execution, wall/pathway promotion, quality/cost, or panel-generality claim",
                "negative_control": "screened 15 gaps remain closed",
            },
        ]
    )
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _next_gate_rows() -> pd.DataFrame:
    rows = pd.DataFrame(
        [
            {
                "next_gate_id": "NG1",
                "next_gate": "freeze_transition_type_panel",
                "priority": 1,
                "status": "recommended_now",
                "rationale": (
                    "The new low-fraction result shows target collapse is not enough; "
                    "the panel should separate transition-band, late-collapse, strict "
                    "analog, cross-surface, and boundary controls."
                ),
                "execution_type": "read_only_design",
            },
            {
                "next_gate_id": "NG2",
                "next_gate": "typed_ladder_relation_rule_contract",
                "priority": 2,
                "status": "next_if_promoting_016_relation",
                "rationale": (
                    "If we want stronger 016 wording, predeclare a typed transient/"
                    "ladder relation rule using 001/007/009/012/020/005 as controls."
                ),
                "execution_type": "contract_before_any_route_expansion",
            },
            {
                "next_gate_id": "NG3",
                "next_gate": "endpoint_object_membership_audit",
                "priority": 3,
                "status": "alternative_if_promoting_object_wall",
                "rationale": (
                    "If wall/object wording is the target, endpoint-object membership "
                    "must be resolved separately from route morphology."
                ),
                "execution_type": "read_only_or_predeclared_membership_audit",
            },
            {
                "next_gate_id": "NG4",
                "next_gate": "screened_gap_expansion",
                "priority": 4,
                "status": "blocked_for_now",
                "rationale": (
                    "The current mechanism question is transition type, not finding "
                    "more weak examples among the 15 screened gaps."
                ),
                "execution_type": "do_not_execute",
            },
        ]
    )
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _decision_rows() -> pd.DataFrame:
    rows = pd.DataFrame(
        [
            {
                "decision_id": "D1",
                "decision": "direction_is_correct",
                "rationale": (
                    "The low-fraction audit exposed a hidden schedule-boundary caveat, "
                    "so the result-review-first direction was justified."
                ),
            },
            {
                "decision_id": "D2",
                "decision": "stop_low_fraction_expansion",
                "rationale": (
                    "001/007 are now explained as late target-collapse guards; more "
                    "low-fraction checks would not answer the 016 mechanism question."
                ),
            },
            {
                "decision_id": "D3",
                "decision": "use_transition_type_not_quality",
                "rationale": (
                    "The discriminant is state-sequence morphology, not Q value or "
                    "target endpoint alone."
                ),
            },
            {
                "decision_id": "D4",
                "decision": "keep_016_diagnostic",
                "rationale": (
                    "016 remains the only transition-band reference, but object-wall "
                    "and pathway claims stay blocked."
                ),
            },
        ]
    )
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _gate_matrix(
    *,
    pathway_summary: dict[str, Any],
    pathway_fraction_rows: pd.DataFrame,
    low_fraction_summary: dict[str, Any],
    surface_panel_summary: dict[str, Any],
    case_rows: pd.DataFrame,
    discriminant_rule_rows: pd.DataFrame,
    next_gate_rows: pd.DataFrame,
) -> pd.DataFrame:
    transient_rows = pathway_fraction_rows[
        pathway_fraction_rows["expected_pathway_state_class"].astype(str).eq(
            "transient_signature"
        )
    ]
    return pd.DataFrame(
        [
            _gate_row(
                "G1_upstream_artifacts_pass",
                "Did the upstream evidence pass without failed gates?",
                {
                    "pathway_failed_gates": pathway_summary.get("failed_gates"),
                    "low_fraction_failed_gates": low_fraction_summary.get("failed_gates"),
                    "surface_panel_failed_gates": surface_panel_summary.get("failed_gates"),
                },
                "all upstream failed_gates lists are empty",
                pathway_summary.get("failed_gates") == []
                and low_fraction_summary.get("failed_gates") == []
                and surface_panel_summary.get("failed_gates") == [],
            ),
            _gate_row(
                "G2_016_transition_band_reference_present",
                "Does 016 have the recurrent transition-band readout?",
                {
                    "transient_fraction_count": int(len(transient_rows)),
                    "all_forward_24": bool(
                        transient_rows["forward_transient_signature_count"]
                        .astype(int)
                        .eq(24)
                        .all()
                    ),
                    "all_reverse_24": bool(
                        transient_rows["reverse_transient_signature_count"]
                        .astype(int)
                        .eq(24)
                        .all()
                    ),
                },
                "six transient fractions, all 24/24 forward and reverse",
                int(len(transient_rows)) == 6
                and bool(
                    transient_rows["forward_transient_signature_count"]
                    .astype(int)
                    .eq(24)
                    .all()
                )
                and bool(
                    transient_rows["reverse_transient_signature_count"]
                    .astype(int)
                    .eq(24)
                    .all()
                ),
            ),
            _gate_row(
                "G3_001_007_late_collapse_controls_present",
                "Do 001/007 serve as target-collapse-not-band controls?",
                {
                    "late_target_collapse_pair_ids": low_fraction_summary.get(
                        "late_target_collapse_pair_ids"
                    ),
                    "single_side_signal_pair_ids": low_fraction_summary.get(
                        "single_side_signal_pair_ids"
                    ),
                },
                "001/007 are late target-collapse guards with no single-side signal",
                set(low_fraction_summary.get("late_target_collapse_pair_ids", []))
                == set(LATE_COLLAPSE_GUARD_IDS)
                and low_fraction_summary.get("single_side_signal_pair_ids") == [],
            ),
            _gate_row(
                "G4_eight_row_scoreable_panel_preserved",
                "Is the scoreable panel preserved as eight rows with 15 gaps closed?",
                {
                    "scoreable_pair_count": low_fraction_summary.get("scoreable_pair_count"),
                    "not_scoreable_pair_count": low_fraction_summary.get(
                        "not_scoreable_pair_count"
                    ),
                },
                "8 scoreable rows and 15 not-scoreable rows",
                int(low_fraction_summary.get("scoreable_pair_count", 0)) == 8
                and int(low_fraction_summary.get("not_scoreable_pair_count", 0)) == 15,
            ),
            _gate_row(
                "G5_rules_and_next_gates_predeclared",
                "Are discriminant rules and next gates materialized?",
                {
                    "rule_count": int(len(discriminant_rule_rows)),
                    "next_gate_count": int(len(next_gate_rows)),
                    "case_count": int(len(case_rows)),
                },
                "at least five rules, four next gates, and eight case rows",
                int(len(discriminant_rule_rows)) >= 5
                and int(len(next_gate_rows)) >= 4
                and int(len(case_rows)) == 8,
            ),
            _gate_row(
                "G6_no_claim_promotion",
                "Are wall, pathway, method, quality, replay, and generality claims closed?",
                {
                    "wall_promotion_status": _count_dict(case_rows["wall_promotion_status"]),
                    "method_status": _count_dict(case_rows["method_status"]),
                    "blocked_next_gates": next_gate_rows.loc[
                        next_gate_rows["status"].astype(str).str.contains("blocked"),
                        "next_gate",
                    ].astype(str).tolist(),
                },
                "no promotion flags and screened gap expansion blocked",
                bool(case_rows["wall_promotion_status"].eq(WALL_PROMOTION_STATUS).all())
                and bool(case_rows["method_status"].eq(METHOD_STATUS).all())
                and "screened_gap_expansion"
                in set(
                    next_gate_rows.loc[
                        next_gate_rows["status"].astype(str).eq("blocked_for_now"),
                        "next_gate",
                    ].astype(str)
                ),
            ),
        ]
    )


def _summary(
    *,
    output_dir: Path,
    pathway_dir: Path,
    low_fraction_audit_dir: Path,
    surface_panel_dir: Path,
    case_rows: pd.DataFrame,
    next_gate_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> dict[str, Any]:
    failed_gates = list(
        gate_matrix.loc[gate_matrix["gate_status"].ne("pass"), "gate_id"].astype(str)
    )
    return {
        "schema": "nanoclustering_g4_8_first_pass_transition_type_panel_contract_summary.v1",
        "status": RUN_STATUS,
        "output_dir": str(output_dir),
        "pathway_dir": str(pathway_dir),
        "low_fraction_audit_dir": str(low_fraction_audit_dir),
        "surface_panel_dir": str(surface_panel_dir),
        "case_row_count": int(len(case_rows)),
        "transition_type_class_counts": _count_dict(case_rows["transition_type_class"]),
        "next_gate_status_counts": _count_dict(next_gate_rows["status"]),
        "recommended_next_gate": "typed_ladder_relation_rule_contract",
        "alternative_next_gate": "endpoint_object_membership_audit",
        "blocked_next_gate": "screened_gap_expansion",
        "direction_status": "correct_to_freeze_transition_type_panel_before_more_expansion",
        "gate_status_counts": _count_dict(gate_matrix["gate_status"]),
        "failed_gates": failed_gates,
        "route_execution_opened": False,
        "wall_claim_ready": False,
        "pathway_claim_ready": False,
        "panel_generality_claim_ready": False,
        "method_claim_ready": False,
        "quality_claim_ready": False,
        "interpretation": (
            "The result direction is correct: 001/007 reveal that target-like "
            "collapse is not enough, while 016 remains the only finite recurrent "
            "transition-band reference. The next work should freeze transition-type "
            "discrimination and then choose either typed-ladder relation rules or "
            "endpoint-object membership, not broaden screened gaps."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _report(
    *,
    summary: dict[str, Any],
    case_rows: pd.DataFrame,
    discriminant_rule_rows: pd.DataFrame,
    next_gate_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# NanoClustering G4.8 First-Pass Transition-Type Panel Contract",
            "",
            f"- status: `{summary['status']}`",
            f"- direction_status: `{summary['direction_status']}`",
            f"- transition_type_class_counts: {summary['transition_type_class_counts']}",
            f"- recommended_next_gate: `{summary['recommended_next_gate']}`",
            f"- alternative_next_gate: `{summary['alternative_next_gate']}`",
            f"- blocked_next_gate: `{summary['blocked_next_gate']}`",
            f"- failed_gates: {summary['failed_gates']}",
            f"- interpretation: {summary['interpretation']}",
            f"- claim_boundary: {CLAIM_BOUNDARY}",
            "",
            "## Case Rows",
            "",
            _markdown_table(
                case_rows,
                [
                    "local_pair_id",
                    "transition_type_role",
                    "transition_type_class",
                    "transition_band_fraction_readout",
                    "target_fraction_readout",
                    "claim_status_after_contract",
                    "next_use",
                ],
            ),
            "",
            "## Discriminant Rules",
            "",
            _markdown_table(
                discriminant_rule_rows,
                [
                    "rule_id",
                    "rule_group",
                    "rule",
                    "positive_requirement",
                    "negative_control",
                ],
            ),
            "",
            "## Next Gates",
            "",
            _markdown_table(
                next_gate_rows,
                ["next_gate_id", "next_gate", "priority", "status", "rationale"],
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
        ]
    )


def run(
    *,
    pathway_dir: Path,
    low_fraction_audit_dir: Path,
    surface_panel_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    pathway_summary = _read_json(pathway_dir / PATHWAY_SUMMARY_JSON)
    low_fraction_summary = _read_json(low_fraction_audit_dir / LOW_FRACTION_AUDIT_SUMMARY_JSON)
    surface_panel_summary = _read_json(surface_panel_dir / SURFACE_PANEL_SUMMARY_JSON)
    pathway_fraction_rows = _read_csv(pathway_dir / PATHWAY_FRACTION_ROWS_CSV)
    low_fraction_pair_rows = _read_csv(
        low_fraction_audit_dir / LOW_FRACTION_AUDIT_PAIR_ROWS_CSV
    )

    case_rows = _case_rows(
        pathway_fraction_rows=pathway_fraction_rows,
        low_fraction_pair_rows=low_fraction_pair_rows,
    )
    discriminant_rule_rows = _discriminant_rule_rows()
    next_gate_rows = _next_gate_rows()
    decision_rows = _decision_rows()
    gate_matrix = _gate_matrix(
        pathway_summary=pathway_summary,
        pathway_fraction_rows=pathway_fraction_rows,
        low_fraction_summary=low_fraction_summary,
        surface_panel_summary=surface_panel_summary,
        case_rows=case_rows,
        discriminant_rule_rows=discriminant_rule_rows,
        next_gate_rows=next_gate_rows,
    )
    summary = _summary(
        output_dir=output_dir,
        pathway_dir=pathway_dir,
        low_fraction_audit_dir=low_fraction_audit_dir,
        surface_panel_dir=surface_panel_dir,
        case_rows=case_rows,
        next_gate_rows=next_gate_rows,
        gate_matrix=gate_matrix,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(case_rows, output_dir / CASE_ROWS_CSV)
    _write_csv(discriminant_rule_rows, output_dir / DISCRIMINANT_RULE_ROWS_CSV)
    _write_csv(next_gate_rows, output_dir / NEXT_GATE_ROWS_CSV)
    _write_csv(decision_rows, output_dir / DECISION_ROWS_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_transition_type_panel_contract_config.v1",
        "pathway_dir": str(pathway_dir),
        "low_fraction_audit_dir": str(low_fraction_audit_dir),
        "surface_panel_dir": str(surface_panel_dir),
        "output_dir": str(output_dir),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / REPORT_MD).write_text(
        _report(
            summary=summary,
            case_rows=case_rows,
            discriminant_rule_rows=discriminant_rule_rows,
            next_gate_rows=next_gate_rows,
            decision_rows=decision_rows,
            gate_matrix=gate_matrix,
        ),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pathway-dir", type=Path, default=DEFAULT_016_PATHWAY_DIR)
    parser.add_argument(
        "--low-fraction-audit-dir",
        type=Path,
        default=DEFAULT_LOW_FRACTION_AUDIT_DIR,
    )
    parser.add_argument("--surface-panel-dir", type=Path, default=DEFAULT_SURFACE_PANEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        pathway_dir=args.pathway_dir,
        low_fraction_audit_dir=args.low_fraction_audit_dir,
        surface_panel_dir=args.surface_panel_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
