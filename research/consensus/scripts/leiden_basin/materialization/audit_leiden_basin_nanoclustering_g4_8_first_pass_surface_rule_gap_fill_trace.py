#!/usr/bin/env python3
"""Audit the first-pass surface-rule gap-fill trace.

This audit consumes the panel-readiness surface, the gap-fill contract, and the
gap-fill trace. It updates only the two executed diagnostic gaps (``001`` and
``007``). Both executed rows become scoreable negative guards because the
predeclared trace found no diagnostic transition-band recurrence.

The audit does not promote wall/pathway, method, quality/cost, full-replay, or
panel-generality claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from audit_leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_panel_readiness import (
    DEFAULT_OUTPUT_DIR as DEFAULT_PANEL_READINESS_DIR,
    PAIR_SURFACE_ROWS_CSV as PANEL_PAIR_SURFACE_ROWS_CSV,
    SUMMARY_JSON as PANEL_SUMMARY_JSON,
)
from design_leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_gap_fill_contract import (
    DEFAULT_OUTPUT_DIR as DEFAULT_GAP_FILL_CONTRACT_DIR,
    GATE_MATRIX_CSV as CONTRACT_GATE_MATRIX_CSV,
    SUMMARY_JSON as CONTRACT_SUMMARY_JSON,
)
from run_leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace import (
    DEFAULT_OUTPUT_DIR as DEFAULT_GAP_FILL_TRACE_DIR,
    GATE_MATRIX_CSV as TRACE_GATE_MATRIX_CSV,
    PAIR_READOUT_ROWS_CSV as TRACE_PAIR_READOUT_ROWS_CSV,
    SEED_ROUTE_ROWS_CSV as TRACE_SEED_ROUTE_ROWS_CSV,
    SUMMARY_JSON as TRACE_SUMMARY_JSON,
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
    / "leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_audit_gamma1e5_20260609"
)

PAIR_SURFACE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_audit_pair_surface_rows.csv"
)
UPDATED_PAIR_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_audit_updated_pair_rows.csv"
)
CLASS_ROWS_CSV = "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_audit_class_rows.csv"
EVIDENCE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_audit_evidence_rows.csv"
)
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_audit_decision_rows.csv"
)
GATE_MATRIX_CSV = "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_audit_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_audit_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_audit_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_audit_report.md"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace"
ROUTE_EXECUTION_STATUS = "audited_surface_rule_gap_fill_trace"
WALL_PROMOTION_STATUS = "not_promoted_gap_fill_trace_audit_only"
METHOD_STATUS = "surface_rule_gap_fill_trace_audit_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass surface-rule gap-fill trace audit only; "
    "updates the executed 001/007 diagnostic gaps into scoreable negative guards "
    "because no diagnostic transition-band recurrence was observed. It keeps "
    "screened gaps closed and does not promote wall, pathway, panel-generality, "
    "quality/cost, full-replay, or method claims."
)

EXECUTED_GAP_FILL_IDS = {"local_pair_001", "local_pair_007"}
FIXED_CORE_IDS = {
    "local_pair_005",
    "local_pair_009",
    "local_pair_012",
    "local_pair_014",
    "local_pair_016",
    "local_pair_020",
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


def _audited_pair_rows(
    panel_rows: pd.DataFrame,
    trace_pair_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows = panel_rows.copy()
    trace_lookup = trace_pair_rows.set_index("local_pair_id").to_dict("index")
    for index, row in rows.iterrows():
        pair_id = str(row["local_pair_id"])
        if pair_id not in EXECUTED_GAP_FILL_IDS:
            continue
        trace = trace_lookup[pair_id]
        route_count = int(trace["route_sequence_count"])
        recurrence_count = int(trace["diagnostic_recurrence_pass_count"])
        if route_count <= 0:
            scoreability = "diagnostic_not_scoreable"
            surface_class = "gap_fill_residual_not_scoreable"
            generalization_status = "not_scoreable_surface_gap"
            promotion_status = "closed_no_promotion"
            readiness_gap = "gap_fill_trace_missing_route_readout"
            decision = "retain_residual_gap"
            claim_status = "diagnostic_only"
        elif recurrence_count > 0:
            scoreability = "scoreable_core"
            surface_class = "gap_fill_diagnostic_recurrence_reference_candidate"
            generalization_status = "additional_diagnostic_reference_not_panel_generality"
            promotion_status = "diagnostic_only_wall_pathway_blocked"
            readiness_gap = "needs_object_surface_and_independent_controls"
            decision = "retain_as_diagnostic_recurrence_candidate"
            claim_status = "diagnostic_only"
        else:
            scoreability = "scoreable_core"
            surface_class = "gap_fill_scoreable_negative_no_recurrence_guard"
            generalization_status = "gap_fill_not_016_like"
            promotion_status = "blocked_negative_guard"
            readiness_gap = "gap_filled_negative_no_transition_band_recurrence"
            decision = "promote_gap_to_scoreable_negative_guard"
            claim_status = "blocked"
        rows.at[index, "scoreability_status"] = scoreability
        rows.at[index, "surface_level"] = "state"
        rows.at[index, "object_status"] = "unknown"
        rows.at[index, "relation_status"] = "unresolved"
        rows.at[index, "claim_status"] = claim_status
        rows.at[index, "surface_rule_class"] = surface_class
        rows.at[index, "generalization_role"] = "gap_fill_executed_candidate"
        rows.at[index, "generalization_status"] = generalization_status
        rows.at[index, "promotion_status"] = promotion_status
        rows.at[index, "readiness_gap"] = readiness_gap
        rows.at[index, "readiness_decision"] = decision
        rows.at[index, "route_trace_pair_present"] = True
        rows.at[index, "route_negative_pair_present"] = recurrence_count == 0
        rows.at[index, "contract_feature_scoreable"] = True
        rows.at[index, "schema_adapter_version"] = SCHEMA_ADAPTER_VERSION
        rows.at[index, "run_status"] = RUN_STATUS
        rows.at[index, "claim_boundary"] = CLAIM_BOUNDARY
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values("local_pair_id", kind="mergesort").reset_index(drop=True)


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


def _evidence_rows(
    *,
    panel_summary: dict[str, Any],
    contract_summary: dict[str, Any],
    trace_summary: dict[str, Any],
    trace_pair_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "evidence_id": "E1_panel_readiness_available",
            "evidence_status": "ready",
            "observed": {
                "failed_gates": panel_summary.get("failed_gates"),
                "pair_row_count": panel_summary.get("pair_row_count"),
                "scoreable_core_pair_count": panel_summary.get("scoreable_core_pair_count"),
                "non_scoreable_pair_count": panel_summary.get("non_scoreable_pair_count"),
            },
            "claim_effect": "provides pre-gap-fill surface accounting baseline",
        },
        {
            "evidence_id": "E2_gap_fill_contract_passed",
            "evidence_status": "ready",
            "observed": {
                "failed_gates": contract_summary.get("failed_gates"),
                "candidate_pair_ids": contract_summary.get("candidate_pair_ids"),
                "route_plan_row_count": contract_summary.get("route_plan_row_count"),
            },
            "claim_effect": "predeclares 001/007-only route scope",
        },
        {
            "evidence_id": "E3_gap_fill_trace_executed",
            "evidence_status": "ready",
            "observed": {
                "failed_gates": trace_summary.get("failed_gates"),
                "trace_row_count": trace_summary.get("trace_row_count"),
                "seed_route_row_count": trace_summary.get("seed_route_row_count"),
                "diagnostic_recurrence_pair_ids": trace_summary.get(
                    "diagnostic_recurrence_pair_ids"
                ),
                "gap_fill_pair_class_counts": trace_summary.get(
                    "gap_fill_pair_class_counts"
                ),
            },
            "claim_effect": "turns 001/007 from unscoreable diagnostics into scoreable negatives",
        },
        {
            "evidence_id": "E4_pair_negative_readout",
            "evidence_status": "scoreable_negative",
            "observed": trace_pair_rows[
                [
                    "local_pair_id",
                    "route_sequence_count",
                    "diagnostic_recurrence_pass_count",
                    "source_family_start_count",
                    "finite_single_side_band_count",
                    "final_target_like_count",
                    "gap_fill_pair_class",
                ]
            ].to_dict("records"),
            "claim_effect": "shows no diagnostic transition-band recurrence for either executed gap",
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
            "decision": "gap_fill_trace_negative",
            "rationale": (
                "001/007 were executed under the predeclared gap-fill contract and "
                "showed zero diagnostic transition-band recurrence."
            ),
        },
        {
            "decision_id": "D2",
            "decision": "promote_001_007_to_scoreable_negative_guards",
            "rationale": (
                "They now have route/fraction readout, so they are no longer "
                "diagnostic-not-scoreable gaps; they are scoreable negative guards."
            ),
        },
        {
            "decision_id": "D3",
            "decision": "retain_15_screened_gaps",
            "rationale": (
                "The contract excluded screened rows, so they remain not-scoreable "
                "surface gaps."
            ),
        },
        {
            "decision_id": "D4",
            "decision": "retain_016_single_reference",
            "rationale": (
                "No additional diagnostic recurrence was observed, so 016 remains "
                "the single transition-band reference."
            ),
        },
        {
            "decision_id": "D5",
            "decision": "no_claim_promotion",
            "rationale": (
                "The audit opens no wall, pathway, panel-generality, method, "
                "quality/cost, full-replay, or route-execution claim."
            ),
        },
    ]
    frame = pd.DataFrame(rows)
    frame["run_status"] = RUN_STATUS
    frame["claim_boundary"] = CLAIM_BOUNDARY
    return frame


def _gate_matrix(
    *,
    contract_gates: pd.DataFrame,
    trace_gates: pd.DataFrame,
    pair_rows: pd.DataFrame,
    trace_pair_rows: pd.DataFrame,
    validation: dict[str, Any],
) -> pd.DataFrame:
    updated = pair_rows[pair_rows["local_pair_id"].astype(str).isin(EXECUTED_GAP_FILL_IDS)]
    screened = pair_rows[
        pair_rows["surface_rule_class"].astype(str).eq("screened_panel_context_not_scoreable")
    ]
    diagnostic_references = pair_rows[
        pair_rows["surface_rule_class"].astype(str).eq(
            "diagnostic_transition_band_surface_reference"
        )
    ]
    return pd.DataFrame(
        [
            _gate_row(
                "G1_upstream_contract_and_trace_pass",
                "Did contract and trace gates pass?",
                {
                    "contract_gate_counts": _count_dict(contract_gates["gate_status"]),
                    "trace_gate_counts": _count_dict(trace_gates["gate_status"]),
                },
                "all upstream contract and trace gates pass",
                bool(contract_gates["gate_status"].astype(str).eq("pass").all())
                and bool(trace_gates["gate_status"].astype(str).eq("pass").all()),
            ),
            _gate_row(
                "G2_executed_scope_is_001_007_only",
                "Were only 001 and 007 updated by execution?",
                {
                    "updated_pair_ids": sorted(updated["local_pair_id"].astype(str).tolist()),
                    "trace_pair_ids": sorted(trace_pair_rows["local_pair_id"].astype(str).tolist()),
                },
                "only 001 and 007 are updated",
                set(updated["local_pair_id"].astype(str)) == EXECUTED_GAP_FILL_IDS
                and set(trace_pair_rows["local_pair_id"].astype(str)) == EXECUTED_GAP_FILL_IDS,
            ),
            _gate_row(
                "G3_001_007_scoreable_negative",
                "Did 001/007 become scoreable negative guards with no recurrence?",
                updated[
                    [
                        "local_pair_id",
                        "scoreability_status",
                        "surface_rule_class",
                        "generalization_status",
                        "promotion_status",
                    ]
                ].to_dict("records"),
                "both updated rows are scoreable negative guards",
                bool(updated["surface_rule_class"].astype(str).eq(
                    "gap_fill_scoreable_negative_no_recurrence_guard"
                ).all())
                and bool(updated["promotion_status"].astype(str).eq(
                    "blocked_negative_guard"
                ).all()),
            ),
            _gate_row(
                "G4_15_screened_gaps_remain_closed",
                "Do the 15 screened gaps remain excluded?",
                {
                    "screened_gap_count": int(len(screened)),
                    "screened_pair_ids": sorted(screened["local_pair_id"].astype(str).tolist()),
                },
                "15 screened rows remain not-scoreable",
                len(screened) == 15
                and bool(screened["scoreability_status"].astype(str).eq(
                    "screened_not_scoreable"
                ).all()),
            ),
            _gate_row(
                "G5_single_016_reference_retained",
                "Is 016 still the only diagnostic transition-band reference?",
                diagnostic_references["local_pair_id"].astype(str).tolist(),
                "only 016 remains the diagnostic transition-band reference",
                diagnostic_references["local_pair_id"].astype(str).tolist()
                == ["local_pair_016"],
            ),
            _gate_row(
                "G6_adapter_validation_passes",
                "Do audited rows satisfy the surface schema?",
                validation,
                "required columns and values are valid",
                bool(validation["required_columns_present"])
                and bool(validation["required_values_valid"]),
            ),
            _gate_row(
                "G7_no_claim_promotion",
                "Are wall, pathway, method, quality, replay, and generality claims closed?",
                {
                    "promotion_status_counts": _count_dict(pair_rows["promotion_status"]),
                    "claim_status_counts": _count_dict(pair_rows["claim_status"]),
                },
                "no row opens a stronger claim",
                not bool(pair_rows["claim_status"].astype(str).eq("open").any()),
            ),
        ]
    )


def _summary(
    *,
    output_dir: Path,
    panel_readiness_dir: Path,
    gap_fill_contract_dir: Path,
    gap_fill_trace_dir: Path,
    pair_rows: pd.DataFrame,
    trace_pair_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
    validation: dict[str, Any],
) -> dict[str, Any]:
    failed_gates = gate_matrix.loc[
        ~gate_matrix["gate_status"].astype(str).eq("pass"),
        "gate_id",
    ].astype(str).tolist()
    scoreable_rows = pair_rows[
        pair_rows["scoreability_status"].astype(str).eq("scoreable_core")
    ]
    not_scoreable_rows = pair_rows[
        pair_rows["scoreability_status"].astype(str).isin(
            {"diagnostic_not_scoreable", "screened_not_scoreable"}
        )
    ]
    return {
        "schema": "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_audit_summary.v1",
        "status": RUN_STATUS,
        "output_dir": str(output_dir),
        "panel_readiness_dir": str(panel_readiness_dir),
        "gap_fill_contract_dir": str(gap_fill_contract_dir),
        "gap_fill_trace_dir": str(gap_fill_trace_dir),
        "claim_boundary": CLAIM_BOUNDARY,
        "pair_row_count": int(len(pair_rows)),
        "scoreable_pair_count": int(len(scoreable_rows)),
        "not_scoreable_pair_count": int(len(not_scoreable_rows)),
        "gap_fill_scoreable_negative_pair_ids": sorted(
            pair_rows.loc[
                pair_rows["surface_rule_class"].astype(str).eq(
                    "gap_fill_scoreable_negative_no_recurrence_guard"
                ),
                "local_pair_id",
            ].astype(str).tolist()
        ),
        "diagnostic_recurrence_pair_ids": sorted(
            trace_pair_rows.loc[
                trace_pair_rows["diagnostic_recurrence_pass_count"].astype(int).gt(0),
                "local_pair_id",
            ].astype(str).tolist()
        ),
        "diagnostic_transition_band_reference_pairs": ["local_pair_016"],
        "remaining_screened_gap_count": int(
            pair_rows["surface_rule_class"].astype(str).eq(
                "screened_panel_context_not_scoreable"
            ).sum()
        ),
        "surface_rule_class_counts": _count_dict(pair_rows["surface_rule_class"]),
        "generalization_status_counts": _count_dict(pair_rows["generalization_status"]),
        "promotion_status_counts": _count_dict(pair_rows["promotion_status"]),
        "gate_status_counts": _count_dict(gate_matrix["gate_status"]),
        "failed_gates": failed_gates,
        "required_columns_present": bool(validation["required_columns_present"]),
        "required_values_valid": bool(validation["required_values_valid"]),
        "invalid_values_by_column": validation["invalid_values_by_column"],
        "panel_generality_established": False,
        "wall_claim_ready": False,
        "pathway_claim_ready": False,
        "method_claim_ready": False,
        "quality_claim_ready": False,
        "interpretation": (
            "The 001/007 gap-fill execution found no diagnostic transition-band "
            "recurrence. The rows are now scoreable negative guards, reducing the "
            "not-scoreable panel from 17 to 15 while keeping 016 as the single "
            "diagnostic reference."
        ),
        "recommended_next_gate": (
            "Do not broaden the screened 15 gaps yet. Use the updated eight-row "
            "scoreable surface as the fixed evidence/guard panel, or predeclare a "
            "separate mechanism question before opening additional screened rows."
        ),
    }


def _report(
    *,
    summary: dict[str, Any],
    updated_pair_rows: pd.DataFrame,
    class_rows: pd.DataFrame,
    evidence_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> str:
    lines = [
        "# NanoClustering G4.8 First-Pass Surface Rule Gap-Fill Trace Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- pair_row_count: {summary['pair_row_count']}",
        f"- scoreable_pair_count: {summary['scoreable_pair_count']}",
        f"- not_scoreable_pair_count: {summary['not_scoreable_pair_count']}",
        f"- gap_fill_scoreable_negative_pair_ids: {summary['gap_fill_scoreable_negative_pair_ids']}",
        f"- diagnostic_recurrence_pair_ids: {summary['diagnostic_recurrence_pair_ids']}",
        f"- diagnostic_transition_band_reference_pairs: {summary['diagnostic_transition_band_reference_pairs']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {summary['claim_boundary']}",
        "",
        "## Updated Pair Rows",
        "",
        _markdown_table(
            updated_pair_rows,
            [
                "local_pair_id",
                "scoreability_status",
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
        "## Evidence Rows",
        "",
        _markdown_table(evidence_rows, ["evidence_id", "evidence_status", "claim_effect", "observed"]),
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
        (
            "This audit updates surface accounting only. It does not promote "
            "wall/pathway, method, quality/cost, full-replay, or panel-generality "
            "claims."
        ),
        "",
    ]
    return "\n".join(lines)


def run(
    *,
    panel_readiness_dir: Path,
    gap_fill_contract_dir: Path,
    gap_fill_trace_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    panel_summary = _read_json(panel_readiness_dir / PANEL_SUMMARY_JSON)
    contract_summary = _read_json(gap_fill_contract_dir / CONTRACT_SUMMARY_JSON)
    trace_summary = _read_json(gap_fill_trace_dir / TRACE_SUMMARY_JSON)
    panel_rows = _read_csv(panel_readiness_dir / PANEL_PAIR_SURFACE_ROWS_CSV)
    contract_gates = _read_csv(gap_fill_contract_dir / CONTRACT_GATE_MATRIX_CSV)
    trace_gates = _read_csv(gap_fill_trace_dir / TRACE_GATE_MATRIX_CSV)
    trace_pair_rows = _read_csv(gap_fill_trace_dir / TRACE_PAIR_READOUT_ROWS_CSV)
    trace_seed_rows = _read_csv(gap_fill_trace_dir / TRACE_SEED_ROUTE_ROWS_CSV)

    pair_rows = _audited_pair_rows(panel_rows, trace_pair_rows)
    updated_pair_rows = pair_rows[
        pair_rows["local_pair_id"].astype(str).isin(EXECUTED_GAP_FILL_IDS)
    ].copy()
    validation = validate_surface_claim_rows(pair_rows)
    class_rows = _class_rows(pair_rows)
    evidence_rows = _evidence_rows(
        panel_summary=panel_summary,
        contract_summary=contract_summary,
        trace_summary=trace_summary,
        trace_pair_rows=trace_pair_rows,
    )
    decision_rows = _decision_rows()
    gate_matrix = _gate_matrix(
        contract_gates=contract_gates,
        trace_gates=trace_gates,
        pair_rows=pair_rows,
        trace_pair_rows=trace_pair_rows,
        validation=validation,
    )
    summary = _summary(
        output_dir=output_dir,
        panel_readiness_dir=panel_readiness_dir,
        gap_fill_contract_dir=gap_fill_contract_dir,
        gap_fill_trace_dir=gap_fill_trace_dir,
        pair_rows=pair_rows,
        trace_pair_rows=trace_pair_rows,
        gate_matrix=gate_matrix,
        validation=validation,
    )
    summary["trace_seed_route_row_count"] = int(len(trace_seed_rows))

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(pair_rows, output_dir / PAIR_SURFACE_ROWS_CSV)
    _write_csv(updated_pair_rows, output_dir / UPDATED_PAIR_ROWS_CSV)
    _write_csv(class_rows, output_dir / CLASS_ROWS_CSV)
    _write_csv(evidence_rows, output_dir / EVIDENCE_ROWS_CSV)
    _write_csv(decision_rows, output_dir / DECISION_ROWS_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace_audit_config.v1",
        "panel_readiness_dir": str(panel_readiness_dir),
        "gap_fill_contract_dir": str(gap_fill_contract_dir),
        "gap_fill_trace_dir": str(gap_fill_trace_dir),
        "output_dir": str(output_dir),
        "executed_gap_fill_ids": sorted(EXECUTED_GAP_FILL_IDS),
        "schema_adapter_version": SCHEMA_ADAPTER_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / REPORT_MD).write_text(
        _report(
            summary=summary,
            updated_pair_rows=updated_pair_rows,
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
    parser.add_argument("--panel-readiness-dir", type=Path, default=DEFAULT_PANEL_READINESS_DIR)
    parser.add_argument("--gap-fill-contract-dir", type=Path, default=DEFAULT_GAP_FILL_CONTRACT_DIR)
    parser.add_argument("--gap-fill-trace-dir", type=Path, default=DEFAULT_GAP_FILL_TRACE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        panel_readiness_dir=args.panel_readiness_dir,
        gap_fill_contract_dir=args.gap_fill_contract_dir,
        gap_fill_trace_dir=args.gap_fill_trace_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
