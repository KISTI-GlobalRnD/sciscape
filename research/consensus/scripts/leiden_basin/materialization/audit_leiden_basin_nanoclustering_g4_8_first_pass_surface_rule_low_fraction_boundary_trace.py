#!/usr/bin/env python3
"""Audit the 001/007 low-fraction schedule-boundary trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from audit_leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace import (
    DEFAULT_OUTPUT_DIR as DEFAULT_GAP_FILL_AUDIT_DIR,
    PAIR_SURFACE_ROWS_CSV as GAP_FILL_AUDIT_PAIR_SURFACE_ROWS_CSV,
    SUMMARY_JSON as GAP_FILL_AUDIT_SUMMARY_JSON,
)
from design_leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_contract import (
    CANDIDATE_IDS,
    DEFAULT_OUTPUT_DIR as DEFAULT_LOW_FRACTION_CONTRACT_DIR,
    GATE_MATRIX_CSV as CONTRACT_GATE_MATRIX_CSV,
    SUMMARY_JSON as CONTRACT_SUMMARY_JSON,
)
from run_leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace import (
    DEFAULT_OUTPUT_DIR as DEFAULT_LOW_FRACTION_TRACE_DIR,
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
    / "leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_audit_gamma1e5_20260609"
)

PAIR_SURFACE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_audit_pair_surface_rows.csv"
)
UPDATED_PAIR_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_audit_updated_pair_rows.csv"
)
CLASS_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_audit_class_rows.csv"
)
EVIDENCE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_audit_evidence_rows.csv"
)
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_audit_decision_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_audit_gate_matrix.csv"
)
SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_audit_summary.json"
)
CONFIG_JSON = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_audit_config.json"
)
REPORT_MD = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_audit_report.md"
)

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace"
ROUTE_EXECUTION_STATUS = "audited_surface_rule_low_fraction_boundary_trace"
WALL_PROMOTION_STATUS = "not_promoted_low_fraction_boundary_trace_audit_only"
METHOD_STATUS = "surface_rule_low_fraction_boundary_trace_audit_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass 001/007 low-fraction schedule-boundary "
    "trace audit only; it qualifies whether prior negative guards were "
    "0.5-bound. It keeps screened gaps closed and does not promote wall, "
    "pathway, panel-generality, quality/cost, full-replay, or method claims."
)


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


def _class_decision(boundary_class: str) -> dict[str, str]:
    if boundary_class == "low_fraction_diagnostic_recurrence_candidate":
        return {
            "scoreability_status": "scoreable_core",
            "surface_rule_class": "low_fraction_diagnostic_recurrence_candidate",
            "generalization_status": "additional_diagnostic_reference_not_panel_generality",
            "promotion_status": "diagnostic_only_wall_pathway_blocked",
            "readiness_gap": "needs_object_surface_and_independent_controls",
            "readiness_decision": "retain_as_low_fraction_diagnostic_candidate",
            "claim_status": "diagnostic_only",
        }
    if boundary_class == "low_fraction_single_side_signal_without_full_recurrence":
        return {
            "scoreability_status": "scoreable_core",
            "surface_rule_class": "low_fraction_single_side_signal_guard_needs_positive_audit",
            "generalization_status": "possible_low_fraction_signal_not_panel_generality",
            "promotion_status": "diagnostic_only_wall_pathway_blocked",
            "readiness_gap": "low_fraction_single_side_signal_without_full_recurrence",
            "readiness_decision": "hold_negative_guard_pending_positive_signal_audit",
            "claim_status": "diagnostic_only",
        }
    if boundary_class == "low_fraction_late_target_collapse_guard":
        return {
            "scoreability_status": "scoreable_core",
            "surface_rule_class": "low_fraction_late_target_collapse_guard",
            "generalization_status": "late_target_collapse_not_016_like",
            "promotion_status": "blocked_boundary_artifact_guard",
            "readiness_gap": "low_fraction_target_like_without_single_side_band",
            "readiness_decision": "reclassify_as_late_target_collapse_guard",
            "claim_status": "blocked",
        }
    return {
        "scoreability_status": "scoreable_core",
        "surface_rule_class": "low_fraction_reinforced_negative_no_recurrence_guard",
        "generalization_status": "low_fraction_not_016_like",
        "promotion_status": "blocked_negative_guard",
        "readiness_gap": "low_fraction_boundary_checked_no_transition_signal",
        "readiness_decision": "retain_scoreable_negative_guard_reinforced",
        "claim_status": "blocked",
    }


def _audited_pair_rows(
    prior_pair_rows: pd.DataFrame,
    trace_pair_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows = prior_pair_rows.copy()
    trace_lookup = trace_pair_rows.set_index("local_pair_id").to_dict("index")
    for index, row in rows.iterrows():
        pair_id = str(row["local_pair_id"])
        if pair_id not in CANDIDATE_IDS:
            continue
        trace = trace_lookup[pair_id]
        decision = _class_decision(str(trace["low_fraction_boundary_class"]))
        for column, value in decision.items():
            rows.at[index, column] = value
        rows.at[index, "surface_level"] = "state"
        rows.at[index, "object_status"] = "unknown"
        rows.at[index, "relation_status"] = "unresolved"
        rows.at[index, "generalization_role"] = "low_fraction_boundary_executed_candidate"
        rows.at[index, "route_state_morphology_class"] = str(
            trace["low_fraction_boundary_class"]
        )
        rows.at[index, "low_fraction_boundary_class"] = str(
            trace["low_fraction_boundary_class"]
        )
        rows.at[index, "low_fraction_single_side_fraction_total"] = int(
            trace["low_fraction_single_side_fraction_total"]
        )
        rows.at[index, "low_fraction_target_like_fraction_total"] = int(
            trace["low_fraction_target_like_fraction_total"]
        )
        rows.at[index, "route_trace_pair_present"] = True
        rows.at[index, "route_negative_pair_present"] = (
            str(trace["low_fraction_boundary_class"])
            == "low_fraction_no_recurrence_negative_guard_reinforced"
        )
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
    prior_summary: dict[str, Any],
    contract_summary: dict[str, Any],
    trace_summary: dict[str, Any],
    trace_pair_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "evidence_id": "E1_prior_gap_fill_audit",
            "evidence_type": "upstream_audit",
            "evidence_summary": {
                "prior_failed_gates": prior_summary.get("failed_gates"),
                "prior_scoreable_pair_count": prior_summary.get("scoreable_pair_count"),
                "prior_gap_fill_negative_pair_ids": prior_summary.get(
                    "gap_fill_scoreable_negative_pair_ids"
                ),
            },
        },
        {
            "evidence_id": "E2_low_fraction_contract",
            "evidence_type": "contract",
            "evidence_summary": {
                "contract_failed_gates": contract_summary.get("failed_gates"),
                "route_plan_row_count": contract_summary.get("route_plan_row_count"),
                "low_fractions": contract_summary.get("low_fractions"),
            },
        },
        {
            "evidence_id": "E3_low_fraction_trace",
            "evidence_type": "trace",
            "evidence_summary": {
                "trace_failed_gates": trace_summary.get("failed_gates"),
                "trace_row_count": trace_summary.get("trace_row_count"),
                "low_fraction_boundary_class_counts": trace_summary.get(
                    "low_fraction_boundary_class_counts"
                ),
                "late_target_collapse_pair_ids": trace_summary.get(
                    "low_fraction_late_target_collapse_pair_ids"
                ),
                "single_side_signal_pair_ids": trace_summary.get(
                    "low_fraction_single_side_signal_pair_ids"
                ),
            },
        },
        {
            "evidence_id": "E4_pair_readout",
            "evidence_type": "pair_rows",
            "evidence_summary": trace_pair_rows[
                [
                    "local_pair_id",
                    "low_fraction_boundary_class",
                    "low_fraction_single_side_fraction_total",
                    "low_fraction_target_like_fraction_total",
                    "schedule_artifact_status",
                ]
            ].to_dict("records"),
        },
    ]
    frame = pd.DataFrame(rows)
    frame["evidence_summary"] = frame["evidence_summary"].map(_json_dump)
    frame["run_status"] = RUN_STATUS
    frame["claim_boundary"] = CLAIM_BOUNDARY
    return frame


def _decision_rows(trace_summary: dict[str, Any]) -> pd.DataFrame:
    late_ids = trace_summary.get("low_fraction_late_target_collapse_pair_ids", [])
    single_ids = trace_summary.get("low_fraction_single_side_signal_pair_ids", [])
    recurrence_ids = trace_summary.get("low_fraction_diagnostic_recurrence_pair_ids", [])
    if recurrence_ids or single_ids:
        decision = "hold_negative_guard_pending_positive_signal_audit"
        rationale = (
            "A low-fraction single-side signal appeared, so the pair cannot be used "
            "as a simple negative guard without a separate positive-signal audit."
        )
    elif late_ids:
        decision = "reclassify_as_late_target_collapse_guard"
        rationale = (
            "The lower schedule produced target-like collapse without a single-side "
            "band, so the prior negative was 0.5-qualified rather than a no-transition guard."
        )
    else:
        decision = "reinforce_negative_guard"
        rationale = (
            "The lower schedule produced neither target-like nor single-side signal, "
            "so the prior negative guard is reinforced under the tested boundary."
        )
    rows = pd.DataFrame(
        [
            {
                "decision_id": "D1",
                "decision": decision,
                "rationale": rationale,
            },
            {
                "decision_id": "D2",
                "decision": "keep_screened_gaps_closed",
                "rationale": "The audit only qualifies 001/007; it does not reopen the 15 screened gaps.",
            },
            {
                "decision_id": "D3",
                "decision": "keep_pathway_claim_closed",
                "rationale": "The trace is independent-fraction readout, not a warm-start pathway.",
            },
        ]
    )
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _gate_matrix(
    *,
    contract_gates: pd.DataFrame,
    trace_gates: pd.DataFrame,
    pair_rows: pd.DataFrame,
    updated_pair_rows: pd.DataFrame,
    trace_pair_rows: pd.DataFrame,
    validation: dict[str, Any],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _gate_row(
                "G1_upstream_gates_pass",
                "Did contract and trace gates pass?",
                {
                    "contract_gate_status_counts": _count_dict(contract_gates["gate_status"]),
                    "trace_gate_status_counts": _count_dict(trace_gates["gate_status"]),
                },
                "all upstream gates pass",
                bool(contract_gates["gate_status"].astype(str).eq("pass").all())
                and bool(trace_gates["gate_status"].astype(str).eq("pass").all()),
            ),
            _gate_row(
                "G2_only_001_007_updated",
                "Were only 001/007 updated by this audit?",
                sorted(updated_pair_rows["local_pair_id"].astype(str).tolist()),
                "updated pair ids equal 001/007",
                set(updated_pair_rows["local_pair_id"].astype(str)) == set(CANDIDATE_IDS)
                and len(updated_pair_rows) == len(CANDIDATE_IDS),
            ),
            _gate_row(
                "G3_low_fraction_classes_consumed",
                "Were trace boundary classes consumed into the pair surface?",
                {
                    "trace_classes": _count_dict(trace_pair_rows["low_fraction_boundary_class"]),
                    "pair_surface_classes": _count_dict(
                        updated_pair_rows["route_state_morphology_class"]
                    ),
                },
                "each updated row carries a low-fraction boundary class",
                bool(
                    updated_pair_rows["route_state_morphology_class"]
                    .astype(str)
                    .str.startswith("low_fraction_")
                    .all()
                ),
            ),
            _gate_row(
                "G4_schema_valid",
                "Does the updated pair surface satisfy the surface schema adapter?",
                validation,
                "required columns present and allowed values valid",
                bool(validation["required_columns_present"])
                and bool(validation["required_values_valid"]),
            ),
            _gate_row(
                "G5_screened_gaps_still_closed",
                "Do screened not-scoreable gaps remain closed?",
                _count_dict(pair_rows["scoreability_status"]),
                "15 screened rows remain not-scoreable",
                int(pair_rows["scoreability_status"].astype(str).eq("screened_not_scoreable").sum())
                == 15,
            ),
            _gate_row(
                "G6_no_claim_promotion",
                "Are wall, pathway, method, quality, replay, and generality claims closed?",
                {
                    "claim_status_counts": _count_dict(pair_rows["claim_status"]),
                    "promotion_status_counts": _count_dict(pair_rows["promotion_status"]),
                },
                "001/007 are blocked or diagnostic-only; no method/pathway promotion",
                not bool(
                    pair_rows["promotion_status"]
                    .astype(str)
                    .str.contains("method|pathway_ready|wall_ready", regex=True)
                    .any()
                ),
            ),
        ]
    )


def _summary(
    *,
    output_dir: Path,
    gap_fill_audit_dir: Path,
    low_fraction_contract_dir: Path,
    low_fraction_trace_dir: Path,
    pair_rows: pd.DataFrame,
    updated_pair_rows: pd.DataFrame,
    trace_pair_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
    validation: dict[str, Any],
) -> dict[str, Any]:
    failed_gates = list(
        gate_matrix.loc[gate_matrix["gate_status"].ne("pass"), "gate_id"].astype(str)
    )
    return {
        "schema": "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_audit_summary.v1",
        "status": RUN_STATUS,
        "output_dir": str(output_dir),
        "gap_fill_audit_dir": str(gap_fill_audit_dir),
        "low_fraction_contract_dir": str(low_fraction_contract_dir),
        "low_fraction_trace_dir": str(low_fraction_trace_dir),
        "pair_row_count": int(len(pair_rows)),
        "updated_pair_count": int(len(updated_pair_rows)),
        "scoreable_pair_count": int(
            pair_rows["scoreability_status"].astype(str).eq("scoreable_core").sum()
        ),
        "not_scoreable_pair_count": int(
            pair_rows["scoreability_status"].astype(str).str.contains("not_scoreable").sum()
        ),
        "low_fraction_boundary_class_counts": _count_dict(
            trace_pair_rows["low_fraction_boundary_class"]
        ),
        "late_target_collapse_pair_ids": sorted(
            updated_pair_rows.loc[
                updated_pair_rows["surface_rule_class"]
                .astype(str)
                .eq("low_fraction_late_target_collapse_guard"),
                "local_pair_id",
            ].astype(str)
        ),
        "reinforced_negative_pair_ids": sorted(
            updated_pair_rows.loc[
                updated_pair_rows["surface_rule_class"]
                .astype(str)
                .eq("low_fraction_reinforced_negative_no_recurrence_guard"),
                "local_pair_id",
            ].astype(str)
        ),
        "single_side_signal_pair_ids": sorted(
            updated_pair_rows.loc[
                updated_pair_rows["surface_rule_class"]
                .astype(str)
                .str.contains("single_side_signal"),
                "local_pair_id",
            ].astype(str)
        ),
        "gate_status_counts": _count_dict(gate_matrix["gate_status"]),
        "failed_gates": failed_gates,
        "schema_validation": validation,
        "wall_claim_ready": False,
        "pathway_claim_ready": False,
        "panel_generality_claim_ready": False,
        "method_claim_ready": False,
        "quality_claim_ready": False,
        "interpretation": (
            "This audit qualifies the prior 001/007 negative guard against the "
            "0.5 schedule boundary only. It does not promote pathway, wall, "
            "panel-generality, quality/cost, replay, or method claims."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
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
    return "\n".join(
        [
            "# NanoClustering G4.8 001/007 Low-Fraction Boundary Trace Audit",
            "",
            f"- status: `{summary['status']}`",
            f"- scoreable_pair_count: {summary['scoreable_pair_count']}",
            f"- not_scoreable_pair_count: {summary['not_scoreable_pair_count']}",
            f"- low_fraction_boundary_class_counts: {summary['low_fraction_boundary_class_counts']}",
            f"- late_target_collapse_pair_ids: {summary['late_target_collapse_pair_ids']}",
            f"- reinforced_negative_pair_ids: {summary['reinforced_negative_pair_ids']}",
            f"- single_side_signal_pair_ids: {summary['single_side_signal_pair_ids']}",
            f"- failed_gates: {summary['failed_gates']}",
            f"- interpretation: {summary['interpretation']}",
            f"- claim_boundary: {CLAIM_BOUNDARY}",
            "",
            "## Updated Pair Rows",
            "",
            _markdown_table(
                updated_pair_rows,
                [
                    "local_pair_id",
                    "surface_rule_class",
                    "generalization_status",
                    "promotion_status",
                    "low_fraction_boundary_class",
                    "low_fraction_single_side_fraction_total",
                    "low_fraction_target_like_fraction_total",
                    "claim_status",
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
            _markdown_table(evidence_rows, ["evidence_id", "evidence_type", "evidence_summary"]),
            "",
            "## Decision Rows",
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
    gap_fill_audit_dir: Path,
    low_fraction_contract_dir: Path,
    low_fraction_trace_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    prior_summary = _read_json(gap_fill_audit_dir / GAP_FILL_AUDIT_SUMMARY_JSON)
    contract_summary = _read_json(low_fraction_contract_dir / CONTRACT_SUMMARY_JSON)
    trace_summary = _read_json(low_fraction_trace_dir / TRACE_SUMMARY_JSON)
    prior_pair_rows = _read_csv(gap_fill_audit_dir / GAP_FILL_AUDIT_PAIR_SURFACE_ROWS_CSV)
    contract_gates = _read_csv(low_fraction_contract_dir / CONTRACT_GATE_MATRIX_CSV)
    trace_gates = _read_csv(low_fraction_trace_dir / TRACE_GATE_MATRIX_CSV)
    trace_pair_rows = _read_csv(low_fraction_trace_dir / TRACE_PAIR_READOUT_ROWS_CSV)
    trace_seed_rows = _read_csv(low_fraction_trace_dir / TRACE_SEED_ROUTE_ROWS_CSV)

    pair_rows = _audited_pair_rows(prior_pair_rows, trace_pair_rows)
    updated_pair_rows = pair_rows[
        pair_rows["local_pair_id"].astype(str).isin(CANDIDATE_IDS)
    ].copy()
    validation = validate_surface_claim_rows(pair_rows)
    class_rows = _class_rows(pair_rows)
    evidence_rows = _evidence_rows(
        prior_summary=prior_summary,
        contract_summary=contract_summary,
        trace_summary=trace_summary,
        trace_pair_rows=trace_pair_rows,
    )
    decision_rows = _decision_rows(trace_summary)
    gate_matrix = _gate_matrix(
        contract_gates=contract_gates,
        trace_gates=trace_gates,
        pair_rows=pair_rows,
        updated_pair_rows=updated_pair_rows,
        trace_pair_rows=trace_pair_rows,
        validation=validation,
    )
    summary = _summary(
        output_dir=output_dir,
        gap_fill_audit_dir=gap_fill_audit_dir,
        low_fraction_contract_dir=low_fraction_contract_dir,
        low_fraction_trace_dir=low_fraction_trace_dir,
        pair_rows=pair_rows,
        updated_pair_rows=updated_pair_rows,
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
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_trace_audit_config.v1",
        "gap_fill_audit_dir": str(gap_fill_audit_dir),
        "low_fraction_contract_dir": str(low_fraction_contract_dir),
        "low_fraction_trace_dir": str(low_fraction_trace_dir),
        "output_dir": str(output_dir),
        "candidate_pair_ids": list(CANDIDATE_IDS),
        "schema_adapter_version": SCHEMA_ADAPTER_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
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
    parser.add_argument("--gap-fill-audit-dir", type=Path, default=DEFAULT_GAP_FILL_AUDIT_DIR)
    parser.add_argument("--low-fraction-contract-dir", type=Path, default=DEFAULT_LOW_FRACTION_CONTRACT_DIR)
    parser.add_argument("--low-fraction-trace-dir", type=Path, default=DEFAULT_LOW_FRACTION_TRACE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        gap_fill_audit_dir=args.gap_fill_audit_dir,
        low_fraction_contract_dir=args.low_fraction_contract_dir,
        low_fraction_trace_dir=args.low_fraction_trace_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
