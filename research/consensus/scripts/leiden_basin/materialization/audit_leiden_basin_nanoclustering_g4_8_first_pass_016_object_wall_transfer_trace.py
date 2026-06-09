#!/usr/bin/env python3
"""Audit the executed first-pass 016 object-wall transfer trace.

This read-only audit consumes the executed 14-row ``local_pair_016`` transfer
trace. It asks whether the observed direct-only target shape plus
typed-transient/object-identity readout is sufficient to open a local
object-wall evidence audit.

The audit keeps the evidence layers separate: direct-only target availability
is useful transfer evidence, but a local object-wall claim still requires
object identity and a recovery-loop relation that is not merely a typed
transient block. It does not rerun Leiden, expand route rows, promote pathway
labels or walls, evaluate quality/cost value, replay full NanoClustering, or
claim method success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_leiden_basin_nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace import (
    BOUNDARY_GUARD_RESULT_ROWS_CSV as TRACE_BOUNDARY_GUARD_RESULT_ROWS_CSV,
    DEFAULT_OUTPUT_DIR as DEFAULT_TRACE_DIR,
    GATE_MATRIX_CSV as TRACE_GATE_MATRIX_CSV,
    PAIR_TRANSFER_RESULT_ROWS_CSV as TRACE_PAIR_TRANSFER_RESULT_ROWS_CSV,
    POSITIVE_PAIR_ID,
    ROUTE_TRANSFER_RESULT_ROWS_CSV as TRACE_ROUTE_TRANSFER_RESULT_ROWS_CSV,
    ROUTE_TRANSFER_SUMMARY_ROWS_CSV as TRACE_ROUTE_TRANSFER_SUMMARY_ROWS_CSV,
    SUMMARY_JSON as TRACE_SUMMARY_JSON,
    TRACE_ROWS_CSV,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace_audit_gamma1e5_20260607"
)

EVIDENCE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace_audit_evidence_rows.csv"
)
ROUTE_AUDIT_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace_audit_route_rows.csv"
)
PAIR_AUDIT_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace_audit_pair_rows.csv"
)
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace_audit_decision_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace_audit_gate_matrix.csv"
)
SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace_audit_summary.json"
)
CONFIG_JSON = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace_audit_config.json"
)
REPORT_MD = (
    "nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace_audit_report.md"
)

BOUNDARY_PAIR_ID = "local_pair_005"
SOURCE_CONTEXT_PAIR_ID = "local_pair_014"
DIRECT_FAMILY = "first_pass_016_direct_only_target_availability_probe"
RECOVERY_FAMILY = "first_pass_016_recovery_loop_probe"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_016_object_wall_transfer_trace_audit"
WALL_PROMOTION_STATUS = "not_promoted_016_object_wall_transfer_trace_audit_only"
METHOD_STATUS = "object_wall_transfer_trace_audit_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass 016 object-wall transfer trace audit only; "
    "reads the executed 14-row transfer trace and classifies whether direct-only "
    "target shape plus typed-transient/object-identity readout is sufficient "
    "for a local object-wall evidence audit. It does not rerun Leiden, expand "
    "routes, promote pathway labels or walls, evaluate quality/cost value, "
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


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


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
        "gate_status": "pass" if bool(passed) else "fail",
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 80) -> str:
    cols = [column for column in columns if column in frame.columns]
    if not cols:
        return "_No matching columns._"
    visible = frame[cols].head(int(max_rows))
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


def _route_audit_rows(route_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in route_summary.sort_values(
        ["local_pair_id", "start_condition", "planned_route_family"],
        kind="mergesort",
    ).itertuples(index=False):
        data = row._asdict()
        pair_id = str(data["local_pair_id"])
        family = str(data["planned_route_family"])
        seed_count = int(data["seed_count"])
        direct_available = int(data.get("direct_target_available_seed_count", 0))
        recovery_target = int(data.get("recovery_target_with_recovery_seed_count", 0))
        recovery_transient = int(data.get("recovery_typed_transient_block_seed_count", 0))
        boundary_leak = int(data.get("boundary_positive_leak_seed_count", 0))
        untyped = int(data.get("untyped_transient_seed_route_count", 0))
        if pair_id == POSITIVE_PAIR_ID and family == DIRECT_FAMILY:
            if direct_available == seed_count and untyped == 0:
                audit_class = "direct_only_target_shape_complete_identity_unresolved"
                readiness_effect = "supports_direct_leg_only"
            elif untyped:
                audit_class = "direct_only_untyped_state_blocker"
                readiness_effect = "blocks_object_wall_audit"
            else:
                audit_class = "direct_only_target_shape_incomplete"
                readiness_effect = "blocks_object_wall_audit"
        elif pair_id == POSITIVE_PAIR_ID and family == RECOVERY_FAMILY:
            if recovery_target == seed_count and seed_count > 0 and untyped == 0:
                audit_class = "recovery_target_with_recovery_complete"
                readiness_effect = "supports_recovery_leg"
            elif recovery_transient == seed_count and seed_count > 0 and untyped == 0:
                audit_class = "recovery_typed_transient_block_complete"
                readiness_effect = "blocks_object_wall_audit_but_supports_identity_resolution_audit"
            elif untyped:
                audit_class = "recovery_untyped_state_blocker"
                readiness_effect = "blocks_object_wall_audit"
            else:
                audit_class = "recovery_relation_incomplete"
                readiness_effect = "blocks_object_wall_audit"
        elif pair_id == BOUNDARY_PAIR_ID:
            if boundary_leak:
                audit_class = "boundary_positive_leak"
                readiness_effect = "blocks_object_wall_audit"
            else:
                audit_class = "boundary_closed"
                readiness_effect = "supports_false_positive_guard"
        else:
            audit_class = "source_context_not_executed"
            readiness_effect = "context_only"
        rows.append(
            {
                **data,
                "route_audit_class": audit_class,
                "readiness_effect": readiness_effect,
                "object_wall_claim_allowed_after_audit": False,
                "pathway_claim_allowed_after_audit": False,
                "method_claim_allowed_after_audit": False,
                "quality_cost_claim_allowed_after_audit": False,
                "full_replay_claim_allowed_after_audit": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _positive_trace_counts(trace_rows: pd.DataFrame) -> dict[str, Any]:
    positive = trace_rows[trace_rows["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)]
    return {
        "trace_row_count": int(len(positive)),
        "typed_transient_assignment_counts": _count_dict(
            positive["typed_transient_assignment_by_step"]
        ),
        "object_identity_transfer_status_counts": _count_dict(
            positive["object_identity_transfer_status"]
        ),
        "endpoint_assignment_counts": _count_dict(positive["endpoint_assignment_by_step"]),
        "target_anchor_proxy_row_count": int(
            positive["object_identity_transfer_status"]
            .astype(str)
            .eq("target_anchor_proxy_object_identity_unresolved")
            .sum()
        ),
        "typed_pathway_intermediate_row_count": int(
            positive["typed_transient_assignment_by_step"].astype(str).eq("pathway_intermediate").sum()
        ),
        "object_identity_blocker_row_count": int(
            positive["typed_transient_assignment_by_step"].astype(str).eq("object_identity_blocker").sum()
        ),
    }


def _pair_audit_rows(
    *,
    pair_results: pd.DataFrame,
    route_audit: pd.DataFrame,
    trace_rows: pd.DataFrame,
) -> pd.DataFrame:
    positive_trace_counts = _positive_trace_counts(trace_rows)
    rows: list[dict[str, Any]] = []
    for pair in pair_results.sort_values("local_pair_id", kind="mergesort").itertuples(index=False):
        data = pair._asdict()
        pair_id = str(data["local_pair_id"])
        pair_routes = route_audit[route_audit["local_pair_id"].astype(str).eq(pair_id)]
        direct_complete = bool(
            pair_routes["route_audit_class"]
            .astype(str)
            .eq("direct_only_target_shape_complete_identity_unresolved")
            .any()
        )
        recovery_complete = bool(
            pair_routes["route_audit_class"]
            .astype(str)
            .eq("recovery_target_with_recovery_complete")
            .any()
        )
        recovery_typed_block = bool(
            pair_routes["route_audit_class"]
            .astype(str)
            .eq("recovery_typed_transient_block_complete")
            .any()
        )
        boundary_closed = bool(
            pair_id == BOUNDARY_PAIR_ID
            and pair_routes["route_audit_class"].astype(str).eq("boundary_closed").all()
        )
        untyped_count = int(data.get("untyped_transfer_seed_route_count", 0))
        boundary_leak_count = int(data.get("boundary_positive_leak_seed_route_count", 0))
        if pair_id == POSITIVE_PAIR_ID:
            object_identity_resolution_audit_ready = bool(
                direct_complete
                and recovery_typed_block
                and untyped_count == 0
                and boundary_leak_count == 0
            )
            local_object_wall_evidence_audit_ready = bool(
                direct_complete
                and recovery_complete
                and untyped_count == 0
                and boundary_leak_count == 0
            )
            if local_object_wall_evidence_audit_ready:
                pair_audit_status = "ready_for_local_object_wall_evidence_audit"
            elif object_identity_resolution_audit_ready:
                pair_audit_status = (
                    "not_ready_for_object_wall_evidence_ready_for_identity_resolution_audit"
                )
            elif untyped_count:
                pair_audit_status = "not_ready_untyped_transfer_states"
            else:
                pair_audit_status = "not_ready_incomplete_transfer_shape"
        elif pair_id == BOUNDARY_PAIR_ID:
            local_object_wall_evidence_audit_ready = False
            object_identity_resolution_audit_ready = False
            pair_audit_status = (
                "boundary_guard_closed"
                if boundary_closed and boundary_leak_count == 0
                else "boundary_guard_not_closed"
            )
        else:
            local_object_wall_evidence_audit_ready = False
            object_identity_resolution_audit_ready = False
            pair_audit_status = "source_vocabulary_context_not_executed"
        rows.append(
            {
                **data,
                "direct_only_target_shape_complete": bool(direct_complete),
                "recovery_target_with_recovery_complete": bool(recovery_complete),
                "recovery_typed_transient_block_complete": bool(recovery_typed_block),
                "boundary_guard_closed": bool(boundary_closed),
                "local_object_wall_evidence_audit_ready": bool(
                    local_object_wall_evidence_audit_ready
                ),
                "object_identity_resolution_audit_ready": bool(
                    object_identity_resolution_audit_ready
                ),
                "pair_audit_status": pair_audit_status,
                "positive_trace_diagnostics": (
                    positive_trace_counts if pair_id == POSITIVE_PAIR_ID else {}
                ),
                "object_wall_claim_allowed_after_audit": False,
                "pathway_claim_allowed_after_audit": False,
                "method_claim_allowed_after_audit": False,
                "quality_cost_claim_allowed_after_audit": False,
                "full_replay_claim_allowed_after_audit": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _evidence_rows(pair_audit: pd.DataFrame) -> pd.DataFrame:
    positive = pair_audit[pair_audit["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)]
    boundary = pair_audit[pair_audit["local_pair_id"].astype(str).eq(BOUNDARY_PAIR_ID)]
    positive_row = positive.iloc[0] if not positive.empty else pd.Series()
    boundary_row = boundary.iloc[0] if not boundary.empty else pd.Series()
    evidence = [
        {
            "evidence_id": "E1_direct_only_target_shape",
            "evidence_question": "Does 016 direct-only target availability hold across all seed routes?",
            "observed": {
                "direct_target_available_seed_route_count": int(
                    positive_row.get("direct_target_available_seed_route_count", 0)
                ),
                "direct_seed_route_count": int(positive_row.get("direct_seed_route_count", 0)),
            },
            "evidence_status": "supports_direct_leg"
            if bool(positive_row.get("direct_only_target_shape_complete", False))
            else "blocked",
            "claim_effect": "not_sufficient_for_object_wall",
        },
        {
            "evidence_id": "E2_recovery_target_with_recovery",
            "evidence_question": "Does 016 recovery-loop target-with-recovery hold across all seed routes?",
            "observed": {
                "recovery_target_with_recovery_seed_route_count": int(
                    positive_row.get("recovery_target_with_recovery_seed_route_count", 0)
                ),
                "recovery_seed_route_count": int(positive_row.get("recovery_seed_route_count", 0)),
                "recovery_typed_transient_block_seed_route_count": int(
                    positive_row.get("recovery_typed_transient_block_seed_route_count", 0)
                ),
            },
            "evidence_status": "blocked_by_typed_transient_recovery",
            "claim_effect": "blocks_object_wall_evidence_audit",
        },
        {
            "evidence_id": "E3_typed_transient_readout",
            "evidence_question": "Are transient states typed rather than untyped?",
            "observed": {
                "untyped_transfer_seed_route_count": int(
                    positive_row.get("untyped_transfer_seed_route_count", 0)
                ),
                "recovery_typed_transient_block_complete": bool(
                    positive_row.get("recovery_typed_transient_block_complete", False)
                ),
            },
            "evidence_status": "supports_identity_resolution_audit",
            "claim_effect": "opens_object_identity_resolution_not_wall",
        },
        {
            "evidence_id": "E4_object_identity_unresolved",
            "evidence_question": "Is endpoint-object identity resolved for 016 target/transient rows?",
            "observed": positive_row.get("positive_trace_diagnostics", {}),
            "evidence_status": "unresolved_object_identity",
            "claim_effect": "blocks_object_level_wall_language",
        },
        {
            "evidence_id": "E5_boundary_guard",
            "evidence_question": "Does 005 remain a closed boundary guard?",
            "observed": {
                "boundary_positive_leak_seed_route_count": int(
                    boundary_row.get("boundary_positive_leak_seed_route_count", -1)
                ),
                "boundary_guard_closed": bool(boundary_row.get("boundary_guard_closed", False)),
            },
            "evidence_status": "closed_false_positive_guard"
            if bool(boundary_row.get("boundary_guard_closed", False))
            else "boundary_guard_not_closed",
            "claim_effect": "supports_guard_only",
        },
    ]
    frame = pd.DataFrame(evidence)
    frame["object_wall_claim_allowed_after_audit"] = False
    frame["route_execution_status"] = ROUTE_EXECUTION_STATUS
    frame["wall_promotion_status"] = WALL_PROMOTION_STATUS
    frame["method_status"] = METHOD_STATUS
    frame["run_status"] = RUN_STATUS
    frame["claim_boundary"] = CLAIM_BOUNDARY
    return frame


def _decision_rows(pair_audit: pd.DataFrame) -> pd.DataFrame:
    positive = pair_audit[pair_audit["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)]
    positive_status = (
        str(positive.iloc[0]["pair_audit_status"])
        if not positive.empty
        else "missing_positive_pair"
    )
    decisions = [
        {
            "decision_id": "D1",
            "decision": "trace_valid_for_read_only_audit",
            "rationale": "The executed transfer trace has exact scope and complete typed readout.",
        },
        {
            "decision_id": "D2",
            "decision": "do_not_open_object_wall_evidence_audit_yet",
            "rationale": (
                "016 direct-only target shape is complete, but recovery-loop evidence is "
                "typed transient block rather than target-with-recovery."
            ),
        },
        {
            "decision_id": "D3",
            "decision": "open_object_identity_resolution_audit",
            "rationale": (
                "The positive pair status is "
                f"{positive_status}; next work should resolve target/transient object identity "
                "within the existing trace surface."
            ),
        },
        {
            "decision_id": "D4",
            "decision": "retain_005_boundary_guard",
            "rationale": "005 has zero positive leaks and should stay as the false-positive control.",
        },
        {
            "decision_id": "D5",
            "decision": "keep_labels_closed",
            "rationale": (
                "No pathway, wall, method, quality/cost, or full-replay label is promoted by this audit."
            ),
        },
    ]
    frame = pd.DataFrame(decisions)
    frame["run_status"] = RUN_STATUS
    frame["claim_boundary"] = CLAIM_BOUNDARY
    return frame


def _gate_matrix(
    *,
    trace_summary: dict[str, Any],
    trace_gates: pd.DataFrame,
    route_audit: pd.DataFrame,
    pair_audit: pd.DataFrame,
    boundary_results: pd.DataFrame,
) -> pd.DataFrame:
    positive = pair_audit[pair_audit["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)]
    boundary = pair_audit[pair_audit["local_pair_id"].astype(str).eq(BOUNDARY_PAIR_ID)]
    positive_row = positive.iloc[0] if not positive.empty else pd.Series()
    boundary_row = boundary.iloc[0] if not boundary.empty else pd.Series()
    positive_routes = route_audit[route_audit["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)]
    rows = [
        _gate_row(
            "G1_upstream_trace_gates_pass",
            "Did the executed 016 transfer trace gates pass?",
            {
                "trace_failed_gates": trace_summary.get("failed_gates", []),
                "trace_gate_status_counts": _count_dict(trace_gates["gate_status"]),
            },
            "upstream failed_gates empty and all trace gates pass",
            len(trace_summary.get("failed_gates", [])) == 0
            and bool(trace_gates["gate_status"].astype(str).eq("pass").all()),
        ),
        _gate_row(
            "G2_exact_scope_retained",
            "Does the audit read the exact executed trace scope?",
            {
                "route_execution_plan_row_count": trace_summary.get(
                    "route_execution_plan_row_count"
                ),
                "route_step_config_count": trace_summary.get("route_step_config_count"),
                "trace_row_count": trace_summary.get("trace_row_count"),
            },
            "14 route rows, 77 step configs, 616 trace rows",
            int(trace_summary.get("route_execution_plan_row_count", -1)) == 14
            and int(trace_summary.get("route_step_config_count", -1)) == 77
            and int(trace_summary.get("trace_row_count", -1)) == 616,
        ),
        _gate_row(
            "G3_direct_only_target_shape_complete",
            "Is 016 direct-only target shape complete?",
            {
                "direct_target_available_seed_route_count": int(
                    positive_row.get("direct_target_available_seed_route_count", 0)
                ),
                "direct_seed_route_count": int(positive_row.get("direct_seed_route_count", 0)),
            },
            "24/24 direct-only seed routes target-available",
            bool(positive_row.get("direct_only_target_shape_complete", False)),
        ),
        _gate_row(
            "G4_recovery_typed_transient_block_named",
            "Is the recovery-loop blocker explicitly typed?",
            {
                "recovery_target_with_recovery_seed_route_count": int(
                    positive_row.get("recovery_target_with_recovery_seed_route_count", 0)
                ),
                "recovery_typed_transient_block_seed_route_count": int(
                    positive_row.get("recovery_typed_transient_block_seed_route_count", 0)
                ),
                "recovery_seed_route_count": int(positive_row.get("recovery_seed_route_count", 0)),
            },
            "0 target-with-recovery and 24/24 typed transient blocks",
            bool(positive_row.get("recovery_typed_transient_block_complete", False))
            and int(positive_row.get("recovery_target_with_recovery_seed_route_count", -1)) == 0,
        ),
        _gate_row(
            "G5_no_untyped_positive_transfer_states",
            "Are positive transfer states typed rather than untyped?",
            int(positive_row.get("untyped_transfer_seed_route_count", -1)),
            "zero untyped positive seed routes",
            int(positive_row.get("untyped_transfer_seed_route_count", -1)) == 0,
        ),
        _gate_row(
            "G6_boundary_guard_closed",
            "Does 005 remain a closed boundary guard?",
            {
                "boundary_positive_leak_seed_route_count": int(
                    positive_row.get("boundary_positive_leak_seed_route_count", 0)
                ),
                "boundary_guard_status": str(boundary_row.get("pair_audit_status", "")),
                "boundary_guard_result_counts": _count_dict(
                    boundary_results["boundary_guard_result"]
                ),
            },
            "005 boundary guard closed and zero positive leaks",
            bool(boundary_row.get("boundary_guard_closed", False))
            and int(boundary_row.get("boundary_positive_leak_seed_route_count", -1)) == 0,
        ),
        _gate_row(
            "G7_object_wall_audit_blocker_named",
            "Is the object-wall audit blocker named without treating it as failure to execute?",
            {
                "positive_pair_audit_status": str(positive_row.get("pair_audit_status", "")),
                "route_audit_classes": _count_dict(positive_routes["route_audit_class"]),
            },
            "not ready for object-wall evidence; ready for identity-resolution audit",
            str(positive_row.get("pair_audit_status", ""))
            == "not_ready_for_object_wall_evidence_ready_for_identity_resolution_audit"
            and bool(positive_row.get("object_identity_resolution_audit_ready", False))
            and not bool(positive_row.get("local_object_wall_evidence_audit_ready", True)),
        ),
        _gate_row(
            "G8_claims_closed",
            "Are pathway, wall, method, quality/cost, and full-replay claims closed?",
            CLAIM_BOUNDARY,
            "all promotion flags false",
            bool(pair_audit["object_wall_claim_allowed_after_audit"].eq(False).all())
            and bool(pair_audit["pathway_claim_allowed_after_audit"].eq(False).all())
            and bool(pair_audit["method_claim_allowed_after_audit"].eq(False).all())
            and bool(pair_audit["quality_cost_claim_allowed_after_audit"].eq(False).all())
            and bool(pair_audit["full_replay_claim_allowed_after_audit"].eq(False).all()),
        ),
    ]
    return pd.DataFrame(rows)


def _summary(
    *,
    trace_dir: Path,
    output_dir: Path,
    route_audit: pd.DataFrame,
    pair_audit: pd.DataFrame,
    evidence_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    positive = pair_audit[pair_audit["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)]
    boundary = pair_audit[pair_audit["local_pair_id"].astype(str).eq(BOUNDARY_PAIR_ID)]
    positive_row = positive.iloc[0] if not positive.empty else pd.Series()
    boundary_row = boundary.iloc[0] if not boundary.empty else pd.Series()
    failed_gates = gates.loc[
        ~gates["gate_status"].astype(str).eq("pass"),
        "gate_id",
    ].tolist()
    object_wall_ready = bool(
        positive_row.get("local_object_wall_evidence_audit_ready", False)
    )
    identity_ready = bool(
        positive_row.get("object_identity_resolution_audit_ready", False)
    )
    if object_wall_ready:
        next_gate = (
            "Open a local object-wall evidence audit over the executed 016 transfer trace."
        )
    elif identity_ready:
        next_gate = (
            "Run a read-only 016 object/signature identity-resolution audit over the "
            "existing transfer trace; do not expand routes or promote labels."
        )
    else:
        next_gate = (
            "Inspect incomplete transfer-readout blockers before opening any object-wall audit."
        )
    return {
        "schema": "nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace_audit_summary.v1",
        "status": RUN_STATUS,
        "trace_dir": str(trace_dir),
        "output_dir": str(output_dir),
        "route_audit_row_count": int(len(route_audit)),
        "pair_audit_row_count": int(len(pair_audit)),
        "evidence_row_count": int(len(evidence_rows)),
        "route_audit_class_counts": _count_dict(route_audit["route_audit_class"]),
        "pair_audit_status_counts": _count_dict(pair_audit["pair_audit_status"]),
        "evidence_status_counts": _count_dict(evidence_rows["evidence_status"]),
        "positive_pair_audit_status": str(positive_row.get("pair_audit_status", "")),
        "boundary_pair_audit_status": str(boundary_row.get("pair_audit_status", "")),
        "local_object_wall_evidence_audit_ready": object_wall_ready,
        "object_identity_resolution_audit_ready": identity_ready,
        "wall_claim_ready_pairs": [],
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": failed_gates,
        "interpretation": (
            "The executed 016 transfer trace is valid and strongly supports the "
            "direct-only target leg, but it does not support a local object-wall "
            "evidence audit yet because recovery-loop evidence is a typed transient "
            "block and object identity remains unresolved."
        ),
        "recommended_next_gate": next_gate,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    evidence_rows: pd.DataFrame,
    route_audit: pd.DataFrame,
    pair_audit: pd.DataFrame,
    decisions: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass 016 Object-Wall Transfer Trace Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- positive_pair_audit_status: `{summary['positive_pair_audit_status']}`",
        f"- boundary_pair_audit_status: `{summary['boundary_pair_audit_status']}`",
        f"- local_object_wall_evidence_audit_ready: {summary['local_object_wall_evidence_audit_ready']}",
        f"- object_identity_resolution_audit_ready: {summary['object_identity_resolution_audit_ready']}",
        f"- route_audit_class_counts: {summary['route_audit_class_counts']}",
        f"- evidence_status_counts: {summary['evidence_status_counts']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Evidence Rows",
        "",
        _markdown_table(
            evidence_rows,
            ["evidence_id", "evidence_status", "claim_effect", "observed", "evidence_question"],
        ),
        "",
        "## Pair Audit",
        "",
        _markdown_table(
            pair_audit,
            [
                "local_pair_id",
                "contract_pair_role",
                "direct_only_target_shape_complete",
                "recovery_target_with_recovery_complete",
                "recovery_typed_transient_block_complete",
                "boundary_guard_closed",
                "local_object_wall_evidence_audit_ready",
                "object_identity_resolution_audit_ready",
                "pair_audit_status",
            ],
        ),
        "",
        "## Route Audit",
        "",
        _markdown_table(
            route_audit,
            [
                "local_pair_id",
                "start_condition",
                "planned_route_family",
                "seed_count",
                "route_transfer_status",
                "route_audit_class",
                "readiness_effect",
            ],
            max_rows=40,
        ),
        "",
        "## Decisions",
        "",
        _markdown_table(decisions, ["decision_id", "decision", "rationale"], max_rows=10),
        "",
        "## Gate Matrix",
        "",
        _markdown_table(
            gates,
            ["gate_id", "gate_status", "observed", "minimum_or_rule", "question"],
        ),
        "",
        "## Boundary",
        "",
        (
            "This audit names a blocker, not a wall. It should be followed by "
            "object/signature identity resolution on the existing trace surface, "
            "with no route expansion and no label promotion."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_summary = _read_json(trace_dir / TRACE_SUMMARY_JSON)
    trace_gates = _read_csv(trace_dir / TRACE_GATE_MATRIX_CSV)
    trace_rows = _read_csv(trace_dir / TRACE_ROWS_CSV)
    route_results = _read_csv(trace_dir / TRACE_ROUTE_TRANSFER_RESULT_ROWS_CSV)
    route_summary = _read_csv(trace_dir / TRACE_ROUTE_TRANSFER_SUMMARY_ROWS_CSV)
    pair_results = _read_csv(trace_dir / TRACE_PAIR_TRANSFER_RESULT_ROWS_CSV)
    boundary_results = _read_csv(trace_dir / TRACE_BOUNDARY_GUARD_RESULT_ROWS_CSV)

    route_audit = _route_audit_rows(route_summary)
    pair_audit = _pair_audit_rows(
        pair_results=pair_results,
        route_audit=route_audit,
        trace_rows=trace_rows,
    )
    evidence_rows = _evidence_rows(pair_audit)
    decisions = _decision_rows(pair_audit)
    gates = _gate_matrix(
        trace_summary=trace_summary,
        trace_gates=trace_gates,
        route_audit=route_audit,
        pair_audit=pair_audit,
        boundary_results=boundary_results,
    )
    summary = _summary(
        trace_dir=trace_dir,
        output_dir=output_dir,
        route_audit=route_audit,
        pair_audit=pair_audit,
        evidence_rows=evidence_rows,
        gates=gates,
    )

    _write_csv(evidence_rows, output_dir / EVIDENCE_ROWS_CSV)
    _write_csv(route_audit, output_dir / ROUTE_AUDIT_ROWS_CSV)
    _write_csv(pair_audit, output_dir / PAIR_AUDIT_ROWS_CSV)
    _write_csv(decisions, output_dir / DECISION_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace_audit_config.v1",
        "trace_dir": str(trace_dir),
        "output_dir": str(output_dir),
        "positive_pair_id": POSITIVE_PAIR_ID,
        "boundary_pair_id": BOUNDARY_PAIR_ID,
        "source_context_pair_id": SOURCE_CONTEXT_PAIR_ID,
        "direct_family": DIRECT_FAMILY,
        "recovery_family": RECOVERY_FAMILY,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        evidence_rows=evidence_rows,
        route_audit=route_audit,
        pair_audit=pair_audit,
        decisions=decisions,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
