#!/usr/bin/env python3
"""Audit primitive wall evidence for the first-pass local_pair_014 probe.

This consumes the executed first-pass 014 pathway-probe trace and asks whether
the accepted direct-only and recovery-loop routes support a local, object-level
primitive wall-evidence claim for ``local_pair_014``. The audit deliberately
keeps method, quality/cost, and full NanoClustering replay claims closed.

The wall-evidence unit is stricter than a single route result: for the same
start condition and seed, the direct-only route must make the exclusive target
available while bridge support is suppressed, and the recovery-loop route must
move from a source-like endpoint object to the exclusive target object and back
to the source-like endpoint object after bridge support is restored. The
``local_pair_005`` boundary control must not leak under matched route families.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_leiden_basin_nanoclustering_g4_8_first_pass_014_pathway_probe_trace import (
    CONTROL_GUARD_RESULT_ROWS_CSV as TRACE_CONTROL_GUARD_RESULT_ROWS_CSV,
    DEFAULT_OUTPUT_DIR as DEFAULT_TRACE_DIR,
    GATE_MATRIX_CSV as TRACE_GATE_MATRIX_CSV,
    PAIR_PROBE_RESULT_ROWS_CSV as TRACE_PAIR_PROBE_RESULT_ROWS_CSV,
    POSITIVE_PAIR_ID,
    ROUTE_PROBE_RESULT_ROWS_CSV as TRACE_ROUTE_PROBE_RESULT_ROWS_CSV,
    ROUTE_PROBE_SUMMARY_ROWS_CSV as TRACE_ROUTE_PROBE_SUMMARY_ROWS_CSV,
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
    / "leiden_basin_nanoclustering_g4_8_first_pass_014_wall_evidence_audit_gamma1e5_20260604"
)

SEED_WALL_ROWS_CSV = "nanoclustering_g4_8_first_pass_014_wall_evidence_seed_rows.csv"
BOUNDARY_GUARD_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_wall_evidence_boundary_guard_rows.csv"
)
PAIR_WALL_ROWS_CSV = "nanoclustering_g4_8_first_pass_014_wall_evidence_pair_rows.csv"
GATE_MATRIX_CSV = "nanoclustering_g4_8_first_pass_014_wall_evidence_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_014_wall_evidence_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_014_wall_evidence_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_014_wall_evidence_report.md"

BOUNDARY_PAIR_ID = "local_pair_005"
DIRECT_FAMILY = "first_pass_014_direct_only_target_availability_probe"
RECOVERY_FAMILY = "first_pass_014_recovery_loop_probe"
BOUNDARY_DIRECT_FAMILY = "first_pass_005_boundary_direct_only_guard"
BOUNDARY_RECOVERY_FAMILY = "first_pass_005_boundary_recovery_loop_guard"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_014_wall_evidence"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass 014 primitive wall-evidence audit only; "
    "reads accepted local direct-only and recovery-loop probe routes. It may "
    "classify local object-level wall evidence for local_pair_014, but it does "
    "not evaluate quality/cost value, replay full NanoClustering, or claim "
    "method/algorithm success."
)

SOURCE_OBJECTS = {"source_endpoint_object", "source_like_endpoint_object"}
TARGET_OBJECT = "exclusive_target_endpoint_object"


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


def _source_like(value: Any) -> bool:
    return str(value) in SOURCE_OBJECTS


def _object_sequence_parts(value: Any) -> list[str]:
    return [part.strip() for part in str(value).split(" -> ") if part.strip()]


def _seed_wall_rows(route_results: pd.DataFrame) -> pd.DataFrame:
    positive = route_results[route_results["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)]
    direct = positive[positive["planned_route_family"].astype(str).eq(DIRECT_FAMILY)].copy()
    recovery = positive[positive["planned_route_family"].astype(str).eq(RECOVERY_FAMILY)].copy()
    key_cols = ["local_pair_id", "branch", "start_condition", "seed"]
    rows = direct.merge(
        recovery,
        on=key_cols,
        suffixes=("_direct", "_recovery"),
        how="outer",
        validate="one_to_one",
    )
    output: list[dict[str, Any]] = []
    for row in rows.sort_values(["start_condition", "seed"], kind="mergesort").itertuples(index=False):
        data = row._asdict()
        direct_sequence = _object_sequence_parts(data.get("endpoint_object_assignment_sequence_direct", ""))
        recovery_sequence = _object_sequence_parts(
            data.get("endpoint_object_assignment_sequence_recovery", "")
        )
        direct_source_object = direct_sequence[0] if direct_sequence else ""
        direct_target_object = direct_sequence[-1] if direct_sequence else ""
        recovery_start_object = recovery_sequence[0] if recovery_sequence else ""
        recovery_final_object = recovery_sequence[-1] if recovery_sequence else ""
        recovery_interior = recovery_sequence[1:-1] if len(recovery_sequence) > 2 else []
        direct_path_accepted = _as_bool(data.get("direct_path_accepted_seed_direct", False))
        recovery_accepted = _as_bool(data.get("recovery_accepted_seed_recovery", False))
        direct_relation_pass = _source_like(direct_source_object) and direct_target_object == TARGET_OBJECT
        recovery_relation_pass = (
            _source_like(recovery_start_object)
            and bool(recovery_interior)
            and all(value == TARGET_OBJECT for value in recovery_interior)
            and _source_like(recovery_final_object)
        )
        objective_wall_shape_pass = bool(
            float(data.get("max_objective_debt_from_start_recovery", 0.0)) > 0.0
            and float(data.get("max_objective_recovery_from_min_recovery", 0.0)) > 0.0
            and _as_bool(data.get("accepted_recovery_after_min_recovery", False))
        )
        no_unknown_or_unresolved_object = bool(
            int(data.get("unknown_step_count_direct", 0)) == 0
            and int(data.get("unknown_step_count_recovery", 0)) == 0
            and int(data.get("ambiguous_step_count_direct", 0)) == 0
            and int(data.get("ambiguous_step_count_recovery", 0)) == 0
        )
        no_support_incompatibility = bool(
            int(data.get("support_incompatibility_step_count_direct", 0)) == 0
            and int(data.get("support_incompatibility_step_count_recovery", 0)) == 0
        )
        raw_anchor_ambiguity_resolved_to_object = bool(
            int(data.get("raw_anchor_ambiguous_step_count_direct", 0)) >= 0
            and int(data.get("raw_anchor_ambiguous_step_count_recovery", 0)) >= 0
            and no_unknown_or_unresolved_object
        )
        wall_seed_ready = all(
            [
                direct_path_accepted,
                recovery_accepted,
                direct_relation_pass,
                recovery_relation_pass,
                objective_wall_shape_pass,
                no_unknown_or_unresolved_object,
                no_support_incompatibility,
            ]
        )
        if wall_seed_ready:
            wall_seed_status = "primitive_object_wall_seed_ready"
        elif not direct_path_accepted:
            wall_seed_status = "direct_path_not_accepted"
        elif not recovery_accepted:
            wall_seed_status = "recovery_loop_not_accepted"
        elif not direct_relation_pass:
            wall_seed_status = "direct_relation_not_source_to_target_object"
        elif not recovery_relation_pass:
            wall_seed_status = "recovery_relation_not_reversible_source_target_source_object"
        elif not objective_wall_shape_pass:
            wall_seed_status = "objective_wall_shape_not_accepted"
        elif not no_unknown_or_unresolved_object:
            wall_seed_status = "unknown_or_unresolved_object_endpoint"
        elif not no_support_incompatibility:
            wall_seed_status = "support_incompatibility_observed"
        else:
            wall_seed_status = "unclassified_wall_seed_reject"
        output.append(
            {
                "wall_seed_id": (
                    f"{data.get('local_pair_id', POSITIVE_PAIR_ID)}__"
                    f"{data.get('start_condition', '')}__seed{int(data.get('seed', -1)):02d}"
                ),
                "local_pair_id": str(data.get("local_pair_id", POSITIVE_PAIR_ID)),
                "branch": str(data.get("branch", "")),
                "start_condition": str(data.get("start_condition", "")),
                "seed": int(data.get("seed", -1)),
                "direct_route_contract_id": str(data.get("route_contract_id_direct", "")),
                "recovery_route_contract_id": str(data.get("route_contract_id_recovery", "")),
                "direct_path_accepted_seed": bool(direct_path_accepted),
                "recovery_accepted_seed": bool(recovery_accepted),
                "direct_source_object": direct_source_object,
                "direct_target_object": direct_target_object,
                "recovery_start_object": recovery_start_object,
                "recovery_interior_target_object_count": int(
                    sum(value == TARGET_OBJECT for value in recovery_interior)
                ),
                "recovery_interior_step_count": int(len(recovery_interior)),
                "recovery_final_object": recovery_final_object,
                "direct_object_relation_pass": bool(direct_relation_pass),
                "recovery_reversible_object_relation_pass": bool(recovery_relation_pass),
                "objective_wall_shape_pass": bool(objective_wall_shape_pass),
                "no_unknown_or_unresolved_object": bool(no_unknown_or_unresolved_object),
                "no_support_incompatibility": bool(no_support_incompatibility),
                "raw_anchor_ambiguity_step_count": int(
                    int(data.get("raw_anchor_ambiguous_step_count_direct", 0))
                    + int(data.get("raw_anchor_ambiguous_step_count_recovery", 0))
                ),
                "raw_anchor_ambiguity_resolved_to_object": bool(
                    raw_anchor_ambiguity_resolved_to_object
                ),
                "direct_bridge_fraction_sequence": str(
                    data.get("bridge_fraction_sequence_direct", "")
                ),
                "recovery_bridge_fraction_sequence": str(
                    data.get("bridge_fraction_sequence_recovery", "")
                ),
                "max_objective_debt_from_start_recovery": float(
                    data.get("max_objective_debt_from_start_recovery", 0.0)
                ),
                "max_objective_recovery_from_min_recovery": float(
                    data.get("max_objective_recovery_from_min_recovery", 0.0)
                ),
                "wall_seed_ready": bool(wall_seed_ready),
                "wall_seed_status": wall_seed_status,
                "wall_evidence_scope": "local_object_level_seed_start_condition",
                "method_claim_allowed_after_audit": False,
                "quality_cost_claim_allowed_after_audit": False,
                "full_replay_claim_allowed_after_audit": False,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(output)


def _boundary_guard_rows(route_results: pd.DataFrame) -> pd.DataFrame:
    boundary = route_results[route_results["local_pair_id"].astype(str).eq(BOUNDARY_PAIR_ID)]
    direct = boundary[
        boundary["planned_route_family"].astype(str).eq(BOUNDARY_DIRECT_FAMILY)
    ].copy()
    recovery = boundary[
        boundary["planned_route_family"].astype(str).eq(BOUNDARY_RECOVERY_FAMILY)
    ].copy()
    key_cols = ["local_pair_id", "branch", "start_condition", "seed"]
    rows = direct.merge(
        recovery,
        on=key_cols,
        suffixes=("_direct", "_recovery"),
        how="outer",
        validate="one_to_one",
    )
    output: list[dict[str, Any]] = []
    for row in rows.sort_values(["start_condition", "seed"], kind="mergesort").itertuples(index=False):
        data = row._asdict()
        direct_leak = _as_bool(data.get("boundary_positive_leak_observed_direct", False))
        recovery_leak = _as_bool(data.get("boundary_positive_leak_observed_recovery", False))
        guard_closed = not direct_leak and not recovery_leak
        output.append(
            {
                "boundary_guard_seed_id": (
                    f"{data.get('local_pair_id', BOUNDARY_PAIR_ID)}__"
                    f"{data.get('start_condition', '')}__seed{int(data.get('seed', -1)):02d}"
                ),
                "local_pair_id": str(data.get("local_pair_id", BOUNDARY_PAIR_ID)),
                "branch": str(data.get("branch", "")),
                "start_condition": str(data.get("start_condition", "")),
                "seed": int(data.get("seed", -1)),
                "direct_boundary_leak_observed": bool(direct_leak),
                "recovery_boundary_leak_observed": bool(recovery_leak),
                "boundary_guard_closed": bool(guard_closed),
                "boundary_guard_status": "closed" if guard_closed else "positive_leak_observed",
                "direct_route_outcome_class": str(data.get("route_probe_outcome_class_direct", "")),
                "recovery_route_outcome_class": str(
                    data.get("route_probe_outcome_class_recovery", "")
                ),
                "method_claim_allowed_after_audit": False,
                "quality_cost_claim_allowed_after_audit": False,
                "full_replay_claim_allowed_after_audit": False,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(output)


def _pair_wall_rows(seed_wall: pd.DataFrame, boundary_guards: pd.DataFrame) -> pd.DataFrame:
    wall_seed_count = int(len(seed_wall))
    wall_ready_seed_count = int(seed_wall["wall_seed_ready"].map(_as_bool).sum())
    boundary_seed_count = int(len(boundary_guards))
    boundary_closed_count = int(boundary_guards["boundary_guard_closed"].map(_as_bool).sum())
    wall_ready = bool(
        wall_seed_count == 32
        and wall_ready_seed_count == wall_seed_count
        and boundary_seed_count == 32
        and boundary_closed_count == boundary_seed_count
    )
    status = (
        "primitive_object_level_wall_evidence_ready_local_only"
        if wall_ready
        else "primitive_object_level_wall_evidence_not_ready"
    )
    return pd.DataFrame(
        [
            {
                "local_pair_id": POSITIVE_PAIR_ID,
                "boundary_pair_id": BOUNDARY_PAIR_ID,
                "wall_seed_count": wall_seed_count,
                "wall_ready_seed_count": wall_ready_seed_count,
                "boundary_guard_seed_count": boundary_seed_count,
                "boundary_guard_closed_seed_count": boundary_closed_count,
                "wall_seed_status_counts": _count_dict(seed_wall["wall_seed_status"]),
                "boundary_guard_status_counts": _count_dict(
                    boundary_guards["boundary_guard_status"]
                ),
                "primitive_wall_evidence_ready": bool(wall_ready),
                "primitive_wall_evidence_status": status,
                "wall_evidence_scope": "local_object_level_first_pass_014_only",
                "wall_claim_limitations": (
                    "local object-level evidence only; no generalization, no exact "
                    "wall-location localization beyond the coarse schedules, no "
                    "quality/cost value, no full replay, and no method claim"
                ),
                "method_claim_allowed_after_audit": False,
                "quality_cost_claim_allowed_after_audit": False,
                "full_replay_claim_allowed_after_audit": False,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ]
    )


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
        "observed": observed,
        "minimum_or_rule": minimum_or_rule,
        "gate_status": "pass" if bool(passed) else "fail",
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _gate_matrix(
    *,
    trace_gates: pd.DataFrame,
    seed_wall: pd.DataFrame,
    boundary_guards: pd.DataFrame,
    pair_wall: pd.DataFrame,
    route_results: pd.DataFrame,
    trace_rows: pd.DataFrame,
) -> pd.DataFrame:
    pair = pair_wall.iloc[0] if not pair_wall.empty else {}
    wall_seed_count = int(pair.get("wall_seed_count", 0))
    wall_ready_seed_count = int(pair.get("wall_ready_seed_count", 0))
    boundary_seed_count = int(pair.get("boundary_guard_seed_count", 0))
    boundary_closed_seed_count = int(pair.get("boundary_guard_closed_seed_count", 0))
    positive_trace = trace_rows[trace_rows["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)]
    target_steps = positive_trace[
        positive_trace["endpoint_object_assignment_by_step"].astype(str).eq(TARGET_OBJECT)
    ]
    source_like_steps = positive_trace[
        positive_trace["endpoint_object_assignment_by_step"].astype(str).isin(SOURCE_OBJECTS)
    ]
    raw_anchor_ambiguity_count = int(
        route_results[
            route_results["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)
        ]["raw_anchor_ambiguous_step_count"]
        .astype(int)
        .sum()
    )
    rows = [
        _gate_row(
            "G1_upstream_pathway_probe_trace_gates_pass",
            "Did every upstream 014 pathway-probe trace gate pass?",
            _count_dict(trace_gates["gate_status"]),
            "all upstream trace gates pass",
            bool(trace_gates["gate_status"].astype(str).eq("pass").all()),
        ),
        _gate_row(
            "G2_same_seed_direct_and_recovery_units_materialized",
            "Were same-start same-seed direct/recovery wall units materialized for 014?",
            f"wall_seed_rows={wall_seed_count}",
            "4 starts * 8 seeds = 32 wall seed rows",
            wall_seed_count == 32,
        ),
        _gate_row(
            "G3_direct_path_object_relation_all_seed_units",
            "Does every 014 wall unit have direct-only source-like to exclusive-target object evidence?",
            f"direct_relation_pass={int(seed_wall['direct_object_relation_pass'].sum())} of {wall_seed_count}",
            "32 of 32 direct object relations pass",
            wall_seed_count == 32
            and bool(seed_wall["direct_object_relation_pass"].map(_as_bool).all()),
        ),
        _gate_row(
            "G4_recovery_loop_reversible_object_relation_all_seed_units",
            "Does every 014 wall unit have source-like to target to source-like recovery-loop evidence?",
            f"recovery_relation_pass={int(seed_wall['recovery_reversible_object_relation_pass'].sum())} of {wall_seed_count}",
            "32 of 32 recovery object relations pass",
            wall_seed_count == 32
            and bool(seed_wall["recovery_reversible_object_relation_pass"].map(_as_bool).all()),
        ),
        _gate_row(
            "G5_objective_debt_recovery_shape_all_seed_units",
            "Does every 014 wall unit show objective debt and accepted recovery after the minimum?",
            f"objective_wall_shape_pass={int(seed_wall['objective_wall_shape_pass'].sum())} of {wall_seed_count}",
            "32 of 32 objective wall-shape checks pass",
            wall_seed_count == 32
            and bool(seed_wall["objective_wall_shape_pass"].map(_as_bool).all()),
        ),
        _gate_row(
            "G6_boundary_guard_no_positive_leak",
            "Does 005 remain a closed false-positive guard under matched direct/recovery probes?",
            f"boundary_closed={boundary_closed_seed_count} of {boundary_seed_count}",
            "32 of 32 boundary guard seed units closed",
            boundary_seed_count == 32
            and boundary_closed_seed_count == boundary_seed_count,
        ),
        _gate_row(
            "G7_endpoint_object_readout_separates_exact_anchor_ambiguity",
            "Is exact-anchor ambiguity resolved as source-like object evidence rather than hidden target failure?",
            {
                "raw_anchor_ambiguity_step_count": raw_anchor_ambiguity_count,
                "source_like_object_steps": int(len(source_like_steps)),
                "target_object_steps": int(len(target_steps)),
                "unresolved_wall_seed_count": int(
                    (~seed_wall["no_unknown_or_unresolved_object"].map(_as_bool)).sum()
                ),
            },
            "raw ambiguity may exist, but zero unresolved object-level wall units",
            raw_anchor_ambiguity_count > 0
            and bool(seed_wall["no_unknown_or_unresolved_object"].map(_as_bool).all()),
        ),
        _gate_row(
            "G8_primitive_wall_evidence_ready_local_only",
            "Is primitive local object-level wall evidence ready for 014?",
            f"wall_ready_seed_units={wall_ready_seed_count} of {wall_seed_count}",
            "32 of 32 wall seed units ready and boundary closed",
            bool(pair.get("primitive_wall_evidence_ready", False)),
        ),
        _gate_row(
            "G9_method_quality_full_replay_claims_closed",
            "Are method, quality/cost, and full-replay claims still closed?",
            CLAIM_BOUNDARY,
            "all non-wall-evidence claims remain false",
            bool(pair_wall["method_claim_allowed_after_audit"].eq(False).all())
            and bool(pair_wall["quality_cost_claim_allowed_after_audit"].eq(False).all())
            and bool(pair_wall["full_replay_claim_allowed_after_audit"].eq(False).all()),
        ),
    ]
    return pd.DataFrame(rows)


def _summary(
    *,
    trace_dir: Path,
    output_dir: Path,
    seed_wall: pd.DataFrame,
    boundary_guards: pd.DataFrame,
    pair_wall: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    pair = pair_wall.iloc[0].to_dict() if not pair_wall.empty else {}
    wall_ready = bool(pair.get("primitive_wall_evidence_ready", False))
    return {
        "schema": "nanoclustering_g4_8_first_pass_014_wall_evidence_summary.v1",
        "status": RUN_STATUS,
        "trace_dir": str(trace_dir),
        "output_dir": str(output_dir),
        "wall_seed_row_count": int(len(seed_wall)),
        "wall_ready_seed_count": int(seed_wall["wall_seed_ready"].map(_as_bool).sum()),
        "boundary_guard_seed_row_count": int(len(boundary_guards)),
        "boundary_guard_closed_seed_count": int(
            boundary_guards["boundary_guard_closed"].map(_as_bool).sum()
        ),
        "primitive_wall_evidence_ready_pair_count": int(wall_ready),
        "primitive_wall_evidence_ready_pairs": [POSITIVE_PAIR_ID] if wall_ready else [],
        "wall_seed_status_counts": _count_dict(seed_wall["wall_seed_status"]),
        "boundary_guard_status_counts": _count_dict(boundary_guards["boundary_guard_status"]),
        "pair_wall_evidence_status_counts": _count_dict(
            pair_wall["primitive_wall_evidence_status"]
        ),
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "interpretation": (
            "local_pair_014 satisfies the primitive local object-level wall "
            "evidence definition: same-start same-seed direct-only routes make "
            "the exclusive target object available, recovery-loop routes move "
            "source-like to target to source-like with objective debt/recovery, "
            "and local_pair_005 remains a closed boundary guard. This is not a "
            "method, quality/cost, full-replay, or generalization claim."
        )
        if wall_ready
        else (
            "local_pair_014 does not yet satisfy the primitive local object-level "
            "wall evidence definition under the predeclared direct/recovery and "
            "boundary guard checks."
        ),
        "recommended_next_gate": (
            "Stress the primitive wall evidence beyond this one pair: repeat the "
            "same paired direct/recovery wall audit over additional clean object "
            "candidates or create a synthetic demo that reproduces the same "
            "object-level wall relation under Leiden+CPM."
        )
        if wall_ready
        else "Inspect failed wall seed units before any broader wall-evidence claim.",
        "claim_boundary": CLAIM_BOUNDARY,
        "written_artifacts": [
            SEED_WALL_ROWS_CSV,
            BOUNDARY_GUARD_ROWS_CSV,
            PAIR_WALL_ROWS_CSV,
            GATE_MATRIX_CSV,
            SUMMARY_JSON,
            CONFIG_JSON,
            REPORT_MD,
        ],
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 60) -> str:
    cols = [col for col in columns if col in frame.columns]
    if not cols:
        return "No columns."
    visible = frame[cols].head(int(max_rows))
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    rows: list[str] = []
    for row in visible.itertuples(index=False):
        values: list[str] = []
        for value in row:
            if isinstance(value, (dict, list, tuple, set)):
                values.append(json.dumps(_json_safe(value), sort_keys=True))
            elif pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value).replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    pair_wall: pd.DataFrame,
    seed_wall: pd.DataFrame,
    boundary_guards: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass 014 Wall-Evidence Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- wall_seed_row_count: {summary['wall_seed_row_count']}",
        f"- wall_ready_seed_count: {summary['wall_ready_seed_count']}",
        f"- boundary_guard_seed_row_count: {summary['boundary_guard_seed_row_count']}",
        f"- boundary_guard_closed_seed_count: {summary['boundary_guard_closed_seed_count']}",
        f"- primitive_wall_evidence_ready_pairs: {summary['primitive_wall_evidence_ready_pairs']}",
        f"- wall_seed_status_counts: {summary['wall_seed_status_counts']}",
        f"- boundary_guard_status_counts: {summary['boundary_guard_status_counts']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Pair Wall Evidence",
        "",
        _markdown_table(
            pair_wall,
            [
                "local_pair_id",
                "boundary_pair_id",
                "wall_seed_count",
                "wall_ready_seed_count",
                "boundary_guard_seed_count",
                "boundary_guard_closed_seed_count",
                "primitive_wall_evidence_ready",
                "primitive_wall_evidence_status",
                "wall_evidence_scope",
                "wall_claim_limitations",
            ],
            max_rows=5,
        ),
        "",
        "## Seed Wall Evidence",
        "",
        _markdown_table(
            seed_wall.sort_values(["start_condition", "seed"], kind="mergesort"),
            [
                "start_condition",
                "seed",
                "direct_source_object",
                "direct_target_object",
                "recovery_start_object",
                "recovery_interior_target_object_count",
                "recovery_final_object",
                "objective_wall_shape_pass",
                "raw_anchor_ambiguity_step_count",
                "wall_seed_ready",
                "wall_seed_status",
            ],
            max_rows=40,
        ),
        "",
        "## Boundary Guards",
        "",
        _markdown_table(
            boundary_guards.sort_values(["start_condition", "seed"], kind="mergesort"),
            [
                "start_condition",
                "seed",
                "direct_boundary_leak_observed",
                "recovery_boundary_leak_observed",
                "boundary_guard_closed",
                "boundary_guard_status",
            ],
            max_rows=40,
        ),
        "",
        "## Gate Matrix",
        "",
        _markdown_table(
            gates,
            ["gate_id", "gate_status", "observed", "minimum_or_rule", "question"],
            max_rows=20,
        ),
        "",
        "## Boundary",
        "",
        (
            "This audit accepts only local primitive object-level wall evidence "
            "for local_pair_014. It does not establish generality, exact wall "
            "location, quality/cost value, full-replay behavior, or method "
            "success."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_rows = _read_csv(trace_dir / TRACE_ROWS_CSV)
    route_results = _read_csv(trace_dir / TRACE_ROUTE_PROBE_RESULT_ROWS_CSV)
    route_summary = _read_csv(trace_dir / TRACE_ROUTE_PROBE_SUMMARY_ROWS_CSV)
    pair_results = _read_csv(trace_dir / TRACE_PAIR_PROBE_RESULT_ROWS_CSV)
    control_results = _read_csv(trace_dir / TRACE_CONTROL_GUARD_RESULT_ROWS_CSV)
    trace_gates = _read_csv(trace_dir / TRACE_GATE_MATRIX_CSV)

    seed_wall = _seed_wall_rows(route_results)
    boundary_guards = _boundary_guard_rows(route_results)
    pair_wall = _pair_wall_rows(seed_wall, boundary_guards)
    gates = _gate_matrix(
        trace_gates=trace_gates,
        seed_wall=seed_wall,
        boundary_guards=boundary_guards,
        pair_wall=pair_wall,
        route_results=route_results,
        trace_rows=trace_rows,
    )
    summary = _summary(
        trace_dir=trace_dir,
        output_dir=output_dir,
        seed_wall=seed_wall,
        boundary_guards=boundary_guards,
        pair_wall=pair_wall,
        gates=gates,
    )

    _write_csv(seed_wall, output_dir / SEED_WALL_ROWS_CSV)
    _write_csv(boundary_guards, output_dir / BOUNDARY_GUARD_ROWS_CSV)
    _write_csv(pair_wall, output_dir / PAIR_WALL_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_014_wall_evidence_config.v1",
        "trace_dir": str(trace_dir),
        "output_dir": str(output_dir),
        "direct_family": DIRECT_FAMILY,
        "recovery_family": RECOVERY_FAMILY,
        "boundary_direct_family": BOUNDARY_DIRECT_FAMILY,
        "boundary_recovery_family": BOUNDARY_RECOVERY_FAMILY,
        "input_route_summary_rows": int(len(route_summary)),
        "input_pair_result_rows": int(len(pair_results)),
        "input_control_result_rows": int(len(control_results)),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        pair_wall=pair_wall,
        seed_wall=seed_wall,
        boundary_guards=boundary_guards,
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
