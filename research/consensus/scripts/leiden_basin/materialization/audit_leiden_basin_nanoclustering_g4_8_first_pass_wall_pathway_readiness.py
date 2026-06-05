#!/usr/bin/env python3
"""Audit wall/pathway readiness for the first-pass object-level candidate.

This audit reads the executed first-pass traces and the bounded symmetric
endpoint-object audit. It keeps ``local_pair_014`` as the only positive
pathway-readiness candidate and keeps ``local_pair_005`` as the boundary
collapse control. It does not promote a wall claim because direct-path
availability and independent wall evidence remain unaccepted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from audit_leiden_basin_nanoclustering_g4_8_first_pass_symmetric_endpoint_objects import (
    DEFAULT_OUTPUT_DIR as DEFAULT_OBJECT_AUDIT_DIR,
    GATE_MATRIX_CSV as OBJECT_GATE_MATRIX_CSV,
    PAIR_SUMMARY_ROWS_CSV as OBJECT_PAIR_SUMMARY_ROWS_CSV,
    RELATION_ROWS_CSV as OBJECT_RELATION_ROWS_CSV,
)
from run_leiden_basin_nanoclustering_g4_8_fresh_axis_b_first_pass_trace import (
    DEFAULT_OUTPUT_DIR as DEFAULT_FIRST_PASS_TRACE_DIR,
    GATE_MATRIX_CSV as FIRST_PASS_TRACE_GATE_MATRIX_CSV,
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
    / "leiden_basin_nanoclustering_g4_8_first_pass_wall_pathway_readiness_audit_gamma1e5_20260604"
)

ROUTE_READINESS_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_wall_pathway_readiness_route_rows.csv"
)
PAIR_READINESS_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_wall_pathway_readiness_pair_rows.csv"
)
GATE_MATRIX_CSV = "nanoclustering_g4_8_first_pass_wall_pathway_readiness_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_wall_pathway_readiness_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_wall_pathway_readiness_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_wall_pathway_readiness_report.md"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_wall_pathway_readiness"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_first_pass_wall_pathway_audit"
WALL_PROMOTION_STATUS = "not_promoted_readiness_only_missing_direct_path_audit"
METHOD_STATUS = "diagnostic_wall_pathway_readiness_audit_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass wall/pathway-readiness audit only; reads "
    "executed route-local traces and bounded endpoint-object audit rows. It "
    "does not rerun Leiden, promote walls, evaluate quality/cost value, replay "
    "full NanoClustering, or claim method success."
)

POSITIVE_PAIR_ID = "local_pair_014"
BOUNDARY_PAIR_ID = "local_pair_005"
AUDIT_PAIR_IDS = (POSITIVE_PAIR_ID, BOUNDARY_PAIR_ID)


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


def _first_step(ordered: pd.DataFrame, mask: pd.Series) -> int | None:
    matches = ordered.loc[mask.astype(bool), "step_index"]
    if matches.empty:
        return None
    return int(matches.iloc[0])


def _step_sequence(values: pd.Series) -> str:
    return " -> ".join(str(value) for value in values.astype(str).tolist())


def _pathway_shape_class(
    *,
    pair_id: str,
    object_relation_class: str,
    has_unknown: bool,
    has_ambiguous: bool,
    first_target_step: int | None,
    max_debt: float,
    max_recovery: float,
) -> str:
    if pair_id == BOUNDARY_PAIR_ID:
        if object_relation_class == "source_target_object_collapse":
            return "boundary_source_target_collapse_not_positive"
        return "boundary_partial_clean_route_not_pair_positive"
    if object_relation_class != "clean_source_to_exclusive_target_object":
        return "not_clean_object_relation"
    if has_unknown:
        return "unknown_intermediate_blocks_pathway_readiness"
    if has_ambiguous:
        return "ambiguous_intermediate_blocks_pathway_readiness"
    timing = "no_target" if first_target_step is None else f"step{int(first_target_step)}"
    if max_debt > 0.0 and max_recovery > 0.0:
        objective = "with_objective_debt_recovery"
    elif max_debt > 0.0:
        objective = "with_objective_debt_without_recovery"
    elif max_recovery > 0.0:
        objective = "without_debt_with_objective_recovery"
    else:
        objective = "without_objective_debt_recovery"
    return f"clean_known_anchor_{timing}_{objective}"


def _route_readiness_rows(
    *,
    trace_rows: pd.DataFrame,
    relation_rows: pd.DataFrame,
    pair_summary: pd.DataFrame,
) -> pd.DataFrame:
    scoped_relations = relation_rows[
        relation_rows["local_pair_id"].astype(str).isin(AUDIT_PAIR_IDS)
    ].copy()
    pair_lookup = pair_summary.set_index("local_pair_id").to_dict("index")
    rows: list[dict[str, Any]] = []
    for relation in scoped_relations.sort_values(
        ["local_pair_id", "start_condition", "seed"], kind="mergesort"
    ).itertuples(index=False):
        pair_id = str(relation.local_pair_id)
        route_contract_id = str(relation.route_contract_id)
        seed = int(relation.seed)
        group = trace_rows[
            trace_rows["route_contract_id"].astype(str).eq(route_contract_id)
            & trace_rows["seed"].astype(int).eq(seed)
        ].sort_values("step_index", kind="mergesort")
        if group.empty:
            continue
        pair_data = pair_lookup.get(pair_id, {})
        endpoint_sequence = _step_sequence(group["endpoint_assignment_by_step"])
        signature_sequence = _step_sequence(group["result_endpoint_signature_id"])
        unknown_count = int(
            group["endpoint_assignment_by_step"].astype(str).eq("unknown_new_endpoint").sum()
        )
        ambiguous_count = int(
            group["endpoint_assignment_by_step"]
            .astype(str)
            .str.startswith("ambiguous_anchor_match")
            .sum()
        )
        post = group[group["step_index"].astype(int).gt(1)].copy()
        post_unknown_count = int(
            post["endpoint_assignment_by_step"].astype(str).eq("unknown_new_endpoint").sum()
        )
        post_ambiguous_count = int(
            post["endpoint_assignment_by_step"]
            .astype(str)
            .str.startswith("ambiguous_anchor_match")
            .sum()
        )
        direct_retained_all_steps = bool(
            group["active_direct_edge_weight"].astype(float).gt(0.0).all()
        )
        bridge_fraction_sequence = ";".join(
            f"{float(value):.2f}" for value in group["bridge_edge_weight_fraction"].tolist()
        )
        bridge_fraction_monotone = group["bridge_edge_weight_fraction"].astype(float).tolist() == [
            1.0,
            0.75,
            0.5,
            0.25,
            0.0,
        ]
        first_target_step = _first_step(
            group,
            group["endpoint_assignment_by_step"].astype(str).eq("drop_bridge_target_anchor"),
        )
        max_debt = float(group["objective_debt_from_start"].astype(float).max())
        max_recovery = float(group["objective_recovery_from_min"].astype(float).max())
        final_delta = float(group["objective_delta_from_start"].astype(float).iloc[-1])
        support_incompatibility_count = int(
            group["support_incompatibility_check"].map(_as_bool).sum()
        )
        post_support_incompatibility_count = int(
            post["support_incompatibility_check"].map(_as_bool).sum()
        )
        polish_reversion_count = int(group["polish_reversion_check"].map(_as_bool).sum())
        post_polish_reversion_count = int(post["polish_reversion_check"].map(_as_bool).sum())
        object_relation_class = str(relation.object_relation_class)
        pathway_shape_class = _pathway_shape_class(
            pair_id=pair_id,
            object_relation_class=object_relation_class,
            has_unknown=post_unknown_count > 0,
            has_ambiguous=post_ambiguous_count > 0,
            first_target_step=first_target_step,
            max_debt=max_debt,
            max_recovery=max_recovery,
        )
        route_pathway_ready = bool(
            pair_id == POSITIVE_PAIR_ID
            and object_relation_class == "clean_source_to_exclusive_target_object"
            and direct_retained_all_steps
            and bridge_fraction_monotone
            and first_target_step is not None
            and post_unknown_count == 0
            and post_ambiguous_count == 0
            and support_incompatibility_count == 0
        )
        if route_pathway_ready:
            readiness_status = "route_pathway_readiness_candidate_wall_claim_closed"
            block_reason = "wall claim still missing independent direct-path and wall evidence"
        elif pair_id == BOUNDARY_PAIR_ID:
            readiness_status = "boundary_control_not_pathway_positive"
            block_reason = "pair has source/target object collapse or mixed target boundary object"
        else:
            readiness_status = "route_not_pathway_ready"
            block_reason = "route fails object, endpoint, or support readiness precheck"
        rows.append(
            {
                "route_contract_id": route_contract_id,
                "local_pair_id": pair_id,
                "branch": str(relation.branch),
                "start_condition": str(relation.start_condition),
                "seed": seed,
                "evidence_role": str(relation.evidence_role),
                "validation_stratum": str(relation.validation_stratum),
                "object_audit_class": str(pair_data.get("object_audit_class", "")),
                "object_relation_class": object_relation_class,
                "first_object_signature_id": str(relation.first_object_signature_id),
                "final_object_signature_id": str(relation.final_object_signature_id),
                "partition_coassignment_distance": float(
                    relation.partition_coassignment_distance
                ),
                "endpoint_sequence": endpoint_sequence,
                "signature_sequence": signature_sequence,
                "bridge_fraction_sequence": bridge_fraction_sequence,
                "bridge_fraction_monotone_predeclared": bridge_fraction_monotone,
                "direct_edge_retained_all_steps": direct_retained_all_steps,
                "first_exclusive_target_step": first_target_step,
                "unknown_endpoint_step_count": unknown_count,
                "post_start_unknown_endpoint_step_count": post_unknown_count,
                "ambiguous_anchor_step_count": ambiguous_count,
                "post_start_ambiguous_anchor_step_count": post_ambiguous_count,
                "support_incompatibility_step_count": support_incompatibility_count,
                "post_start_support_incompatibility_step_count": post_support_incompatibility_count,
                "polish_reversion_step_count": polish_reversion_count,
                "post_start_polish_reversion_step_count": post_polish_reversion_count,
                "max_objective_debt_from_start": max_debt,
                "max_objective_recovery_from_min": max_recovery,
                "final_objective_delta_from_start": final_delta,
                "pathway_shape_class": pathway_shape_class,
                "route_pathway_readiness_candidate": route_pathway_ready,
                "route_wall_claim_ready": False,
                "route_readiness_status": readiness_status,
                "route_readiness_block_reason": block_reason,
                "wall_claim_missing_fields": (
                    "independent_direct_path_availability;accepted_wall_debt;"
                    "accepted_wall_recovery;independent_support_incompatibility;"
                    "full_replay_reproducibility"
                ),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _pair_readiness_rows(
    *,
    pair_summary: pd.DataFrame,
    route_readiness: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair_id in AUDIT_PAIR_IDS:
        pair = pair_summary[pair_summary["local_pair_id"].astype(str).eq(pair_id)].iloc[0]
        group = route_readiness[route_readiness["local_pair_id"].astype(str).eq(pair_id)]
        route_count = int(len(group))
        ready_count = int(group["route_pathway_readiness_candidate"].map(_as_bool).sum())
        wall_ready_count = int(group["route_wall_claim_ready"].map(_as_bool).sum())
        if pair_id == POSITIVE_PAIR_ID and route_count and ready_count == route_count:
            pair_status = "pair_pathway_readiness_candidate_wall_claim_closed"
            pathway_candidate = True
            block_reason = "direct-path and independent wall evidence remain missing"
        elif pair_id == BOUNDARY_PAIR_ID:
            pair_status = "boundary_control_not_pathway_positive"
            pathway_candidate = False
            block_reason = "source/target collapse boundary control"
        else:
            pair_status = "not_pathway_ready"
            pathway_candidate = False
            block_reason = "route-level readiness incomplete"
        rows.append(
            {
                "local_pair_id": pair_id,
                "branch": str(pair["branch"]),
                "evidence_role": str(pair["evidence_role"]),
                "validation_stratum": str(pair["validation_stratum"]),
                "object_audit_class": str(pair["object_audit_class"]),
                "route_count": route_count,
                "route_pathway_readiness_candidate_count": ready_count,
                "route_wall_claim_ready_count": wall_ready_count,
                "pathway_shape_class_counts": _count_dict(group["pathway_shape_class"]),
                "route_readiness_status_counts": _count_dict(group["route_readiness_status"]),
                "min_partition_coassignment_distance": float(
                    group["partition_coassignment_distance"].astype(float).min()
                )
                if not group.empty
                else None,
                "median_partition_coassignment_distance": float(
                    group["partition_coassignment_distance"].astype(float).median()
                )
                if not group.empty
                else None,
                "max_partition_coassignment_distance": float(
                    group["partition_coassignment_distance"].astype(float).max()
                )
                if not group.empty
                else None,
                "max_objective_debt_from_start": float(
                    group["max_objective_debt_from_start"].astype(float).max()
                )
                if not group.empty
                else None,
                "max_objective_recovery_from_min": float(
                    group["max_objective_recovery_from_min"].astype(float).max()
                )
                if not group.empty
                else None,
                "pair_pathway_readiness_candidate": pathway_candidate,
                "pair_wall_claim_ready": False,
                "pair_readiness_status": pair_status,
                "pair_readiness_block_reason": block_reason,
                "wall_claim_missing_fields": (
                    "independent_direct_path_availability;accepted_wall_debt;"
                    "accepted_wall_recovery;independent_support_incompatibility;"
                    "full_replay_reproducibility"
                ),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


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
    }


def _gate_matrix(
    *,
    first_pass_trace_gates: pd.DataFrame,
    object_gates: pd.DataFrame,
    route_readiness: pd.DataFrame,
    pair_readiness: pd.DataFrame,
) -> pd.DataFrame:
    positive = pair_readiness[
        pair_readiness["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)
    ].iloc[0]
    boundary = pair_readiness[
        pair_readiness["local_pair_id"].astype(str).eq(BOUNDARY_PAIR_ID)
    ].iloc[0]
    observed_pairs = sorted(pair_readiness["local_pair_id"].astype(str).unique().tolist())
    rows = [
        _gate_row(
            "G1_upstream_trace_and_object_gates_pass",
            "Did upstream first-pass trace and object-audit gates pass?",
            {
                "first_pass_trace": _count_dict(first_pass_trace_gates["gate_status"]),
                "object_audit": _count_dict(object_gates["gate_status"]),
            },
            "all upstream gates pass",
            bool(first_pass_trace_gates["gate_status"].astype(str).eq("pass").all())
            and bool(object_gates["gate_status"].astype(str).eq("pass").all()),
        ),
        _gate_row(
            "G2_scope_positive_014_boundary_005_only",
            "Is readiness scoped to 014 as positive and 005 as boundary control?",
            observed_pairs,
            "exactly local_pair_014 and local_pair_005",
            observed_pairs == sorted(AUDIT_PAIR_IDS),
        ),
        _gate_row(
            "G3_positive_pair_all_routes_pathway_ready",
            "Does local_pair_014 pass route-local pathway-readiness on all routes?",
            positive.to_dict(),
            "32/32 routes pathway-ready, wall claim closed",
            int(positive["route_pathway_readiness_candidate_count"]) == 32
            and not bool(positive["pair_wall_claim_ready"]),
        ),
        _gate_row(
            "G4_boundary_pair_not_pathway_positive",
            "Is local_pair_005 retained as boundary control rather than positive evidence?",
            boundary.to_dict(),
            "0 pathway-ready positive routes and pair candidate false",
            int(boundary["route_pathway_readiness_candidate_count"]) == 0
            and not bool(boundary["pair_pathway_readiness_candidate"]),
        ),
        _gate_row(
            "G5_pathway_shape_fields_materialized",
            "Were objective, endpoint, support, direct-edge, and bridge-schedule fields materialized?",
            sorted(route_readiness.columns.tolist()),
            "route readiness fields present and nonempty",
            not route_readiness.empty
            and {
                "direct_edge_retained_all_steps",
                "bridge_fraction_monotone_predeclared",
                "first_exclusive_target_step",
                "max_objective_debt_from_start",
                "max_objective_recovery_from_min",
                "support_incompatibility_step_count",
            }.issubset(set(route_readiness.columns)),
        ),
        _gate_row(
            "G6_wall_claims_closed_missing_independent_direct_path",
            "Are wall claims still closed because independent direct-path and wall evidence are missing?",
            pair_readiness[
                ["local_pair_id", "pair_wall_claim_ready", "wall_claim_missing_fields"]
            ].to_dict("records"),
            "all pair wall-claim flags false",
            bool(pair_readiness["pair_wall_claim_ready"].eq(False).all()),
        ),
    ]
    return pd.DataFrame(rows)


def _summary(
    *,
    first_pass_trace_dir: Path,
    object_audit_dir: Path,
    output_dir: Path,
    route_readiness: pd.DataFrame,
    pair_readiness: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema": "nanoclustering_g4_8_first_pass_wall_pathway_readiness_summary.v1",
        "status": RUN_STATUS,
        "first_pass_trace_dir": str(first_pass_trace_dir),
        "object_audit_dir": str(object_audit_dir),
        "output_dir": str(output_dir),
        "route_readiness_row_count": int(len(route_readiness)),
        "pair_readiness_row_count": int(len(pair_readiness)),
        "pair_readiness_status_counts": _count_dict(pair_readiness["pair_readiness_status"]),
        "route_readiness_status_counts": _count_dict(route_readiness["route_readiness_status"]),
        "pathway_shape_class_counts": _count_dict(route_readiness["pathway_shape_class"]),
        "pathway_readiness_candidates": pair_readiness.loc[
            pair_readiness["pair_pathway_readiness_candidate"].map(_as_bool),
            "local_pair_id",
        ].tolist(),
        "boundary_controls": pair_readiness.loc[
            pair_readiness["pair_readiness_status"]
            .astype(str)
            .eq("boundary_control_not_pathway_positive"),
            "local_pair_id",
        ].tolist(),
        "wall_claim_ready_pairs": pair_readiness.loc[
            pair_readiness["pair_wall_claim_ready"].map(_as_bool),
            "local_pair_id",
        ].tolist(),
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"), "gate_id"
        ].tolist(),
        "interpretation": (
            "local_pair_014 is ready for a predeclared pathway probe, while "
            "local_pair_005 remains a collapse boundary control. No wall claim is "
            "opened because independent direct-path evidence, accepted recovery, "
            "and independent wall evidence remain missing."
        ),
        "recommended_next_gate": (
            "Design a predeclared local_pair_014 wall/pathway probe that adds "
            "independent direct-path availability and accepted wall recovery "
            "checks; keep local_pair_005 as the boundary control."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
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
    pair_readiness: pd.DataFrame,
    route_readiness: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass Wall/Pathway Readiness Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- route_readiness_row_count: {summary['route_readiness_row_count']}",
        f"- pair_readiness_status_counts: {summary['pair_readiness_status_counts']}",
        f"- route_readiness_status_counts: {summary['route_readiness_status_counts']}",
        f"- pathway_shape_class_counts: {summary['pathway_shape_class_counts']}",
        f"- pathway_readiness_candidates: {summary['pathway_readiness_candidates']}",
        f"- boundary_controls: {summary['boundary_controls']}",
        f"- wall_claim_ready_pairs: {summary['wall_claim_ready_pairs']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Pair Readiness",
        "",
        _markdown_table(
            pair_readiness,
            [
                "local_pair_id",
                "object_audit_class",
                "route_count",
                "route_pathway_readiness_candidate_count",
                "pair_pathway_readiness_candidate",
                "pair_wall_claim_ready",
                "pair_readiness_status",
                "pair_readiness_block_reason",
            ],
            max_rows=10,
        ),
        "",
        "## Route Readiness",
        "",
        _markdown_table(
            route_readiness.sort_values(
                ["local_pair_id", "start_condition", "seed"], kind="mergesort"
            ),
            [
                "local_pair_id",
                "start_condition",
                "seed",
                "object_relation_class",
                "first_exclusive_target_step",
                "direct_edge_retained_all_steps",
                "bridge_fraction_monotone_predeclared",
                "post_start_unknown_endpoint_step_count",
                "post_start_ambiguous_anchor_step_count",
                "support_incompatibility_step_count",
                "max_objective_debt_from_start",
                "max_objective_recovery_from_min",
                "pathway_shape_class",
                "route_readiness_status",
            ],
            max_rows=80,
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
            "This is a readiness audit, not wall evidence. The positive output is "
            "a scoped next-probe candidate: local_pair_014. Wall promotion stays "
            "closed until independent direct-path and wall-evidence checks are "
            "designed and executed."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    first_pass_trace_dir = Path(args.first_pass_trace_dir)
    object_audit_dir = Path(args.object_audit_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_rows = _read_csv(first_pass_trace_dir / TRACE_ROWS_CSV)
    first_pass_trace_gates = _read_csv(first_pass_trace_dir / FIRST_PASS_TRACE_GATE_MATRIX_CSV)
    object_gates = _read_csv(object_audit_dir / OBJECT_GATE_MATRIX_CSV)
    object_relations = _read_csv(object_audit_dir / OBJECT_RELATION_ROWS_CSV)
    object_pair_summary = _read_csv(object_audit_dir / OBJECT_PAIR_SUMMARY_ROWS_CSV)

    route_readiness = _route_readiness_rows(
        trace_rows=trace_rows,
        relation_rows=object_relations,
        pair_summary=object_pair_summary,
    )
    pair_readiness = _pair_readiness_rows(
        pair_summary=object_pair_summary,
        route_readiness=route_readiness,
    )
    gates = _gate_matrix(
        first_pass_trace_gates=first_pass_trace_gates,
        object_gates=object_gates,
        route_readiness=route_readiness,
        pair_readiness=pair_readiness,
    )
    summary = _summary(
        first_pass_trace_dir=first_pass_trace_dir,
        object_audit_dir=object_audit_dir,
        output_dir=output_dir,
        route_readiness=route_readiness,
        pair_readiness=pair_readiness,
        gates=gates,
    )

    _write_csv(route_readiness, output_dir / ROUTE_READINESS_ROWS_CSV)
    _write_csv(pair_readiness, output_dir / PAIR_READINESS_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_wall_pathway_readiness_config.v1",
        "first_pass_trace_dir": str(first_pass_trace_dir),
        "object_audit_dir": str(object_audit_dir),
        "output_dir": str(output_dir),
        "positive_pair_id": POSITIVE_PAIR_ID,
        "boundary_pair_id": BOUNDARY_PAIR_ID,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        pair_readiness=pair_readiness,
        route_readiness=route_readiness,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-pass-trace-dir", type=Path, default=DEFAULT_FIRST_PASS_TRACE_DIR)
    parser.add_argument("--object-audit-dir", type=Path, default=DEFAULT_OBJECT_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
