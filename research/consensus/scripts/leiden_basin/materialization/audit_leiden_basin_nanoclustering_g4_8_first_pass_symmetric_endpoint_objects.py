#!/usr/bin/env python3
"""Audit first-pass symmetric endpoint objects for the clean and partial cases.

This audit is read-only over the executed first-pass traces and exclusive-target
contrast rows. It treats each local endpoint signature as an object, then
compares source-like first-step objects and final bridge-target objects for
``local_pair_014`` and ``local_pair_005`` only.

It does not rerun Leiden, promote walls, evaluate quality/cost value, replay
full NanoClustering, or claim method success.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from audit_leiden_basin_nanoclustering_g4_8_first_pass_exclusive_target_contrast import (
    DEFAULT_OUTPUT_DIR as DEFAULT_EXCLUSIVE_TARGET_CONTRAST_DIR,
    PAIR_CONTRAST_ROWS_CSV,
    ROUTE_CONTRAST_ROWS_CSV,
)
from run_leiden_basin_nanoclustering_g4_8_fresh_axis_b_first_pass_trace import (
    DEFAULT_OUTPUT_DIR as DEFAULT_FIRST_PASS_TRACE_DIR,
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
    / "leiden_basin_nanoclustering_g4_8_first_pass_symmetric_endpoint_objects_audit_gamma1e5_20260604"
)

OBJECT_ROWS_CSV = "nanoclustering_g4_8_first_pass_symmetric_endpoint_object_rows.csv"
RELATION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_symmetric_endpoint_object_relation_rows.csv"
)
PAIR_SUMMARY_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_symmetric_endpoint_object_pair_summary_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_symmetric_endpoint_object_gate_matrix.csv"
)
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_symmetric_endpoint_object_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_symmetric_endpoint_object_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_symmetric_endpoint_object_report.md"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_symmetric_endpoint_objects"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_first_pass_object_audit"
WALL_PROMOTION_STATUS = "not_promoted_symmetric_endpoint_object_audit_only"
METHOD_STATUS = "diagnostic_endpoint_object_audit_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass symmetric endpoint-object audit only; reads "
    "executed route-local traces and exclusive-target contrast rows for the "
    "clean and partial cases. It does not rerun Leiden, promote walls, evaluate "
    "quality/cost value, replay full NanoClustering, or claim method success."
)

AUDIT_PAIR_IDS = ("local_pair_014", "local_pair_005")
CLEAN_CANDIDATE_ID = "local_pair_014"
PARTIAL_BOUNDARY_ID = "local_pair_005"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _parse_groups(value: Any) -> tuple[tuple[int, ...], ...]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return tuple()
    parsed = json.loads(str(value))
    groups: list[tuple[int, ...]] = []
    for group in parsed:
        groups.append(tuple(sorted(int(node) for node in group)))
    return tuple(sorted(groups))


def _coassignment_pairs(groups: tuple[tuple[int, ...], ...]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for group in groups:
        for left, right in combinations(sorted(group), 2):
            pairs.add((int(left), int(right)))
    return pairs


def _partition_jaccard(
    left_groups: tuple[tuple[int, ...], ...],
    right_groups: tuple[tuple[int, ...], ...],
) -> float:
    left_pairs = _coassignment_pairs(left_groups)
    right_pairs = _coassignment_pairs(right_groups)
    union = left_pairs | right_pairs
    if not union:
        return 1.0
    return float(len(left_pairs & right_pairs) / len(union))


def _group_stats(groups: tuple[tuple[int, ...], ...]) -> dict[str, Any]:
    sizes = [len(group) for group in groups]
    node_ids = sorted({node for group in groups for node in group})
    return {
        "local_node_count": int(len(node_ids)),
        "cluster_count": int(len(groups)),
        "singleton_cluster_count": int(sum(1 for size in sizes if size == 1)),
        "largest_cluster_node_count": int(max(sizes) if sizes else 0),
        "coassigned_pair_count": int(len(_coassignment_pairs(groups))),
        "node_ids": ";".join(str(node) for node in node_ids),
    }


def _trace_object_candidates(trace_rows: pd.DataFrame) -> pd.DataFrame:
    scoped = trace_rows[
        trace_rows["local_pair_id"].astype(str).isin(AUDIT_PAIR_IDS)
        & trace_rows["step_index"].astype(int).isin([1, 5])
    ].copy()
    scoped["endpoint_object_step_role"] = scoped["step_index"].astype(int).map(
        {1: "first_source_like_object", 5: "final_bridge_target_candidate"}
    )
    return scoped


def _object_rows(
    *,
    trace_rows: pd.DataFrame,
    route_contrast: pd.DataFrame,
) -> pd.DataFrame:
    scoped = _trace_object_candidates(trace_rows)
    route_lookup = route_contrast.set_index(["route_contract_id", "seed"]).to_dict("index")
    enriched: list[dict[str, Any]] = []
    for row in scoped.itertuples(index=False):
        route_key = (str(row.route_contract_id), int(row.seed))
        contrast = route_lookup.get(route_key, {})
        enriched.append(
            {
                **row._asdict(),
                "contrast_class": str(contrast.get("contrast_class", "")),
                "all_positive_requirements_pass": _as_bool(
                    contrast.get("all_positive_requirements_pass", False)
                ),
                "source_target_signature_collapse": _as_bool(
                    contrast.get("source_target_signature_collapse", False)
                ),
                "guard_anchor_collapse": _as_bool(
                    contrast.get("guard_anchor_collapse", False)
                ),
                "has_unknown_post_start": _as_bool(
                    contrast.get("has_unknown_post_start", False)
                ),
            }
        )
    enriched_frame = pd.DataFrame(enriched)
    rows: list[dict[str, Any]] = []
    group_cols = [
        "local_pair_id",
        "branch",
        "endpoint_object_step_role",
        "result_endpoint_signature_id",
    ]
    for keys, group in enriched_frame.groupby(group_cols, sort=False):
        local_pair_id, branch, step_role, signature_id = keys
        first_signature = str(group["result_endpoint_signature"].iloc[0])
        groups = _parse_groups(first_signature)
        stats = _group_stats(groups)
        route_count = int(len(group))
        exclusive_count = int(group["all_positive_requirements_pass"].map(_as_bool).sum())
        collapse_count = int(group["source_target_signature_collapse"].map(_as_bool).sum())
        if step_role == "final_bridge_target_candidate" and exclusive_count == route_count:
            object_class = "exclusive_bridge_target_object"
        elif step_role == "final_bridge_target_candidate" and collapse_count == route_count:
            object_class = "collapsed_source_target_object"
        elif step_role == "final_bridge_target_candidate" and exclusive_count > 0 and collapse_count > 0:
            object_class = "mixed_target_boundary_object"
        elif step_role == "first_source_like_object":
            object_class = "source_like_object"
        else:
            object_class = "nonexclusive_endpoint_object"
        rows.append(
            {
                "local_pair_id": str(local_pair_id),
                "branch": str(branch),
                "endpoint_object_step_role": str(step_role),
                "endpoint_object_signature_id": str(signature_id),
                "endpoint_object_class": object_class,
                "route_count": route_count,
                "seed_count": int(group["seed"].nunique()),
                "start_condition_count": int(group["start_condition"].nunique()),
                "start_conditions": ";".join(
                    sorted(group["start_condition"].astype(str).unique().tolist())
                ),
                "seeds": ";".join(
                    str(int(seed)) for seed in sorted(group["seed"].dropna().unique().tolist())
                ),
                "exclusive_bridge_target_pass_count": exclusive_count,
                "source_target_signature_collapse_count": collapse_count,
                "guard_anchor_collapse_count": int(
                    group["guard_anchor_collapse"].map(_as_bool).sum()
                ),
                "intermediate_unknown_route_count": int(
                    group["has_unknown_post_start"].map(_as_bool).sum()
                ),
                "endpoint_assignment_counts": group[
                    "endpoint_assignment_by_step"
                ].value_counts().to_dict(),
                "contrast_class_counts": group["contrast_class"].value_counts().to_dict(),
                "signature_groups_json": first_signature,
                **stats,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _relation_rows(
    *,
    trace_rows: pd.DataFrame,
    route_contrast: pd.DataFrame,
) -> pd.DataFrame:
    scoped = trace_rows[
        trace_rows["local_pair_id"].astype(str).isin(AUDIT_PAIR_IDS)
        & trace_rows["step_index"].astype(int).isin([1, 5])
    ].copy()
    first_rows = scoped[scoped["step_index"].astype(int).eq(1)].set_index(
        ["route_contract_id", "seed"]
    )
    final_rows = scoped[scoped["step_index"].astype(int).eq(5)].set_index(
        ["route_contract_id", "seed"]
    )
    rows: list[dict[str, Any]] = []
    for route in route_contrast[
        route_contrast["local_pair_id"].astype(str).isin(AUDIT_PAIR_IDS)
    ].itertuples(index=False):
        key = (str(route.route_contract_id), int(route.seed))
        if key not in first_rows.index or key not in final_rows.index:
            continue
        first = first_rows.loc[key]
        final = final_rows.loc[key]
        first_groups = _parse_groups(first["result_endpoint_signature"])
        final_groups = _parse_groups(final["result_endpoint_signature"])
        partition_jaccard = _partition_jaccard(first_groups, final_groups)
        first_final_signature_same = str(first["result_endpoint_signature_id"]) == str(
            final["result_endpoint_signature_id"]
        )
        if _as_bool(route.all_positive_requirements_pass):
            relation_class = "clean_source_to_exclusive_target_object"
        elif _as_bool(route.source_target_signature_collapse) or first_final_signature_same:
            relation_class = "source_target_object_collapse"
        elif _as_bool(route.guard_anchor_collapse):
            relation_class = "guard_anchor_object_collapse"
        elif _as_bool(route.has_unknown_post_start):
            relation_class = "unknown_intermediate_before_target_object"
        else:
            relation_class = "other_object_relation_failure"
        rows.append(
            {
                "route_contract_id": str(route.route_contract_id),
                "local_pair_id": str(route.local_pair_id),
                "branch": str(route.branch),
                "start_condition": str(route.start_condition),
                "seed": int(route.seed),
                "evidence_role": str(route.evidence_role),
                "validation_stratum": str(route.validation_stratum),
                "first_object_signature_id": str(first["result_endpoint_signature_id"]),
                "final_object_signature_id": str(final["result_endpoint_signature_id"]),
                "first_endpoint_assignment": str(first["endpoint_assignment_by_step"]),
                "final_endpoint_assignment": str(final["endpoint_assignment_by_step"]),
                "first_final_signature_same": first_final_signature_same,
                "partition_coassignment_jaccard": partition_jaccard,
                "partition_coassignment_distance": float(1.0 - partition_jaccard),
                "contrast_class": str(route.contrast_class),
                "object_relation_class": relation_class,
                "wall_claim_allowed_after_object_audit": False,
                "method_claim_allowed_after_object_audit": False,
                "quality_cost_claim_allowed_after_object_audit": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _pair_summary_rows(
    *,
    pair_contrast: pd.DataFrame,
    object_rows: pd.DataFrame,
    relation_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair_id in AUDIT_PAIR_IDS:
        pair = pair_contrast[pair_contrast["local_pair_id"].astype(str).eq(pair_id)].iloc[0]
        objects = object_rows[object_rows["local_pair_id"].astype(str).eq(pair_id)]
        relations = relation_rows[relation_rows["local_pair_id"].astype(str).eq(pair_id)]
        final_objects = objects[
            objects["endpoint_object_step_role"].astype(str).eq(
                "final_bridge_target_candidate"
            )
        ]
        clean_relations = relations[
            relations["object_relation_class"]
            .astype(str)
            .eq("clean_source_to_exclusive_target_object")
        ]
        collapse_relations = relations[
            relations["object_relation_class"].astype(str).eq("source_target_object_collapse")
        ]
        clean_final_objects = final_objects[
            final_objects["endpoint_object_class"]
            .astype(str)
            .eq("exclusive_bridge_target_object")
        ]
        if pair_id == CLEAN_CANDIDATE_ID and len(clean_relations) == len(relations):
            object_audit_class = "clean_symmetric_endpoint_object_candidate"
            escalation_allowed = True
        elif pair_id == PARTIAL_BOUNDARY_ID and not clean_relations.empty and not collapse_relations.empty:
            object_audit_class = "partial_boundary_source_target_collapse"
            escalation_allowed = False
        else:
            object_audit_class = "not_object_escalation_candidate"
            escalation_allowed = False
        rows.append(
            {
                "local_pair_id": pair_id,
                "branch": str(pair["branch"]),
                "evidence_role": str(pair["evidence_role"]),
                "validation_stratum": str(pair["validation_stratum"]),
                "exclusive_target_contrast_class": str(pair["next_escalation_class"]),
                "route_count": int(len(relations)),
                "source_object_count": int(
                    objects["endpoint_object_step_role"]
                    .astype(str)
                    .eq("first_source_like_object")
                    .sum()
                ),
                "final_object_count": int(len(final_objects)),
                "exclusive_target_object_count": int(len(clean_final_objects)),
                "clean_relation_count": int(len(clean_relations)),
                "source_target_collapse_relation_count": int(len(collapse_relations)),
                "object_relation_class_counts": relations[
                    "object_relation_class"
                ].value_counts().to_dict(),
                "min_partition_coassignment_distance": float(
                    relations["partition_coassignment_distance"].min()
                )
                if not relations.empty
                else math.nan,
                "median_partition_coassignment_distance": float(
                    relations["partition_coassignment_distance"].median()
                )
                if not relations.empty
                else math.nan,
                "max_partition_coassignment_distance": float(
                    relations["partition_coassignment_distance"].max()
                )
                if not relations.empty
                else math.nan,
                "object_audit_class": object_audit_class,
                "object_audit_escalation_allowed": escalation_allowed,
                "wall_claim_allowed_after_object_audit": False,
                "method_claim_allowed_after_object_audit": False,
                "quality_cost_claim_allowed_after_object_audit": False,
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
    pair_contrast: pd.DataFrame,
    object_rows: pd.DataFrame,
    relation_rows: pd.DataFrame,
    pair_summary: pd.DataFrame,
) -> pd.DataFrame:
    observed_pair_ids = sorted(pair_summary["local_pair_id"].astype(str).unique().tolist())
    clean_row = pair_summary[
        pair_summary["local_pair_id"].astype(str).eq(CLEAN_CANDIDATE_ID)
    ].iloc[0]
    partial_row = pair_summary[
        pair_summary["local_pair_id"].astype(str).eq(PARTIAL_BOUNDARY_ID)
    ].iloc[0]
    rows = [
        _gate_row(
            "G1_scope_limited_to_clean_and_partial_pairs",
            "Is the object audit restricted to the clean and partial first-pass pairs?",
            observed_pair_ids,
            "exactly local_pair_014 and local_pair_005",
            observed_pair_ids == sorted(AUDIT_PAIR_IDS),
        ),
        _gate_row(
            "G2_upstream_contrast_supports_scope",
            "Do upstream contrast rows identify one clean and one partial pair?",
            pair_contrast[
                pair_contrast["local_pair_id"].astype(str).isin(AUDIT_PAIR_IDS)
            ][["local_pair_id", "next_escalation_class"]].to_dict("records"),
            "014 clean, 005 partial",
            str(clean_row["exclusive_target_contrast_class"]) == "clean_exclusive_target_candidate"
            and str(partial_row["exclusive_target_contrast_class"])
            == "partial_exclusive_target_candidate",
        ),
        _gate_row(
            "G3_objects_and_relations_materialized",
            "Were endpoint objects and source-target object relations materialized?",
            f"object_rows={len(object_rows)} relation_rows={len(relation_rows)}",
            "nonempty object rows and 64 relation rows",
            not object_rows.empty and len(relation_rows) == 64,
        ),
        _gate_row(
            "G4_clean_pair_has_stable_exclusive_target_object",
            "Does the clean pair have all routes mapped to clean source-to-target object relations?",
            clean_row.to_dict(),
            "32 clean relations and one exclusive final object",
            int(clean_row["clean_relation_count"]) == 32
            and int(clean_row["source_target_collapse_relation_count"]) == 0
            and int(clean_row["exclusive_target_object_count"]) == 1,
        ),
        _gate_row(
            "G5_partial_pair_kept_as_boundary_not_positive",
            "Is the partial pair explicitly kept as a source-target-collapse boundary?",
            partial_row.to_dict(),
            "has clean and collapse relations, escalation disallowed",
            int(partial_row["clean_relation_count"]) == 24
            and int(partial_row["source_target_collapse_relation_count"]) == 8
            and not bool(partial_row["object_audit_escalation_allowed"]),
        ),
        _gate_row(
            "G6_wall_method_quality_claims_closed",
            "Are wall, method, and quality/cost claims still closed?",
            CLAIM_BOUNDARY,
            "all claim flags remain false",
            bool(pair_summary["wall_claim_allowed_after_object_audit"].eq(False).all())
            and bool(pair_summary["method_claim_allowed_after_object_audit"].eq(False).all())
            and bool(pair_summary["quality_cost_claim_allowed_after_object_audit"].eq(False).all()),
        ),
    ]
    return pd.DataFrame(rows)


def _summary(
    *,
    first_pass_trace_dir: Path,
    exclusive_target_contrast_dir: Path,
    output_dir: Path,
    object_rows: pd.DataFrame,
    relation_rows: pd.DataFrame,
    pair_summary: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema": "nanoclustering_g4_8_first_pass_symmetric_endpoint_object_summary.v1",
        "status": RUN_STATUS,
        "first_pass_trace_dir": str(first_pass_trace_dir),
        "exclusive_target_contrast_dir": str(exclusive_target_contrast_dir),
        "output_dir": str(output_dir),
        "audited_pair_ids": list(AUDIT_PAIR_IDS),
        "object_row_count": int(len(object_rows)),
        "relation_row_count": int(len(relation_rows)),
        "pair_summary_row_count": int(len(pair_summary)),
        "object_class_counts": object_rows["endpoint_object_class"].value_counts().to_dict(),
        "relation_class_counts": relation_rows["object_relation_class"].value_counts().to_dict(),
        "pair_object_audit_classes": pair_summary[
            ["local_pair_id", "object_audit_class"]
        ].to_dict("records"),
        "clean_object_candidates": pair_summary.loc[
            pair_summary["object_audit_class"]
            .astype(str)
            .eq("clean_symmetric_endpoint_object_candidate"),
            "local_pair_id",
        ].tolist(),
        "boundary_pairs": pair_summary.loc[
            pair_summary["object_audit_class"]
            .astype(str)
            .eq("partial_boundary_source_target_collapse"),
            "local_pair_id",
        ].tolist(),
        "gate_status_counts": gates["gate_status"].value_counts().to_dict(),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"), "gate_id"
        ].tolist(),
        "interpretation": (
            "The first-pass object audit confirms local_pair_014 as a clean "
            "symmetric endpoint-object candidate and local_pair_005 as a "
            "partial source/target-collapse boundary case. This still does not "
            "promote a wall."
        ),
        "recommended_next_gate": (
            "Use local_pair_014 only for the next wall/pathway-readiness audit. "
            "Keep local_pair_005 as a negative boundary control for collapse."
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
    pair_summary: pd.DataFrame,
    object_rows: pd.DataFrame,
    relation_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass Symmetric Endpoint-Object Audit",
        "",
        f"- status: `{summary['status']}`",
        f"- audited_pair_ids: {summary['audited_pair_ids']}",
        f"- object_row_count: {summary['object_row_count']}",
        f"- relation_row_count: {summary['relation_row_count']}",
        f"- object_class_counts: {summary['object_class_counts']}",
        f"- relation_class_counts: {summary['relation_class_counts']}",
        f"- pair_object_audit_classes: {summary['pair_object_audit_classes']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Pair Summary",
        "",
        _markdown_table(
            pair_summary,
            [
                "local_pair_id",
                "exclusive_target_contrast_class",
                "route_count",
                "source_object_count",
                "final_object_count",
                "exclusive_target_object_count",
                "clean_relation_count",
                "source_target_collapse_relation_count",
                "median_partition_coassignment_distance",
                "object_audit_class",
                "object_audit_escalation_allowed",
            ],
            max_rows=10,
        ),
        "",
        "## Endpoint Objects",
        "",
        _markdown_table(
            object_rows.sort_values(
                ["local_pair_id", "endpoint_object_step_role", "route_count"],
                ascending=[True, True, False],
                kind="mergesort",
            ),
            [
                "local_pair_id",
                "endpoint_object_step_role",
                "endpoint_object_signature_id",
                "endpoint_object_class",
                "route_count",
                "seed_count",
                "start_condition_count",
                "exclusive_bridge_target_pass_count",
                "source_target_signature_collapse_count",
                "cluster_count",
                "largest_cluster_node_count",
            ],
            max_rows=30,
        ),
        "",
        "## Object Relations",
        "",
        _markdown_table(
            relation_rows.sort_values(
                ["local_pair_id", "start_condition", "seed"], kind="mergesort"
            ),
            [
                "local_pair_id",
                "start_condition",
                "seed",
                "first_object_signature_id",
                "final_object_signature_id",
                "partition_coassignment_distance",
                "object_relation_class",
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
            "This audit identifies endpoint-object consistency inside the "
            "route-local first-pass surface. It does not establish a basin wall "
            "or method value."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    first_pass_trace_dir = Path(args.first_pass_trace_dir)
    exclusive_target_contrast_dir = Path(args.exclusive_target_contrast_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_rows = _read_csv(first_pass_trace_dir / TRACE_ROWS_CSV)
    route_contrast = _read_csv(exclusive_target_contrast_dir / ROUTE_CONTRAST_ROWS_CSV)
    pair_contrast = _read_csv(exclusive_target_contrast_dir / PAIR_CONTRAST_ROWS_CSV)

    object_rows = _object_rows(trace_rows=trace_rows, route_contrast=route_contrast)
    relation_rows = _relation_rows(trace_rows=trace_rows, route_contrast=route_contrast)
    pair_summary = _pair_summary_rows(
        pair_contrast=pair_contrast,
        object_rows=object_rows,
        relation_rows=relation_rows,
    )
    gates = _gate_matrix(
        pair_contrast=pair_contrast,
        object_rows=object_rows,
        relation_rows=relation_rows,
        pair_summary=pair_summary,
    )
    summary = _summary(
        first_pass_trace_dir=first_pass_trace_dir,
        exclusive_target_contrast_dir=exclusive_target_contrast_dir,
        output_dir=output_dir,
        object_rows=object_rows,
        relation_rows=relation_rows,
        pair_summary=pair_summary,
        gates=gates,
    )

    _write_csv(object_rows, output_dir / OBJECT_ROWS_CSV)
    _write_csv(relation_rows, output_dir / RELATION_ROWS_CSV)
    _write_csv(pair_summary, output_dir / PAIR_SUMMARY_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_symmetric_endpoint_object_config.v1",
        "first_pass_trace_dir": str(first_pass_trace_dir),
        "exclusive_target_contrast_dir": str(exclusive_target_contrast_dir),
        "output_dir": str(output_dir),
        "audited_pair_ids": list(AUDIT_PAIR_IDS),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        pair_summary=pair_summary,
        object_rows=object_rows,
        relation_rows=relation_rows,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-pass-trace-dir", type=Path, default=DEFAULT_FIRST_PASS_TRACE_DIR)
    parser.add_argument(
        "--exclusive-target-contrast-dir",
        type=Path,
        default=DEFAULT_EXCLUSIVE_TARGET_CONTRAST_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
