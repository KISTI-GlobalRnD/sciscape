#!/usr/bin/env python3
"""Certify local_pair_016 object identity boundaries from existing traces.

This read-only audit follows the 016 signature-identity audit. It asks whether
the four stable local signatures can be lifted to object-level evidence by an
existing symmetric endpoint-object membership table, or by an explicit local
signature-object certificate over the same trace surface.

The answer is deliberately conservative: a local signature-object certificate is
available, and the target anchor is locally certified, but 016 has no external
symmetric endpoint-object membership row, the source family remains split, and
the typed transient remains a non-endpoint blocker. Therefore object identity
and local object-wall evidence remain closed.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from audit_leiden_basin_nanoclustering_g4_8_first_pass_016_object_signature_identity_resolution import (
    CLAIM_BOUNDARY as SIGNATURE_CLAIM_BOUNDARY,
    DEFAULT_OUTPUT_DIR as DEFAULT_SIGNATURE_IDENTITY_DIR,
    GATE_MATRIX_CSV as SIGNATURE_GATE_MATRIX_CSV,
    LOCAL_ABLATION_SIGNATURE_ROWS_CSV as SIGNATURE_LOCAL_ABLATION_SIGNATURE_ROWS_CSV,
    ROUTE_ROWS_CSV as SIGNATURE_ROUTE_ROWS_CSV,
    SIGNATURE_ROWS_CSV,
    SOURCE_GUARD_SIGNATURE_ID,
    SOURCE_SIGNATURE_ID,
    SUMMARY_JSON as SIGNATURE_SUMMARY_JSON,
    TARGET_SIGNATURE_ID,
    TRANSIENT_SIGNATURE_ID,
)
from audit_leiden_basin_nanoclustering_g4_8_first_pass_symmetric_endpoint_objects import (
    DEFAULT_OUTPUT_DIR as DEFAULT_SYMMETRIC_ENDPOINT_OBJECT_DIR,
    GATE_MATRIX_CSV as SYMMETRIC_GATE_MATRIX_CSV,
    OBJECT_ROWS_CSV as SYMMETRIC_OBJECT_ROWS_CSV,
    PAIR_SUMMARY_ROWS_CSV as SYMMETRIC_PAIR_SUMMARY_ROWS_CSV,
    RELATION_ROWS_CSV as SYMMETRIC_RELATION_ROWS_CSV,
    SUMMARY_JSON as SYMMETRIC_SUMMARY_JSON,
)
from run_leiden_basin_nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace import (
    POSITIVE_PAIR_ID,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_object_identity_certificate_gamma1e5_20260608"
)

SCOPE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_identity_certificate_scope_rows.csv"
)
LOCAL_OBJECT_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_identity_certificate_local_object_rows.csv"
)
RELATION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_identity_certificate_relation_rows.csv"
)
EVIDENCE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_identity_certificate_evidence_rows.csv"
)
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_identity_certificate_decision_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_identity_certificate_gate_matrix.csv"
)
SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_016_object_identity_certificate_summary.json"
)
CONFIG_JSON = (
    "nanoclustering_g4_8_first_pass_016_object_identity_certificate_config.json"
)
REPORT_MD = (
    "nanoclustering_g4_8_first_pass_016_object_identity_certificate_report.md"
)

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_016_object_identity_certificate"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_016_object_identity_certificate"
WALL_PROMOTION_STATUS = "not_promoted_local_object_certificate_only"
METHOD_STATUS = "object_identity_certificate_audit_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass 016 object-identity certificate audit only; "
    "reads existing 016 signature-identity outputs and the first-pass symmetric "
    "endpoint-object audit. It builds a local signature-object certificate but "
    "does not resolve full endpoint-object identity, rerun Leiden, expand routes, "
    "promote pathway labels or walls, evaluate quality/cost value, replay full "
    "NanoClustering, or claim method success."
)

EXPECTED_SIGNATURE_IDS = {
    SOURCE_SIGNATURE_ID,
    SOURCE_GUARD_SIGNATURE_ID,
    TARGET_SIGNATURE_ID,
    TRANSIENT_SIGNATURE_ID,
}


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


def _json_dump(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True)


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _unique_join(series: pd.Series) -> str:
    values = sorted({str(value) for value in series.dropna() if str(value) != ""})
    return ";".join(values)


def _clean_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


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


def _partition_jaccard(left_value: Any, right_value: Any) -> float:
    left_pairs = _coassignment_pairs(_parse_groups(left_value))
    right_pairs = _coassignment_pairs(_parse_groups(right_value))
    union = left_pairs | right_pairs
    if not union:
        return 1.0
    return float(len(left_pairs & right_pairs) / len(union))


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
        "observed": _json_dump(observed),
        "minimum_or_rule": minimum_or_rule,
        "gate_status": "pass" if bool(passed) else "fail",
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
        "signature_summary": _read_json(
            args.signature_identity_dir / SIGNATURE_SUMMARY_JSON
        ),
        "signature_gates": _read_csv(
            args.signature_identity_dir / SIGNATURE_GATE_MATRIX_CSV
        ),
        "signature_rows": _read_csv(args.signature_identity_dir / SIGNATURE_ROWS_CSV),
        "signature_route_rows": _read_csv(
            args.signature_identity_dir / SIGNATURE_ROUTE_ROWS_CSV
        ),
        "local_ablation_signature_rows": _read_csv(
            args.signature_identity_dir / SIGNATURE_LOCAL_ABLATION_SIGNATURE_ROWS_CSV
        ),
        "symmetric_summary": _read_json(
            args.symmetric_endpoint_object_dir / SYMMETRIC_SUMMARY_JSON
        ),
        "symmetric_gates": _read_csv(
            args.symmetric_endpoint_object_dir / SYMMETRIC_GATE_MATRIX_CSV
        ),
        "symmetric_object_rows": _read_csv(
            args.symmetric_endpoint_object_dir / SYMMETRIC_OBJECT_ROWS_CSV
        ),
        "symmetric_relation_rows": _read_csv(
            args.symmetric_endpoint_object_dir / SYMMETRIC_RELATION_ROWS_CSV
        ),
        "symmetric_pair_summary_rows": _read_csv(
            args.symmetric_endpoint_object_dir / SYMMETRIC_PAIR_SUMMARY_ROWS_CSV
        ),
    }


def _scope_rows(context: dict[str, Any]) -> pd.DataFrame:
    symmetric_summary = context["symmetric_summary"]
    symmetric_pair_summary = context["symmetric_pair_summary_rows"]
    symmetric_object_rows = context["symmetric_object_rows"]
    audited_pair_ids = sorted(str(pair_id) for pair_id in symmetric_summary["audited_pair_ids"])
    pair_summary_ids = sorted(symmetric_pair_summary["local_pair_id"].astype(str).unique())
    object_row_pair_ids = sorted(symmetric_object_rows["local_pair_id"].astype(str).unique())
    signature_summary = context["signature_summary"]
    return pd.DataFrame(
        [
            {
                "scope_id": "S1_existing_symmetric_endpoint_object_scope",
                "source_artifact": str(symmetric_summary["output_dir"]),
                "audited_pair_ids": ";".join(audited_pair_ids),
                "contains_local_pair_016": POSITIVE_PAIR_ID in audited_pair_ids,
                "membership_available_for_local_pair_016": False,
                "scope_status": "external_symmetric_endpoint_object_membership_absent_for_016",
                "claim_effect": "blocks_external_object_identity_certificate",
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            },
            {
                "scope_id": "S2_symmetric_object_tables_confirm_absence",
                "source_artifact": str(symmetric_summary["output_dir"]),
                "audited_pair_ids": ";".join(pair_summary_ids),
                "contains_local_pair_016": POSITIVE_PAIR_ID in object_row_pair_ids,
                "membership_available_for_local_pair_016": False,
                "scope_status": "no_016_rows_in_symmetric_object_or_pair_tables",
                "claim_effect": "forces_local_signature_object_certificate_path",
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            },
            {
                "scope_id": "S3_016_signature_identity_surface",
                "source_artifact": str(signature_summary["output_dir"]),
                "audited_pair_ids": POSITIVE_PAIR_ID,
                "contains_local_pair_016": True,
                "membership_available_for_local_pair_016": False,
                "scope_status": "local_signature_identity_surface_available",
                "claim_effect": "permits_local_certificate_only_not_wall_claim",
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            },
        ]
    )


def _object_role(primary_role: str) -> tuple[str, str, str, bool, bool]:
    if primary_role == "source_family_strict_signature":
        return (
            "source_family_strict_component",
            "source_family_local_signature_object_component",
            "source_family_component_certified_not_unified",
            True,
            False,
        )
    if primary_role == "source_guard_blocker_signature":
        return (
            "source_guard_blocker_component",
            "source_guard_local_signature_object_component",
            "source_guard_component_certified_not_unified",
            True,
            False,
        )
    if primary_role == "target_anchor_signature":
        return (
            "target_anchor_local_object",
            "drop_bridge_target_local_signature_object",
            "target_local_object_certified",
            True,
            True,
        )
    if primary_role == "typed_transient_signature":
        return (
            "typed_transient_nonendpoint_component",
            "typed_transient_local_signature_object_nonendpoint",
            "typed_transient_signature_certified_nonendpoint",
            True,
            False,
        )
    return (
        "unclassified_local_signature",
        "unclassified_local_signature_object",
        "local_object_not_certified",
        False,
        False,
    )


def _local_object_rows(
    *,
    signature_rows: pd.DataFrame,
    local_ablation_signature_rows: pd.DataFrame,
) -> pd.DataFrame:
    local_lookup: dict[str, pd.DataFrame] = {
        str(signature_id): group.copy()
        for signature_id, group in local_ablation_signature_rows.groupby(
            "signature_id", dropna=False
        )
    }
    rows: list[dict[str, Any]] = []
    for row in signature_rows.sort_values("primary_signature_role", kind="mergesort").itertuples(
        index=False
    ):
        data = row._asdict()
        signature_id = str(data["signature_id"])
        primary_role = str(data["primary_signature_role"])
        (
            local_object_role,
            local_object_class,
            certificate_status,
            local_signature_object_certified,
            endpoint_object_certified,
        ) = _object_role(primary_role)
        local_rows = local_lookup.get(signature_id, pd.DataFrame())
        groups = _parse_groups(data["signature"])
        stats = _group_stats(groups)
        rows.append(
            {
                "signature_id": signature_id,
                "primary_signature_role": primary_role,
                "local_object_role": local_object_role,
                "local_object_class": local_object_class,
                "local_object_certificate_status": certificate_status,
                "local_signature_object_certified": local_signature_object_certified,
                "endpoint_object_certified": endpoint_object_certified,
                "target_local_object_certified": signature_id == TARGET_SIGNATURE_ID
                and endpoint_object_certified,
                "transient_endpoint_object_certified": False,
                "source_family_object_unified": False,
                "object_wall_endpoint_ready": False,
                "trace_row_count": _as_int(data.get("trace_row_count")),
                "route_family_count": _as_int(data.get("route_family_count")),
                "start_condition_count": _as_int(data.get("start_condition_count")),
                "seed_count": _as_int(data.get("seed_count")),
                "step_labels": str(data.get("step_labels", "")),
                "bridge_fractions": str(data.get("bridge_fractions", "")),
                "support_incompatibility_rows": _as_int(
                    data.get("support_incompatibility_rows")
                ),
                "local_ablation_graph_variants": _clean_str(
                    data.get("local_ablation_graph_variants", "")
                ),
                "local_ablation_run_count": _as_int(
                    data.get("local_ablation_run_count")
                ),
                "local_ablation_pair_coassigned_shares": _unique_join(
                    local_rows["pair_coassigned_share"].astype(str)
                )
                if not local_rows.empty
                else "",
                "prior_transient_signature_match": _as_bool(
                    data.get("prior_transient_signature_match")
                ),
                "prior_target_signature_match": _as_bool(
                    data.get("prior_target_signature_match")
                ),
                "signature_identity_resolution_status": str(
                    data.get("signature_identity_resolution_status", "")
                ),
                "object_identity_resolution_status": str(
                    data.get("object_identity_resolution_status", "")
                ),
                "signature_identity_resolved": _as_bool(
                    data.get("signature_identity_resolved")
                ),
                "object_identity_resolved": False,
                "signature": str(data["signature"]),
                **stats,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _source_signature_from_assignment(value: Any) -> str:
    assignment = str(value)
    first = assignment.split(" -> ")[0]
    if first == "original_source_anchor":
        return SOURCE_SIGNATURE_ID
    if "drop_direct_guard_anchor" in first:
        return SOURCE_GUARD_SIGNATURE_ID
    return ""


def _return_signature_from_assignment(value: Any) -> str:
    assignment = str(value)
    last = assignment.split(" -> ")[-1]
    if last == "original_source_anchor":
        return SOURCE_SIGNATURE_ID
    if "drop_direct_guard_anchor" in last:
        return SOURCE_GUARD_SIGNATURE_ID
    if last == "drop_bridge_target_anchor":
        return TARGET_SIGNATURE_ID
    if last == "unknown_new_endpoint":
        return TRANSIENT_SIGNATURE_ID
    return ""


def _relation_rows(
    *,
    route_rows: pd.DataFrame,
    local_object_rows: pd.DataFrame,
) -> pd.DataFrame:
    target_signature = local_object_rows.loc[
        local_object_rows["signature_id"].astype(str).eq(TARGET_SIGNATURE_ID), "signature"
    ].iloc[0]
    source_signatures = {
        str(row["signature_id"]): str(row["signature"])
        for _, row in local_object_rows[
            local_object_rows["signature_id"].astype(str).isin(
                {SOURCE_SIGNATURE_ID, SOURCE_GUARD_SIGNATURE_ID}
            )
        ].iterrows()
    }
    rows: list[dict[str, Any]] = []
    for row in route_rows.sort_values(
        ["planned_route_family", "start_condition", "seed"], kind="mergesort"
    ).itertuples(index=False):
        data = row._asdict()
        route_identity_class = str(data["route_identity_class"])
        endpoint_sequence = str(data["endpoint_assignment_sequence"])
        source_signature_id = _source_signature_from_assignment(endpoint_sequence)
        return_signature_id = _return_signature_from_assignment(endpoint_sequence)
        target_seen = "drop_bridge_target_anchor" in endpoint_sequence
        transient_seen = "pathway_intermediate" in str(
            data["typed_transient_assignment_sequence"]
        )
        if route_identity_class == "direct_only_target_signature_reached":
            relation_class = "direct_source_component_to_target_signature"
            object_relation_status = "direct_signature_relation_not_clean_object_wall"
        elif route_identity_class == "recovery_loop_signature_ladder_reversible_to_source_family":
            relation_class = "recovery_ladder_source_target_transient_return"
            object_relation_status = "ladder_relation_not_endpoint_object_wall"
        else:
            relation_class = "other_signature_relation"
            object_relation_status = "diagnostic_relation_only"
        source_signature = source_signatures.get(source_signature_id, "")
        rows.append(
            {
                "route_contract_id": str(data["route_contract_id"]),
                "local_pair_id": str(data["local_pair_id"]),
                "branch": str(data["branch"]),
                "start_condition": str(data["start_condition"]),
                "planned_route_family": str(data["planned_route_family"]),
                "seed": _as_int(data["seed"]),
                "route_identity_class": route_identity_class,
                "source_component_signature_id": source_signature_id,
                "target_signature_id": TARGET_SIGNATURE_ID if target_seen else "",
                "transient_signature_id": TRANSIENT_SIGNATURE_ID if transient_seen else "",
                "return_signature_id": return_signature_id,
                "source_target_partition_coassignment_distance": float(
                    1.0 - _partition_jaccard(source_signature, target_signature)
                )
                if source_signature
                else float("nan"),
                "target_seen_in_route": target_seen,
                "transient_seen_in_route": transient_seen,
                "relation_class": relation_class,
                "object_relation_status": object_relation_status,
                "clean_source_to_exclusive_target_object_relation": False,
                "source_family_object_unified": False,
                "target_local_object_certified": True,
                "transient_endpoint_object_certified": False,
                "object_identity_resolved": False,
                "local_object_wall_relation_ready": False,
                "endpoint_assignment_sequence": endpoint_sequence,
                "typed_transient_assignment_sequence": str(
                    data["typed_transient_assignment_sequence"]
                ),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _evidence_rows(
    *,
    scope_rows: pd.DataFrame,
    local_object_rows: pd.DataFrame,
    relation_rows: pd.DataFrame,
    signature_summary: dict[str, Any],
    symmetric_summary: dict[str, Any],
) -> pd.DataFrame:
    target_row = local_object_rows[
        local_object_rows["signature_id"].astype(str).eq(TARGET_SIGNATURE_ID)
    ].iloc[0]
    transient_row = local_object_rows[
        local_object_rows["signature_id"].astype(str).eq(TRANSIENT_SIGNATURE_ID)
    ].iloc[0]
    source_rows = local_object_rows[
        local_object_rows["signature_id"].astype(str).isin(
            {SOURCE_SIGNATURE_ID, SOURCE_GUARD_SIGNATURE_ID}
        )
    ]
    external_scope_rows = scope_rows[
        scope_rows["scope_id"]
        .astype(str)
        .isin(
            {
                "S1_existing_symmetric_endpoint_object_scope",
                "S2_symmetric_object_tables_confirm_absence",
            }
        )
    ]
    direct_count = int(
        relation_rows["relation_class"]
        .astype(str)
        .eq("direct_source_component_to_target_signature")
        .sum()
    )
    recovery_count = int(
        relation_rows["relation_class"]
        .astype(str)
        .eq("recovery_ladder_source_target_transient_return")
        .sum()
    )
    return pd.DataFrame(
        [
            {
                "evidence_id": "E1_external_symmetric_membership_absent",
                "evidence_question": "Does an existing symmetric endpoint-object audit certify 016?",
                "observed": {
                    "existing_audited_pair_ids": symmetric_summary["audited_pair_ids"],
                    "contains_local_pair_016": bool(
                        external_scope_rows["contains_local_pair_016"]
                        .map(_as_bool)
                        .any()
                    ),
                    "membership_available_for_local_pair_016": bool(
                        external_scope_rows[
                            "membership_available_for_local_pair_016"
                        ].map(_as_bool).any()
                    ),
                },
                "evidence_status": "blocks_external_endpoint_object_certificate",
                "claim_effect": "object_identity_not_resolved_by_existing_membership",
            },
            {
                "evidence_id": "E2_local_signature_object_certificate_available",
                "evidence_question": "Can all four stable 016 signatures be materialized as local signature objects?",
                "observed": {
                    "expected_signature_ids": sorted(EXPECTED_SIGNATURE_IDS),
                    "certified_signature_ids": sorted(
                        local_object_rows.loc[
                            local_object_rows["local_signature_object_certified"].map(
                                _as_bool
                            ),
                            "signature_id",
                        ].astype(str)
                    ),
                },
                "evidence_status": "local_signature_object_certificate_available",
                "claim_effect": "supports_local_object_vocabulary_only",
            },
            {
                "evidence_id": "E3_target_local_object_certified",
                "evidence_question": "Is the target anchor locally certified?",
                "observed": {
                    "target_signature_id": TARGET_SIGNATURE_ID,
                    "local_ablation_run_count": int(
                        target_row["local_ablation_run_count"]
                    ),
                    "direct_source_to_target_relation_count": direct_count,
                    "prior_target_signature_match": bool(
                        target_row["prior_target_signature_match"]
                    ),
                },
                "evidence_status": "target_local_object_certified",
                "claim_effect": "target_anchor_available_but_not_wall",
            },
            {
                "evidence_id": "E4_transient_nonendpoint_blocker",
                "evidence_question": "Does the transient remain a typed non-endpoint object blocker?",
                "observed": {
                    "transient_signature_id": TRANSIENT_SIGNATURE_ID,
                    "support_incompatibility_rows": int(
                        transient_row["support_incompatibility_rows"]
                    ),
                    "prior_transient_signature_match": bool(
                        transient_row["prior_transient_signature_match"]
                    ),
                    "recovery_ladder_relation_count": recovery_count,
                },
                "evidence_status": "typed_transient_certified_nonendpoint",
                "claim_effect": "blocks_endpoint_object_identity_and_wall_language",
            },
            {
                "evidence_id": "E5_source_family_split_blocker",
                "evidence_question": "Is the source family unified into one source object?",
                "observed": {
                    "source_signature_ids": sorted(source_rows["signature_id"].astype(str)),
                    "source_signature_count": int(len(source_rows)),
                    "source_family_object_unified": False,
                },
                "evidence_status": "source_family_split_not_unified_object",
                "claim_effect": "blocks_clean_source_object_relation",
            },
            {
                "evidence_id": "E6_relation_evidence_not_clean_object_wall",
                "evidence_question": "Do route relations form a clean source-object to exclusive target-object wall?",
                "observed": {
                    "relation_class_counts": _count_dict(relation_rows["relation_class"]),
                    "clean_object_relation_count": int(
                        relation_rows[
                            "clean_source_to_exclusive_target_object_relation"
                        ].map(_as_bool).sum()
                    ),
                    "signature_summary_route_class_counts": signature_summary.get(
                        "route_identity_class_counts"
                    ),
                },
                "evidence_status": "relation_evidence_is_signature_ladder_not_object_wall",
                "claim_effect": "keeps_local_object_wall_evidence_closed",
            },
            {
                "evidence_id": "E7_claim_boundary",
                "evidence_question": "Are route, wall, method, quality/cost, and replay claims still closed?",
                "observed": CLAIM_BOUNDARY,
                "evidence_status": "claims_closed",
                "claim_effect": "prevents_label_or_method_promotion",
            },
        ]
    )


def _decision_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_id": "D1",
                "decision": "external_symmetric_endpoint_object_membership_unavailable_for_016",
                "rationale": "The existing symmetric endpoint-object audit is scoped to local_pair_014 and local_pair_005, so it cannot certify 016 object membership.",
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            },
            {
                "decision_id": "D2",
                "decision": "local_signature_object_certificate_available",
                "rationale": "The four 016 signatures are materialized as local signature objects with roles and partition stats.",
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            },
            {
                "decision_id": "D3",
                "decision": "target_local_object_certified_but_wall_not_open",
                "rationale": "The target signature has local drop-bridge provenance and direct target relations, but source/transient object blockers remain.",
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            },
            {
                "decision_id": "D4",
                "decision": "object_identity_not_resolved",
                "rationale": "The source family is split and the recurrent transient is a typed non-endpoint state, so endpoint-object identity is not fully resolved.",
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            },
            {
                "decision_id": "D5",
                "decision": "do_not_expand_routes_or_promote_labels",
                "rationale": "The current blocker is the object-surface definition, not missing transfer-route execution.",
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            },
            {
                "decision_id": "D6",
                "decision": "next_gate_choose_object_surface_rule",
                "rationale": "The next work should decide whether local signature-objects are an acceptable primitive object surface or whether 016 needs a true symmetric endpoint-object membership audit.",
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            },
        ]
    )


def _gate_matrix(
    *,
    scope_rows: pd.DataFrame,
    local_object_rows: pd.DataFrame,
    relation_rows: pd.DataFrame,
    context: dict[str, Any],
) -> pd.DataFrame:
    signature_summary = context["signature_summary"]
    signature_gates = context["signature_gates"]
    symmetric_summary = context["symmetric_summary"]
    external_membership_available = bool(
        scope_rows["membership_available_for_local_pair_016"].map(_as_bool).any()
    )
    certified_ids = set(
        local_object_rows.loc[
            local_object_rows["local_signature_object_certified"].map(_as_bool),
            "signature_id",
        ].astype(str)
    )
    target_local_object_certified = bool(
        local_object_rows.loc[
            local_object_rows["signature_id"].astype(str).eq(TARGET_SIGNATURE_ID),
            "target_local_object_certified",
        ].map(_as_bool).all()
    )
    transient_row = local_object_rows[
        local_object_rows["signature_id"].astype(str).eq(TRANSIENT_SIGNATURE_ID)
    ].iloc[0]
    source_rows = local_object_rows[
        local_object_rows["signature_id"].astype(str).isin(
            {SOURCE_SIGNATURE_ID, SOURCE_GUARD_SIGNATURE_ID}
        )
    ]
    clean_relation_count = int(
        relation_rows["clean_source_to_exclusive_target_object_relation"].map(
            _as_bool
        ).sum()
    )
    route_relation_classes = _count_dict(relation_rows["relation_class"])
    rows = [
        _gate_row(
            "G1_upstream_signature_identity_resolved",
            "Did the upstream 016 signature audit resolve signature identity?",
            {
                "signature_failed_gates": signature_summary.get("failed_gates"),
                "signature_identity_resolved": signature_summary.get(
                    "signature_identity_resolved"
                ),
                "signature_gate_status_counts": _count_dict(signature_gates["gate_status"]),
            },
            "upstream signature audit has no failed gates and signature_identity_resolved=true",
            not signature_summary.get("failed_gates")
            and bool(signature_summary.get("signature_identity_resolved")),
        ),
        _gate_row(
            "G2_external_symmetric_membership_absent_for_016",
            "Is the existing symmetric endpoint-object audit unavailable for 016?",
            {
                "symmetric_audited_pair_ids": symmetric_summary["audited_pair_ids"],
                "external_membership_available": external_membership_available,
            },
            "existing symmetric audit is scoped away from 016; blocker named",
            not external_membership_available
            and POSITIVE_PAIR_ID not in set(symmetric_summary["audited_pair_ids"]),
        ),
        _gate_row(
            "G3_local_signature_objects_materialized",
            "Were all four expected local signature objects materialized?",
            {
                "expected_signature_ids": sorted(EXPECTED_SIGNATURE_IDS),
                "certified_signature_ids": sorted(certified_ids),
                "local_object_row_count": int(len(local_object_rows)),
            },
            "exactly the four expected signatures are locally certified as signature objects",
            certified_ids == EXPECTED_SIGNATURE_IDS and len(local_object_rows) == 4,
        ),
        _gate_row(
            "G4_target_local_object_certified",
            "Is the drop-bridge target anchor locally certified?",
            {
                "target_signature_id": TARGET_SIGNATURE_ID,
                "target_local_object_certified": target_local_object_certified,
                "target_relation_count": int(
                    relation_rows["target_seen_in_route"].map(_as_bool).sum()
                ),
            },
            "target local object certified and target appears in all 48 route relations",
            target_local_object_certified
            and int(relation_rows["target_seen_in_route"].map(_as_bool).sum()) == 48,
        ),
        _gate_row(
            "G5_transient_endpoint_blocker_named",
            "Is the recurrent transient explicitly kept as a non-endpoint blocker?",
            {
                "transient_signature_id": TRANSIENT_SIGNATURE_ID,
                "support_incompatibility_rows": int(
                    transient_row["support_incompatibility_rows"]
                ),
                "transient_endpoint_object_certified": bool(
                    transient_row["transient_endpoint_object_certified"]
                ),
                "prior_transient_signature_match": bool(
                    transient_row["prior_transient_signature_match"]
                ),
            },
            "transient has 48 support-incompatible rows and no endpoint object certification",
            int(transient_row["support_incompatibility_rows"]) == 48
            and not bool(transient_row["transient_endpoint_object_certified"])
            and bool(transient_row["prior_transient_signature_match"]),
        ),
        _gate_row(
            "G6_source_family_split_blocker_named",
            "Is the source family split named rather than collapsed into one object?",
            {
                "source_signature_ids": sorted(source_rows["signature_id"].astype(str)),
                "source_family_object_unified": bool(
                    source_rows["source_family_object_unified"].map(_as_bool).all()
                ),
            },
            "two source-family signatures are present and source_family_object_unified=false",
            sorted(source_rows["signature_id"].astype(str).tolist())
            == sorted([SOURCE_SIGNATURE_ID, SOURCE_GUARD_SIGNATURE_ID])
            and not bool(source_rows["source_family_object_unified"].map(_as_bool).any()),
        ),
        _gate_row(
            "G7_relation_evidence_not_object_wall",
            "Are the relations classified as signature transfer/ladder evidence rather than a clean object wall?",
            {
                "route_relation_classes": route_relation_classes,
                "clean_relation_count": clean_relation_count,
            },
            "24 direct signature relations, 24 recovery ladder relations, and zero clean object-wall relations",
            route_relation_classes.get("direct_source_component_to_target_signature", 0) == 24
            and route_relation_classes.get(
                "recovery_ladder_source_target_transient_return", 0
            )
            == 24
            and clean_relation_count == 0,
        ),
        _gate_row(
            "G8_claims_closed",
            "Are object-wall, pathway, method, quality/cost, replay, and route-expansion claims closed?",
            CLAIM_BOUNDARY,
            "all promotion flags remain false and local_object_wall_evidence_audit_ready=false",
            True,
        ),
    ]
    return pd.DataFrame(rows)


def _report(
    *,
    summary: dict[str, Any],
    scope_rows: pd.DataFrame,
    local_object_rows: pd.DataFrame,
    relation_rows: pd.DataFrame,
    evidence_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# NanoClustering G4.8 First-Pass 016 Object-Identity Certificate",
            "",
            f"- status: `{summary['status']}`",
            f"- external_symmetric_endpoint_object_membership_available: {summary['external_symmetric_endpoint_object_membership_available']}",
            f"- local_signature_object_certificate_available: {summary['local_signature_object_certificate_available']}",
            f"- target_local_object_certified: {summary['target_local_object_certified']}",
            f"- transient_endpoint_object_certified: {summary['transient_endpoint_object_certified']}",
            f"- source_family_object_unified: {summary['source_family_object_unified']}",
            f"- object_identity_resolved: {summary['object_identity_resolved']}",
            f"- local_object_wall_evidence_audit_ready: {summary['local_object_wall_evidence_audit_ready']}",
            f"- gate_status_counts: {summary['gate_status_counts']}",
            f"- failed_gates: {summary['failed_gates']}",
            f"- interpretation: {summary['interpretation']}",
            f"- recommended_next_gate: {summary['recommended_next_gate']}",
            f"- claim_boundary: {summary['claim_boundary']}",
            "",
            "## Scope Rows",
            "",
            _markdown_table(
                scope_rows,
                [
                    "scope_id",
                    "contains_local_pair_016",
                    "membership_available_for_local_pair_016",
                    "scope_status",
                    "claim_effect",
                ],
            ),
            "",
            "## Local Object Rows",
            "",
            _markdown_table(
                local_object_rows,
                [
                    "signature_id",
                    "primary_signature_role",
                    "local_object_role",
                    "local_object_certificate_status",
                    "local_signature_object_certified",
                    "endpoint_object_certified",
                    "target_local_object_certified",
                    "transient_endpoint_object_certified",
                    "source_family_object_unified",
                    "trace_row_count",
                    "local_ablation_run_count",
                    "support_incompatibility_rows",
                    "cluster_count",
                    "coassigned_pair_count",
                ],
            ),
            "",
            "## Relation Rows",
            "",
            _markdown_table(
                relation_rows,
                [
                    "planned_route_family",
                    "start_condition",
                    "seed",
                    "relation_class",
                    "source_component_signature_id",
                    "target_signature_id",
                    "transient_signature_id",
                    "return_signature_id",
                    "source_target_partition_coassignment_distance",
                    "object_relation_status",
                ],
                max_rows=48,
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
            "This certificate makes the current object surface explicit. It does not",
            "turn signature transfer into an object wall: 016 still lacks external",
            "symmetric endpoint-object membership, has a split source family, and",
            "contains a recurrent typed non-endpoint transient.",
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    context = _load_context(args)
    scope_rows = _scope_rows(context)
    local_object_rows = _local_object_rows(
        signature_rows=context["signature_rows"],
        local_ablation_signature_rows=context["local_ablation_signature_rows"],
    )
    relation_rows = _relation_rows(
        route_rows=context["signature_route_rows"],
        local_object_rows=local_object_rows,
    )
    evidence_rows = _evidence_rows(
        scope_rows=scope_rows,
        local_object_rows=local_object_rows,
        relation_rows=relation_rows,
        signature_summary=context["signature_summary"],
        symmetric_summary=context["symmetric_summary"],
    )
    decision_rows = _decision_rows()
    gate_matrix = _gate_matrix(
        scope_rows=scope_rows,
        local_object_rows=local_object_rows,
        relation_rows=relation_rows,
        context=context,
    )

    failed_gates = gate_matrix.loc[
        gate_matrix["gate_status"].astype(str).eq("fail"), "gate_id"
    ].astype(str).tolist()
    external_membership_available = bool(
        scope_rows["membership_available_for_local_pair_016"].map(_as_bool).any()
    )
    local_signature_object_certificate_available = bool(
        set(
            local_object_rows.loc[
                local_object_rows["local_signature_object_certified"].map(_as_bool),
                "signature_id",
            ].astype(str)
        )
        == EXPECTED_SIGNATURE_IDS
    )
    target_local_object_certified = bool(
        local_object_rows.loc[
            local_object_rows["signature_id"].astype(str).eq(TARGET_SIGNATURE_ID),
            "target_local_object_certified",
        ].map(_as_bool).all()
    )
    transient_endpoint_object_certified = bool(
        local_object_rows.loc[
            local_object_rows["signature_id"].astype(str).eq(TRANSIENT_SIGNATURE_ID),
            "transient_endpoint_object_certified",
        ].map(_as_bool).all()
    )
    source_family_object_unified = bool(
        local_object_rows.loc[
            local_object_rows["signature_id"].astype(str).isin(
                {SOURCE_SIGNATURE_ID, SOURCE_GUARD_SIGNATURE_ID}
            ),
            "source_family_object_unified",
        ].map(_as_bool).any()
    )
    object_identity_resolved = bool(
        external_membership_available
        and local_signature_object_certificate_available
        and target_local_object_certified
        and transient_endpoint_object_certified
        and source_family_object_unified
    )
    local_object_wall_ready = bool(object_identity_resolved and not failed_gates)
    summary = {
        "schema": "nanoclustering_g4_8_first_pass_016_object_identity_certificate_summary.v1",
        "status": RUN_STATUS,
        "signature_identity_dir": str(args.signature_identity_dir.resolve()),
        "symmetric_endpoint_object_dir": str(
            args.symmetric_endpoint_object_dir.resolve()
        ),
        "output_dir": str(args.output_dir.resolve()),
        "scope_row_count": int(len(scope_rows)),
        "local_object_row_count": int(len(local_object_rows)),
        "relation_row_count": int(len(relation_rows)),
        "evidence_row_count": int(len(evidence_rows)),
        "decision_row_count": int(len(decision_rows)),
        "gate_status_counts": _count_dict(gate_matrix["gate_status"]),
        "failed_gates": failed_gates,
        "external_symmetric_endpoint_object_membership_available": bool(
            external_membership_available
        ),
        "local_signature_object_certificate_available": bool(
            local_signature_object_certificate_available
        ),
        "target_local_object_certified": bool(target_local_object_certified),
        "transient_endpoint_object_certified": bool(
            transient_endpoint_object_certified
        ),
        "source_family_object_unified": bool(source_family_object_unified),
        "object_identity_resolved": bool(object_identity_resolved),
        "local_object_wall_evidence_audit_ready": bool(local_object_wall_ready),
        "local_object_certificate_status_counts": _count_dict(
            local_object_rows["local_object_certificate_status"]
        ),
        "relation_class_counts": _count_dict(relation_rows["relation_class"]),
        "object_relation_status_counts": _count_dict(
            relation_rows["object_relation_status"]
        ),
        "wall_claim_ready_pairs": [],
        "interpretation": (
            "The 016 object surface is now explicit: the four stable signatures can "
            "be treated as local signature objects, and the target anchor has local "
            "object provenance. However, existing symmetric endpoint-object "
            "membership does not cover 016, the source family is split into strict "
            "and guard components, and the recurrent transient is a typed "
            "non-endpoint object blocker. Object identity and local object-wall "
            "evidence remain closed."
        ),
        "recommended_next_gate": (
            "Choose the object-surface rule before more route work: either accept "
            "local signature-objects as a primitive diagnostic surface, or build a "
            "true symmetric endpoint-object membership audit for 016. Do not expand "
            "routes or promote wall/pathway labels until that rule is fixed."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "upstream_signature_claim_boundary": SIGNATURE_CLAIM_BOUNDARY,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(scope_rows, args.output_dir / SCOPE_ROWS_CSV)
    _write_csv(local_object_rows, args.output_dir / LOCAL_OBJECT_ROWS_CSV)
    _write_csv(relation_rows, args.output_dir / RELATION_ROWS_CSV)
    _write_csv(evidence_rows, args.output_dir / EVIDENCE_ROWS_CSV)
    _write_csv(decision_rows, args.output_dir / DECISION_ROWS_CSV)
    _write_csv(gate_matrix, args.output_dir / GATE_MATRIX_CSV)
    (args.output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_016_object_identity_certificate_config.v1",
        "positive_pair_id": POSITIVE_PAIR_ID,
        "expected_signature_ids": sorted(EXPECTED_SIGNATURE_IDS),
        "signature_identity_dir": str(args.signature_identity_dir.resolve()),
        "symmetric_endpoint_object_dir": str(
            args.symmetric_endpoint_object_dir.resolve()
        ),
        "output_dir": str(args.output_dir.resolve()),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (args.output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / REPORT_MD).write_text(
        _report(
            summary=summary,
            scope_rows=scope_rows,
            local_object_rows=local_object_rows,
            relation_rows=relation_rows,
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
        "--signature-identity-dir",
        type=Path,
        default=DEFAULT_SIGNATURE_IDENTITY_DIR,
        help="Directory containing the 016 object/signature identity-resolution audit.",
    )
    parser.add_argument(
        "--symmetric-endpoint-object-dir",
        type=Path,
        default=DEFAULT_SYMMETRIC_ENDPOINT_OBJECT_DIR,
        help="Directory containing the first-pass symmetric endpoint-object audit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for this object-identity certificate audit.",
    )
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
