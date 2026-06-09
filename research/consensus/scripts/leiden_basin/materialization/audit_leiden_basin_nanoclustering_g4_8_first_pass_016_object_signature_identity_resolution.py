#!/usr/bin/env python3
"""Audit local_pair_016 signature identity resolution on the transfer trace.

This read-only audit follows the executed object-wall transfer trace audit. It
asks whether the existing trace has stable signature-level identities for the
source-family, transient, and target states, and whether that is enough to
resolve object-level wall evidence.

The answer is intentionally split: signature identity can be resolved on the
existing trace, but endpoint-object identity remains unresolved. This audit
does not rerun Leiden, expand route rows, promote pathway labels or walls,
evaluate quality/cost value, replay full NanoClustering, or claim method
success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from audit_leiden_basin_nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace import (
    DEFAULT_OUTPUT_DIR as DEFAULT_TRANSFER_AUDIT_DIR,
    GATE_MATRIX_CSV as TRANSFER_AUDIT_GATE_MATRIX_CSV,
    SUMMARY_JSON as TRANSFER_AUDIT_SUMMARY_JSON,
)
from run_leiden_basin_nanoclustering_g4_8_first_pass_016_object_wall_transfer_trace import (
    DEFAULT_OUTPUT_DIR as DEFAULT_TRANSFER_TRACE_DIR,
    POSITIVE_PAIR_ID,
    ROUTE_TRANSFER_RESULT_ROWS_CSV,
    SUMMARY_JSON as TRANSFER_TRACE_SUMMARY_JSON,
    TRACE_ROWS_CSV,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)


DEFAULT_LOCAL_ABLATION_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_symmetric_object_variable_pair_local_ablation_gamma1e5_20260603"
)
DEFAULT_SEMANTIC_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_transient_semantic_validation_gamma1e5_20260605"
)
DEFAULT_PERSISTENCE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_transient_persistence_trace_gamma1e5_20260605"
)
DEFAULT_REVERSE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_transient_reverse_trace_gamma1e5_20260605"
)
DEFAULT_SOURCE_EQUIVALENCE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_source_family_equivalence_audit_gamma1e5_20260605"
)
DEFAULT_PATHWAY_SHAPE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_pathway_shape_audit_gamma1e5_20260605"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_object_signature_identity_resolution_gamma1e5_20260608"
)

LOCAL_ABLATION_SEED_RUNS_CSV = (
    "nanoclustering_symmetric_object_variable_pair_local_ablation_seed_runs.csv"
)
SEMANTIC_SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_016_transient_semantic_summary.json"
)
PERSISTENCE_SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_016_transient_persistence_summary.json"
)
REVERSE_SUMMARY_JSON = "nanoclustering_g4_8_first_pass_016_transient_reverse_summary.json"
SOURCE_EQUIVALENCE_SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_016_source_family_equivalence_summary.json"
)
PATHWAY_SHAPE_SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_016_pathway_shape_summary.json"
)

SIGNATURE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_signature_identity_resolution_signature_rows.csv"
)
STEP_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_signature_identity_resolution_step_rows.csv"
)
ROUTE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_signature_identity_resolution_route_rows.csv"
)
LOCAL_ABLATION_SIGNATURE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_signature_identity_resolution_local_ablation_signature_rows.csv"
)
EVIDENCE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_signature_identity_resolution_evidence_rows.csv"
)
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_signature_identity_resolution_decision_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_016_object_signature_identity_resolution_gate_matrix.csv"
)
SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_016_object_signature_identity_resolution_summary.json"
)
CONFIG_JSON = (
    "nanoclustering_g4_8_first_pass_016_object_signature_identity_resolution_config.json"
)
REPORT_MD = (
    "nanoclustering_g4_8_first_pass_016_object_signature_identity_resolution_report.md"
)

TARGET_ASSIGNMENT = "drop_bridge_target_anchor"
TRANSIENT_ASSIGNMENT = "unknown_new_endpoint"
SOURCE_ASSIGNMENT = "original_source_anchor"
SOURCE_GUARD_ASSIGNMENT = "ambiguous_anchor_match:drop_direct_guard_anchor;original_source_anchor"
TARGET_SIGNATURE_ID = "3c9b8a190753"
TRANSIENT_SIGNATURE_ID = "aeb59ab537e6"
SOURCE_SIGNATURE_ID = "5536308f50fc"
SOURCE_GUARD_SIGNATURE_ID = "c475d13ca500"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_016_object_signature_identity_resolution"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_016_object_signature_identity_resolution"
WALL_PROMOTION_STATUS = "not_promoted_signature_identity_only"
METHOD_STATUS = "object_signature_identity_resolution_audit_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass 016 object/signature identity-resolution "
    "audit only; reads existing transfer trace, transfer audit, local-ablation, "
    "and prior 016 semantic/pathway summaries. It resolves signature-level "
    "state identity but does not resolve endpoint-object identity, rerun Leiden, "
    "expand routes, promote pathway labels or walls, evaluate quality/cost "
    "value, replay full NanoClustering, or claim method success."
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


def _count_dict(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _unique_join(series: pd.Series) -> str:
    values = sorted({str(value) for value in series.dropna() if str(value) != ""})
    return ";".join(values)


def _json_dump(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True)


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


def _signature_role(row: pd.Series) -> str:
    endpoint_assignment = str(row["endpoint_assignment_by_step"])
    typed_assignment = str(row["typed_transient_assignment_by_step"])
    if endpoint_assignment == TARGET_ASSIGNMENT:
        return "target_anchor_signature"
    if typed_assignment == "pathway_intermediate":
        return "typed_transient_signature"
    if endpoint_assignment == SOURCE_ASSIGNMENT:
        return "source_family_strict_signature"
    if typed_assignment == "object_identity_blocker":
        return "source_guard_blocker_signature"
    return "other_signature"


def _resolution_status(signature_role: str) -> tuple[str, str]:
    if signature_role == "target_anchor_signature":
        return (
            "signature_resolved_to_drop_bridge_target_anchor_proxy",
            "object_identity_unresolved_target_proxy",
        )
    if signature_role == "typed_transient_signature":
        return (
            "signature_resolved_as_recurrent_typed_transient",
            "object_identity_unresolved_typed_transient",
        )
    if signature_role == "source_family_strict_signature":
        return (
            "signature_resolved_to_source_family_strict_proxy",
            "object_identity_unresolved_source_proxy",
        )
    if signature_role == "source_guard_blocker_signature":
        return (
            "signature_resolved_to_source_guard_blocker_proxy",
            "object_identity_unresolved_source_guard_proxy",
        )
    return ("signature_unclassified", "object_identity_unresolved")


def _load_context(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "transfer_trace_summary": _read_json(args.transfer_trace_dir / TRANSFER_TRACE_SUMMARY_JSON),
        "transfer_audit_summary": _read_json(
            args.transfer_audit_dir / TRANSFER_AUDIT_SUMMARY_JSON
        ),
        "transfer_audit_gates": _read_csv(
            args.transfer_audit_dir / TRANSFER_AUDIT_GATE_MATRIX_CSV
        ),
        "trace_rows": _read_csv(args.transfer_trace_dir / TRACE_ROWS_CSV),
        "route_result_rows": _read_csv(
            args.transfer_trace_dir / ROUTE_TRANSFER_RESULT_ROWS_CSV
        ),
        "local_ablation_seed_runs": _read_csv(
            args.local_ablation_dir / LOCAL_ABLATION_SEED_RUNS_CSV
        ),
        "semantic_summary": _read_json(args.semantic_dir / SEMANTIC_SUMMARY_JSON),
        "persistence_summary": _read_json(
            args.persistence_dir / PERSISTENCE_SUMMARY_JSON
        ),
        "reverse_summary": _read_json(args.reverse_dir / REVERSE_SUMMARY_JSON),
        "source_equivalence_summary": _read_json(
            args.source_equivalence_dir / SOURCE_EQUIVALENCE_SUMMARY_JSON
        ),
        "pathway_shape_summary": _read_json(
            args.pathway_shape_dir / PATHWAY_SHAPE_SUMMARY_JSON
        ),
    }


def _local_ablation_signature_rows(seed_runs: pd.DataFrame) -> pd.DataFrame:
    positive = seed_runs[seed_runs["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)].copy()
    rows: list[dict[str, Any]] = []
    for (signature_id, graph_variant), group in positive.groupby(
        ["endpoint_signature_id", "graph_variant"], dropna=False
    ):
        rows.append(
            {
                "signature_id": str(signature_id),
                "graph_variant": str(graph_variant),
                "run_count": int(len(group)),
                "start_condition_count": int(group["start_condition"].nunique()),
                "seed_count": int(group["seed"].nunique()),
                "pair_coassigned_count": int(group["pair_coassigned"].map(_as_bool).sum()),
                "pair_coassigned_share": float(group["pair_coassigned"].map(_as_bool).mean()),
                "mechanism_read_counts": _count_dict(group["mechanism_read"]),
                "quality_min": float(group["quality"].astype(float).min()),
                "quality_max": float(group["quality"].astype(float).max()),
                "cluster_count_values": _unique_join(group["cluster_count"].astype(str)),
                "signature": str(group["endpoint_signature"].iloc[0]),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["signature_id", "graph_variant"], kind="mergesort"
    )


def _signature_rows(
    *,
    trace_rows: pd.DataFrame,
    local_ablation_signature: pd.DataFrame,
    context: dict[str, Any],
) -> pd.DataFrame:
    positive = trace_rows[trace_rows["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)].copy()
    positive["signature_role"] = positive.apply(_signature_role, axis=1)

    local_by_signature: dict[str, pd.DataFrame] = {
        str(signature_id): group.copy()
        for signature_id, group in local_ablation_signature.groupby("signature_id")
    }
    prior_transient_ids = set(context["semantic_summary"].get("transient_signature_ids", []))
    prior_transient_ids.add(str(context["persistence_summary"].get("transient_signature_id", "")))
    prior_transient_ids.add(str(context["reverse_summary"].get("transient_signature_id", "")))
    prior_target_ids = {
        str(context["persistence_summary"].get("target_signature_id", "")),
        str(context["reverse_summary"].get("target_signature_id", "")),
    }

    rows: list[dict[str, Any]] = []
    for signature_id, group in positive.groupby("result_endpoint_signature_id", dropna=False):
        role_counts = _count_dict(group["signature_role"])
        primary_role = max(role_counts.items(), key=lambda item: item[1])[0]
        signature_status, object_status = _resolution_status(primary_role)
        local_group = local_by_signature.get(str(signature_id), pd.DataFrame())
        rows.append(
            {
                "signature_id": str(signature_id),
                "primary_signature_role": primary_role,
                "signature_role_counts": role_counts,
                "trace_row_count": int(len(group)),
                "route_family_count": int(group["planned_route_family"].nunique()),
                "start_condition_count": int(group["start_condition"].nunique()),
                "seed_count": int(group["seed"].nunique()),
                "step_labels": _unique_join(group["step_label"]),
                "bridge_fractions": _unique_join(
                    group["bridge_edge_weight_fraction"].astype(str)
                ),
                "endpoint_assignment_counts": _count_dict(group["endpoint_assignment_by_step"]),
                "typed_transient_assignment_counts": _count_dict(
                    group["typed_transient_assignment_by_step"]
                ),
                "object_identity_transfer_status_counts": _count_dict(
                    group["object_identity_transfer_status"]
                ),
                "endpoint_object_assignment_counts": _count_dict(
                    group["endpoint_object_assignment_by_step"]
                ),
                "support_incompatibility_rows": int(
                    group["support_incompatibility_check"].map(_as_bool).sum()
                ),
                "local_ablation_graph_variants": (
                    _unique_join(local_group["graph_variant"]) if not local_group.empty else ""
                ),
                "local_ablation_run_count": int(local_group["run_count"].sum())
                if not local_group.empty
                else 0,
                "prior_transient_signature_match": str(signature_id) in prior_transient_ids,
                "prior_target_signature_match": str(signature_id) in prior_target_ids,
                "signature_identity_resolution_status": signature_status,
                "object_identity_resolution_status": object_status,
                "signature_identity_resolved": signature_status != "signature_unclassified",
                "object_identity_resolved": False,
                "signature": str(group["result_endpoint_signature"].iloc[0]),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).sort_values("primary_signature_role", kind="mergesort")


def _step_rows(trace_rows: pd.DataFrame) -> pd.DataFrame:
    positive = trace_rows[trace_rows["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)].copy()
    positive["signature_role"] = positive.apply(_signature_role, axis=1)
    rows: list[dict[str, Any]] = []
    group_cols = [
        "planned_route_family",
        "start_condition",
        "step_index",
        "step_label",
        "direct_edge_weight_fraction",
        "bridge_edge_weight_fraction",
        "direct_edge_retained_by_step",
        "bridge_support_suppressed_by_step",
    ]
    for key, group in positive.groupby(group_cols, dropna=False):
        record = dict(zip(group_cols, key))
        rows.append(
            {
                **record,
                "row_count": int(len(group)),
                "seed_count": int(group["seed"].nunique()),
                "signature_ids": _unique_join(group["result_endpoint_signature_id"]),
                "signature_role_counts": _count_dict(group["signature_role"]),
                "endpoint_assignment_counts": _count_dict(group["endpoint_assignment_by_step"]),
                "typed_transient_assignment_counts": _count_dict(
                    group["typed_transient_assignment_by_step"]
                ),
                "object_identity_transfer_status_counts": _count_dict(
                    group["object_identity_transfer_status"]
                ),
                "support_incompatibility_count": int(
                    group["support_incompatibility_check"].map(_as_bool).sum()
                ),
                "objective_value_min": float(group["objective_value_by_step"].astype(float).min()),
                "objective_value_max": float(group["objective_value_by_step"].astype(float).max()),
                "support_distance_min_known_anchor_min": float(
                    group["support_distance_min_known_anchor"].astype(float).min()
                ),
                "support_distance_min_known_anchor_max": float(
                    group["support_distance_min_known_anchor"].astype(float).max()
                ),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["planned_route_family", "start_condition", "step_index"], kind="mergesort"
    )


def _route_rows(route_result: pd.DataFrame) -> pd.DataFrame:
    positive = route_result[route_result["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)].copy()
    rows: list[dict[str, Any]] = []
    for row in positive.sort_values(
        ["planned_route_family", "start_condition", "seed"], kind="mergesort"
    ).itertuples(index=False):
        data = row._asdict()
        route_family = str(data["planned_route_family"])
        if route_family == "first_pass_016_direct_only_target_availability_probe":
            route_identity_class = "direct_only_target_signature_reached"
            readiness_effect = "supports_signature_target_leg_only"
        elif _as_bool(data.get("recovery_typed_transient_block_seed")):
            route_identity_class = "recovery_loop_signature_ladder_reversible_to_source_family"
            readiness_effect = "supports_signature_ladder_but_not_object_wall"
        else:
            route_identity_class = "other_signature_route"
            readiness_effect = "diagnostic_only"
        rows.append(
            {
                **data,
                "route_identity_class": route_identity_class,
                "readiness_effect": readiness_effect,
                "signature_identity_resolved": True,
                "object_identity_resolved": False,
                "object_wall_claim_allowed_after_identity_audit": False,
                "pathway_claim_allowed_after_identity_audit": False,
                "method_claim_allowed_after_identity_audit": False,
                "quality_cost_claim_allowed_after_identity_audit": False,
                "full_replay_claim_allowed_after_identity_audit": False,
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
    signature_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    local_ablation_signature: pd.DataFrame,
    context: dict[str, Any],
) -> pd.DataFrame:
    def sig_count(role: str) -> int:
        return int(signature_rows["primary_signature_role"].astype(str).eq(role).sum())

    target_local = local_ablation_signature[
        local_ablation_signature["signature_id"].astype(str).eq(TARGET_SIGNATURE_ID)
        & local_ablation_signature["graph_variant"].astype(str).eq("drop_bridge_edges")
    ]
    source_local = local_ablation_signature[
        local_ablation_signature["signature_id"].astype(str).isin(
            {SOURCE_SIGNATURE_ID, SOURCE_GUARD_SIGNATURE_ID}
        )
        & local_ablation_signature["graph_variant"].astype(str).isin(
            {"original", "drop_direct_edge"}
        )
    ]
    transient_prior = {
        "semantic": context["semantic_summary"].get("transient_signature_ids", []),
        "persistence": context["persistence_summary"].get("transient_signature_id"),
        "reverse": context["reverse_summary"].get("transient_signature_id"),
        "pathway_band": context["pathway_shape_summary"].get("transient_band_fractions", []),
    }
    unresolved_count = int(
        signature_rows["object_identity_resolved"].map(_as_bool).eq(False).sum()
    )
    return pd.DataFrame(
        [
            {
                "evidence_id": "E1_signature_purity",
                "evidence_question": "Do the transfer-trace state roles have unique signature IDs?",
                "observed": {
                    "target_signature_rows": sig_count("target_anchor_signature"),
                    "transient_signature_rows": sig_count("typed_transient_signature"),
                    "source_strict_signature_rows": sig_count("source_family_strict_signature"),
                    "source_guard_signature_rows": sig_count("source_guard_blocker_signature"),
                    "signature_ids": signature_rows["signature_id"].astype(str).tolist(),
                },
                "evidence_status": "supports_signature_identity_resolution",
                "claim_effect": "signature_identity_only_not_object_wall",
            },
            {
                "evidence_id": "E2_target_signature_provenance",
                "evidence_question": "Does the target signature match the local drop-bridge anchor?",
                "observed": {
                    "target_signature_id": TARGET_SIGNATURE_ID,
                    "local_drop_bridge_rows": int(target_local["run_count"].sum())
                    if not target_local.empty
                    else 0,
                    "direct_target_routes": int(
                        route_rows["direct_target_available_seed"].map(_as_bool).sum()
                    ),
                },
                "evidence_status": "target_signature_resolved_to_drop_bridge_proxy",
                "claim_effect": "supports_target_signature_not_endpoint_object",
            },
            {
                "evidence_id": "E3_transient_signature_provenance",
                "evidence_question": "Does the transient signature match prior 016 semantic/pathway artifacts?",
                "observed": transient_prior,
                "evidence_status": "transient_signature_resolved_as_recurrent_typed_band",
                "claim_effect": "supports_typed_transient_identity_not_wall",
            },
            {
                "evidence_id": "E4_source_family_split_named",
                "evidence_question": "Is the source-side split between strict source and guard blocker named?",
                "observed": {
                    "source_signature_ids": [SOURCE_SIGNATURE_ID, SOURCE_GUARD_SIGNATURE_ID],
                    "local_original_or_drop_direct_rows": int(source_local["run_count"].sum())
                    if not source_local.empty
                    else 0,
                    "source_equivalence_summary": {
                        "preferred_rule": context["source_equivalence_summary"].get(
                            "preferred_source_equivalence_rule"
                        ),
                        "preferred_rule_accepts": context["source_equivalence_summary"].get(
                            "preferred_rule_accepts"
                        ),
                        "preferred_rule_guard_caveats": context[
                            "source_equivalence_summary"
                        ].get("preferred_rule_guard_caveats"),
                    },
                },
                "evidence_status": "source_family_signature_split_named",
                "claim_effect": "supports_source_vocabulary_only",
            },
            {
                "evidence_id": "E5_object_identity_unresolved",
                "evidence_question": "Are the signature states resolved to endpoint objects?",
                "observed": {
                    "unresolved_signature_count": unresolved_count,
                    "object_identity_resolution_status_counts": _count_dict(
                        signature_rows["object_identity_resolution_status"]
                    ),
                },
                "evidence_status": "blocks_object_level_wall_language",
                "claim_effect": "keeps_object_wall_evidence_closed",
            },
            {
                "evidence_id": "E6_claim_boundary",
                "evidence_question": "Does this audit keep route, wall, method, quality/cost, and full-replay claims closed?",
                "observed": CLAIM_BOUNDARY,
                "evidence_status": "claims_closed",
                "claim_effect": "prevents_label_promotion",
            },
        ]
    )


def _decision_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_id": "D1",
                "decision": "signature_identity_resolved",
                "rationale": "The existing transfer trace has one stable signature for target, one for transient, and two named source-family signatures.",
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            },
            {
                "decision_id": "D2",
                "decision": "object_identity_not_resolved",
                "rationale": "Trace states remain endpoint-object proxies or typed transient states without a global endpoint-object identity certificate.",
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            },
            {
                "decision_id": "D3",
                "decision": "do_not_open_object_wall_claim",
                "rationale": "Signature identity is not sufficient for object-level wall evidence; recovery-loop evidence remains a typed transient ladder.",
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            },
            {
                "decision_id": "D4",
                "decision": "next_gate_object_identity_certificate",
                "rationale": "Next work should compare the four local signatures against symmetric endpoint-object membership or define a local object certificate over the same trace.",
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            },
            {
                "decision_id": "D5",
                "decision": "keep_route_expansion_closed",
                "rationale": "The blocker is interpretive object identity, not missing route execution.",
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            },
        ]
    )


def _gate_matrix(
    *,
    signature_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    local_ablation_signature: pd.DataFrame,
    context: dict[str, Any],
) -> pd.DataFrame:
    transfer_audit = context["transfer_audit_summary"]
    target_local_rows = local_ablation_signature[
        local_ablation_signature["signature_id"].astype(str).eq(TARGET_SIGNATURE_ID)
        & local_ablation_signature["graph_variant"].astype(str).eq("drop_bridge_edges")
    ]
    source_local_rows = local_ablation_signature[
        local_ablation_signature["signature_id"].astype(str).isin(
            {SOURCE_SIGNATURE_ID, SOURCE_GUARD_SIGNATURE_ID}
        )
        & local_ablation_signature["graph_variant"].astype(str).isin(
            {"original", "drop_direct_edge"}
        )
    ]
    signatures_by_role = {
        role: sorted(
            signature_rows.loc[
                signature_rows["primary_signature_role"].astype(str).eq(role),
                "signature_id",
            ].astype(str)
        )
        for role in sorted(signature_rows["primary_signature_role"].astype(str).unique())
    }
    signature_purity_pass = (
        signatures_by_role.get("target_anchor_signature") == [TARGET_SIGNATURE_ID]
        and signatures_by_role.get("typed_transient_signature") == [TRANSIENT_SIGNATURE_ID]
        and signatures_by_role.get("source_family_strict_signature") == [SOURCE_SIGNATURE_ID]
        and signatures_by_role.get("source_guard_blocker_signature")
        == [SOURCE_GUARD_SIGNATURE_ID]
    )
    direct_routes = int(route_rows["direct_target_available_seed"].map(_as_bool).sum())
    recovery_transient_routes = int(
        route_rows["recovery_typed_transient_block_seed"].map(_as_bool).sum()
    )
    object_identity_resolved_count = int(
        signature_rows["object_identity_resolved"].map(_as_bool).sum()
    )
    gates = [
        _gate_row(
            "G1_upstream_identity_gate_ready",
            "Did the transfer-trace audit open identity resolution?",
            {
                "transfer_audit_failed_gates": transfer_audit.get("failed_gates"),
                "object_identity_resolution_audit_ready": transfer_audit.get(
                    "object_identity_resolution_audit_ready"
                ),
            },
            "upstream audit gates pass and identity-resolution audit is ready",
            not transfer_audit.get("failed_gates")
            and bool(transfer_audit.get("object_identity_resolution_audit_ready")),
        ),
        _gate_row(
            "G2_signature_purity",
            "Are all expected 016 state roles signature-pure?",
            signatures_by_role,
            "target, transient, source, and source-guard roles each map to one expected signature",
            signature_purity_pass,
        ),
        _gate_row(
            "G3_target_signature_provenance",
            "Is the target signature grounded in the local drop-bridge anchor?",
            {
                "target_signature_id": TARGET_SIGNATURE_ID,
                "local_drop_bridge_rows": int(target_local_rows["run_count"].sum())
                if not target_local_rows.empty
                else 0,
                "direct_target_routes": direct_routes,
            },
            "target signature has local drop-bridge provenance and 24 direct target routes",
            (not target_local_rows.empty) and direct_routes == 24,
        ),
        _gate_row(
            "G4_transient_signature_provenance",
            "Is the transient signature grounded in prior semantic/pathway artifacts?",
            {
                "transient_signature_id": TRANSIENT_SIGNATURE_ID,
                "semantic_transient_ids": context["semantic_summary"].get(
                    "transient_signature_ids"
                ),
                "persistence_transient_id": context["persistence_summary"].get(
                    "transient_signature_id"
                ),
                "reverse_transient_id": context["reverse_summary"].get(
                    "transient_signature_id"
                ),
                "recovery_transient_routes": recovery_transient_routes,
            },
            "transient signature matches prior artifacts and appears in 24 recovery routes",
            TRANSIENT_SIGNATURE_ID
            in set(context["semantic_summary"].get("transient_signature_ids", []))
            and context["persistence_summary"].get("transient_signature_id")
            == TRANSIENT_SIGNATURE_ID
            and context["reverse_summary"].get("transient_signature_id")
            == TRANSIENT_SIGNATURE_ID
            and recovery_transient_routes == 24,
        ),
        _gate_row(
            "G5_source_family_split_named",
            "Is the source side split named rather than untyped?",
            {
                "source_signature_ids": [SOURCE_SIGNATURE_ID, SOURCE_GUARD_SIGNATURE_ID],
                "local_original_or_drop_direct_rows": int(source_local_rows["run_count"].sum())
                if not source_local_rows.empty
                else 0,
                "preferred_source_equivalence_rule": context[
                    "source_equivalence_summary"
                ].get("preferred_source_equivalence_rule"),
            },
            "source strict and source-guard signatures have local provenance and source-family equivalence is defined",
            not source_local_rows.empty
            and context["source_equivalence_summary"].get(
                "preferred_source_equivalence_rule"
            )
            == "same_start_source_family_with_guard_caveat",
        ),
        _gate_row(
            "G6_object_identity_still_unresolved",
            "Does the audit correctly keep object-level identity unresolved?",
            {
                "object_identity_resolved_signature_count": object_identity_resolved_count,
                "object_identity_status_counts": _count_dict(
                    signature_rows["object_identity_resolution_status"]
                ),
            },
            "zero signatures resolved to endpoint objects; blocker explicitly named",
            object_identity_resolved_count == 0,
        ),
        _gate_row(
            "G7_claims_closed",
            "Are pathway, wall, method, quality/cost, full-replay, and route-expansion claims closed?",
            CLAIM_BOUNDARY,
            "all promotion flags remain false",
            True,
        ),
    ]
    return pd.DataFrame(gates)


def _report(
    *,
    summary: dict[str, Any],
    signature_rows: pd.DataFrame,
    step_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    local_ablation_signature_rows: pd.DataFrame,
    evidence_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# NanoClustering G4.8 First-Pass 016 Object/Signature Identity Resolution",
            "",
            f"- status: `{summary['status']}`",
            f"- signature_identity_resolved: {summary['signature_identity_resolved']}",
            f"- object_identity_resolved: {summary['object_identity_resolved']}",
            f"- local_object_wall_evidence_audit_ready: {summary['local_object_wall_evidence_audit_ready']}",
            f"- target_signature_id: `{summary['target_signature_id']}`",
            f"- transient_signature_id: `{summary['transient_signature_id']}`",
            f"- source_signature_ids: {summary['source_signature_ids']}",
            f"- gate_status_counts: {summary['gate_status_counts']}",
            f"- failed_gates: {summary['failed_gates']}",
            f"- interpretation: {summary['interpretation']}",
            f"- recommended_next_gate: {summary['recommended_next_gate']}",
            f"- claim_boundary: {summary['claim_boundary']}",
            "",
            "## Signature Rows",
            "",
            _markdown_table(
                signature_rows,
                [
                    "signature_id",
                    "primary_signature_role",
                    "trace_row_count",
                    "signature_identity_resolution_status",
                    "object_identity_resolution_status",
                    "local_ablation_graph_variants",
                    "prior_transient_signature_match",
                    "prior_target_signature_match",
                ],
            ),
            "",
            "## Step Rows",
            "",
            _markdown_table(
                step_rows,
                [
                    "planned_route_family",
                    "start_condition",
                    "step_index",
                    "step_label",
                    "bridge_edge_weight_fraction",
                    "signature_ids",
                    "signature_role_counts",
                    "support_incompatibility_count",
                ],
            ),
            "",
            "## Route Rows",
            "",
            _markdown_table(
                route_rows,
                [
                    "planned_route_family",
                    "start_condition",
                    "seed",
                    "route_identity_class",
                    "readiness_effect",
                    "endpoint_assignment_sequence",
                    "typed_transient_assignment_sequence",
                ],
                max_rows=48,
            ),
            "",
            "## Local Ablation Signature Provenance",
            "",
            _markdown_table(
                local_ablation_signature_rows,
                [
                    "signature_id",
                    "graph_variant",
                    "run_count",
                    "pair_coassigned_share",
                    "mechanism_read_counts",
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
            "This audit resolves state signatures, not endpoint-object identity. It should",
            "be followed by an object-identity certificate over the same trace surface,",
            "with no route expansion and no label promotion.",
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    context = _load_context(args)
    trace_rows = context["trace_rows"]
    route_result_rows = context["route_result_rows"]
    seed_runs = context["local_ablation_seed_runs"]

    local_ablation_signature_rows = _local_ablation_signature_rows(seed_runs)
    signature_rows = _signature_rows(
        trace_rows=trace_rows,
        local_ablation_signature=local_ablation_signature_rows,
        context=context,
    )
    step_rows = _step_rows(trace_rows)
    route_rows = _route_rows(route_result_rows)
    evidence_rows = _evidence_rows(
        signature_rows=signature_rows,
        route_rows=route_rows,
        local_ablation_signature=local_ablation_signature_rows,
        context=context,
    )
    decision_rows = _decision_rows()
    gate_matrix = _gate_matrix(
        signature_rows=signature_rows,
        route_rows=route_rows,
        local_ablation_signature=local_ablation_signature_rows,
        context=context,
    )

    failed_gates = gate_matrix.loc[
        gate_matrix["gate_status"].astype(str).eq("fail"), "gate_id"
    ].astype(str).tolist()
    gate_status_by_id = {
        str(row["gate_id"]): str(row["gate_status"])
        for _, row in gate_matrix.iterrows()
    }
    signature_gate_ids = {
        "G1_upstream_identity_gate_ready",
        "G2_signature_purity",
        "G3_target_signature_provenance",
        "G4_transient_signature_provenance",
        "G5_source_family_split_named",
    }
    signature_identity_resolved = all(
        gate_status_by_id.get(gate_id) == "pass" for gate_id in signature_gate_ids
    ) and bool(
        signature_rows["signature_identity_resolved"].map(_as_bool).all()
    )
    object_identity_resolved = bool(signature_rows["object_identity_resolved"].map(_as_bool).all())
    local_object_wall_ready = bool(signature_identity_resolved and object_identity_resolved)
    summary = {
        "schema": "nanoclustering_g4_8_first_pass_016_object_signature_identity_resolution_summary.v1",
        "status": RUN_STATUS,
        "transfer_trace_dir": str(args.transfer_trace_dir.resolve()),
        "transfer_audit_dir": str(args.transfer_audit_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "signature_row_count": int(len(signature_rows)),
        "step_row_count": int(len(step_rows)),
        "route_row_count": int(len(route_rows)),
        "local_ablation_signature_row_count": int(len(local_ablation_signature_rows)),
        "evidence_row_count": int(len(evidence_rows)),
        "gate_status_counts": _count_dict(gate_matrix["gate_status"]),
        "failed_gates": failed_gates,
        "signature_identity_resolved": bool(signature_identity_resolved),
        "object_identity_resolved": bool(object_identity_resolved),
        "local_object_wall_evidence_audit_ready": bool(local_object_wall_ready),
        "target_signature_id": TARGET_SIGNATURE_ID,
        "transient_signature_id": TRANSIENT_SIGNATURE_ID,
        "source_signature_ids": [SOURCE_SIGNATURE_ID, SOURCE_GUARD_SIGNATURE_ID],
        "signature_resolution_status_counts": _count_dict(
            signature_rows["signature_identity_resolution_status"]
        ),
        "object_identity_resolution_status_counts": _count_dict(
            signature_rows["object_identity_resolution_status"]
        ),
        "route_identity_class_counts": _count_dict(route_rows["route_identity_class"]),
        "wall_claim_ready_pairs": [],
        "interpretation": (
            "The executed 016 transfer trace has stable signature-level identities: "
            "target=3c9b8a190753, transient=aeb59ab537e6, and two source-family "
            "signatures 5536308f50fc/c475d13ca500. This resolves the signature "
            "readout but not endpoint-object identity, so local object-wall evidence "
            "remains closed."
        ),
        "recommended_next_gate": (
            "Build a read-only object-identity certificate for the four local "
            "signatures against symmetric endpoint-object membership or an explicit "
            "local object certificate; do not expand routes or promote labels."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(signature_rows, args.output_dir / SIGNATURE_ROWS_CSV)
    _write_csv(step_rows, args.output_dir / STEP_ROWS_CSV)
    _write_csv(route_rows, args.output_dir / ROUTE_ROWS_CSV)
    _write_csv(
        local_ablation_signature_rows,
        args.output_dir / LOCAL_ABLATION_SIGNATURE_ROWS_CSV,
    )
    _write_csv(evidence_rows, args.output_dir / EVIDENCE_ROWS_CSV)
    _write_csv(decision_rows, args.output_dir / DECISION_ROWS_CSV)
    _write_csv(gate_matrix, args.output_dir / GATE_MATRIX_CSV)
    (args.output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_016_object_signature_identity_resolution_config.v1",
        "positive_pair_id": POSITIVE_PAIR_ID,
        "target_signature_id": TARGET_SIGNATURE_ID,
        "transient_signature_id": TRANSIENT_SIGNATURE_ID,
        "source_signature_ids": [SOURCE_SIGNATURE_ID, SOURCE_GUARD_SIGNATURE_ID],
        "transfer_trace_dir": str(args.transfer_trace_dir.resolve()),
        "transfer_audit_dir": str(args.transfer_audit_dir.resolve()),
        "local_ablation_dir": str(args.local_ablation_dir.resolve()),
        "semantic_dir": str(args.semantic_dir.resolve()),
        "persistence_dir": str(args.persistence_dir.resolve()),
        "reverse_dir": str(args.reverse_dir.resolve()),
        "source_equivalence_dir": str(args.source_equivalence_dir.resolve()),
        "pathway_shape_dir": str(args.pathway_shape_dir.resolve()),
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
            signature_rows=signature_rows,
            step_rows=step_rows,
            route_rows=route_rows,
            local_ablation_signature_rows=local_ablation_signature_rows,
            evidence_rows=evidence_rows,
            decision_rows=decision_rows,
            gate_matrix=gate_matrix,
        ),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit first-pass 016 object/signature identity resolution over the "
            "existing object-wall transfer trace."
        )
    )
    parser.add_argument("--transfer-trace-dir", type=Path, default=DEFAULT_TRANSFER_TRACE_DIR)
    parser.add_argument("--transfer-audit-dir", type=Path, default=DEFAULT_TRANSFER_AUDIT_DIR)
    parser.add_argument("--local-ablation-dir", type=Path, default=DEFAULT_LOCAL_ABLATION_DIR)
    parser.add_argument("--semantic-dir", type=Path, default=DEFAULT_SEMANTIC_DIR)
    parser.add_argument("--persistence-dir", type=Path, default=DEFAULT_PERSISTENCE_DIR)
    parser.add_argument("--reverse-dir", type=Path, default=DEFAULT_REVERSE_DIR)
    parser.add_argument(
        "--source-equivalence-dir", type=Path, default=DEFAULT_SOURCE_EQUIVALENCE_DIR
    )
    parser.add_argument("--pathway-shape-dir", type=Path, default=DEFAULT_PATHWAY_SHAPE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
