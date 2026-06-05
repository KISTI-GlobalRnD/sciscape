#!/usr/bin/env python3
"""Audit role stability of first-pass 014 transition-band signatures.

This read-only audit turns endpoint signatures into role-level cluster objects
for the 12-node ``local_pair_014`` graph. It distinguishes the left/right pair
nodes from the ranked bridge nodes and classifies recurrent transition-band
signatures by role composition, not only by row-local anchor labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from audit_leiden_basin_nanoclustering_g4_8_first_pass_014_wall_localization_signature_identity import (
    DEFAULT_OUTPUT_DIR as DEFAULT_SIGNATURE_IDENTITY_AUDIT_DIR,
    GATE_MATRIX_CSV as SIGNATURE_IDENTITY_GATE_MATRIX_CSV,
    SIGNATURE_IDENTITY_ROWS_CSV,
)
from run_leiden_basin_nanoclustering_g4_8_first_pass_014_wall_localization_trace import (
    POSITIVE_PAIR_ID,
    TRACE_ROWS_CSV,
    DEFAULT_OUTPUT_DIR as DEFAULT_TRACE_DIR,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)
from run_leiden_basin_nanoclustering_symmetric_object_variable_pair_local_ablation import (
    DEFAULT_OUTPUT_DIR as DEFAULT_LOCAL_ABLATION_DIR,
    LOCAL_GRAPH_ROWS_CSV,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_014_intermediate_role_stability_audit_gamma1e5_20260605"
)

NODE_ROLE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_intermediate_role_stability_node_rows.csv"
)
SIGNATURE_ROLE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_intermediate_role_stability_signature_rows.csv"
)
SEED_ROUTE_ROLE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_intermediate_role_stability_seed_route_rows.csv"
)
PAIR_ROLE_SUMMARY_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_intermediate_role_stability_pair_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_014_intermediate_role_stability_gate_matrix.csv"
)
SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_014_intermediate_role_stability_summary.json"
)
CONFIG_JSON = (
    "nanoclustering_g4_8_first_pass_014_intermediate_role_stability_config.json"
)
REPORT_MD = (
    "nanoclustering_g4_8_first_pass_014_intermediate_role_stability_report.md"
)

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_014_intermediate_role_stability"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass local_pair_014 intermediate role-stability "
    "audit only; reads the executed localization trace and local graph metadata "
    "to classify recurrent transition-band signatures by L/R/bridge cluster "
    "roles. It does not promote wall generality, evaluate quality/cost value, "
    "replay full NanoClustering, or claim method success."
)


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


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]

    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def _parse_bridge_meta(spec: pd.Series) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for token in str(spec["selected_bridge_rank_scope_weight"]).split(";"):
        parts = token.split(":")
        if len(parts) != 4:
            continue
        rank, node_id, scope, weight = parts
        rows[int(node_id)] = {
            "bridge_rank": int(rank),
            "bridge_scope": str(scope),
            "bridge_min_pair_edge_weight": float(weight),
        }
    return rows


def _node_role_rows(local_ablation_dir: Path) -> tuple[pd.DataFrame, dict[int, str], int, int]:
    graph_rows = _read_csv(local_ablation_dir / LOCAL_GRAPH_ROWS_CSV)
    spec = graph_rows[graph_rows["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)]
    if len(spec) != 1:
        raise ValueError(f"expected one local graph row for {POSITIVE_PAIR_ID}, got {len(spec)}")
    row = spec.iloc[0]
    left = int(row["left_node_id"])
    right = int(row["right_node_id"])
    bridge_meta = _parse_bridge_meta(row)
    node_ids = [int(value) for value in str(row["local_node_ids"]).split(";") if value]
    role_by_node: dict[int, str] = {}
    output: list[dict[str, Any]] = []
    for node_id in node_ids:
        if node_id == left:
            role = "L"
            family = "left_pair_node"
            bridge_rank = None
            bridge_scope = "pair"
            bridge_weight = None
        elif node_id == right:
            role = "R"
            family = "right_pair_node"
            bridge_rank = None
            bridge_scope = "pair"
            bridge_weight = None
        else:
            meta = bridge_meta.get(node_id, {})
            bridge_rank = int(meta.get("bridge_rank", -1))
            role = f"B{bridge_rank}" if bridge_rank > 0 else f"B?{node_id}"
            family = "selected_bridge_node"
            bridge_scope = str(meta.get("bridge_scope", "unknown"))
            bridge_weight = meta.get("bridge_min_pair_edge_weight")
        role_by_node[node_id] = role
        output.append(
            {
                "local_pair_id": POSITIVE_PAIR_ID,
                "node_id": int(node_id),
                "node_role": role,
                "node_role_family": family,
                "bridge_rank": bridge_rank,
                "bridge_scope": bridge_scope,
                "bridge_min_pair_edge_weight": bridge_weight,
                "method_claim_allowed_after_audit": False,
                "quality_cost_claim_allowed_after_audit": False,
                "wall_generality_claim_allowed_after_audit": False,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(output), role_by_node, left, right


def _cluster_roles(signature_json: str, role_by_node: dict[int, str]) -> tuple[str, dict[str, int], list[list[str]]]:
    groups = json.loads(signature_json)
    role_groups: list[list[str]] = []
    membership: dict[str, int] = {}
    for cluster_index, group in enumerate(groups):
        roles = [role_by_node.get(int(node_id), str(node_id)) for node_id in group]
        role_groups.append(roles)
        for role in roles:
            membership[role] = cluster_index
    return " | ".join("+".join(roles) for roles in role_groups), membership, role_groups


def _cluster_for_role(role_groups: list[list[str]], role: str) -> list[str]:
    for group in role_groups:
        if role in group:
            return group
    return []


def _role_class(row: dict[str, Any]) -> str:
    status = str(row["signature_identity_status"])
    pair_coassigned = bool(row["pair_coassigned_any"])
    left_cluster = set(str(row["left_cluster_roles"]).split("+")) if row["left_cluster_roles"] else set()
    right_cluster = set(str(row["right_cluster_roles"]).split("+")) if row["right_cluster_roles"] else set()
    if status == "stable_positive_target_signature":
        return "target_anchor_pair_only"
    if status == "stable_source_like_signature" and pair_coassigned:
        return "source_anchor_pair_with_bridges"
    if status == "stable_source_like_signature" and not pair_coassigned:
        return "source_like_separated_bridge_split"
    if status == "row_local_unresolved_but_signature_known_elsewhere":
        return "hidden_known_source_guard_intermediate"
    if status == "signature_level_unresolved_positive_intermediate" and pair_coassigned:
        return "unresolved_pair_coassigned_intermediate"
    if status == "signature_level_unresolved_positive_intermediate" and not pair_coassigned:
        if "L" in left_cluster and "R" in right_cluster and left_cluster != right_cluster:
            return "unresolved_pair_separated_bridge_reassignment"
        return "unresolved_pair_separated_intermediate"
    return "other_signature_role"


def _signature_role_rows(
    *,
    trace_rows: pd.DataFrame,
    signature_identity_rows: pd.DataFrame,
    role_by_node: dict[int, str],
) -> pd.DataFrame:
    positive = trace_rows[trace_rows["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)].copy()
    identity = signature_identity_rows[
        signature_identity_rows["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)
    ].set_index("result_endpoint_signature_id")
    rows: list[dict[str, Any]] = []
    for signature_id, group in positive.groupby("result_endpoint_signature_id", sort=True):
        signature_json = str(group["result_endpoint_signature"].iloc[0])
        cluster_signature, membership, role_groups = _cluster_roles(signature_json, role_by_node)
        left_cluster_roles = _cluster_for_role(role_groups, "L")
        right_cluster_roles = _cluster_for_role(role_groups, "R")
        pair_coassigned_any = bool(group["pair_coassigned"].map(_as_bool).any())
        pair_coassigned_rate = float(group["pair_coassigned"].mean())
        identity_row = identity.loc[str(signature_id)].to_dict()
        bridge_roles_left = sorted(role for role in left_cluster_roles if role.startswith("B"))
        bridge_roles_right = sorted(role for role in right_cluster_roles if role.startswith("B"))
        bridge_roles_pair_cluster = sorted(
            set(bridge_roles_left + bridge_roles_right)
            if pair_coassigned_any
            else set()
        )
        row = {
            "local_pair_id": POSITIVE_PAIR_ID,
            "result_endpoint_signature_id": str(signature_id),
            "signature_role_cluster_signature": cluster_signature,
            "left_cluster_roles": "+".join(left_cluster_roles),
            "right_cluster_roles": "+".join(right_cluster_roles),
            "left_bridge_roles": "+".join(bridge_roles_left),
            "right_bridge_roles": "+".join(bridge_roles_right),
            "pair_cluster_bridge_roles": "+".join(bridge_roles_pair_cluster),
            "pair_coassigned_any": bool(pair_coassigned_any),
            "pair_coassigned_rate": pair_coassigned_rate,
            "signature_row_count": int(len(group)),
            "start_conditions": ";".join(sorted(set(group["start_condition"].astype(str)))),
            "seeds": ";".join(str(value) for value in sorted(set(group["seed"].astype(int)))),
            "route_family_roles": ";".join(sorted(set(group["route_family_role"].astype(str)))),
            "bridge_fractions": ";".join(
                f"{value:.3g}"
                for value in sorted(set(group["bridge_edge_weight_fraction"].astype(float)))
            ),
            "objective_value_mean": float(group["objective_value_by_step"].mean()),
            "objective_value_min": float(group["objective_value_by_step"].min()),
            "objective_value_max": float(group["objective_value_by_step"].max()),
            "support_distance_to_original_min": float(group["support_distance_to_original"].min()),
            "support_distance_to_drop_bridge_edges_min": float(
                group["support_distance_to_drop_bridge_edges"].min()
            ),
            "support_distance_to_drop_direct_edge_min": float(
                group["support_distance_to_drop_direct_edge"].min()
            ),
            "support_distance_to_drop_direct_and_bridge_edges_min": float(
                group["support_distance_to_drop_direct_and_bridge_edges"].min()
            ),
            "observed_endpoint_object_assignments": str(
                identity_row.get("observed_endpoint_object_assignments", "")
            ),
            "signature_identity_status": str(identity_row.get("signature_identity_status", "")),
            "signature_level_unresolved": bool(
                _as_bool(identity_row.get("signature_level_unresolved", False))
            ),
            "signature_known_elsewhere": bool(
                _as_bool(identity_row.get("signature_known_elsewhere", False))
            ),
            "method_claim_allowed_after_audit": False,
            "quality_cost_claim_allowed_after_audit": False,
            "wall_generality_claim_allowed_after_audit": False,
            "run_status": RUN_STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        row["signature_role_class"] = _role_class(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["objective_value_mean", "signature_row_count"],
        ascending=[True, False],
        kind="mergesort",
    )


def _seed_route_role_rows(
    trace_rows: pd.DataFrame,
    signature_role_rows: pd.DataFrame,
) -> pd.DataFrame:
    role_class = signature_role_rows.set_index("result_endpoint_signature_id")[
        "signature_role_class"
    ].to_dict()
    positive = trace_rows[trace_rows["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)].copy()
    rows: list[dict[str, Any]] = []
    key_cols = ["local_pair_id", "branch", "start_condition", "seed", "route_family_role"]
    for key, group in positive.groupby(key_cols, sort=True):
        data = dict(zip(key_cols, key, strict=True))
        route = group.sort_values("step_index", kind="mergesort")
        signature_sequence = route["result_endpoint_signature_id"].astype(str).tolist()
        role_sequence = [role_class.get(signature_id, "untyped_signature") for signature_id in signature_sequence]
        rows.append(
            {
                **data,
                "route_signature_sequence": " -> ".join(signature_sequence),
                "route_role_class_sequence": " -> ".join(role_sequence),
                "unique_signature_count": int(len(set(signature_sequence))),
                "unique_role_class_count": int(len(set(role_sequence))),
                "signature_level_unresolved_step_count": int(
                    sum("unresolved_" in role for role in role_sequence)
                ),
                "hidden_known_source_guard_step_count": int(
                    sum(role == "hidden_known_source_guard_intermediate" for role in role_sequence)
                ),
                "target_anchor_step_count": int(
                    sum(role == "target_anchor_pair_only" for role in role_sequence)
                ),
                "source_anchor_step_count": int(
                    sum(role.startswith("source_") for role in role_sequence)
                ),
                "method_claim_allowed_after_audit": False,
                "quality_cost_claim_allowed_after_audit": False,
                "wall_generality_claim_allowed_after_audit": False,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _pair_summary_rows(
    signature_rows: pd.DataFrame,
    seed_route_rows: pd.DataFrame,
) -> pd.DataFrame:
    unresolved = signature_rows[
        signature_rows["signature_level_unresolved"].map(_as_bool)
    ]
    hidden_known = signature_rows[
        signature_rows["signature_role_class"].astype(str).eq(
            "hidden_known_source_guard_intermediate"
        )
    ]
    rows = [
        {
            "local_pair_id": POSITIVE_PAIR_ID,
            "typed_signature_count": int(len(signature_rows)),
            "signature_role_class_counts": json.dumps(
                _count_dict(signature_rows["signature_role_class"]),
                ensure_ascii=True,
                sort_keys=True,
            ),
            "signature_level_unresolved_signature_count": int(len(unresolved)),
            "signature_level_unresolved_signature_ids": ";".join(
                unresolved["result_endpoint_signature_id"].astype(str)
            ),
            "hidden_known_source_guard_signature_count": int(len(hidden_known)),
            "hidden_known_source_guard_signature_ids": ";".join(
                hidden_known["result_endpoint_signature_id"].astype(str)
            ),
            "seed_route_row_count": int(len(seed_route_rows)),
            "seed_route_with_unresolved_intermediate_count": int(
                seed_route_rows["signature_level_unresolved_step_count"].gt(0).sum()
            ),
            "seed_route_with_hidden_known_source_guard_count": int(
                seed_route_rows["hidden_known_source_guard_step_count"].gt(0).sum()
            ),
            "pair_role_stability_status": (
                "typed_recurrent_transition_band_signatures_with_unresolved_intermediates"
                if len(unresolved) > 0 and len(hidden_known) > 0
                else "typed_transition_band_signatures"
            ),
            "method_claim_allowed_after_audit": False,
            "quality_cost_claim_allowed_after_audit": False,
            "wall_generality_claim_allowed_after_audit": False,
            "run_status": RUN_STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
        }
    ]
    return pd.DataFrame(rows)


def _gate_matrix(
    *,
    signature_identity_gate_matrix: pd.DataFrame,
    node_role_rows: pd.DataFrame,
    signature_role_rows: pd.DataFrame,
    seed_route_rows: pd.DataFrame,
    pair_summary_rows: pd.DataFrame,
) -> pd.DataFrame:
    identity_gate_counts = _count_dict(signature_identity_gate_matrix["gate_status"])
    unresolved_count = int(signature_role_rows["signature_level_unresolved"].map(_as_bool).sum())
    hidden_known_count = int(
        signature_role_rows["signature_role_class"]
        .astype(str)
        .eq("hidden_known_source_guard_intermediate")
        .sum()
    )
    all_claims_closed = bool(
        not pair_summary_rows["method_claim_allowed_after_audit"].map(_as_bool).any()
        and not pair_summary_rows["quality_cost_claim_allowed_after_audit"].map(_as_bool).any()
        and not pair_summary_rows["wall_generality_claim_allowed_after_audit"].map(_as_bool).any()
    )
    rows = [
        {
            "gate_id": "G1_signature_identity_gates_pass",
            "question": "Did the upstream signature-identity gates pass?",
            "observed": json.dumps(identity_gate_counts, ensure_ascii=True, sort_keys=True),
            "minimum_or_rule": "all signature-identity gates pass",
            "gate_status": "pass"
            if identity_gate_counts.get("pass", 0) == len(signature_identity_gate_matrix)
            else "fail",
        },
        {
            "gate_id": "G2_node_roles_materialized",
            "question": "Were L/R plus ten bridge roles materialized?",
            "observed": f"node_role_rows={len(node_role_rows)}",
            "minimum_or_rule": "12 local-node role rows",
            "gate_status": "pass" if len(node_role_rows) == 12 else "fail",
        },
        {
            "gate_id": "G3_positive_signatures_typed",
            "question": "Were all positive endpoint signatures typed by role composition?",
            "observed": f"signature_role_rows={len(signature_role_rows)}",
            "minimum_or_rule": "6 positive signatures",
            "gate_status": "pass" if len(signature_role_rows) == 6 else "fail",
        },
        {
            "gate_id": "G4_unresolved_intermediates_preserved",
            "question": "Are signature-level unresolved intermediates retained after role typing?",
            "observed": f"unresolved_signature_count={unresolved_count}",
            "minimum_or_rule": "at least two unresolved intermediate signatures from identity audit",
            "gate_status": "pass" if unresolved_count >= 2 else "fail",
        },
        {
            "gate_id": "G5_hidden_known_source_guard_typed",
            "question": "Was the hidden-known source/guard intermediate typed separately?",
            "observed": f"hidden_known_source_guard_signature_count={hidden_known_count}",
            "minimum_or_rule": "at least one hidden-known source/guard signature",
            "gate_status": "pass" if hidden_known_count >= 1 else "fail",
        },
        {
            "gate_id": "G6_seed_route_role_sequences_materialized",
            "question": "Were role-class sequences materialized for every positive seed route?",
            "observed": f"seed_route_rows={len(seed_route_rows)}",
            "minimum_or_rule": "64 positive seed-route rows",
            "gate_status": "pass" if len(seed_route_rows) == 64 else "fail",
        },
        {
            "gate_id": "G7_claims_closed",
            "question": "Are method, quality/cost, and wall-generality claims closed?",
            "observed": CLAIM_BOUNDARY,
            "minimum_or_rule": "all claim flags false",
            "gate_status": "pass" if all_claims_closed else "fail",
        },
    ]
    return pd.DataFrame(rows)


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    node_role_rows: pd.DataFrame,
    signature_role_rows: pd.DataFrame,
    seed_route_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> None:
    route_preview = seed_route_rows[
        [
            "start_condition",
            "seed",
            "route_family_role",
            "signature_level_unresolved_step_count",
            "hidden_known_source_guard_step_count",
            "route_role_class_sequence",
        ]
    ].head(16)
    report = [
        "# NanoClustering G4.8 First-Pass 014 Intermediate Role-Stability Audit",
        "",
        f"- status: `{RUN_STATUS}`",
        f"- typed_signature_count: {summary['typed_signature_count']}",
        f"- signature_level_unresolved_signature_count: {summary['signature_level_unresolved_signature_count']}",
        f"- hidden_known_source_guard_signature_count: {summary['hidden_known_source_guard_signature_count']}",
        f"- seed_route_with_unresolved_intermediate_count: {summary['seed_route_with_unresolved_intermediate_count']}",
        f"- seed_route_with_hidden_known_source_guard_count: {summary['seed_route_with_hidden_known_source_guard_count']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        "- interpretation: The transition band contains typed recurrent "
        "signatures, not arbitrary noise. One signature is a hidden-known "
        "source/guard intermediate, while two signatures remain true "
        "signature-level unresolved intermediates with distinct pair-coassigned "
        "and pair-separated role compositions.",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Pair Role Summary",
        "",
        _markdown_table(pair_rows),
        "",
        "## Signature Role Rows",
        "",
        _markdown_table(
            signature_role_rows[
                [
                    "result_endpoint_signature_id",
                    "signature_role_class",
                    "signature_identity_status",
                    "signature_row_count",
                    "bridge_fractions",
                    "objective_value_mean",
                    "pair_coassigned_rate",
                    "left_cluster_roles",
                    "right_cluster_roles",
                    "signature_role_cluster_signature",
                ]
            ]
        ),
        "",
        "## Node Roles",
        "",
        _markdown_table(
            node_role_rows[
                [
                    "node_id",
                    "node_role",
                    "node_role_family",
                    "bridge_rank",
                    "bridge_scope",
                    "bridge_min_pair_edge_weight",
                ]
            ]
        ),
        "",
        "## Route Role Sequence Preview",
        "",
        _markdown_table(route_preview),
        "",
        "## Gate Matrix",
        "",
        _markdown_table(gate_matrix),
        "",
        "## Boundary",
        "",
        "This audit is a local object-typing layer over an already executed trace. "
        "It does not promote wall generality, method success, full replay, or "
        "quality/cost value.",
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(report), encoding="utf-8")


def run_audit(
    *,
    trace_dir: Path,
    signature_identity_audit_dir: Path,
    local_ablation_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    trace_rows = _read_csv(trace_dir / TRACE_ROWS_CSV)
    signature_identity_rows = _read_csv(
        signature_identity_audit_dir / SIGNATURE_IDENTITY_ROWS_CSV
    )
    signature_identity_gate_matrix = _read_csv(
        signature_identity_audit_dir / SIGNATURE_IDENTITY_GATE_MATRIX_CSV
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    node_role_rows, role_by_node, _left, _right = _node_role_rows(local_ablation_dir)
    signature_role_rows = _signature_role_rows(
        trace_rows=trace_rows,
        signature_identity_rows=signature_identity_rows,
        role_by_node=role_by_node,
    )
    seed_route_rows = _seed_route_role_rows(trace_rows, signature_role_rows)
    pair_rows = _pair_summary_rows(signature_role_rows, seed_route_rows)
    gate_matrix = _gate_matrix(
        signature_identity_gate_matrix=signature_identity_gate_matrix,
        node_role_rows=node_role_rows,
        signature_role_rows=signature_role_rows,
        seed_route_rows=seed_route_rows,
        pair_summary_rows=pair_rows,
    )

    _write_csv(node_role_rows, output_dir / NODE_ROLE_ROWS_CSV)
    _write_csv(signature_role_rows, output_dir / SIGNATURE_ROLE_ROWS_CSV)
    _write_csv(seed_route_rows, output_dir / SEED_ROUTE_ROLE_ROWS_CSV)
    _write_csv(pair_rows, output_dir / PAIR_ROLE_SUMMARY_ROWS_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)

    pair_row = pair_rows.iloc[0].to_dict()
    summary = {
        "run_status": RUN_STATUS,
        "trace_dir": str(trace_dir),
        "signature_identity_audit_dir": str(signature_identity_audit_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "typed_signature_count": int(pair_row["typed_signature_count"]),
        "signature_role_class_counts": json.loads(str(pair_row["signature_role_class_counts"])),
        "signature_level_unresolved_signature_count": int(
            pair_row["signature_level_unresolved_signature_count"]
        ),
        "signature_level_unresolved_signature_ids": str(
            pair_row["signature_level_unresolved_signature_ids"]
        ),
        "hidden_known_source_guard_signature_count": int(
            pair_row["hidden_known_source_guard_signature_count"]
        ),
        "hidden_known_source_guard_signature_ids": str(
            pair_row["hidden_known_source_guard_signature_ids"]
        ),
        "seed_route_row_count": int(pair_row["seed_route_row_count"]),
        "seed_route_with_unresolved_intermediate_count": int(
            pair_row["seed_route_with_unresolved_intermediate_count"]
        ),
        "seed_route_with_hidden_known_source_guard_count": int(
            pair_row["seed_route_with_hidden_known_source_guard_count"]
        ),
        "pair_role_stability_status": str(pair_row["pair_role_stability_status"]),
        "gate_status_counts": _count_dict(gate_matrix["gate_status"]),
        "failed_gates": gate_matrix[gate_matrix["gate_status"].astype(str).ne("pass")][
            "gate_id"
        ].astype(str).tolist(),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    config = {
        "trace_dir": str(trace_dir),
        "signature_identity_audit_dir": str(signature_identity_audit_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "read_only_trace_audit": True,
        "positive_pair_id": POSITIVE_PAIR_ID,
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        node_role_rows=node_role_rows,
        signature_role_rows=signature_role_rows,
        seed_route_rows=seed_route_rows,
        pair_rows=pair_rows,
        gate_matrix=gate_matrix,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument(
        "--signature-identity-audit-dir",
        type=Path,
        default=DEFAULT_SIGNATURE_IDENTITY_AUDIT_DIR,
    )
    parser.add_argument("--local-ablation-dir", type=Path, default=DEFAULT_LOCAL_ABLATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_audit(
        trace_dir=args.trace_dir,
        signature_identity_audit_dir=args.signature_identity_audit_dir,
        local_ablation_dir=args.local_ablation_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
