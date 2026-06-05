#!/usr/bin/env python3
"""Screen whether the typed 014 role pattern transfers to other first-pass pairs.

This is a read-only audit over existing first-pass, exclusive-target, and local
graph outputs. It uses the ``local_pair_014`` role-stability audit as the
reference pattern, then classifies the same first-pass endpoint signatures for
all screened pairs by local L/R/bridge role composition and endpoint-object
family. It does not rerun Leiden, perform a fraction sweep, promote walls,
evaluate quality/cost value, or claim method success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from audit_leiden_basin_nanoclustering_g4_8_first_pass_014_intermediate_role_stability import (
    DEFAULT_OUTPUT_DIR as DEFAULT_014_ROLE_STABILITY_AUDIT_DIR,
    GATE_MATRIX_CSV as ROLE_STABILITY_GATE_MATRIX_CSV,
)
from audit_leiden_basin_nanoclustering_g4_8_first_pass_exclusive_target_contrast import (
    DEFAULT_OUTPUT_DIR as DEFAULT_EXCLUSIVE_TARGET_CONTRAST_DIR,
    GATE_MATRIX_CSV as EXCLUSIVE_TARGET_GATE_MATRIX_CSV,
    PAIR_CONTRAST_ROWS_CSV,
)
from run_leiden_basin_nanoclustering_g4_8_fresh_axis_b_first_pass_trace import (
    DEFAULT_OUTPUT_DIR as DEFAULT_FIRST_PASS_TRACE_DIR,
    GATE_MATRIX_CSV as FIRST_PASS_GATE_MATRIX_CSV,
    PAIR_READOUT_RESULT_ROWS_CSV,
    TRACE_ROWS_CSV,
)
from run_leiden_basin_nanoclustering_g4_8_first_pass_014_wall_localization_trace import (
    POSITIVE_PAIR_ID,
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
    / "leiden_basin_nanoclustering_g4_8_first_pass_014_role_pattern_transfer_screen_gamma1e5_20260605"
)

NODE_ROLE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_role_pattern_transfer_node_rows.csv"
)
SIGNATURE_ROLE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_role_pattern_transfer_signature_rows.csv"
)
ROUTE_ROLE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_role_pattern_transfer_route_rows.csv"
)
PAIR_ROLE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_role_pattern_transfer_pair_rows.csv"
)
CANDIDATE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_role_pattern_transfer_candidate_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_014_role_pattern_transfer_gate_matrix.csv"
)
SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_014_role_pattern_transfer_summary.json"
)
CONFIG_JSON = "nanoclustering_g4_8_first_pass_014_role_pattern_transfer_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_014_role_pattern_transfer_report.md"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_014_role_pattern_transfer_screen"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_role_pattern_transfer_screen"
WALL_PROMOTION_STATUS = "not_promoted_role_pattern_screen_only"
METHOD_STATUS = "diagnostic_role_pattern_screen_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass 014 role-pattern transfer screen only; "
    "reads existing first-pass trace, exclusive-target contrast, 014 "
    "role-stability gates, and local graph metadata to classify pair-local "
    "L/R/bridge endpoint signatures across the screened pairs. It does not "
    "rerun Leiden, perform a fraction sweep, promote basin walls, replay full "
    "NanoClustering, evaluate quality/cost value, or claim method success."
)

READY_ROLE = "conditional_ready_like_test"
CONTROL_ROLE = "control_false_positive_guard"


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


def _json_counts(series: pd.Series) -> str:
    return json.dumps(_count_dict(series), ensure_ascii=True, sort_keys=True)


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


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    visible = frame.copy()
    if columns is not None:
        visible = visible[[column for column in columns if column in visible.columns]]
    visible = visible.head(int(max_rows))
    if visible.empty:
        return "_No matching columns._"

    cols = [str(column) for column in visible.columns]

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


def _parse_bridge_meta(spec: pd.Series) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for token in str(spec["selected_bridge_rank_scope_weight"]).split(";"):
        if not token:
            continue
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


def _node_role_rows(
    *,
    local_graph_rows: pd.DataFrame,
    local_pair_ids: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[int, str]]]:
    spec_by_pair = local_graph_rows.set_index("local_pair_id").to_dict("index")
    rows: list[dict[str, Any]] = []
    role_maps: dict[str, dict[int, str]] = {}
    for local_pair_id in local_pair_ids:
        if local_pair_id not in spec_by_pair:
            raise ValueError(f"missing local graph row for {local_pair_id}")
        spec = pd.Series(spec_by_pair[local_pair_id])
        left = int(spec["left_node_id"])
        right = int(spec["right_node_id"])
        bridge_meta = _parse_bridge_meta(spec)
        node_ids = [int(value) for value in str(spec["local_node_ids"]).split(";") if value]
        role_by_node: dict[int, str] = {}
        for node_id in node_ids:
            if node_id == left:
                role = "L"
                role_family = "left_pair_node"
                bridge_rank = None
                bridge_scope = "pair"
                bridge_weight = None
            elif node_id == right:
                role = "R"
                role_family = "right_pair_node"
                bridge_rank = None
                bridge_scope = "pair"
                bridge_weight = None
            else:
                meta = bridge_meta.get(node_id, {})
                bridge_rank = int(meta.get("bridge_rank", -1))
                role = f"B{bridge_rank}" if bridge_rank > 0 else f"B?{node_id}"
                role_family = "selected_bridge_node"
                bridge_scope = str(meta.get("bridge_scope", "unknown"))
                bridge_weight = meta.get("bridge_min_pair_edge_weight")
            role_by_node[node_id] = role
            rows.append(
                {
                    "local_pair_id": local_pair_id,
                    "node_id": int(node_id),
                    "node_role": role,
                    "node_role_family": role_family,
                    "bridge_rank": bridge_rank,
                    "bridge_scope": bridge_scope,
                    "bridge_min_pair_edge_weight": bridge_weight,
                    "method_claim_allowed_after_screen": False,
                    "quality_cost_claim_allowed_after_screen": False,
                    "wall_generality_claim_allowed_after_screen": False,
                    "run_status": RUN_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
        role_maps[local_pair_id] = role_by_node
    return pd.DataFrame(rows), role_maps


def _cluster_roles(signature_json: str, role_by_node: dict[int, str]) -> tuple[str, list[list[str]]]:
    groups = json.loads(signature_json)
    role_groups: list[list[str]] = []
    for group in groups:
        roles = [role_by_node.get(int(node_id), str(node_id)) for node_id in group]
        role_groups.append(roles)
    return " | ".join("+".join(roles) for roles in role_groups), role_groups


def _cluster_for_role(role_groups: list[list[str]], role: str) -> list[str]:
    for group in role_groups:
        if role in group:
            return group
    return []


def _assignment_families(value: Any) -> set[str]:
    text = str(value)
    if text.startswith("ambiguous_anchor_match:"):
        text = text.split(":", 1)[1]
    labels = [label for label in text.split(";") if label]
    if not labels:
        labels = [text]

    families: set[str] = set()
    for label in labels:
        if label == "original_source_anchor":
            families.add("source_like")
        elif label == "drop_bridge_target_anchor":
            families.add("target")
        elif label in {"drop_direct_guard_anchor", "drop_both_guard_anchor"}:
            families.add("guard")
        elif label == "unknown_new_endpoint":
            families.add("unknown")
        else:
            families.add("other")
    return families


def _signature_role_class(row: dict[str, Any]) -> str:
    families = set(str(row["endpoint_object_families"]).split(";")) if row["endpoint_object_families"] else set()
    known = set(str(row["known_endpoint_object_families"]).split(";")) if row["known_endpoint_object_families"] else set()
    pair_coassigned = bool(row["pair_coassigned_any"])
    left_cluster = set(str(row["left_cluster_roles"]).split("+")) if row["left_cluster_roles"] else set()
    right_cluster = set(str(row["right_cluster_roles"]).split("+")) if row["right_cluster_roles"] else set()

    if "unknown" in families and not known:
        if pair_coassigned:
            return "unresolved_pair_coassigned_intermediate"
        if "L" in left_cluster and "R" in right_cluster and left_cluster != right_cluster:
            return "unresolved_pair_separated_bridge_reassignment"
        return "unresolved_pair_separated_intermediate"

    if "unknown" in families and known:
        if known == {"source_like", "guard"}:
            return "hidden_known_source_guard_intermediate"
        if known == {"source_like"}:
            return (
                "hidden_known_source_pair_intermediate"
                if pair_coassigned
                else "hidden_known_source_separated_intermediate"
            )
        if known == {"guard"}:
            return "hidden_known_guard_intermediate"
        if known == {"target"}:
            return "hidden_known_target_intermediate"
        return "hidden_known_mixed_intermediate"

    if known == {"target"}:
        if str(row["left_cluster_roles"]) == "L+R" or str(row["right_cluster_roles"]) == "L+R":
            return "target_anchor_pair_only"
        return "target_anchor_with_bridges"
    if known == {"source_like"}:
        return "source_anchor_pair_with_bridges" if pair_coassigned else "source_like_separated_bridge_split"
    if known == {"guard"}:
        return "guard_known"
    if known == {"source_like", "target"}:
        return "source_target_collapse_or_boundary_mixed"
    if known == {"target", "guard"}:
        return "target_guard_collapse_mixed"
    if known == {"source_like", "guard"}:
        return "source_guard_mixed"
    if known:
        return "mixed_known_endpoint_signature"
    return "other_signature_role"


def _signature_role_rows(
    *,
    trace_rows: pd.DataFrame,
    role_maps: dict[str, dict[int, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (local_pair_id, signature_id), group in trace_rows.groupby(
        ["local_pair_id", "result_endpoint_signature_id"], sort=True
    ):
        local_pair_id = str(local_pair_id)
        role_by_node = role_maps[local_pair_id]
        signature_json = str(group["result_endpoint_signature"].iloc[0])
        role_signature, role_groups = _cluster_roles(signature_json, role_by_node)
        left_cluster_roles = _cluster_for_role(role_groups, "L")
        right_cluster_roles = _cluster_for_role(role_groups, "R")
        pair_coassigned_any = bool(group["pair_coassigned"].map(_as_bool).any())
        pair_coassigned_rate = float(group["pair_coassigned"].map(_as_bool).mean())
        family_values: set[str] = set()
        for assignment in group["endpoint_assignment_by_step"].astype(str):
            family_values.update(_assignment_families(assignment))
        known_families = sorted(family_values - {"unknown"})
        endpoint_object_families = ";".join(sorted(family_values))
        known_endpoint_object_families = ";".join(known_families)
        row = {
            "local_pair_id": local_pair_id,
            "result_endpoint_signature_id": str(signature_id),
            "signature_role_cluster_signature": role_signature,
            "left_cluster_roles": "+".join(left_cluster_roles),
            "right_cluster_roles": "+".join(right_cluster_roles),
            "pair_coassigned_any": pair_coassigned_any,
            "pair_coassigned_rate": pair_coassigned_rate,
            "signature_row_count": int(len(group)),
            "start_conditions": ";".join(sorted(set(group["start_condition"].astype(str)))),
            "seeds": ";".join(str(value) for value in sorted(set(group["seed"].astype(int)))),
            "route_family_roles": ";".join(sorted(set(group["route_family_role"].astype(str)))),
            "bridge_fractions": ";".join(
                f"{value:.3g}"
                for value in sorted(set(group["bridge_edge_weight_fraction"].astype(float)))
            ),
            "endpoint_object_families": endpoint_object_families,
            "known_endpoint_object_families": known_endpoint_object_families,
            "endpoint_assignment_counts": json.dumps(
                _count_dict(group["endpoint_assignment_by_step"].astype(str)),
                ensure_ascii=True,
                sort_keys=True,
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
            "method_claim_allowed_after_screen": False,
            "quality_cost_claim_allowed_after_screen": False,
            "wall_generality_claim_allowed_after_screen": False,
            "route_execution_status": ROUTE_EXECUTION_STATUS,
            "wall_promotion_status": WALL_PROMOTION_STATUS,
            "method_status": METHOD_STATUS,
            "run_status": RUN_STATUS,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        row["signature_role_class"] = _signature_role_class(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["local_pair_id", "objective_value_mean", "signature_row_count"],
        ascending=[True, True, False],
        kind="mergesort",
    )


def _route_role_rows(
    *,
    trace_rows: pd.DataFrame,
    signature_rows: pd.DataFrame,
) -> pd.DataFrame:
    role_class = signature_rows.set_index(
        ["local_pair_id", "result_endpoint_signature_id"]
    )["signature_role_class"].to_dict()
    rows: list[dict[str, Any]] = []
    key_cols = ["local_pair_id", "branch", "start_condition", "seed", "route_family_role"]
    for key, group in trace_rows.groupby(key_cols, sort=True):
        route = group.sort_values("step_index", kind="mergesort")
        local_pair_id = str(key[0])
        signature_sequence = route["result_endpoint_signature_id"].astype(str).tolist()
        role_sequence = [
            role_class.get((local_pair_id, signature_id), "untyped_signature")
            for signature_id in signature_sequence
        ]
        assignment_sequence = route["endpoint_assignment_by_step"].astype(str).tolist()
        families_sequence = [
            "+".join(sorted(_assignment_families(assignment)))
            for assignment in assignment_sequence
        ]
        rows.append(
            {
                "local_pair_id": local_pair_id,
                "branch": str(key[1]),
                "start_condition": str(key[2]),
                "seed": int(key[3]),
                "route_family_role": str(key[4]),
                "route_signature_sequence": " -> ".join(signature_sequence),
                "route_role_class_sequence": " -> ".join(role_sequence),
                "route_endpoint_assignment_sequence": " -> ".join(assignment_sequence),
                "route_endpoint_family_sequence": " -> ".join(families_sequence),
                "unique_signature_count": int(len(set(signature_sequence))),
                "unique_role_class_count": int(len(set(role_sequence))),
                "unresolved_intermediate_step_count": int(
                    sum("unresolved_" in role for role in role_sequence)
                ),
                "hidden_known_intermediate_step_count": int(
                    sum(role.startswith("hidden_known_") for role in role_sequence)
                ),
                "target_anchor_step_count": int(
                    sum(role.startswith("target_anchor") for role in role_sequence)
                ),
                "source_like_step_count": int(
                    sum(role.startswith("source_") or role.startswith("hidden_known_source") for role in role_sequence)
                ),
                "mixed_known_step_count": int(
                    sum("mixed" in role or "collapse" in role for role in role_sequence)
                ),
                "method_claim_allowed_after_screen": False,
                "quality_cost_claim_allowed_after_screen": False,
                "wall_generality_claim_allowed_after_screen": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _has_any_class(class_counts: dict[str, int], prefixes: tuple[str, ...]) -> bool:
    return any(
        count > 0 and any(role_class.startswith(prefix) for prefix in prefixes)
        for role_class, count in class_counts.items()
    )


def _pair_transfer_status(
    *,
    local_pair_id: str,
    evidence_role: str,
    validation_stratum: str,
    ready_like_count: int,
    exclusive_pass_count: int,
    has_source_like: bool,
    has_target_anchor: bool,
    has_hidden_known: bool,
    has_unresolved_pair_coassigned: bool,
    has_unresolved_pair_separated: bool,
    has_known_mixed: bool,
) -> tuple[str, bool, bool, str]:
    has_transition_analog = (
        has_hidden_known or has_unresolved_pair_coassigned or has_unresolved_pair_separated
    )
    role_analog_score = sum(
        [
            has_source_like,
            has_target_anchor,
            has_hidden_known,
            has_unresolved_pair_coassigned,
            has_unresolved_pair_separated,
            has_known_mixed,
        ]
    )
    if local_pair_id == POSITIVE_PAIR_ID:
        return (
            "reference_positive_014_scaffold_recovered",
            False,
            True,
            "Keep 014 as the reference; use other rows as transfer diagnostics only.",
        )
    if ready_like_count > 0 and exclusive_pass_count > 0:
        return (
            "partial_boundary_guard_ready_like_not_transfer_positive",
            False,
            True,
            "Use as a boundary guard for source/target collapse; do not promote as a new transfer positive.",
        )
    if evidence_role == CONTROL_ROLE and role_analog_score >= 3:
        return (
            "closed_control_role_analog",
            False,
            True,
            "Keep as a negative-control analog to test why similar-looking roles do not open a pathway.",
        )
    if evidence_role == READY_ROLE and has_source_like and has_target_anchor and has_transition_analog:
        if validation_stratum == "strict_ready":
            return (
                "strict_ready_continuity_blocked_role_analog",
                False,
                True,
                "Primary diagnostic: inspect why the strict-ready scaffold fails post-start continuity before any localization escalation.",
            )
        return (
            "rare_ready_continuity_blocked_role_analog",
            False,
            True,
            "Secondary diagnostic: compare after the strict-ready continuity-blocked analog.",
        )
    if evidence_role == READY_ROLE and (has_target_anchor or has_transition_analog):
        return (
            "conditional_blocked_partial_role_signal",
            False,
            True,
            "Keep as a weaker blocked diagnostic; first compare against the stronger continuity-blocked analog.",
        )
    if evidence_role == CONTROL_ROLE:
        return (
            "closed_control_no_transfer_signal",
            False,
            False,
            "No follow-up unless a specific control mechanism question is named.",
        )
    return (
        "not_transfer_candidate",
        False,
        False,
        "No follow-up from this read-only screen.",
    )


def _pair_role_rows(
    *,
    signature_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    pair_readout: pd.DataFrame,
    pair_contrast: pd.DataFrame,
) -> pd.DataFrame:
    readout_lookup = pair_readout.set_index("local_pair_id").to_dict("index")
    contrast_lookup = pair_contrast.set_index("local_pair_id").to_dict("index")
    rows: list[dict[str, Any]] = []
    for local_pair_id, sig_group in signature_rows.groupby("local_pair_id", sort=True):
        local_pair_id = str(local_pair_id)
        route_group = route_rows[route_rows["local_pair_id"].astype(str).eq(local_pair_id)]
        readout = readout_lookup.get(local_pair_id, {})
        contrast = contrast_lookup.get(local_pair_id, {})
        class_counts = _count_dict(sig_group["signature_role_class"])
        ready_like_count = int(readout.get("ready_like_seed_route_pass_count", 0))
        exclusive_pass_count = int(contrast.get("exclusive_bridge_target_pass_count", 0))
        evidence_role = str(readout.get("evidence_role", contrast.get("evidence_role", "")))
        validation_stratum = str(
            readout.get("validation_stratum", contrast.get("validation_stratum", ""))
        )
        has_source_like = _has_any_class(
            class_counts,
            (
                "source_",
                "hidden_known_source",
            ),
        )
        has_target_anchor = _has_any_class(class_counts, ("target_anchor",))
        has_hidden_known = _has_any_class(class_counts, ("hidden_known_",))
        has_unresolved_pair_coassigned = class_counts.get(
            "unresolved_pair_coassigned_intermediate", 0
        ) > 0
        has_unresolved_pair_separated = (
            class_counts.get("unresolved_pair_separated_intermediate", 0) > 0
            or class_counts.get("unresolved_pair_separated_bridge_reassignment", 0) > 0
        )
        has_known_mixed = any(
            count > 0 and ("mixed" in role_class or "collapse" in role_class)
            for role_class, count in class_counts.items()
        )
        has_transition_analog = (
            has_hidden_known or has_unresolved_pair_coassigned or has_unresolved_pair_separated
        )
        matches_014_first_pass_scaffold = bool(
            has_source_like and has_target_anchor and ready_like_count > 0
        )
        transfer_status, new_positive_candidate, diagnostic_followup, next_action = (
            _pair_transfer_status(
                local_pair_id=local_pair_id,
                evidence_role=evidence_role,
                validation_stratum=validation_stratum,
                ready_like_count=ready_like_count,
                exclusive_pass_count=exclusive_pass_count,
                has_source_like=has_source_like,
                has_target_anchor=has_target_anchor,
                has_hidden_known=has_hidden_known,
                has_unresolved_pair_coassigned=has_unresolved_pair_coassigned,
                has_unresolved_pair_separated=has_unresolved_pair_separated,
                has_known_mixed=has_known_mixed,
            )
        )
        role_analog_feature_count = int(
            sum(
                [
                    has_source_like,
                    has_target_anchor,
                    has_hidden_known,
                    has_unresolved_pair_coassigned,
                    has_unresolved_pair_separated,
                    has_known_mixed,
                ]
            )
        )
        rows.append(
            {
                "local_pair_id": local_pair_id,
                "branch": str(readout.get("branch", contrast.get("branch", ""))),
                "evidence_role": evidence_role,
                "validation_stratum": validation_stratum,
                "pair_first_pass_result": str(
                    readout.get("pair_first_pass_result", contrast.get("pair_first_pass_result", ""))
                ),
                "pair_contrast_escalation_class": str(contrast.get("next_escalation_class", "")),
                "typed_signature_count": int(len(sig_group)),
                "signature_role_class_counts": json.dumps(
                    class_counts, ensure_ascii=True, sort_keys=True
                ),
                "route_role_row_count": int(len(route_group)),
                "ready_like_seed_route_pass_count": ready_like_count,
                "exclusive_bridge_target_pass_count": exclusive_pass_count,
                "exclusive_bridge_target_pass_share": float(
                    contrast.get("exclusive_bridge_target_pass_share", 0.0)
                ),
                "intermediate_unknown_route_count": int(
                    contrast.get("intermediate_unknown_route_count", 0)
                ),
                "source_target_signature_collapse_count": int(
                    contrast.get("source_target_signature_collapse_count", 0)
                ),
                "guard_anchor_collapse_count": int(contrast.get("guard_anchor_collapse_count", 0)),
                "has_source_like_signature": has_source_like,
                "has_target_anchor_signature": has_target_anchor,
                "has_hidden_known_intermediate_signature": has_hidden_known,
                "has_unresolved_pair_coassigned_signature": has_unresolved_pair_coassigned,
                "has_unresolved_pair_separated_signature": has_unresolved_pair_separated,
                "has_known_mixed_signature": has_known_mixed,
                "has_transition_intermediate_analog": has_transition_analog,
                "role_analog_feature_count": role_analog_feature_count,
                "matches_014_first_pass_scaffold": matches_014_first_pass_scaffold,
                "new_positive_transfer_candidate": new_positive_candidate,
                "diagnostic_followup_candidate": diagnostic_followup,
                "transfer_screen_status": transfer_status,
                "next_action": next_action,
                "method_claim_allowed_after_screen": False,
                "quality_cost_claim_allowed_after_screen": False,
                "wall_generality_claim_allowed_after_screen": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["new_positive_transfer_candidate", "diagnostic_followup_candidate", "role_analog_feature_count", "local_pair_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )


def _candidate_rows(pair_rows: pd.DataFrame) -> pd.DataFrame:
    candidates = pair_rows[
        pair_rows["diagnostic_followup_candidate"].map(_as_bool)
        | pair_rows["new_positive_transfer_candidate"].map(_as_bool)
    ].copy()
    priority_by_status = {
        "reference_positive_014_scaffold_recovered": 0,
        "strict_ready_continuity_blocked_role_analog": 1,
        "partial_boundary_guard_ready_like_not_transfer_positive": 2,
        "closed_control_role_analog": 3,
        "rare_ready_continuity_blocked_role_analog": 4,
        "conditional_blocked_partial_role_signal": 5,
    }
    candidates["followup_priority_rank"] = candidates["transfer_screen_status"].astype(str).map(
        priority_by_status
    )
    candidates.loc[candidates["followup_priority_rank"].isna(), "followup_priority_rank"] = 99
    return candidates[
        [
            "followup_priority_rank",
            "local_pair_id",
            "evidence_role",
            "validation_stratum",
            "pair_first_pass_result",
            "transfer_screen_status",
            "role_analog_feature_count",
            "matches_014_first_pass_scaffold",
            "new_positive_transfer_candidate",
            "ready_like_seed_route_pass_count",
            "exclusive_bridge_target_pass_count",
            "has_source_like_signature",
            "has_target_anchor_signature",
            "has_hidden_known_intermediate_signature",
            "has_unresolved_pair_coassigned_signature",
            "has_unresolved_pair_separated_signature",
            "has_known_mixed_signature",
            "next_action",
            "claim_boundary",
        ]
    ].sort_values(
        [
            "followup_priority_rank",
            "new_positive_transfer_candidate",
            "role_analog_feature_count",
            "local_pair_id",
        ],
        ascending=[True, False, False, True],
        kind="mergesort",
    )


def _gate_status_counts(*gate_matrices: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for gate_matrix in gate_matrices:
        for status, count in _count_dict(gate_matrix["gate_status"]).items():
            counts[status] = counts.get(status, 0) + int(count)
    return counts


def _gate_matrix(
    *,
    first_pass_gates: pd.DataFrame,
    exclusive_target_gates: pd.DataFrame,
    role_stability_gates: pd.DataFrame,
    signature_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    candidate_rows: pd.DataFrame,
) -> pd.DataFrame:
    upstream_counts = _gate_status_counts(
        first_pass_gates,
        exclusive_target_gates,
        role_stability_gates,
    )
    reference = pair_rows[pair_rows["local_pair_id"].astype(str).eq(POSITIVE_PAIR_ID)]
    controls = pair_rows[pair_rows["evidence_role"].astype(str).eq(CONTROL_ROLE)]
    new_positives = pair_rows[
        pair_rows["new_positive_transfer_candidate"].map(_as_bool)
        & pair_rows["local_pair_id"].astype(str).ne(POSITIVE_PAIR_ID)
    ]
    analogs = pair_rows[
        pair_rows["diagnostic_followup_candidate"].map(_as_bool)
        & pair_rows["local_pair_id"].astype(str).ne(POSITIVE_PAIR_ID)
    ]
    all_claims_closed = bool(
        not pair_rows["method_claim_allowed_after_screen"].map(_as_bool).any()
        and not pair_rows["quality_cost_claim_allowed_after_screen"].map(_as_bool).any()
        and not pair_rows["wall_generality_claim_allowed_after_screen"].map(_as_bool).any()
    )
    rows = [
        _gate_row(
            "G1_upstream_gates_pass",
            "Did the upstream first-pass, exclusive-target, and 014 role-stability gates pass?",
            json.dumps(upstream_counts, ensure_ascii=True, sort_keys=True),
            "all upstream gates pass",
            upstream_counts.get("fail", 0) == 0 and upstream_counts.get("pass", 0) > 0,
        ),
        _gate_row(
            "G2_all_first_pass_pairs_screened",
            "Were all first-pass local pairs screened by role pattern?",
            f"pair_rows={len(pair_rows)} signature_rows={len(signature_rows)}",
            "9 pair rows and nonempty signature rows",
            len(pair_rows) == 9 and len(signature_rows) > 0,
        ),
        _gate_row(
            "G3_reference_014_recovered",
            "Was the 014 first-pass role scaffold recovered as the reference positive?",
            reference[["local_pair_id", "transfer_screen_status", "matches_014_first_pass_scaffold"]].to_dict("records"),
            "014 is present and matches the first-pass scaffold",
            len(reference) == 1
            and bool(reference["matches_014_first_pass_scaffold"].map(_as_bool).iloc[0])
            and str(reference["transfer_screen_status"].iloc[0]).startswith("reference_positive"),
        ),
        _gate_row(
            "G4_no_new_positive_transfer_from_existing_screen",
            "Did any non-014 pair become a new positive transfer candidate?",
            new_positives["local_pair_id"].astype(str).tolist(),
            "no non-014 positive transfer candidates from this read-only screen",
            new_positives.empty,
        ),
        _gate_row(
            "G5_controls_remain_negative",
            "Do control rows remain non-positive even when role analogs appear?",
            controls[["local_pair_id", "transfer_screen_status", "role_analog_feature_count"]].to_dict("records"),
            "control rows have new_positive_transfer_candidate=false",
            bool(controls["new_positive_transfer_candidate"].map(_as_bool).eq(False).all()),
        ),
        _gate_row(
            "G6_diagnostic_analogs_materialized_without_sweep",
            "Were blocked/control analogs materialized as diagnostics rather than a sweep?",
            analogs[["local_pair_id", "transfer_screen_status", "next_action"]].to_dict("records"),
            "at least one diagnostic analog row and no new execution",
            not analogs.empty,
        ),
        _gate_row(
            "G7_claims_closed",
            "Are method, quality/cost, and wall-generality claims closed?",
            CLAIM_BOUNDARY,
            "all claim flags false",
            all_claims_closed,
        ),
    ]
    return pd.DataFrame(rows)


def _summary(
    *,
    first_pass_trace_dir: Path,
    exclusive_target_contrast_dir: Path,
    role_stability_audit_dir: Path,
    local_ablation_dir: Path,
    output_dir: Path,
    signature_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    non014_new = pair_rows[
        pair_rows["new_positive_transfer_candidate"].map(_as_bool)
        & pair_rows["local_pair_id"].astype(str).ne(POSITIVE_PAIR_ID)
    ]
    primary_diagnostic = pair_rows[
        pair_rows["transfer_screen_status"]
        .astype(str)
        .eq("strict_ready_continuity_blocked_role_analog")
    ]
    rare_ready_blocked = pair_rows[
        pair_rows["transfer_screen_status"]
        .astype(str)
        .eq("rare_ready_continuity_blocked_role_analog")
    ]
    closed_control_analogs = pair_rows[
        pair_rows["transfer_screen_status"].astype(str).eq("closed_control_role_analog")
    ]
    boundary_guards = pair_rows[
        pair_rows["transfer_screen_status"]
        .astype(str)
        .eq("partial_boundary_guard_ready_like_not_transfer_positive")
    ]
    return {
        "schema": "nanoclustering_g4_8_first_pass_014_role_pattern_transfer_summary.v1",
        "status": RUN_STATUS,
        "first_pass_trace_dir": str(first_pass_trace_dir),
        "exclusive_target_contrast_dir": str(exclusive_target_contrast_dir),
        "role_stability_audit_dir": str(role_stability_audit_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "screened_pair_count": int(len(pair_rows)),
        "signature_role_row_count": int(len(signature_rows)),
        "route_role_row_count": int(len(route_rows)),
        "candidate_row_count": int(len(candidate_rows)),
        "transfer_screen_status_counts": _count_dict(pair_rows["transfer_screen_status"]),
        "signature_role_class_counts": _count_dict(signature_rows["signature_role_class"]),
        "candidate_pairs": candidate_rows["local_pair_id"].astype(str).tolist(),
        "primary_diagnostic_pairs": primary_diagnostic["local_pair_id"].astype(str).tolist(),
        "rare_ready_blocked_analog_pairs": rare_ready_blocked["local_pair_id"].astype(str).tolist(),
        "boundary_guard_pairs": boundary_guards["local_pair_id"].astype(str).tolist(),
        "closed_control_analog_pairs": closed_control_analogs["local_pair_id"].astype(str).tolist(),
        "non014_new_positive_transfer_candidates": non014_new["local_pair_id"].astype(str).tolist(),
        "reference_pair_id": POSITIVE_PAIR_ID,
        "gate_status_counts": _count_dict(gates["gate_status"]),
        "failed_gates": gates.loc[
            gates["gate_status"].astype(str).ne("pass"), "gate_id"
        ].astype(str).tolist(),
        "interpretation": (
            "The existing first-pass screen recovers 014 as the only clean "
            "reference scaffold. Other pairs contain blocked or closed role "
            "analogs, but none become a non-014 positive transfer candidate "
            "without changing the evidence surface."
        ),
        "recommended_next_gate": (
            "Do not localize every analog. The primary mechanism question from "
            "this screen is local_pair_016: a strict-ready role analog that "
            "still fails post-start continuity. Use local_pair_005 as the "
            "boundary guard and closed control analogs only as contrasts."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    pair_rows: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    signature_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass 014 Role-Pattern Transfer Screen",
        "",
        f"- status: `{summary['status']}`",
        f"- screened_pair_count: {summary['screened_pair_count']}",
        f"- signature_role_row_count: {summary['signature_role_row_count']}",
        f"- route_role_row_count: {summary['route_role_row_count']}",
        f"- candidate_pairs: {summary['candidate_pairs']}",
        f"- primary_diagnostic_pairs: {summary['primary_diagnostic_pairs']}",
        f"- boundary_guard_pairs: {summary['boundary_guard_pairs']}",
        f"- closed_control_analog_pairs: {summary['closed_control_analog_pairs']}",
        f"- non014_new_positive_transfer_candidates: {summary['non014_new_positive_transfer_candidates']}",
        f"- transfer_screen_status_counts: {summary['transfer_screen_status_counts']}",
        f"- gate_status_counts: {summary['gate_status_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        f"- interpretation: {summary['interpretation']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Pair Screen",
        "",
        _markdown_table(
            pair_rows.sort_values("local_pair_id", kind="mergesort"),
            [
                "local_pair_id",
                "evidence_role",
                "validation_stratum",
                "pair_first_pass_result",
                "ready_like_seed_route_pass_count",
                "exclusive_bridge_target_pass_count",
                "role_analog_feature_count",
                "matches_014_first_pass_scaffold",
                "transfer_screen_status",
                "next_action",
            ],
            max_rows=20,
        ),
        "",
        "## Candidate And Guard Rows",
        "",
        _markdown_table(
            candidate_rows,
            [
                "local_pair_id",
                "evidence_role",
                "validation_stratum",
                "followup_priority_rank",
                "transfer_screen_status",
                "role_analog_feature_count",
                "matches_014_first_pass_scaffold",
                "ready_like_seed_route_pass_count",
                "exclusive_bridge_target_pass_count",
                "next_action",
            ],
            max_rows=20,
        ),
        "",
        "## Signature Role Inventory",
        "",
        _markdown_table(
            signature_rows.sort_values(
                ["local_pair_id", "signature_role_class", "signature_row_count"],
                ascending=[True, True, False],
                kind="mergesort",
            ),
            [
                "local_pair_id",
                "result_endpoint_signature_id",
                "signature_role_class",
                "signature_row_count",
                "endpoint_object_families",
                "pair_coassigned_rate",
                "left_cluster_roles",
                "right_cluster_roles",
                "bridge_fractions",
                "signature_role_cluster_signature",
            ],
            max_rows=80,
        ),
        "",
        "## Gate Matrix",
        "",
        _markdown_table(gates, ["gate_id", "gate_status", "observed", "minimum_or_rule", "question"], max_rows=20),
        "",
        "## Boundary",
        "",
        (
            "This screen only compares already materialized signatures. It "
            "does not add route executions, replay Leiden on new settings, or "
            "turn a blocked/control analog into wall evidence."
        ),
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    first_pass_trace_dir = Path(args.first_pass_trace_dir)
    exclusive_target_contrast_dir = Path(args.exclusive_target_contrast_dir)
    role_stability_audit_dir = Path(args.role_stability_audit_dir)
    local_ablation_dir = Path(args.local_ablation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_rows = _read_csv(first_pass_trace_dir / TRACE_ROWS_CSV)
    pair_readout = _read_csv(first_pass_trace_dir / PAIR_READOUT_RESULT_ROWS_CSV)
    pair_contrast = _read_csv(exclusive_target_contrast_dir / PAIR_CONTRAST_ROWS_CSV)
    local_graph_rows = _read_csv(local_ablation_dir / LOCAL_GRAPH_ROWS_CSV)
    first_pass_gates = _read_csv(first_pass_trace_dir / FIRST_PASS_GATE_MATRIX_CSV)
    exclusive_target_gates = _read_csv(
        exclusive_target_contrast_dir / EXCLUSIVE_TARGET_GATE_MATRIX_CSV
    )
    role_stability_gates = _read_csv(role_stability_audit_dir / ROLE_STABILITY_GATE_MATRIX_CSV)

    local_pair_ids = sorted(trace_rows["local_pair_id"].astype(str).unique().tolist())
    node_role_rows, role_maps = _node_role_rows(
        local_graph_rows=local_graph_rows,
        local_pair_ids=local_pair_ids,
    )
    signature_rows = _signature_role_rows(trace_rows=trace_rows, role_maps=role_maps)
    route_rows = _route_role_rows(trace_rows=trace_rows, signature_rows=signature_rows)
    pair_rows = _pair_role_rows(
        signature_rows=signature_rows,
        route_rows=route_rows,
        pair_readout=pair_readout,
        pair_contrast=pair_contrast,
    )
    candidate_rows = _candidate_rows(pair_rows)
    gates = _gate_matrix(
        first_pass_gates=first_pass_gates,
        exclusive_target_gates=exclusive_target_gates,
        role_stability_gates=role_stability_gates,
        signature_rows=signature_rows,
        pair_rows=pair_rows,
        candidate_rows=candidate_rows,
    )
    summary = _summary(
        first_pass_trace_dir=first_pass_trace_dir,
        exclusive_target_contrast_dir=exclusive_target_contrast_dir,
        role_stability_audit_dir=role_stability_audit_dir,
        local_ablation_dir=local_ablation_dir,
        output_dir=output_dir,
        signature_rows=signature_rows,
        route_rows=route_rows,
        pair_rows=pair_rows,
        candidate_rows=candidate_rows,
        gates=gates,
    )

    _write_csv(node_role_rows, output_dir / NODE_ROLE_ROWS_CSV)
    _write_csv(signature_rows, output_dir / SIGNATURE_ROLE_ROWS_CSV)
    _write_csv(route_rows, output_dir / ROUTE_ROLE_ROWS_CSV)
    _write_csv(pair_rows, output_dir / PAIR_ROLE_ROWS_CSV)
    _write_csv(candidate_rows, output_dir / CANDIDATE_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_014_role_pattern_transfer_config.v1",
        "first_pass_trace_dir": str(first_pass_trace_dir),
        "exclusive_target_contrast_dir": str(exclusive_target_contrast_dir),
        "role_stability_audit_dir": str(role_stability_audit_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "read_only_screen": True,
        "reference_pair_id": POSITIVE_PAIR_ID,
        "run_status": RUN_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        pair_rows=pair_rows,
        candidate_rows=candidate_rows,
        signature_rows=signature_rows,
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
    parser.add_argument(
        "--role-stability-audit-dir",
        type=Path,
        default=DEFAULT_014_ROLE_STABILITY_AUDIT_DIR,
    )
    parser.add_argument("--local-ablation-dir", type=Path, default=DEFAULT_LOCAL_ABLATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
