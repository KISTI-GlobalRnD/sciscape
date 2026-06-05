#!/usr/bin/env python3
"""Screen typed-transient predicate choices for the basin/pathway definition.

This read-only audit takes the existing G4.8 first-pass, 014 role-pattern
transfer, 016 continuity-block, and transition-synthesis outputs and turns the
definition question into explicit accept/reject predicates. It intentionally
includes two over-broad negative predicates so the boundary/control leaks are
visible in the same table as the guarded candidate.

The script does not rerun Leiden, execute new routes, promote wall claims,
evaluate quality/cost value, or claim method success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)


DEFAULT_FIRST_PASS_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_g4_8_fresh_axis_b_first_pass_trace_gamma1e5_20260604"
)
DEFAULT_TRANSFER_SCREEN_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_014_role_pattern_transfer_screen_gamma1e5_20260605"
)
DEFAULT_CONTINUITY_016_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_continuity_block_audit_gamma1e5_20260605"
)
DEFAULT_SYNTHESIS_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_transition_evidence_synthesis_gamma1e5_20260605"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_typed_transient_predicate_screen_gamma1e5_20260605"
)

PAIR_FEATURE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_typed_transient_predicate_pair_feature_rows.csv"
)
PAIR_PREDICATE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_typed_transient_predicate_pair_predicate_rows.csv"
)
PREDICATE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_typed_transient_predicate_rows.csv"
)
DEFINITION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_typed_transient_predicate_definition_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_typed_transient_predicate_gate_matrix.csv"
)
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_typed_transient_predicate_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_typed_transient_predicate_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_typed_transient_predicate_report.md"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_typed_transient_predicate_screen"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_predicate_screen"
WALL_PROMOTION_STATUS = "not_promoted_predicate_screen_only"
METHOD_STATUS = "definition_predicate_screen_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass typed-transient predicate screen only; "
    "reads existing first-pass, transfer-screen, 016 continuity-block, and "
    "transition-synthesis outputs to compare basin/pathway definition "
    "predicates. It does not rerun Leiden, perform a fraction sweep, promote "
    "basin walls, replay full NanoClustering, evaluate quality/cost value, or "
    "claim method success."
)

REFERENCE_PAIR_ID = "local_pair_014"
PRIMARY_TYPED_TRANSIENT_PAIR_ID = "local_pair_016"
BOUNDARY_GUARD_PAIR_ID = "local_pair_005"


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


def _as_int(value: Any) -> int:
    if pd.isna(value):
        return 0
    return int(value)


def _bool_count(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int(frame[column].map(_as_bool).sum())


def _list_json(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=True)


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


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
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


def _load_context(args: argparse.Namespace) -> dict[str, Any]:
    first_pass_dir = Path(args.first_pass_dir)
    transfer_screen_dir = Path(args.transfer_screen_dir)
    continuity_016_dir = Path(args.continuity_016_dir)
    synthesis_dir = Path(args.synthesis_dir)

    summaries = {
        "first_pass": _read_json(
            first_pass_dir / "nanoclustering_g4_8_fresh_axis_b_first_pass_trace_summary.json"
        ),
        "transfer_screen": _read_json(
            transfer_screen_dir / "nanoclustering_g4_8_first_pass_014_role_pattern_transfer_summary.json"
        ),
        "continuity_016": _read_json(
            continuity_016_dir / "nanoclustering_g4_8_first_pass_016_continuity_block_summary.json"
        ),
        "synthesis": _read_json(
            synthesis_dir / "nanoclustering_g4_8_first_pass_transition_evidence_synthesis_summary.json"
        ),
    }
    tables = {
        "first_pass_pair": _read_csv(
            first_pass_dir / "nanoclustering_g4_8_fresh_axis_b_first_pass_pair_readout_result_rows.csv"
        ),
        "first_pass_route": _read_csv(
            first_pass_dir / "nanoclustering_g4_8_fresh_axis_b_first_pass_route_readout_result_rows.csv"
        ),
        "transfer_pair": _read_csv(
            transfer_screen_dir / "nanoclustering_g4_8_first_pass_014_role_pattern_transfer_pair_rows.csv"
        ),
        "transfer_route": _read_csv(
            transfer_screen_dir / "nanoclustering_g4_8_first_pass_014_role_pattern_transfer_route_rows.csv"
        ),
        "continuity_016_route": _read_csv(
            continuity_016_dir / "nanoclustering_g4_8_first_pass_016_continuity_block_route_rows.csv"
        ),
        "synthesis_pair": _read_csv(
            synthesis_dir / "nanoclustering_g4_8_first_pass_transition_evidence_synthesis_pair_rows.csv"
        ),
    }
    return {
        "paths": {
            "first_pass_dir": first_pass_dir,
            "transfer_screen_dir": transfer_screen_dir,
            "continuity_016_dir": continuity_016_dir,
            "synthesis_dir": synthesis_dir,
        },
        "summaries": summaries,
        "tables": tables,
    }


def _pair_order(context: dict[str, Any]) -> list[str]:
    synthesis_pair = context["tables"]["synthesis_pair"]
    first_pass_pair = context["tables"]["first_pass_pair"]
    transfer_pair = context["tables"]["transfer_pair"]
    ordered = []
    for frame in (synthesis_pair, first_pass_pair, transfer_pair):
        if "local_pair_id" not in frame.columns:
            continue
        for pair_id in frame["local_pair_id"].astype(str).tolist():
            if pair_id not in ordered:
                ordered.append(pair_id)
    return ordered


def _first_row(frame: pd.DataFrame, pair_id: str) -> dict[str, Any]:
    if frame.empty or "local_pair_id" not in frame.columns:
        return {}
    rows = frame[frame["local_pair_id"].astype(str) == pair_id]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _build_pair_features(context: dict[str, Any]) -> pd.DataFrame:
    tables = context["tables"]
    transfer_summary = context["summaries"]["transfer_screen"]
    closed_control_pairs = set(transfer_summary.get("closed_control_analog_pairs", []))
    rare_ready_pairs = set(transfer_summary.get("rare_ready_blocked_analog_pairs", []))
    primary_diagnostic_pairs = set(transfer_summary.get("primary_diagnostic_pairs", []))

    first_pair = tables["first_pass_pair"]
    first_route = tables["first_pass_route"]
    transfer_pair = tables["transfer_pair"]
    transfer_route = tables["transfer_route"]
    continuity_route = tables["continuity_016_route"]
    synthesis_pair = tables["synthesis_pair"]

    rows: list[dict[str, Any]] = []
    for pair_id in _pair_order(context):
        first_pair_row = _first_row(first_pair, pair_id)
        transfer_pair_row = _first_row(transfer_pair, pair_id)
        synthesis_pair_row = _first_row(synthesis_pair, pair_id)
        pair_first_routes = first_route[first_route["local_pair_id"].astype(str) == pair_id]
        pair_transfer_routes = transfer_route[transfer_route["local_pair_id"].astype(str) == pair_id]
        pair_continuity_routes = continuity_route[
            continuity_route["local_pair_id"].astype(str) == pair_id
        ]

        route_n = len(pair_first_routes)
        transfer_route_n = len(pair_transfer_routes)
        continuity_route_n = len(pair_continuity_routes)

        source_start_count = _bool_count(pair_first_routes, "source_start_support_pass")
        post_start_count = _bool_count(pair_first_routes, "post_start_endpoint_continuity_pass")
        target_final_count = _bool_count(pair_first_routes, "target_final_continuity_pass")
        target_exclusive_count = _bool_count(
            pair_first_routes, "target_final_bridge_exclusive_pass"
        )
        direct_retention_count = _bool_count(pair_first_routes, "direct_edge_retention_pass")
        all_positive_count = _bool_count(pair_first_routes, "all_positive_requirements_pass")
        control_leak_count = _bool_count(pair_first_routes, "control_trap_leak_observed")

        role_sequence = (
            pair_transfer_routes["route_role_class_sequence"].fillna("").astype(str)
            if not pair_transfer_routes.empty
            else pd.Series(dtype=str)
        )
        unresolved_counts = (
            pair_transfer_routes["unresolved_intermediate_step_count"].fillna(0).astype(int)
            if "unresolved_intermediate_step_count" in pair_transfer_routes.columns
            else pd.Series(dtype=int)
        )
        target_anchor_counts = (
            pair_transfer_routes["target_anchor_step_count"].fillna(0).astype(int)
            if "target_anchor_step_count" in pair_transfer_routes.columns
            else pd.Series(dtype=int)
        )
        source_like_counts = (
            pair_transfer_routes["source_like_step_count"].fillna(0).astype(int)
            if "source_like_step_count" in pair_transfer_routes.columns
            else pd.Series(dtype=int)
        )
        hidden_known_counts = (
            pair_transfer_routes["hidden_known_intermediate_step_count"].fillna(0).astype(int)
            if "hidden_known_intermediate_step_count" in pair_transfer_routes.columns
            else pd.Series(dtype=int)
        )
        mixed_known_counts = (
            pair_transfer_routes["mixed_known_step_count"].fillna(0).astype(int)
            if "mixed_known_step_count" in pair_transfer_routes.columns
            else pd.Series(dtype=int)
        )

        separated_seq_count = int(
            role_sequence.str.contains("unresolved_pair_separated_bridge_reassignment").sum()
        )
        coassigned_seq_count = int(
            role_sequence.str.contains("unresolved_pair_coassigned_intermediate").sum()
        )
        unresolved_route_count = int((unresolved_counts > 0).sum())
        single_unresolved_route_count = int((unresolved_counts == 1).sum())
        hidden_known_route_count = int((hidden_known_counts > 0).sum())
        mixed_known_route_count = int((mixed_known_counts > 0).sum())
        target_anchor_min = int(target_anchor_counts.min()) if len(target_anchor_counts) else 0
        target_anchor_max = int(target_anchor_counts.max()) if len(target_anchor_counts) else 0
        source_like_min = int(source_like_counts.min()) if len(source_like_counts) else 0
        source_like_max = int(source_like_counts.max()) if len(source_like_counts) else 0

        typed_single_separated_all = (
            transfer_route_n > 0
            and single_unresolved_route_count == transfer_route_n
            and separated_seq_count == transfer_route_n
            and coassigned_seq_count == 0
            and hidden_known_route_count == 0
            and target_anchor_min >= 3
            and source_like_min >= 1
        )
        typed_coassigned_all = (
            transfer_route_n > 0
            and unresolved_route_count == transfer_route_n
            and coassigned_seq_count == transfer_route_n
            and separated_seq_count == 0
        )

        continuity_single_step_count = _bool_count(
            pair_continuity_routes, "single_step_bridge_reassignment_block"
        )
        continuity_source_start_count = _bool_count(
            pair_continuity_routes, "source_start_support_pass"
        )
        continuity_post_start_count = _bool_count(
            pair_continuity_routes, "post_start_endpoint_continuity_pass"
        )
        continuity_target_exclusive_count = _bool_count(
            pair_continuity_routes, "target_final_bridge_exclusive_pass"
        )
        continuity_direct_retention_count = _bool_count(
            pair_continuity_routes, "direct_edge_retention_pass"
        )

        evidence_role = str(
            synthesis_pair_row.get(
                "evidence_role", first_pair_row.get("evidence_role", transfer_pair_row.get("evidence_role", ""))
            )
        )
        validation_stratum = str(
            synthesis_pair_row.get(
                "validation_stratum",
                first_pair_row.get("validation_stratum", transfer_pair_row.get("validation_stratum", "")),
            )
        )
        is_control = evidence_role == "control_false_positive_guard"
        is_boundary_guard = pair_id == BOUNDARY_GUARD_PAIR_ID
        is_reference = pair_id == REFERENCE_PAIR_ID
        is_primary_typed_transient = pair_id == PRIMARY_TYPED_TRANSIENT_PAIR_ID

        if is_reference:
            guard_family = "reference_positive"
        elif is_primary_typed_transient or pair_id in primary_diagnostic_pairs:
            guard_family = "primary_typed_transient_diagnostic"
        elif is_boundary_guard:
            guard_family = "boundary_guard"
        elif pair_id in closed_control_pairs or is_control:
            guard_family = "closed_control"
        elif pair_id in rare_ready_pairs or validation_stratum == "rare_ready":
            guard_family = "rare_ready_blocked"
        else:
            guard_family = "candidate_or_other"

        rows.append(
            {
                "local_pair_id": pair_id,
                "evidence_role": evidence_role,
                "validation_stratum": validation_stratum,
                "guard_family": guard_family,
                "is_reference_pair": is_reference,
                "is_primary_typed_transient_pair": is_primary_typed_transient,
                "is_boundary_guard_pair": is_boundary_guard,
                "is_control_pair": is_control,
                "first_pass_result": first_pair_row.get(
                    "pair_first_pass_result", synthesis_pair_row.get("pair_first_pass_result", "")
                ),
                "route_readout_row_count": route_n,
                "seed_route_result_count": _as_int(
                    first_pair_row.get("seed_route_result_count", route_n)
                ),
                "source_start_pass_count": source_start_count,
                "post_start_continuity_pass_count": post_start_count,
                "target_final_pass_count": target_final_count,
                "target_final_bridge_exclusive_pass_count": target_exclusive_count,
                "direct_edge_retention_pass_count": direct_retention_count,
                "all_positive_requirements_pass_count": all_positive_count,
                "control_trap_leak_count": control_leak_count,
                "source_start_all_routes": route_n > 0 and source_start_count == route_n,
                "post_start_continuity_all_routes": route_n > 0 and post_start_count == route_n,
                "target_final_all_routes": route_n > 0 and target_final_count == route_n,
                "target_final_bridge_exclusive_all_routes": (
                    route_n > 0 and target_exclusive_count == route_n
                ),
                "direct_edge_retention_all_routes": route_n > 0 and direct_retention_count == route_n,
                "all_positive_requirements_all_routes": route_n > 0 and all_positive_count == route_n,
                "has_source_like_signature": _as_bool(
                    transfer_pair_row.get(
                        "has_source_like_signature",
                        synthesis_pair_row.get("has_source_like_signature", False),
                    )
                ),
                "has_target_anchor_signature": _as_bool(
                    transfer_pair_row.get(
                        "has_target_anchor_signature",
                        synthesis_pair_row.get("has_target_anchor_signature", False),
                    )
                ),
                "has_hidden_known_intermediate_signature": _as_bool(
                    transfer_pair_row.get(
                        "has_hidden_known_intermediate_signature",
                        synthesis_pair_row.get("has_hidden_known_intermediate_signature", False),
                    )
                ),
                "has_unresolved_pair_coassigned_signature": _as_bool(
                    transfer_pair_row.get(
                        "has_unresolved_pair_coassigned_signature",
                        synthesis_pair_row.get("has_unresolved_pair_coassigned_signature", False),
                    )
                ),
                "has_unresolved_pair_separated_signature": _as_bool(
                    transfer_pair_row.get(
                        "has_unresolved_pair_separated_signature",
                        synthesis_pair_row.get("has_unresolved_pair_separated_signature", False),
                    )
                ),
                "has_known_mixed_signature": _as_bool(
                    transfer_pair_row.get("has_known_mixed_signature", False)
                ),
                "has_transition_intermediate_analog": _as_bool(
                    transfer_pair_row.get("has_transition_intermediate_analog", False)
                ),
                "role_analog_feature_count": _as_int(
                    transfer_pair_row.get(
                        "role_analog_feature_count",
                        synthesis_pair_row.get("role_analog_feature_count", 0),
                    )
                ),
                "source_target_signature_collapse_count": _as_int(
                    transfer_pair_row.get("source_target_signature_collapse_count", 0)
                ),
                "guard_anchor_collapse_count": _as_int(
                    transfer_pair_row.get("guard_anchor_collapse_count", 0)
                ),
                "intermediate_unknown_route_count": _as_int(
                    transfer_pair_row.get("intermediate_unknown_route_count", 0)
                ),
                "transfer_route_row_count": transfer_route_n,
                "transfer_route_unresolved_count": unresolved_route_count,
                "transfer_route_single_unresolved_count": single_unresolved_route_count,
                "transfer_route_separated_sequence_count": separated_seq_count,
                "transfer_route_coassigned_sequence_count": coassigned_seq_count,
                "transfer_route_hidden_known_count": hidden_known_route_count,
                "transfer_route_mixed_known_count": mixed_known_route_count,
                "transfer_route_target_anchor_step_min": target_anchor_min,
                "transfer_route_target_anchor_step_max": target_anchor_max,
                "transfer_route_source_like_step_min": source_like_min,
                "transfer_route_source_like_step_max": source_like_max,
                "typed_single_separated_transient_all_routes": typed_single_separated_all,
                "typed_coassigned_transient_all_routes": typed_coassigned_all,
                "continuity_016_route_row_count": continuity_route_n,
                "continuity_016_single_step_bridge_reassignment_count": continuity_single_step_count,
                "continuity_016_single_step_bridge_reassignment_all_routes": (
                    continuity_route_n > 0 and continuity_single_step_count == continuity_route_n
                ),
                "continuity_016_source_start_all_routes": (
                    continuity_route_n > 0 and continuity_source_start_count == continuity_route_n
                ),
                "continuity_016_post_start_continuity_pass_count": continuity_post_start_count,
                "continuity_016_target_exclusive_all_routes": (
                    continuity_route_n > 0 and continuity_target_exclusive_count == continuity_route_n
                ),
                "continuity_016_direct_retention_all_routes": (
                    continuity_route_n > 0 and continuity_direct_retention_count == continuity_route_n
                ),
                "method_claim_allowed_after_predicate_screen": False,
                "quality_cost_claim_allowed_after_predicate_screen": False,
                "wall_generality_claim_allowed_after_predicate_screen": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


PREDICATE_DEFINITIONS = [
    {
        "predicate_id": "P0_strict_all_positive_baseline",
        "predicate_family": "guarded_baseline",
        "definition_question": "Which pair survives the original strict first-pass requirements?",
        "acceptance_rule": (
            "all routes pass source-start, post-start continuity, target-final, "
            "exclusive target, direct retention, and all-positive requirements"
        ),
        "expected_role": "reference_positive_only",
        "valid_definition_candidate": True,
        "expected_to_leak_guards": False,
    },
    {
        "predicate_id": "P1_guarded_single_step_separated_transient_candidate",
        "predicate_family": "guarded_typed_transient_candidate",
        "definition_question": (
            "Can a strict-ready pair with target-final success and exactly one recurrent "
            "separated bridge-reassignment transient be held as a definition candidate?"
        ),
        "acceptance_rule": (
            "strict-ready, non-control, non-boundary pair; source-start, target-final, "
            "exclusive target, and direct retention all pass; post-start continuity and "
            "all-positive fail; all transfer routes contain exactly one separated "
            "bridge-reassignment transient; no coassigned or hidden-known route transient; "
            "no source-target or guard-anchor collapse"
        ),
        "expected_role": "typed_transient_candidate_only",
        "valid_definition_candidate": True,
        "expected_to_leak_guards": False,
    },
    {
        "predicate_id": "P2_endpoint_only_negative_control",
        "predicate_family": "invalid_broadened_endpoint_only",
        "definition_question": "What happens if continuity is ignored and only endpoint success is used?",
        "acceptance_rule": (
            "source-start, target-final, and direct retention all pass; post-start continuity, "
            "exclusive target closure, and typed-transient shape are ignored"
        ),
        "expected_role": "negative_demonstration_only",
        "valid_definition_candidate": False,
        "expected_to_leak_guards": True,
    },
    {
        "predicate_id": "P3_role_analog_only_negative_control",
        "predicate_family": "invalid_broadened_role_analog_only",
        "definition_question": "What happens if role analog presence alone defines a pathway candidate?",
        "acceptance_rule": (
            "source-like signature, target-anchor signature, and any transition-intermediate "
            "analog are present; route-level endpoint and guard closure are ignored"
        ),
        "expected_role": "negative_demonstration_only",
        "valid_definition_candidate": False,
        "expected_to_leak_guards": True,
    },
]


def _accepted_by_predicate(pair: dict[str, Any], predicate_id: str) -> bool:
    if predicate_id == "P0_strict_all_positive_baseline":
        return bool(
            pair["all_positive_requirements_all_routes"]
            and pair["source_start_all_routes"]
            and pair["post_start_continuity_all_routes"]
            and pair["target_final_all_routes"]
            and pair["target_final_bridge_exclusive_all_routes"]
            and pair["direct_edge_retention_all_routes"]
            and pair["control_trap_leak_count"] == 0
        )
    if predicate_id == "P1_guarded_single_step_separated_transient_candidate":
        return bool(
            pair["validation_stratum"] == "strict_ready"
            and not pair["is_control_pair"]
            and not pair["is_boundary_guard_pair"]
            and not pair["is_reference_pair"]
            and pair["source_start_all_routes"]
            and pair["target_final_all_routes"]
            and pair["target_final_bridge_exclusive_all_routes"]
            and pair["direct_edge_retention_all_routes"]
            and not pair["post_start_continuity_all_routes"]
            and pair["all_positive_requirements_pass_count"] == 0
            and pair["has_source_like_signature"]
            and pair["has_target_anchor_signature"]
            and pair["has_unresolved_pair_separated_signature"]
            and pair["typed_single_separated_transient_all_routes"]
            and pair["source_target_signature_collapse_count"] == 0
            and pair["guard_anchor_collapse_count"] == 0
            and pair["continuity_016_single_step_bridge_reassignment_all_routes"]
        )
    if predicate_id == "P2_endpoint_only_negative_control":
        return bool(
            pair["source_start_all_routes"]
            and pair["target_final_all_routes"]
            and pair["direct_edge_retention_all_routes"]
        )
    if predicate_id == "P3_role_analog_only_negative_control":
        return bool(
            pair["has_source_like_signature"]
            and pair["has_target_anchor_signature"]
            and pair["has_transition_intermediate_analog"]
        )
    raise KeyError(predicate_id)


def _guard_violation_type(pair: dict[str, Any]) -> str:
    if pair["is_control_pair"]:
        return "control_guard_leak"
    if pair["is_boundary_guard_pair"]:
        return "boundary_guard_leak"
    if pair["validation_stratum"] != "strict_ready" and not pair["is_reference_pair"]:
        return "non_strict_or_rare_ready_leak"
    return "none"


def _evaluate_predicates(pair_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows: list[dict[str, Any]] = []
    for predicate in PREDICATE_DEFINITIONS:
        predicate_id = predicate["predicate_id"]
        for pair in pair_features.to_dict(orient="records"):
            accepted = _accepted_by_predicate(pair, predicate_id)
            guard_violation_type = _guard_violation_type(pair) if accepted else "none"
            if not accepted:
                claim_level = "rejected"
            elif predicate_id == "P0_strict_all_positive_baseline":
                claim_level = "reference_positive_only"
            elif predicate_id == "P1_guarded_single_step_separated_transient_candidate":
                claim_level = "typed_transient_candidate_only"
            else:
                claim_level = "invalid_negative_demonstration_only"
            pair_rows.append(
                {
                    "predicate_id": predicate_id,
                    "predicate_family": predicate["predicate_family"],
                    "local_pair_id": pair["local_pair_id"],
                    "accepted_by_predicate": accepted,
                    "claim_level_if_accepted": claim_level,
                    "guard_family": pair["guard_family"],
                    "guard_violation_type": guard_violation_type,
                    "is_guard_violation": guard_violation_type != "none",
                    "is_reference_pair": pair["is_reference_pair"],
                    "is_primary_typed_transient_pair": pair["is_primary_typed_transient_pair"],
                    "is_boundary_guard_pair": pair["is_boundary_guard_pair"],
                    "is_control_pair": pair["is_control_pair"],
                    "validation_stratum": pair["validation_stratum"],
                    "source_start_all_routes": pair["source_start_all_routes"],
                    "post_start_continuity_all_routes": pair[
                        "post_start_continuity_all_routes"
                    ],
                    "target_final_bridge_exclusive_all_routes": pair[
                        "target_final_bridge_exclusive_all_routes"
                    ],
                    "direct_edge_retention_all_routes": pair["direct_edge_retention_all_routes"],
                    "typed_single_separated_transient_all_routes": pair[
                        "typed_single_separated_transient_all_routes"
                    ],
                    "has_transition_intermediate_analog": pair[
                        "has_transition_intermediate_analog"
                    ],
                    "source_target_signature_collapse_count": pair[
                        "source_target_signature_collapse_count"
                    ],
                    "guard_anchor_collapse_count": pair["guard_anchor_collapse_count"],
                    "method_claim_allowed_after_predicate_screen": False,
                    "quality_cost_claim_allowed_after_predicate_screen": False,
                    "wall_generality_claim_allowed_after_predicate_screen": False,
                    "route_execution_status": ROUTE_EXECUTION_STATUS,
                    "wall_promotion_status": WALL_PROMOTION_STATUS,
                    "method_status": METHOD_STATUS,
                    "run_status": RUN_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
    pair_predicate_rows = pd.DataFrame(pair_rows)

    predicate_rows: list[dict[str, Any]] = []
    for predicate in PREDICATE_DEFINITIONS:
        predicate_id = predicate["predicate_id"]
        rows = pair_predicate_rows[pair_predicate_rows["predicate_id"] == predicate_id]
        accepted = rows[rows["accepted_by_predicate"]]
        accepted_pairs = accepted["local_pair_id"].astype(str).tolist()
        boundary_leaks = accepted[accepted["is_boundary_guard_pair"]][
            "local_pair_id"
        ].astype(str).tolist()
        control_leaks = accepted[accepted["is_control_pair"]][
            "local_pair_id"
        ].astype(str).tolist()
        rare_leaks = accepted[
            (accepted["validation_stratum"] != "strict_ready")
            & ~accepted["is_reference_pair"]
            & ~accepted["is_control_pair"]
        ]["local_pair_id"].astype(str).tolist()
        guard_leaks = boundary_leaks + control_leaks + rare_leaks

        if predicate_id == "P0_strict_all_positive_baseline":
            validation_status = (
                "valid_strict_reference_baseline"
                if accepted_pairs == [REFERENCE_PAIR_ID] and not guard_leaks
                else "invalid_strict_baseline_leak_or_missing_reference"
            )
            next_action = "keep_as_baseline_positive_reference_only"
        elif predicate_id == "P1_guarded_single_step_separated_transient_candidate":
            validation_status = (
                "candidate_valid_for_next_gate"
                if accepted_pairs == [PRIMARY_TYPED_TRANSIENT_PAIR_ID] and not guard_leaks
                else "invalid_typed_transient_candidate_leak_or_missing_016"
            )
            next_action = "validate semantics before treating as pathway evidence"
        else:
            validation_status = (
                "invalid_as_expected_guard_leaks_observed"
                if bool(guard_leaks)
                else "unexpectedly_no_guard_leak"
            )
            next_action = "do_not_use_as_definition_rule"

        predicate_rows.append(
            {
                "predicate_id": predicate_id,
                "predicate_family": predicate["predicate_family"],
                "definition_question": predicate["definition_question"],
                "acceptance_rule": predicate["acceptance_rule"],
                "expected_role": predicate["expected_role"],
                "valid_definition_candidate": bool(predicate["valid_definition_candidate"]),
                "expected_to_leak_guards": bool(predicate["expected_to_leak_guards"]),
                "accepted_pair_count": int(len(accepted_pairs)),
                "accepted_pair_ids": _list_json(accepted_pairs),
                "boundary_guard_leak_pair_ids": _list_json(boundary_leaks),
                "control_guard_leak_pair_ids": _list_json(control_leaks),
                "rare_or_non_strict_leak_pair_ids": _list_json(rare_leaks),
                "guard_leak_count": int(len(guard_leaks)),
                "reference_pair_accepted": REFERENCE_PAIR_ID in accepted_pairs,
                "primary_typed_transient_pair_accepted": (
                    PRIMARY_TYPED_TRANSIENT_PAIR_ID in accepted_pairs
                ),
                "validation_status": validation_status,
                "next_action": next_action,
                "method_claim_allowed_after_predicate_screen": False,
                "quality_cost_claim_allowed_after_predicate_screen": False,
                "wall_generality_claim_allowed_after_predicate_screen": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pair_predicate_rows, pd.DataFrame(predicate_rows)


def _definition_rows(pair_features: pd.DataFrame, predicate_rows: pd.DataFrame) -> pd.DataFrame:
    feature = pair_features.set_index("local_pair_id").to_dict(orient="index")
    p_status = predicate_rows.set_index("predicate_id").to_dict(orient="index")
    rows = [
        {
            "definition_id": "D01_endpoint_success_is_necessary_not_sufficient",
            "definition_component": "endpoint identity",
            "data_observation": (
                "014, 016, 005, 007, 008, 002, 003, 013, and 022 all satisfy source-start, "
                "target-final, and direct-retention counts under "
                "the endpoint-only predicate."
            ),
            "predicate_implication": (
                "Endpoint evidence remains necessary, but accepting endpoint success "
                "alone leaks boundary, rare-ready, and control rows."
            ),
            "decision_status": "reject_as_standalone_rule",
            "next_gate": "endpoint checks must be paired with typed path shape and guard closure",
        },
        {
            "definition_id": "D02_strict_continuity_baseline",
            "definition_component": "post-start continuity",
            "data_observation": (
                f"{REFERENCE_PAIR_ID} is the only pair accepted by "
                "P0_strict_all_positive_baseline."
            ),
            "predicate_implication": (
                "The strict rule is clean but may be too narrow for typed transient "
                "intermediates such as 016."
            ),
            "decision_status": p_status["P0_strict_all_positive_baseline"]["validation_status"],
            "next_gate": "retain as reference baseline, not as final basin definition",
        },
        {
            "definition_id": "D03_guarded_typed_transient_exception",
            "definition_component": "typed transient semantics",
            "data_observation": (
                f"{PRIMARY_TYPED_TRANSIENT_PAIR_ID} has source-start, target-final, "
                "exclusive-target, and direct-retention success on all audited routes, "
                "but all routes fail post-start continuity through one separated "
                "bridge-reassignment transient."
            ),
            "predicate_implication": (
                "A guarded single-step separated transient can be held as a candidate "
                "pathway object without promoting it to positive wall evidence."
            ),
            "decision_status": p_status[
                "P1_guarded_single_step_separated_transient_candidate"
            ]["validation_status"],
            "next_gate": "validate whether the transient is a meaningful basin-wall crossing or a local route artifact",
        },
        {
            "definition_id": "D04_boundary_guard_closure",
            "definition_component": "boundary guard",
            "data_observation": (
                f"{BOUNDARY_GUARD_PAIR_ID} has 24 ready-like first-pass routes but "
                "lacks a target-anchor signature and has source-target signature collapse."
            ),
            "predicate_implication": (
                "Any broadened definition must still reject boundary-like partial "
                "ready cases."
            ),
            "decision_status": "guard_required",
            "next_gate": "keep 005 as a mandatory rejection row for any candidate predicate",
        },
        {
            "definition_id": "D05_role_analog_is_diagnostic_not_decisive",
            "definition_component": "role-pattern analogs",
            "data_observation": (
                "The role-analog-only negative predicate accepts closed controls and "
                "rare-ready analogs."
            ),
            "predicate_implication": (
                "Role analogs can nominate diagnostic follow-up, but cannot define "
                "basin membership or pathways by themselves."
            ),
            "decision_status": p_status["P3_role_analog_only_negative_control"][
                "validation_status"
            ],
            "next_gate": "require route-level and guard-level closure before localization",
        },
    ]
    if PRIMARY_TYPED_TRANSIENT_PAIR_ID in feature:
        rows.append(
            {
                "definition_id": "D06_016_current_status",
                "definition_component": "primary diagnostic pair",
                "data_observation": (
                    f"{PRIMARY_TYPED_TRANSIENT_PAIR_ID} is accepted only by the guarded "
                    "typed-transient predicate, not by the strict baseline."
                ),
                "predicate_implication": (
                    "016 is a definition candidate for the second-step semantics, not "
                    "a positive transfer or generality claim."
                ),
                "decision_status": "diagnostic_candidate_only",
                "next_gate": "inspect the transient mechanism before new broad trace execution",
            }
        )
    return pd.DataFrame(rows)


def _build_gate_matrix(
    context: dict[str, Any],
    pair_features: pd.DataFrame,
    predicate_rows: pd.DataFrame,
    pair_predicate_rows: pd.DataFrame,
) -> pd.DataFrame:
    summaries = context["summaries"]
    upstream_failed = {
        name: summary.get("failed_gates", [])
        for name, summary in summaries.items()
        if name in {"synthesis", "transfer_screen", "continuity_016", "first_pass"}
    }
    predicate_ids = set(predicate_rows["predicate_id"].astype(str).tolist())
    p_status = predicate_rows.set_index("predicate_id").to_dict(orient="index")

    strict_pairs = json.loads(p_status["P0_strict_all_positive_baseline"]["accepted_pair_ids"])
    typed_pairs = json.loads(
        p_status["P1_guarded_single_step_separated_transient_candidate"][
            "accepted_pair_ids"
        ]
    )
    endpoint_negative = p_status["P2_endpoint_only_negative_control"]
    role_negative = p_status["P3_role_analog_only_negative_control"]
    endpoint_negative_pairs = json.loads(endpoint_negative["accepted_pair_ids"])
    role_negative_pairs = json.loads(role_negative["accepted_pair_ids"])
    valid_claim_flags = (
        not pair_features["method_claim_allowed_after_predicate_screen"].map(_as_bool).any()
        and not pair_features["quality_cost_claim_allowed_after_predicate_screen"].map(_as_bool).any()
        and not pair_features["wall_generality_claim_allowed_after_predicate_screen"].map(
            _as_bool
        ).any()
        and not pair_predicate_rows["method_claim_allowed_after_predicate_screen"].map(
            _as_bool
        ).any()
        and not predicate_rows["method_claim_allowed_after_predicate_screen"].map(_as_bool).any()
    )

    gates = [
        _gate_row(
            "G1",
            "Did upstream read-only audits pass before predicate screening?",
            upstream_failed,
            "all referenced upstream failed_gates lists are empty",
            all(not failed for failed in upstream_failed.values()),
        ),
        _gate_row(
            "G2",
            "Is the predicate family explicit and comparative?",
            sorted(predicate_ids),
            "include strict baseline, guarded typed-transient candidate, endpoint-only negative, and role-analog negative",
            {
                "P0_strict_all_positive_baseline",
                "P1_guarded_single_step_separated_transient_candidate",
                "P2_endpoint_only_negative_control",
                "P3_role_analog_only_negative_control",
            }.issubset(predicate_ids),
        ),
        _gate_row(
            "G3",
            "Does the strict baseline preserve only the reference scaffold?",
            strict_pairs,
            f"accepted pairs exactly [{REFERENCE_PAIR_ID}]",
            strict_pairs == [REFERENCE_PAIR_ID],
        ),
        _gate_row(
            "G4",
            "Does the guarded typed-transient candidate isolate 016 without guard leaks?",
            {
                "accepted_pairs": typed_pairs,
                "guard_leak_count": p_status[
                    "P1_guarded_single_step_separated_transient_candidate"
                ]["guard_leak_count"],
            },
            f"accepted pairs exactly [{PRIMARY_TYPED_TRANSIENT_PAIR_ID}] and zero guard leaks",
            typed_pairs == [PRIMARY_TYPED_TRANSIENT_PAIR_ID]
            and int(
                p_status["P1_guarded_single_step_separated_transient_candidate"][
                    "guard_leak_count"
                ]
            )
            == 0,
        ),
        _gate_row(
            "G5",
            "Does endpoint-only broadening visibly fail the guards?",
            {
                "accepted_pairs": endpoint_negative_pairs,
                "guard_leak_count": endpoint_negative["guard_leak_count"],
            },
            "endpoint-only negative predicate must leak at least one guard row",
            int(endpoint_negative["guard_leak_count"]) > 0,
        ),
        _gate_row(
            "G6",
            "Does role-analog-only broadening visibly fail the guards?",
            {
                "accepted_pairs": role_negative_pairs,
                "guard_leak_count": role_negative["guard_leak_count"],
            },
            "role-analog-only negative predicate must leak at least one guard row",
            int(role_negative["guard_leak_count"]) > 0,
        ),
        _gate_row(
            "G7",
            "Are claim boundaries preserved?",
            {
                "method_quality_wall_flags_all_false": valid_claim_flags,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
            },
            "no method, quality/cost, wall-generality, wall-promotion, or new-route claim",
            valid_claim_flags,
        ),
    ]
    return pd.DataFrame(gates)


def _write_report(
    output_dir: Path,
    pair_features: pd.DataFrame,
    pair_predicate_rows: pd.DataFrame,
    predicate_rows: pd.DataFrame,
    definition_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    accepted_rows = pair_predicate_rows[pair_predicate_rows["accepted_by_predicate"]]
    lines = [
        "# NanoClustering G4.8 First-Pass Typed-Transient Predicate Screen",
        "",
        "## Claim Boundary",
        "",
        CLAIM_BOUNDARY,
        "",
        "## Summary",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {
                        "pair_feature_row_count": summary["pair_feature_row_count"],
                        "pair_predicate_row_count": summary["pair_predicate_row_count"],
                        "predicate_row_count": summary["predicate_row_count"],
                        "definition_row_count": summary["definition_row_count"],
                        "failed_gates": _list_json(summary["failed_gates"]),
                        "recommended_next_gate": summary["recommended_next_gate"],
                    }
                ]
            ),
            [
                "pair_feature_row_count",
                "pair_predicate_row_count",
                "predicate_row_count",
                "definition_row_count",
                "failed_gates",
                "recommended_next_gate",
            ],
        ),
        "",
        "## Pair Features",
        "",
        _markdown_table(
            pair_features,
            [
                "local_pair_id",
                "guard_family",
                "validation_stratum",
                "source_start_pass_count",
                "post_start_continuity_pass_count",
                "target_final_bridge_exclusive_pass_count",
                "all_positive_requirements_pass_count",
                "typed_single_separated_transient_all_routes",
                "source_target_signature_collapse_count",
                "guard_anchor_collapse_count",
            ],
        ),
        "",
        "## Predicate Outcomes",
        "",
        _markdown_table(
            predicate_rows,
            [
                "predicate_id",
                "accepted_pair_ids",
                "guard_leak_count",
                "boundary_guard_leak_pair_ids",
                "control_guard_leak_pair_ids",
                "rare_or_non_strict_leak_pair_ids",
                "validation_status",
                "next_action",
            ],
        ),
        "",
        "## Accepted Pair Rows",
        "",
        _markdown_table(
            accepted_rows,
            [
                "predicate_id",
                "local_pair_id",
                "claim_level_if_accepted",
                "guard_family",
                "guard_violation_type",
                "typed_single_separated_transient_all_routes",
                "post_start_continuity_all_routes",
            ],
            max_rows=80,
        ),
        "",
        "## Definition Decisions",
        "",
        _markdown_table(
            definition_rows,
            [
                "definition_id",
                "definition_component",
                "decision_status",
                "next_gate",
            ],
        ),
        "",
        "## Gates",
        "",
        _markdown_table(
            gate_matrix,
            ["gate_id", "question", "observed", "minimum_or_rule", "gate_status"],
        ),
        "",
        "## Interpretation",
        "",
        "- P0 remains the clean strict baseline and only recovers local_pair_014.",
        "- P1 isolates local_pair_016 as a guarded typed-transient candidate, not as positive wall evidence.",
        "- P2 and P3 fail deliberately: they admit boundary, control, or rare-ready rows and therefore explain why endpoint-only or role-analog-only definitions are too broad.",
        "- The next step is semantic validation of the typed transient, not a new broad localization sweep.",
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    context = _load_context(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_features = _build_pair_features(context)
    pair_predicate_rows, predicate_rows = _evaluate_predicates(pair_features)
    definition_rows = _definition_rows(pair_features, predicate_rows)
    gate_matrix = _build_gate_matrix(
        context, pair_features, predicate_rows, pair_predicate_rows
    )

    failed_gates = gate_matrix.loc[
        gate_matrix["gate_status"].astype(str) != "pass", "gate_id"
    ].astype(str).tolist()
    summary = {
        "schema": "nanoclustering_g4_8_first_pass_typed_transient_predicate_summary.v1",
        "status": RUN_STATUS,
        "output_dir": str(output_dir.resolve()),
        "source_dirs": {
            key: str(path.resolve()) for key, path in context["paths"].items()
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "reference_pair": REFERENCE_PAIR_ID,
        "primary_typed_transient_pair": PRIMARY_TYPED_TRANSIENT_PAIR_ID,
        "boundary_guard_pair": BOUNDARY_GUARD_PAIR_ID,
        "pair_feature_row_count": int(len(pair_features)),
        "pair_predicate_row_count": int(len(pair_predicate_rows)),
        "predicate_row_count": int(len(predicate_rows)),
        "definition_row_count": int(len(definition_rows)),
        "gate_status_counts": {
            str(k): int(v)
            for k, v in gate_matrix["gate_status"].value_counts().to_dict().items()
        },
        "failed_gates": failed_gates,
        "predicate_validation_status": {
            str(row["predicate_id"]): str(row["validation_status"])
            for row in predicate_rows.to_dict(orient="records")
        },
        "accepted_pairs_by_predicate": {
            str(row["predicate_id"]): json.loads(str(row["accepted_pair_ids"]))
            for row in predicate_rows.to_dict(orient="records")
        },
        "guard_leaks_by_predicate": {
            str(row["predicate_id"]): {
                "boundary": json.loads(str(row["boundary_guard_leak_pair_ids"])),
                "control": json.loads(str(row["control_guard_leak_pair_ids"])),
                "rare_or_non_strict": json.loads(
                    str(row["rare_or_non_strict_leak_pair_ids"])
                ),
            }
            for row in predicate_rows.to_dict(orient="records")
        },
        "definition_decision_focus": (
            "Treat 016 as a guarded typed-transient definition candidate only; "
            "reject endpoint-only and role-analog-only broadenings because they leak guards."
        ),
        "recommended_next_gate": (
            "Validate the semantic meaning of the 016 single-step separated bridge-reassignment "
            "transient before any new broad trace execution or wall promotion."
        ),
    }
    config = {
        "first_pass_dir": str(Path(args.first_pass_dir).resolve()),
        "transfer_screen_dir": str(Path(args.transfer_screen_dir).resolve()),
        "continuity_016_dir": str(Path(args.continuity_016_dir).resolve()),
        "synthesis_dir": str(Path(args.synthesis_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "claim_boundary": CLAIM_BOUNDARY,
        "predicate_definitions": PREDICATE_DEFINITIONS,
    }

    _write_csv(pair_features, output_dir / PAIR_FEATURE_ROWS_CSV)
    _write_csv(pair_predicate_rows, output_dir / PAIR_PREDICATE_ROWS_CSV)
    _write_csv(predicate_rows, output_dir / PREDICATE_ROWS_CSV)
    _write_csv(definition_rows, output_dir / DEFINITION_ROWS_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_report(
        output_dir,
        pair_features,
        pair_predicate_rows,
        predicate_rows,
        definition_rows,
        gate_matrix,
        summary,
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Screen typed-transient predicate choices for basin/pathway definition."
    )
    parser.add_argument("--first-pass-dir", type=Path, default=DEFAULT_FIRST_PASS_DIR)
    parser.add_argument("--transfer-screen-dir", type=Path, default=DEFAULT_TRANSFER_SCREEN_DIR)
    parser.add_argument("--continuity-016-dir", type=Path, default=DEFAULT_CONTINUITY_016_DIR)
    parser.add_argument("--synthesis-dir", type=Path, default=DEFAULT_SYNTHESIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
