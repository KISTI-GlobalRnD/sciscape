#!/usr/bin/env python3
"""Materialize basin-state assignment evidence for six route-scoreable pairs.

This read-only audit attaches endpoint-anchor, endpoint-object, continuity,
and primitive wall evidence to the current route-state morphology bridge. It
separates local basin-state assignment from pathway or method promotion: route
anchors can create candidates, but accepted basin-state pairs require
object-level endpoint identity evidence.
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


DEFAULT_BRIDGE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_basin_state_route_morphology_bridge_gamma1e5_20260606"
)
DEFAULT_ROUTE_TRACE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_gamma1e5_20260605"
)
DEFAULT_ROUTE_NEGATIVE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_route_negative_explanation_audit_gamma1e5_20260605"
)
DEFAULT_CONTINUITY_016_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_continuity_block_audit_gamma1e5_20260605"
)
DEFAULT_ENDPOINT_OBJECT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_symmetric_endpoint_objects_audit_gamma1e5_20260604"
)
DEFAULT_WALL_EVIDENCE_014_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_014_wall_evidence_audit_gamma1e5_20260604"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_basin_state_assignment_surface_gamma1e5_20260606"
)

BRIDGE_PAIR_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_basin_state_route_morphology_bridge_pair_rows.csv"
)
BRIDGE_GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_basin_state_route_morphology_bridge_gate_matrix.csv"
)
BRIDGE_SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_basin_state_route_morphology_bridge_summary.json"
)
ROUTE_TRACE_ROUTE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_route_rows.csv"
)
ROUTE_NEGATIVE_PAIR_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_route_negative_explanation_pair_rows.csv"
)
ROUTE_NEGATIVE_SUBSTRATE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_route_negative_explanation_substrate_rows.csv"
)
CONTINUITY_016_ROUTE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_continuity_block_route_rows.csv"
)
CONTINUITY_016_SUMMARY_JSON = "nanoclustering_g4_8_first_pass_016_continuity_block_summary.json"
ENDPOINT_OBJECT_PAIR_SUMMARY_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_symmetric_endpoint_object_pair_summary_rows.csv"
)
ENDPOINT_OBJECT_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_symmetric_endpoint_object_rows.csv"
)
ENDPOINT_OBJECT_SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_symmetric_endpoint_object_summary.json"
)
WALL_EVIDENCE_PAIR_ROWS_CSV = "nanoclustering_g4_8_first_pass_014_wall_evidence_pair_rows.csv"
WALL_EVIDENCE_BOUNDARY_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_014_wall_evidence_boundary_guard_rows.csv"
)
WALL_EVIDENCE_SUMMARY_JSON = "nanoclustering_g4_8_first_pass_014_wall_evidence_summary.json"

PAIR_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_basin_state_assignment_surface_pair_rows.csv"
)
EVIDENCE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_basin_state_assignment_surface_evidence_rows.csv"
)
CLASS_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_basin_state_assignment_surface_class_rows.csv"
)
REQUIREMENT_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_basin_state_assignment_surface_requirement_rows.csv"
)
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_basin_state_assignment_surface_decision_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_basin_state_assignment_surface_gate_matrix.csv"
)
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_basin_state_assignment_surface_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_basin_state_assignment_surface_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_basin_state_assignment_surface_report.md"

EXPECTED_ROUTE_SCOREABLE_PAIR_IDS = (
    "local_pair_016",
    "local_pair_014",
    "local_pair_009",
    "local_pair_012",
    "local_pair_020",
    "local_pair_005",
)
POSITIVE_ROUTE_MORPHOLOGY_CLASS = "stable_finite_single_side_plateau_reference"
NEGATIVE_ROUTE_MORPHOLOGY_CLASSES = {
    "abrupt_source_target_switch_negative",
    "fragmented_or_point_single_side_negative",
}
BOUNDARY_ROUTE_MORPHOLOGY_CLASS = "boundary_or_endpoint_surface_control"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_basin_state_assignment_surface"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_basin_state_assignment_surface"
WALL_PROMOTION_STATUS = "not_promoted_assignment_surface_only"
METHOD_STATUS = "basin_state_assignment_surface_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass basin-state assignment surface audit only; "
    "reads existing bridge, route-trace, route-negative, 016 continuity, "
    "014/005 endpoint-object, and 014 wall-evidence artifacts. It may assign "
    "local basin-state candidate status, but it does not execute Leiden, "
    "promote pathway labels, promote a general wall, replay full NanoClustering, "
    "evaluate quality/cost value, or claim method success."
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


def _safe_str(value: Any, default: str = "") -> str:
    if pd.isna(value):
        return default
    return str(value)


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
        "gate_status": "pass" if passed else "fail",
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


def _index_by_pair(frame: pd.DataFrame) -> dict[str, pd.Series]:
    if frame.empty or "local_pair_id" not in frame.columns:
        return {}
    return {str(row["local_pair_id"]): row for _, row in frame.iterrows()}


def _route_anchor_aggregates(route_rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if route_rows.empty:
        return rows
    grouped = route_rows.groupby("local_pair_id", dropna=False)
    for pair_id, group in grouped:
        route_count = int(len(group))
        rows[str(pair_id)] = {
            "route_anchor_source": "mechanism_generalization_route_trace",
            "route_anchor_route_count": route_count,
            "source_anchor_pass_count": int(group["source_family_start"].map(_as_bool).sum()),
            "target_anchor_pass_count": int(group["final_target_like"].map(_as_bool).sum()),
            "target_expected_anchor_pass_count": int(
                group["final_matches_expected_anchor"].map(_as_bool).sum()
            ),
            "continuity_pass_count": None,
            "unknown_endpoint_step_count": None,
            "route_anchor_note": "fixed-predicate route trace source/start and final target anchor readout",
        }
    return rows


def _continuity_016_aggregate(route_rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if route_rows.empty:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    grouped = route_rows.groupby("local_pair_id", dropna=False)
    for pair_id, group in grouped:
        route_count = int(len(group))
        rows[str(pair_id)] = {
            "route_anchor_source": "016_continuity_block_audit",
            "route_anchor_route_count": route_count,
            "source_anchor_pass_count": int(
                group["source_start_support_pass"].map(_as_bool).sum()
            ),
            "target_anchor_pass_count": int(
                group["target_final_bridge_exclusive_pass"].map(_as_bool).sum()
            ),
            "target_expected_anchor_pass_count": int(
                group["target_final_bridge_exclusive_pass"].map(_as_bool).sum()
            ),
            "continuity_pass_count": int(
                group["post_start_endpoint_continuity_pass"].map(_as_bool).sum()
            ),
            "unknown_endpoint_step_count": int(group["unknown_endpoint_step_count"].sum()),
            "route_anchor_note": (
                "016 continuity audit source/target anchors are present, but "
                "post-start endpoint continuity is blocked by a typed transient"
            ),
        }
    return rows


def _object_evidence_by_pair(pair_summary_rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for pair_id, row in _index_by_pair(pair_summary_rows).items():
        audit_class = _safe_str(row.get("object_audit_class"))
        route_count = _as_int(row.get("route_count"))
        clean_count = _as_int(row.get("clean_relation_count"))
        collapse_count = _as_int(row.get("source_target_collapse_relation_count"))
        exclusive_target_count = _as_int(row.get("exclusive_target_object_count"))
        if (
            audit_class == "clean_symmetric_endpoint_object_candidate"
            and route_count > 0
            and clean_count == route_count
            and collapse_count == 0
            and exclusive_target_count >= 1
        ):
            evidence_status = "accepted_clean_local_endpoint_object_pair"
            source_assignment = "accepted_local_object_source_basin_candidate"
            target_assignment = "accepted_local_object_target_basin_candidate"
            accepted_object_pair = True
        elif audit_class == "partial_boundary_source_target_collapse":
            evidence_status = "rejected_boundary_source_target_collapse"
            source_assignment = "rejected_boundary_source_basin_candidate"
            target_assignment = "rejected_boundary_target_basin_candidate"
            accepted_object_pair = False
        else:
            evidence_status = "object_evidence_present_but_not_accepted"
            source_assignment = "object_source_candidate_not_accepted"
            target_assignment = "object_target_candidate_not_accepted"
            accepted_object_pair = False
        rows[pair_id] = {
            "object_evidence_status": evidence_status,
            "object_audit_class": audit_class,
            "object_route_count": route_count,
            "source_object_count": _as_int(row.get("source_object_count")),
            "final_object_count": _as_int(row.get("final_object_count")),
            "exclusive_target_object_count": exclusive_target_count,
            "clean_relation_count": clean_count,
            "source_target_collapse_relation_count": collapse_count,
            "source_object_assignment_status": source_assignment,
            "target_object_assignment_status": target_assignment,
            "accepted_object_endpoint_pair": accepted_object_pair,
        }
    return rows


def _wall_evidence_by_pair(
    wall_pair_rows: pd.DataFrame,
    boundary_rows: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for pair_id, row in _index_by_pair(wall_pair_rows).items():
        ready = _as_bool(row.get("primitive_wall_evidence_ready"))
        rows[pair_id] = {
            "wall_evidence_status": _safe_str(row.get("primitive_wall_evidence_status")),
            "wall_evidence_ready_local_only": ready,
            "wall_evidence_scope": _safe_str(row.get("wall_evidence_scope")),
            "wall_evidence_source": "014_wall_evidence_pair_rows",
        }
    if not boundary_rows.empty:
        grouped = boundary_rows.groupby("local_pair_id", dropna=False)
        for pair_id, group in grouped:
            if str(pair_id) in rows:
                continue
            closed_count = int(group["boundary_guard_closed"].map(_as_bool).sum())
            rows[str(pair_id)] = {
                "wall_evidence_status": (
                    "boundary_guard_closed_no_positive_wall"
                    if closed_count == len(group)
                    else "boundary_guard_not_closed"
                ),
                "wall_evidence_ready_local_only": False,
                "wall_evidence_scope": "boundary_guard_from_014_wall_evidence_audit",
                "wall_evidence_source": "014_wall_evidence_boundary_guard_rows",
            }
    return rows


def _substrate_by_pair(substrate_rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for pair_id, row in _index_by_pair(substrate_rows).items():
        rows[pair_id] = {
            "local_gate_class": _safe_str(row.get("local_gate_class")),
            "local_gate_status": _safe_str(row.get("local_gate_status")),
            "fixed_016_local_signature_pass": _as_bool(
                row.get("fixed_016_local_signature_pass")
            ),
            "pair_scope": _safe_str(row.get("pair_scope")),
            "counterfactual_class": _safe_str(row.get("counterfactual_class")),
        }
    return rows


def _source_status(
    *,
    object_row: dict[str, Any] | None,
    source_anchor_pass_count: int,
    route_anchor_route_count: int,
) -> tuple[str, bool]:
    if object_row and object_row["source_object_assignment_status"].startswith("accepted"):
        return object_row["source_object_assignment_status"], True
    if object_row and object_row["source_object_assignment_status"].startswith("rejected"):
        return object_row["source_object_assignment_status"], False
    if route_anchor_route_count > 0 and source_anchor_pass_count == route_anchor_route_count:
        return "route_anchor_source_candidate_needs_object_identity", False
    if source_anchor_pass_count == 0:
        return "missing_source_basin_context", False
    return "partial_route_anchor_source_candidate_needs_object_identity", False


def _target_status(
    *,
    object_row: dict[str, Any] | None,
    target_anchor_pass_count: int,
    route_anchor_route_count: int,
) -> tuple[str, bool]:
    if object_row and object_row["target_object_assignment_status"].startswith("accepted"):
        return object_row["target_object_assignment_status"], True
    if object_row and object_row["target_object_assignment_status"].startswith("rejected"):
        return object_row["target_object_assignment_status"], False
    if route_anchor_route_count > 0 and target_anchor_pass_count == route_anchor_route_count:
        return "route_anchor_target_candidate_needs_object_identity", False
    if target_anchor_pass_count == 0:
        return "missing_target_basin_context", False
    return "partial_route_anchor_target_candidate_needs_object_identity", False


def _assignment_class(
    *,
    pair_id: str,
    accepted_pair: bool,
    morphology_class: str,
    source_status: str,
    target_status: str,
) -> str:
    if accepted_pair:
        return "accepted_local_object_basin_pair_current_morphology_guard"
    if pair_id == "local_pair_005" or "missing_source" in source_status:
        return "rejected_boundary_missing_or_collapsed_source"
    if morphology_class == POSITIVE_ROUTE_MORPHOLOGY_CLASS:
        return "positive_morphology_route_anchor_pair_needs_object_wall_evidence"
    if morphology_class in NEGATIVE_ROUTE_MORPHOLOGY_CLASSES:
        return "negative_morphology_route_anchor_pair_needs_object_wall_evidence"
    if "rejected" in source_status or "rejected" in target_status:
        return "rejected_object_boundary_or_collapse"
    return "diagnostic_assignment_surface_unresolved"


def _pathway_promotion_status(
    *,
    morphology_class: str,
    accepted_pair: bool,
    wall_status: str,
) -> str:
    if (
        accepted_pair
        and wall_status == "primitive_object_level_wall_evidence_ready_local_only"
        and morphology_class != POSITIVE_ROUTE_MORPHOLOGY_CLASS
    ):
        return "blocked_current_route_morphology_guard_despite_local_wall_evidence"
    if morphology_class == POSITIVE_ROUTE_MORPHOLOGY_CLASS and not accepted_pair:
        return "blocked_positive_morphology_missing_object_identity_and_wall_evidence"
    if morphology_class in NEGATIVE_ROUTE_MORPHOLOGY_CLASSES:
        return "blocked_negative_route_morphology_and_missing_wall_evidence"
    if morphology_class == BOUNDARY_ROUTE_MORPHOLOGY_CLASS:
        return "blocked_boundary_control"
    return "blocked_assignment_surface_only"


def _build_pair_rows(context: dict[str, Any]) -> pd.DataFrame:
    bridge_rows = context["bridge_pair_rows"].copy()
    bridge_rows = bridge_rows[
        bridge_rows["local_pair_id"].astype(str).isin(EXPECTED_ROUTE_SCOREABLE_PAIR_IDS)
    ].copy()
    order = {pair_id: index for index, pair_id in enumerate(EXPECTED_ROUTE_SCOREABLE_PAIR_IDS)}
    bridge_rows = (
        bridge_rows.assign(
            _sort_key=lambda frame: frame["local_pair_id"].map(order).fillna(999)
        )
        .sort_values(["_sort_key", "local_pair_id"], kind="mergesort")
        .drop(columns=["_sort_key"])
    )
    route_anchors = _route_anchor_aggregates(context["route_trace_route_rows"])
    route_anchors.update(_continuity_016_aggregate(context["continuity_016_route_rows"]))
    object_evidence = _object_evidence_by_pair(context["endpoint_object_pair_summary_rows"])
    wall_evidence = _wall_evidence_by_pair(
        context["wall_evidence_pair_rows"],
        context["wall_evidence_boundary_rows"],
    )
    substrate = _substrate_by_pair(context["route_negative_substrate_rows"])

    rows: list[dict[str, Any]] = []
    for _, bridge_row in bridge_rows.iterrows():
        pair_id = str(bridge_row["local_pair_id"])
        morphology_class = str(bridge_row["route_state_morphology_class"])
        route_anchor = route_anchors.get(
            pair_id,
            {
                "route_anchor_source": "missing_route_anchor_rows",
                "route_anchor_route_count": 0,
                "source_anchor_pass_count": 0,
                "target_anchor_pass_count": 0,
                "target_expected_anchor_pass_count": 0,
                "continuity_pass_count": None,
                "unknown_endpoint_step_count": None,
                "route_anchor_note": "no route-anchor rows available",
            },
        )
        object_row = object_evidence.get(pair_id)
        wall_row = wall_evidence.get(
            pair_id,
            {
                "wall_evidence_status": "not_tested_no_object_endpoint_pair",
                "wall_evidence_ready_local_only": False,
                "wall_evidence_scope": "not_tested",
                "wall_evidence_source": "none",
            },
        )
        substrate_row = substrate.get(pair_id, {})
        source_status, accepted_source = _source_status(
            object_row=object_row,
            source_anchor_pass_count=route_anchor["source_anchor_pass_count"],
            route_anchor_route_count=route_anchor["route_anchor_route_count"],
        )
        target_status, accepted_target = _target_status(
            object_row=object_row,
            target_anchor_pass_count=route_anchor["target_anchor_pass_count"],
            route_anchor_route_count=route_anchor["route_anchor_route_count"],
        )
        accepted_pair = accepted_source and accepted_target
        assignment_class = _assignment_class(
            pair_id=pair_id,
            accepted_pair=accepted_pair,
            morphology_class=morphology_class,
            source_status=source_status,
            target_status=target_status,
        )
        pathway_status = _pathway_promotion_status(
            morphology_class=morphology_class,
            accepted_pair=accepted_pair,
            wall_status=wall_row["wall_evidence_status"],
        )
        rows.append(
            {
                "local_pair_id": pair_id,
                "route_state_morphology_class": morphology_class,
                "basin_state_assignment_class": assignment_class,
                "source_basin_assignment_status": source_status,
                "target_basin_assignment_status": target_status,
                "accepted_source_basin_candidate": accepted_source,
                "accepted_target_basin_candidate": accepted_target,
                "accepted_local_object_basin_pair": accepted_pair,
                "route_anchor_source": route_anchor["route_anchor_source"],
                "route_anchor_route_count": route_anchor["route_anchor_route_count"],
                "source_anchor_pass_count": route_anchor["source_anchor_pass_count"],
                "target_anchor_pass_count": route_anchor["target_anchor_pass_count"],
                "target_expected_anchor_pass_count": route_anchor[
                    "target_expected_anchor_pass_count"
                ],
                "continuity_pass_count": route_anchor["continuity_pass_count"],
                "unknown_endpoint_step_count": route_anchor["unknown_endpoint_step_count"],
                "object_evidence_status": (
                    object_row["object_evidence_status"]
                    if object_row
                    else "missing_endpoint_object_identity_evidence"
                ),
                "object_audit_class": object_row["object_audit_class"] if object_row else "",
                "object_route_count": object_row["object_route_count"] if object_row else 0,
                "source_object_count": object_row["source_object_count"] if object_row else 0,
                "exclusive_target_object_count": (
                    object_row["exclusive_target_object_count"] if object_row else 0
                ),
                "clean_relation_count": object_row["clean_relation_count"] if object_row else 0,
                "source_target_collapse_relation_count": (
                    object_row["source_target_collapse_relation_count"] if object_row else 0
                ),
                "wall_evidence_status": wall_row["wall_evidence_status"],
                "wall_evidence_ready_local_only": wall_row["wall_evidence_ready_local_only"],
                "wall_evidence_scope": wall_row["wall_evidence_scope"],
                "pathway_label_promotion_status": pathway_status,
                "accepted_pathway_label": False,
                "demo_readiness_status": _demo_readiness_status(
                    pair_id=pair_id,
                    assignment_class=assignment_class,
                    pathway_status=pathway_status,
                ),
                "local_gate_class": substrate_row.get("local_gate_class", ""),
                "local_gate_status": substrate_row.get("local_gate_status", ""),
                "pair_scope": substrate_row.get("pair_scope", ""),
                "counterfactual_class": substrate_row.get("counterfactual_class", ""),
                "evidence_boundary": _evidence_boundary(
                    pair_id=pair_id,
                    assignment_class=assignment_class,
                    pathway_status=pathway_status,
                ),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows)


def _demo_readiness_status(
    *,
    pair_id: str,
    assignment_class: str,
    pathway_status: str,
) -> str:
    if pair_id == "local_pair_014":
        return "local_object_wall_demo_candidate_requires_route_morphology_reconciliation"
    if assignment_class.startswith("positive_morphology"):
        return "needs_endpoint_object_and_wall_evidence_before_demo"
    if assignment_class.startswith("negative_morphology"):
        return "negative_guard_not_demo_target"
    if "boundary" in assignment_class:
        return "boundary_control_not_demo_target"
    return f"not_demo_ready:{pathway_status}"


def _evidence_boundary(
    *,
    pair_id: str,
    assignment_class: str,
    pathway_status: str,
) -> str:
    if pair_id == "local_pair_014":
        return (
            "Local object endpoint pair and primitive wall evidence are available "
            "for 014, but current fixed-predicate morphology is a guard; do not "
            "collapse this into a general pathway claim."
        )
    if pair_id == "local_pair_016":
        return (
            "016 has positive route morphology and source/target anchors, but "
            "object endpoint identity and wall evidence are missing."
        )
    if "boundary" in assignment_class:
        return "Boundary/control evidence rejects basin-state promotion."
    return f"Route-anchor candidate only; {pathway_status}."


def _build_evidence_rows(pair_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in pair_rows.iterrows():
        pair_id = str(row["local_pair_id"])
        evidence_specs = [
            (
                "route_anchor_source",
                row["route_anchor_source"],
                {
                    "route_count": row["route_anchor_route_count"],
                    "source_anchor_pass_count": row["source_anchor_pass_count"],
                    "target_anchor_pass_count": row["target_anchor_pass_count"],
                    "target_expected_anchor_pass_count": row[
                        "target_expected_anchor_pass_count"
                    ],
                    "continuity_pass_count": row["continuity_pass_count"],
                    "unknown_endpoint_step_count": row["unknown_endpoint_step_count"],
                },
            ),
            (
                "endpoint_object_identity",
                row["object_evidence_status"],
                {
                    "object_audit_class": row["object_audit_class"],
                    "object_route_count": row["object_route_count"],
                    "source_object_count": row["source_object_count"],
                    "exclusive_target_object_count": row["exclusive_target_object_count"],
                    "clean_relation_count": row["clean_relation_count"],
                    "source_target_collapse_relation_count": row[
                        "source_target_collapse_relation_count"
                    ],
                },
            ),
            (
                "wall_evidence",
                row["wall_evidence_status"],
                {
                    "wall_evidence_ready_local_only": row["wall_evidence_ready_local_only"],
                    "wall_evidence_scope": row["wall_evidence_scope"],
                },
            ),
            (
                "assignment_decision",
                row["basin_state_assignment_class"],
                {
                    "source_basin_assignment_status": row[
                        "source_basin_assignment_status"
                    ],
                    "target_basin_assignment_status": row[
                        "target_basin_assignment_status"
                    ],
                    "accepted_local_object_basin_pair": row[
                        "accepted_local_object_basin_pair"
                    ],
                    "pathway_label_promotion_status": row[
                        "pathway_label_promotion_status"
                    ],
                },
            ),
        ]
        for evidence_type, evidence_status, evidence_payload in evidence_specs:
            rows.append(
                {
                    "local_pair_id": pair_id,
                    "evidence_type": evidence_type,
                    "evidence_status": evidence_status,
                    "evidence_payload_json": json.dumps(
                        _json_safe(evidence_payload), sort_keys=True
                    ),
                    "claim_boundary": CLAIM_BOUNDARY,
                    "run_status": RUN_STATUS,
                }
            )
    return pd.DataFrame(rows)


def _build_class_rows(pair_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = pair_rows.groupby("basin_state_assignment_class", dropna=False)
    for assignment_class, group in grouped:
        rows.append(
            {
                "basin_state_assignment_class": assignment_class,
                "pair_count": int(len(group)),
                "local_pair_ids": ";".join(group["local_pair_id"].astype(str)),
                "accepted_local_object_basin_pair_count": int(
                    group["accepted_local_object_basin_pair"].map(_as_bool).sum()
                ),
                "accepted_pathway_label_count": int(
                    group["accepted_pathway_label"].map(_as_bool).sum()
                ),
                "route_morphology_classes": ";".join(
                    sorted(set(group["route_state_morphology_class"].astype(str)))
                ),
                "class_interpretation": _class_interpretation(str(assignment_class)),
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    order = {
        "accepted_local_object_basin_pair_current_morphology_guard": 0,
        "positive_morphology_route_anchor_pair_needs_object_wall_evidence": 1,
        "negative_morphology_route_anchor_pair_needs_object_wall_evidence": 2,
        "rejected_boundary_missing_or_collapsed_source": 3,
        "rejected_object_boundary_or_collapse": 4,
        "diagnostic_assignment_surface_unresolved": 5,
    }
    return (
        pd.DataFrame(rows)
        .assign(_sort_key=lambda frame: frame["basin_state_assignment_class"].map(order).fillna(99))
        .sort_values(["_sort_key", "basin_state_assignment_class"], kind="mergesort")
        .drop(columns=["_sort_key"])
    )


def _class_interpretation(assignment_class: str) -> str:
    if assignment_class == "accepted_local_object_basin_pair_current_morphology_guard":
        return (
            "Local endpoint-object basin pair evidence exists, but current "
            "fixed-predicate route morphology is not a promotable pathway label."
        )
    if assignment_class == "positive_morphology_route_anchor_pair_needs_object_wall_evidence":
        return (
            "Positive route morphology exists with route anchors, but object "
            "endpoint identity and wall evidence are still missing."
        )
    if assignment_class == "negative_morphology_route_anchor_pair_needs_object_wall_evidence":
        return (
            "Source/target route anchors exist, but morphology is a negative guard "
            "and object/wall evidence is missing."
        )
    if assignment_class == "rejected_boundary_missing_or_collapsed_source":
        return "Boundary/control evidence lacks or collapses source context."
    return "Diagnostic assignment class; no promotion allowed."


def _build_requirement_rows(pair_rows: pd.DataFrame) -> pd.DataFrame:
    pair_count = int(len(pair_rows))
    accepted_pair_count = int(
        pair_rows["accepted_local_object_basin_pair"].map(_as_bool).sum()
    )
    accepted_pathway_count = int(pair_rows["accepted_pathway_label"].map(_as_bool).sum())
    route_anchor_pair_count = int(
        (
            (pair_rows["source_anchor_pass_count"] == pair_rows["route_anchor_route_count"])
            & (pair_rows["target_anchor_pass_count"] == pair_rows["route_anchor_route_count"])
            & (pair_rows["route_anchor_route_count"] > 0)
        ).sum()
    )
    object_missing_count = int(
        pair_rows["object_evidence_status"]
        .astype(str)
        .eq("missing_endpoint_object_identity_evidence")
        .sum()
    )
    wall_ready_count = int(pair_rows["wall_evidence_ready_local_only"].map(_as_bool).sum())
    rows = [
        {
            "requirement_id": "R1_route_anchor_attachment",
            "requirement_question": "Were route-anchor source/target readouts attached?",
            "observed": f"{route_anchor_pair_count} of {pair_count} rows have source and target route anchors",
            "requirement_status": "satisfied_as_candidate_evidence",
            "blocker": "route anchors are candidates, not accepted basin identity",
        },
        {
            "requirement_id": "R2_endpoint_object_identity",
            "requirement_question": "Which rows have accepted endpoint-object basin identity?",
            "observed": f"{accepted_pair_count} accepted local object endpoint pair; {object_missing_count} rows missing object evidence",
            "requirement_status": "partially_satisfied_local_only",
            "blocker": "only 014 has accepted clean object endpoint evidence; 016/009/012/020 lack object identity evidence and 005 is boundary/collapse",
        },
        {
            "requirement_id": "R3_wall_evidence_attachment",
            "requirement_question": "Which rows have wall evidence attached?",
            "observed": f"{wall_ready_count} local-only primitive wall evidence row",
            "requirement_status": "partially_satisfied_local_only",
            "blocker": "wall evidence exists only for 014 local object audit and is not general",
        },
        {
            "requirement_id": "R4_pathway_label_promotion",
            "requirement_question": "Can any current fixed-predicate pathway label be accepted?",
            "observed": f"{accepted_pathway_count} accepted pathway labels",
            "requirement_status": "blocked",
            "blocker": "014 has wall evidence but current morphology is a guard; 016 has positive morphology but no object/wall evidence",
        },
        {
            "requirement_id": "R5_next_gate",
            "requirement_question": "What should the next execution gate test?",
            "observed": "assignment surface separates 014 local object-wall evidence from 016 positive route morphology",
            "requirement_status": "design_ready",
            "blocker": "need object-identity/wall evidence for 016-like morphology or explicit reconciliation with the 014 wall-object surface",
        },
    ]
    for row in rows:
        row["claim_boundary"] = CLAIM_BOUNDARY
        row["run_status"] = RUN_STATUS
    return pd.DataFrame(rows)


def _build_decision_rows(pair_rows: pd.DataFrame) -> pd.DataFrame:
    accepted_pairs = list(
        pair_rows.loc[
            pair_rows["accepted_local_object_basin_pair"].map(_as_bool), "local_pair_id"
        ].astype(str)
    )
    positive_anchor_needs_object = list(
        pair_rows.loc[
            pair_rows["basin_state_assignment_class"].astype(str).eq(
                "positive_morphology_route_anchor_pair_needs_object_wall_evidence"
            ),
            "local_pair_id",
        ].astype(str)
    )
    negative_guard_pairs = list(
        pair_rows.loc[
            pair_rows["route_state_morphology_class"]
            .astype(str)
            .isin(NEGATIVE_ROUTE_MORPHOLOGY_CLASSES),
            "local_pair_id",
        ].astype(str)
    )
    decisions = [
        {
            "decision_id": "D1_route_anchors_are_candidates_not_basin_identity",
            "decision": (
                "Source/target route-anchor readouts can seed basin-state candidates, "
                "but they are not accepted basin identity without object evidence."
            ),
            "evidence": "route anchors exist for 016/014/009/012/020; 005 lacks source context",
            "decision_status": "accepted_guardrail",
        },
        {
            "decision_id": "D2_014_only_accepted_local_object_basin_pair",
            "decision": (
                "Only local_pair_014 has accepted clean local endpoint-object basin "
                "pair evidence on the current evidence surface."
            ),
            "evidence": f"accepted_local_object_basin_pair_ids={';'.join(accepted_pairs) or 'none'}",
            "decision_status": "accepted_local_only",
        },
        {
            "decision_id": "D3_016_positive_morphology_not_basin_state_ready",
            "decision": (
                "local_pair_016 remains a positive route-morphology reference, but "
                "not an accepted basin-state pair."
            ),
            "evidence": f"positive_anchor_needs_object={';'.join(positive_anchor_needs_object) or 'none'}",
            "decision_status": "accepted_blocker",
        },
        {
            "decision_id": "D4_negative_guards_do_not_promote",
            "decision": (
                "009/012/014/020 remain guards for current fixed-predicate route "
                "morphology; object or wall evidence does not automatically promote "
                "a current pathway label."
            ),
            "evidence": f"negative_guard_pairs={';'.join(negative_guard_pairs) or 'none'}",
            "decision_status": "accepted_guardrail",
        },
        {
            "decision_id": "D5_next_gate_reconcile_assignment_and_morphology",
            "decision": (
                "The next gate should attach object-identity and wall evidence to "
                "016-like positive morphology, or explicitly reconcile why 014's "
                "object-wall evidence belongs to a different route surface."
            ),
            "evidence": "014 has local object-wall evidence; 016 has positive morphology but lacks object-wall evidence",
            "decision_status": "next_gate",
        },
    ]
    for row in decisions:
        row["claim_boundary"] = CLAIM_BOUNDARY
        row["run_status"] = RUN_STATUS
    return pd.DataFrame(decisions)


def _build_gate_matrix(
    *,
    context: dict[str, Any],
    pair_rows: pd.DataFrame,
    requirement_rows: pd.DataFrame,
    next_gate: str,
) -> pd.DataFrame:
    pair_ids = list(pair_rows["local_pair_id"].astype(str))
    accepted_pair_ids = list(
        pair_rows.loc[
            pair_rows["accepted_local_object_basin_pair"].map(_as_bool), "local_pair_id"
        ].astype(str)
    )
    accepted_pathway_ids = list(
        pair_rows.loc[pair_rows["accepted_pathway_label"].map(_as_bool), "local_pair_id"]
        .astype(str)
    )
    object_statuses = dict(
        zip(
            pair_rows["local_pair_id"].astype(str),
            pair_rows["object_evidence_status"].astype(str),
            strict=False,
        )
    )
    assignment_classes = dict(
        zip(
            pair_rows["local_pair_id"].astype(str),
            pair_rows["basin_state_assignment_class"].astype(str),
            strict=False,
        )
    )
    requirement_statuses = dict(
        zip(
            requirement_rows["requirement_id"].astype(str),
            requirement_rows["requirement_status"].astype(str),
            strict=False,
        )
    )
    gates = [
        _gate_row(
            "G1_sources_readable",
            "Were bridge, route, continuity, endpoint-object, and wall-evidence sources readable?",
            {
                "bridge_pair_rows": int(len(context["bridge_pair_rows"])),
                "bridge_failed_gates": context["bridge_summary"].get("failed_gates"),
                "route_trace_route_rows": int(len(context["route_trace_route_rows"])),
                "continuity_016_route_rows": int(len(context["continuity_016_route_rows"])),
                "endpoint_object_pair_summary_rows": int(
                    len(context["endpoint_object_pair_summary_rows"])
                ),
                "wall_evidence_pair_rows": int(len(context["wall_evidence_pair_rows"])),
            },
            "all required sources have rows and upstream bridge has no failed gates",
            bool(len(context["bridge_pair_rows"]))
            and bool(len(context["route_trace_route_rows"]))
            and bool(len(context["continuity_016_route_rows"]))
            and bool(len(context["endpoint_object_pair_summary_rows"]))
            and bool(len(context["wall_evidence_pair_rows"]))
            and not context["bridge_summary"].get("failed_gates"),
        ),
        _gate_row(
            "G2_route_scoreable_surface_preserved",
            "Does the assignment surface cover exactly the six route-scoreable pairs?",
            pair_ids,
            "exactly 016,014,009,012,020,005",
            tuple(pair_ids) == EXPECTED_ROUTE_SCOREABLE_PAIR_IDS,
        ),
        _gate_row(
            "G3_route_anchor_evidence_attached",
            "Were source/target route-anchor candidates attached without accepting them as identity?",
            {
                "route_anchor_counts": {
                    str(row["local_pair_id"]): {
                        "route_count": int(row["route_anchor_route_count"]),
                        "source": int(row["source_anchor_pass_count"]),
                        "target": int(row["target_anchor_pass_count"]),
                    }
                    for _, row in pair_rows.iterrows()
                }
            },
            "016/014/009/012/020 have full source-target anchors; 005 lacks source context",
            assignment_classes.get("local_pair_005")
            == "rejected_boundary_missing_or_collapsed_source"
            and all(
                pair_rows.loc[
                    pair_rows["local_pair_id"].astype(str).eq(pair_id),
                    "source_anchor_pass_count",
                ].iloc[0]
                > 0
                for pair_id in ("local_pair_016", "local_pair_014", "local_pair_009", "local_pair_012", "local_pair_020")
            ),
        ),
        _gate_row(
            "G4_object_identity_limited_to_014",
            "Is accepted endpoint-object identity limited to 014 and is 005 rejected as boundary/collapse?",
            object_statuses,
            "014 accepted clean object pair; 005 rejected boundary; 016/009/012/020 missing object evidence",
            object_statuses.get("local_pair_014")
            == "accepted_clean_local_endpoint_object_pair"
            and object_statuses.get("local_pair_005")
            == "rejected_boundary_source_target_collapse"
            and all(
                object_statuses.get(pair_id)
                == "missing_endpoint_object_identity_evidence"
                for pair_id in ("local_pair_016", "local_pair_009", "local_pair_012", "local_pair_020")
            ),
        ),
        _gate_row(
            "G5_basin_state_pair_acceptance_local_only",
            "Is basin-state pair acceptance local-only and restricted to 014?",
            accepted_pair_ids,
            "only local_pair_014 accepted as local object endpoint pair",
            accepted_pair_ids == ["local_pair_014"],
        ),
        _gate_row(
            "G6_pathway_label_promotion_blocked",
            "Are current fixed-predicate pathway labels still blocked?",
            {
                "accepted_pathway_label_ids": accepted_pathway_ids,
                "pathway_statuses": dict(
                    zip(
                        pair_rows["local_pair_id"].astype(str),
                        pair_rows["pathway_label_promotion_status"].astype(str),
                        strict=False,
                    )
                ),
            },
            "zero accepted pathway labels",
            accepted_pathway_ids == [],
        ),
        _gate_row(
            "G7_next_gate_reconciles_assignment_and_morphology",
            "Is the next gate object/wall attachment for positive morphology or reconciliation of 014 versus 016 surfaces?",
            {
                "next_gate": next_gate,
                "requirement_statuses": requirement_statuses,
            },
            "next gate is not route rerun, candidate expansion, or method promotion",
            "object-identity and wall evidence" in next_gate
            and requirement_statuses.get("R4_pathway_label_promotion") == "blocked",
        ),
        _gate_row(
            "G8_claim_boundaries_closed",
            "Are method, general wall, pathway-label, quality/cost, and full-replay claims closed?",
            CLAIM_BOUNDARY,
            "read-only assignment surface only",
            all(
                pair_rows[column].astype(str).eq(expected).all()
                for column, expected in {
                    "route_execution_status": ROUTE_EXECUTION_STATUS,
                    "wall_promotion_status": WALL_PROMOTION_STATUS,
                    "method_status": METHOD_STATUS,
                }.items()
            )
            and not pair_rows["accepted_pathway_label"].map(_as_bool).any(),
        ),
    ]
    return pd.DataFrame(gates)


def _load_context(
    *,
    bridge_dir: Path,
    route_trace_dir: Path,
    route_negative_dir: Path,
    continuity_016_dir: Path,
    endpoint_object_dir: Path,
    wall_evidence_014_dir: Path,
) -> dict[str, Any]:
    return {
        "paths": {
            "bridge_dir": bridge_dir,
            "route_trace_dir": route_trace_dir,
            "route_negative_dir": route_negative_dir,
            "continuity_016_dir": continuity_016_dir,
            "endpoint_object_dir": endpoint_object_dir,
            "wall_evidence_014_dir": wall_evidence_014_dir,
        },
        "bridge_pair_rows": _read_csv(bridge_dir / BRIDGE_PAIR_ROWS_CSV),
        "bridge_gate_rows": _read_csv(bridge_dir / BRIDGE_GATE_MATRIX_CSV),
        "bridge_summary": _read_json(bridge_dir / BRIDGE_SUMMARY_JSON),
        "route_trace_route_rows": _read_csv(route_trace_dir / ROUTE_TRACE_ROUTE_ROWS_CSV),
        "route_negative_pair_rows": _read_csv(route_negative_dir / ROUTE_NEGATIVE_PAIR_ROWS_CSV),
        "route_negative_substrate_rows": _read_csv(
            route_negative_dir / ROUTE_NEGATIVE_SUBSTRATE_ROWS_CSV
        ),
        "continuity_016_route_rows": _read_csv(
            continuity_016_dir / CONTINUITY_016_ROUTE_ROWS_CSV
        ),
        "continuity_016_summary": _read_json(continuity_016_dir / CONTINUITY_016_SUMMARY_JSON),
        "endpoint_object_pair_summary_rows": _read_csv(
            endpoint_object_dir / ENDPOINT_OBJECT_PAIR_SUMMARY_ROWS_CSV
        ),
        "endpoint_object_rows": _read_csv(endpoint_object_dir / ENDPOINT_OBJECT_ROWS_CSV),
        "endpoint_object_summary": _read_json(
            endpoint_object_dir / ENDPOINT_OBJECT_SUMMARY_JSON
        ),
        "wall_evidence_pair_rows": _read_csv(
            wall_evidence_014_dir / WALL_EVIDENCE_PAIR_ROWS_CSV
        ),
        "wall_evidence_boundary_rows": _read_csv(
            wall_evidence_014_dir / WALL_EVIDENCE_BOUNDARY_ROWS_CSV
        ),
        "wall_evidence_summary": _read_json(
            wall_evidence_014_dir / WALL_EVIDENCE_SUMMARY_JSON
        ),
    }


def _write_report(
    *,
    output_dir: Path,
    pair_rows: pd.DataFrame,
    evidence_rows: pd.DataFrame,
    class_rows: pd.DataFrame,
    requirement_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    report = f"""# NanoClustering G4.8 First-Pass Basin-State Assignment Surface

## Status

- Assignment status: `{summary["assignment_surface_status"]}`
- Accepted local object basin pairs: `{", ".join(summary["accepted_local_object_basin_pair_ids"]) or "none"}`
- Accepted pathway labels: `{", ".join(summary["accepted_pathway_label_ids"]) or "none"}`
- Recommended next gate: {summary["recommended_next_gate"]}

## Pair Assignments

{_markdown_table(
    pair_rows,
    [
        "local_pair_id",
        "route_state_morphology_class",
        "basin_state_assignment_class",
        "source_basin_assignment_status",
        "target_basin_assignment_status",
        "object_evidence_status",
        "wall_evidence_status",
        "pathway_label_promotion_status",
    ],
)}

## Assignment Classes

{_markdown_table(
    class_rows,
    [
        "basin_state_assignment_class",
        "pair_count",
        "local_pair_ids",
        "accepted_local_object_basin_pair_count",
        "accepted_pathway_label_count",
        "class_interpretation",
    ],
)}

## Requirements

{_markdown_table(
    requirement_rows,
    [
        "requirement_id",
        "requirement_status",
        "observed",
        "blocker",
    ],
)}

## Decisions

{_markdown_table(
    decision_rows,
    [
        "decision_id",
        "decision_status",
        "decision",
        "evidence",
    ],
)}

## Gates

{_markdown_table(
    gate_matrix,
    [
        "gate_id",
        "gate_status",
        "question",
        "minimum_or_rule",
    ],
)}

## Evidence Rows

{_markdown_table(
    evidence_rows,
    [
        "local_pair_id",
        "evidence_type",
        "evidence_status",
    ],
)}

## Claim Boundary

{CLAIM_BOUNDARY}
"""
    (output_dir / REPORT_MD).write_text(report, encoding="utf-8")


def run(
    *,
    bridge_dir: Path,
    route_trace_dir: Path,
    route_negative_dir: Path,
    continuity_016_dir: Path,
    endpoint_object_dir: Path,
    wall_evidence_014_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    context = _load_context(
        bridge_dir=bridge_dir,
        route_trace_dir=route_trace_dir,
        route_negative_dir=route_negative_dir,
        continuity_016_dir=continuity_016_dir,
        endpoint_object_dir=endpoint_object_dir,
        wall_evidence_014_dir=wall_evidence_014_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows = _build_pair_rows(context)
    evidence_rows = _build_evidence_rows(pair_rows)
    class_rows = _build_class_rows(pair_rows)
    requirement_rows = _build_requirement_rows(pair_rows)
    decision_rows = _build_decision_rows(pair_rows)
    recommended_next_gate = (
        "Attach object-identity and wall evidence to 016-like positive route "
        "morphology, or explicitly reconcile why 014's local object-wall evidence "
        "belongs to a different route surface; do not promote pathway labels, "
        "rerun routes, or broaden candidates before that distinction is resolved."
    )
    gate_matrix = _build_gate_matrix(
        context=context,
        pair_rows=pair_rows,
        requirement_rows=requirement_rows,
        next_gate=recommended_next_gate,
    )
    failed_gates = list(
        gate_matrix.loc[gate_matrix["gate_status"].astype(str).eq("fail"), "gate_id"].astype(str)
    )
    accepted_local_object_basin_pair_ids = list(
        pair_rows.loc[
            pair_rows["accepted_local_object_basin_pair"].map(_as_bool), "local_pair_id"
        ].astype(str)
    )
    accepted_pathway_label_ids = list(
        pair_rows.loc[pair_rows["accepted_pathway_label"].map(_as_bool), "local_pair_id"]
        .astype(str)
    )
    summary = {
        "assignment_surface_status": (
            "local_object_basin_pair_only_for_014_positive_morphology_still_unassigned"
        ),
        "output_dir": str(output_dir),
        "route_scoreable_pair_ids": list(pair_rows["local_pair_id"].astype(str)),
        "accepted_local_object_basin_pair_ids": accepted_local_object_basin_pair_ids,
        "accepted_pathway_label_ids": accepted_pathway_label_ids,
        "positive_morphology_unassigned_pair_ids": list(
            pair_rows.loc[
                pair_rows["basin_state_assignment_class"].astype(str).eq(
                    "positive_morphology_route_anchor_pair_needs_object_wall_evidence"
                ),
                "local_pair_id",
            ].astype(str)
        ),
        "assignment_class_counts": [
            {
                "basin_state_assignment_class": row["basin_state_assignment_class"],
                "pair_count": int(row["pair_count"]),
            }
            for _, row in class_rows.iterrows()
        ],
        "failed_gates": failed_gates,
        "recommended_next_gate": recommended_next_gate,
        "claim_boundary": CLAIM_BOUNDARY,
        "run_status": RUN_STATUS,
    }
    config = {
        "bridge_dir": str(bridge_dir),
        "route_trace_dir": str(route_trace_dir),
        "route_negative_dir": str(route_negative_dir),
        "continuity_016_dir": str(continuity_016_dir),
        "endpoint_object_dir": str(endpoint_object_dir),
        "wall_evidence_014_dir": str(wall_evidence_014_dir),
        "output_dir": str(output_dir),
        "expected_route_scoreable_pair_ids": list(EXPECTED_ROUTE_SCOREABLE_PAIR_IDS),
        "route_execution_status": ROUTE_EXECUTION_STATUS,
        "wall_promotion_status": WALL_PROMOTION_STATUS,
        "method_status": METHOD_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "run_status": RUN_STATUS,
    }

    _write_csv(pair_rows, output_dir / PAIR_ROWS_CSV)
    _write_csv(evidence_rows, output_dir / EVIDENCE_ROWS_CSV)
    _write_csv(class_rows, output_dir / CLASS_ROWS_CSV)
    _write_csv(requirement_rows, output_dir / REQUIREMENT_ROWS_CSV)
    _write_csv(decision_rows, output_dir / DECISION_ROWS_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        pair_rows=pair_rows,
        evidence_rows=evidence_rows,
        class_rows=class_rows,
        requirement_rows=requirement_rows,
        decision_rows=decision_rows,
        gate_matrix=gate_matrix,
        summary=summary,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize first-pass basin-state assignment evidence."
    )
    parser.add_argument("--bridge-dir", type=Path, default=DEFAULT_BRIDGE_DIR)
    parser.add_argument("--route-trace-dir", type=Path, default=DEFAULT_ROUTE_TRACE_DIR)
    parser.add_argument("--route-negative-dir", type=Path, default=DEFAULT_ROUTE_NEGATIVE_DIR)
    parser.add_argument("--continuity-016-dir", type=Path, default=DEFAULT_CONTINUITY_016_DIR)
    parser.add_argument("--endpoint-object-dir", type=Path, default=DEFAULT_ENDPOINT_OBJECT_DIR)
    parser.add_argument("--wall-evidence-014-dir", type=Path, default=DEFAULT_WALL_EVIDENCE_014_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = run(
        bridge_dir=args.bridge_dir,
        route_trace_dir=args.route_trace_dir,
        route_negative_dir=args.route_negative_dir,
        continuity_016_dir=args.continuity_016_dir,
        endpoint_object_dir=args.endpoint_object_dir,
        wall_evidence_014_dir=args.wall_evidence_014_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
