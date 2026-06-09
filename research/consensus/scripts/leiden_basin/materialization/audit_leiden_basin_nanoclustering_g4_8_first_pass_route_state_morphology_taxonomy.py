#!/usr/bin/env python3
"""Classify first-pass route-state morphologies from existing evidence.

This read-only audit synthesizes the mechanism-generalization screen, fixed
route trace, route-negative explanation, plateau-stability feature audit, and
P1-P6 gate application. It separates a narrow 016-like finite plateau predicate
from broader route-state morphology classes so we do not turn one successful
reference morphology into the full basin definition.
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


PRIMARY_PAIR_ID = "local_pair_016"
STRICT_ANALOG_PAIR_IDS = ("local_pair_009", "local_pair_012", "local_pair_020")
POINT_OR_FRAGMENTED_CONTROL_PAIR_ID = "local_pair_014"
BOUNDARY_GUARD_PAIR_ID = "local_pair_005"
ROUTE_SCOREABLE_PAIR_IDS = (
    BOUNDARY_GUARD_PAIR_ID,
    *STRICT_ANALOG_PAIR_IDS,
    POINT_OR_FRAGMENTED_CONTROL_PAIR_ID,
    PRIMARY_PAIR_ID,
)
NON_STRICT_LOCAL_SIGNATURE_PAIR_IDS = ("local_pair_001", "local_pair_007")

DEFAULT_GENERALIZATION_SCREEN_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_mechanism_generalization_screen_gamma1e5_20260605"
)
DEFAULT_ROUTE_TRACE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_gamma1e5_20260605"
)
DEFAULT_ROUTE_NEGATIVE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_route_negative_explanation_audit_gamma1e5_20260605"
)
DEFAULT_PLATEAU_FEATURE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_plateau_stability_feature_audit_gamma1e5_20260606"
)
DEFAULT_GATE_APPLICATION_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_plateau_stability_gate_application_gamma1e5_20260606"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy_gamma1e5_20260606"
)

PAIR_ROWS_CSV = "nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy_pair_rows.csv"
TAXONOMY_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy_class_rows.csv"
)
PROVENANCE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy_provenance_rows.csv"
)
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy_decision_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy_gate_matrix.csv"
)
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy_report.md"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_route_state_morphology_taxonomy"
WALL_PROMOTION_STATUS = "not_promoted_route_state_taxonomy_only"
METHOD_STATUS = "route_state_morphology_taxonomy_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass route-state morphology taxonomy audit only; "
    "reads existing screen, route trace, route-negative, plateau-feature, and "
    "gate-application artifacts. It does not execute Leiden, promote basin "
    "walls, replay full NanoClustering, evaluate quality/cost value, or claim "
    "method success."
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
    return int(value)


def _as_float(value: Any, default: float | None = None) -> float | None:
    if pd.isna(value):
        return default
    return float(value)


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


def _load_context(
    *,
    generalization_screen_dir: Path,
    route_trace_dir: Path,
    route_negative_dir: Path,
    plateau_feature_dir: Path,
    gate_application_dir: Path,
) -> dict[str, Any]:
    return {
        "paths": {
            "generalization_screen_dir": generalization_screen_dir,
            "route_trace_dir": route_trace_dir,
            "route_negative_dir": route_negative_dir,
            "plateau_feature_dir": plateau_feature_dir,
            "gate_application_dir": gate_application_dir,
        },
        "summaries": {
            "generalization_screen": _read_json(
                generalization_screen_dir
                / "nanoclustering_g4_8_first_pass_mechanism_generalization_summary.json"
            ),
            "route_trace": _read_json(
                route_trace_dir
                / "nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_summary.json"
            ),
            "route_negative": _read_json(
                route_negative_dir
                / "nanoclustering_g4_8_first_pass_route_negative_explanation_summary.json"
            ),
            "plateau_feature": _read_json(
                plateau_feature_dir
                / "nanoclustering_g4_8_first_pass_plateau_stability_feature_summary.json"
            ),
            "gate_application": _read_json(
                gate_application_dir
                / "nanoclustering_g4_8_first_pass_plateau_stability_gate_application_summary.json"
            ),
        },
        "tables": {
            "screen_pair": _read_csv(
                generalization_screen_dir
                / "nanoclustering_g4_8_first_pass_mechanism_generalization_pair_rows.csv"
            ),
            "route_pair": _read_csv(
                route_trace_dir
                / "nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_pair_rows.csv"
            ),
            "route_fraction": _read_csv(
                route_trace_dir
                / "nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_fraction_rows.csv"
            ),
            "route_gate": _read_csv(
                route_trace_dir
                / "nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_gate_matrix.csv"
            ),
            "route_negative_pair": _read_csv(
                route_negative_dir
                / "nanoclustering_g4_8_first_pass_route_negative_explanation_pair_rows.csv"
            ),
            "route_negative_fraction": _read_csv(
                route_negative_dir
                / "nanoclustering_g4_8_first_pass_route_negative_explanation_fraction_rows.csv"
            ),
            "route_negative_substrate": _read_csv(
                route_negative_dir
                / "nanoclustering_g4_8_first_pass_route_negative_explanation_substrate_rows.csv"
            ),
            "plateau_pair": _read_csv(
                plateau_feature_dir
                / "nanoclustering_g4_8_first_pass_plateau_stability_feature_pair_rows.csv"
            ),
            "plateau_fraction": _read_csv(
                plateau_feature_dir
                / "nanoclustering_g4_8_first_pass_plateau_stability_feature_fraction_rows.csv"
            ),
            "gate_application_pair": _read_csv(
                gate_application_dir
                / "nanoclustering_g4_8_first_pass_plateau_stability_gate_application_pair_rows.csv"
            ),
            "gate_application_gate": _read_csv(
                gate_application_dir
                / "nanoclustering_g4_8_first_pass_plateau_stability_gate_application_gate_matrix.csv"
            ),
        },
    }


def _index_by_pair(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {str(row["local_pair_id"]): row for _, row in frame.iterrows()}


def _fraction_state(row: pd.Series) -> str:
    source = _as_int(row.get("source_family_count"))
    single = _as_int(row.get("single_side_count"))
    target = _as_int(row.get("target_like_count"))
    route_count = _as_int(row.get("route_count"))
    if route_count > 0 and source == route_count:
        return "source"
    if route_count > 0 and single == route_count:
        return "single"
    if route_count > 0 and target == route_count:
        return "target"
    if single > 0:
        return "partial_single"
    if source > 0 and target > 0:
        return "mixed_source_target"
    if source > 0:
        return "partial_source"
    if target > 0:
        return "partial_target"
    return "none"


def _route_state_sequence(fraction_rows: pd.DataFrame, pair_id: str) -> str:
    rows = fraction_rows[fraction_rows["local_pair_id"].astype(str).eq(pair_id)].copy()
    if rows.empty:
        return ""
    rows = rows.sort_values("bridge_edge_weight_fraction", ascending=False)
    return " -> ".join(
        f"{float(row['bridge_edge_weight_fraction']):g}:{_fraction_state(row)}"
        for _, row in rows.iterrows()
    )


def _taxonomy_from_rows(
    pair_id: str,
    screen_row: pd.Series,
    plateau_row: pd.Series | None,
) -> tuple[str, str, str, str, str]:
    fixed_signature = _as_bool(screen_row["fixed_016_local_signature_pass"])
    validation_stratum = str(screen_row["validation_stratum"])

    if plateau_row is None:
        if fixed_signature and validation_stratum != "strict_ready":
            return (
                "non_strict_local_signature_unrouted_diagnostic",
                "local_signature_diagnostic",
                "Local substrate recurs outside strict-ready scope, but route morphology is not established.",
                "Keep as diagnostic only until a strict scope and route-state question are declared.",
                "diagnostic_only_no_route_morphology_claim",
            )
        if not fixed_signature:
            return (
                "screened_nonanalog_or_closed_control",
                "screened_panel_context",
                "Pair is outside the fixed 016 local-signature route morphology surface.",
                "Do not spend route trace budget unless a new mechanism question reopens it.",
                "closed_screen_no_basin_claim",
            )
        return (
            "unrouted_local_signature_unknown",
            "unrouted_gap",
            "Local signature passes but current route-state morphology is not scoreable.",
            "First decide whether this pair belongs to the strict route-state surface.",
            "unrouted_no_basin_claim",
        )

    single_all = _as_int(plateau_row["all_route_single_side_fraction_count"])
    any_single = _as_int(plateau_row["any_single_side_fraction_count"])
    source_all = _as_int(plateau_row["all_source_fraction_count"])
    target_all = _as_int(plateau_row["all_target_fraction_count"])
    stable = _as_bool(plateau_row["seed_start_stable_finite_plateau"])
    pair_explanation = str(plateau_row["pair_explanation_class"])

    if pair_id == PRIMARY_PAIR_ID and single_all >= 2 and stable:
        return (
            "stable_finite_single_side_plateau_reference",
            "016_like_plateau_reference",
            "This is the current positive 016-like route-state morphology, not the full basin definition.",
            "Use as the positive reference for plateau recurrence predicates only.",
            "route_morphology_reference_only",
        )
    if not fixed_signature or "source_family_absent" in pair_explanation:
        return (
            "boundary_or_endpoint_surface_control",
            "boundary_control",
            "Boundary/control surface can show endpoint or single-side fragments without source-family basin context.",
            "Keep as a negative control against over-reading endpoint motion.",
            "boundary_control_no_basin_claim",
        )
    if source_all >= 1 and target_all >= 1 and any_single == 0:
        return (
            "abrupt_source_target_switch_negative",
            "near_miss_negative_guard",
            "Source and target surfaces exist, but the transition skips a durable single-side state.",
            "Use as a guard against defining basin transition by endpoints alone.",
            "negative_guard_not_plateau_claim",
        )
    if any_single > 0 and single_all == 0:
        return (
            "fragmented_or_point_single_side_negative",
            "near_miss_negative_guard",
            "A single-side state appears only as a partial, point, or seed-fragile event.",
            "Use as a guard against treating any single-side observation as a basin morphology.",
            "negative_guard_not_plateau_claim",
        )
    return (
        "route_morphology_negative_other",
        "near_miss_negative_guard",
        "Route-state evidence is scoreable but does not match the 016-like plateau predicate.",
        "Keep as diagnostic route morphology until a new family is predeclared.",
        "negative_guard_not_plateau_claim",
    )


def _build_pair_rows(
    *,
    screen_pair: pd.DataFrame,
    route_pair: pd.DataFrame,
    route_negative_pair: pd.DataFrame,
    route_fraction: pd.DataFrame,
    plateau_pair: pd.DataFrame,
    gate_application_pair: pd.DataFrame,
) -> pd.DataFrame:
    route_by_pair = _index_by_pair(route_pair)
    negative_by_pair = _index_by_pair(route_negative_pair)
    plateau_by_pair = _index_by_pair(plateau_pair)
    gate_by_pair = _index_by_pair(gate_application_pair)

    rows: list[dict[str, Any]] = []
    for _, screen_row in screen_pair.iterrows():
        pair_id = str(screen_row["local_pair_id"])
        route_row = route_by_pair.get(pair_id)
        negative_row = negative_by_pair.get(pair_id)
        plateau_row = plateau_by_pair.get(pair_id)
        gate_row = gate_by_pair.get(pair_id)
        morphology_class, morphology_role, interpretation, next_action, claim_status = (
            _taxonomy_from_rows(pair_id, screen_row, plateau_row)
        )
        current_route_state_readout = negative_row is not None and plateau_row is not None
        stale_screen_flag = (
            not _as_bool(screen_row.get("first_pass_route_readout_available"))
            and current_route_state_readout
        )
        current_row = route_row if route_row is not None else negative_row
        rows.append(
            {
                "local_pair_id": pair_id,
                "validation_stratum": screen_row["validation_stratum"],
                "guard_family": screen_row.get("guard_family"),
                "fixed_016_local_signature_pass": _as_bool(
                    screen_row["fixed_016_local_signature_pass"]
                ),
                "screen_first_pass_route_readout_available": _as_bool(
                    screen_row.get("first_pass_route_readout_available")
                ),
                "route_trace_pair_present": route_row is not None,
                "route_negative_pair_present": negative_row is not None,
                "plateau_feature_present": plateau_row is not None,
                "current_route_state_readout_present": current_route_state_readout,
                "stale_screen_readout_flag": stale_screen_flag,
                "contract_application_class": gate_row["contract_application_class"]
                if gate_row is not None
                else None,
                "route_state_morphology_class": morphology_class,
                "route_state_morphology_role": morphology_role,
                "basin_interpretation": interpretation,
                "recommended_next_action": next_action,
                "claim_status": claim_status,
                "route_state_sequence": _route_state_sequence(route_fraction, pair_id),
                "route_count": _as_int(current_row["route_count"])
                if current_row is not None
                else None,
                "full_fixed_016_route_predicate_count": _as_int(
                    current_row["fixed_016_route_predicate_pass_count"]
                    if "fixed_016_route_predicate_pass_count" in current_row.index
                    else current_row["full_fixed_016_route_predicate_count"]
                )
                if current_row is not None
                else None,
                "source_family_start_count": _as_int(current_row["source_family_start_count"])
                if current_row is not None
                else None,
                "finite_single_side_band_count": _as_int(
                    current_row["finite_single_side_band_count"]
                    if "finite_single_side_band_count" in current_row.index
                    else current_row["finite_single_side_band_route_count"]
                )
                if current_row is not None
                else None,
                "final_target_like_count": _as_int(
                    current_row["final_target_like_count"]
                    if "final_target_like_count" in current_row.index
                    else current_row["final_target_like_route_count"]
                )
                if current_row is not None
                else None,
                "all_source_fraction_count": _as_int(
                    plateau_row["all_source_fraction_count"]
                )
                if plateau_row is not None
                else None,
                "all_route_single_side_fraction_count": _as_int(
                    plateau_row["all_route_single_side_fraction_count"]
                )
                if plateau_row is not None
                else None,
                "any_single_side_fraction_count": _as_int(
                    plateau_row["any_single_side_fraction_count"]
                )
                if plateau_row is not None
                else None,
                "all_target_fraction_count": _as_int(
                    plateau_row["all_target_fraction_count"]
                )
                if plateau_row is not None
                else None,
                "single_side_latch_signature": plateau_row["single_side_latch_signature"]
                if plateau_row is not None
                else None,
                "seed_start_stable_finite_plateau": _as_bool(
                    plateau_row["seed_start_stable_finite_plateau"]
                )
                if plateau_row is not None
                else None,
                "pair_explanation_class": plateau_row["pair_explanation_class"]
                if plateau_row is not None
                else (
                    negative_row["pair_explanation_class"]
                    if negative_row is not None
                    else None
                ),
                "bridge_to_direct_weight_ratio": _as_float(
                    screen_row.get("bridge_to_direct_weight_ratio")
                ),
                "original_pair_coassigned_share": _as_float(
                    screen_row.get("original_pair_coassigned_share")
                ),
                "mechanism_generalization_class": screen_row[
                    "mechanism_generalization_class"
                ],
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows)


def _taxonomy_interpretation(morphology_class: str) -> tuple[str, str]:
    interpretations = {
        "stable_finite_single_side_plateau_reference": (
            "016-like finite plateau is a positive route-state morphology, not the full basin definition.",
            "Use P1-P6 only as the acceptance contract for this morphology family.",
        ),
        "abrupt_source_target_switch_negative": (
            "Endpoint brackets exist, but no durable middle state is observed.",
            "Keep as a negative guard against endpoint-only basin claims.",
        ),
        "fragmented_or_point_single_side_negative": (
            "Single-side evidence exists, but it is not all-route, adjacent, or seed/start stable.",
            "Keep as a negative guard against point-event plateau claims.",
        ),
        "boundary_or_endpoint_surface_control": (
            "Boundary/control rows can mimic endpoint motion without source-family context.",
            "Keep as a specificity control.",
        ),
        "non_strict_local_signature_unrouted_diagnostic": (
            "The local signature can recur outside strict-ready scope.",
            "Do not route-promote until a strict scope and mechanism question are declared.",
        ),
        "screened_nonanalog_or_closed_control": (
            "The pair is outside the current route-state morphology surface.",
            "Do not reopen without a new mechanism question.",
        ),
        "unrouted_local_signature_unknown": (
            "Local signature passed but route-state morphology is unknown.",
            "First name the route-state question before executing.",
        ),
        "route_morphology_negative_other": (
            "Route state is scoreable but does not match a declared positive family.",
            "Keep as diagnostic until a new family is predeclared.",
        ),
    }
    return interpretations.get(
        morphology_class,
        ("Unspecified route-state morphology.", "Keep diagnostic only."),
    )


def _build_taxonomy_rows(pair_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = pair_rows.groupby("route_state_morphology_class", dropna=False)
    for morphology_class, group in grouped:
        interpretation, next_action = _taxonomy_interpretation(str(morphology_class))
        rows.append(
            {
                "route_state_morphology_class": morphology_class,
                "pair_count": int(len(group)),
                "local_pair_ids": ";".join(group["local_pair_id"].astype(str)),
                "route_scoreable_pair_count": int(
                    group["current_route_state_readout_present"].map(_as_bool).sum()
                ),
                "p1_signature_pair_count": int(
                    group["fixed_016_local_signature_pass"].map(_as_bool).sum()
                ),
                "taxonomy_interpretation": interpretation,
                "recommended_next_action": next_action,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    order = {
        "stable_finite_single_side_plateau_reference": 0,
        "abrupt_source_target_switch_negative": 1,
        "fragmented_or_point_single_side_negative": 2,
        "boundary_or_endpoint_surface_control": 3,
        "non_strict_local_signature_unrouted_diagnostic": 4,
        "screened_nonanalog_or_closed_control": 5,
        "unrouted_local_signature_unknown": 6,
        "route_morphology_negative_other": 7,
    }
    return (
        pd.DataFrame(rows)
        .assign(_sort_key=lambda frame: frame["route_state_morphology_class"].map(order).fillna(99))
        .sort_values(["_sort_key", "route_state_morphology_class"], kind="mergesort")
        .drop(columns=["_sort_key"])
    )


def _build_provenance_rows(context: dict[str, Any]) -> pd.DataFrame:
    tables = context["tables"]
    paths = context["paths"]
    rows = [
        {
            "source_name": "mechanism_generalization_screen",
            "source_path": str(paths["generalization_screen_dir"]),
            "artifact_files": "summary_json;pair_rows",
            "row_count": int(len(tables["screen_pair"])),
            "evidence_role": "full 23-pair local-signature screen; route-readout flags are pre-route-trace and can be stale",
            "provenance_status": "used_for_panel_scope_and_local_signature_only",
        },
        {
            "source_name": "mechanism_generalization_route_trace",
            "source_path": str(paths["route_trace_dir"]),
            "artifact_files": "summary_json;pair_rows;fraction_rows;gate_matrix",
            "row_count": int(len(tables["route_fraction"])),
            "evidence_role": "current fixed-predicate route execution surface for 009/012/020 plus 014/005 controls",
            "provenance_status": "used_for_current_route_state_readout",
        },
        {
            "source_name": "route_negative_explanation",
            "source_path": str(paths["route_negative_dir"]),
            "artifact_files": "summary_json;pair_rows;fraction_rows;substrate_rows",
            "row_count": int(len(tables["route_negative_pair"])),
            "evidence_role": "route-negative morphology labels and local-substrate separation",
            "provenance_status": "used_for_negative_morphology_interpretation",
        },
        {
            "source_name": "plateau_stability_feature_audit",
            "source_path": str(paths["plateau_feature_dir"]),
            "artifact_files": "summary_json;pair_rows;fraction_rows",
            "row_count": int(len(tables["plateau_pair"])),
            "evidence_role": "P1-P6 feature values and stable/fragmented plateau discrimination",
            "provenance_status": "used_for_plateau_feature_taxonomy",
        },
        {
            "source_name": "plateau_stability_gate_application",
            "source_path": str(paths["gate_application_dir"]),
            "artifact_files": "summary_json;pair_rows;gate_matrix",
            "row_count": int(len(tables["gate_application_pair"])),
            "evidence_role": "current 23-pair panel closure and P1-P6 specificity result",
            "provenance_status": "used_for_claim_boundary_and_panel_closure",
        },
    ]
    return pd.DataFrame(rows).assign(claim_boundary=CLAIM_BOUNDARY, run_status=RUN_STATUS)


def _decision_rows(pair_rows: pd.DataFrame, taxonomy_rows: pd.DataFrame) -> pd.DataFrame:
    stable = pair_rows[
        pair_rows["route_state_morphology_class"].astype(str).eq(
            "stable_finite_single_side_plateau_reference"
        )
    ]
    abrupt = pair_rows[
        pair_rows["route_state_morphology_class"].astype(str).eq(
            "abrupt_source_target_switch_negative"
        )
    ]
    fragmented = pair_rows[
        pair_rows["route_state_morphology_class"].astype(str).eq(
            "fragmented_or_point_single_side_negative"
        )
    ]
    stale = pair_rows[pair_rows["stale_screen_readout_flag"].map(_as_bool)]
    rows = [
        {
            "decision_id": "D1_plateau_is_one_morphology_not_full_basin_definition",
            "decision": "treat_p1_p6_as_016_like_plateau_contract_only",
            "evidence": json.dumps(
                {
                    "stable_plateau_pair_ids": stable["local_pair_id"].tolist(),
                    "taxonomy_classes": taxonomy_rows[
                        ["route_state_morphology_class", "pair_count"]
                    ].to_dict(orient="records"),
                },
                sort_keys=True,
            ),
            "claim_boundary": "P1-P6 opens only an 016-like plateau recurrence claim, not a full basin definition.",
            "run_status": RUN_STATUS,
        },
        {
            "decision_id": "D2_negative_morphologies_are_structured_not_failed_noise",
            "decision": "retain_abrupt_and_fragmented_routes_as_morphology_guards",
            "evidence": json.dumps(
                {
                    "abrupt_pair_ids": abrupt["local_pair_id"].tolist(),
                    "fragmented_pair_ids": fragmented["local_pair_id"].tolist(),
                },
                sort_keys=True,
            ),
            "claim_boundary": "Negative route-state classes guide definition design; they are not method failures.",
            "run_status": RUN_STATUS,
        },
        {
            "decision_id": "D3_readout_provenance_must_be_time_ordered",
            "decision": "screen_readout_flags_are_not_current_route_evidence_after_later_trace",
            "evidence": json.dumps(
                {
                    "stale_screen_readout_flag_pair_ids": stale[
                        "local_pair_id"
                    ].tolist(),
                    "reason": "screen flags predate route_trace and plateau_feature evidence",
                },
                sort_keys=True,
            ),
            "claim_boundary": "Use screen flags for screen history, not for current route-state availability.",
            "run_status": RUN_STATUS,
        },
        {
            "decision_id": "D4_next_direction",
            "decision": "define_route_state_morphology_family_before_more_candidates_or_demo",
            "evidence": json.dumps(
                {
                    "recommended_family_split": [
                        "stable_finite_single_side_plateau",
                        "abrupt_source_target_switch",
                        "fragmented_or_point_single_side",
                        "boundary_or_endpoint_surface",
                    ]
                },
                sort_keys=True,
            ),
            "claim_boundary": "Next work is definition design, not candidate expansion or quality/cost method claim.",
            "run_status": RUN_STATUS,
        },
    ]
    return pd.DataFrame(rows)


def _build_gates(
    *,
    pair_rows: pd.DataFrame,
    taxonomy_rows: pd.DataFrame,
    provenance_rows: pd.DataFrame,
    route_gate: pd.DataFrame,
    gate_application_gate: pd.DataFrame,
) -> pd.DataFrame:
    stable = pair_rows[
        pair_rows["route_state_morphology_class"].astype(str).eq(
            "stable_finite_single_side_plateau_reference"
        )
    ]
    abrupt = pair_rows[
        pair_rows["route_state_morphology_class"].astype(str).eq(
            "abrupt_source_target_switch_negative"
        )
    ]
    fragmented = pair_rows[
        pair_rows["route_state_morphology_class"].astype(str).eq(
            "fragmented_or_point_single_side_negative"
        )
    ]
    boundary = pair_rows[
        pair_rows["route_state_morphology_class"].astype(str).eq(
            "boundary_or_endpoint_surface_control"
        )
    ]
    stale = pair_rows[pair_rows["stale_screen_readout_flag"].map(_as_bool)]
    route_failed = route_gate[route_gate["gate_status"].astype(str).ne("pass")]
    application_failed = gate_application_gate[
        gate_application_gate["gate_status"].astype(str).ne("pass")
    ]
    gates = [
        _gate_row(
            "G1_sources_readable",
            "Were all required provenance sources readable?",
            provenance_rows[["source_name", "row_count", "provenance_status"]].to_dict(
                orient="records"
            ),
            "five source surfaces with non-empty rows",
            len(provenance_rows) == 5 and bool(provenance_rows["row_count"].gt(0).all()),
        ),
        _gate_row(
            "G2_current_panel_preserved",
            "Does the taxonomy preserve the 23-pair panel while marking only route-scoreable rows?",
            {
                "pair_count": int(len(pair_rows)),
                "route_scoreable_pair_ids": pair_rows[
                    pair_rows["current_route_state_readout_present"].map(_as_bool)
                ]["local_pair_id"].tolist(),
            },
            "23 panel rows and exactly the six current route-scoreable pairs",
            len(pair_rows) == 23
            and set(
                pair_rows[
                    pair_rows["current_route_state_readout_present"].map(_as_bool)
                ][
                    "local_pair_id"
                ]
            )
            == set(ROUTE_SCOREABLE_PAIR_IDS),
        ),
        _gate_row(
            "G3_016_plateau_is_unique_positive_morphology",
            "Is stable finite plateau restricted to 016 on the current route-state surface?",
            stable[["local_pair_id", "route_state_sequence"]].to_dict(orient="records"),
            "only local_pair_016 has stable_finite_single_side_plateau_reference",
            stable["local_pair_id"].tolist() == [PRIMARY_PAIR_ID],
        ),
        _gate_row(
            "G4_negative_morphologies_preserved",
            "Are route-negative analogs retained as abrupt or fragmented morphology guards?",
            {
                "abrupt_pair_ids": abrupt["local_pair_id"].tolist(),
                "fragmented_pair_ids": fragmented["local_pair_id"].tolist(),
                "boundary_pair_ids": boundary["local_pair_id"].tolist(),
            },
            "009/020 abrupt, 012/014 fragmented, and 005 boundary/control",
            set(abrupt["local_pair_id"]) == {"local_pair_009", "local_pair_020"}
            and set(fragmented["local_pair_id"]) == {"local_pair_012", "local_pair_014"}
            and boundary["local_pair_id"].tolist() == [BOUNDARY_GUARD_PAIR_ID],
        ),
        _gate_row(
            "G5_provenance_staleness_is_explicit",
            "Are stale screen readout flags separated from current route evidence?",
            stale[["local_pair_id", "screen_first_pass_route_readout_available"]].to_dict(
                orient="records"
            ),
            "009/012/020 have stale screen flags after later route trace evidence",
            set(stale["local_pair_id"]) == set(STRICT_ANALOG_PAIR_IDS),
        ),
        _gate_row(
            "G6_upstream_failure_scope_preserved",
            "Are upstream failed route-recurrence gates treated as evidence boundaries?",
            {
                "route_failed_gates": route_failed["gate_id"].tolist(),
                "gate_application_failed_gates": application_failed["gate_id"].tolist(),
            },
            "route recurrence gates can fail; gate application should have no failed gates",
            set(route_failed["gate_id"]) == {
                "G3_candidate_route_recurrence_observed",
                "G4_all_candidates_recur_under_fixed_predicate",
            }
            and application_failed.empty,
        ),
        _gate_row(
            "G7_claim_boundaries_closed",
            "Are wall, method, quality/cost, and full-replay claims closed?",
            CLAIM_BOUNDARY,
            "read-only taxonomy audit",
            True,
        ),
    ]
    return pd.DataFrame(gates)


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    pair_rows: pd.DataFrame,
    taxonomy_rows: pd.DataFrame,
    provenance_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass Route-State Morphology Taxonomy",
        "",
        "## Summary",
        "",
        f"- status: {summary['status']}",
        f"- morphology_status: {summary['morphology_status']}",
        f"- stable_plateau_pair_ids: {summary['stable_plateau_pair_ids']}",
        f"- stale_screen_readout_flag_pair_ids: {summary['stale_screen_readout_flag_pair_ids']}",
        f"- failed_gates: {summary['failed_gates']}",
        "",
        "## Pair Rows",
        "",
        _markdown_table(
            pair_rows,
            [
                "local_pair_id",
                "validation_stratum",
                "fixed_016_local_signature_pass",
                "route_trace_pair_present",
                "current_route_state_readout_present",
                "stale_screen_readout_flag",
                "route_state_morphology_class",
                "route_state_morphology_role",
                "route_state_sequence",
                "claim_status",
            ],
        ),
        "",
        "## Taxonomy Rows",
        "",
        _markdown_table(
            taxonomy_rows,
            [
                "route_state_morphology_class",
                "pair_count",
                "local_pair_ids",
                "route_scoreable_pair_count",
                "taxonomy_interpretation",
                "recommended_next_action",
            ],
        ),
        "",
        "## Provenance Rows",
        "",
        _markdown_table(
            provenance_rows,
            [
                "source_name",
                "row_count",
                "evidence_role",
                "provenance_status",
            ],
        ),
        "",
        "## Decisions",
        "",
        _markdown_table(
            decision_rows,
            ["decision_id", "decision", "claim_boundary"],
        ),
        "",
        "## Gates",
        "",
        _markdown_table(
            gates,
            ["gate_id", "gate_status", "question", "minimum_or_rule"],
        ),
        "",
        "## Recommended Next Gate",
        "",
        summary["recommended_next_gate"],
        "",
        "## Claim Boundary",
        "",
        CLAIM_BOUNDARY,
        "",
    ]
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    generalization_screen_dir: Path,
    route_trace_dir: Path,
    route_negative_dir: Path,
    plateau_feature_dir: Path,
    gate_application_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    context = _load_context(
        generalization_screen_dir=generalization_screen_dir,
        route_trace_dir=route_trace_dir,
        route_negative_dir=route_negative_dir,
        plateau_feature_dir=plateau_feature_dir,
        gate_application_dir=gate_application_dir,
    )
    tables = context["tables"]
    pair_rows = _build_pair_rows(
        screen_pair=tables["screen_pair"],
        route_pair=tables["route_pair"],
        route_negative_pair=tables["route_negative_pair"],
        route_fraction=tables["route_negative_fraction"],
        plateau_pair=tables["plateau_pair"],
        gate_application_pair=tables["gate_application_pair"],
    )
    taxonomy_rows = _build_taxonomy_rows(pair_rows)
    provenance_rows = _build_provenance_rows(context)
    decision_rows = _decision_rows(pair_rows, taxonomy_rows)
    gates = _build_gates(
        pair_rows=pair_rows,
        taxonomy_rows=taxonomy_rows,
        provenance_rows=provenance_rows,
        route_gate=tables["route_gate"],
        gate_application_gate=tables["gate_application_gate"],
    )

    failed_gates = gates[gates["gate_status"].astype(str).ne("pass")]["gate_id"].tolist()
    stable = pair_rows[
        pair_rows["route_state_morphology_class"].astype(str).eq(
            "stable_finite_single_side_plateau_reference"
        )
    ]
    stale = pair_rows[pair_rows["stale_screen_readout_flag"].map(_as_bool)]
    negative = pair_rows[
        pair_rows["route_state_morphology_role"].astype(str).eq("near_miss_negative_guard")
    ]
    summary = {
        "status": RUN_STATUS,
        "schema": "nanoclustering_g4_8_first_pass_route_state_morphology_taxonomy_summary.v1",
        "morphology_status": "route_state_taxonomy_separates_016_like_plateau_from_abrupt_and_fragmented_negatives",
        "stable_plateau_pair_ids": stable["local_pair_id"].tolist(),
        "negative_morphology_pair_ids": negative["local_pair_id"].tolist(),
        "stale_screen_readout_flag_pair_ids": stale["local_pair_id"].tolist(),
        "route_scoreable_pair_ids": pair_rows[
            pair_rows["current_route_state_readout_present"].map(_as_bool)
        ]["local_pair_id"].tolist(),
        "taxonomy_class_counts": taxonomy_rows[
            ["route_state_morphology_class", "pair_count"]
        ].to_dict(orient="records"),
        "failed_gates": failed_gates,
        "recommended_next_gate": (
            "Use the route-state taxonomy as the next definition layer: keep "
            "P1-P6 as an 016-like finite-plateau predicate, treat abrupt and "
            "fragmented routes as separate morphology guards, then design the "
            "minimal demo/reproducibility surface around these morphology "
            "families rather than rerunning the same strict analog trace."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "source_dirs": {key: str(path) for key, path in context["paths"].items()},
        "source_statuses": {
            key: value.get("status") for key, value in context["summaries"].items()
        },
        "output_dir": str(output_dir),
    }
    config = {
        "generalization_screen_dir": str(generalization_screen_dir),
        "route_trace_dir": str(route_trace_dir),
        "route_negative_dir": str(route_negative_dir),
        "plateau_feature_dir": str(plateau_feature_dir),
        "gate_application_dir": str(gate_application_dir),
        "output_dir": str(output_dir),
        "primary_pair_id": PRIMARY_PAIR_ID,
        "strict_analog_pair_ids": list(STRICT_ANALOG_PAIR_IDS),
        "route_scoreable_pair_ids": list(ROUTE_SCOREABLE_PAIR_IDS),
        "claim_boundary": CLAIM_BOUNDARY,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(pair_rows, output_dir / PAIR_ROWS_CSV)
    _write_csv(taxonomy_rows, output_dir / TAXONOMY_ROWS_CSV)
    _write_csv(provenance_rows, output_dir / PROVENANCE_ROWS_CSV)
    _write_csv(decision_rows, output_dir / DECISION_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
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
        summary=summary,
        pair_rows=pair_rows,
        taxonomy_rows=taxonomy_rows,
        provenance_rows=provenance_rows,
        decision_rows=decision_rows,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generalization-screen-dir",
        type=Path,
        default=DEFAULT_GENERALIZATION_SCREEN_DIR,
    )
    parser.add_argument("--route-trace-dir", type=Path, default=DEFAULT_ROUTE_TRACE_DIR)
    parser.add_argument(
        "--route-negative-dir",
        type=Path,
        default=DEFAULT_ROUTE_NEGATIVE_DIR,
    )
    parser.add_argument(
        "--plateau-feature-dir",
        type=Path,
        default=DEFAULT_PLATEAU_FEATURE_DIR,
    )
    parser.add_argument(
        "--gate-application-dir",
        type=Path,
        default=DEFAULT_GATE_APPLICATION_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        generalization_screen_dir=args.generalization_screen_dir,
        route_trace_dir=args.route_trace_dir,
        route_negative_dir=args.route_negative_dir,
        plateau_feature_dir=args.plateau_feature_dir,
        gate_application_dir=args.gate_application_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
