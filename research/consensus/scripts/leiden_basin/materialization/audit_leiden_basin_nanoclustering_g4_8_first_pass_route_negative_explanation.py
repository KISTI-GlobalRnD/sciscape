#!/usr/bin/env python3
"""Explain why strict local-signature analogs fail the fixed 016 route predicate.

This read-only audit compares the positive ``016`` persistence trace against
the route-negative strict analog queue (``009``, ``012``, ``020``) and controls
(``014``, ``005``). It separates local substrate recurrence from fractional
route morphology so we do not mistake a coarse local-ablation signature for a
route-level mechanism.
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
CANDIDATE_PAIR_IDS = ("local_pair_009", "local_pair_012", "local_pair_020")
REFERENCE_PAIR_ID = "local_pair_014"
BOUNDARY_GUARD_PAIR_ID = "local_pair_005"
CONTROL_PAIR_IDS = (REFERENCE_PAIR_ID, BOUNDARY_GUARD_PAIR_ID)
AUDIT_PAIR_IDS = (*CANDIDATE_PAIR_IDS, REFERENCE_PAIR_ID, PRIMARY_PAIR_ID, BOUNDARY_GUARD_PAIR_ID)

SOURCE_FAMILY_MECHANISMS = {
    "pair_coassigned_with_selected_bridge",
    "pair_separated_bridge_split",
}
SINGLE_SIDE_MECHANISM = "pair_separated_single_side_bridge"
TARGET_LIKE_MECHANISM = "pair_coassigned_without_selected_bridge"

DEFAULT_LOCAL_ABLATION_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_symmetric_object_variable_pair_local_ablation_gamma1e5_20260603"
)
DEFAULT_GENERALIZATION_SCREEN_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_mechanism_generalization_screen_gamma1e5_20260605"
)
DEFAULT_016_PERSISTENCE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_transient_persistence_trace_gamma1e5_20260605"
)
DEFAULT_ROUTE_TRACE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_gamma1e5_20260605"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_route_negative_explanation_audit_gamma1e5_20260605"
)

SUBSTRATE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_route_negative_explanation_substrate_rows.csv"
)
PAIR_EXPLANATION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_route_negative_explanation_pair_rows.csv"
)
FRACTION_PROFILE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_route_negative_explanation_fraction_rows.csv"
)
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_route_negative_explanation_decision_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_route_negative_explanation_gate_matrix.csv"
)
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_route_negative_explanation_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_route_negative_explanation_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_route_negative_explanation_report.md"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_route_negative_explanation"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_route_negative_explanation"
WALL_PROMOTION_STATUS = "not_promoted_route_negative_explanation_only"
METHOD_STATUS = "route_negative_explanation_audit_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass route-negative explanation audit only; "
    "reads local-ablation, 016 persistence, mechanism-generalization screen, "
    "and fixed-predicate route-trace artifacts. It does not rerun Leiden, "
    "promote basin walls, replay full NanoClustering, evaluate quality/cost "
    "value, or claim method success."
)

EPS = 1e-9


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


def _load_context(
    *,
    local_ablation_dir: Path,
    generalization_screen_dir: Path,
    persistence_016_dir: Path,
    route_trace_dir: Path,
) -> dict[str, Any]:
    return {
        "paths": {
            "local_ablation_dir": local_ablation_dir,
            "generalization_screen_dir": generalization_screen_dir,
            "persistence_016_dir": persistence_016_dir,
            "route_trace_dir": route_trace_dir,
        },
        "summaries": {
            "generalization_screen": _read_json(
                generalization_screen_dir
                / "nanoclustering_g4_8_first_pass_mechanism_generalization_summary.json"
            ),
            "persistence_016": _read_json(
                persistence_016_dir
                / "nanoclustering_g4_8_first_pass_016_transient_persistence_summary.json"
            ),
            "route_trace": _read_json(
                route_trace_dir
                / "nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_summary.json"
            ),
            "local_ablation": _read_json(
                local_ablation_dir
                / "nanoclustering_symmetric_object_variable_pair_local_ablation_summary.json"
            ),
        },
        "tables": {
            "local_graph": _read_csv(
                local_ablation_dir
                / "nanoclustering_symmetric_object_variable_pair_local_ablation_graph_rows.csv"
            ),
            "pair_gate": _read_csv(
                local_ablation_dir
                / "nanoclustering_symmetric_object_variable_pair_local_ablation_pair_gate_rows.csv"
            ),
            "generalization_pair": _read_csv(
                generalization_screen_dir
                / "nanoclustering_g4_8_first_pass_mechanism_generalization_pair_rows.csv"
            ),
            "persistence_016_trace": _read_csv(
                persistence_016_dir
                / "nanoclustering_g4_8_first_pass_016_transient_persistence_trace_rows.csv"
            ),
            "route_trace": _read_csv(
                route_trace_dir
                / "nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_rows.csv"
            ),
            "route_pair": _read_csv(
                route_trace_dir
                / "nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_pair_rows.csv"
            ),
            "route_gate": _read_csv(
                route_trace_dir
                / "nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_gate_matrix.csv"
            ),
        },
    }


def _pair_role(pair_id: str) -> str:
    if pair_id == PRIMARY_PAIR_ID:
        return "primary_016_finite_band_reference"
    if pair_id in CANDIDATE_PAIR_IDS:
        return "strict_nonboundary_local_signature_analog"
    if pair_id == REFERENCE_PAIR_ID:
        return "positive_reference_control"
    if pair_id == BOUNDARY_GUARD_PAIR_ID:
        return "boundary_guard_control"
    return "other"


def _substrate_rows(context: dict[str, Any]) -> pd.DataFrame:
    graph = context["tables"]["local_graph"]
    gate = context["tables"]["pair_gate"]
    screen = context["tables"]["generalization_pair"]
    rows: list[dict[str, Any]] = []
    for pair_id in AUDIT_PAIR_IDS:
        graph_row = graph[graph["local_pair_id"].astype(str).eq(pair_id)].iloc[0].to_dict()
        gate_row = gate[gate["local_pair_id"].astype(str).eq(pair_id)].iloc[0].to_dict()
        screen_match = screen[screen["local_pair_id"].astype(str).eq(pair_id)]
        screen_row = screen_match.iloc[0].to_dict() if not screen_match.empty else {}
        rows.append(
            {
                "local_pair_id": pair_id,
                "pair_role": _pair_role(pair_id),
                "branch": graph_row.get("branch"),
                "pair_scope": graph_row.get("pair_scope"),
                "counterfactual_class": graph_row.get("counterfactual_class"),
                "direct_cpm_delta_q": float(graph_row.get("direct_cpm_delta_q")),
                "direct_edge_weight": float(graph_row.get("direct_edge_weight")),
                "bridge_to_direct_weight_ratio": float(
                    graph_row.get("bridge_to_direct_weight_ratio")
                ),
                "selected_bridge_count": int(graph_row.get("selected_bridge_count")),
                "local_node_count": int(graph_row.get("local_node_count")),
                "local_gate_class": gate_row.get("gate_class"),
                "local_gate_status": gate_row.get("gate_status"),
                "original_pair_coassigned_share": float(
                    gate_row.get("original_pair_coassigned_share")
                ),
                "drop_direct_pair_coassigned_share": float(
                    gate_row.get("drop_direct_pair_coassigned_share")
                ),
                "drop_bridge_pair_coassigned_share": float(
                    gate_row.get("drop_bridge_pair_coassigned_share")
                ),
                "fixed_016_local_signature_pass": _as_bool(
                    screen_row.get(
                        "fixed_016_local_signature_pass",
                        pair_id == PRIMARY_PAIR_ID,
                    )
                ),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows)


def _combined_trace_rows(context: dict[str, Any]) -> pd.DataFrame:
    trace_016 = context["tables"]["persistence_016_trace"].copy()
    trace_016["contract_pair_role"] = "primary_016_finite_band_reference"
    route_trace = context["tables"]["route_trace"].copy()
    combined = pd.concat([trace_016, route_trace], ignore_index=True, sort=False)
    combined = combined[combined["local_pair_id"].astype(str).isin(AUDIT_PAIR_IDS)].copy()
    combined["pair_role"] = combined["local_pair_id"].astype(str).map(_pair_role)
    combined["is_source_family"] = combined["mechanism_read"].astype(str).isin(
        SOURCE_FAMILY_MECHANISMS
    )
    combined["is_single_side"] = combined["mechanism_read"].astype(str).eq(
        SINGLE_SIDE_MECHANISM
    )
    combined["is_target_like"] = combined["mechanism_read"].astype(str).eq(
        TARGET_LIKE_MECHANISM
    )
    return combined


def _fraction_profile_rows(trace_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in trace_rows.groupby(
        ["local_pair_id", "pair_role", "bridge_edge_weight_fraction"],
        sort=False,
    ):
        pair_id, pair_role, fraction = keys
        mechanism_counts = group["mechanism_read"].astype(str).value_counts().to_dict()
        rows.append(
            {
                "local_pair_id": pair_id,
                "pair_role": pair_role,
                "bridge_edge_weight_fraction": float(fraction),
                "trace_row_count": int(len(group)),
                "route_count": int(
                    group[["route_contract_id", "seed"]].drop_duplicates().shape[0]
                ),
                "source_family_count": int(group["is_source_family"].sum()),
                "single_side_count": int(group["is_single_side"].sum()),
                "target_like_count": int(group["is_target_like"].sum()),
                "source_family_all_routes": bool(group["is_source_family"].all()),
                "single_side_all_routes": bool(group["is_single_side"].all()),
                "target_like_all_routes": bool(group["is_target_like"].all()),
                "dominant_mechanism_read": max(
                    mechanism_counts.items(), key=lambda item: item[1]
                )[0]
                if mechanism_counts
                else "",
                "mechanism_read_counts": mechanism_counts,
                "objective_value_mean": float(group["objective_value_by_step"].mean()),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["local_pair_id", "bridge_edge_weight_fraction"],
        ascending=[True, False],
        kind="mergesort",
    )


def _route_predicate_rows(trace_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "route_contract_id",
        "local_pair_id",
        "pair_role",
        "start_condition",
        "seed",
    ]
    for keys, group in trace_rows.groupby(group_cols, sort=False):
        route_contract_id, pair_id, pair_role, start, seed = keys
        ordered = group.sort_values("step_index", kind="mergesort")
        first = ordered.iloc[0]
        final = ordered.iloc[-1]
        single = ordered[ordered["is_single_side"]]
        single_steps = [int(value) for value in single["step_index"].tolist()]
        single_fractions = [
            float(value) for value in single["bridge_edge_weight_fraction"].tolist()
        ]
        adjacent = (
            len(single_steps) >= 2
            and max(single_steps) - min(single_steps) == len(single_steps) - 1
        )
        source_start = _as_bool(first["is_source_family"])
        final_target = _as_bool(final["is_target_like"])
        full = bool(source_start and adjacent and final_target)
        if full:
            route_class = "full_fixed_016_route_predicate"
        elif not source_start:
            route_class = "source_family_start_absent"
        elif single.empty and final_target:
            route_class = "source_to_target_without_single_side_band"
        elif not single.empty and not adjacent:
            route_class = "point_or_fragmented_single_side_not_band"
        elif not single.empty and not final_target:
            route_class = "single_side_without_target_final"
        elif not final_target:
            route_class = "target_final_absent"
        else:
            route_class = "other_route_negative"
        rows.append(
            {
                "route_contract_id": route_contract_id,
                "local_pair_id": pair_id,
                "pair_role": pair_role,
                "start_condition": start,
                "seed": int(seed),
                "source_family_start": source_start,
                "single_side_fraction_count": int(len(single)),
                "single_side_fractions": ";".join(f"{value:.5g}" for value in single_fractions),
                "single_side_adjacent_fraction_band": bool(adjacent),
                "final_target_like": final_target,
                "full_fixed_016_route_predicate": full,
                "route_explanation_class": route_class,
            }
        )
    return pd.DataFrame(rows)


def _pair_explanation_class(
    *,
    pair_id: str,
    role: str,
    route_count: int,
    full_count: int,
    source_count: int,
    band_count: int,
    target_count: int,
    any_single_route_count: int,
    all_route_single_side_fraction_count: int,
    point_or_fragmented_count: int,
    no_band_target_count: int,
    source_absent_count: int,
    fraction_rows: pd.DataFrame,
) -> str:
    if pair_id == PRIMARY_PAIR_ID and full_count == route_count and all_route_single_side_fraction_count >= 2:
        return "reference_finite_single_side_band_route_mechanism"
    if source_absent_count == route_count:
        return "source_family_absent_boundary_or_target_surface"
    if full_count > 0:
        return "unexpected_partial_or_full_fixed_route_recurrence"
    if any_single_route_count > 0:
        return "point_or_seed_fragile_single_side_not_finite_band"
    if source_count == route_count and target_count == route_count:
        source_all_fractions = set(
            fraction_rows[
                fraction_rows["source_family_all_routes"].map(_as_bool)
            ]["bridge_edge_weight_fraction"].astype(float)
        )
        if 0.625 in source_all_fractions and 0.5 not in source_all_fractions:
            return "delayed_abrupt_source_to_target_switch_without_band"
        return "abrupt_source_to_target_switch_without_band"
    if no_band_target_count > 0:
        return "mixed_source_target_motion_without_single_side_band"
    return "unexplained_route_negative"


def _pair_explanation_rows(
    *,
    substrate_rows: pd.DataFrame,
    fraction_rows: pd.DataFrame,
    route_predicates: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair_id in AUDIT_PAIR_IDS:
        role = _pair_role(pair_id)
        pair_routes = route_predicates[route_predicates["local_pair_id"].astype(str).eq(pair_id)]
        pair_fractions = fraction_rows[fraction_rows["local_pair_id"].astype(str).eq(pair_id)]
        substrate = substrate_rows[substrate_rows["local_pair_id"].astype(str).eq(pair_id)].iloc[0]
        route_count = int(len(pair_routes))
        full_count = int(pair_routes["full_fixed_016_route_predicate"].map(_as_bool).sum())
        source_count = int(pair_routes["source_family_start"].map(_as_bool).sum())
        band_count = int(pair_routes["single_side_adjacent_fraction_band"].map(_as_bool).sum())
        target_count = int(pair_routes["final_target_like"].map(_as_bool).sum())
        any_single_route_count = int(pair_routes["single_side_fraction_count"].astype(int).gt(0).sum())
        point_or_fragmented_count = int(
            pair_routes["route_explanation_class"]
            .astype(str)
            .eq("point_or_fragmented_single_side_not_band")
            .sum()
        )
        no_band_target_count = int(
            pair_routes["route_explanation_class"]
            .astype(str)
            .eq("source_to_target_without_single_side_band")
            .sum()
        )
        source_absent_count = int(
            pair_routes["route_explanation_class"].astype(str).eq("source_family_start_absent").sum()
        )
        all_route_single_side_fraction_count = int(
            pair_fractions["single_side_all_routes"].map(_as_bool).sum()
        )
        explanation = _pair_explanation_class(
            pair_id=pair_id,
            role=role,
            route_count=route_count,
            full_count=full_count,
            source_count=source_count,
            band_count=band_count,
            target_count=target_count,
            any_single_route_count=any_single_route_count,
            all_route_single_side_fraction_count=all_route_single_side_fraction_count,
            point_or_fragmented_count=point_or_fragmented_count,
            no_band_target_count=no_band_target_count,
            source_absent_count=source_absent_count,
            fraction_rows=pair_fractions,
        )
        rows.append(
            {
                "local_pair_id": pair_id,
                "pair_role": role,
                "fixed_016_local_signature_pass": _as_bool(
                    substrate["fixed_016_local_signature_pass"]
                ),
                "bridge_to_direct_weight_ratio": float(
                    substrate["bridge_to_direct_weight_ratio"]
                ),
                "original_pair_coassigned_share": float(
                    substrate["original_pair_coassigned_share"]
                ),
                "route_count": route_count,
                "full_fixed_016_route_predicate_count": full_count,
                "source_family_start_count": source_count,
                "finite_single_side_band_route_count": band_count,
                "any_single_side_route_count": any_single_route_count,
                "all_route_single_side_fraction_count": all_route_single_side_fraction_count,
                "final_target_like_route_count": target_count,
                "route_explanation_class_counts": pair_routes[
                    "route_explanation_class"
                ].value_counts().to_dict(),
                "pair_explanation_class": explanation,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows)


def _decision_rows(pair_rows: pd.DataFrame) -> pd.DataFrame:
    candidates = pair_rows[
        pair_rows["pair_role"].astype(str).eq("strict_nonboundary_local_signature_analog")
    ]
    return pd.DataFrame(
        [
            {
                "decision_id": "D1_local_signature_not_sufficient",
                "decision": "fixed_local_signature_recurrence_does_not_imply_route_mechanism",
                "evidence": {
                    "candidate_local_signature_pass_count": int(
                        candidates["fixed_016_local_signature_pass"].map(_as_bool).sum()
                    ),
                    "candidate_full_route_predicate_count": int(
                        candidates["full_fixed_016_route_predicate_count"].astype(int).gt(0).sum()
                    ),
                },
                "claim_boundary": "Local substrate remains a screen, not a route-level basin mechanism.",
                "run_status": RUN_STATUS,
            },
            {
                "decision_id": "D2_route_negative_morphologies",
                "decision": "strict_analogs_split_into_abrupt_or_point_only_route_negatives",
                "evidence": {
                    str(row.local_pair_id): str(row.pair_explanation_class)
                    for row in candidates.itertuples(index=False)
                },
                "claim_boundary": "Route-negative classes explain the failed generalization gate without changing thresholds.",
                "run_status": RUN_STATUS,
            },
            {
                "decision_id": "D3_next_gate",
                "decision": "inspect_stable_single_side_plateau_conditions_before_new_candidates",
                "evidence": {
                    "positive_reference": PRIMARY_PAIR_ID,
                    "route_negative_candidates": list(CANDIDATE_PAIR_IDS),
                },
                "claim_boundary": "Next work should explain plateau stability, not broaden candidate or policy sweeps.",
                "run_status": RUN_STATUS,
            },
        ]
    )


def _gate_matrix(
    *,
    context: dict[str, Any],
    substrate_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    fraction_rows: pd.DataFrame,
) -> pd.DataFrame:
    route_gates = context["tables"]["route_gate"]
    candidates = pair_rows[
        pair_rows["pair_role"].astype(str).eq("strict_nonboundary_local_signature_analog")
    ]
    controls = pair_rows[
        pair_rows["pair_role"].astype(str).isin(
            {"positive_reference_control", "boundary_guard_control"}
        )
    ]
    reference = pair_rows[pair_rows["local_pair_id"].astype(str).eq(PRIMARY_PAIR_ID)]
    known_negative_classes = {
        "abrupt_source_to_target_switch_without_band",
        "delayed_abrupt_source_to_target_switch_without_band",
        "point_or_seed_fragile_single_side_not_finite_band",
    }
    return pd.DataFrame(
        [
            _gate_row(
                "G1_inputs_readable",
                "Were all route-negative explanation inputs readable?",
                {
                    "substrate_rows": int(len(substrate_rows)),
                    "pair_rows": int(len(pair_rows)),
                    "fraction_rows": int(len(fraction_rows)),
                },
                "6 substrate/pair rows and non-empty fraction profiles",
                len(substrate_rows) == 6 and len(pair_rows) == 6 and not fraction_rows.empty,
            ),
            _gate_row(
                "G2_016_finite_band_reference_reproduced",
                "Does 016 remain the finite single-side band reference?",
                reference[
                    [
                        "route_count",
                        "full_fixed_016_route_predicate_count",
                        "all_route_single_side_fraction_count",
                        "pair_explanation_class",
                    ]
                ].to_dict("records"),
                "016 has every route pass and at least two all-route single-side fractions",
                not reference.empty
                and int(reference.iloc[0]["full_fixed_016_route_predicate_count"])
                == int(reference.iloc[0]["route_count"])
                and int(reference.iloc[0]["all_route_single_side_fraction_count"]) >= 2,
            ),
            _gate_row(
                "G3_candidate_route_negatives_explained",
                "Are all strict analog route negatives assigned to known morphology classes?",
                candidates[
                    ["local_pair_id", "pair_explanation_class"]
                ].to_dict("records"),
                "009/012/020 are abrupt or point-only negatives, not unexplained",
                set(candidates["pair_explanation_class"].astype(str)).issubset(
                    known_negative_classes
                ),
            ),
            _gate_row(
                "G4_local_substrate_not_sufficient",
                "Do local-signature-positive candidates still fail the route predicate?",
                candidates[
                    [
                        "local_pair_id",
                        "fixed_016_local_signature_pass",
                        "full_fixed_016_route_predicate_count",
                    ]
                ].to_dict("records"),
                "all candidates pass local signature and have zero full route predicates",
                bool(candidates["fixed_016_local_signature_pass"].map(_as_bool).all())
                and int(candidates["full_fixed_016_route_predicate_count"].astype(int).sum()) == 0,
            ),
            _gate_row(
                "G5_controls_stay_negative",
                "Do 014/005 controls stay negative for the full route predicate?",
                controls[
                    ["local_pair_id", "full_fixed_016_route_predicate_count", "pair_explanation_class"]
                ].to_dict("records"),
                "zero full route predicates among controls",
                int(controls["full_fixed_016_route_predicate_count"].astype(int).sum()) == 0,
            ),
            _gate_row(
                "G6_prior_route_trace_scope_closed",
                "Did the upstream route trace keep exact scope and claim boundaries?",
                route_gates[["gate_id", "gate_status"]].to_dict("records"),
                "scope and claim gates pass upstream",
                bool(
                    route_gates[
                        route_gates["gate_id"].astype(str).isin(
                            {"G2_exact_trace_scope", "G6_claim_boundaries_closed"}
                        )
                    ]["gate_status"].astype(str).eq("pass").all()
                ),
            ),
            _gate_row(
                "G7_claim_boundaries_closed",
                "Are method, wall, quality/cost, and full-replay claims closed in this audit?",
                CLAIM_BOUNDARY,
                "read-only explanation audit",
                True,
            ),
        ]
    )


def _summary(
    *,
    context: dict[str, Any],
    output_dir: Path,
    pair_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    candidates = pair_rows[
        pair_rows["pair_role"].astype(str).eq("strict_nonboundary_local_signature_analog")
    ]
    return {
        "schema": "nanoclustering_g4_8_first_pass_route_negative_explanation_summary.v1",
        "status": RUN_STATUS,
        "output_dir": str(output_dir),
        "source_dirs": {key: str(value) for key, value in context["paths"].items()},
        "route_negative_explanation_status": (
            "local_substrate_recurrence_without_route_mechanism_explained"
        ),
        "candidate_pair_explanation_classes": {
            str(row.local_pair_id): str(row.pair_explanation_class)
            for row in candidates.itertuples(index=False)
        },
        "reference_pair_explanation_class": str(
            pair_rows[pair_rows["local_pair_id"].astype(str).eq(PRIMARY_PAIR_ID)].iloc[0][
                "pair_explanation_class"
            ]
        ),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "gate_status_counts": gates["gate_status"].value_counts().to_dict(),
        "interpretation": (
            "The fixed local signature is not sufficient for the 016 route "
            "mechanism. The route-positive feature is a stable finite "
            "single-side plateau over bridge fractions, which the first strict "
            "analog queue does not reproduce."
        ),
        "recommended_next_gate": (
            "Before opening more candidates, inspect what makes 016's "
            "single-side state stable across adjacent bridge fractions: bridge "
            "side-balance, endpoint-side support geometry, and seed/start "
            "stability should be audited as predeclared features."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "source_statuses": {
            key: summary.get("status", summary.get("run_status"))
            for key, summary in context["summaries"].items()
        },
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    substrate_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    fraction_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass Route-Negative Explanation Audit",
        "",
        "## Summary",
        "",
        f"- status: {summary['status']}",
        (
            "- route_negative_explanation_status: "
            f"{summary['route_negative_explanation_status']}"
        ),
        (
            "- candidate_pair_explanation_classes: "
            f"{summary['candidate_pair_explanation_classes']}"
        ),
        f"- failed_gates: {summary['failed_gates']}",
        "",
        "## Pair Explanations",
        "",
        _markdown_table(
            pair_rows,
            [
                "local_pair_id",
                "pair_role",
                "fixed_016_local_signature_pass",
                "bridge_to_direct_weight_ratio",
                "original_pair_coassigned_share",
                "route_count",
                "full_fixed_016_route_predicate_count",
                "all_route_single_side_fraction_count",
                "final_target_like_route_count",
                "pair_explanation_class",
                "route_explanation_class_counts",
            ],
            max_rows=20,
        ),
        "",
        "## Fraction Profiles",
        "",
        _markdown_table(
            fraction_rows,
            [
                "local_pair_id",
                "bridge_edge_weight_fraction",
                "route_count",
                "source_family_count",
                "single_side_count",
                "target_like_count",
                "source_family_all_routes",
                "single_side_all_routes",
                "target_like_all_routes",
                "dominant_mechanism_read",
            ],
            max_rows=80,
        ),
        "",
        "## Local Substrate Rows",
        "",
        _markdown_table(
            substrate_rows,
            [
                "local_pair_id",
                "pair_role",
                "direct_cpm_delta_q",
                "bridge_to_direct_weight_ratio",
                "selected_bridge_count",
                "local_node_count",
                "original_pair_coassigned_share",
                "drop_direct_pair_coassigned_share",
                "drop_bridge_pair_coassigned_share",
                "fixed_016_local_signature_pass",
            ],
            max_rows=20,
        ),
        "",
        "## Decisions",
        "",
        _markdown_table(
            decision_rows,
            ["decision_id", "decision", "evidence", "claim_boundary"],
            max_rows=20,
        ),
        "",
        "## Gates",
        "",
        _markdown_table(
            gates,
            ["gate_id", "question", "observed", "minimum_or_rule", "gate_status"],
            max_rows=20,
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


def run_audit(
    *,
    local_ablation_dir: Path = DEFAULT_LOCAL_ABLATION_DIR,
    generalization_screen_dir: Path = DEFAULT_GENERALIZATION_SCREEN_DIR,
    persistence_016_dir: Path = DEFAULT_016_PERSISTENCE_DIR,
    route_trace_dir: Path = DEFAULT_ROUTE_TRACE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    context = _load_context(
        local_ablation_dir=local_ablation_dir,
        generalization_screen_dir=generalization_screen_dir,
        persistence_016_dir=persistence_016_dir,
        route_trace_dir=route_trace_dir,
    )
    substrate_rows = _substrate_rows(context)
    trace_rows = _combined_trace_rows(context)
    fraction_rows = _fraction_profile_rows(trace_rows)
    route_predicates = _route_predicate_rows(trace_rows)
    pair_rows = _pair_explanation_rows(
        substrate_rows=substrate_rows,
        fraction_rows=fraction_rows,
        route_predicates=route_predicates,
    )
    decision_rows = _decision_rows(pair_rows)
    gates = _gate_matrix(
        context=context,
        substrate_rows=substrate_rows,
        pair_rows=pair_rows,
        fraction_rows=fraction_rows,
    )
    summary = _summary(
        context=context,
        output_dir=output_dir,
        pair_rows=pair_rows,
        gates=gates,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(substrate_rows, output_dir / SUBSTRATE_ROWS_CSV)
    _write_csv(pair_rows, output_dir / PAIR_EXPLANATION_ROWS_CSV)
    _write_csv(fraction_rows, output_dir / FRACTION_PROFILE_ROWS_CSV)
    _write_csv(decision_rows, output_dir / DECISION_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_route_negative_explanation_config.v1",
        "local_ablation_dir": str(local_ablation_dir),
        "generalization_screen_dir": str(generalization_screen_dir),
        "persistence_016_dir": str(persistence_016_dir),
        "route_trace_dir": str(route_trace_dir),
        "output_dir": str(output_dir),
        "audit_pair_ids": list(AUDIT_PAIR_IDS),
        "source_family_mechanisms": sorted(SOURCE_FAMILY_MECHANISMS),
        "single_side_mechanism": SINGLE_SIDE_MECHANISM,
        "target_like_mechanism": TARGET_LIKE_MECHANISM,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        substrate_rows=substrate_rows,
        pair_rows=pair_rows,
        fraction_rows=fraction_rows,
        decision_rows=decision_rows,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-ablation-dir", type=Path, default=DEFAULT_LOCAL_ABLATION_DIR)
    parser.add_argument(
        "--generalization-screen-dir",
        type=Path,
        default=DEFAULT_GENERALIZATION_SCREEN_DIR,
    )
    parser.add_argument("--persistence-016-dir", type=Path, default=DEFAULT_016_PERSISTENCE_DIR)
    parser.add_argument("--route-trace-dir", type=Path, default=DEFAULT_ROUTE_TRACE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_audit(
        local_ablation_dir=Path(args.local_ablation_dir),
        generalization_screen_dir=Path(args.generalization_screen_dir),
        persistence_016_dir=Path(args.persistence_016_dir),
        route_trace_dir=Path(args.route_trace_dir),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
