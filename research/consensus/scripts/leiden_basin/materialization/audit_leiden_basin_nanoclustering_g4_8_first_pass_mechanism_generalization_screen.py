#!/usr/bin/env python3
"""Screen 016 mechanism predicates across the local-pair panel.

This read-only audit asks whether the local mechanism named for
``local_pair_016`` is unique, recurrent only as a local substrate, or already
supported as a route-level generalization. It deliberately separates the
local-ablation substrate from route/pathway evidence so that an under-measured
strict analog is not promoted into a basin, wall, quality, or method claim.
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
REFERENCE_PAIR_ID = "local_pair_014"
BOUNDARY_GUARD_PAIR_ID = "local_pair_005"
P1_PREDICATE_ID = "P1_guarded_single_step_separated_transient_candidate"

DEFAULT_LOCAL_ABLATION_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_symmetric_object_variable_pair_local_ablation_gamma1e5_20260603"
)
DEFAULT_TYPED_PREDICATE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_typed_transient_predicate_screen_gamma1e5_20260605"
)
DEFAULT_LOCAL_VALIDATION_DIR = (
    BASE_RESULT_DIR / "leiden_basin_nanoclustering_g4_8_local_validation_readout_gamma1e5_20260604"
)
DEFAULT_016_MECHANISM_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_mechanism_interpretation_audit_gamma1e5_20260605"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_mechanism_generalization_screen_gamma1e5_20260605"
)

PAIR_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_mechanism_generalization_pair_rows.csv"
)
CLASS_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_mechanism_generalization_class_rows.csv"
)
NEXT_GATE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_mechanism_generalization_next_gate_rows.csv"
)
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_mechanism_generalization_decision_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_mechanism_generalization_gate_matrix.csv"
)
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_mechanism_generalization_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_mechanism_generalization_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_mechanism_generalization_report.md"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_mechanism_generalization_screen"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_mechanism_generalization_screen"
WALL_PROMOTION_STATUS = "not_promoted_mechanism_generalization_screen_only"
METHOD_STATUS = "mechanism_generalization_screen_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass mechanism-generalization screen only; "
    "reads existing local-ablation, local-validation, typed-predicate, and "
    "016 mechanism-interpretation artifacts. It does not rerun Leiden, execute "
    "new route/fraction traces, promote basin walls, replay full NanoClustering, "
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


def _as_float(value: Any, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    return float(value)


def _as_int(value: Any, default: int = 0) -> int:
    if pd.isna(value):
        return default
    return int(value)


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
    typed_predicate_dir: Path,
    local_validation_dir: Path,
    mechanism_016_dir: Path,
) -> dict[str, Any]:
    return {
        "paths": {
            "local_ablation_dir": local_ablation_dir,
            "typed_predicate_dir": typed_predicate_dir,
            "local_validation_dir": local_validation_dir,
            "mechanism_016_dir": mechanism_016_dir,
        },
        "summaries": {
            "local_ablation": _read_json(
                local_ablation_dir
                / "nanoclustering_symmetric_object_variable_pair_local_ablation_summary.json"
            ),
            "typed_predicate": _read_json(
                typed_predicate_dir
                / "nanoclustering_g4_8_first_pass_typed_transient_predicate_summary.json"
            ),
            "local_validation": _read_json(
                local_validation_dir
                / "nanoclustering_g4_8_local_validation_readout_summary.json"
            ),
            "mechanism_016": _read_json(
                mechanism_016_dir
                / "nanoclustering_g4_8_first_pass_016_mechanism_summary.json"
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
            "variant_summary": _read_csv(
                local_ablation_dir
                / "nanoclustering_symmetric_object_variable_pair_local_ablation_variant_summary.csv"
            ),
            "typed_pair_feature": _read_csv(
                typed_predicate_dir
                / "nanoclustering_g4_8_first_pass_typed_transient_predicate_pair_feature_rows.csv"
            ),
            "typed_pair_predicate": _read_csv(
                typed_predicate_dir
                / "nanoclustering_g4_8_first_pass_typed_transient_predicate_pair_predicate_rows.csv"
            ),
            "local_validation_pair": _read_csv(
                local_validation_dir
                / "nanoclustering_g4_8_local_validation_readout_pair_rows.csv"
            ),
        },
    }


def _variant_lookup(variant_summary: pd.DataFrame) -> dict[str, dict[str, dict[str, Any]]]:
    lookup: dict[str, dict[str, dict[str, Any]]] = {}
    for row in variant_summary.to_dict(orient="records"):
        pair_id = str(row["local_pair_id"])
        variant = str(row["graph_variant"])
        lookup.setdefault(pair_id, {})[variant] = row
    return lookup


def _first_row(frame: pd.DataFrame, pair_id: str) -> dict[str, Any]:
    if frame.empty or "local_pair_id" not in frame.columns:
        return {}
    matched = frame[frame["local_pair_id"].astype(str) == str(pair_id)]
    if matched.empty:
        return {}
    return matched.iloc[0].to_dict()


def _p1_row(frame: pd.DataFrame, pair_id: str) -> dict[str, Any]:
    if frame.empty:
        return {}
    matched = frame[
        (frame["local_pair_id"].astype(str) == str(pair_id))
        & (frame["predicate_id"].astype(str) == P1_PREDICATE_ID)
    ]
    if matched.empty:
        return {}
    return matched.iloc[0].to_dict()


def _variant_share(variants: dict[str, dict[str, Any]], variant: str) -> float:
    return _as_float(variants.get(variant, {}).get("pair_coassigned_share"), default=-1.0)


def _variant_mechanism_has(
    variants: dict[str, dict[str, Any]], variant: str, mechanism_name: str
) -> bool:
    return mechanism_name in str(variants.get(variant, {}).get("mechanism_read_counts", ""))


def _classify_pair(
    *,
    pair_id: str,
    validation_stratum: str,
    guard_family: str,
    fixed_016_local_signature_pass: bool,
    p1_route_predicate_accepts: bool,
    first_pass_route_readout_available: bool,
    is_primary: bool,
    is_reference: bool,
    is_boundary: bool,
    is_control: bool,
) -> tuple[str, str]:
    if pair_id == PRIMARY_PAIR_ID and fixed_016_local_signature_pass and p1_route_predicate_accepts:
        return (
            "primary_016_full_mechanism_reference",
            "016 has both fixed local signature and available route-level typed transient evidence.",
        )
    if is_boundary:
        if fixed_016_local_signature_pass:
            return (
                "boundary_guard_partial_local_overlap",
                "boundary guard overlaps locally but fails the fixed 016 local signature or route predicate.",
            )
        return (
            "boundary_guard_rejected",
            "boundary guard is rejected by the fixed 016 signature.",
        )
    if is_control:
        if fixed_016_local_signature_pass:
            return (
                "control_local_overlap_not_generalization",
                "control overlap is treated as a local-only leak and blocks local-only promotion.",
            )
        return ("closed_control_rejected", "closed control is rejected by the fixed local signature.")
    if is_reference:
        if fixed_016_local_signature_pass and not p1_route_predicate_accepts:
            return (
                "reference_local_signature_without_single_side_transient",
                "014 matches the local substrate/readout but is the positive reference, not the 016 typed transient.",
            )
        return ("reference_not_016_mechanism", "reference pair does not pass the 016 full predicate.")
    if fixed_016_local_signature_pass and validation_stratum == "strict_ready":
        if not first_pass_route_readout_available:
            return (
                "strict_local_signature_analog_missing_route_readout",
                "strict-ready local analog needs the fixed 016 route/fraction predicate before generality.",
            )
        if not p1_route_predicate_accepts:
            return (
                "strict_local_signature_analog_route_negative",
                "strict-ready local analog has available route readout but not the 016 typed transient.",
            )
    if fixed_016_local_signature_pass:
        return (
            "non_strict_local_signature_analog_not_generalization",
            "local signature recurs outside strict-ready scope and is diagnostic only.",
        )
    if validation_stratum == "strict_ready":
        return (
            "strict_ready_nonanalog",
            "strict-ready pair does not match the fixed 016 local signature.",
        )
    if guard_family:
        return ("guard_or_control_nonanalog", "guard/control pair rejected or kept as diagnostic context.")
    return ("nonanalog", "pair does not match the fixed 016 local signature.")


def _build_pair_rows(context: dict[str, Any]) -> pd.DataFrame:
    local_graph = context["tables"]["local_graph"].copy()
    pair_gate = context["tables"]["pair_gate"]
    variant_lookup = _variant_lookup(context["tables"]["variant_summary"])
    typed_pair_feature = context["tables"]["typed_pair_feature"]
    typed_pair_predicate = context["tables"]["typed_pair_predicate"]
    local_validation_pair = context["tables"]["local_validation_pair"]

    rows: list[dict[str, Any]] = []
    for graph_row in local_graph.to_dict(orient="records"):
        pair_id = str(graph_row["local_pair_id"])
        variants = variant_lookup.get(pair_id, {})
        gate_row = _first_row(pair_gate, pair_id)
        feature_row = _first_row(typed_pair_feature, pair_id)
        validation_row = _first_row(local_validation_pair, pair_id)
        p1_row = _p1_row(typed_pair_predicate, pair_id)

        validation_stratum = str(
            feature_row.get(
                "validation_stratum",
                validation_row.get("validation_stratum", ""),
            )
        )
        guard_family = str(feature_row.get("guard_family", ""))
        is_primary = _as_bool(feature_row.get("is_primary_typed_transient_pair", pair_id == PRIMARY_PAIR_ID))
        is_reference = _as_bool(feature_row.get("is_reference_pair", pair_id == REFERENCE_PAIR_ID))
        is_boundary = _as_bool(feature_row.get("is_boundary_guard_pair", pair_id == BOUNDARY_GUARD_PAIR_ID))
        is_control = _as_bool(feature_row.get("is_control_pair", False))

        direct_positive_weak_pair = str(graph_row.get("mechanism_label")) == "direct_positive_weak_pair_at_gamma"
        direct_positive_at_gamma = _as_bool(graph_row.get("direct_positive_at_gamma"))
        direct_edge_needed = _as_bool(graph_row.get("direct_edge_needed_for_input_gamma_positive"))
        selected_bridge_count = _as_int(graph_row.get("selected_bridge_count"))
        bridge_to_direct_weight_ratio = _as_float(
            graph_row.get("bridge_to_direct_weight_ratio"), default=0.0
        )
        local_substrate_pass = (
            direct_positive_weak_pair
            and direct_positive_at_gamma
            and direct_edge_needed
            and selected_bridge_count > 0
            and bridge_to_direct_weight_ratio > 1.0
            and str(gate_row.get("gate_status"))
            == "diagnostic_supports_local_mechanism_reproduction"
        )

        original_share = _variant_share(variants, "original")
        drop_direct_share = _variant_share(variants, "drop_direct_edge")
        drop_bridge_share = _variant_share(variants, "drop_bridge_edges")
        drop_both_share = _variant_share(variants, "drop_direct_and_bridge_edges")
        original_partial = 0.0 < original_share < 1.0
        drop_direct_bridge_split = (
            drop_direct_share == 0.0
            and _variant_mechanism_has(
                variants, "drop_direct_edge", "pair_separated_bridge_split"
            )
        )
        drop_bridge_target_like = (
            drop_bridge_share == 1.0
            and _variant_mechanism_has(
                variants, "drop_bridge_edges", "pair_coassigned_without_selected_bridge"
            )
        )
        drop_direct_and_bridge_separated = (
            drop_both_share == 0.0
            and _variant_mechanism_has(
                variants,
                "drop_direct_and_bridge_edges",
                "pair_separated_no_selected_bridge",
            )
        )
        fixed_016_local_signature_pass = (
            local_substrate_pass
            and original_partial
            and drop_direct_bridge_split
            and drop_bridge_target_like
            and drop_direct_and_bridge_separated
        )

        first_pass_route_readout_available = bool(feature_row)
        typed_single_all = _as_bool(feature_row.get("typed_single_separated_transient_all_routes"))
        p1_route_predicate_accepts = _as_bool(p1_row.get("accepted_by_predicate"))
        strict_nonboundary_route_gap = (
            validation_stratum == "strict_ready"
            and fixed_016_local_signature_pass
            and not is_primary
            and not is_reference
            and not is_boundary
            and not is_control
            and not first_pass_route_readout_available
        )

        mechanism_class, mechanism_class_reason = _classify_pair(
            pair_id=pair_id,
            validation_stratum=validation_stratum,
            guard_family=guard_family,
            fixed_016_local_signature_pass=fixed_016_local_signature_pass,
            p1_route_predicate_accepts=p1_route_predicate_accepts,
            first_pass_route_readout_available=first_pass_route_readout_available,
            is_primary=is_primary,
            is_reference=is_reference,
            is_boundary=is_boundary,
            is_control=is_control,
        )

        rows.append(
            {
                "local_pair_id": pair_id,
                "validation_stratum": validation_stratum,
                "guard_family": guard_family,
                "is_primary_typed_transient_pair": is_primary,
                "is_reference_pair": is_reference,
                "is_boundary_guard_pair": is_boundary,
                "is_control_pair": is_control,
                "direct_positive_weak_pair": direct_positive_weak_pair,
                "direct_positive_at_gamma": direct_positive_at_gamma,
                "direct_edge_needed_for_input_gamma_positive": direct_edge_needed,
                "selected_bridge_count": selected_bridge_count,
                "bridge_to_direct_weight_ratio": bridge_to_direct_weight_ratio,
                "local_substrate_pass": local_substrate_pass,
                "original_pair_coassigned_share": original_share,
                "drop_direct_pair_coassigned_share": drop_direct_share,
                "drop_bridge_pair_coassigned_share": drop_bridge_share,
                "drop_direct_and_bridge_pair_coassigned_share": drop_both_share,
                "original_partial_pair_coassignment": original_partial,
                "drop_direct_bridge_split": drop_direct_bridge_split,
                "drop_bridge_target_like": drop_bridge_target_like,
                "drop_direct_and_bridge_separated": drop_direct_and_bridge_separated,
                "fixed_016_local_signature_pass": fixed_016_local_signature_pass,
                "first_pass_route_readout_available": first_pass_route_readout_available,
                "typed_single_separated_transient_all_routes": typed_single_all,
                "p1_route_predicate_accepts": p1_route_predicate_accepts,
                "strict_nonboundary_route_gap": strict_nonboundary_route_gap,
                "mechanism_generalization_class": mechanism_class,
                "mechanism_generalization_reason": mechanism_class_reason,
                "run_status": RUN_STATUS,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )

    return pd.DataFrame(rows).sort_values(
        [
            "fixed_016_local_signature_pass",
            "validation_stratum",
            "mechanism_generalization_class",
            "local_pair_id",
        ],
        ascending=[False, True, True, True],
    )


def _build_class_rows(pair_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for class_name, group in pair_rows.groupby("mechanism_generalization_class", dropna=False):
        rows.append(
            {
                "mechanism_generalization_class": class_name,
                "pair_count": int(len(group)),
                "local_pair_ids": ";".join(sorted(group["local_pair_id"].astype(str).tolist())),
                "strict_ready_count": int((group["validation_stratum"] == "strict_ready").sum()),
                "fixed_016_local_signature_pass_count": int(
                    group["fixed_016_local_signature_pass"].map(_as_bool).sum()
                ),
                "p1_route_predicate_accept_count": int(
                    group["p1_route_predicate_accepts"].map(_as_bool).sum()
                ),
                "route_gap_count": int(group["strict_nonboundary_route_gap"].map(_as_bool).sum()),
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["route_gap_count", "fixed_016_local_signature_pass_count", "pair_count"],
        ascending=[False, False, False],
    )


def _build_next_gate_rows(pair_rows: pd.DataFrame) -> pd.DataFrame:
    next_gate = pair_rows[pair_rows["strict_nonboundary_route_gap"].map(_as_bool)].copy()
    if next_gate.empty:
        return pd.DataFrame(
            columns=[
                "local_pair_id",
                "recommended_next_gate_role",
                "predeclared_mechanism_predicate",
                "required_controls",
                "allowed_claim_after_execution",
                "claim_boundary",
            ]
        )
    next_gate = next_gate.sort_values(
        ["bridge_to_direct_weight_ratio", "original_pair_coassigned_share"],
        ascending=[False, True],
    )
    rows = []
    for row in next_gate.to_dict(orient="records"):
        rows.append(
            {
                "local_pair_id": row["local_pair_id"],
                "recommended_next_gate_role": "strict_nonboundary_local_signature_analog_needs_route_trace",
                "predeclared_mechanism_predicate": (
                    "same fixed 016 predicate: source-family start, target "
                    "anchor at lower bridge fraction, and finite "
                    "pair-separated single-side-bridge transition band"
                ),
                "required_controls": (
                    f"{REFERENCE_PAIR_ID} positive reference and "
                    f"{BOUNDARY_GUARD_PAIR_ID} boundary guard; no broad "
                    "threshold or policy sweep"
                ),
                "allowed_claim_after_execution": (
                    "route-level mechanism recurrence only if the same typed "
                    "band appears; no wall, quality/cost, full replay, or "
                    "method claim"
                ),
                "bridge_to_direct_weight_ratio": row["bridge_to_direct_weight_ratio"],
                "original_pair_coassigned_share": row["original_pair_coassigned_share"],
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _build_decision_rows(pair_rows: pd.DataFrame, next_gate_rows: pd.DataFrame) -> pd.DataFrame:
    local_signature_pairs = sorted(
        pair_rows[pair_rows["fixed_016_local_signature_pass"].map(_as_bool)][
            "local_pair_id"
        ]
        .astype(str)
        .tolist()
    )
    route_accept_pairs = sorted(
        pair_rows[pair_rows["p1_route_predicate_accepts"].map(_as_bool)]["local_pair_id"]
        .astype(str)
        .tolist()
    )
    strict_route_gap_pairs = sorted(next_gate_rows["local_pair_id"].astype(str).tolist())
    return pd.DataFrame(
        [
            {
                "decision_id": "D1_local_substrate_recurrence",
                "decision": "fixed_016_local_signature_is_recurrent_but_not_sufficient",
                "evidence": {
                    "local_signature_pair_count": len(local_signature_pairs),
                    "local_signature_pairs": local_signature_pairs,
                },
                "claim_boundary": "Local signature recurrence is substrate evidence only.",
                "run_status": RUN_STATUS,
            },
            {
                "decision_id": "D2_route_level_generality",
                "decision": "route_level_generality_not_established",
                "evidence": {
                    "route_accept_pair_count": len(route_accept_pairs),
                    "route_accept_pairs": route_accept_pairs,
                    "strict_route_gap_pairs": strict_route_gap_pairs,
                },
                "claim_boundary": "Only existing route predicate acceptance is 016; analogs need narrow execution.",
                "run_status": RUN_STATUS,
            },
            {
                "decision_id": "D3_next_execution_scope",
                "decision": "predeclare_narrow_route_trace_for_strict_nonboundary_analogs",
                "evidence": {
                    "candidate_pairs": strict_route_gap_pairs,
                    "reference_pair": REFERENCE_PAIR_ID,
                    "boundary_guard_pair": BOUNDARY_GUARD_PAIR_ID,
                },
                "claim_boundary": "This is a screen for the next gate, not a method or wall result.",
                "run_status": RUN_STATUS,
            },
        ]
    )


def _build_gate_matrix(pair_rows: pd.DataFrame, next_gate_rows: pd.DataFrame) -> pd.DataFrame:
    primary = pair_rows[pair_rows["local_pair_id"] == PRIMARY_PAIR_ID]
    local_signature_pairs = pair_rows[pair_rows["fixed_016_local_signature_pass"].map(_as_bool)]
    route_accept_pairs = pair_rows[pair_rows["p1_route_predicate_accepts"].map(_as_bool)]
    boundary_or_control_full = pair_rows[
        (pair_rows["p1_route_predicate_accepts"].map(_as_bool))
        & (
            pair_rows["is_boundary_guard_pair"].map(_as_bool)
            | pair_rows["is_control_pair"].map(_as_bool)
        )
    ]
    strict_route_gap_count = int(len(next_gate_rows))
    gate_rows = [
        _gate_row(
            "G1_input_artifacts_present",
            "Were all fixed-predicate source artifacts readable?",
            {
                "pair_rows": int(len(pair_rows)),
                "local_signature_pair_count": int(len(local_signature_pairs)),
            },
            "23 local-pair rows and non-empty fixed-signature rows",
            len(pair_rows) == 23 and len(local_signature_pairs) > 0,
        ),
        _gate_row(
            "G2_016_reproduces_fixed_full_predicate",
            "Does 016 still pass both local and route predicates?",
            primary[
                [
                    "fixed_016_local_signature_pass",
                    "p1_route_predicate_accepts",
                    "typed_single_separated_transient_all_routes",
                ]
            ].to_dict(orient="records"),
            "016 local signature pass and P1 route predicate accept",
            (
                not primary.empty
                and _as_bool(primary.iloc[0]["fixed_016_local_signature_pass"])
                and _as_bool(primary.iloc[0]["p1_route_predicate_accepts"])
            ),
        ),
        _gate_row(
            "G3_local_signature_recurrence_observed",
            "Does the fixed 016 local signature recur outside 016?",
            {
                "non_016_local_signature_pairs": sorted(
                    local_signature_pairs[
                        local_signature_pairs["local_pair_id"] != PRIMARY_PAIR_ID
                    ]["local_pair_id"]
                    .astype(str)
                    .tolist()
                )
            },
            "at least one non-016 local signature analog",
            len(local_signature_pairs[local_signature_pairs["local_pair_id"] != PRIMARY_PAIR_ID]) > 0,
        ),
        _gate_row(
            "G4_route_level_generality_not_yet_established",
            "Is route-level generality already established beyond 016?",
            {
                "route_accept_pairs": sorted(route_accept_pairs["local_pair_id"].astype(str).tolist()),
                "strict_nonboundary_route_gap_pairs": sorted(
                    next_gate_rows["local_pair_id"].astype(str).tolist()
                ),
            },
            "requires at least one non-016 route predicate accept; currently expected to fail if only 016 is instrumented",
            len(route_accept_pairs[route_accept_pairs["local_pair_id"] != PRIMARY_PAIR_ID]) > 0,
        ),
        _gate_row(
            "G5_no_boundary_or_control_full_predicate_leak",
            "Do boundary/control pairs avoid the full 016 route predicate?",
            {
                "boundary_or_control_full_predicate_pairs": sorted(
                    boundary_or_control_full["local_pair_id"].astype(str).tolist()
                )
            },
            "zero boundary/control full-predicate accepts",
            boundary_or_control_full.empty,
        ),
        _gate_row(
            "G6_next_gate_queue_is_narrow",
            "Is the next execution queue narrow and mechanism-predeclared?",
            {
                "strict_nonboundary_route_gap_count": strict_route_gap_count,
                "strict_nonboundary_route_gap_pairs": sorted(
                    next_gate_rows["local_pair_id"].astype(str).tolist()
                ),
            },
            "1-5 strict nonboundary fixed-signature analogs, no threshold sweep",
            1 <= strict_route_gap_count <= 5,
        ),
    ]
    return pd.DataFrame(gate_rows)


def _build_summary(
    *,
    context: dict[str, Any],
    pair_rows: pd.DataFrame,
    class_rows: pd.DataFrame,
    next_gate_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    local_signature_pairs = sorted(
        pair_rows[pair_rows["fixed_016_local_signature_pass"].map(_as_bool)][
            "local_pair_id"
        ]
        .astype(str)
        .tolist()
    )
    route_accept_pairs = sorted(
        pair_rows[pair_rows["p1_route_predicate_accepts"].map(_as_bool)]["local_pair_id"]
        .astype(str)
        .tolist()
    )
    strict_route_gap_pairs = sorted(next_gate_rows["local_pair_id"].astype(str).tolist())
    failed_gates = sorted(
        gate_matrix[gate_matrix["gate_status"] == "fail"]["gate_id"].astype(str).tolist()
    )
    return {
        "schema": "nanoclustering_g4_8_first_pass_mechanism_generalization_summary.v1",
        "status": "mechanism_generalization_screen_materialized_with_route_gap",
        "run_status": RUN_STATUS,
        "output_dir": str(output_dir),
        "source_dirs": {key: str(value) for key, value in context["paths"].items()},
        "pair_count": int(len(pair_rows)),
        "fixed_016_local_signature_pair_count": len(local_signature_pairs),
        "fixed_016_local_signature_pairs": local_signature_pairs,
        "strict_ready_fixed_016_local_signature_pairs": sorted(
            pair_rows[
                (pair_rows["validation_stratum"] == "strict_ready")
                & pair_rows["fixed_016_local_signature_pass"].map(_as_bool)
            ]["local_pair_id"]
            .astype(str)
            .tolist()
        ),
        "p1_route_predicate_accept_pair_count": len(route_accept_pairs),
        "p1_route_predicate_accept_pairs": route_accept_pairs,
        "strict_nonboundary_route_gap_pair_count": len(strict_route_gap_pairs),
        "strict_nonboundary_route_gap_pairs": strict_route_gap_pairs,
        "mechanism_generalization_class_counts": {
            str(row["mechanism_generalization_class"]): int(row["pair_count"])
            for row in class_rows.to_dict(orient="records")
        },
        "failed_gates": failed_gates,
        "gate_status_counts": {
            status: int(count)
            for status, count in gate_matrix["gate_status"].value_counts().to_dict().items()
        },
        "recommended_next_gate": (
            "Execute a narrow fixed-predicate route/fraction trace for strict "
            "nonboundary local-signature analogs "
            f"{strict_route_gap_pairs}, with {REFERENCE_PAIR_ID} and "
            f"{BOUNDARY_GUARD_PAIR_ID} retained as controls; do not run a "
            "broad threshold, policy, or localization sweep."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "source_statuses": {
            key: summary.get("status", summary.get("run_status"))
            for key, summary in context["summaries"].items()
        },
        "written_artifacts": [
            PAIR_ROWS_CSV,
            CLASS_ROWS_CSV,
            NEXT_GATE_ROWS_CSV,
            DECISION_ROWS_CSV,
            GATE_MATRIX_CSV,
            CONFIG_JSON,
            SUMMARY_JSON,
            REPORT_MD,
        ],
    }


def _write_report(
    *,
    output_dir: Path,
    pair_rows: pd.DataFrame,
    class_rows: pd.DataFrame,
    next_gate_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass Mechanism Generalization Screen",
        "",
        "## Summary",
        "",
        f"- status: {summary['status']}",
        (
            "- fixed_016_local_signature_pair_count: "
            f"{summary['fixed_016_local_signature_pair_count']}"
        ),
        (
            "- fixed_016_local_signature_pairs: "
            f"{', '.join(summary['fixed_016_local_signature_pairs'])}"
        ),
        (
            "- p1_route_predicate_accept_pairs: "
            f"{', '.join(summary['p1_route_predicate_accept_pairs'])}"
        ),
        (
            "- strict_nonboundary_route_gap_pairs: "
            f"{', '.join(summary['strict_nonboundary_route_gap_pairs'])}"
        ),
        f"- failed_gates: {', '.join(summary['failed_gates']) if summary['failed_gates'] else 'none'}",
        "",
        "## Mechanism Classes",
        "",
        _markdown_table(
            class_rows,
            [
                "mechanism_generalization_class",
                "pair_count",
                "local_pair_ids",
                "strict_ready_count",
                "fixed_016_local_signature_pass_count",
                "p1_route_predicate_accept_count",
                "route_gap_count",
            ],
            max_rows=40,
        ),
        "",
        "## Pair Rows",
        "",
        _markdown_table(
            pair_rows,
            [
                "local_pair_id",
                "validation_stratum",
                "guard_family",
                "fixed_016_local_signature_pass",
                "first_pass_route_readout_available",
                "typed_single_separated_transient_all_routes",
                "p1_route_predicate_accepts",
                "strict_nonboundary_route_gap",
                "mechanism_generalization_class",
                "bridge_to_direct_weight_ratio",
                "original_pair_coassigned_share",
                "drop_direct_pair_coassigned_share",
                "drop_bridge_pair_coassigned_share",
            ],
            max_rows=60,
        ),
        "",
        "## Next Gate Queue",
        "",
        _markdown_table(
            next_gate_rows,
            [
                "local_pair_id",
                "recommended_next_gate_role",
                "bridge_to_direct_weight_ratio",
                "original_pair_coassigned_share",
                "predeclared_mechanism_predicate",
                "required_controls",
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
        "## Gate Matrix",
        "",
        _markdown_table(
            gate_matrix,
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
    typed_predicate_dir: Path = DEFAULT_TYPED_PREDICATE_DIR,
    local_validation_dir: Path = DEFAULT_LOCAL_VALIDATION_DIR,
    mechanism_016_dir: Path = DEFAULT_016_MECHANISM_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    context = _load_context(
        local_ablation_dir=local_ablation_dir,
        typed_predicate_dir=typed_predicate_dir,
        local_validation_dir=local_validation_dir,
        mechanism_016_dir=mechanism_016_dir,
    )
    pair_rows = _build_pair_rows(context)
    class_rows = _build_class_rows(pair_rows)
    next_gate_rows = _build_next_gate_rows(pair_rows)
    decision_rows = _build_decision_rows(pair_rows, next_gate_rows)
    gate_matrix = _build_gate_matrix(pair_rows, next_gate_rows)
    summary = _build_summary(
        context=context,
        pair_rows=pair_rows,
        class_rows=class_rows,
        next_gate_rows=next_gate_rows,
        gate_matrix=gate_matrix,
        output_dir=output_dir,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(pair_rows, output_dir / PAIR_ROWS_CSV)
    _write_csv(class_rows, output_dir / CLASS_ROWS_CSV)
    _write_csv(next_gate_rows, output_dir / NEXT_GATE_ROWS_CSV)
    _write_csv(decision_rows, output_dir / DECISION_ROWS_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            {
                "local_ablation_dir": str(local_ablation_dir),
                "typed_predicate_dir": str(typed_predicate_dir),
                "local_validation_dir": str(local_validation_dir),
                "mechanism_016_dir": str(mechanism_016_dir),
                "output_dir": str(output_dir),
                "primary_pair_id": PRIMARY_PAIR_ID,
                "reference_pair_id": REFERENCE_PAIR_ID,
                "boundary_guard_pair_id": BOUNDARY_GUARD_PAIR_ID,
                "p1_predicate_id": P1_PREDICATE_ID,
                "run_status": RUN_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        pair_rows=pair_rows,
        class_rows=class_rows,
        next_gate_rows=next_gate_rows,
        decision_rows=decision_rows,
        gate_matrix=gate_matrix,
        summary=summary,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen 016 mechanism generalization over local pairs."
    )
    parser.add_argument("--local-ablation-dir", type=Path, default=DEFAULT_LOCAL_ABLATION_DIR)
    parser.add_argument("--typed-predicate-dir", type=Path, default=DEFAULT_TYPED_PREDICATE_DIR)
    parser.add_argument("--local-validation-dir", type=Path, default=DEFAULT_LOCAL_VALIDATION_DIR)
    parser.add_argument("--mechanism-016-dir", type=Path, default=DEFAULT_016_MECHANISM_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_audit(
        local_ablation_dir=Path(args.local_ablation_dir),
        typed_predicate_dir=Path(args.typed_predicate_dir),
        local_validation_dir=Path(args.local_validation_dir),
        mechanism_016_dir=Path(args.mechanism_016_dir),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
