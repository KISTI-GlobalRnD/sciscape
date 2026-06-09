#!/usr/bin/env python3
"""Run the fixed-predicate mechanism-generalization route trace.

The runner consumes
``design_leiden_basin_nanoclustering_g4_8_first_pass_mechanism_generalization_route_contract.py``.
It executes only the predeclared strict analog/control source-start rows and
classifies each seed route with the fixed ``016`` mechanism predicate:
source-family start, finite pair-separated single-side bridge band, and
target-like bridge-release final state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import run_leiden_basin_nanoclustering_g4_8_scoped_pathway_probe_trace as scoped_trace
from design_leiden_basin_nanoclustering_g4_8_first_pass_mechanism_generalization_route_contract import (
    ALL_EXECUTION_PAIR_IDS,
    BOUNDARY_GUARD_PAIR_ID,
    CANDIDATE_PAIR_IDS,
    CLAIM_BOUNDARY as CONTRACT_CLAIM_BOUNDARY,
    CONTROL_PAIR_IDS,
    DEFAULT_OUTPUT_DIR as DEFAULT_CONTRACT_DIR,
    FINE_BRIDGE_FRACTIONS,
    FRACTION_STEP_ROWS_CSV as CONTRACT_FRACTION_STEP_ROWS_CSV,
    GATE_MATRIX_CSV as CONTRACT_GATE_MATRIX_CSV,
    PLANNED_ROUTE_FAMILY,
    READOUT_RULE_ROWS_CSV as CONTRACT_READOUT_RULE_ROWS_CSV,
    REFERENCE_PAIR_ID,
    ROUTE_PLAN_ROWS_CSV as CONTRACT_ROUTE_PLAN_ROWS_CSV,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)
from run_leiden_basin_nanoclustering_symmetric_object_variable_pair_local_ablation import (
    DEFAULT_OUTPUT_DIR as DEFAULT_LOCAL_ABLATION_DIR,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_gamma1e5_20260605"
)

TRACE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_rows.csv"
)
ROUTE_MECHANISM_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_route_rows.csv"
)
PAIR_MECHANISM_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_pair_rows.csv"
)
FRACTION_MECHANISM_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_fraction_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_gate_matrix.csv"
)
SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_summary.json"
)
CONFIG_JSON = (
    "nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_config.json"
)
REPORT_MD = (
    "nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_report.md"
)

RUN_STATUS = "executed_nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace"
ROUTE_EXECUTION_STATUS = "executed_fixed_predicate_mechanism_generalization_route_trace"
WALL_PROMOTION_STATUS = "not_promoted_mechanism_generalization_route_trace_only"
METHOD_STATUS = "mechanism_generalization_route_trace_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass mechanism-generalization route trace only; "
    "executes the predeclared fixed-predicate source-start bridge-fraction "
    "trace for strict local-signature analogs and controls. It does not "
    "promote basin walls, replay full NanoClustering, evaluate quality/cost "
    "value, or claim method success."
)

SOURCE_FAMILY_MECHANISMS = {
    "pair_coassigned_with_selected_bridge",
    "pair_separated_bridge_split",
}
SINGLE_SIDE_MECHANISM = "pair_separated_single_side_bridge"
TARGET_LIKE_MECHANISM = "pair_coassigned_without_selected_bridge"
EPS = 1e-9


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


def _install_contract_schedule(fraction_steps: pd.DataFrame) -> None:
    for family, group in fraction_steps.groupby("planned_route_family", sort=False):
        steps: list[dict[str, Any]] = []
        for row in group.sort_values("step_index", kind="mergesort").drop_duplicates(
            "step_index"
        ).itertuples(index=False):
            steps.append(
                {
                    "step_index": int(row.step_index),
                    "step_label": str(row.step_label),
                    "direct_edge_weight_fraction": float(row.direct_edge_weight_fraction),
                    "bridge_edge_weight_fraction": float(row.bridge_edge_weight_fraction),
                    "expected_final_anchor_variant": str(row.expected_final_anchor_variant),
                }
            )
        scoped_trace.SCHEDULES[str(family)] = tuple(steps)
        final_variant = str(steps[-1]["expected_final_anchor_variant"])
        scoped_trace.EXPECTED_FINAL_ASSIGNMENT[str(family)] = (
            scoped_trace.ANCHOR_VARIANT_TO_ASSIGNMENT[final_variant]
        )


def _override_status_columns(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    rows = rows.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    rows["run_status"] = RUN_STATUS
    rows["wall_generality_claim_allowed_after_trace"] = False
    rows["method_claim_allowed_after_trace"] = False
    rows["quality_cost_claim_allowed_after_trace"] = False
    return rows


def _annotate_trace_rows(trace_rows: pd.DataFrame, route_plan: pd.DataFrame) -> pd.DataFrame:
    if trace_rows.empty:
        return trace_rows
    metadata_cols = [
        "route_contract_id",
        "contract_pair_role",
        "source_start_macro_role",
        "source_start_condition",
        "source_start_expected_validation_pass",
    ]
    metadata = route_plan[metadata_cols].drop_duplicates("route_contract_id")
    return trace_rows.merge(metadata, on="route_contract_id", how="left", validate="many_to_one")


def _adjacent_step_band(step_indices: list[int]) -> bool:
    if len(step_indices) < 2:
        return False
    ordered = sorted(step_indices)
    return max(ordered) - min(ordered) == len(ordered) - 1


def _mechanism_sequence(ordered: pd.DataFrame) -> str:
    return " -> ".join(
        (
            f"{float(row.bridge_edge_weight_fraction):.5g}:"
            f"{str(row.mechanism_read)}"
        )
        for row in ordered.itertuples(index=False)
    )


def _route_mechanism_rows(trace_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if trace_rows.empty:
        return pd.DataFrame(rows)
    group_cols = [
        "route_contract_id",
        "validation_unit_id",
        "local_pair_id",
        "contract_pair_role",
        "start_condition",
        "planned_route_family",
        "seed",
    ]
    for keys, group in trace_rows.groupby(group_cols, sort=False):
        key_data = dict(zip(group_cols, keys, strict=True))
        ordered = group.sort_values("step_index", kind="mergesort").copy()
        first = ordered.iloc[0]
        final = ordered.iloc[-1]
        source_rows = ordered[
            ordered["mechanism_read"].astype(str).isin(SOURCE_FAMILY_MECHANISMS)
        ]
        single_side_rows = ordered[
            ordered["mechanism_read"].astype(str).eq(SINGLE_SIDE_MECHANISM)
        ]
        target_rows = ordered[
            ordered["mechanism_read"].astype(str).eq(TARGET_LIKE_MECHANISM)
        ]
        single_side_steps = [int(value) for value in single_side_rows["step_index"].tolist()]
        single_side_fractions = [
            float(value) for value in single_side_rows["bridge_edge_weight_fraction"].tolist()
        ]
        source_family_start = str(first["mechanism_read"]) in SOURCE_FAMILY_MECHANISMS
        final_target_like = str(final["mechanism_read"]) == TARGET_LIKE_MECHANISM
        finite_single_side_band = _adjacent_step_band(single_side_steps)
        full_predicate = (
            source_family_start
            and finite_single_side_band
            and final_target_like
        )
        if full_predicate:
            route_class = "fixed_016_route_predicate_pass"
        elif not source_family_start:
            route_class = "source_family_start_absent"
        elif not single_side_steps and final_target_like:
            route_class = "source_to_target_without_single_side_band"
        elif single_side_steps and not finite_single_side_band:
            route_class = "single_side_nonfinite_or_fragmented"
        elif single_side_steps and not final_target_like:
            route_class = "single_side_band_without_target_final"
        elif not final_target_like:
            route_class = "target_like_final_absent"
        else:
            route_class = "other_fixed_predicate_failure"

        objective_values = list(map(float, ordered["objective_value_by_step"].tolist()))
        objective_diffs = [
            objective_values[index + 1] - objective_values[index]
            for index in range(len(objective_values) - 1)
        ]
        rows.append(
            {
                **key_data,
                "route_key": f"{key_data['start_condition']}|seed={int(key_data['seed'])}",
                "fraction_count": int(len(ordered)),
                "source_family_start": bool(source_family_start),
                "source_family_fraction_count": int(len(source_rows)),
                "source_family_fractions": ";".join(
                    f"{float(value):.5g}"
                    for value in source_rows["bridge_edge_weight_fraction"].tolist()
                ),
                "single_side_fraction_count": int(len(single_side_rows)),
                "single_side_fractions": ";".join(
                    f"{value:.5g}" for value in single_side_fractions
                ),
                "single_side_adjacent_fraction_band": bool(finite_single_side_band),
                "target_like_fraction_count": int(len(target_rows)),
                "target_like_fractions": ";".join(
                    f"{float(value):.5g}"
                    for value in target_rows["bridge_edge_weight_fraction"].tolist()
                ),
                "final_target_like": bool(final_target_like),
                "final_matches_expected_anchor": _as_bool(
                    final.get("matches_expected_final_anchor", False)
                ),
                "distinct_mechanism_read_count": int(ordered["mechanism_read"].nunique()),
                "mechanism_read_sequence": _mechanism_sequence(ordered),
                "objective_monotone_nonincreasing_with_bridge_release": bool(
                    all(delta <= EPS for delta in objective_diffs)
                ),
                "max_objective_debt_from_start": float(
                    ordered["objective_debt_from_start"].max()
                ),
                "max_objective_recovery_from_min": float(
                    ordered["objective_recovery_from_min"].max()
                ),
                "fixed_016_route_predicate_pass": bool(full_predicate),
                "route_mechanism_class": route_class,
                "wall_claim_ready_after_trace": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows)


def _pair_mechanism_rows(route_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if route_rows.empty:
        return pd.DataFrame(rows)
    for keys, group in route_rows.groupby(
        ["local_pair_id", "contract_pair_role"],
        sort=False,
    ):
        pair_id, pair_role = keys
        route_count = int(len(group))
        pass_count = int(group["fixed_016_route_predicate_pass"].map(_as_bool).sum())
        source_count = int(group["source_family_start"].map(_as_bool).sum())
        band_count = int(group["single_side_adjacent_fraction_band"].map(_as_bool).sum())
        target_count = int(group["final_target_like"].map(_as_bool).sum())
        if pair_role == "strict_nonboundary_local_signature_analog":
            if pass_count == route_count and route_count > 0:
                recurrence_class = "candidate_full_route_level_recurrence"
            elif pass_count > 0:
                recurrence_class = "candidate_partial_route_level_recurrence"
            else:
                recurrence_class = "candidate_route_level_recurrence_absent"
        elif pair_role == "positive_reference_control":
            recurrence_class = (
                "positive_reference_full_predicate_leak"
                if pass_count > 0
                else "positive_reference_control_negative_for_016_predicate"
            )
        elif pair_role == "boundary_guard_control":
            recurrence_class = (
                "boundary_guard_full_predicate_leak"
                if pass_count > 0
                else "boundary_guard_control_negative_for_016_predicate"
            )
        else:
            recurrence_class = "unclassified_pair_role"
        rows.append(
            {
                "local_pair_id": pair_id,
                "contract_pair_role": pair_role,
                "route_count": route_count,
                "fixed_016_route_predicate_pass_count": pass_count,
                "source_family_start_count": source_count,
                "finite_single_side_band_count": band_count,
                "final_target_like_count": target_count,
                "route_mechanism_class_counts": group[
                    "route_mechanism_class"
                ].value_counts().to_dict(),
                "mechanism_recurrence_class": recurrence_class,
                "wall_claim_ready_after_trace": False,
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows)


def _fraction_mechanism_rows(trace_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if trace_rows.empty:
        return pd.DataFrame(rows)
    for keys, group in trace_rows.groupby(
        ["local_pair_id", "contract_pair_role", "bridge_edge_weight_fraction"],
        sort=False,
    ):
        pair_id, pair_role, fraction = keys
        mechanism_counts = group["mechanism_read"].astype(str).value_counts()
        dominant = str(mechanism_counts.index[0]) if not mechanism_counts.empty else ""
        rows.append(
            {
                "local_pair_id": pair_id,
                "contract_pair_role": pair_role,
                "bridge_edge_weight_fraction": float(fraction),
                "trace_row_count": int(len(group)),
                "route_count": int(
                    group[["route_contract_id", "seed"]].drop_duplicates().shape[0]
                ),
                "source_family_count": int(
                    group["mechanism_read"].astype(str).isin(SOURCE_FAMILY_MECHANISMS).sum()
                ),
                "single_side_count": int(
                    group["mechanism_read"].astype(str).eq(SINGLE_SIDE_MECHANISM).sum()
                ),
                "target_like_count": int(
                    group["mechanism_read"].astype(str).eq(TARGET_LIKE_MECHANISM).sum()
                ),
                "dominant_mechanism_read": dominant,
                "mechanism_read_counts": mechanism_counts.to_dict(),
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


def _gate_matrix(
    *,
    contract_gates: pd.DataFrame,
    route_plan: pd.DataFrame,
    trace_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    step_config_count: int,
    seeds: int,
) -> pd.DataFrame:
    expected_trace_rows = int(len(route_plan)) * len(FINE_BRIDGE_FRACTIONS) * int(seeds)
    candidate_pairs = pair_rows[
        pair_rows["contract_pair_role"].astype(str).eq(
            "strict_nonboundary_local_signature_analog"
        )
    ]
    full_candidate_pairs = candidate_pairs[
        candidate_pairs["mechanism_recurrence_class"].astype(str).eq(
            "candidate_full_route_level_recurrence"
        )
    ]
    partial_candidate_pairs = candidate_pairs[
        candidate_pairs["mechanism_recurrence_class"].astype(str).eq(
            "candidate_partial_route_level_recurrence"
        )
    ]
    controls = pair_rows[
        pair_rows["contract_pair_role"].astype(str).isin(
            {"positive_reference_control", "boundary_guard_control"}
        )
    ]
    control_leaks = controls[controls["fixed_016_route_predicate_pass_count"].astype(int).gt(0)]
    return pd.DataFrame(
        [
            _gate_row(
                "G1_contract_gates_pass",
                "Did every upstream route contract gate pass?",
                contract_gates["gate_status"].value_counts().to_dict(),
                "all contract gates pass",
                bool(contract_gates["gate_status"].astype(str).eq("pass").all()),
            ),
            _gate_row(
                "G2_exact_trace_scope",
                "Was execution restricted to the predeclared pair/start/fraction/seed grid?",
                {
                    "route_plan_rows": int(len(route_plan)),
                    "route_step_config_count": int(step_config_count),
                    "seed_count": int(seeds),
                    "trace_rows": int(len(trace_rows)),
                    "expected_trace_rows": expected_trace_rows,
                    "executed_pairs": sorted(trace_rows["local_pair_id"].astype(str).unique().tolist()),
                },
                "12 route rows * 9 fractions * 8 seeds = 864 trace rows, only predeclared pairs",
                int(len(route_plan)) == 12
                and int(step_config_count) == int(len(route_plan)) * len(FINE_BRIDGE_FRACTIONS)
                and int(len(trace_rows)) == expected_trace_rows
                and set(trace_rows["local_pair_id"].astype(str)) == set(ALL_EXECUTION_PAIR_IDS),
            ),
            _gate_row(
                "G3_candidate_route_recurrence_observed",
                "Does at least one strict analog pass the full fixed 016 route predicate?",
                {
                    "full_candidate_pairs": sorted(full_candidate_pairs["local_pair_id"].astype(str).tolist()),
                    "partial_candidate_pairs": sorted(partial_candidate_pairs["local_pair_id"].astype(str).tolist()),
                    "candidate_pair_rows": candidate_pairs[
                        [
                            "local_pair_id",
                            "route_count",
                            "fixed_016_route_predicate_pass_count",
                            "mechanism_recurrence_class",
                        ]
                    ].to_dict("records"),
                },
                "at least one candidate has all source-start seed routes pass",
                not full_candidate_pairs.empty,
            ),
            _gate_row(
                "G4_all_candidates_recur_under_fixed_predicate",
                "Do all three strict analogs pass the fixed route predicate?",
                {
                    "candidate_recurrence_classes": candidate_pairs[
                        ["local_pair_id", "mechanism_recurrence_class"]
                    ].to_dict("records")
                },
                "009, 012, and 020 all pass all source-start seed routes",
                len(full_candidate_pairs) == len(CANDIDATE_PAIR_IDS),
            ),
            _gate_row(
                "G5_controls_do_not_accept_full_016_predicate",
                "Do 014 and 005 avoid the full 016 route predicate?",
                {
                    "control_leak_pairs": sorted(control_leaks["local_pair_id"].astype(str).tolist()),
                    "control_rows": controls[
                        [
                            "local_pair_id",
                            "fixed_016_route_predicate_pass_count",
                            "mechanism_recurrence_class",
                        ]
                    ].to_dict("records"),
                },
                "zero full-predicate route accepts among controls",
                control_leaks.empty,
            ),
            _gate_row(
                "G6_claim_boundaries_closed",
                "Are wall, method, quality/cost, and full-replay claims closed?",
                {
                    "wall_flags_all_false": bool(
                        trace_rows["wall_generality_claim_allowed_after_trace"].eq(False).all()
                    ),
                    "method_flags_all_false": bool(
                        trace_rows["method_claim_allowed_after_trace"].eq(False).all()
                    ),
                    "quality_flags_all_false": bool(
                        trace_rows["quality_cost_claim_allowed_after_trace"].eq(False).all()
                    ),
                    "contract_boundary": CONTRACT_CLAIM_BOUNDARY,
                },
                "all claim flags false",
                bool(trace_rows["wall_generality_claim_allowed_after_trace"].eq(False).all())
                and bool(trace_rows["method_claim_allowed_after_trace"].eq(False).all())
                and bool(trace_rows["quality_cost_claim_allowed_after_trace"].eq(False).all()),
            ),
        ]
    )


def _recurrence_status(pair_rows: pd.DataFrame) -> str:
    candidates = pair_rows[
        pair_rows["contract_pair_role"].astype(str).eq(
            "strict_nonboundary_local_signature_analog"
        )
    ]
    controls = pair_rows[
        pair_rows["contract_pair_role"].astype(str).isin(
            {"positive_reference_control", "boundary_guard_control"}
        )
    ]
    control_leak = bool(controls["fixed_016_route_predicate_pass_count"].astype(int).gt(0).any())
    full_count = int(
        candidates["mechanism_recurrence_class"].astype(str).eq(
            "candidate_full_route_level_recurrence"
        ).sum()
    )
    partial_count = int(
        candidates["mechanism_recurrence_class"].astype(str).eq(
            "candidate_partial_route_level_recurrence"
        ).sum()
    )
    if control_leak:
        return "invalid_control_leak_under_fixed_predicate"
    if full_count == len(CANDIDATE_PAIR_IDS):
        return "full_route_level_recurrence_across_all_strict_analogs"
    if full_count > 0:
        return "partial_pair_level_route_recurrence_observed"
    if partial_count > 0:
        return "partial_seed_route_recurrence_only"
    return "route_level_recurrence_not_observed"


def _summary(
    *,
    contract_dir: Path,
    local_ablation_dir: Path,
    output_dir: Path,
    route_plan: pd.DataFrame,
    trace_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    fraction_rows: pd.DataFrame,
    gates: pd.DataFrame,
    step_config_count: int,
    seeds: int,
) -> dict[str, Any]:
    status = _recurrence_status(pair_rows)
    if status == "full_route_level_recurrence_across_all_strict_analogs":
        recommended_next_gate = (
            "Audit source-family equivalence and objective/barrier shape for "
            "the recurring analogs before any wall or method language."
        )
    elif status in {
        "partial_pair_level_route_recurrence_observed",
        "partial_seed_route_recurrence_only",
    }:
        recommended_next_gate = (
            "Stratify recurrence by pair, start, and seed; keep the fixed "
            "predicate and do not broaden thresholds until the partial route "
            "mechanism is explained."
        )
    elif status == "invalid_control_leak_under_fixed_predicate":
        recommended_next_gate = (
            "Do not generalize the 016 predicate. Inspect why controls accept "
            "the predicate and tighten the source/target/transition readout."
        )
    else:
        recommended_next_gate = (
            "Treat the 016 route mechanism as not reproduced in these strict "
            "analogs under the fixed predicate. Revisit candidate selection or "
            "route-state definition before more execution."
        )
    return {
        "schema": "nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_summary.v1",
        "status": RUN_STATUS,
        "mechanism_route_recurrence_status": status,
        "contract_dir": str(contract_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "candidate_pair_ids": list(CANDIDATE_PAIR_IDS),
        "control_pair_ids": list(CONTROL_PAIR_IDS),
        "route_plan_row_count": int(len(route_plan)),
        "route_step_config_count": int(step_config_count),
        "seed_count": int(seeds),
        "trace_row_count": int(len(trace_rows)),
        "route_mechanism_row_count": int(len(route_rows)),
        "pair_mechanism_row_count": int(len(pair_rows)),
        "fraction_mechanism_row_count": int(len(fraction_rows)),
        "pair_recurrence_classes": {
            str(row.local_pair_id): str(row.mechanism_recurrence_class)
            for row in pair_rows.itertuples(index=False)
        },
        "pair_full_predicate_counts": {
            str(row.local_pair_id): int(row.fixed_016_route_predicate_pass_count)
            for row in pair_rows.itertuples(index=False)
        },
        "route_mechanism_class_counts": route_rows[
            "route_mechanism_class"
        ].value_counts().to_dict()
        if not route_rows.empty
        else {},
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "gate_status_counts": gates["gate_status"].value_counts().to_dict(),
        "interpretation": (
            "The trace tests route-level recurrence of the fixed 016 mechanism "
            "predicate only. Substrate recurrence, wall claims, method claims, "
            "full replay, and quality/cost value remain separate questions."
        ),
        "recommended_next_gate": recommended_next_gate,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    pair_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    fraction_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass Mechanism Generalization Route Trace",
        "",
        "## Summary",
        "",
        f"- status: {summary['status']}",
        f"- mechanism_route_recurrence_status: {summary['mechanism_route_recurrence_status']}",
        f"- trace_row_count: {summary['trace_row_count']}",
        f"- pair_full_predicate_counts: {summary['pair_full_predicate_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        "",
        "## Pair Mechanism Rows",
        "",
        _markdown_table(
            pair_rows,
            [
                "local_pair_id",
                "contract_pair_role",
                "route_count",
                "fixed_016_route_predicate_pass_count",
                "source_family_start_count",
                "finite_single_side_band_count",
                "final_target_like_count",
                "mechanism_recurrence_class",
                "route_mechanism_class_counts",
            ],
            max_rows=20,
        ),
        "",
        "## Route Mechanism Rows",
        "",
        _markdown_table(
            route_rows,
            [
                "local_pair_id",
                "contract_pair_role",
                "start_condition",
                "seed",
                "source_family_start",
                "single_side_fraction_count",
                "single_side_fractions",
                "single_side_adjacent_fraction_band",
                "final_target_like",
                "fixed_016_route_predicate_pass",
                "route_mechanism_class",
            ],
            max_rows=80,
        ),
        "",
        "## Fraction Mechanism Rows",
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
                "dominant_mechanism_read",
                "mechanism_read_counts",
            ],
            max_rows=80,
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


def run_trace(
    *,
    contract_dir: Path = DEFAULT_CONTRACT_DIR,
    local_ablation_dir: Path = DEFAULT_LOCAL_ABLATION_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    gamma: float = 1.0e-5,
    seeds: int = 8,
    n_iterations: int = 2,
    edge_chunk_size: int = 5_000_000,
) -> dict[str, Any]:
    route_plan = _read_csv(contract_dir / CONTRACT_ROUTE_PLAN_ROWS_CSV)
    fraction_steps = _read_csv(contract_dir / CONTRACT_FRACTION_STEP_ROWS_CSV)
    contract_gates = _read_csv(contract_dir / CONTRACT_GATE_MATRIX_CSV)
    readout_rules = _read_csv(contract_dir / CONTRACT_READOUT_RULE_ROWS_CSV)
    _install_contract_schedule(fraction_steps)
    trace_rows, step_config_count, candidate_pair_count = scoped_trace._trace_rows(
        route_plan=route_plan,
        contract_dir=contract_dir,
        local_ablation_dir=local_ablation_dir,
        gamma=float(gamma),
        seeds=int(seeds),
        n_iterations=int(n_iterations),
        edge_chunk_size=int(edge_chunk_size),
    )
    trace_rows = _annotate_trace_rows(trace_rows, route_plan)
    trace_rows = _override_status_columns(trace_rows)
    if candidate_pair_count != len(ALL_EXECUTION_PAIR_IDS):
        raise ValueError(
            f"expected {len(ALL_EXECUTION_PAIR_IDS)} executed pairs, got {candidate_pair_count}"
        )
    route_rows = _route_mechanism_rows(trace_rows)
    pair_rows = _pair_mechanism_rows(route_rows)
    fraction_rows = _fraction_mechanism_rows(trace_rows)
    gates = _gate_matrix(
        contract_gates=contract_gates,
        route_plan=route_plan,
        trace_rows=trace_rows,
        route_rows=route_rows,
        pair_rows=pair_rows,
        step_config_count=step_config_count,
        seeds=seeds,
    )
    summary = _summary(
        contract_dir=contract_dir,
        local_ablation_dir=local_ablation_dir,
        output_dir=output_dir,
        route_plan=route_plan,
        trace_rows=trace_rows,
        route_rows=route_rows,
        pair_rows=pair_rows,
        fraction_rows=fraction_rows,
        gates=gates,
        step_config_count=step_config_count,
        seeds=seeds,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(trace_rows, output_dir / TRACE_ROWS_CSV)
    _write_csv(route_rows, output_dir / ROUTE_MECHANISM_ROWS_CSV)
    _write_csv(pair_rows, output_dir / PAIR_MECHANISM_ROWS_CSV)
    _write_csv(fraction_rows, output_dir / FRACTION_MECHANISM_ROWS_CSV)
    _write_csv(gates, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_mechanism_generalization_route_trace_config.v1",
        "contract_dir": str(contract_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "gamma": float(gamma),
        "seeds": int(seeds),
        "n_iterations": int(n_iterations),
        "edge_chunk_size": int(edge_chunk_size),
        "planned_route_family": PLANNED_ROUTE_FAMILY,
        "fine_bridge_fractions": list(map(float, FINE_BRIDGE_FRACTIONS)),
        "source_family_mechanisms": sorted(SOURCE_FAMILY_MECHANISMS),
        "single_side_mechanism": SINGLE_SIDE_MECHANISM,
        "target_like_mechanism": TARGET_LIKE_MECHANISM,
        "readout_rules": readout_rules.to_dict("records"),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        pair_rows=pair_rows,
        route_rows=route_rows,
        fraction_rows=fraction_rows,
        gates=gates,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT_DIR)
    parser.add_argument("--local-ablation-dir", type=Path, default=DEFAULT_LOCAL_ABLATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gamma", type=float, default=1.0e-5)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--n-iterations", type=int, default=2)
    parser.add_argument("--edge-chunk-size", type=int, default=5_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_trace(
        contract_dir=Path(args.contract_dir),
        local_ablation_dir=Path(args.local_ablation_dir),
        output_dir=Path(args.output_dir),
        gamma=float(args.gamma),
        seeds=int(args.seeds),
        n_iterations=int(args.n_iterations),
        edge_chunk_size=int(args.edge_chunk_size),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
