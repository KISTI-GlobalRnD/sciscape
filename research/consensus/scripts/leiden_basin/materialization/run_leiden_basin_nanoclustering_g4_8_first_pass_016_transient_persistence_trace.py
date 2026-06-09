#!/usr/bin/env python3
"""Execute the local_pair_016 transient persistence contract.

This runner consumes
``design_leiden_basin_nanoclustering_g4_8_first_pass_016_transient_persistence_contract.py``.
It executes only the three predeclared ``local_pair_016`` route rows and their
fine bridge-fraction schedule, then classifies whether the previous recurrent
step-2 transient is a finite fraction band or a point-only saddle.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

import run_leiden_basin_nanoclustering_g4_8_scoped_pathway_probe_trace as scoped_trace
from design_leiden_basin_nanoclustering_g4_8_first_pass_016_transient_persistence_contract import (
    CLAIM_BOUNDARY as CONTRACT_CLAIM_BOUNDARY,
    DEFAULT_OUTPUT_DIR as DEFAULT_CONTRACT_DIR,
    FRACTION_STEP_ROWS_CSV as CONTRACT_FRACTION_STEP_ROWS_CSV,
    GATE_MATRIX_CSV as CONTRACT_GATE_MATRIX_CSV,
    PLANNED_ROUTE_FAMILY,
    PRIMARY_PAIR_ID,
    ROUTE_PLAN_ROWS_CSV as CONTRACT_ROUTE_PLAN_ROWS_CSV,
    TARGET_SIGNATURE_ID,
    TRANSIENT_SIGNATURE_ID,
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
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_transient_persistence_trace_gamma1e5_20260605"
)

TRACE_ROWS_CSV = "nanoclustering_g4_8_first_pass_016_transient_persistence_trace_rows.csv"
ROUTE_PERSISTENCE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_transient_persistence_route_rows.csv"
)
FRACTION_SUMMARY_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_transient_persistence_fraction_rows.csv"
)
GATE_MATRIX_CSV = "nanoclustering_g4_8_first_pass_016_transient_persistence_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_016_transient_persistence_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_016_transient_persistence_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_016_transient_persistence_report.md"

RUN_STATUS = "executed_nanoclustering_g4_8_first_pass_016_transient_persistence_trace"
ROUTE_EXECUTION_STATUS = "executed_016_transient_fine_bridge_persistence_trace"
WALL_PROMOTION_STATUS = "not_promoted_016_transient_persistence_only"
METHOD_STATUS = "transient_persistence_trace_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass local_pair_016 transient-persistence trace "
    "only; executes the predeclared fine bridge-fraction scan around the "
    "previous recurrent step-2 transient. It does not promote basin walls, run "
    "reverse hysteresis, replay full NanoClustering, evaluate quality/cost "
    "value, or claim method/algorithm success."
)

EPS = 1e-9
SUPPORT_ANCHOR_DISTANCE_COLUMNS = (
    "support_distance_to_original",
    "support_distance_to_drop_bridge_edges",
    "support_distance_to_drop_direct_edge",
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
        steps = []
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


def _support_equidistant_to_three_anchors(row: pd.Series) -> bool:
    values: list[float] = []
    for column in SUPPORT_ANCHOR_DISTANCE_COLUMNS:
        value = row.get(column)
        if value is None or pd.isna(value):
            return False
        values.append(float(value))
    return max(values) - min(values) <= EPS


def _endpoint_class(row: pd.Series) -> str:
    signature_id = str(row["result_endpoint_signature_id"])
    assignment = str(row["endpoint_assignment_by_step"])
    if signature_id == TRANSIENT_SIGNATURE_ID:
        return "transient_signature"
    if signature_id == TARGET_SIGNATURE_ID or "drop_bridge_target_anchor" in assignment:
        return "target_anchor"
    if _as_bool(row.get("matches_original_anchor", False)):
        return "source_anchor"
    if assignment == "unknown_new_endpoint":
        return "unknown_other"
    return "other_anchor"


def _route_persistence_rows(trace_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if trace_rows.empty:
        return pd.DataFrame(rows)
    classified = trace_rows.copy()
    classified["endpoint_class"] = classified.apply(_endpoint_class, axis=1)
    classified["support_equidistant_to_three_primary_anchors"] = classified.apply(
        _support_equidistant_to_three_anchors,
        axis=1,
    )
    group_cols = [
        "route_contract_id",
        "validation_unit_id",
        "local_pair_id",
        "start_condition",
        "planned_route_family",
        "seed",
    ]
    for keys, group in classified.groupby(group_cols, sort=False):
        key_data = dict(zip(group_cols, keys, strict=True))
        ordered = group.sort_values("step_index", kind="mergesort").copy()
        saddle = ordered[ordered["endpoint_class"].eq("transient_signature")]
        target = ordered[ordered["endpoint_class"].eq("target_anchor")]
        source = ordered[ordered["endpoint_class"].eq("source_anchor")]
        saddle_fractions = [
            float(value) for value in saddle["bridge_edge_weight_fraction"].tolist()
        ]
        saddle_indices = [int(value) for value in saddle["step_index"].tolist()]
        adjacent_band = (
            len(saddle_indices) >= 2
            and max(saddle_indices) - min(saddle_indices) == len(saddle_indices) - 1
        )
        saddle_equidistant_count = int(
            saddle["support_equidistant_to_three_primary_anchors"].astype(bool).sum()
        )
        if len(saddle_fractions) >= 2 and adjacent_band and saddle_equidistant_count == len(saddle_fractions):
            route_class = "finite_saddle_band_candidate"
        elif len(saddle_fractions) == 1 and math.isclose(saddle_fractions[0], 0.75, abs_tol=EPS):
            route_class = "point_saddle_only_candidate"
        elif len(saddle_fractions) == 1:
            route_class = "shifted_point_saddle_candidate"
        elif len(saddle_fractions) > 1:
            route_class = "fragmented_saddle_candidate"
        else:
            route_class = "saddle_absent_after_fine_scan"
        endpoint_sequence = " -> ".join(
            f"{float(row.bridge_edge_weight_fraction):.5g}:{row.endpoint_class}"
            for row in ordered.itertuples(index=False)
        )
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
                "saddle_fraction_count": int(len(saddle_fractions)),
                "saddle_fractions": ";".join(f"{value:.5g}" for value in saddle_fractions),
                "saddle_fraction_min": min(saddle_fractions) if saddle_fractions else None,
                "saddle_fraction_max": max(saddle_fractions) if saddle_fractions else None,
                "saddle_fraction_width": (
                    max(saddle_fractions) - min(saddle_fractions)
                    if len(saddle_fractions) >= 2
                    else 0.0
                ),
                "saddle_adjacent_fraction_band": bool(adjacent_band),
                "saddle_support_equidistant_fraction_count": saddle_equidistant_count,
                "target_fraction_count": int(len(target)),
                "target_fractions": ";".join(
                    f"{float(value):.5g}" for value in target["bridge_edge_weight_fraction"].tolist()
                ),
                "source_fraction_count": int(len(source)),
                "source_fractions": ";".join(
                    f"{float(value):.5g}" for value in source["bridge_edge_weight_fraction"].tolist()
                ),
                "endpoint_class_sequence": endpoint_sequence,
                "distinct_signature_count": int(ordered["result_endpoint_signature_id"].nunique()),
                "objective_monotone_nonincreasing_with_bridge_release": bool(
                    all(delta <= EPS for delta in objective_diffs)
                ),
                "max_objective_debt_from_start": float(ordered["objective_debt_from_start"].max()),
                "max_objective_recovery_from_min": float(
                    ordered["objective_recovery_from_min"].max()
                ),
                "route_persistence_class": route_class,
                "wall_claim_ready_after_trace": False,
                "wall_claim_block_reason": (
                    "persistence trace classifies the 016 transient but does not "
                    "include reverse hysteresis, endpoint-basin proof, or positive "
                    "objective recovery evidence"
                ),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows)


def _fraction_summary_rows(trace_rows: pd.DataFrame) -> pd.DataFrame:
    if trace_rows.empty:
        return pd.DataFrame()
    classified = trace_rows.copy()
    classified["endpoint_class"] = classified.apply(_endpoint_class, axis=1)
    classified["support_equidistant_to_three_primary_anchors"] = classified.apply(
        _support_equidistant_to_three_anchors,
        axis=1,
    )
    rows: list[dict[str, Any]] = []
    for fraction, group in classified.groupby("bridge_edge_weight_fraction", sort=False):
        signature_counts = group["result_endpoint_signature_id"].astype(str).value_counts()
        dominant_signature_id = str(signature_counts.index[0]) if not signature_counts.empty else ""
        dominant_count = int(signature_counts.iloc[0]) if not signature_counts.empty else 0
        rows.append(
            {
                "bridge_edge_weight_fraction": float(fraction),
                "trace_row_count": int(len(group)),
                "route_count": int(
                    group[["route_contract_id", "seed"]].drop_duplicates().shape[0]
                ),
                "source_anchor_count": int(group["endpoint_class"].eq("source_anchor").sum()),
                "transient_signature_count": int(
                    group["endpoint_class"].eq("transient_signature").sum()
                ),
                "target_anchor_count": int(group["endpoint_class"].eq("target_anchor").sum()),
                "unknown_other_count": int(group["endpoint_class"].eq("unknown_other").sum()),
                "other_anchor_count": int(group["endpoint_class"].eq("other_anchor").sum()),
                "distinct_signature_count": int(group["result_endpoint_signature_id"].nunique()),
                "dominant_signature_id": dominant_signature_id,
                "dominant_signature_count": dominant_count,
                "support_equidistant_to_three_primary_anchors_count": int(
                    group["support_equidistant_to_three_primary_anchors"].astype(bool).sum()
                ),
                "objective_value_mean": float(group["objective_value_by_step"].mean()),
                "objective_value_min": float(group["objective_value_by_step"].min()),
                "objective_value_max": float(group["objective_value_by_step"].max()),
                "route_execution_status": ROUTE_EXECUTION_STATUS,
                "wall_promotion_status": WALL_PROMOTION_STATUS,
                "method_status": METHOD_STATUS,
                "claim_boundary": CLAIM_BOUNDARY,
                "run_status": RUN_STATUS,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "bridge_edge_weight_fraction",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)


def _semantic_persistence_class(route_rows: pd.DataFrame) -> str:
    if route_rows.empty:
        return "not_classified_no_route_rows"
    expected = int(len(route_rows))
    finite = int(route_rows["route_persistence_class"].eq("finite_saddle_band_candidate").sum())
    point = int(route_rows["route_persistence_class"].eq("point_saddle_only_candidate").sum())
    shifted = int(route_rows["route_persistence_class"].eq("shifted_point_saddle_candidate").sum())
    fragmented = int(route_rows["route_persistence_class"].eq("fragmented_saddle_candidate").sum())
    absent = int(route_rows["route_persistence_class"].eq("saddle_absent_after_fine_scan").sum())
    if finite == expected:
        return "persistent_finite_saddle_band_candidate_not_wall"
    if point == expected:
        return "point_saddle_only_candidate_not_persistent_band"
    if absent == expected:
        return "coarse_transient_not_reproduced_under_fine_scan"
    if shifted + point == expected:
        return "point_saddle_candidate_with_fraction_shift"
    if finite + fragmented == expected:
        return "multi_fraction_saddle_candidate_mixed_band_quality"
    return "mixed_seed_start_persistence_class"


def _gate_matrix(
    *,
    contract_gates: pd.DataFrame,
    route_plan: pd.DataFrame,
    fraction_steps: pd.DataFrame,
    trace_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    fraction_rows: pd.DataFrame,
    step_config_count: int,
    seeds: int,
) -> pd.DataFrame:
    expected_trace_rows = int(step_config_count) * int(seeds)
    expected_seed_routes = int(len(route_plan)) * int(seeds)
    saddle_route_count = (
        int(route_rows["saddle_fraction_count"].gt(0).sum()) if not route_rows.empty else 0
    )
    finite_band_route_count = (
        int(route_rows["route_persistence_class"].eq("finite_saddle_band_candidate").sum())
        if not route_rows.empty
        else 0
    )
    saddle_rows = trace_rows[
        trace_rows["result_endpoint_signature_id"].astype(str).eq(TRANSIENT_SIGNATURE_ID)
    ]
    saddle_support_equidistant_count = int(
        saddle_rows.apply(_support_equidistant_to_three_anchors, axis=1).sum()
    ) if not saddle_rows.empty else 0
    return pd.DataFrame(
        [
            _gate_row(
                "G1_contract_gates_pass",
                "Did every upstream 016 persistence contract gate pass?",
                contract_gates["gate_status"].value_counts().to_dict(),
                "all contract gates pass",
                bool(contract_gates["gate_status"].astype(str).eq("pass").all()),
            ),
            _gate_row(
                "G2_exact_trace_scope",
                "Was execution restricted to the predeclared 016 route/fraction/seed grid?",
                {
                    "route_plan_rows": len(route_plan),
                    "step_config_count": step_config_count,
                    "seeds": seeds,
                    "trace_rows": len(trace_rows),
                    "expected_trace_rows": expected_trace_rows,
                    "executed_pairs": sorted(trace_rows["local_pair_id"].astype(str).unique().tolist()),
                },
                "3 route rows * 9 fractions * 8 seeds = 216 trace rows, only local_pair_016",
                len(route_plan) == 3
                and int(step_config_count) == 27
                and len(trace_rows) == expected_trace_rows
                and set(trace_rows["local_pair_id"].astype(str)) == {PRIMARY_PAIR_ID},
            ),
            _gate_row(
                "G3_transient_reproduced_in_fine_scan",
                "Does the previous recurrent transient signature reappear after fine scanning?",
                {
                    "seed_route_count": len(route_rows),
                    "saddle_route_count": saddle_route_count,
                    "fraction_rows": fraction_rows[
                        [
                            "bridge_edge_weight_fraction",
                            "transient_signature_count",
                            "dominant_signature_id",
                        ]
                    ].to_dict("records")
                    if not fraction_rows.empty
                    else [],
                },
                "transient signature appears in all 24 seed routes",
                len(route_rows) == expected_seed_routes
                and saddle_route_count == expected_seed_routes,
            ),
            _gate_row(
                "G4_finite_saddle_band_evidence",
                "Is the transient a finite adjacent fraction band rather than a single point?",
                {
                    "route_persistence_class_counts": route_rows[
                        "route_persistence_class"
                    ].value_counts().to_dict()
                    if not route_rows.empty
                    else {},
                    "finite_band_route_count": finite_band_route_count,
                    "expected_seed_routes": expected_seed_routes,
                },
                "all 24 seed routes have the transient at two or more adjacent fine fractions",
                len(route_rows) == expected_seed_routes
                and finite_band_route_count == expected_seed_routes,
            ),
            _gate_row(
                "G5_support_geometry_still_blocks_endpoint_promotion",
                "Do transient rows remain support-equidistant to the primary anchors?",
                {
                    "saddle_trace_rows": len(saddle_rows),
                    "saddle_support_equidistant_count": saddle_support_equidistant_count,
                },
                "every transient row is equidistant to original/drop-bridge/drop-direct anchors",
                not saddle_rows.empty
                and saddle_support_equidistant_count == len(saddle_rows),
            ),
            _gate_row(
                "G6_claim_boundaries_closed",
                "Are wall, method, quality/cost, reverse-hysteresis, and full replay claims closed?",
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
                    "wall_promotion_status": WALL_PROMOTION_STATUS,
                    "contract_boundary": CONTRACT_CLAIM_BOUNDARY,
                },
                "all claim flags false and wall promotion status closed",
                bool(trace_rows["wall_generality_claim_allowed_after_trace"].eq(False).all())
                and bool(trace_rows["method_claim_allowed_after_trace"].eq(False).all())
                and bool(trace_rows["quality_cost_claim_allowed_after_trace"].eq(False).all()),
            ),
        ]
    )


def _summary(
    *,
    contract_dir: Path,
    local_ablation_dir: Path,
    output_dir: Path,
    route_plan: pd.DataFrame,
    fraction_steps: pd.DataFrame,
    trace_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    fraction_rows: pd.DataFrame,
    gates: pd.DataFrame,
    step_config_count: int,
    candidate_pair_count: int,
    seeds: int,
) -> dict[str, Any]:
    persistence_class = _semantic_persistence_class(route_rows)
    if persistence_class == "persistent_finite_saddle_band_candidate_not_wall":
        recommended_next_gate = (
            "Design a same-seed target-anchor reverse trace to test hysteresis; "
            "do not promote wall language until reverse evidence and objective "
            "recovery/debt behavior are audited."
        )
    elif persistence_class == "point_saddle_only_candidate_not_persistent_band":
        recommended_next_gate = (
            "Treat 016 as a recurrent but point-local gateway object, not a "
            "finite basin band. Revisit the basin/pathway definition before "
            "building a reverse runner."
        )
    else:
        recommended_next_gate = (
            "Stratify by start condition and seed before reverse execution; the "
            "fine scan did not produce a single clean persistence class."
        )
    return {
        "schema": "nanoclustering_g4_8_first_pass_016_transient_persistence_summary.v1",
        "status": RUN_STATUS,
        "contract_dir": str(contract_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "primary_pair": PRIMARY_PAIR_ID,
        "planned_route_family": PLANNED_ROUTE_FAMILY,
        "candidate_pair_count": int(candidate_pair_count),
        "route_plan_row_count": int(len(route_plan)),
        "fraction_step_row_count": int(len(fraction_steps)),
        "route_step_config_count": int(step_config_count),
        "seed_count": int(seeds),
        "trace_row_count": int(len(trace_rows)),
        "route_persistence_row_count": int(len(route_rows)),
        "fraction_summary_row_count": int(len(fraction_rows)),
        "semantic_persistence_class": persistence_class,
        "route_persistence_class_counts": route_rows["route_persistence_class"].value_counts().to_dict()
        if not route_rows.empty
        else {},
        "fraction_transient_counts": {
            f"{float(row.bridge_edge_weight_fraction):.5g}": int(row.transient_signature_count)
            for row in fraction_rows.itertuples(index=False)
        }
        if not fraction_rows.empty
        else {},
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "gate_status_counts": gates["gate_status"].value_counts().to_dict(),
        "transient_signature_id": TRANSIENT_SIGNATURE_ID,
        "target_signature_id": TARGET_SIGNATURE_ID,
        "interpretation": (
            "The run executes a narrow fine-fraction persistence test for the "
            "016 transient. It classifies persistence behavior only; basin-wall, "
            "reverse-hysteresis, full-replay, method, and quality/cost claims "
            "remain closed."
        ),
        "recommended_next_gate": recommended_next_gate,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    path: Path,
    summary: dict[str, Any],
    route_rows: pd.DataFrame,
    fraction_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass 016 Transient Persistence Trace",
        "",
        "## Summary",
        "",
        f"- status: {summary['status']}",
        f"- semantic_persistence_class: {summary['semantic_persistence_class']}",
        f"- trace_row_count: {summary['trace_row_count']}",
        f"- route_persistence_class_counts: {summary['route_persistence_class_counts']}",
        f"- failed_gates: {summary['failed_gates']}",
        "",
        "## Fraction Summary",
        "",
        _markdown_table(
            fraction_rows,
            [
                "bridge_edge_weight_fraction",
                "route_count",
                "source_anchor_count",
                "transient_signature_count",
                "target_anchor_count",
                "unknown_other_count",
                "dominant_signature_id",
                "support_equidistant_to_three_primary_anchors_count",
                "objective_value_mean",
            ],
            max_rows=20,
        ),
        "",
        "## Route Persistence Rows",
        "",
        _markdown_table(
            route_rows,
            [
                "route_key",
                "saddle_fraction_count",
                "saddle_fractions",
                "target_fractions",
                "source_fractions",
                "route_persistence_class",
                "max_objective_debt_from_start",
                "max_objective_recovery_from_min",
            ],
            max_rows=40,
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
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
        "## Recommended Next Gate",
        "",
        summary["recommended_next_gate"],
        "",
        "## Claim Boundary",
        "",
        summary["claim_boundary"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT_DIR)
    parser.add_argument("--local-ablation-dir", type=Path, default=DEFAULT_LOCAL_ABLATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gamma", type=float, default=1e-5)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--n-iterations", type=int, default=2)
    parser.add_argument("--edge-chunk-size", type=int, default=1_000_000)
    args = parser.parse_args()

    contract_dir = Path(args.contract_dir)
    local_ablation_dir = Path(args.local_ablation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    route_plan = _read_csv(contract_dir / CONTRACT_ROUTE_PLAN_ROWS_CSV)
    fraction_steps = _read_csv(contract_dir / CONTRACT_FRACTION_STEP_ROWS_CSV)
    contract_gates = _read_csv(contract_dir / CONTRACT_GATE_MATRIX_CSV)
    _install_contract_schedule(fraction_steps)

    trace_rows, step_config_count, candidate_pair_count = scoped_trace._trace_rows(
        route_plan=route_plan,
        contract_dir=contract_dir,
        local_ablation_dir=local_ablation_dir,
        gamma=float(args.gamma),
        seeds=int(args.seeds),
        n_iterations=int(args.n_iterations),
        edge_chunk_size=int(args.edge_chunk_size),
    )
    trace_rows = _override_status_columns(trace_rows)
    route_rows = _route_persistence_rows(trace_rows)
    fraction_rows = _fraction_summary_rows(trace_rows)
    gates = _gate_matrix(
        contract_gates=contract_gates,
        route_plan=route_plan,
        fraction_steps=fraction_steps,
        trace_rows=trace_rows,
        route_rows=route_rows,
        fraction_rows=fraction_rows,
        step_config_count=step_config_count,
        seeds=int(args.seeds),
    )
    summary = _summary(
        contract_dir=contract_dir,
        local_ablation_dir=local_ablation_dir,
        output_dir=output_dir,
        route_plan=route_plan,
        fraction_steps=fraction_steps,
        trace_rows=trace_rows,
        route_rows=route_rows,
        fraction_rows=fraction_rows,
        gates=gates,
        step_config_count=step_config_count,
        candidate_pair_count=candidate_pair_count,
        seeds=int(args.seeds),
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_016_transient_persistence_config.v1",
        "contract_dir": str(contract_dir),
        "local_ablation_dir": str(local_ablation_dir),
        "output_dir": str(output_dir),
        "gamma": float(args.gamma),
        "seeds": int(args.seeds),
        "n_iterations": int(args.n_iterations),
        "edge_chunk_size": int(args.edge_chunk_size),
        "claim_boundary": CLAIM_BOUNDARY,
    }

    _write_csv(trace_rows, output_dir / TRACE_ROWS_CSV)
    _write_csv(route_rows, output_dir / ROUTE_PERSISTENCE_ROWS_CSV)
    _write_csv(fraction_rows, output_dir / FRACTION_SUMMARY_ROWS_CSV)
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
        path=output_dir / REPORT_MD,
        summary=summary,
        route_rows=route_rows,
        fraction_rows=fraction_rows,
        gates=gates,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
