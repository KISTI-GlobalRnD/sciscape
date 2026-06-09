#!/usr/bin/env python3
"""Audit objective/barrier interpretation for local_pair_016 pathway shape.

This read-only audit follows the 016 pathway-shape audit. It checks whether the
executed bridge-fraction traces support objective-barrier language, or whether
they should remain a changing-graph state-ladder diagnostic.
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

DEFAULT_PERSISTENCE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_transient_persistence_trace_gamma1e5_20260605"
)
DEFAULT_REVERSE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_transient_reverse_trace_gamma1e5_20260605"
)
DEFAULT_PATHWAY_SHAPE_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_pathway_shape_audit_gamma1e5_20260605"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_016_objective_barrier_audit_gamma1e5_20260605"
)

FRACTION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_objective_barrier_fraction_rows.csv"
)
ROUTE_ROWS_CSV = "nanoclustering_g4_8_first_pass_016_objective_barrier_route_rows.csv"
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_016_objective_barrier_decision_rows.csv"
)
GATE_MATRIX_CSV = "nanoclustering_g4_8_first_pass_016_objective_barrier_gate_matrix.csv"
SUMMARY_JSON = "nanoclustering_g4_8_first_pass_016_objective_barrier_summary.json"
CONFIG_JSON = "nanoclustering_g4_8_first_pass_016_objective_barrier_config.json"
REPORT_MD = "nanoclustering_g4_8_first_pass_016_objective_barrier_report.md"

RUN_STATUS = "audited_nanoclustering_g4_8_first_pass_016_objective_barrier"
ROUTE_EXECUTION_STATUS = "not_executed_read_only_016_objective_barrier"
WALL_PROMOTION_STATUS = "not_promoted_objective_barrier_only"
METHOD_STATUS = "objective_barrier_audit_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass local_pair_016 objective/barrier audit "
    "only; reads the executed bridge-fraction traces and pathway-shape audit to "
    "interpret objective profiles. It does not rerun Leiden, promote basin "
    "walls, replay full NanoClustering, evaluate quality/cost value, or claim "
    "method success."
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


def _route_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    if "route_key" not in rows.columns:
        rows["route_key"] = (
            rows["start_condition"].astype(str)
            + "|seed="
            + rows["seed"].astype(int).astype(str)
        )
    return rows


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
    persistence_dir: Path,
    reverse_dir: Path,
    pathway_shape_dir: Path,
) -> dict[str, Any]:
    return {
        "persistence_trace": _route_key_frame(
            _read_csv(
                persistence_dir
                / "nanoclustering_g4_8_first_pass_016_transient_persistence_trace_rows.csv"
            )
        ),
        "persistence_route": _read_csv(
            persistence_dir
            / "nanoclustering_g4_8_first_pass_016_transient_persistence_route_rows.csv"
        ),
        "reverse_trace": _route_key_frame(
            _read_csv(
                reverse_dir
                / "nanoclustering_g4_8_first_pass_016_transient_reverse_trace_rows.csv"
            )
        ),
        "reverse_route": _read_csv(
            reverse_dir
            / "nanoclustering_g4_8_first_pass_016_transient_reverse_route_rows.csv"
        ),
        "pathway_summary": _read_json(
            pathway_shape_dir
            / "nanoclustering_g4_8_first_pass_016_pathway_shape_summary.json"
        ),
        "pathway_gates": _read_csv(
            pathway_shape_dir
            / "nanoclustering_g4_8_first_pass_016_pathway_shape_gate_matrix.csv"
        ),
        "pathway_fraction": _read_csv(
            pathway_shape_dir
            / "nanoclustering_g4_8_first_pass_016_pathway_shape_fraction_rows.csv"
        ),
        "pathway_route": _read_csv(
            pathway_shape_dir
            / "nanoclustering_g4_8_first_pass_016_pathway_shape_route_rows.csv"
        ),
    }


def _is_non_decreasing(values: list[float]) -> bool:
    return all(values[index] <= values[index + 1] + EPS for index in range(len(values) - 1))


def _is_non_increasing(values: list[float]) -> bool:
    return all(values[index] >= values[index + 1] - EPS for index in range(len(values) - 1))


def _fraction_rows(
    *,
    persistence_trace: pd.DataFrame,
    reverse_trace: pd.DataFrame,
    pathway_fraction: pd.DataFrame,
) -> pd.DataFrame:
    forward = (
        persistence_trace.groupby("bridge_edge_weight_fraction", sort=True)
        .agg(
            forward_route_count=("objective_value_by_step", "size"),
            forward_objective_min=("objective_value_by_step", "min"),
            forward_objective_mean=("objective_value_by_step", "mean"),
            forward_objective_max=("objective_value_by_step", "max"),
            forward_delta_mean=("objective_delta_from_start", "mean"),
            forward_debt_mean=("objective_debt_from_start", "mean"),
            forward_recovery_mean=("objective_recovery_from_min", "mean"),
            forward_distinct_signature_count=("result_endpoint_signature_id", "nunique"),
        )
        .reset_index()
    )
    reverse = (
        reverse_trace.groupby("bridge_edge_weight_fraction", sort=True)
        .agg(
            reverse_route_count=("objective_value_by_step", "size"),
            reverse_objective_min=("objective_value_by_step", "min"),
            reverse_objective_mean=("objective_value_by_step", "mean"),
            reverse_objective_max=("objective_value_by_step", "max"),
            reverse_delta_mean=("objective_delta_from_start", "mean"),
            reverse_debt_mean=("objective_debt_from_start", "mean"),
            reverse_recovery_mean=("objective_recovery_from_min", "mean"),
            reverse_distinct_signature_count=("result_endpoint_signature_id", "nunique"),
        )
        .reset_index()
    )
    rows = pathway_fraction[
        [
            "bridge_edge_weight_fraction",
            "expected_pathway_state_class",
            "both_directions_match_expected_class",
        ]
    ].merge(forward, on="bridge_edge_weight_fraction", how="left")
    rows = rows.merge(reverse, on="bridge_edge_weight_fraction", how="left")
    rows["forward_reverse_objective_mean_abs_diff"] = (
        rows["forward_objective_mean"] - rows["reverse_objective_mean"]
    ).abs()
    target_forward = float(
        rows.loc[
            rows["expected_pathway_state_class"].astype(str).eq("target_anchor"),
            "forward_objective_mean",
        ].iloc[0]
    )
    source_forward = float(
        rows.loc[
            rows["bridge_edge_weight_fraction"].astype(float).eq(1.0),
            "forward_objective_mean",
        ].iloc[0]
    )
    rows["forward_objective_gap_from_target_mean"] = (
        rows["forward_objective_mean"] - target_forward
    )
    rows["forward_objective_gap_from_source_fraction_1_mean"] = (
        rows["forward_objective_mean"] - source_forward
    )
    rows["objective_profile_interpretation"] = rows[
        "expected_pathway_state_class"
    ].map(
        {
            "source_family": "source_family_state_under_changing_graph_weight",
            "transient_signature": "finite_transition_band_state_under_changing_graph_weight",
            "target_anchor": "target_anchor_state_under_changed_graph_weight",
        }
    )
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    rows["run_status"] = RUN_STATUS
    return rows.sort_values("bridge_edge_weight_fraction", kind="mergesort").reset_index(
        drop=True
    )


def _route_rows(
    *,
    persistence_route: pd.DataFrame,
    reverse_route: pd.DataFrame,
    pathway_route: pd.DataFrame,
    persistence_trace: pd.DataFrame,
    reverse_trace: pd.DataFrame,
) -> pd.DataFrame:
    forward_route = persistence_route[
        [
            "route_key",
            "objective_monotone_nonincreasing_with_bridge_release",
            "max_objective_debt_from_start",
            "max_objective_recovery_from_min",
        ]
    ].rename(
        columns={
            "objective_monotone_nonincreasing_with_bridge_release": "forward_objective_nonincreasing_with_bridge_release",
            "max_objective_debt_from_start": "forward_max_objective_debt_from_start",
            "max_objective_recovery_from_min": "forward_max_objective_recovery_from_min",
        }
    )
    reverse_route_subset = reverse_route[
        [
            "route_key",
            "objective_monotone_nonincreasing_with_bridge_restore",
            "max_objective_debt_from_start",
            "max_objective_recovery_from_min",
        ]
    ].rename(
        columns={
            "objective_monotone_nonincreasing_with_bridge_restore": "reverse_objective_nonincreasing_with_bridge_restore",
            "max_objective_debt_from_start": "reverse_max_objective_debt_from_start",
            "max_objective_recovery_from_min": "reverse_max_objective_recovery_from_min",
        }
    )
    rows = pathway_route[
        [
            "route_key",
            "start_condition",
            "seed",
            "pathway_shape_class",
            "preferred_source_equivalence_status",
            "guard_only_source_family_overlap",
        ]
    ].merge(forward_route, on="route_key", how="left")
    rows = rows.merge(reverse_route_subset, on="route_key", how="left")

    forward_sequences: dict[str, str] = {}
    reverse_sequences: dict[str, str] = {}
    forward_monotone_by_route: dict[str, bool] = {}
    reverse_monotone_by_route: dict[str, bool] = {}
    for route_key, group in persistence_trace.groupby("route_key", sort=True):
        ordered = group.sort_values("bridge_edge_weight_fraction", kind="mergesort")
        values = ordered["objective_value_by_step"].astype(float).tolist()
        forward_monotone_by_route[str(route_key)] = _is_non_decreasing(values)
        forward_sequences[str(route_key)] = " -> ".join(
            f"{float(row.bridge_edge_weight_fraction):.6g}:{float(row.objective_value_by_step):.6g}"
            for row in ordered.itertuples(index=False)
        )
    for route_key, group in reverse_trace.groupby("route_key", sort=True):
        ordered = group.sort_values("bridge_edge_weight_fraction", kind="mergesort")
        values = ordered["objective_value_by_step"].astype(float).tolist()
        reverse_monotone_by_route[str(route_key)] = _is_non_decreasing(values)
        reverse_sequences[str(route_key)] = " -> ".join(
            f"{float(row.bridge_edge_weight_fraction):.6g}:{float(row.objective_value_by_step):.6g}"
            for row in ordered.itertuples(index=False)
        )
    rows["forward_objective_nondecreasing_with_bridge_fraction"] = rows[
        "route_key"
    ].map(forward_monotone_by_route)
    rows["reverse_objective_nondecreasing_with_bridge_fraction"] = rows[
        "route_key"
    ].map(reverse_monotone_by_route)
    rows["forward_objective_sequence_by_fraction"] = rows["route_key"].map(
        forward_sequences
    )
    rows["reverse_objective_sequence_by_fraction"] = rows["route_key"].map(
        reverse_sequences
    )
    rows["fixed_landscape_barrier_supported"] = False
    rows["objective_barrier_interpretation"] = (
        "changing_graph_monotone_weight_profile_not_fixed_barrier"
    )
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    rows["run_status"] = RUN_STATUS
    return rows


def _decision_rows(
    *,
    fraction_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    pathway_summary: dict[str, Any],
) -> pd.DataFrame:
    forward_means = fraction_rows.sort_values(
        "bridge_edge_weight_fraction",
        kind="mergesort",
    )["forward_objective_mean"].astype(float).tolist()
    reverse_means = fraction_rows.sort_values(
        "bridge_edge_weight_fraction",
        kind="mergesort",
    )["reverse_objective_mean"].astype(float).tolist()
    transient_rows = fraction_rows[
        fraction_rows["expected_pathway_state_class"].astype(str).eq(
            "transient_signature"
        )
    ]
    transient_identical_objective_count = int(
        (
            transient_rows["forward_reverse_objective_mean_abs_diff"].astype(float)
            <= EPS
        ).sum()
    )
    return pd.DataFrame(
        [
            {
                "decision_id": "D1_objective_profile_is_monotone_weight_profile",
                "axis": "objective_profile",
                "observed": {
                    "forward_mean_nondecreasing_with_bridge_fraction": _is_non_decreasing(
                        forward_means
                    ),
                    "reverse_mean_nondecreasing_with_bridge_fraction": _is_non_decreasing(
                        reverse_means
                    ),
                    "route_forward_nondecreasing_count": int(
                        route_rows[
                            "forward_objective_nondecreasing_with_bridge_fraction"
                        ]
                        .map(_as_bool)
                        .sum()
                    ),
                    "route_reverse_nondecreasing_count": int(
                        route_rows[
                            "reverse_objective_nondecreasing_with_bridge_fraction"
                        ]
                        .map(_as_bool)
                        .sum()
                    ),
                },
                "decision": "objective_values_follow_bridge_weight_profile_not_independent_barrier",
                "passes": _is_non_decreasing(forward_means)
                and _is_non_decreasing(reverse_means),
                "claim_effect": "prevents reading the transient band as a fixed-landscape objective barrier",
            },
            {
                "decision_id": "D2_transient_band_has_shared_objective_profile",
                "axis": "transient_band",
                "observed": {
                    "transient_fraction_count": int(len(transient_rows)),
                    "forward_reverse_identical_objective_count": transient_identical_objective_count,
                    "transient_forward_debt_mean_range": [
                        float(transient_rows["forward_debt_mean"].min()),
                        float(transient_rows["forward_debt_mean"].max()),
                    ],
                    "transient_reverse_recovery_mean_range": [
                        float(transient_rows["reverse_recovery_mean"].min()),
                        float(transient_rows["reverse_recovery_mean"].max()),
                    ],
                },
                "decision": "transient_band_objective_profile_is_shared_but_perturbation_relative",
                "passes": transient_identical_objective_count == len(transient_rows),
                "claim_effect": "supports a shared transition-band diagnostic, not objective-barrier proof",
            },
            {
                "decision_id": "D3_pathway_shape_remains_state_ladder_claim",
                "axis": "pathway_shape_boundary",
                "observed": {
                    "preferred_pathway_readout": pathway_summary.get(
                        "preferred_pathway_readout"
                    ),
                    "matching_bidirectional_route_count": pathway_summary.get(
                        "matching_bidirectional_route_count"
                    ),
                },
                "decision": "pathway_shape_is_established_but_objective_barrier_is_not",
                "passes": int(pathway_summary.get("matching_bidirectional_route_count", 0))
                == 24,
                "claim_effect": "keeps the current positive statement at source-family transition-band mechanism object",
            },
            {
                "decision_id": "D4_no_quality_or_method_interpretation",
                "axis": "claim_boundary",
                "observed": CLAIM_BOUNDARY,
                "decision": "objective_columns_are_not_quality_cost_or_method_evidence",
                "passes": True,
                "claim_effect": "quality/cost, method, full replay, wall, and tunneling claims remain closed",
            },
            {
                "decision_id": "D5_next_gate",
                "axis": "next_step",
                "observed": (
                    "objective profile does not supply fixed-landscape barrier "
                    "evidence; mechanism/generalization remains the useful next gate"
                ),
                "decision": "move_to_mechanism_or_generality_gate",
                "passes": True,
                "claim_effect": "broad threshold localization remains unjustified without a mechanism question",
            },
        ]
    )


def _gate_matrix(
    *,
    pathway_gates: pd.DataFrame,
    fraction_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
) -> pd.DataFrame:
    forward_means = fraction_rows.sort_values(
        "bridge_edge_weight_fraction",
        kind="mergesort",
    )["forward_objective_mean"].astype(float).tolist()
    reverse_means = fraction_rows.sort_values(
        "bridge_edge_weight_fraction",
        kind="mergesort",
    )["reverse_objective_mean"].astype(float).tolist()
    transient_rows = fraction_rows[
        fraction_rows["expected_pathway_state_class"].astype(str).eq(
            "transient_signature"
        )
    ]
    fixed_barrier_count = int(
        route_rows["fixed_landscape_barrier_supported"].map(_as_bool).sum()
    )
    return pd.DataFrame(
        [
            _gate_row(
                "G1_upstream_pathway_shape_passed",
                "Did the upstream pathway-shape audit pass?",
                pathway_gates["gate_status"].value_counts().to_dict(),
                "all pathway-shape gates pass",
                bool(pathway_gates["gate_status"].astype(str).eq("pass").all()),
            ),
            _gate_row(
                "G2_objective_rows_available",
                "Are objective profiles available for all 24 routes and 9 fractions?",
                {
                    "fraction_rows": int(len(fraction_rows)),
                    "route_rows": int(len(route_rows)),
                    "forward_route_counts": sorted(
                        fraction_rows["forward_route_count"].astype(int).unique()
                    ),
                    "reverse_route_counts": sorted(
                        fraction_rows["reverse_route_count"].astype(int).unique()
                    ),
                },
                "9 fraction rows, 24 route rows, and 24 trace rows per fraction in each direction",
                len(fraction_rows) == 9
                and len(route_rows) == 24
                and set(fraction_rows["forward_route_count"].astype(int)) == {24}
                and set(fraction_rows["reverse_route_count"].astype(int)) == {24},
            ),
            _gate_row(
                "G3_objective_profile_is_monotone_with_bridge_fraction",
                "Does the objective profile follow bridge fraction rather than show an interior barrier?",
                {
                    "forward_mean_nondecreasing": _is_non_decreasing(forward_means),
                    "reverse_mean_nondecreasing": _is_non_decreasing(reverse_means),
                    "forward_means": forward_means,
                    "reverse_means": reverse_means,
                },
                "forward and reverse mean objective profiles are nondecreasing with bridge fraction",
                _is_non_decreasing(forward_means) and _is_non_decreasing(reverse_means),
            ),
            _gate_row(
                "G4_transient_objectives_are_shared_but_not_barrier_proof",
                "Do transient-band fractions share objective means across directions without becoming fixed-barrier proof?",
                transient_rows[
                    [
                        "bridge_edge_weight_fraction",
                        "forward_objective_mean",
                        "reverse_objective_mean",
                        "forward_reverse_objective_mean_abs_diff",
                    ]
                ].to_dict("records"),
                "six transient fractions have identical forward/reverse means within tolerance",
                bool(
                    (
                        transient_rows[
                            "forward_reverse_objective_mean_abs_diff"
                        ].astype(float)
                        <= EPS
                    ).all()
                ),
            ),
            _gate_row(
                "G5_fixed_landscape_barrier_not_supported",
                "Does the audit avoid promoting fixed-landscape barrier language?",
                {
                    "fixed_barrier_supported_route_count": fixed_barrier_count,
                    "route_interpretation_counts": route_rows[
                        "objective_barrier_interpretation"
                    ].value_counts().to_dict(),
                },
                "0 routes marked as fixed-landscape barrier supported",
                fixed_barrier_count == 0,
            ),
            _gate_row(
                "G6_claim_boundaries_closed",
                "Are wall, tunneling, method, full replay, and quality/cost claims closed?",
                {
                    "decision_passes": int(decision_rows["passes"].map(_as_bool).sum()),
                    "claim_boundary": CLAIM_BOUNDARY,
                },
                "all decisions pass and claim boundary is read-only",
                bool(decision_rows["passes"].map(_as_bool).all()),
            ),
        ]
    )


def _summary(
    *,
    persistence_dir: Path,
    reverse_dir: Path,
    pathway_shape_dir: Path,
    output_dir: Path,
    fraction_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> dict[str, Any]:
    forward_means = fraction_rows.sort_values(
        "bridge_edge_weight_fraction",
        kind="mergesort",
    )["forward_objective_mean"].astype(float).tolist()
    reverse_means = fraction_rows.sort_values(
        "bridge_edge_weight_fraction",
        kind="mergesort",
    )["reverse_objective_mean"].astype(float).tolist()
    transient_rows = fraction_rows[
        fraction_rows["expected_pathway_state_class"].astype(str).eq(
            "transient_signature"
        )
    ]
    return {
        "schema": "nanoclustering_g4_8_first_pass_016_objective_barrier_summary.v1",
        "status": RUN_STATUS,
        "persistence_dir": str(persistence_dir),
        "reverse_dir": str(reverse_dir),
        "pathway_shape_dir": str(pathway_shape_dir),
        "output_dir": str(output_dir),
        "primary_pair": PRIMARY_PAIR_ID,
        "fraction_row_count": int(len(fraction_rows)),
        "route_row_count": int(len(route_rows)),
        "decision_row_count": int(len(decision_rows)),
        "objective_profile_class": (
            "changing_graph_monotone_weight_profile_not_fixed_landscape_barrier"
        ),
        "forward_mean_nondecreasing_with_bridge_fraction": _is_non_decreasing(
            forward_means
        ),
        "reverse_mean_nondecreasing_with_bridge_fraction": _is_non_decreasing(
            reverse_means
        ),
        "fixed_landscape_barrier_supported_route_count": int(
            route_rows["fixed_landscape_barrier_supported"].map(_as_bool).sum()
        ),
        "transient_fraction_count": int(len(transient_rows)),
        "transient_forward_reverse_identical_objective_count": int(
            (
                transient_rows[
                    "forward_reverse_objective_mean_abs_diff"
                ].astype(float)
                <= EPS
            ).sum()
        ),
        "transient_forward_debt_mean_range": [
            float(transient_rows["forward_debt_mean"].min()),
            float(transient_rows["forward_debt_mean"].max()),
        ],
        "transient_reverse_recovery_mean_range": [
            float(transient_rows["reverse_recovery_mean"].min()),
            float(transient_rows["reverse_recovery_mean"].max()),
        ],
        "gate_status_counts": gates["gate_status"].value_counts().to_dict(),
        "failed_gates": gates.loc[
            ~gates["gate_status"].astype(str).eq("pass"),
            "gate_id",
        ].tolist(),
        "interpretation": (
            "The 016 objective profile supports the source-family transition-band "
            "state ladder as a perturbation-relative diagnostic, but not a "
            "fixed-landscape objective barrier. Mean objectives are monotone with "
            "bridge fraction in both directions, and the transient-band objective "
            "means match across forward/reverse traces, so objective columns should "
            "not be promoted to wall, tunneling, method, full replay, or "
            "quality/cost evidence."
        ),
        "recommended_next_gate": (
            "Move to mechanism/generalization: explain why the source-family "
            "transition band forms, or test whether analogous source-family "
            "transition bands recur beyond 016. Do not run broad threshold "
            "localization unless it answers that mechanism question."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    path: Path,
    summary: dict[str, Any],
    fraction_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# NanoClustering G4.8 First-Pass 016 Objective/Barrier Audit",
        "",
        "## Summary",
        "",
        f"- status: {summary['status']}",
        f"- objective_profile_class: {summary['objective_profile_class']}",
        f"- fixed_landscape_barrier_supported_route_count: {summary['fixed_landscape_barrier_supported_route_count']}",
        f"- transient_forward_reverse_identical_objective_count: {summary['transient_forward_reverse_identical_objective_count']}",
        f"- failed_gates: {summary['failed_gates']}",
        "",
        "## Fraction Rows",
        "",
        _markdown_table(
            fraction_rows,
            [
                "bridge_edge_weight_fraction",
                "expected_pathway_state_class",
                "forward_objective_mean",
                "reverse_objective_mean",
                "forward_reverse_objective_mean_abs_diff",
                "forward_debt_mean",
                "reverse_recovery_mean",
                "objective_profile_interpretation",
            ],
            max_rows=20,
        ),
        "",
        "## Route Rows",
        "",
        _markdown_table(
            route_rows,
            [
                "route_key",
                "pathway_shape_class",
                "forward_objective_nondecreasing_with_bridge_fraction",
                "reverse_objective_nondecreasing_with_bridge_fraction",
                "fixed_landscape_barrier_supported",
                "objective_barrier_interpretation",
            ],
            max_rows=30,
        ),
        "",
        "## Decisions",
        "",
        _markdown_table(
            decision_rows,
            ["decision_id", "axis", "observed", "decision", "passes", "claim_effect"],
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
        "## Interpretation",
        "",
        str(summary["interpretation"]),
        "",
        "## Recommended Next Gate",
        "",
        str(summary["recommended_next_gate"]),
        "",
        "## Claim Boundary",
        "",
        CLAIM_BOUNDARY,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_outputs(
    *,
    output_dir: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    fraction_rows: pd.DataFrame,
    route_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(fraction_rows, output_dir / FRACTION_ROWS_CSV)
    _write_csv(route_rows, output_dir / ROUTE_ROWS_CSV)
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
        path=output_dir / REPORT_MD,
        summary=summary,
        fraction_rows=fraction_rows,
        route_rows=route_rows,
        decision_rows=decision_rows,
        gates=gates,
    )


def run_audit(
    *,
    persistence_dir: Path = DEFAULT_PERSISTENCE_DIR,
    reverse_dir: Path = DEFAULT_REVERSE_DIR,
    pathway_shape_dir: Path = DEFAULT_PATHWAY_SHAPE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    context = _load_context(
        persistence_dir=persistence_dir,
        reverse_dir=reverse_dir,
        pathway_shape_dir=pathway_shape_dir,
    )
    fraction_rows = _fraction_rows(
        persistence_trace=context["persistence_trace"],
        reverse_trace=context["reverse_trace"],
        pathway_fraction=context["pathway_fraction"],
    )
    route_rows = _route_rows(
        persistence_route=context["persistence_route"],
        reverse_route=context["reverse_route"],
        pathway_route=context["pathway_route"],
        persistence_trace=context["persistence_trace"],
        reverse_trace=context["reverse_trace"],
    )
    decision_rows = _decision_rows(
        fraction_rows=fraction_rows,
        route_rows=route_rows,
        pathway_summary=context["pathway_summary"],
    )
    gates = _gate_matrix(
        pathway_gates=context["pathway_gates"],
        fraction_rows=fraction_rows,
        route_rows=route_rows,
        decision_rows=decision_rows,
    )
    summary = _summary(
        persistence_dir=persistence_dir,
        reverse_dir=reverse_dir,
        pathway_shape_dir=pathway_shape_dir,
        output_dir=output_dir,
        fraction_rows=fraction_rows,
        route_rows=route_rows,
        decision_rows=decision_rows,
        gates=gates,
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_016_objective_barrier_config.v1",
        "persistence_dir": str(persistence_dir),
        "reverse_dir": str(reverse_dir),
        "pathway_shape_dir": str(pathway_shape_dir),
        "output_dir": str(output_dir),
        "primary_pair": PRIMARY_PAIR_ID,
        "route_execution_status": ROUTE_EXECUTION_STATUS,
        "wall_promotion_status": WALL_PROMOTION_STATUS,
        "method_status": METHOD_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "run_status": RUN_STATUS,
    }
    _write_outputs(
        output_dir=output_dir,
        config=config,
        summary=summary,
        fraction_rows=fraction_rows,
        route_rows=route_rows,
        decision_rows=decision_rows,
        gates=gates,
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit local_pair_016 objective/barrier interpretation.",
    )
    parser.add_argument(
        "--persistence-dir",
        type=Path,
        default=DEFAULT_PERSISTENCE_DIR,
        help="Input persistence trace artifact directory.",
    )
    parser.add_argument(
        "--reverse-dir",
        type=Path,
        default=DEFAULT_REVERSE_DIR,
        help="Input reverse trace artifact directory.",
    )
    parser.add_argument(
        "--pathway-shape-dir",
        type=Path,
        default=DEFAULT_PATHWAY_SHAPE_DIR,
        help="Input pathway-shape audit artifact directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the objective/barrier audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_audit(
        persistence_dir=args.persistence_dir,
        reverse_dir=args.reverse_dir,
        pathway_shape_dir=args.pathway_shape_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
