#!/usr/bin/env python3
"""Refine the sparse G4.8D diagonal ready ridge with intermediate cells.

G4.8D found ready opportunity only at three cells on an increasing
pair-bridge/bridge-host diagonal: ``(1.32, 1.44)``, ``(1.35, 1.45)``, and
``(1.38, 1.46)``. This G4.8E diagnostic freezes the G4.3 bridge-release handle,
G4.5 selector, and G4.6 schedule, then probes a predeclared narrow strip around
the diagonal connecting those cells.

The goal is to decide whether the G4.8D signal is a continuous thin ridge, a
finite-width diagonal band, or isolated resonance points. This is mechanism
cartography only. It is not selector retuning, threshold search for a policy,
wall promotion, pathway identification, quality/cost evaluation,
NanoClustering replay, or an algorithm-level claim.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from analyze_leiden_cpm_variable_pair_synthetic_g4_4_restart_comparison import (
    CASE_SUMMARY_CSV as G4_4_CASE_SUMMARY_CSV,
    SUMMARY_JSON as G4_4_SUMMARY_JSON,
    analyze as analyze_g4_4,
)
from analyze_leiden_cpm_variable_pair_synthetic_g4_5_selector_suppression import (
    SELECTOR_CASE_SUMMARY_CSV as G4_5_CASE_SUMMARY_CSV,
    SUMMARY_JSON as G4_5_SUMMARY_JSON,
    analyze as analyze_g4_5,
)
from analyze_leiden_cpm_variable_pair_synthetic_g4_6_schedule_accounting import (
    SCHEDULE_CASE_SUMMARY_CSV as G4_6_CASE_SUMMARY_CSV,
    SUMMARY_JSON as G4_6_SUMMARY_JSON,
    analyze as analyze_g4_6,
)
from run_leiden_cpm_variable_pair_synthetic_demo import (
    BASE_RESULT_DIR,
    _json_safe,
    _write_csv,
)
from run_leiden_cpm_variable_pair_synthetic_g4_3_handle_generalization import (
    BASELINE_RUNS_CSV,
    CONFIG_JSON as G4_3_CONFIG_JSON,
    ENDPOINT_SUMMARY_CSV,
    GRAPH_EDGES_CSV,
    GRAPH_MANIFEST_CSV,
    HANDLE_POLICY_SUMMARY_CSV,
    HANDLE_RUNS_CSV,
    PANEL_CASES_CSV,
    REPORT_MD as G4_3_REPORT_MD,
    SUMMARY_JSON as G4_3_SUMMARY_JSON,
    VARIANT_GATE_ROWS_CSV,
    CLAIM_BOUNDARY as G4_3_CLAIM_BOUNDARY,
    HANDLE_POLICIES,
    PanelCase,
    _endpoint_summary,
    _graph_manifest_and_edges,
    _handle_policy_summary,
    _panel_case_to_synthetic,
    _run_baseline,
    _run_handles,
    _summary as _g4_3_summary,
    _variant_gate_rows,
    _write_report as _write_g4_3_report,
)
from run_leiden_cpm_variable_pair_synthetic_g4_8c_opportunity_cartography import (
    _cartography_regime,
    _cartography_role,
    _cartography_status,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_variable_pair_synthetic_g4_8e_diagonal_ridge_refinement_v1_20260603"
)

G4_3_DIRNAME = "g4_3_handle_probe"
G4_4_DIRNAME = "g4_4_restart_comparison"
G4_5_DIRNAME = "g4_5_selector_suppression"
G4_6_DIRNAME = "g4_6_schedule_accounting"
PANEL_DESIGN_CSV = "variable_pair_synthetic_g4_8e_panel_design.csv"
CASE_SUMMARY_CSV = "variable_pair_synthetic_g4_8e_case_summary.csv"
PAIR_SUMMARY_CSV = "variable_pair_synthetic_g4_8e_pair_bridge_summary.csv"
OFFSET_SUMMARY_CSV = "variable_pair_synthetic_g4_8e_offset_summary.csv"
ROLE_MATRIX_CSV = "variable_pair_synthetic_g4_8e_role_matrix.csv"
STATUS_MATRIX_CSV = "variable_pair_synthetic_g4_8e_status_matrix.csv"
SUMMARY_JSON = "variable_pair_synthetic_g4_8e_summary.json"
CONFIG_JSON = "variable_pair_synthetic_g4_8e_config.json"
REPORT_MD = "variable_pair_synthetic_g4_8e_report.md"

DIRECT_WEIGHT = 1.08
HOST_CLIQUE_WEIGHT = 1.25
PAIR_BRIDGE_VALUES = tuple(round(1.320 + step * 0.005, 3) for step in range(13))
BRIDGE_HOST_OFFSETS = (-0.004, -0.002, 0.0, 0.002, 0.004)
DIAGONAL_START_PAIR = 1.32
DIAGONAL_START_BRIDGE = 1.44
DIAGONAL_SLOPE = 1.0 / 3.0
EPS = 1.0e-12

CLAIM_BOUNDARY = (
    "Variable-pair synthetic G4.8E diagonal-ridge refinement only; a "
    "predeclared narrow strip around the G4.8D ready diagonal replays the "
    "frozen G4.3 handle, G4.5 selector, and G4.6 schedule. The diagnostic "
    "tests whether the ready signal is a continuous thin ridge, finite-width "
    "band, or isolated resonance points. It does not retune selectors, search "
    "thresholds for a policy, promote walls, identify pathways, replay "
    "NanoClustering, evaluate quality/cost value, or make algorithm-level "
    "claims."
)
ROUTE_EXECUTION_STATUS = "executed_g4_8e_diagonal_ridge_refinement"
WALL_PROMOTION_STATUS = "not_promoted_diagonal_refinement_only"
METHOD_STATUS = "diagonal_ridge_cartography_not_algorithm_claim"


@dataclass(frozen=True)
class RidgeCase:
    case_id: str
    pair_bridge_weight: float
    diagonal_bridge_host_weight: float
    bridge_host_offset: float
    bridge_host_weight: float
    pair_bridge_index: int
    offset_index: int

    def to_panel_case(self) -> PanelCase:
        return PanelCase(
            case_id=self.case_id,
            panel_role="positive_holdout",
            expected_gate="diagonal_ridge_refinement_no_expected_success",
            direct_weight=DIRECT_WEIGHT,
            pair_bridge_weight=self.pair_bridge_weight,
            bridge_host_weight=self.bridge_host_weight,
            host_clique_weight=HOST_CLIQUE_WEIGHT,
            note=(
                "Predeclared narrow-strip refinement cell around the G4.8D "
                "pair-bridge/bridge-host ready diagonal."
            ),
        )


def _diagonal_bridge_host(pair_bridge: float) -> float:
    return round(
        DIAGONAL_START_BRIDGE
        + (float(pair_bridge) - DIAGONAL_START_PAIR) * DIAGONAL_SLOPE,
        3,
    )


def _case_id(pair_bridge: float, bridge_host: float, offset: float) -> str:
    pair_milli = int(round(pair_bridge * 1000))
    bridge_milli = int(round(bridge_host * 1000))
    offset_milli = int(round(offset * 1000))
    sign = "p" if offset_milli >= 0 else "m"
    return (
        f"g4_8e_pair{pair_milli:04d}_bridge{bridge_milli:04d}"
        f"_o{sign}{abs(offset_milli):03d}"
    )


def _ridge_cases() -> tuple[RidgeCase, ...]:
    cases: list[RidgeCase] = []
    for pair_index, pair_bridge in enumerate(PAIR_BRIDGE_VALUES):
        diagonal_bridge = _diagonal_bridge_host(pair_bridge)
        for offset_index, offset in enumerate(BRIDGE_HOST_OFFSETS):
            bridge_host = round(diagonal_bridge + offset, 3)
            cases.append(
                RidgeCase(
                    case_id=_case_id(pair_bridge, bridge_host, offset),
                    pair_bridge_weight=float(pair_bridge),
                    diagonal_bridge_host_weight=float(diagonal_bridge),
                    bridge_host_offset=float(offset),
                    bridge_host_weight=float(bridge_host),
                    pair_bridge_index=pair_index,
                    offset_index=offset_index,
                )
            )
    return tuple(cases)


RIDGE_CASES = _ridge_cases()


def _claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _panel_design_rows() -> pd.DataFrame:
    rows = []
    for case in RIDGE_CASES:
        rows.append(
            {
                "case_id": case.case_id,
                "pair_bridge_weight": case.pair_bridge_weight,
                "diagonal_bridge_host_weight": case.diagonal_bridge_host_weight,
                "bridge_host_offset": case.bridge_host_offset,
                "bridge_host_weight": case.bridge_host_weight,
                "pair_bridge_index": case.pair_bridge_index,
                "offset_index": case.offset_index,
                "direct_weight": DIRECT_WEIGHT,
                "host_clique_weight": HOST_CLIQUE_WEIGHT,
                "panel_role": "positive_holdout",
                "expected_gate": "diagonal_ridge_refinement_no_expected_success",
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _run_g4_3_stage(
    *,
    output_dir: Path,
    baseline_seeds: int,
    handle_seeds: int,
    n_iterations: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_cases = tuple(case.to_panel_case() for case in RIDGE_CASES)
    cases = [_panel_case_to_synthetic(case) for case in panel_cases]
    panel_rows, graph_manifest, graph_edges = _graph_manifest_and_edges(
        cases,
        panel_cases,
    )
    baseline_runs = _run_baseline(
        cases=cases,
        panel_cases=panel_cases,
        seeds=baseline_seeds,
        n_iterations=n_iterations,
    )
    endpoint_summary = _endpoint_summary(baseline_runs)
    handle_runs = _run_handles(
        cases=cases,
        panel_cases=panel_cases,
        endpoint_summary=endpoint_summary,
        seeds=handle_seeds,
        n_iterations=n_iterations,
    )
    policy_summary = _handle_policy_summary(handle_runs)
    gate_rows = _variant_gate_rows(
        endpoint_summary=endpoint_summary,
        policy_summary=policy_summary,
        panel_cases=panel_cases,
    )
    _write_csv(panel_rows, output_dir / PANEL_CASES_CSV)
    _write_csv(graph_manifest, output_dir / GRAPH_MANIFEST_CSV)
    _write_csv(graph_edges, output_dir / GRAPH_EDGES_CSV)
    _write_csv(baseline_runs, output_dir / BASELINE_RUNS_CSV)
    _write_csv(endpoint_summary, output_dir / ENDPOINT_SUMMARY_CSV)
    _write_csv(handle_runs, output_dir / HANDLE_RUNS_CSV)
    _write_csv(policy_summary, output_dir / HANDLE_POLICY_SUMMARY_CSV)
    _write_csv(gate_rows, output_dir / VARIANT_GATE_ROWS_CSV)
    summary = _g4_3_summary(
        output_dir=output_dir,
        baseline_runs=baseline_runs,
        endpoint_summary=endpoint_summary,
        handle_runs=handle_runs,
        policy_summary=policy_summary,
        gate_rows=gate_rows,
    )
    (output_dir / G4_3_SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "variable_pair_synthetic_g4_8e_g4_3_config.v1",
        "output_dir": str(output_dir),
        "panel_cases": [case.__dict__ for case in RIDGE_CASES],
        "handle_policies": list(HANDLE_POLICIES),
        "baseline_seeds": int(baseline_seeds),
        "handle_seeds": int(handle_seeds),
        "n_iterations": int(n_iterations),
        "stage_claim_boundary": G4_3_CLAIM_BOUNDARY,
        "g4_8e_claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / G4_3_CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_g4_3_report(
        output_dir=output_dir,
        summary=summary,
        gate_rows=gate_rows,
        policy_summary=policy_summary,
    )
    return summary


def _case_summary(
    *,
    panel_design: pd.DataFrame,
    g4_3_dir: Path,
    g4_4_dir: Path,
    g4_5_dir: Path,
    g4_6_dir: Path,
) -> pd.DataFrame:
    g4_3 = pd.read_csv(g4_3_dir / VARIANT_GATE_ROWS_CSV)
    g4_4 = pd.read_csv(g4_4_dir / G4_4_CASE_SUMMARY_CSV)
    g4_5 = pd.read_csv(g4_5_dir / G4_5_CASE_SUMMARY_CSV)
    g4_6 = pd.read_csv(g4_6_dir / G4_6_CASE_SUMMARY_CSV)
    design_cols = [
        "case_id",
        "pair_bridge_weight",
        "diagonal_bridge_host_weight",
        "bridge_host_offset",
        "bridge_host_weight",
        "pair_bridge_index",
        "offset_index",
        "direct_weight",
        "host_clique_weight",
    ]
    rows = (
        g4_6.merge(panel_design[design_cols], on="case_id", how="left")
        .merge(
            g4_3[
                [
                    "case_id",
                    "baseline_pair_coassigned_run_share",
                    "separated_endpoint_count",
                    "coassigned_endpoint_count",
                    "bridge_handle_eligible_source_count",
                    "bridge_handle_robust_pair_coassignment_count",
                    "pair_relation_only_robust_pair_coassignment_count",
                    "bridge_handle_pair_rate_median",
                    "gate_passed",
                    "g4_3_gate_status",
                ]
            ],
            on="case_id",
            how="left",
        )
        .merge(
            g4_4[
                [
                    "case_id",
                    "g4_4_case_status",
                    "source_count",
                    "handle_known_coassigned_hit_rate_median",
                    "baseline_over_handle_expected_run_ratio_median",
                ]
            ],
            on="case_id",
            how="left",
        )
        .merge(
            g4_5[
                [
                    "case_id",
                    "g4_5_case_status",
                    "selector_selected_count",
                    "selector_suppressed_count",
                    "selected_positive_win_count",
                    "suppressed_positive_win_count",
                    "selected_control_leak_count",
                ]
            ],
            on="case_id",
            how="left",
        )
    )
    rows["endpoint_coexistence"] = (
        rows["separated_endpoint_count"].fillna(0).astype(int).gt(0)
        & rows["coassigned_endpoint_count"].fillna(0).astype(int).gt(0)
    )
    rows["target_saturated"] = (
        rows["baseline_pair_coassigned_run_share"].fillna(0.0).astype(float).ge(1.0)
    )
    rows["target_absent"] = (
        rows["baseline_pair_coassigned_run_share"].fillna(0.0).astype(float).le(0.0)
    )
    rows["bridge_release_eligible"] = (
        rows["bridge_handle_eligible_source_count"].fillna(0).astype(int).gt(0)
    )
    rows["bridge_release_robust"] = (
        rows["bridge_handle_robust_pair_coassignment_count"]
        .fillna(0)
        .astype(int)
        .gt(0)
    )
    rows["pair_only_robust"] = (
        rows["pair_relation_only_robust_pair_coassignment_count"]
        .fillna(0)
        .astype(int)
        .gt(0)
    )
    rows["selector_source_available"] = (
        rows["selected_source_count"].fillna(0).astype(int).gt(0)
    )
    rows["observed_opportunity_regime"] = [
        _cartography_regime(row) for row in rows.to_dict("records")
    ]
    rows["observed_next_gate_role"] = [
        _cartography_role(regime) for regime in rows["observed_opportunity_regime"]
    ]
    rows["source_handle_fire"] = (
        rows["handle_application_rate"].fillna(0.0).astype(float).gt(0.0)
    )
    rows["ready_opportunity_preserved"] = (
        rows["observed_next_gate_role"].eq("ready_positive_anchor")
        & rows["source_handle_fire"]
        & rows["schedule_known_coassigned_hit_rate"].fillna(0.0).astype(float).gt(
            rows["baseline_known_coassigned_hit_rate"].fillna(0.0).astype(float)
            + EPS
        )
    )
    rows["cartography_status"] = [
        _cartography_status(row) for row in rows.to_dict("records")
    ]
    rows["role_symbol"] = [_role_symbol(row) for row in rows.to_dict("records")]
    return _claim_columns(rows)


def _role_symbol(row: dict[str, Any]) -> str:
    status = str(row["cartography_status"])
    if status == "ready_opportunity_with_source_handle_fire":
        return "R"
    if status == "collapsed_to_target_saturation":
        return "T"
    if status == "coexistence_bridge_eligible_nonrobust":
        return "N"
    if status == "pair_only_boundary":
        return "P"
    if status == "no_target_boundary":
        return "Z"
    return "?"


def _group_summary(case_summary: pd.DataFrame, group_col: str) -> pd.DataFrame:
    order_col = "bridge_host_offset" if group_col == "pair_bridge_weight" else "pair_bridge_weight"
    rows: list[dict[str, Any]] = []
    for key, group in case_summary.groupby(group_col, sort=True):
        rows.append(
            {
                group_col: float(key),
                "case_count": int(len(group)),
                "ready_count": int(group["role_symbol"].eq("R").sum()),
                "target_saturation_count": int(group["role_symbol"].eq("T").sum()),
                "nonrobust_coexistence_count": int(group["role_symbol"].eq("N").sum()),
                "pair_only_count": int(group["role_symbol"].eq("P").sum()),
                "source_handle_fire_count": int(
                    group["source_handle_fire"].astype(bool).sum()
                ),
                "role_symbols": "".join(
                    group.sort_values(order_col)["role_symbol"].astype(str)
                ),
                "case_ids": ";".join(sorted(group["case_id"].astype(str))),
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _matrix_rows(case_summary: pd.DataFrame, value_col: str) -> pd.DataFrame:
    matrix = case_summary.pivot(
        index="pair_bridge_weight",
        columns="bridge_host_offset",
        values=value_col,
    ).sort_index(ascending=True)
    matrix = matrix.reset_index()
    matrix.columns = [_column_label(col) for col in matrix.columns]
    return _claim_columns(matrix)


def _column_label(col: Any) -> str:
    if isinstance(col, float):
        return f"{col:+.3f}"
    return str(col)


def _ready_components(case_summary: pd.DataFrame) -> list[dict[str, Any]]:
    ready = case_summary[case_summary["role_symbol"].eq("R")].copy()
    ready_coords = {
        (int(row["pair_bridge_index"]), int(row["offset_index"])): row
        for row in ready.to_dict("records")
    }
    seen: set[tuple[int, int]] = set()
    components: list[dict[str, Any]] = []
    for coord in sorted(ready_coords):
        if coord in seen:
            continue
        stack = [coord]
        seen.add(coord)
        members: list[dict[str, Any]] = []
        while stack:
            cur = stack.pop()
            member = ready_coords[cur]
            members.append(member)
            pair_index, offset_index = cur
            for nxt in (
                (pair_index - 1, offset_index),
                (pair_index + 1, offset_index),
                (pair_index, offset_index - 1),
                (pair_index, offset_index + 1),
            ):
                if nxt in ready_coords and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(
            {
                "component_size": int(len(members)),
                "pair_bridge_min": float(
                    min(row["pair_bridge_weight"] for row in members)
                ),
                "pair_bridge_max": float(
                    max(row["pair_bridge_weight"] for row in members)
                ),
                "bridge_host_min": float(
                    min(row["bridge_host_weight"] for row in members)
                ),
                "bridge_host_max": float(
                    max(row["bridge_host_weight"] for row in members)
                ),
                "offset_min": float(min(row["bridge_host_offset"] for row in members)),
                "offset_max": float(max(row["bridge_host_offset"] for row in members)),
                "case_ids": ";".join(sorted(str(row["case_id"]) for row in members)),
            }
        )
    return sorted(components, key=lambda row: row["component_size"], reverse=True)


def _contiguous_ready_runs(case_summary: pd.DataFrame) -> list[dict[str, Any]]:
    diagonal = (
        case_summary[
            case_summary["bridge_host_offset"].astype(float).abs().le(EPS)
            & case_summary["role_symbol"].eq("R")
        ]
        .sort_values("pair_bridge_index")
        .copy()
    )
    if diagonal.empty:
        return []
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_index: int | None = None
    for row in diagonal.to_dict("records"):
        pair_index = int(row["pair_bridge_index"])
        if previous_index is None or pair_index == previous_index + 1:
            current.append(row)
        else:
            runs.append(current)
            current = [row]
        previous_index = pair_index
    if current:
        runs.append(current)
    return [
        {
            "run_size": int(len(run)),
            "pair_bridge_min": float(min(row["pair_bridge_weight"] for row in run)),
            "pair_bridge_max": float(max(row["pair_bridge_weight"] for row in run)),
            "case_ids": ";".join(sorted(str(row["case_id"]) for row in run)),
        }
        for run in runs
    ]


def _ridge_read(case_summary: pd.DataFrame) -> dict[str, Any]:
    ready = case_summary[case_summary["role_symbol"].eq("R")].copy()
    ready_count = int(len(ready))
    components = _ready_components(case_summary)
    largest_component = int(components[0]["component_size"]) if components else 0
    diagonal = case_summary[
        case_summary["bridge_host_offset"].astype(float).abs().le(EPS)
    ]
    diagonal_ready_count = int(diagonal["role_symbol"].eq("R").sum())
    diagonal_run_rows = _contiguous_ready_runs(case_summary)
    largest_diagonal_run = (
        max(row["run_size"] for row in diagonal_run_rows) if diagonal_run_rows else 0
    )
    ready_offsets = (
        sorted(ready["bridge_host_offset"].astype(float).unique().tolist())
        if ready_count
        else []
    )
    max_ready_offset_abs = (
        float(max(abs(value) for value in ready_offsets)) if ready_offsets else 0.0
    )
    centerline_fraction = (
        diagonal_ready_count / float(len(PAIR_BRIDGE_VALUES))
        if PAIR_BRIDGE_VALUES
        else 0.0
    )
    if ready_count == 0:
        ridge_status = "no_ready_opportunity"
    elif (
        largest_diagonal_run >= max(3, int(round(len(PAIR_BRIDGE_VALUES) * 0.6)))
        and max_ready_offset_abs <= EPS
    ):
        ridge_status = "continuous_thin_diagonal_ridge"
    elif largest_component >= max(5, int(round(len(PAIR_BRIDGE_VALUES) * 0.6))):
        ridge_status = "finite_width_diagonal_band"
    elif ready_count >= 3 and max_ready_offset_abs <= EPS:
        ridge_status = "centerline_resonance_lattice"
    elif ready_count <= 3 and largest_component <= 1:
        ridge_status = "isolated_resonance_points"
    else:
        ridge_status = "fragmented_diagonal_readiness"
    return {
        "ready_count": ready_count,
        "ready_component_count": int(len(components)),
        "largest_ready_component_size": largest_component,
        "diagonal_ready_count": diagonal_ready_count,
        "diagonal_ready_fraction": float(centerline_fraction),
        "largest_diagonal_ready_run": int(largest_diagonal_run),
        "ready_offsets": ready_offsets,
        "max_ready_offset_abs": max_ready_offset_abs,
        "ready_pair_bridge_values": sorted(
            ready["pair_bridge_weight"].astype(float).unique().tolist()
        ),
        "ready_bridge_host_values": sorted(
            ready["bridge_host_weight"].astype(float).unique().tolist()
        ),
        "ridge_status": ridge_status,
        "ready_components": components,
        "diagonal_ready_runs": diagonal_run_rows,
    }


def _summary(
    *,
    output_dir: Path,
    g4_3_summary: dict[str, Any],
    g4_5_summary: dict[str, Any],
    g4_6_summary: dict[str, Any],
    case_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    offset_summary: pd.DataFrame,
) -> dict[str, Any]:
    ridge = _ridge_read(case_summary)
    return {
        "schema": "variable_pair_synthetic_g4_8e_diagonal_ridge_refinement_summary.v1",
        "status": ROUTE_EXECUTION_STATUS,
        "output_dir": str(output_dir),
        "case_count": int(len(case_summary)),
        "pair_bridge_values": list(PAIR_BRIDGE_VALUES),
        "bridge_host_offsets": list(BRIDGE_HOST_OFFSETS),
        "diagonal_rule": (
            "bridge_host = 1.44 + (pair_bridge - 1.32) / 3, rounded to 0.001"
        ),
        "ready_count": int(ridge["ready_count"]),
        "target_saturation_count": int(case_summary["role_symbol"].eq("T").sum()),
        "nonrobust_coexistence_count": int(case_summary["role_symbol"].eq("N").sum()),
        "pair_only_count": int(case_summary["role_symbol"].eq("P").sum()),
        "source_handle_fire_count": int(
            case_summary["source_handle_fire"].astype(bool).sum()
        ),
        "cartography_status_counts": case_summary[
            "cartography_status"
        ].value_counts().to_dict(),
        "observed_role_counts": case_summary[
            "observed_next_gate_role"
        ].value_counts().to_dict(),
        "ready_component_count": int(ridge["ready_component_count"]),
        "largest_ready_component_size": int(ridge["largest_ready_component_size"]),
        "diagonal_ready_count": int(ridge["diagonal_ready_count"]),
        "diagonal_ready_fraction": float(ridge["diagonal_ready_fraction"]),
        "largest_diagonal_ready_run": int(ridge["largest_diagonal_ready_run"]),
        "ready_offsets": ridge["ready_offsets"],
        "max_ready_offset_abs": float(ridge["max_ready_offset_abs"]),
        "ready_pair_bridge_values": ridge["ready_pair_bridge_values"],
        "ready_bridge_host_values": ridge["ready_bridge_host_values"],
        "ridge_status": str(ridge["ridge_status"]),
        "ready_components": ridge["ready_components"],
        "diagonal_ready_runs": ridge["diagonal_ready_runs"],
        "pair_summary_row_count": int(len(pair_summary)),
        "offset_summary_row_count": int(len(offset_summary)),
        "g4_3_positive_pass_count": int(g4_3_summary.get("positive_pass_count", 0)),
        "g4_5_selector_gate_passed": bool(g4_5_summary.get("selector_gate_passed", False)),
        "g4_6_schedule_gate_passed": bool(g4_6_summary.get("schedule_gate_passed", False)),
        "recommended_next_gate": _recommended_next_gate(str(ridge["ridge_status"])),
        "stage_summary_paths": {
            "g4_3": str(output_dir / G4_3_DIRNAME / G4_3_SUMMARY_JSON),
            "g4_4": str(output_dir / G4_4_DIRNAME / G4_4_SUMMARY_JSON),
            "g4_5": str(output_dir / G4_5_DIRNAME / G4_5_SUMMARY_JSON),
            "g4_6": str(output_dir / G4_6_DIRNAME / G4_6_SUMMARY_JSON),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _recommended_next_gate(ridge_status: str) -> str:
    if ridge_status == "continuous_thin_diagonal_ridge":
        return (
            "Freeze the diagonal construction relation and test it on fresh "
            "direct/host contexts before source-discovery replacement."
        )
    if ridge_status == "finite_width_diagonal_band":
        return (
            "Derive a construction-band rule from ready cells, then test fresh "
            "contexts and seed robustness before source-discovery replacement."
        )
    if ridge_status == "centerline_resonance_lattice":
        return (
            "Audit why only separated centerline cells fire by comparing ready "
            "and neighboring non-ready endpoint/source signatures before "
            "promoting any construction rule."
        )
    if ridge_status == "isolated_resonance_points":
        return (
            "Treat the G4.8D ready cells as isolated resonances under this "
            "synthetic family; redesign the construction family before source "
            "discovery or pathway work."
        )
    if ridge_status == "no_ready_opportunity":
        return (
            "Stop this narrow-strip construction path and redesign graph "
            "construction before source-discovery replacement."
        )
    return (
        "Inspect fragmented ready cells, then either define a mechanistic "
        "construction rule or retire this synthetic family."
    )


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    case_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    offset_summary: pd.DataFrame,
    role_matrix: pd.DataFrame,
) -> None:
    lines = [
        "# Variable-Pair Synthetic G4.8E Diagonal Ridge Refinement",
        "",
        f"- status: `{summary['status']}`",
        f"- ridge_status: {summary['ridge_status']}",
        f"- ready_count: {summary['ready_count']}",
        f"- largest_ready_component_size: {summary['largest_ready_component_size']}",
        f"- diagonal_ready_count: {summary['diagonal_ready_count']}",
        f"- diagonal_ready_fraction: {summary['diagonal_ready_fraction']:.3f}",
        f"- largest_diagonal_ready_run: {summary['largest_diagonal_ready_run']}",
        f"- ready_offsets: {summary['ready_offsets']}",
        f"- max_ready_offset_abs: {summary['max_ready_offset_abs']:.3f}",
        f"- ready_pair_bridge_values: {summary['ready_pair_bridge_values']}",
        f"- ready_bridge_host_values: {summary['ready_bridge_host_values']}",
        f"- target_saturation_count: {summary['target_saturation_count']}",
        f"- nonrobust_coexistence_count: {summary['nonrobust_coexistence_count']}",
        f"- source_handle_fire_count: {summary['source_handle_fire_count']}",
        f"- cartography_status_counts: {summary['cartography_status_counts']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Role Matrix",
        "",
        "Rows are pair-bridge values. Columns are bridge-host offsets from the "
        "predeclared diagonal. Symbols: R=ready with source-handle fire, "
        "T=target saturation, N=nonrobust coexistence.",
        "",
    ]
    display_cols = ["pair_bridge_weight"] + [
        f"{offset:+.3f}" for offset in BRIDGE_HOST_OFFSETS
    ]
    lines.extend(_markdown_table(role_matrix[display_cols]))
    lines.extend(["", "## Pair-Bridge Summary"])
    for row in pair_summary.itertuples(index=False):
        lines.append(
            "- "
            f"pair_bridge={row.pair_bridge_weight:.3f}: roles={row.role_symbols}, "
            f"ready={row.ready_count}, target={row.target_saturation_count}, "
            f"nonrobust={row.nonrobust_coexistence_count}, "
            f"source_fire={row.source_handle_fire_count}"
        )
    lines.extend(["", "## Offset Summary"])
    for row in offset_summary.itertuples(index=False):
        lines.append(
            "- "
            f"offset={row.bridge_host_offset:+.3f}: roles={row.role_symbols}, "
            f"ready={row.ready_count}, target={row.target_saturation_count}, "
            f"nonrobust={row.nonrobust_coexistence_count}, "
            f"source_fire={row.source_handle_fire_count}"
        )
    lines.extend(["", "## Ready Components"])
    for component in summary["ready_components"]:
        lines.append(
            "- "
            f"size={component['component_size']}, "
            f"pair={component['pair_bridge_min']:.3f}-{component['pair_bridge_max']:.3f}, "
            f"bridge={component['bridge_host_min']:.3f}-{component['bridge_host_max']:.3f}, "
            f"offset={component['offset_min']:+.3f}-{component['offset_max']:+.3f}, "
            f"ids={component['case_ids']}"
        )
    lines.extend(["", "## Diagonal Ready Runs"])
    for run in summary["diagonal_ready_runs"]:
        lines.append(
            "- "
            f"size={run['run_size']}, "
            f"pair={run['pair_bridge_min']:.3f}-{run['pair_bridge_max']:.3f}, "
            f"ids={run['case_ids']}"
        )
    lines.extend(["", "## Case Summary"])
    for row in case_summary.itertuples(index=False):
        lines.append(
            "- "
            f"{row.case_id}: {row.role_symbol}/{row.cartography_status}; "
            f"pair_bridge={row.pair_bridge_weight:.3f}, "
            f"diag_bridge={row.diagonal_bridge_host_weight:.3f}, "
            f"offset={row.bridge_host_offset:+.3f}, "
            f"bridge_host={row.bridge_host_weight:.3f}, "
            f"pair_share={row.baseline_pair_coassigned_run_share:.3f}, "
            f"eligible={row.bridge_handle_eligible_source_count}, "
            f"robust_bridge={row.bridge_handle_robust_pair_coassignment_count}, "
            f"selected_sources={row.selected_source_count}, "
            f"schedule_p={row.schedule_known_coassigned_hit_rate:.3f}"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "G4.8E is a narrow-strip mechanism map. It should decide "
                "whether the G4.8D diagonal can become a construction rule "
                "before any source-discovery, pathway, wall, quality/cost, or "
                "method claim is reopened."
            ),
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    cols = [str(col) for col in frame.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in frame.to_dict("records"):
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return lines


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    g4_3_dir = output_dir / G4_3_DIRNAME
    g4_4_dir = output_dir / G4_4_DIRNAME
    g4_5_dir = output_dir / G4_5_DIRNAME
    g4_6_dir = output_dir / G4_6_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_design = _panel_design_rows()
    _write_csv(panel_design, output_dir / PANEL_DESIGN_CSV)
    g4_3_summary = _run_g4_3_stage(
        output_dir=g4_3_dir,
        baseline_seeds=int(args.baseline_seeds),
        handle_seeds=int(args.handle_seeds),
        n_iterations=int(args.n_iterations),
    )
    g4_4_summary = analyze_g4_4(
        Namespace(g4_3_dir=g4_3_dir, output_dir=g4_4_dir)
    )
    g4_5_summary = analyze_g4_5(
        Namespace(g4_3_dir=g4_3_dir, g4_4_dir=g4_4_dir, output_dir=g4_5_dir)
    )
    g4_6_summary = analyze_g4_6(
        Namespace(
            g4_3_dir=g4_3_dir,
            g4_4_dir=g4_4_dir,
            g4_5_dir=g4_5_dir,
            output_dir=g4_6_dir,
        )
    )
    case_summary = _case_summary(
        panel_design=panel_design,
        g4_3_dir=g4_3_dir,
        g4_4_dir=g4_4_dir,
        g4_5_dir=g4_5_dir,
        g4_6_dir=g4_6_dir,
    )
    pair_summary = _group_summary(case_summary, "pair_bridge_weight")
    offset_summary = _group_summary(case_summary, "bridge_host_offset")
    role_matrix = _matrix_rows(case_summary, "role_symbol")
    status_matrix = _matrix_rows(case_summary, "cartography_status")
    _write_csv(case_summary, output_dir / CASE_SUMMARY_CSV)
    _write_csv(pair_summary, output_dir / PAIR_SUMMARY_CSV)
    _write_csv(offset_summary, output_dir / OFFSET_SUMMARY_CSV)
    _write_csv(role_matrix, output_dir / ROLE_MATRIX_CSV)
    _write_csv(status_matrix, output_dir / STATUS_MATRIX_CSV)
    summary = _summary(
        output_dir=output_dir,
        g4_3_summary=g4_3_summary,
        g4_5_summary=g4_5_summary,
        g4_6_summary=g4_6_summary,
        case_summary=case_summary,
        pair_summary=pair_summary,
        offset_summary=offset_summary,
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "variable_pair_synthetic_g4_8e_diagonal_ridge_refinement_config.v1",
        "output_dir": str(output_dir),
        "stage_dirs": {
            "g4_3": str(g4_3_dir),
            "g4_4": str(g4_4_dir),
            "g4_5": str(g4_5_dir),
            "g4_6": str(g4_6_dir),
        },
        "direct_weight": DIRECT_WEIGHT,
        "host_clique_weight": HOST_CLIQUE_WEIGHT,
        "pair_bridge_values": list(PAIR_BRIDGE_VALUES),
        "bridge_host_offsets": list(BRIDGE_HOST_OFFSETS),
        "diagonal_start_pair": DIAGONAL_START_PAIR,
        "diagonal_start_bridge": DIAGONAL_START_BRIDGE,
        "diagonal_slope": DIAGONAL_SLOPE,
        "baseline_seeds": int(args.baseline_seeds),
        "handle_seeds": int(args.handle_seeds),
        "n_iterations": int(args.n_iterations),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        case_summary=case_summary,
        pair_summary=pair_summary,
        offset_summary=offset_summary,
        role_matrix=role_matrix,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-seeds", type=int, default=16)
    parser.add_argument("--handle-seeds", type=int, default=16)
    parser.add_argument("--n-iterations", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
