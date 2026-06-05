#!/usr/bin/env python3
"""Map the pair-bridge by bridge-host ready-opportunity balance surface.

G4.8C isolated the active construction boundary to the balance between
``pair_bridge`` and ``bridge_host`` support. This G4.8D diagnostic freezes the
G4.3 bridge-release handle, G4.5 selector, and G4.6 schedule, then runs a
predeclared 2D grid around the ``1.35/1.45`` ready anchor.

The goal is to decide whether ready opportunity is a reproducible local band or
a knife-edge artifact. This is mechanism cartography only. It is not selector
retuning, threshold search for a policy, wall promotion, pathway identification,
quality/cost evaluation, NanoClustering replay, or an algorithm-level claim.
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
    CLAIM_BOUNDARY as G4_3_CLAIM_BOUNDARY,
    HANDLE_POLICIES,
)
from run_leiden_cpm_variable_pair_synthetic_g4_8c_opportunity_cartography import (
    _cartography_regime,
    _cartography_role,
    _cartography_status,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_variable_pair_synthetic_g4_8d_balance_cartography_v1_20260603"
)

G4_3_DIRNAME = "g4_3_handle_probe"
G4_4_DIRNAME = "g4_4_restart_comparison"
G4_5_DIRNAME = "g4_5_selector_suppression"
G4_6_DIRNAME = "g4_6_schedule_accounting"
PANEL_DESIGN_CSV = "variable_pair_synthetic_g4_8d_panel_design.csv"
CASE_SUMMARY_CSV = "variable_pair_synthetic_g4_8d_case_summary.csv"
PAIR_SUMMARY_CSV = "variable_pair_synthetic_g4_8d_pair_bridge_summary.csv"
BRIDGE_SUMMARY_CSV = "variable_pair_synthetic_g4_8d_bridge_host_summary.csv"
ROLE_MATRIX_CSV = "variable_pair_synthetic_g4_8d_role_matrix.csv"
STATUS_MATRIX_CSV = "variable_pair_synthetic_g4_8d_status_matrix.csv"
SUMMARY_JSON = "variable_pair_synthetic_g4_8d_summary.json"
CONFIG_JSON = "variable_pair_synthetic_g4_8d_config.json"
REPORT_MD = "variable_pair_synthetic_g4_8d_report.md"

DIRECT_WEIGHT = 1.08
HOST_CLIQUE_WEIGHT = 1.25
PAIR_BRIDGE_VALUES = (1.32, 1.33, 1.34, 1.35, 1.36, 1.37, 1.38, 1.40)
BRIDGE_HOST_VALUES = (1.42, 1.43, 1.44, 1.45, 1.46, 1.47, 1.48)
EPS = 1.0e-12

CLAIM_BOUNDARY = (
    "Variable-pair synthetic G4.8D 2D balance cartography only; a predeclared "
    "pair-bridge by bridge-host grid replays the frozen G4.3 handle, G4.5 "
    "selector, and G4.6 schedule. The diagnostic tests whether ready "
    "opportunity is a reproducible local band or a knife-edge artifact. It does "
    "not retune selectors, search thresholds for a policy, promote walls, "
    "identify pathways, replay NanoClustering, evaluate quality/cost value, or "
    "make algorithm-level claims."
)
ROUTE_EXECUTION_STATUS = "executed_g4_8d_2d_balance_cartography"
WALL_PROMOTION_STATUS = "not_promoted_2d_balance_cartography_only"
METHOD_STATUS = "balance_cartography_not_algorithm_claim"


@dataclass(frozen=True)
class BalanceCase:
    case_id: str
    pair_bridge_weight: float
    bridge_host_weight: float
    pair_bridge_index: int
    bridge_host_index: int

    def to_panel_case(self) -> PanelCase:
        return PanelCase(
            case_id=self.case_id,
            panel_role="positive_holdout",
            expected_gate="balance_cartography_no_expected_success",
            direct_weight=DIRECT_WEIGHT,
            pair_bridge_weight=self.pair_bridge_weight,
            bridge_host_weight=self.bridge_host_weight,
            host_clique_weight=HOST_CLIQUE_WEIGHT,
            note=(
                "Predeclared 2D pair-bridge by bridge-host balance cartography "
                "cell around the G4.3 ready anchor."
            ),
        )


def _case_id(pair: float, bridge: float) -> str:
    return f"g4_8d_pair{int(round(pair * 100)):03d}_bridge{int(round(bridge * 100)):03d}"


def _balance_cases() -> tuple[BalanceCase, ...]:
    cases: list[BalanceCase] = []
    for pair_index, pair in enumerate(PAIR_BRIDGE_VALUES):
        for bridge_index, bridge in enumerate(BRIDGE_HOST_VALUES):
            cases.append(
                BalanceCase(
                    case_id=_case_id(pair, bridge),
                    pair_bridge_weight=float(pair),
                    bridge_host_weight=float(bridge),
                    pair_bridge_index=pair_index,
                    bridge_host_index=bridge_index,
                )
            )
    return tuple(cases)


BALANCE_CASES = _balance_cases()


def _claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _panel_design_rows() -> pd.DataFrame:
    rows = []
    for case in BALANCE_CASES:
        rows.append(
            {
                "case_id": case.case_id,
                "pair_bridge_weight": case.pair_bridge_weight,
                "bridge_host_weight": case.bridge_host_weight,
                "pair_bridge_index": case.pair_bridge_index,
                "bridge_host_index": case.bridge_host_index,
                "direct_weight": DIRECT_WEIGHT,
                "host_clique_weight": HOST_CLIQUE_WEIGHT,
                "panel_role": "positive_holdout",
                "expected_gate": "balance_cartography_no_expected_success",
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
    panel_cases = tuple(case.to_panel_case() for case in BALANCE_CASES)
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
        "schema": "variable_pair_synthetic_g4_8d_g4_3_config.v1",
        "output_dir": str(output_dir),
        "panel_cases": [case.__dict__ for case in BALANCE_CASES],
        "handle_policies": list(HANDLE_POLICIES),
        "baseline_seeds": int(baseline_seeds),
        "handle_seeds": int(handle_seeds),
        "n_iterations": int(n_iterations),
        "stage_claim_boundary": G4_3_CLAIM_BOUNDARY,
        "g4_8d_claim_boundary": CLAIM_BOUNDARY,
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
        "bridge_host_weight",
        "pair_bridge_index",
        "bridge_host_index",
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
            rows["baseline_known_coassigned_hit_rate"].fillna(0.0).astype(float) + EPS
        )
    )
    rows["cartography_status"] = [
        _cartography_status(row) for row in rows.to_dict("records")
    ]
    rows["role_symbol"] = [
        _role_symbol(row) for row in rows.to_dict("records")
    ]
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
                "role_symbols": "".join(group.sort_values("bridge_host_weight" if group_col == "pair_bridge_weight" else "pair_bridge_weight")["role_symbol"].astype(str)),
                "case_ids": ";".join(sorted(group["case_id"].astype(str))),
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _matrix_rows(case_summary: pd.DataFrame, value_col: str) -> pd.DataFrame:
    matrix = case_summary.pivot(
        index="bridge_host_weight",
        columns="pair_bridge_weight",
        values=value_col,
    ).sort_index(ascending=False)
    matrix = matrix.reset_index()
    matrix.columns = [str(col) for col in matrix.columns]
    return _claim_columns(matrix)


def _ready_components(case_summary: pd.DataFrame) -> list[dict[str, Any]]:
    ready = case_summary[case_summary["role_symbol"].eq("R")].copy()
    ready_coords = {
        (int(row["pair_bridge_index"]), int(row["bridge_host_index"])): row
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
            x, y = cur
            for nxt in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nxt in ready_coords and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(
            {
                "component_size": int(len(members)),
                "pair_bridge_min": float(min(row["pair_bridge_weight"] for row in members)),
                "pair_bridge_max": float(max(row["pair_bridge_weight"] for row in members)),
                "bridge_host_min": float(min(row["bridge_host_weight"] for row in members)),
                "bridge_host_max": float(max(row["bridge_host_weight"] for row in members)),
                "case_ids": ";".join(sorted(str(row["case_id"]) for row in members)),
            }
        )
    return sorted(components, key=lambda row: row["component_size"], reverse=True)


def _balance_read(case_summary: pd.DataFrame) -> dict[str, Any]:
    ready_count = int(case_summary["role_symbol"].eq("R").sum())
    components = _ready_components(case_summary)
    largest_component = int(components[0]["component_size"]) if components else 0
    anchor = case_summary[
        case_summary["pair_bridge_weight"].eq(1.35)
        & case_summary["bridge_host_weight"].eq(1.45)
    ]
    anchor_ready = bool(anchor["role_symbol"].eq("R").all()) if not anchor.empty else False
    diagonal_ready = _has_sparse_diagonal_ready_ridge(case_summary)
    if ready_count == 0:
        band_status = "no_ready_opportunity"
    elif ready_count == 1:
        band_status = "single_cell_knife_edge"
    elif diagonal_ready:
        band_status = "sparse_diagonal_ready_ridge"
    elif largest_component >= 3:
        band_status = "multi_cell_ready_band"
    else:
        band_status = "fragmented_ready_cells"
    ready_rows = case_summary[case_summary["role_symbol"].eq("R")]
    return {
        "ready_count": ready_count,
        "ready_component_count": int(len(components)),
        "largest_ready_component_size": largest_component,
        "anchor_ready": anchor_ready,
        "sparse_diagonal_ready_ridge": diagonal_ready,
        "ready_pair_bridge_values": sorted(
            ready_rows["pair_bridge_weight"].astype(float).unique().tolist()
        ),
        "ready_bridge_host_values": sorted(
            ready_rows["bridge_host_weight"].astype(float).unique().tolist()
        ),
        "band_status": band_status,
        "ready_components": components,
    }


def _has_sparse_diagonal_ready_ridge(case_summary: pd.DataFrame) -> bool:
    ready = (
        case_summary[case_summary["role_symbol"].eq("R")]
        .sort_values(["pair_bridge_weight", "bridge_host_weight"])
        .copy()
    )
    if len(ready) < 3:
        return False
    pairs = ready["pair_bridge_weight"].astype(float).tolist()
    bridges = ready["bridge_host_weight"].astype(float).tolist()
    if len(set(pairs)) != len(pairs) or len(set(bridges)) != len(bridges):
        return False
    return all(
        pairs[index] < pairs[index + 1] and bridges[index] < bridges[index + 1]
        for index in range(len(pairs) - 1)
    )


def _summary(
    *,
    output_dir: Path,
    g4_3_summary: dict[str, Any],
    g4_5_summary: dict[str, Any],
    g4_6_summary: dict[str, Any],
    case_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    bridge_summary: pd.DataFrame,
) -> dict[str, Any]:
    balance = _balance_read(case_summary)
    return {
        "schema": "variable_pair_synthetic_g4_8d_balance_cartography_summary.v1",
        "status": ROUTE_EXECUTION_STATUS,
        "output_dir": str(output_dir),
        "case_count": int(len(case_summary)),
        "pair_bridge_values": list(PAIR_BRIDGE_VALUES),
        "bridge_host_values": list(BRIDGE_HOST_VALUES),
        "ready_count": int(balance["ready_count"]),
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
        "ready_component_count": int(balance["ready_component_count"]),
        "largest_ready_component_size": int(balance["largest_ready_component_size"]),
        "anchor_ready": bool(balance["anchor_ready"]),
        "sparse_diagonal_ready_ridge": bool(balance["sparse_diagonal_ready_ridge"]),
        "ready_pair_bridge_values": balance["ready_pair_bridge_values"],
        "ready_bridge_host_values": balance["ready_bridge_host_values"],
        "band_status": str(balance["band_status"]),
        "ready_components": balance["ready_components"],
        "pair_summary_row_count": int(len(pair_summary)),
        "bridge_summary_row_count": int(len(bridge_summary)),
        "g4_3_positive_pass_count": int(g4_3_summary.get("positive_pass_count", 0)),
        "g4_5_selector_gate_passed": bool(g4_5_summary.get("selector_gate_passed", False)),
        "g4_6_schedule_gate_passed": bool(g4_6_summary.get("schedule_gate_passed", False)),
        "recommended_next_gate": _recommended_next_gate(str(balance["band_status"])),
        "stage_summary_paths": {
            "g4_3": str(output_dir / G4_3_DIRNAME / G4_3_SUMMARY_JSON),
            "g4_4": str(output_dir / G4_4_DIRNAME / G4_4_SUMMARY_JSON),
            "g4_5": str(output_dir / G4_5_DIRNAME / G4_5_SUMMARY_JSON),
            "g4_6": str(output_dir / G4_6_DIRNAME / G4_6_SUMMARY_JSON),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _recommended_next_gate(band_status: str) -> str:
    if band_status == "sparse_diagonal_ready_ridge":
        return (
            "Refine along the diagonal ready ridge with intermediate "
            "pair-bridge/bridge-host cells before source-discovery replacement."
        )
    if band_status == "multi_cell_ready_band":
        return (
            "Freeze a ready-band construction rule and test it on fresh direct/"
            "host contexts before source-discovery replacement."
        )
    if band_status == "single_cell_knife_edge":
        return (
            "Treat ready opportunity as knife-edge under this synthetic family; "
            "redesign graph construction before source-discovery replacement."
        )
    return (
        "Use the 2D map to refine the construction hypothesis before any "
        "source-discovery replacement."
    )


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    case_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    bridge_summary: pd.DataFrame,
    role_matrix: pd.DataFrame,
) -> None:
    lines = [
        "# Variable-Pair Synthetic G4.8D Balance Cartography",
        "",
        f"- status: `{summary['status']}`",
        f"- band_status: {summary['band_status']}",
        f"- ready_count: {summary['ready_count']}",
        f"- largest_ready_component_size: {summary['largest_ready_component_size']}",
        f"- anchor_ready: {summary['anchor_ready']}",
        f"- sparse_diagonal_ready_ridge: {summary['sparse_diagonal_ready_ridge']}",
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
        "Symbols: R=ready with source-handle fire, T=target saturation, "
        "N=nonrobust coexistence.",
        "",
    ]
    display_cols = ["bridge_host_weight"] + [str(value) for value in PAIR_BRIDGE_VALUES]
    lines.extend(_markdown_table(role_matrix[display_cols]))
    lines.extend(["", "## Pair-Bridge Summary"])
    for row in pair_summary.itertuples(index=False):
        lines.append(
            "- "
            f"pair_bridge={row.pair_bridge_weight:.2f}: roles={row.role_symbols}, "
            f"ready={row.ready_count}, target={row.target_saturation_count}, "
            f"nonrobust={row.nonrobust_coexistence_count}, "
            f"source_fire={row.source_handle_fire_count}"
        )
    lines.extend(["", "## Bridge-Host Summary"])
    for row in bridge_summary.itertuples(index=False):
        lines.append(
            "- "
            f"bridge_host={row.bridge_host_weight:.2f}: roles={row.role_symbols}, "
            f"ready={row.ready_count}, target={row.target_saturation_count}, "
            f"nonrobust={row.nonrobust_coexistence_count}, "
            f"source_fire={row.source_handle_fire_count}"
        )
    lines.extend(["", "## Ready Components"])
    for component in summary["ready_components"]:
        lines.append(
            "- "
            f"size={component['component_size']}, "
            f"pair={component['pair_bridge_min']:.2f}-{component['pair_bridge_max']:.2f}, "
            f"bridge={component['bridge_host_min']:.2f}-{component['bridge_host_max']:.2f}, "
            f"ids={component['case_ids']}"
        )
    lines.extend(["", "## Case Summary"])
    for row in case_summary.itertuples(index=False):
        lines.append(
            "- "
            f"{row.case_id}: {row.role_symbol}/{row.cartography_status}; "
            f"pair_bridge={row.pair_bridge_weight:.2f}, "
            f"bridge_host={row.bridge_host_weight:.2f}, "
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
                "G4.8D is a 2D mechanism map. It should decide whether a later "
                "construction rule is defensible before any source-discovery or "
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
    bridge_summary = _group_summary(case_summary, "bridge_host_weight")
    role_matrix = _matrix_rows(case_summary, "role_symbol")
    status_matrix = _matrix_rows(case_summary, "cartography_status")
    _write_csv(case_summary, output_dir / CASE_SUMMARY_CSV)
    _write_csv(pair_summary, output_dir / PAIR_SUMMARY_CSV)
    _write_csv(bridge_summary, output_dir / BRIDGE_SUMMARY_CSV)
    _write_csv(role_matrix, output_dir / ROLE_MATRIX_CSV)
    _write_csv(status_matrix, output_dir / STATUS_MATRIX_CSV)
    summary = _summary(
        output_dir=output_dir,
        g4_3_summary=g4_3_summary,
        g4_5_summary=g4_5_summary,
        g4_6_summary=g4_6_summary,
        case_summary=case_summary,
        pair_summary=pair_summary,
        bridge_summary=bridge_summary,
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "variable_pair_synthetic_g4_8d_balance_cartography_config.v1",
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
        "bridge_host_values": list(BRIDGE_HOST_VALUES),
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
        bridge_summary=bridge_summary,
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
