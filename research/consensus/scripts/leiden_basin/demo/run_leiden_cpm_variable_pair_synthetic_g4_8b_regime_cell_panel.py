#!/usr/bin/env python3
"""Run the frozen schedule on a predeclared G4.8B regime-cell panel.

This diagnostic follows the G4.8A opportunity-regime design. It creates a fresh
synthetic panel by regime cell, then replays the frozen G4.3 handle, G4.5
selector, and G4.6 schedule without retuning. The pass condition is
classification-level: source-handle schedule fire should occur only in
ready bridge-release opportunity cells, while suppressed, saturated, pair-only,
and no-target boundary cells should have no added source-handle leak.

This is still a synthetic regime-cell gate. It does not promote basin walls,
identify pathways, replace source discovery, replay NanoClustering, measure
quality/cost value, or make algorithm-level claims.
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
from analyze_leiden_cpm_variable_pair_synthetic_g4_8_opportunity_regime_design import (
    _classify_opportunity,
    _next_gate_role,
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


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_variable_pair_synthetic_g4_8b_regime_cell_panel_v1_20260603"
)

G4_3_DIRNAME = "g4_3_handle_probe"
G4_4_DIRNAME = "g4_4_restart_comparison"
G4_5_DIRNAME = "g4_5_selector_suppression"
G4_6_DIRNAME = "g4_6_schedule_accounting"
PANEL_DESIGN_CSV = "variable_pair_synthetic_g4_8b_panel_design.csv"
CASE_SUMMARY_CSV = "variable_pair_synthetic_g4_8b_case_summary.csv"
REGIME_SUMMARY_CSV = "variable_pair_synthetic_g4_8b_regime_summary.csv"
SUMMARY_JSON = "variable_pair_synthetic_g4_8b_summary.json"
CONFIG_JSON = "variable_pair_synthetic_g4_8b_config.json"
REPORT_MD = "variable_pair_synthetic_g4_8b_report.md"

EPS = 1.0e-12

CLAIM_BOUNDARY = (
    "Variable-pair synthetic G4.8B predeclared regime-cell diagnostic only; "
    "fresh synthetic cell variants replay the frozen G4.3 handle, G4.5 selector, "
    "and G4.6 schedule without selector retuning. The gate tests opportunity "
    "regime classification and source-handle no-leak behavior, not wall "
    "promotion, pathway identification, independent source discovery, full "
    "NanoClustering replay, quality/cost value, or algorithm-level claims."
)
ROUTE_EXECUTION_STATUS = "executed_g4_8b_predeclared_regime_cell_panel"
WALL_PROMOTION_STATUS = "not_promoted_regime_cell_gate_only"
METHOD_STATUS = "regime_cell_gate_not_algorithm_claim"


@dataclass(frozen=True)
class RegimePanelCase:
    case_id: str
    panel_role: str
    expected_gate: str
    direct_weight: float
    pair_bridge_weight: float
    bridge_host_weight: float
    host_clique_weight: float
    expected_opportunity_regime: str
    expected_next_gate_role: str
    predeclared_regime_cell: str
    pair_node_size: int = 1
    note: str = ""

    def to_panel_case(self) -> PanelCase:
        return PanelCase(
            case_id=self.case_id,
            panel_role=self.panel_role,
            expected_gate=self.expected_gate,
            direct_weight=self.direct_weight,
            pair_bridge_weight=self.pair_bridge_weight,
            bridge_host_weight=self.bridge_host_weight,
            host_clique_weight=self.host_clique_weight,
            pair_node_size=self.pair_node_size,
            note=self.note,
        )


REGIME_PANEL_CASES: tuple[RegimePanelCase, ...] = (
    RegimePanelCase(
        case_id="g4_8b_ready_bridge_mid_106",
        panel_role="positive_holdout",
        expected_gate="bridge_release_robust_pair_coassignment",
        direct_weight=1.06,
        pair_bridge_weight=1.34,
        bridge_host_weight=1.46,
        host_clique_weight=1.23,
        expected_opportunity_regime="bridge_release_opportunity_ready",
        expected_next_gate_role="ready_positive_anchor",
        predeclared_regime_cell="ready_bridge_release_opportunity",
        note="Fresh ready cell inside the G4.8A bridge-release opportunity band.",
    ),
    RegimePanelCase(
        case_id="g4_8b_ready_host_high_110",
        panel_role="positive_holdout",
        expected_gate="bridge_release_robust_pair_coassignment",
        direct_weight=1.10,
        pair_bridge_weight=1.36,
        bridge_host_weight=1.46,
        host_clique_weight=1.29,
        expected_opportunity_regime="bridge_release_opportunity_ready",
        expected_next_gate_role="ready_positive_anchor",
        predeclared_regime_cell="ready_bridge_release_opportunity",
        note="Fresh ready cell with stronger host clique than the mid anchor.",
    ),
    RegimePanelCase(
        case_id="g4_8b_suppressed_context_low_139",
        panel_role="matched_control",
        expected_gate="bridge_release_not_robust_pair_coassignment",
        direct_weight=1.09,
        pair_bridge_weight=1.34,
        bridge_host_weight=1.39,
        host_clique_weight=1.25,
        expected_opportunity_regime="coexistence_control_suppressed",
        expected_next_gate_role="suppressed_control_anchor",
        predeclared_regime_cell="suppressed_coexistence_control",
        note="Fresh suppressed-control cell below the bridge-host release band.",
    ),
    RegimePanelCase(
        case_id="g4_8b_suppressed_pair_compete_142",
        panel_role="matched_control",
        expected_gate="bridge_release_not_robust_pair_coassignment",
        direct_weight=1.09,
        pair_bridge_weight=1.42,
        bridge_host_weight=1.46,
        host_clique_weight=1.25,
        expected_opportunity_regime="coexistence_control_suppressed",
        expected_next_gate_role="suppressed_control_anchor",
        predeclared_regime_cell="suppressed_coexistence_control",
        note="Fresh suppressed-control cell with high pair-bridge competition.",
    ),
    RegimePanelCase(
        case_id="g4_8b_saturated_direct_high_115",
        panel_role="positive_holdout",
        expected_gate="bridge_release_not_robust_pair_coassignment",
        direct_weight=1.15,
        pair_bridge_weight=1.30,
        bridge_host_weight=1.49,
        host_clique_weight=1.24,
        expected_opportunity_regime="target_saturated_no_source_opportunity",
        expected_next_gate_role="target_saturation_boundary",
        predeclared_regime_cell="target_saturation_boundary",
        note="Fresh saturated boundary with high direct support and low pair competition.",
    ),
    RegimePanelCase(
        case_id="g4_8b_saturated_pair_low_108",
        panel_role="positive_holdout",
        expected_gate="bridge_release_not_robust_pair_coassignment",
        direct_weight=1.08,
        pair_bridge_weight=1.27,
        bridge_host_weight=1.49,
        host_clique_weight=1.24,
        expected_opportunity_regime="target_saturated_no_source_opportunity",
        expected_next_gate_role="target_saturation_boundary",
        predeclared_regime_cell="target_saturation_boundary",
        note="Fresh saturated boundary with lower pair-bridge competition.",
    ),
    RegimePanelCase(
        case_id="g4_8b_pair_only_floor_101",
        panel_role="positive_holdout",
        expected_gate="bridge_release_not_robust_pair_coassignment",
        direct_weight=1.01,
        pair_bridge_weight=1.31,
        bridge_host_weight=1.48,
        host_clique_weight=1.24,
        expected_opportunity_regime="pair_only_opportunity_not_bridge_release",
        expected_next_gate_role="competing_pair_only_boundary",
        predeclared_regime_cell="pair_only_boundary",
        note="Fresh pair-only boundary near the direct-support floor.",
    ),
    RegimePanelCase(
        case_id="g4_8b_pair_only_host_low_102",
        panel_role="positive_holdout",
        expected_gate="bridge_release_not_robust_pair_coassignment",
        direct_weight=1.02,
        pair_bridge_weight=1.30,
        bridge_host_weight=1.47,
        host_clique_weight=1.18,
        expected_opportunity_regime="pair_only_opportunity_not_bridge_release",
        expected_next_gate_role="competing_pair_only_boundary",
        predeclared_regime_cell="pair_only_boundary",
        note="Fresh pair-only boundary with weaker host context.",
    ),
    RegimePanelCase(
        case_id="g4_8b_no_target_direct_floor_096",
        panel_role="negative_control",
        expected_gate="bridge_release_not_robust_pair_coassignment",
        direct_weight=0.96,
        pair_bridge_weight=1.33,
        bridge_host_weight=1.48,
        host_clique_weight=1.25,
        expected_opportunity_regime="target_absent_no_bridge_source",
        expected_next_gate_role="no_target_boundary",
        predeclared_regime_cell="no_target_boundary",
        note="Fresh no-target boundary below the direct support floor.",
    ),
    RegimePanelCase(
        case_id="g4_8b_no_target_weak_context_100",
        panel_role="negative_control",
        expected_gate="bridge_release_not_robust_pair_coassignment",
        direct_weight=1.06,
        pair_bridge_weight=1.34,
        bridge_host_weight=1.00,
        host_clique_weight=1.25,
        expected_opportunity_regime="target_absent_bridge_release_no_value",
        expected_next_gate_role="no_target_boundary",
        predeclared_regime_cell="no_target_boundary",
        note="Fresh no-target boundary with weak bridge context.",
    ),
)


def _claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _panel_design_rows() -> pd.DataFrame:
    rows = []
    for case in REGIME_PANEL_CASES:
        rows.append(
            {
                "case_id": case.case_id,
                "panel_role": case.panel_role,
                "expected_gate": case.expected_gate,
                "direct_weight": case.direct_weight,
                "pair_bridge_weight": case.pair_bridge_weight,
                "bridge_host_weight": case.bridge_host_weight,
                "host_clique_weight": case.host_clique_weight,
                "pair_node_size": case.pair_node_size,
                "expected_opportunity_regime": case.expected_opportunity_regime,
                "expected_next_gate_role": case.expected_next_gate_role,
                "predeclared_regime_cell": case.predeclared_regime_cell,
                "note": case.note,
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
    panel_cases = tuple(case.to_panel_case() for case in REGIME_PANEL_CASES)
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
        "schema": "variable_pair_synthetic_g4_8b_g4_3_config.v1",
        "output_dir": str(output_dir),
        "panel_cases": [case.__dict__ for case in REGIME_PANEL_CASES],
        "handle_policies": list(HANDLE_POLICIES),
        "baseline_seeds": int(baseline_seeds),
        "handle_seeds": int(handle_seeds),
        "n_iterations": int(n_iterations),
        "stage_claim_boundary": G4_3_CLAIM_BOUNDARY,
        "g4_8b_claim_boundary": CLAIM_BOUNDARY,
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
        "predeclared_regime_cell",
        "expected_opportunity_regime",
        "expected_next_gate_role",
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
        _classify_opportunity(row) for row in rows.to_dict("records")
    ]
    rows["observed_next_gate_role"] = [
        _next_gate_role({"opportunity_regime": regime})
        for regime in rows["observed_opportunity_regime"]
    ]
    rows["expected_role_reproduced"] = (
        rows["observed_next_gate_role"].astype(str)
        == rows["expected_next_gate_role"].astype(str)
    )
    rows["source_handle_fire"] = (
        rows["handle_application_rate"].fillna(0.0).astype(float).gt(0.0)
    )
    rows["no_added_source_handle_leak"] = (
        rows["schedule_probability_lift_vs_baseline"].fillna(0.0).astype(float) <= EPS
    ) & (~rows["source_handle_fire"])
    rows["g4_8b_case_status"] = [
        _case_status(row) for row in rows.to_dict("records")
    ]
    rows["g4_8b_case_passed"] = [
        _case_passed(row) for row in rows.to_dict("records")
    ]
    return _claim_columns(rows)


def _case_status(row: dict[str, Any]) -> str:
    if not bool(row["expected_role_reproduced"]):
        return "expected_regime_not_reproduced"
    expected_role = str(row["expected_next_gate_role"])
    if expected_role == "ready_positive_anchor":
        if bool(row["source_handle_fire"]) and str(row["g4_6_case_status"]) == (
            "positive_schedule_beats_restart_with_source_accounting"
        ):
            return "ready_cell_source_handle_fire_success"
        return "ready_cell_missing_source_handle_fire"
    if bool(row["no_added_source_handle_leak"]):
        return "boundary_cell_no_added_source_handle_leak"
    return "boundary_cell_source_handle_leak"


def _case_passed(row: dict[str, Any]) -> bool:
    status = str(row["g4_8b_case_status"])
    return status in {
        "ready_cell_source_handle_fire_success",
        "boundary_cell_no_added_source_handle_leak",
    }


def _regime_summary(case_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["predeclared_regime_cell", "expected_next_gate_role"]
    for keys, group in case_summary.groupby(group_cols, sort=True):
        cell, role = keys
        rows.append(
            {
                "predeclared_regime_cell": str(cell),
                "expected_next_gate_role": str(role),
                "case_count": int(len(group)),
                "case_pass_count": int(group["g4_8b_case_passed"].astype(bool).sum()),
                "case_fail_count": int((~group["g4_8b_case_passed"].astype(bool)).sum()),
                "observed_roles": ";".join(sorted(group["observed_next_gate_role"].unique())),
                "case_ids": ";".join(sorted(group["case_id"].astype(str))),
                "source_handle_fire_count": int(group["source_handle_fire"].astype(bool).sum()),
                "max_schedule_lift_vs_baseline": float(
                    group["schedule_probability_lift_vs_baseline"].max()
                ),
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _summary(
    *,
    output_dir: Path,
    g4_3_summary: dict[str, Any],
    g4_4_summary: dict[str, Any],
    g4_5_summary: dict[str, Any],
    g4_6_summary: dict[str, Any],
    case_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
) -> dict[str, Any]:
    ready = case_summary[case_summary["expected_next_gate_role"].eq("ready_positive_anchor")]
    boundary = case_summary[
        ~case_summary["expected_next_gate_role"].eq("ready_positive_anchor")
    ]
    return {
        "schema": "variable_pair_synthetic_g4_8b_regime_cell_panel_summary.v1",
        "status": ROUTE_EXECUTION_STATUS,
        "output_dir": str(output_dir),
        "case_count": int(len(case_summary)),
        "regime_cell_count": int(case_summary["predeclared_regime_cell"].nunique()),
        "case_pass_count": int(case_summary["g4_8b_case_passed"].astype(bool).sum()),
        "case_fail_count": int((~case_summary["g4_8b_case_passed"].astype(bool)).sum()),
        "ready_case_count": int(len(ready)),
        "ready_source_handle_fire_success_count": int(
            ready["g4_8b_case_status"].eq("ready_cell_source_handle_fire_success").sum()
        ),
        "boundary_case_count": int(len(boundary)),
        "boundary_no_added_leak_count": int(
            boundary["g4_8b_case_status"]
            .eq("boundary_cell_no_added_source_handle_leak")
            .sum()
        ),
        "expected_role_reproduced_count": int(
            case_summary["expected_role_reproduced"].astype(bool).sum()
        ),
        "source_handle_fire_outside_ready_count": int(
            boundary["source_handle_fire"].astype(bool).sum()
        ),
        "boundary_schedule_lift_max": float(
            boundary["schedule_probability_lift_vs_baseline"].max()
        )
        if not boundary.empty
        else 0.0,
        "case_status_counts": case_summary["g4_8b_case_status"].value_counts().to_dict(),
        "observed_role_counts": case_summary["observed_next_gate_role"].value_counts().to_dict(),
        "regime_summary_row_count": int(len(regime_summary)),
        "g4_3_positive_pass_count": int(g4_3_summary.get("positive_pass_count", 0)),
        "g4_5_selector_gate_passed": bool(g4_5_summary.get("selector_gate_passed", False)),
        "g4_6_schedule_gate_passed": bool(g4_6_summary.get("schedule_gate_passed", False)),
        "regime_cell_gate_passed": bool(
            case_summary["g4_8b_case_passed"].astype(bool).all()
            and int(boundary["source_handle_fire"].astype(bool).sum()) == 0
        ),
        "stage_summary_paths": {
            "g4_3": str(output_dir / G4_3_DIRNAME / G4_3_SUMMARY_JSON),
            "g4_4": str(output_dir / G4_4_DIRNAME / G4_4_SUMMARY_JSON),
            "g4_5": str(output_dir / G4_5_DIRNAME / G4_5_SUMMARY_JSON),
            "g4_6": str(output_dir / G4_6_DIRNAME / G4_6_SUMMARY_JSON),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    case_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Variable-Pair Synthetic G4.8B Regime-Cell Panel",
        "",
        f"- status: `{summary['status']}`",
        f"- regime_cell_gate_passed: {summary['regime_cell_gate_passed']}",
        f"- case_pass_count: {summary['case_pass_count']}",
        f"- case_fail_count: {summary['case_fail_count']}",
        f"- ready_source_handle_fire_success_count: {summary['ready_source_handle_fire_success_count']}",
        f"- boundary_no_added_leak_count: {summary['boundary_no_added_leak_count']}",
        f"- source_handle_fire_outside_ready_count: {summary['source_handle_fire_outside_ready_count']}",
        f"- boundary_schedule_lift_max: {summary['boundary_schedule_lift_max']}",
        f"- case_status_counts: {summary['case_status_counts']}",
        f"- observed_role_counts: {summary['observed_role_counts']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Regime Summary",
    ]
    for row in regime_summary.itertuples(index=False):
        lines.append(
            "- "
            f"{row.predeclared_regime_cell}: expected_role={row.expected_next_gate_role}, "
            f"pass={row.case_pass_count}/{row.case_count}, "
            f"observed_roles={row.observed_roles}, "
            f"source_handle_fire_count={row.source_handle_fire_count}, "
            f"max_lift={row.max_schedule_lift_vs_baseline:.6g}, "
            f"ids={row.case_ids}"
        )
    lines.extend(["", "## Case Summary"])
    for row in case_summary.itertuples(index=False):
        lines.append(
            "- "
            f"{row.case_id} ({row.predeclared_regime_cell}): "
            f"{row.g4_8b_case_status}; expected={row.expected_next_gate_role}, "
            f"observed={row.observed_next_gate_role}/{row.observed_opportunity_regime}, "
            f"baseline_pair_share={row.baseline_pair_coassigned_run_share:.3f}, "
            f"selected_sources={row.selected_source_count}, "
            f"handle_fire={row.source_handle_fire}, "
            f"schedule_p={row.schedule_known_coassigned_hit_rate:.3f}, "
            f"baseline_p={row.baseline_known_coassigned_hit_rate:.3f}, "
            f"lift={row.schedule_probability_lift_vs_baseline:.3f}"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "G4.8B is a regime-cell gate. It tests whether the frozen schedule "
                "has source-handle fire only in ready bridge-release opportunity "
                "cells. It is not a selector retuning loop or method claim."
            ),
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


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
    regime_summary = _regime_summary(case_summary)
    _write_csv(case_summary, output_dir / CASE_SUMMARY_CSV)
    _write_csv(regime_summary, output_dir / REGIME_SUMMARY_CSV)
    summary = _summary(
        output_dir=output_dir,
        g4_3_summary=g4_3_summary,
        g4_4_summary=g4_4_summary,
        g4_5_summary=g4_5_summary,
        g4_6_summary=g4_6_summary,
        case_summary=case_summary,
        regime_summary=regime_summary,
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "variable_pair_synthetic_g4_8b_regime_cell_panel_config.v1",
        "output_dir": str(output_dir),
        "stage_dirs": {
            "g4_3": str(g4_3_dir),
            "g4_4": str(g4_4_dir),
            "g4_5": str(g4_5_dir),
            "g4_6": str(g4_6_dir),
        },
        "panel_cases": [case.__dict__ for case in REGIME_PANEL_CASES],
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
        regime_summary=regime_summary,
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
