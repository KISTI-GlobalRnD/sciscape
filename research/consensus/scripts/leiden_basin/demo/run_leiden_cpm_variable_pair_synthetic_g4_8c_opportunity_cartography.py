#!/usr/bin/env python3
"""Map the opportunity-construction surface around G4.3 ready anchors.

G4.8B showed that fresh predeclared ready and pair-only cells can collapse into
target saturation before the frozen schedule has a separated source endpoint to
act on. This G4.8C diagnostic therefore does not replace source availability
with source discovery yet. It first maps which local graph-weight perturbations
preserve endpoint coexistence and bridge-release eligible separated sources.

The G4.3 bridge-release handle, G4.5 selector, and G4.6 schedule are frozen.
The panel is predeclared as anchor replays, one-axis local perturbations, and a
small decomposition of the G4.8B collapse direction. This is mechanism
cartography only, not selector retuning, wall promotion, pathway identification,
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
    / "leiden_basin_variable_pair_synthetic_g4_8c_opportunity_cartography_v1_20260603"
)

G4_3_DIRNAME = "g4_3_handle_probe"
G4_4_DIRNAME = "g4_4_restart_comparison"
G4_5_DIRNAME = "g4_5_selector_suppression"
G4_6_DIRNAME = "g4_6_schedule_accounting"
PANEL_DESIGN_CSV = "variable_pair_synthetic_g4_8c_panel_design.csv"
CASE_SUMMARY_CSV = "variable_pair_synthetic_g4_8c_case_summary.csv"
AXIS_SUMMARY_CSV = "variable_pair_synthetic_g4_8c_axis_summary.csv"
REGIME_SUMMARY_CSV = "variable_pair_synthetic_g4_8c_regime_summary.csv"
SUMMARY_JSON = "variable_pair_synthetic_g4_8c_summary.json"
CONFIG_JSON = "variable_pair_synthetic_g4_8c_config.json"
REPORT_MD = "variable_pair_synthetic_g4_8c_report.md"

EPS = 1.0e-12

CLAIM_BOUNDARY = (
    "Variable-pair synthetic G4.8C opportunity-construction cartography only; "
    "fresh anchor and perturbation cases replay the frozen G4.3 handle, G4.5 "
    "selector, and G4.6 schedule. The diagnostic maps endpoint coexistence and "
    "bridge-release eligible separated sources before any source-discovery "
    "replacement. No selector retuning, wall promotion, pathway identification, "
    "full NanoClustering replay, quality/cost value, or algorithm-level claims."
)
ROUTE_EXECUTION_STATUS = "executed_g4_8c_opportunity_construction_cartography"
WALL_PROMOTION_STATUS = "not_promoted_opportunity_cartography_only"
METHOD_STATUS = "opportunity_cartography_not_algorithm_claim"


@dataclass(frozen=True)
class CartographyCase:
    case_id: str
    cartography_family: str
    axis_group: str
    perturbation_label: str
    direct_weight: float
    pair_bridge_weight: float
    bridge_host_weight: float
    host_clique_weight: float
    panel_role: str = "positive_holdout"
    expected_gate: str = "opportunity_cartography_no_expected_success"
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


def _case_id(prefix: str, direct: float, pair: float, bridge: float, host: float) -> str:
    return (
        f"{prefix}_d{int(round(direct * 100)):03d}"
        f"_p{int(round(pair * 100)):03d}"
        f"_b{int(round(bridge * 100)):03d}"
        f"_h{int(round(host * 100)):03d}"
    )


def _cartography_cases() -> tuple[CartographyCase, ...]:
    cases: list[CartographyCase] = []
    base = {
        "direct_weight": 1.08,
        "pair_bridge_weight": 1.35,
        "bridge_host_weight": 1.45,
        "host_clique_weight": 1.25,
    }

    anchor_specs = [
        ("g4_3_anchor_direct_low_104", 1.04, 1.35, 1.45, 1.25),
        ("g4_3_anchor_direct_high_112", 1.12, 1.35, 1.45, 1.25),
        ("g4_3_anchor_host_low_120", 1.08, 1.35, 1.45, 1.20),
        ("g4_3_anchor_host_high_130", 1.08, 1.35, 1.45, 1.30),
    ]
    for label, direct, pair, bridge, host in anchor_specs:
        cases.append(
            CartographyCase(
                case_id=f"g4_8c_{label}",
                cartography_family="g4_3_ready_anchor_replay",
                axis_group="anchor_replay",
                perturbation_label=label,
                direct_weight=direct,
                pair_bridge_weight=pair,
                bridge_host_weight=bridge,
                host_clique_weight=host,
                expected_gate="bridge_release_robust_pair_coassignment",
                note="Exact G4.3 ready anchor replay for opportunity preservation.",
            )
        )

    axis_values = {
        "direct": [1.04, 1.06, 1.08, 1.10, 1.12],
        "pair_bridge": [1.32, 1.34, 1.35, 1.36, 1.38, 1.40],
        "bridge_host": [1.42, 1.44, 1.45, 1.46, 1.48],
        "host_clique": [1.20, 1.23, 1.25, 1.27, 1.30],
    }
    for axis, values in axis_values.items():
        for value in values:
            weights = dict(base)
            if axis == "direct":
                weights["direct_weight"] = value
            elif axis == "pair_bridge":
                weights["pair_bridge_weight"] = value
            elif axis == "bridge_host":
                weights["bridge_host_weight"] = value
            elif axis == "host_clique":
                weights["host_clique_weight"] = value
            else:
                raise ValueError(axis)
            direct = float(weights["direct_weight"])
            pair = float(weights["pair_bridge_weight"])
            bridge = float(weights["bridge_host_weight"])
            host = float(weights["host_clique_weight"])
            cases.append(
                CartographyCase(
                    case_id=_case_id(f"g4_8c_axis_{axis}", direct, pair, bridge, host),
                    cartography_family="one_axis_local_perturbation",
                    axis_group=axis,
                    perturbation_label=f"{axis}={value:.2f}",
                    direct_weight=direct,
                    pair_bridge_weight=pair,
                    bridge_host_weight=bridge,
                    host_clique_weight=host,
                    note=(
                        "One-axis perturbation around the G4.3 ready-anchor "
                        "central point."
                    ),
                )
            )

    decomposition_specs = [
        ("direct_106_only", 1.06, 1.35, 1.45, 1.25),
        ("pair_134_only", 1.08, 1.34, 1.45, 1.25),
        ("bridge_146_only", 1.08, 1.35, 1.46, 1.25),
        ("host_123_only", 1.08, 1.35, 1.45, 1.23),
        ("g4_8b_ready_mid_combo", 1.06, 1.34, 1.46, 1.23),
    ]
    for label, direct, pair, bridge, host in decomposition_specs:
        cases.append(
            CartographyCase(
                case_id=_case_id(f"g4_8c_decomp_{label}", direct, pair, bridge, host),
                cartography_family="g4_8b_collapse_decomposition",
                axis_group="combined_decomposition",
                perturbation_label=label,
                direct_weight=direct,
                pair_bridge_weight=pair,
                bridge_host_weight=bridge,
                host_clique_weight=host,
                note=(
                    "Decomposes the G4.8B ready-cell collapse direction into "
                    "single and combined shifts."
                ),
            )
        )
    return tuple(cases)


CARTOGRAPHY_CASES = _cartography_cases()


def _claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _panel_design_rows() -> pd.DataFrame:
    rows = []
    for case in CARTOGRAPHY_CASES:
        rows.append(
            {
                "case_id": case.case_id,
                "cartography_family": case.cartography_family,
                "axis_group": case.axis_group,
                "perturbation_label": case.perturbation_label,
                "panel_role": case.panel_role,
                "expected_gate": case.expected_gate,
                "direct_weight": case.direct_weight,
                "pair_bridge_weight": case.pair_bridge_weight,
                "bridge_host_weight": case.bridge_host_weight,
                "host_clique_weight": case.host_clique_weight,
                "pair_node_size": case.pair_node_size,
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
    panel_cases = tuple(case.to_panel_case() for case in CARTOGRAPHY_CASES)
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
        "schema": "variable_pair_synthetic_g4_8c_g4_3_config.v1",
        "output_dir": str(output_dir),
        "panel_cases": [case.__dict__ for case in CARTOGRAPHY_CASES],
        "handle_policies": list(HANDLE_POLICIES),
        "baseline_seeds": int(baseline_seeds),
        "handle_seeds": int(handle_seeds),
        "n_iterations": int(n_iterations),
        "stage_claim_boundary": G4_3_CLAIM_BOUNDARY,
        "g4_8c_claim_boundary": CLAIM_BOUNDARY,
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
        "cartography_family",
        "axis_group",
        "perturbation_label",
        "direct_weight",
        "pair_bridge_weight",
        "bridge_host_weight",
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
        _cartography_role(regime)
        for regime in rows["observed_opportunity_regime"]
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
    return _claim_columns(rows)


def _cartography_regime(row: dict[str, Any]) -> str:
    regime = _classify_opportunity(row)
    if regime == "unclassified_opportunity_boundary":
        if (
            bool(row["endpoint_coexistence"])
            and bool(row["bridge_release_eligible"])
            and not bool(row["bridge_release_robust"])
        ):
            return "coexistence_bridge_eligible_nonrobust_boundary"
        if bool(row["endpoint_coexistence"]):
            return "coexistence_nonready_boundary"
    return regime


def _cartography_role(regime: str) -> str:
    if regime == "coexistence_bridge_eligible_nonrobust_boundary":
        return "nonrobust_coexistence_boundary"
    if regime == "coexistence_nonready_boundary":
        return "nonready_coexistence_boundary"
    return _next_gate_role({"opportunity_regime": regime})


def _cartography_status(row: dict[str, Any]) -> str:
    role = str(row["observed_next_gate_role"])
    if role == "ready_positive_anchor":
        if bool(row["source_handle_fire"]):
            return "ready_opportunity_with_source_handle_fire"
        return "ready_opportunity_without_schedule_fire"
    if role == "target_saturation_boundary":
        return "collapsed_to_target_saturation"
    if role == "nonrobust_coexistence_boundary":
        return "coexistence_bridge_eligible_nonrobust"
    if role == "nonready_coexistence_boundary":
        return "coexistence_nonready_boundary"
    if role == "suppressed_control_anchor":
        return "coexistence_but_selector_suppressed"
    if role == "competing_pair_only_boundary":
        return "pair_only_boundary"
    if role == "no_target_boundary":
        return "no_target_boundary"
    return "unclassified_cartography_status"


def _group_summary(case_summary: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in case_summary.groupby(group_col, sort=True):
        rows.append(
            {
                group_col: str(key),
                "case_count": int(len(group)),
                "ready_opportunity_count": int(
                    group["observed_next_gate_role"].eq("ready_positive_anchor").sum()
                ),
                "source_handle_fire_count": int(
                    group["source_handle_fire"].astype(bool).sum()
                ),
                "target_saturation_count": int(
                    group["observed_next_gate_role"].eq("target_saturation_boundary").sum()
                ),
                "suppressed_control_count": int(
                    group["observed_next_gate_role"].eq("suppressed_control_anchor").sum()
                ),
                "nonrobust_coexistence_count": int(
                    group["observed_next_gate_role"]
                    .eq("nonrobust_coexistence_boundary")
                    .sum()
                ),
                "pair_only_count": int(
                    group["observed_next_gate_role"]
                    .eq("competing_pair_only_boundary")
                    .sum()
                ),
                "no_target_count": int(
                    group["observed_next_gate_role"].eq("no_target_boundary").sum()
                ),
                "baseline_pair_share_min": float(
                    group["baseline_pair_coassigned_run_share"].min()
                ),
                "baseline_pair_share_max": float(
                    group["baseline_pair_coassigned_run_share"].max()
                ),
                "case_ids": ";".join(sorted(group["case_id"].astype(str))),
                "status_counts": json.dumps(
                    group["cartography_status"].value_counts().to_dict(),
                    sort_keys=True,
                ),
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _summary(
    *,
    output_dir: Path,
    g4_3_summary: dict[str, Any],
    g4_5_summary: dict[str, Any],
    g4_6_summary: dict[str, Any],
    case_summary: pd.DataFrame,
    axis_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema": "variable_pair_synthetic_g4_8c_opportunity_cartography_summary.v1",
        "status": ROUTE_EXECUTION_STATUS,
        "output_dir": str(output_dir),
        "case_count": int(len(case_summary)),
        "axis_group_count": int(case_summary["axis_group"].nunique()),
        "cartography_family_count": int(case_summary["cartography_family"].nunique()),
        "ready_opportunity_count": int(
            case_summary["observed_next_gate_role"].eq("ready_positive_anchor").sum()
        ),
        "ready_with_source_handle_fire_count": int(
            case_summary["cartography_status"]
            .eq("ready_opportunity_with_source_handle_fire")
            .sum()
        ),
        "target_saturation_count": int(
            case_summary["observed_next_gate_role"].eq("target_saturation_boundary").sum()
        ),
        "suppressed_control_count": int(
            case_summary["observed_next_gate_role"].eq("suppressed_control_anchor").sum()
        ),
        "nonrobust_coexistence_count": int(
            case_summary["observed_next_gate_role"]
            .eq("nonrobust_coexistence_boundary")
            .sum()
        ),
        "pair_only_count": int(
            case_summary["observed_next_gate_role"]
            .eq("competing_pair_only_boundary")
            .sum()
        ),
        "no_target_count": int(
            case_summary["observed_next_gate_role"].eq("no_target_boundary").sum()
        ),
        "source_handle_fire_count": int(
            case_summary["source_handle_fire"].astype(bool).sum()
        ),
        "cartography_status_counts": case_summary[
            "cartography_status"
        ].value_counts().to_dict(),
        "observed_role_counts": case_summary[
            "observed_next_gate_role"
        ].value_counts().to_dict(),
        "axis_summary_row_count": int(len(axis_summary)),
        "regime_summary_row_count": int(len(regime_summary)),
        "g4_3_positive_pass_count": int(g4_3_summary.get("positive_pass_count", 0)),
        "g4_5_selector_gate_passed": bool(g4_5_summary.get("selector_gate_passed", False)),
        "g4_6_schedule_gate_passed": bool(g4_6_summary.get("schedule_gate_passed", False)),
        "recommended_next_gate": (
            "Use the G4.8C ready-preserving cases to define a construction rule "
            "before replacing observed source availability with source discovery."
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
    axis_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Variable-Pair Synthetic G4.8C Opportunity Cartography",
        "",
        f"- status: `{summary['status']}`",
        f"- ready_opportunity_count: {summary['ready_opportunity_count']}",
        (
            "- ready_with_source_handle_fire_count: "
            f"{summary['ready_with_source_handle_fire_count']}"
        ),
        f"- target_saturation_count: {summary['target_saturation_count']}",
        f"- suppressed_control_count: {summary['suppressed_control_count']}",
        f"- nonrobust_coexistence_count: {summary['nonrobust_coexistence_count']}",
        f"- pair_only_count: {summary['pair_only_count']}",
        f"- no_target_count: {summary['no_target_count']}",
        f"- source_handle_fire_count: {summary['source_handle_fire_count']}",
        f"- cartography_status_counts: {summary['cartography_status_counts']}",
        f"- observed_role_counts: {summary['observed_role_counts']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Axis Summary",
    ]
    for row in axis_summary.itertuples(index=False):
        lines.append(
            "- "
            f"{row.axis_group}: cases={row.case_count}, "
            f"ready={row.ready_opportunity_count}, "
            f"source_fire={row.source_handle_fire_count}, "
            f"target_saturation={row.target_saturation_count}, "
            f"suppressed={row.suppressed_control_count}, "
            f"nonrobust={row.nonrobust_coexistence_count}, "
            f"pair_only={row.pair_only_count}, "
            f"pair_share_range={row.baseline_pair_share_min:.3f}-"
            f"{row.baseline_pair_share_max:.3f}, "
            f"statuses={row.status_counts}"
        )
    lines.extend(["", "## Regime Summary"])
    for row in regime_summary.itertuples(index=False):
        lines.append(
            "- "
            f"{row.observed_next_gate_role}: cases={row.case_count}, "
            f"ready={row.ready_opportunity_count}, "
            f"source_fire={row.source_handle_fire_count}, "
            f"target_saturation={row.target_saturation_count}, "
            f"nonrobust={row.nonrobust_coexistence_count}, "
            f"ids={row.case_ids}"
        )
    lines.extend(["", "## Case Summary"])
    for row in case_summary.itertuples(index=False):
        lines.append(
            "- "
            f"{row.case_id} ({row.axis_group}, {row.perturbation_label}): "
            f"{row.cartography_status}; observed={row.observed_next_gate_role}/"
            f"{row.observed_opportunity_regime}, "
            f"weights=d{row.direct_weight:.2f}/p{row.pair_bridge_weight:.2f}/"
            f"b{row.bridge_host_weight:.2f}/h{row.host_clique_weight:.2f}, "
            f"pair_share={row.baseline_pair_coassigned_run_share:.3f}, "
            f"separated={row.separated_endpoint_count}, "
            f"coassigned={row.coassigned_endpoint_count}, "
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
                "G4.8C is mechanism cartography. It should decide whether a "
                "later source-discovery gate has a stable ready-opportunity "
                "surface to discover. It does not tune the selector or promote "
                "a method claim."
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
    axis_summary = _group_summary(case_summary, "axis_group")
    regime_summary = _group_summary(case_summary, "observed_next_gate_role")
    _write_csv(case_summary, output_dir / CASE_SUMMARY_CSV)
    _write_csv(axis_summary, output_dir / AXIS_SUMMARY_CSV)
    _write_csv(regime_summary, output_dir / REGIME_SUMMARY_CSV)
    summary = _summary(
        output_dir=output_dir,
        g4_3_summary=g4_3_summary,
        g4_5_summary=g4_5_summary,
        g4_6_summary=g4_6_summary,
        case_summary=case_summary,
        axis_summary=axis_summary,
        regime_summary=regime_summary,
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "variable_pair_synthetic_g4_8c_opportunity_cartography_config.v1",
        "output_dir": str(output_dir),
        "stage_dirs": {
            "g4_3": str(g4_3_dir),
            "g4_4": str(g4_4_dir),
            "g4_5": str(g4_5_dir),
            "g4_6": str(g4_6_dir),
        },
        "panel_cases": [case.__dict__ for case in CARTOGRAPHY_CASES],
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
        axis_summary=axis_summary,
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
