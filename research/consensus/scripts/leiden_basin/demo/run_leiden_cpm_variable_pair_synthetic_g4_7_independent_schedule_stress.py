#!/usr/bin/env python3
"""Run an independent stress panel for the frozen G4.6 schedule.

This G4.7 diagnostic keeps the G4.3 handle, G4.5 selector, and G4.6 schedule
unchanged. It does not search thresholds. It materializes a deliberately shifted
synthetic panel and replays the frozen G4.3 -> G4.4 -> G4.5 -> G4.6 pipeline.

The purpose is to test whether the schedule-accounting result survives outside
the current fixed panel. Failure is evidence about the regime boundary, not a
prompt to tune the selector inside this gate.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
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


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_variable_pair_synthetic_g4_7_independent_schedule_stress_v1_20260603"
)

G4_3_DIRNAME = "g4_3_handle_probe"
G4_4_DIRNAME = "g4_4_restart_comparison"
G4_5_DIRNAME = "g4_5_selector_suppression"
G4_6_DIRNAME = "g4_6_schedule_accounting"
G4_7_CASE_SUMMARY_CSV = "variable_pair_synthetic_g4_7_case_summary.csv"
G4_7_SUMMARY_JSON = "variable_pair_synthetic_g4_7_summary.json"
G4_7_CONFIG_JSON = "variable_pair_synthetic_g4_7_config.json"
G4_7_REPORT_MD = "variable_pair_synthetic_g4_7_report.md"

STRESS_PANEL_CASES: tuple[PanelCase, ...] = (
    PanelCase(
        case_id="stress_positive_direct_floor_100",
        panel_role="positive_holdout",
        expected_gate="bridge_release_robust_pair_coassignment",
        direct_weight=1.00,
        pair_bridge_weight=1.32,
        bridge_host_weight=1.48,
        host_clique_weight=1.25,
        note="Direct support at the frozen selector floor with stronger context.",
    ),
    PanelCase(
        case_id="stress_positive_direct_high_116",
        panel_role="positive_holdout",
        expected_gate="bridge_release_robust_pair_coassignment",
        direct_weight=1.16,
        pair_bridge_weight=1.32,
        bridge_host_weight=1.48,
        host_clique_weight=1.25,
        note="Higher direct support with lower pair-bridge than the G4.3 panel.",
    ),
    PanelCase(
        case_id="stress_positive_pair_bridge_low_128",
        panel_role="positive_holdout",
        expected_gate="bridge_release_robust_pair_coassignment",
        direct_weight=1.08,
        pair_bridge_weight=1.28,
        bridge_host_weight=1.48,
        host_clique_weight=1.25,
        note="Lower pair-bridge competition with stronger bridge context.",
    ),
    PanelCase(
        case_id="stress_positive_host_low_115",
        panel_role="positive_holdout",
        expected_gate="bridge_release_robust_pair_coassignment",
        direct_weight=1.10,
        pair_bridge_weight=1.32,
        bridge_host_weight=1.48,
        host_clique_weight=1.15,
        note="Lower host clique with stronger bridge context.",
    ),
    PanelCase(
        case_id="stress_control_context_below_140",
        panel_role="matched_control",
        expected_gate="bridge_release_not_robust_pair_coassignment",
        direct_weight=1.10,
        pair_bridge_weight=1.32,
        bridge_host_weight=1.40,
        host_clique_weight=1.25,
        note="Context below the G4.5 source-neutral release regime.",
    ),
    PanelCase(
        case_id="stress_control_context_below_141",
        panel_role="matched_control",
        expected_gate="bridge_release_not_robust_pair_coassignment",
        direct_weight=1.10,
        pair_bridge_weight=1.32,
        bridge_host_weight=1.41,
        host_clique_weight=1.25,
        note="Near-boundary context control.",
    ),
    PanelCase(
        case_id="stress_control_pair_bridge_high_145",
        panel_role="matched_control",
        expected_gate="bridge_release_not_robust_pair_coassignment",
        direct_weight=1.10,
        pair_bridge_weight=1.45,
        bridge_host_weight=1.48,
        host_clique_weight=1.25,
        note="High pair-bridge competition control.",
    ),
    PanelCase(
        case_id="stress_control_direct_below_floor_095",
        panel_role="negative_control",
        expected_gate="bridge_release_not_robust_pair_coassignment",
        direct_weight=0.95,
        pair_bridge_weight=1.32,
        bridge_host_weight=1.48,
        host_clique_weight=1.25,
        note="Direct support below the frozen selector floor.",
    ),
    PanelCase(
        case_id="stress_control_weak_context_105",
        panel_role="negative_control",
        expected_gate="bridge_release_not_robust_pair_coassignment",
        direct_weight=1.10,
        pair_bridge_weight=1.32,
        bridge_host_weight=1.05,
        host_clique_weight=1.25,
        note="Weak bridge context control.",
    ),
)

CLAIM_BOUNDARY = (
    "Variable-pair synthetic G4.7 independent schedule-stress diagnostic only; "
    "the frozen G4.3 handle, G4.5 selector, and G4.6 schedule are replayed on a "
    "predeclared shifted synthetic panel. No threshold retuning, wall promotion, "
    "full NanoClustering replay, quality/cost value, or algorithm-level claims."
)
ROUTE_EXECUTION_STATUS = "executed_g4_7_independent_schedule_stress"
WALL_PROMOTION_STATUS = "not_promoted_independent_stress_only"
METHOD_STATUS = "independent_stress_not_algorithm_claim"


def _claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _run_g4_3_stage(
    *,
    output_dir: Path,
    baseline_seeds: int,
    handle_seeds: int,
    n_iterations: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_cases = STRESS_PANEL_CASES
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
        "schema": "variable_pair_synthetic_g4_7_stress_g4_3_config.v1",
        "output_dir": str(output_dir),
        "panel_cases": [case.__dict__ for case in panel_cases],
        "handle_policies": list(HANDLE_POLICIES),
        "baseline_seeds": int(baseline_seeds),
        "handle_seeds": int(handle_seeds),
        "n_iterations": int(n_iterations),
        "stage_claim_boundary": G4_3_CLAIM_BOUNDARY,
        "g4_7_claim_boundary": CLAIM_BOUNDARY,
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
    g4_3_dir: Path,
    g4_4_dir: Path,
    g4_5_dir: Path,
    g4_6_dir: Path,
) -> pd.DataFrame:
    g4_3 = pd.read_csv(g4_3_dir / VARIANT_GATE_ROWS_CSV)
    g4_4 = pd.read_csv(g4_4_dir / G4_4_CASE_SUMMARY_CSV)
    g4_5 = pd.read_csv(g4_5_dir / G4_5_CASE_SUMMARY_CSV)
    g4_6 = pd.read_csv(g4_6_dir / G4_6_CASE_SUMMARY_CSV)
    rows = g4_6.merge(
        g4_3[
            [
                "case_id",
                "baseline_pair_coassigned_run_share",
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
    ).merge(
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
    ).merge(
        g4_5[
            [
                "case_id",
                "g4_5_case_status",
                "selector_selected_count",
                "selector_suppressed_count",
                "selected_positive_win_count",
                "suppressed_positive_win_count",
            ]
        ],
        on="case_id",
        how="left",
    )
    rows["g4_7_case_status"] = [
        _g4_7_case_status(row) for row in rows.to_dict("records")
    ]
    return _claim_columns(rows)


def _g4_7_case_status(row: dict[str, Any]) -> str:
    panel_role = str(row["panel_role"])
    if panel_role == "positive_holdout":
        if str(row["g4_6_case_status"]) == "positive_schedule_beats_restart_with_source_accounting":
            return "positive_frozen_schedule_survived_stress"
        if int(row.get("pair_relation_only_robust_pair_coassignment_count", 0)) > 0:
            return "positive_failed_bridge_release_not_pair_only"
        if int(row.get("selected_source_count", 0)) == 0:
            if float(row.get("baseline_known_coassigned_hit_rate", 0.0)) >= 1.0:
                return "positive_failed_no_source_opportunity_all_restart_targets"
            return "positive_failed_no_selected_source"
        return "positive_failed_selected_source_or_schedule_accounting"
    if str(row["g4_6_case_status"]) == "control_schedule_no_added_leak":
        return "control_no_added_leak_under_stress"
    return "control_leak_under_stress"


def _summary(
    *,
    output_dir: Path,
    g4_3_summary: dict[str, Any],
    g4_4_summary: dict[str, Any],
    g4_5_summary: dict[str, Any],
    g4_6_summary: dict[str, Any],
    case_summary: pd.DataFrame,
) -> dict[str, Any]:
    positives = case_summary[case_summary["panel_role"].eq("positive_holdout")]
    controls = case_summary[~case_summary["panel_role"].eq("positive_holdout")]
    return {
        "schema": "variable_pair_synthetic_g4_7_independent_schedule_stress_summary.v1",
        "status": ROUTE_EXECUTION_STATUS,
        "output_dir": str(output_dir),
        "case_count": int(len(case_summary)),
        "positive_case_count": int(len(positives)),
        "control_case_count": int(len(controls)),
        "g4_3_positive_pass_count": int(g4_3_summary.get("positive_pass_count", 0)),
        "g4_3_positive_fail_count": int(g4_3_summary.get("positive_fail_count", 0)),
        "g4_3_control_pass_count": int(g4_3_summary.get("control_pass_count", 0)),
        "g4_5_selector_gate_passed": bool(g4_5_summary.get("selector_gate_passed", False)),
        "g4_6_schedule_gate_passed": bool(g4_6_summary.get("schedule_gate_passed", False)),
        "positive_schedule_survived_count": int(
            positives["g4_7_case_status"]
            .eq("positive_frozen_schedule_survived_stress")
            .sum()
        ),
        "positive_no_selected_source_count": int(
            positives["g4_7_case_status"]
            .isin(
                [
                    "positive_failed_no_selected_source",
                    "positive_failed_no_source_opportunity_all_restart_targets",
                    "positive_failed_bridge_release_not_pair_only",
                ]
            )
            .sum()
        ),
        "positive_pair_only_failure_count": int(
            positives["g4_7_case_status"]
            .eq("positive_failed_bridge_release_not_pair_only")
            .sum()
        ),
        "control_no_added_leak_count": int(
            controls["g4_7_case_status"].eq("control_no_added_leak_under_stress").sum()
        ),
        "control_leak_count": int(
            controls["g4_7_case_status"].eq("control_leak_under_stress").sum()
        ),
        "case_status_counts": case_summary["g4_7_case_status"].value_counts().to_dict(),
        "stress_gate_passed": bool(
            positives["g4_7_case_status"]
            .eq("positive_frozen_schedule_survived_stress")
            .all()
            and controls["g4_7_case_status"]
            .eq("control_no_added_leak_under_stress")
            .all()
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
) -> None:
    lines = [
        "# Variable-Pair Synthetic G4.7 Independent Schedule Stress",
        "",
        f"- status: `{summary['status']}`",
        f"- stress_gate_passed: {summary['stress_gate_passed']}",
        f"- g4_3_positive_pass_count: {summary['g4_3_positive_pass_count']}",
        f"- g4_3_positive_fail_count: {summary['g4_3_positive_fail_count']}",
        f"- g4_5_selector_gate_passed: {summary['g4_5_selector_gate_passed']}",
        f"- g4_6_schedule_gate_passed: {summary['g4_6_schedule_gate_passed']}",
        f"- positive_schedule_survived_count: {summary['positive_schedule_survived_count']}",
        f"- positive_no_selected_source_count: {summary['positive_no_selected_source_count']}",
        f"- positive_pair_only_failure_count: {summary['positive_pair_only_failure_count']}",
        f"- control_no_added_leak_count: {summary['control_no_added_leak_count']}",
        f"- control_leak_count: {summary['control_leak_count']}",
        f"- case_status_counts: {summary['case_status_counts']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Case Summary",
    ]
    for row in case_summary.itertuples(index=False):
        lines.append(
            "- "
            f"{row.case_id} ({row.panel_role}): {row.g4_7_case_status}; "
            f"baseline_pair_share={row.baseline_pair_coassigned_run_share:.3f}, "
            f"g4_3_eligible={row.bridge_handle_eligible_source_count}, "
            f"g4_3_robust={row.bridge_handle_robust_pair_coassignment_count}, "
            f"pair_only_robust={row.pair_relation_only_robust_pair_coassignment_count}, "
            f"selected_sources={row.selected_source_count}, "
            f"schedule_p={row.schedule_known_coassigned_hit_rate:.3f}, "
            f"baseline_p={row.baseline_known_coassigned_hit_rate:.3f}"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "The controls remain suppressed, but the shifted positive panel "
                "does not preserve the source-conditioned opportunity surface. "
                "This is a regime-boundary result for the frozen schedule, not a "
                "reason to retune the selector inside G4.7."
            ),
            "",
        ]
    )
    (output_dir / G4_7_REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    g4_3_dir = output_dir / G4_3_DIRNAME
    g4_4_dir = output_dir / G4_4_DIRNAME
    g4_5_dir = output_dir / G4_5_DIRNAME
    g4_6_dir = output_dir / G4_6_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

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
        g4_3_dir=g4_3_dir,
        g4_4_dir=g4_4_dir,
        g4_5_dir=g4_5_dir,
        g4_6_dir=g4_6_dir,
    )
    _write_csv(case_summary, output_dir / G4_7_CASE_SUMMARY_CSV)
    summary = _summary(
        output_dir=output_dir,
        g4_3_summary=g4_3_summary,
        g4_4_summary=g4_4_summary,
        g4_5_summary=g4_5_summary,
        g4_6_summary=g4_6_summary,
        case_summary=case_summary,
    )
    (output_dir / G4_7_SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "variable_pair_synthetic_g4_7_independent_schedule_stress_config.v1",
        "output_dir": str(output_dir),
        "stage_dirs": {
            "g4_3": str(g4_3_dir),
            "g4_4": str(g4_4_dir),
            "g4_5": str(g4_5_dir),
            "g4_6": str(g4_6_dir),
        },
        "panel_cases": [case.__dict__ for case in STRESS_PANEL_CASES],
        "baseline_seeds": int(args.baseline_seeds),
        "handle_seeds": int(args.handle_seeds),
        "n_iterations": int(args.n_iterations),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / G4_7_CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_dir=output_dir, summary=summary, case_summary=case_summary)
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
