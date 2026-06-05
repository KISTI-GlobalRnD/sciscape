#!/usr/bin/env python3
"""Validate the G4.8F signature split on fresh synthetic contexts.

G4.8F explains the G4.8E centerline roles as a source-signature split: ready
cells expose the full 8-source set, nonrobust cells expose only two-side
sources with source-nonneutral release, and target cells expose no source. This
G4.8G diagnostic freezes that construction-read hypothesis and tests it on
fresh direct/host contexts, while keeping the G4.3 handle, G4.5 selector, and
G4.6 schedule unchanged.

This is a fresh-context mechanism validation only. It is not selector retuning,
threshold search, source-discovery replacement, wall/pathway promotion,
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
    SUMMARY_JSON as G4_4_SUMMARY_JSON,
    analyze as analyze_g4_4,
)
from analyze_leiden_cpm_variable_pair_synthetic_g4_5_selector_suppression import (
    SELECTOR_SOURCE_ROWS_CSV as G4_5_SELECTOR_SOURCE_ROWS_CSV,
    SUMMARY_JSON as G4_5_SUMMARY_JSON,
    analyze as analyze_g4_5,
)
from analyze_leiden_cpm_variable_pair_synthetic_g4_6_schedule_accounting import (
    SOURCE_AVAILABILITY_ROWS_CSV as G4_6_SOURCE_AVAILABILITY_ROWS_CSV,
    SUMMARY_JSON as G4_6_SUMMARY_JSON,
    analyze as analyze_g4_6,
)
from analyze_leiden_cpm_variable_pair_synthetic_g4_8f_centerline_signature_audit import (
    _case_rows as _signature_case_rows,
    _endpoint_signature_rows,
    _role_summary,
    _signature_presence,
    _source_signature_rows,
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
from run_leiden_cpm_variable_pair_synthetic_g4_8e_diagonal_ridge_refinement import (
    _case_summary as _g4_8e_case_summary,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_variable_pair_synthetic_g4_8g_fresh_context_signature_validation_v1_20260603"
)

G4_3_DIRNAME = "g4_3_handle_probe"
G4_4_DIRNAME = "g4_4_restart_comparison"
G4_5_DIRNAME = "g4_5_selector_suppression"
G4_6_DIRNAME = "g4_6_schedule_accounting"
PANEL_DESIGN_CSV = "variable_pair_synthetic_g4_8g_panel_design.csv"
CASE_SUMMARY_CSV = "variable_pair_synthetic_g4_8g_case_summary.csv"
SIGNATURE_CASE_SUMMARY_CSV = (
    "variable_pair_synthetic_g4_8g_signature_case_summary.csv"
)
ENDPOINT_SIGNATURES_CSV = "variable_pair_synthetic_g4_8g_endpoint_signatures.csv"
SOURCE_SIGNATURES_CSV = "variable_pair_synthetic_g4_8g_source_signatures.csv"
CONTEXT_SUMMARY_CSV = "variable_pair_synthetic_g4_8g_context_summary.csv"
EXPECTED_ROLE_SUMMARY_CSV = (
    "variable_pair_synthetic_g4_8g_expected_role_summary.csv"
)
SIGNATURE_PRESENCE_MATRIX_CSV = (
    "variable_pair_synthetic_g4_8g_signature_presence_matrix.csv"
)
SUMMARY_JSON = "variable_pair_synthetic_g4_8g_summary.json"
CONFIG_JSON = "variable_pair_synthetic_g4_8g_config.json"
REPORT_MD = "variable_pair_synthetic_g4_8g_report.md"

HOST_CONTEXTS = (
    ("direct_low_host_low", 1.06, 1.23),
    ("direct_low_host_high", 1.06, 1.27),
    ("direct_high_host_low", 1.10, 1.23),
    ("direct_high_host_high", 1.10, 1.27),
)
PAIR_BRIDGE_VALUES = tuple(round(1.320 + step * 0.005, 3) for step in range(13))
DIAGONAL_START_PAIR = 1.32
DIAGONAL_START_BRIDGE = 1.44
DIAGONAL_SLOPE = 1.0 / 3.0
BRIDGE_HOST_OFFSET = 0.0

CLAIM_BOUNDARY = (
    "Variable-pair synthetic G4.8G fresh-context signature validation only; "
    "fresh direct/host contexts replay the frozen G4.3 handle, G4.5 selector, "
    "and G4.6 schedule to test the G4.8F construction-read hypothesis. No "
    "selector retuning, threshold search, source-discovery replacement, wall "
    "or pathway promotion, quality/cost value, NanoClustering replay, or "
    "algorithm-level claims."
)
ROUTE_EXECUTION_STATUS = "executed_g4_8g_fresh_context_signature_validation"
WALL_PROMOTION_STATUS = "not_promoted_fresh_context_signature_validation_only"
METHOD_STATUS = "fresh_context_signature_validation_not_algorithm_claim"


@dataclass(frozen=True)
class FreshContextCase:
    case_id: str
    context_id: str
    direct_weight: float
    host_clique_weight: float
    pair_bridge_weight: float
    diagonal_bridge_host_weight: float
    bridge_host_weight: float
    bridge_host_offset: float
    centerline_index: int
    context_index: int
    expected_role_symbol: str

    def to_panel_case(self) -> PanelCase:
        return PanelCase(
            case_id=self.case_id,
            panel_role="positive_holdout",
            expected_gate=f"fresh_context_signature_expected_{self.expected_role_symbol}",
            direct_weight=self.direct_weight,
            pair_bridge_weight=self.pair_bridge_weight,
            bridge_host_weight=self.bridge_host_weight,
            host_clique_weight=self.host_clique_weight,
            note=(
                "Fresh-context test of the frozen G4.8F centerline signature "
                "split under changed direct/host support."
            ),
        )


def _diagonal_bridge_host(pair_bridge: float) -> float:
    return round(
        DIAGONAL_START_BRIDGE
        + (float(pair_bridge) - DIAGONAL_START_PAIR) * DIAGONAL_SLOPE,
        3,
    )


def _expected_role(centerline_index: int) -> str:
    return ("R", "T", "N")[centerline_index % 3]


def _case_id(
    *,
    context_id: str,
    pair_bridge: float,
    bridge_host: float,
    expected_role: str,
) -> str:
    return (
        f"g4_8g_{context_id}_pair{int(round(pair_bridge * 1000)):04d}"
        f"_bridge{int(round(bridge_host * 1000)):04d}_exp{expected_role}"
    )


def _fresh_context_cases() -> tuple[FreshContextCase, ...]:
    cases: list[FreshContextCase] = []
    for context_index, (context_id, direct_weight, host_clique_weight) in enumerate(
        HOST_CONTEXTS
    ):
        for centerline_index, pair_bridge in enumerate(PAIR_BRIDGE_VALUES):
            bridge_host = _diagonal_bridge_host(pair_bridge)
            expected_role = _expected_role(centerline_index)
            cases.append(
                FreshContextCase(
                    case_id=_case_id(
                        context_id=context_id,
                        pair_bridge=pair_bridge,
                        bridge_host=bridge_host,
                        expected_role=expected_role,
                    ),
                    context_id=context_id,
                    direct_weight=float(direct_weight),
                    host_clique_weight=float(host_clique_weight),
                    pair_bridge_weight=float(pair_bridge),
                    diagonal_bridge_host_weight=float(bridge_host),
                    bridge_host_weight=float(bridge_host),
                    bridge_host_offset=BRIDGE_HOST_OFFSET,
                    centerline_index=centerline_index,
                    context_index=context_index,
                    expected_role_symbol=expected_role,
                )
            )
    return tuple(cases)


FRESH_CONTEXT_CASES = _fresh_context_cases()


def _claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.drop(
        columns=[
            "route_execution_status",
            "wall_promotion_status",
            "method_status",
            "claim_boundary",
        ],
        errors="ignore",
    ).copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _panel_design_rows() -> pd.DataFrame:
    rows = []
    for case in FRESH_CONTEXT_CASES:
        rows.append(
            {
                "case_id": case.case_id,
                "context_id": case.context_id,
                "context_index": case.context_index,
                "centerline_index": case.centerline_index,
                "expected_role_symbol": case.expected_role_symbol,
                "pair_bridge_weight": case.pair_bridge_weight,
                "diagonal_bridge_host_weight": case.diagonal_bridge_host_weight,
                "bridge_host_offset": case.bridge_host_offset,
                "bridge_host_weight": case.bridge_host_weight,
                "pair_bridge_index": case.centerline_index,
                "offset_index": 0,
                "direct_weight": case.direct_weight,
                "host_clique_weight": case.host_clique_weight,
                "panel_role": "positive_holdout",
                "expected_gate": (
                    f"fresh_context_signature_expected_{case.expected_role_symbol}"
                ),
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
    panel_cases = tuple(case.to_panel_case() for case in FRESH_CONTEXT_CASES)
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
        "schema": "variable_pair_synthetic_g4_8g_g4_3_config.v1",
        "output_dir": str(output_dir),
        "panel_cases": [case.__dict__ for case in FRESH_CONTEXT_CASES],
        "handle_policies": list(HANDLE_POLICIES),
        "baseline_seeds": int(baseline_seeds),
        "handle_seeds": int(handle_seeds),
        "n_iterations": int(n_iterations),
        "stage_claim_boundary": G4_3_CLAIM_BOUNDARY,
        "g4_8g_claim_boundary": CLAIM_BOUNDARY,
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
    rows = _g4_8e_case_summary(
        panel_design=panel_design,
        g4_3_dir=g4_3_dir,
        g4_4_dir=g4_4_dir,
        g4_5_dir=g4_5_dir,
        g4_6_dir=g4_6_dir,
    )
    design_cols = [
        "case_id",
        "context_id",
        "context_index",
        "centerline_index",
        "expected_role_symbol",
    ]
    rows = rows.merge(panel_design[design_cols], on="case_id", how="left")
    rows["observed_role_matches_expected"] = rows["role_symbol"].astype(str).eq(
        rows["expected_role_symbol"].astype(str)
    )
    return _claim_columns(rows)


def _signature_rows(
    *,
    case_summary: pd.DataFrame,
    g4_3_dir: Path,
    g4_5_dir: Path,
    g4_6_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    endpoint_summary = pd.read_csv(g4_3_dir / ENDPOINT_SUMMARY_CSV)
    handle_policy_summary = pd.read_csv(g4_3_dir / HANDLE_POLICY_SUMMARY_CSV)
    selector_source_rows = pd.read_csv(g4_5_dir / G4_5_SELECTOR_SOURCE_ROWS_CSV)
    source_availability_rows = pd.read_csv(
        g4_6_dir / G4_6_SOURCE_AVAILABILITY_ROWS_CSV
    )
    endpoint_rows = _claim_columns(
        _endpoint_signature_rows(
            centerline=case_summary,
            endpoint_summary=endpoint_summary,
        )
    )
    source_rows = _claim_columns(
        _source_signature_rows(
            endpoint_rows=endpoint_rows,
            handle_policy_summary=handle_policy_summary,
            selector_source_rows=selector_source_rows,
            source_availability_rows=source_availability_rows,
        )
    )
    signature_case_rows = _claim_columns(
        _signature_case_rows(
            centerline=case_summary,
            endpoint_rows=endpoint_rows,
            source_rows=source_rows,
        )
    )
    design_cols = [
        "case_id",
        "context_id",
        "context_index",
        "centerline_index",
        "expected_role_symbol",
    ]
    design = case_summary[design_cols].drop_duplicates()
    for frame in (endpoint_rows, source_rows, signature_case_rows):
        missing = [col for col in design_cols[1:] if col not in frame.columns]
        if missing:
            frame[missing] = None
    endpoint_rows = _claim_columns(
        endpoint_rows.drop(columns=design_cols[1:], errors="ignore").merge(
            design,
            on="case_id",
            how="left",
        )
    )
    source_rows = _claim_columns(
        source_rows.drop(columns=design_cols[1:], errors="ignore").merge(
            design,
            on="case_id",
            how="left",
        )
    )
    signature_case_rows = _claim_columns(
        signature_case_rows.drop(columns=design_cols[1:], errors="ignore").merge(
            design,
            on="case_id",
            how="left",
        )
    )
    signature_case_rows["observed_role_matches_expected"] = signature_case_rows[
        "role_symbol"
    ].astype(str).eq(signature_case_rows["expected_role_symbol"].astype(str))
    signature_case_rows["signature_expectation_status"] = [
        _signature_expectation_status(row)
        for row in signature_case_rows.to_dict("records")
    ]
    signature_case_rows["signature_expectation_passed"] = signature_case_rows[
        "signature_expectation_status"
    ].eq("signature_expectation_passed")
    presence = _claim_columns(_signature_presence(endpoint_rows, source_rows))
    return endpoint_rows, source_rows, signature_case_rows, presence


def _signature_expectation_status(row: dict[str, Any]) -> str:
    expected = str(row["expected_role_symbol"])
    observed = str(row["role_symbol"])
    if observed != expected:
        return "role_mismatch"
    if expected == "R":
        if (
            int(row["source_signature_count"]) == 8
            and int(row["single_side_release_source_count"]) == 4
            and int(row["two_side_release_source_count"]) == 4
            and int(row["source_neutral_count"]) == 8
            and int(row["selected_source_count"]) == 8
            and int(row["robust_bridge_release_source_count"]) == 8
            and float(row["handle_known_hit_rate_median"]) >= 1.0
        ):
            return "signature_expectation_passed"
        return "ready_signature_split_failed"
    if expected == "N":
        if (
            int(row["source_signature_count"]) == 4
            and int(row["single_side_release_source_count"]) == 0
            and int(row["two_side_release_source_count"]) == 4
            and int(row["source_neutral_count"]) == 0
            and int(row["selected_source_count"]) == 0
            and int(row["robust_bridge_release_source_count"]) == 0
            and float(row["handle_known_hit_rate_median"]) < 1.0
            and float(row["initial_quality_delta_median"]) < 0.0
        ):
            return "signature_expectation_passed"
        return "nonrobust_signature_split_failed"
    if expected == "T":
        if (
            float(row["baseline_pair_coassigned_run_share"]) >= 1.0
            and int(row["source_signature_count"]) == 0
            and int(row["separated_endpoint_count"]) == 0
        ):
            return "signature_expectation_passed"
        return "target_signature_split_failed"
    return "unknown_expected_role"


def _context_summary(signature_case_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for context_id, group in signature_case_rows.groupby("context_id", sort=True):
        rows.append(_summary_row("context_id", str(context_id), group))
    return _claim_columns(pd.DataFrame(rows))


def _expected_role_summary(signature_case_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for role, group in signature_case_rows.groupby("expected_role_symbol", sort=True):
        rows.append(_summary_row("expected_role_symbol", str(role), group))
    return _claim_columns(pd.DataFrame(rows))


def _summary_row(group_col: str, key: str, group: pd.DataFrame) -> dict[str, Any]:
    return {
        group_col: key,
        "case_count": int(len(group)),
        "observed_role_match_count": int(
            group["observed_role_matches_expected"].astype(bool).sum()
        ),
        "signature_expectation_pass_count": int(
            group["signature_expectation_passed"].astype(bool).sum()
        ),
        "observed_role_sequence": "".join(group.sort_values("centerline_index")[
            "role_symbol"
        ].astype(str)),
        "expected_role_sequence": "".join(group.sort_values("centerline_index")[
            "expected_role_symbol"
        ].astype(str)),
        "signature_status_counts": json.dumps(
            group["signature_expectation_status"].value_counts().to_dict(),
            sort_keys=True,
        ),
        "case_ids": ";".join(sorted(group["case_id"].astype(str))),
    }


def _summary(
    *,
    output_dir: Path,
    g4_3_summary: dict[str, Any],
    g4_5_summary: dict[str, Any],
    g4_6_summary: dict[str, Any],
    case_summary: pd.DataFrame,
    signature_case_rows: pd.DataFrame,
    context_summary: pd.DataFrame,
    expected_role_summary: pd.DataFrame,
) -> dict[str, Any]:
    case_count = int(len(signature_case_rows))
    role_match_count = int(
        signature_case_rows["observed_role_matches_expected"].astype(bool).sum()
    )
    signature_pass_count = int(
        signature_case_rows["signature_expectation_passed"].astype(bool).sum()
    )
    if signature_pass_count == case_count and role_match_count == case_count:
        validation_status = "fresh_context_signature_split_validated"
    elif role_match_count == case_count:
        validation_status = "roles_validated_signature_split_failed"
    else:
        validation_status = "fresh_context_signature_split_not_validated"
    return {
        "schema": "variable_pair_synthetic_g4_8g_fresh_context_signature_validation_summary.v1",
        "status": ROUTE_EXECUTION_STATUS,
        "validation_status": validation_status,
        "output_dir": str(output_dir),
        "case_count": case_count,
        "context_count": int(signature_case_rows["context_id"].nunique()),
        "fresh_contexts": [context[0] for context in HOST_CONTEXTS],
        "centerline_case_count_per_context": int(len(PAIR_BRIDGE_VALUES)),
        "observed_role_match_count": role_match_count,
        "signature_expectation_pass_count": signature_pass_count,
        "signature_status_counts": signature_case_rows[
            "signature_expectation_status"
        ].value_counts().to_dict(),
        "observed_role_counts": signature_case_rows["role_symbol"].value_counts().to_dict(),
        "expected_role_counts": signature_case_rows[
            "expected_role_symbol"
        ].value_counts().to_dict(),
        "cartography_status_counts": case_summary[
            "cartography_status"
        ].value_counts().to_dict(),
        "context_summary_row_count": int(len(context_summary)),
        "expected_role_summary_row_count": int(len(expected_role_summary)),
        "g4_3_positive_pass_count": int(g4_3_summary.get("positive_pass_count", 0)),
        "g4_5_selector_gate_passed": bool(g4_5_summary.get("selector_gate_passed", False)),
        "g4_6_schedule_gate_passed": bool(g4_6_summary.get("schedule_gate_passed", False)),
        "recommended_next_gate": _recommended_next_gate(validation_status),
        "stage_summary_paths": {
            "g4_3": str(output_dir / G4_3_DIRNAME / G4_3_SUMMARY_JSON),
            "g4_4": str(output_dir / G4_4_DIRNAME / G4_4_SUMMARY_JSON),
            "g4_5": str(output_dir / G4_5_DIRNAME / G4_5_SUMMARY_JSON),
            "g4_6": str(output_dir / G4_6_DIRNAME / G4_6_SUMMARY_JSON),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _recommended_next_gate(validation_status: str) -> str:
    if validation_status == "fresh_context_signature_split_validated":
        return (
            "Freeze the construction-read rule and design a bounded "
            "source-discovery replacement smoke that must recover the full "
            "8-source ready signature set without reading target outcomes."
        )
    if validation_status == "roles_validated_signature_split_failed":
        return (
            "Keep role cartography but inspect failed signature rows before "
            "source-discovery replacement."
        )
    return (
        "Do not proceed to source discovery; redesign the construction-read "
        "hypothesis or fresh-context panel."
    )


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    signature_case_rows: pd.DataFrame,
    context_summary: pd.DataFrame,
    expected_role_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Variable-Pair Synthetic G4.8G Fresh-Context Signature Validation",
        "",
        f"- status: `{summary['status']}`",
        f"- validation_status: {summary['validation_status']}",
        f"- case_count: {summary['case_count']}",
        f"- observed_role_match_count: {summary['observed_role_match_count']}",
        f"- signature_expectation_pass_count: {summary['signature_expectation_pass_count']}",
        f"- signature_status_counts: {summary['signature_status_counts']}",
        f"- observed_role_counts: {summary['observed_role_counts']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Context Summary",
        "",
    ]
    lines.extend(
        _markdown_table(
            context_summary[
                [
                    "context_id",
                    "case_count",
                    "observed_role_match_count",
                    "signature_expectation_pass_count",
                    "observed_role_sequence",
                    "expected_role_sequence",
                    "signature_status_counts",
                ]
            ]
        )
    )
    lines.extend(["", "## Expected Role Summary", ""])
    lines.extend(
        _markdown_table(
            expected_role_summary[
                [
                    "expected_role_symbol",
                    "case_count",
                    "observed_role_match_count",
                    "signature_expectation_pass_count",
                    "observed_role_sequence",
                    "expected_role_sequence",
                    "signature_status_counts",
                ]
            ]
        )
    )
    lines.extend(["", "## Signature Case Summary", ""])
    display_cols = [
        "context_id",
        "pair_bridge_weight",
        "bridge_host_weight",
        "expected_role_symbol",
        "role_symbol",
        "signature_expectation_status",
        "baseline_pair_coassigned_run_share",
        "source_signature_count",
        "single_side_release_source_count",
        "source_neutral_count",
        "selected_source_count",
        "robust_bridge_release_source_count",
        "handle_known_hit_rate_median",
        "initial_quality_delta_median",
    ]
    lines.extend(_markdown_table(signature_case_rows[display_cols]))
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "G4.8G validates a construction-read hypothesis on fresh "
                "synthetic contexts. It still does not discover sources, prove "
                "walls/pathways, compare quality/cost, replay NanoClustering, "
                "or make a method claim."
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
        lines.append("| " + " | ".join(_format_cell(row[col]) for col in cols) + " |")
    return lines


def _format_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


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
    analyze_g4_4(Namespace(g4_3_dir=g4_3_dir, output_dir=g4_4_dir))
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
    endpoint_rows, source_rows, signature_case_rows, presence = _signature_rows(
        case_summary=case_summary,
        g4_3_dir=g4_3_dir,
        g4_5_dir=g4_5_dir,
        g4_6_dir=g4_6_dir,
    )
    context_summary = _context_summary(signature_case_rows)
    expected_role_summary = _expected_role_summary(signature_case_rows)
    _write_csv(case_summary, output_dir / CASE_SUMMARY_CSV)
    _write_csv(signature_case_rows, output_dir / SIGNATURE_CASE_SUMMARY_CSV)
    _write_csv(endpoint_rows, output_dir / ENDPOINT_SIGNATURES_CSV)
    _write_csv(source_rows, output_dir / SOURCE_SIGNATURES_CSV)
    _write_csv(context_summary, output_dir / CONTEXT_SUMMARY_CSV)
    _write_csv(expected_role_summary, output_dir / EXPECTED_ROLE_SUMMARY_CSV)
    _write_csv(presence, output_dir / SIGNATURE_PRESENCE_MATRIX_CSV)
    summary = _summary(
        output_dir=output_dir,
        g4_3_summary=g4_3_summary,
        g4_5_summary=g4_5_summary,
        g4_6_summary=g4_6_summary,
        case_summary=case_summary,
        signature_case_rows=signature_case_rows,
        context_summary=context_summary,
        expected_role_summary=expected_role_summary,
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "variable_pair_synthetic_g4_8g_fresh_context_signature_validation_config.v1",
        "output_dir": str(output_dir),
        "stage_dirs": {
            "g4_3": str(g4_3_dir),
            "g4_4": str(g4_4_dir),
            "g4_5": str(g4_5_dir),
            "g4_6": str(g4_6_dir),
        },
        "host_contexts": [
            {
                "context_id": context_id,
                "direct_weight": direct_weight,
                "host_clique_weight": host_clique_weight,
            }
            for context_id, direct_weight, host_clique_weight in HOST_CONTEXTS
        ],
        "pair_bridge_values": list(PAIR_BRIDGE_VALUES),
        "diagonal_rule": (
            "bridge_host = 1.44 + (pair_bridge - 1.32) / 3, rounded to 0.001"
        ),
        "expected_role_pattern": "R,T,N repeated from pair_bridge=1.320",
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
        signature_case_rows=signature_case_rows,
        context_summary=context_summary,
        expected_role_summary=expected_role_summary,
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
