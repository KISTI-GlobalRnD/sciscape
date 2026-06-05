#!/usr/bin/env python3
"""Run an off-center failure-mode panel after G4.8I.

G4.8I shows that the frozen G4.8H target-free source-discovery rule can drive
schedule accounting on fresh centerline edge-mid contexts. This G4.8J runner
tests the first failure-mode expansion instead of replaying more success
cases: it keeps the same edge-mid direct/host contexts but shifts the
bridge-host value by ``-0.002`` and ``+0.002`` away from the centerline.

The predeclared expectation follows the G4.8E/G4.8I construction read:

- negative offset should become nonrobust coexistence (``N``): release sources
  may exist, but ready sources should be suppressed by source-nonneutral
  bridge release, so the discovered-source schedule should add no handle;
- positive offset should become target saturation (``T``): no source should be
  discovered and the baseline should already coassign the pair.

This is a synthetic failure-mode diagnostic only. It does not retune the
selector, promote walls/pathways, evaluate wall-clock quality/cost, replay
NanoClustering, or claim an algorithm-level method.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

import run_leiden_cpm_variable_pair_synthetic_g4_8i_discovered_source_schedule_panel as g4_8i
from analyze_leiden_cpm_variable_pair_synthetic_g4_8h_source_discovery_smoke import (
    DECISION_INPUT_COLUMNS,
    EVALUATION_ONLY_COLUMNS,
    SOURCE_DISCOVERY_RULE_ID,
)
from run_leiden_cpm_variable_pair_synthetic_demo import (
    BASE_RESULT_DIR,
    _json_safe,
    _write_csv,
)
from run_leiden_cpm_variable_pair_synthetic_g4_3_handle_generalization import (
    PanelCase,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_variable_pair_synthetic_g4_8j_off_center_failure_mode_panel_v1_20260604"
)

G4_3_DIRNAME = g4_8i.G4_3_DIRNAME
PANEL_DESIGN_CSV = "variable_pair_synthetic_g4_8j_panel_design.csv"
ENDPOINT_DISCOVERY_ROWS_CSV = (
    "variable_pair_synthetic_g4_8j_endpoint_discovery_rows.csv"
)
SCHEDULE_RUN_ROWS_CSV = "variable_pair_synthetic_g4_8j_schedule_run_rows.csv"
CASE_SUMMARY_CSV = "variable_pair_synthetic_g4_8j_case_summary.csv"
CONTEXT_SUMMARY_CSV = "variable_pair_synthetic_g4_8j_context_summary.csv"
OFFSET_SUMMARY_CSV = "variable_pair_synthetic_g4_8j_offset_summary.csv"
ROLE_SUMMARY_CSV = "variable_pair_synthetic_g4_8j_role_summary.csv"
SUMMARY_JSON = "variable_pair_synthetic_g4_8j_summary.json"
CONFIG_JSON = "variable_pair_synthetic_g4_8j_config.json"
REPORT_MD = "variable_pair_synthetic_g4_8j_report.md"

SCHEDULE_RULE_ID = "restart_then_g4_8h_discovered_source_then_one_g4_3_handle_v1"
HOST_CONTEXTS = g4_8i.HOST_CONTEXTS
PAIR_BRIDGE_VALUES = g4_8i.PAIR_BRIDGE_VALUES
BRIDGE_HOST_OFFSETS = (-0.002, 0.002)

CLAIM_BOUNDARY = (
    "Variable-pair synthetic G4.8J off-center failure-mode panel only; a fresh "
    "predeclared edge-mid direct/host panel shifts bridge-host support by "
    "-0.002 and +0.002 around the G4.8I centerline, then drives schedule "
    "accounting with the frozen G4.8H target-free source-discovery rule. No "
    "selector retuning, no oracle source-signature reads for schedule "
    "decisions, no wall or pathway promotion, no wall-clock quality/cost "
    "value, no NanoClustering replay, and no algorithm-level claims."
)
ROUTE_EXECUTION_STATUS = "executed_g4_8j_off_center_failure_mode_panel"
WALL_PROMOTION_STATUS = "not_promoted_off_center_failure_mode_only"
METHOD_STATUS = "off_center_failure_mode_panel_not_method_claim"


@dataclass(frozen=True)
class OffCenterCase:
    case_id: str
    context_id: str
    context_index: int
    centerline_index: int
    pair_bridge_weight: float
    centerline_bridge_host_weight: float
    bridge_host_offset: float
    bridge_host_weight: float
    direct_weight: float
    host_clique_weight: float
    expected_role_symbol: str

    def to_panel_case(self) -> PanelCase:
        return PanelCase(
            case_id=self.case_id,
            panel_role="positive_holdout",
            expected_gate=f"off_center_failure_expected_{self.expected_role_symbol}",
            direct_weight=self.direct_weight,
            pair_bridge_weight=self.pair_bridge_weight,
            bridge_host_weight=self.bridge_host_weight,
            host_clique_weight=self.host_clique_weight,
            note=(
                "Fresh off-center failure-mode panel for target-free "
                "discovered-source schedule accounting."
            ),
        )


def _expected_role(offset: float) -> str:
    return "N" if offset < 0.0 else "T"


def _offset_label(offset: float) -> str:
    return "m002" if offset < 0.0 else "p002"


def _case_id(
    *,
    context_id: str,
    pair_bridge: float,
    bridge_host: float,
    offset: float,
    expected_role: str,
) -> str:
    return (
        f"g4_8j_{context_id}_pair{int(round(pair_bridge * 1000)):04d}"
        f"_bridge{int(round(bridge_host * 1000)):04d}"
        f"_off{_offset_label(offset)}_exp{expected_role}"
    )


def _panel_cases() -> tuple[OffCenterCase, ...]:
    cases: list[OffCenterCase] = []
    for context_index, (context_id, direct_weight, host_clique_weight) in enumerate(
        HOST_CONTEXTS
    ):
        for centerline_index, pair_bridge in enumerate(PAIR_BRIDGE_VALUES):
            centerline_bridge = g4_8i._diagonal_bridge_host(pair_bridge)
            for offset in BRIDGE_HOST_OFFSETS:
                bridge_host = round(centerline_bridge + offset, 3)
                expected_role = _expected_role(offset)
                cases.append(
                    OffCenterCase(
                        case_id=_case_id(
                            context_id=context_id,
                            pair_bridge=pair_bridge,
                            bridge_host=bridge_host,
                            offset=offset,
                            expected_role=expected_role,
                        ),
                        context_id=context_id,
                        context_index=context_index,
                        centerline_index=centerline_index,
                        pair_bridge_weight=float(pair_bridge),
                        centerline_bridge_host_weight=float(centerline_bridge),
                        bridge_host_offset=float(offset),
                        bridge_host_weight=float(bridge_host),
                        direct_weight=float(direct_weight),
                        host_clique_weight=float(host_clique_weight),
                        expected_role_symbol=expected_role,
                    )
                )
    return tuple(cases)


PANEL_CASES = _panel_cases()


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


def _patch_g4_8i_globals() -> None:
    g4_8i.PANEL_CASES = PANEL_CASES
    g4_8i.CLAIM_BOUNDARY = CLAIM_BOUNDARY
    g4_8i.ROUTE_EXECUTION_STATUS = ROUTE_EXECUTION_STATUS
    g4_8i.WALL_PROMOTION_STATUS = WALL_PROMOTION_STATUS
    g4_8i.METHOD_STATUS = METHOD_STATUS
    g4_8i.SCHEDULE_RULE_ID = SCHEDULE_RULE_ID


def _panel_design_rows() -> pd.DataFrame:
    return _claim_columns(
        pd.DataFrame(
            [
                {
                    "case_id": case.case_id,
                    "context_id": case.context_id,
                    "context_index": case.context_index,
                    "centerline_index": case.centerline_index,
                    "expected_role_symbol": case.expected_role_symbol,
                    "pair_bridge_weight": case.pair_bridge_weight,
                    "centerline_bridge_host_weight": (
                        case.centerline_bridge_host_weight
                    ),
                    "bridge_host_offset": case.bridge_host_offset,
                    "bridge_host_weight": case.bridge_host_weight,
                    "direct_weight": case.direct_weight,
                    "host_clique_weight": case.host_clique_weight,
                    "panel_role": "positive_holdout",
                    "expected_gate": (
                        f"off_center_failure_expected_{case.expected_role_symbol}"
                    ),
                }
                for case in PANEL_CASES
            ]
        )
    )


def _case_summary(
    *,
    panel_design: pd.DataFrame,
    endpoint_discovery: pd.DataFrame,
    schedule_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows = g4_8i._case_summary(
        panel_design=panel_design,
        endpoint_discovery=endpoint_discovery,
        schedule_rows=schedule_rows,
    )
    design_cols = [
        "case_id",
        "centerline_bridge_host_weight",
        "bridge_host_offset",
    ]
    rows = rows.merge(panel_design[design_cols], on="case_id", how="left")
    rows["offset_side"] = rows["bridge_host_offset"].astype(float).map(
        lambda value: "negative_offset" if value < 0.0 else "positive_offset"
    )
    rows["failure_mode_expectation_status"] = [
        _failure_mode_expectation_status(row) for row in rows.to_dict("records")
    ]
    rows["failure_mode_expectation_passed"] = rows[
        "failure_mode_expectation_status"
    ].eq("failure_mode_expectation_passed")
    return _claim_columns(rows)


def _failure_mode_expectation_status(row: dict[str, Any]) -> str:
    if str(row["role_symbol"]) != str(row["expected_role_symbol"]):
        return "role_mismatch"
    role = str(row["role_symbol"])
    release_count = int(row["release_source_candidate_count"])
    ready_count = int(row["ready_source_candidate_count"])
    baseline = float(row["baseline_pair_coassigned_run_share"])
    schedule = float(row["schedule_known_coassigned_hit_rate"])
    if role == "N":
        if release_count == 4 and ready_count == 0 and abs(schedule - baseline) < 1e-12:
            return "failure_mode_expectation_passed"
        return "negative_offset_nonrobust_contract_failed"
    if role == "T":
        if release_count == 0 and ready_count == 0 and schedule >= 1.0 - 1e-12:
            return "failure_mode_expectation_passed"
        return "positive_offset_target_saturation_contract_failed"
    return "unexpected_role_for_off_center_panel"


def _group_summary(case_rows: pd.DataFrame, group_col: str, key_col: str) -> pd.DataFrame:
    rows = g4_8i._group_summary(case_rows, group_col, key_col)
    rows["failure_mode_expectation_pass_count"] = [
        int(
            case_rows[case_rows[group_col].astype(str).eq(str(row[key_col]))][
                "failure_mode_expectation_passed"
            ]
            .astype(bool)
            .sum()
        )
        for row in rows.to_dict("records")
    ]
    return _claim_columns(rows)


def _summary(
    *,
    output_dir: Path,
    g4_3_summary: dict[str, Any],
    endpoint_discovery: pd.DataFrame,
    schedule_rows: pd.DataFrame,
    case_rows: pd.DataFrame,
    context_summary: pd.DataFrame,
    offset_summary: pd.DataFrame,
    role_summary: pd.DataFrame,
) -> dict[str, Any]:
    case_count = int(len(case_rows))
    schedule_pass_count = int(
        case_rows["schedule_expectation_passed"].astype(bool).sum()
    )
    failure_pass_count = int(
        case_rows["failure_mode_expectation_passed"].astype(bool).sum()
    )
    role_match_count = int(
        case_rows["observed_role_matches_expected"].astype(bool).sum()
    )
    if (
        schedule_pass_count == case_count
        and failure_pass_count == case_count
        and role_match_count == case_count
    ):
        panel_status = "off_center_failure_mode_panel_passed"
    elif role_match_count == case_count:
        panel_status = "roles_passed_failure_contract_failed"
    else:
        panel_status = "off_center_failure_mode_panel_failed"
    return {
        "schema": "variable_pair_synthetic_g4_8j_off_center_failure_mode_panel_summary.v1",
        "status": ROUTE_EXECUTION_STATUS,
        "panel_status": panel_status,
        "output_dir": str(output_dir),
        "case_count": case_count,
        "endpoint_discovery_row_count": int(len(endpoint_discovery)),
        "schedule_run_row_count": int(len(schedule_rows)),
        "observed_role_match_count": role_match_count,
        "schedule_expectation_pass_count": schedule_pass_count,
        "failure_mode_expectation_pass_count": failure_pass_count,
        "failure_mode_status_counts": case_rows[
            "failure_mode_expectation_status"
        ].value_counts().to_dict(),
        "role_counts": case_rows["role_symbol"].value_counts().to_dict(),
        "release_source_candidate_count_by_role": case_rows.groupby("role_symbol")[
            "release_source_candidate_count"
        ].sum().astype(int).to_dict(),
        "ready_source_candidate_count_by_role": case_rows.groupby("role_symbol")[
            "ready_source_candidate_count"
        ].sum().astype(int).to_dict(),
        "schedule_hit_rate_median_by_role": case_rows.groupby("role_symbol")[
            "schedule_known_coassigned_hit_rate"
        ].median().to_dict(),
        "discovered_source_availability_median_by_role": case_rows.groupby(
            "role_symbol"
        )["discovered_source_availability_rate"].median().to_dict(),
        "restart_plus_handle_unit_median_by_role": case_rows.groupby("role_symbol")[
            "expected_restart_plus_handle_unit_per_restart"
        ].median().to_dict(),
        "g4_3_positive_pass_count": int(g4_3_summary.get("positive_pass_count", 0)),
        "context_summary_row_count": int(len(context_summary)),
        "offset_summary_row_count": int(len(offset_summary)),
        "role_summary_row_count": int(len(role_summary)),
        "source_discovery_rule_id": SOURCE_DISCOVERY_RULE_ID,
        "schedule_rule_id": SCHEDULE_RULE_ID,
        "recommended_next_gate": _recommended_next_gate(panel_status),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _recommended_next_gate(panel_status: str) -> str:
    if panel_status == "off_center_failure_mode_panel_passed":
        return (
            "Freeze the off-center failure contract and choose the next "
            "non-replay stress: wider offsets, context expansion, or a small "
            "real-data analog screen."
        )
    if panel_status == "roles_passed_failure_contract_failed":
        return (
            "Inspect schedule or source-discovery accounting mismatches before "
            "widening offsets."
        )
    return (
        "Do not expand; inspect which offset/context breaks the construction "
        "read before any further schedule panel."
    )


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    case_rows: pd.DataFrame,
    context_summary: pd.DataFrame,
    offset_summary: pd.DataFrame,
    role_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Variable-Pair Synthetic G4.8J Off-Center Failure-Mode Panel",
        "",
        f"- status: `{summary['status']}`",
        f"- panel_status: {summary['panel_status']}",
        f"- case_count: {summary['case_count']}",
        f"- observed_role_match_count: {summary['observed_role_match_count']}",
        f"- schedule_expectation_pass_count: {summary['schedule_expectation_pass_count']}",
        f"- failure_mode_expectation_pass_count: {summary['failure_mode_expectation_pass_count']}",
        f"- failure_mode_status_counts: {summary['failure_mode_status_counts']}",
        f"- source_discovery_rule_id: `{SOURCE_DISCOVERY_RULE_ID}`",
        f"- schedule_rule_id: `{SCHEDULE_RULE_ID}`",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Offset Summary",
        "",
    ]
    summary_cols = [
        "case_count",
        "failure_mode_expectation_pass_count",
        "schedule_expectation_pass_count",
        "observed_role_match_count",
        "release_source_candidate_count_sum",
        "ready_source_candidate_count_sum",
        "schedule_known_coassigned_hit_rate_median",
        "discovered_source_availability_rate_median",
        "expected_restart_plus_handle_unit_per_restart_median",
        "role_sequence",
        "expected_role_sequence",
        "status_counts",
    ]
    lines.extend(_markdown_table(offset_summary[["offset_side", *summary_cols]]))
    lines.extend(["", "## Context Summary", ""])
    lines.extend(_markdown_table(context_summary[["context_id", *summary_cols]]))
    lines.extend(["", "## Role Summary", ""])
    role_cols = [
        "role_symbol",
        "case_count",
        "failure_mode_expectation_pass_count",
        "schedule_expectation_pass_count",
        "release_source_candidate_count_sum",
        "ready_source_candidate_count_sum",
        "schedule_known_coassigned_hit_rate_median",
        "discovered_source_availability_rate_median",
        "expected_restart_plus_handle_unit_per_restart_median",
        "status_counts",
    ]
    lines.extend(_markdown_table(role_summary[role_cols]))
    lines.extend(["", "## Case Summary", ""])
    case_cols = [
        "context_id",
        "centerline_index",
        "bridge_host_offset",
        "expected_role_symbol",
        "role_symbol",
        "release_source_candidate_count",
        "ready_source_candidate_count",
        "baseline_pair_coassigned_run_share",
        "schedule_known_coassigned_hit_rate",
        "discovered_source_availability_rate",
        "failure_mode_expectation_status",
    ]
    lines.extend(_markdown_table(case_rows[case_cols]))
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "G4.8J is a synthetic off-center failure-mode diagnostic. It "
                "tests whether the discovered-source schedule correctly turns "
                "off at negative nonrobust offsets and no-ops at positive "
                "target-saturated offsets. It does not establish wall/pathway, "
                "quality/cost, NanoClustering, real-data source discovery, or "
                "algorithm-level method claims."
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
    _patch_g4_8i_globals()
    output_dir = Path(args.output_dir)
    g4_3_dir = output_dir / G4_3_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_design = _panel_design_rows()
    _write_csv(panel_design, output_dir / PANEL_DESIGN_CSV)
    g4_3_summary = g4_8i._run_g4_3_stage(
        output_dir=g4_3_dir,
        baseline_seeds=int(args.baseline_seeds),
        handle_seeds=int(args.handle_seeds),
        n_iterations=int(args.n_iterations),
    )
    endpoint_discovery = g4_8i._endpoint_discovery_rows(
        panel_design=panel_design,
        g4_3_dir=g4_3_dir,
    )
    schedule_rows = g4_8i._schedule_run_rows(
        g4_3_dir=g4_3_dir,
        endpoint_discovery=endpoint_discovery,
    )
    case_rows = _case_summary(
        panel_design=panel_design,
        endpoint_discovery=endpoint_discovery,
        schedule_rows=schedule_rows,
    )
    context_summary = _group_summary(case_rows, "context_id", "context_id")
    offset_summary = _group_summary(case_rows, "offset_side", "offset_side")
    role_summary = _group_summary(case_rows, "role_symbol", "role_symbol")
    _write_csv(endpoint_discovery, output_dir / ENDPOINT_DISCOVERY_ROWS_CSV)
    _write_csv(schedule_rows, output_dir / SCHEDULE_RUN_ROWS_CSV)
    _write_csv(case_rows, output_dir / CASE_SUMMARY_CSV)
    _write_csv(context_summary, output_dir / CONTEXT_SUMMARY_CSV)
    _write_csv(offset_summary, output_dir / OFFSET_SUMMARY_CSV)
    _write_csv(role_summary, output_dir / ROLE_SUMMARY_CSV)
    summary = _summary(
        output_dir=output_dir,
        g4_3_summary=g4_3_summary,
        endpoint_discovery=endpoint_discovery,
        schedule_rows=schedule_rows,
        case_rows=case_rows,
        context_summary=context_summary,
        offset_summary=offset_summary,
        role_summary=role_summary,
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "variable_pair_synthetic_g4_8j_off_center_failure_mode_panel_config.v1",
        "output_dir": str(output_dir),
        "g4_3_dir": str(g4_3_dir),
        "host_contexts": [
            {
                "context_id": context_id,
                "direct_weight": direct_weight,
                "host_clique_weight": host_clique_weight,
            }
            for context_id, direct_weight, host_clique_weight in HOST_CONTEXTS
        ],
        "pair_bridge_values": list(PAIR_BRIDGE_VALUES),
        "bridge_host_offsets": list(BRIDGE_HOST_OFFSETS),
        "negative_offset_expectation": "N/nonrobust coexistence with release sources but zero ready sources",
        "positive_offset_expectation": "T/target saturation with zero source candidates",
        "source_discovery_rule_id": SOURCE_DISCOVERY_RULE_ID,
        "schedule_rule_id": SCHEDULE_RULE_ID,
        "decision_input_columns": list(DECISION_INPUT_COLUMNS),
        "evaluation_only_columns": list(EVALUATION_ONLY_COLUMNS),
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
        case_rows=case_rows,
        context_summary=context_summary,
        offset_summary=offset_summary,
        role_summary=role_summary,
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
