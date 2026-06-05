#!/usr/bin/env python3
"""Account for a minimal schedule using the frozen G4.5 selector.

This G4.6 diagnostic keeps the G4.3 bridge-release handle and the G4.5
``neutral_release_with_direct_support_v1`` selector frozen. It asks a narrower
question than a full method claim:

1. ordinary Leiden+CPM produces an endpoint;
2. if the endpoint is already a known coassigned target, the schedule succeeds;
3. otherwise, if the endpoint is selected by the frozen G4.5 source-local
   selector, apply the frozen bridge-release handle once;
4. otherwise, no-op.

The accounting includes source availability in the ordinary restart pool and a
simple restart-plus-handle unit cost. It does not include wall identification,
source discovery beyond the observed restart pool, wall-clock timing, full
NanoClustering replay, quality/cost value, or algorithm-level claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from analyze_leiden_cpm_variable_pair_synthetic_g4_4_restart_comparison import (
    DEFAULT_OUTPUT_DIR as DEFAULT_G4_4_DIR,
    TARGET_TABLE_CSV,
    _expected_runs,
    _geometric_quantile,
    _hit_probability,
)
from analyze_leiden_cpm_variable_pair_synthetic_g4_5_selector_suppression import (
    DEFAULT_OUTPUT_DIR as DEFAULT_G4_5_DIR,
    SELECTOR_SOURCE_ROWS_CSV,
)
from run_leiden_cpm_variable_pair_synthetic_demo import (
    BASE_RESULT_DIR,
    _json_safe,
    _write_csv,
)
from run_leiden_cpm_variable_pair_synthetic_g4_3_handle_generalization import (
    BASELINE_RUNS_CSV,
    DEFAULT_OUTPUT_DIR as DEFAULT_G4_3_DIR,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_variable_pair_synthetic_g4_6_schedule_accounting_v1_20260603"
)

SCHEDULE_RUN_ROWS_CSV = "variable_pair_synthetic_g4_6_schedule_run_rows.csv"
SOURCE_AVAILABILITY_ROWS_CSV = (
    "variable_pair_synthetic_g4_6_source_availability_rows.csv"
)
SCHEDULE_CASE_SUMMARY_CSV = "variable_pair_synthetic_g4_6_schedule_case_summary.csv"
SCHEDULE_BUDGET_CURVE_ROWS_CSV = (
    "variable_pair_synthetic_g4_6_schedule_budget_curve_rows.csv"
)
SUMMARY_JSON = "variable_pair_synthetic_g4_6_summary.json"
CONFIG_JSON = "variable_pair_synthetic_g4_6_config.json"
REPORT_MD = "variable_pair_synthetic_g4_6_report.md"

BUDGETS = (1, 2, 3, 4, 5, 8, 10, 16)
HANDLE_UNIT_COST = 1.0
RESTART_UNIT_COST = 1.0
SELECTOR_EVALS_PER_RESTART = 1.0
EPS = 1.0e-12

CLAIM_BOUNDARY = (
    "Variable-pair synthetic G4.6 minimal schedule-accounting diagnostic only; "
    "the frozen G4.5 selector and frozen G4.3 bridge-release handle are applied "
    "to endpoints available in the observed ordinary Leiden+CPM restart pool. "
    "The accounting includes source availability and restart-plus-handle unit "
    "cost, but not wall-clock timing, wall identification, source discovery "
    "beyond the observed pool, full NanoClustering replay, quality/cost value, "
    "or algorithm-level claims."
)
ROUTE_EXECUTION_STATUS = "executed_g4_6_minimal_schedule_accounting"
WALL_PROMOTION_STATUS = "not_promoted_schedule_accounting_only"
METHOD_STATUS = "minimal_schedule_accounting_not_algorithm_claim"
SCHEDULE_RULE_ID = "restart_then_g4_5_selector_then_one_g4_3_handle_v1"


def _claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _target_signatures(value: Any) -> set[str]:
    text = "" if pd.isna(value) else str(value)
    return {item for item in text.split(";") if item}


def _selected_source_lookup(selector_rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    selected = selector_rows[selector_rows["selector_selected"].astype(bool)].copy()
    return {
        str(row["source_endpoint_signature_id"]): row
        for row in selected.to_dict("records")
    }


def _schedule_run_rows(
    *,
    baseline_runs: pd.DataFrame,
    target_table: pd.DataFrame,
    selector_rows: pd.DataFrame,
) -> pd.DataFrame:
    target_lookup = {
        str(row["case_id"]): _target_signatures(row["known_coassigned_target_signatures"])
        for row in target_table.to_dict("records")
    }
    selected_by_case = {
        str(case_id): _selected_source_lookup(group)
        for case_id, group in selector_rows.groupby("case_id", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for source in baseline_runs.to_dict("records"):
        case_id = str(source["case_id"])
        endpoint_id = str(source["endpoint_signature_id"])
        target_signatures = target_lookup.get(case_id, set())
        selected_sources = selected_by_case.get(case_id, {})
        target_hit = endpoint_id in target_signatures
        selected_row = selected_sources.get(endpoint_id)
        selector_selected = selected_row is not None
        handle_rate = (
            float(selected_row["handle_known_coassigned_hit_rate"])
            if selected_row is not None
            else 0.0
        )
        handle_applied = bool(selector_selected and not target_hit)
        if target_hit:
            schedule_hit_probability = 1.0
            schedule_run_status = "target_hit_without_handle"
        elif handle_applied:
            schedule_hit_probability = handle_rate
            schedule_run_status = "selected_source_handle_applied"
        else:
            schedule_hit_probability = 0.0
            schedule_run_status = "no_selected_source_noop"
        rows.append(
            {
                "case_id": case_id,
                "panel_role": str(source["panel_role"]),
                "expected_gate": str(source["expected_gate"]),
                "start_condition": str(source["start_condition"]),
                "seed": int(source["seed"]),
                "endpoint_signature_id": endpoint_id,
                "baseline_target_hit": bool(target_hit),
                "selector_selected_source": bool(selector_selected),
                "handle_applied": bool(handle_applied),
                "selected_handle_known_hit_rate": float(handle_rate),
                "schedule_hit_probability": float(schedule_hit_probability),
                "schedule_run_status": schedule_run_status,
                "schedule_rule_id": SCHEDULE_RULE_ID,
                "restart_unit_cost": RESTART_UNIT_COST,
                "selector_eval_count": SELECTOR_EVALS_PER_RESTART,
                "expected_handle_unit_cost": float(HANDLE_UNIT_COST if handle_applied else 0.0),
                "expected_restart_plus_handle_unit_cost": float(
                    RESTART_UNIT_COST + (HANDLE_UNIT_COST if handle_applied else 0.0)
                ),
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _source_availability_rows(
    *,
    baseline_runs: pd.DataFrame,
    selector_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    run_counts = baseline_runs.groupby("case_id", sort=False).size().to_dict()
    endpoint_counts = (
        baseline_runs.groupby(["case_id", "endpoint_signature_id"], sort=False)
        .size()
        .to_dict()
    )
    for source in selector_rows.to_dict("records"):
        case_id = str(source["case_id"])
        endpoint_id = str(source["source_endpoint_signature_id"])
        run_count = int(run_counts.get(case_id, 0))
        observed_count = int(endpoint_counts.get((case_id, endpoint_id), 0))
        rows.append(
            {
                "case_id": case_id,
                "panel_role": str(source["panel_role"]),
                "expected_gate": str(source["expected_gate"]),
                "source_endpoint_signature_id": endpoint_id,
                "selector_selected": bool(source["selector_selected"]),
                "g4_5_selector_status": str(source["g4_5_selector_status"]),
                "selector_suppression_reason": str(source["selector_suppression_reason"]),
                "baseline_run_count": run_count,
                "source_observed_count": observed_count,
                "source_availability_rate": float(observed_count / run_count)
                if run_count
                else 0.0,
                "handle_known_coassigned_hit_rate": float(
                    source["handle_known_coassigned_hit_rate"]
                ),
                "selected_schedule_contribution_rate": float(
                    observed_count / run_count * float(source["handle_known_coassigned_hit_rate"])
                )
                if run_count and bool(source["selector_selected"])
                else 0.0,
                "schedule_rule_id": SCHEDULE_RULE_ID,
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _case_status(row: dict[str, Any]) -> str:
    positive = str(row["panel_role"]) == "positive_holdout"
    schedule_rate = float(row["schedule_known_coassigned_hit_rate"])
    baseline_rate = float(row["baseline_known_coassigned_hit_rate"])
    cost_ratio = row["baseline_over_schedule_restart_plus_handle_unit_ratio"]
    if positive:
        if schedule_rate > baseline_rate + EPS and cost_ratio is not None and cost_ratio > 1.0:
            return "positive_schedule_beats_restart_with_source_accounting"
        return "positive_schedule_does_not_beat_restart_with_accounting"
    if schedule_rate <= baseline_rate + EPS and int(row["selected_source_count"]) == 0:
        return "control_schedule_no_added_leak"
    return "control_schedule_added_leak_or_selected_source"


def _case_summary(
    *,
    schedule_rows: pd.DataFrame,
    source_availability: pd.DataFrame,
    target_table: pd.DataFrame,
) -> pd.DataFrame:
    target_lookup = {
        str(row["case_id"]): row
        for row in target_table.to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    for case_id, group in schedule_rows.groupby("case_id", sort=True):
        source_group = source_availability[
            source_availability["case_id"].astype(str).eq(str(case_id))
        ]
        selected_sources = source_group[source_group["selector_selected"].astype(bool)]
        target = target_lookup[str(case_id)]
        baseline_rate = float(target["baseline_known_coassigned_hit_rate"])
        schedule_rate = float(group["schedule_hit_probability"].mean())
        source_discovery_rate = float(group["selector_selected_source"].astype(bool).mean())
        handle_application_rate = float(group["handle_applied"].astype(bool).mean())
        schedule_expected = _expected_runs(schedule_rate)
        baseline_expected = _expected_runs(baseline_rate)
        expected_units_per_cycle = float(
            RESTART_UNIT_COST + HANDLE_UNIT_COST * handle_application_rate
        )
        schedule_units = (
            float(schedule_expected * expected_units_per_cycle)
            if schedule_expected is not None
            else None
        )
        baseline_units = (
            float(baseline_expected * RESTART_UNIT_COST)
            if baseline_expected is not None
            else None
        )
        cost_ratio = (
            float(baseline_units / schedule_units)
            if baseline_units is not None and schedule_units is not None and schedule_units > 0.0
            else None
        )
        row = {
            "case_id": str(case_id),
            "panel_role": str(group["panel_role"].iloc[0]),
            "expected_gate": str(group["expected_gate"].iloc[0]),
            "baseline_run_count": int(len(group)),
            "baseline_known_coassigned_hit_rate": baseline_rate,
            "selected_source_count": int(len(selected_sources)),
            "selected_source_discovery_rate": source_discovery_rate,
            "handle_application_rate": handle_application_rate,
            "schedule_known_coassigned_hit_rate": schedule_rate,
            "schedule_probability_lift_vs_baseline": float(schedule_rate - baseline_rate),
            "baseline_expected_restarts_to_known_coassigned": baseline_expected,
            "schedule_expected_cycles_to_known_coassigned": schedule_expected,
            "expected_restart_plus_handle_units_per_cycle": expected_units_per_cycle,
            "baseline_expected_restart_units_to_known_coassigned": baseline_units,
            "schedule_expected_restart_plus_handle_units_to_known_coassigned": schedule_units,
            "baseline_over_schedule_restart_plus_handle_unit_ratio": cost_ratio,
            "schedule_expected_selector_evals_to_known_coassigned": (
                float(schedule_expected * SELECTOR_EVALS_PER_RESTART)
                if schedule_expected is not None
                else None
            ),
            "schedule_expected_handle_units_to_known_coassigned": (
                float(schedule_expected * handle_application_rate * HANDLE_UNIT_COST)
                if schedule_expected is not None
                else None
            ),
            "baseline_p50_cycles_to_known_coassigned": _geometric_quantile(
                baseline_rate,
                0.50,
            ),
            "schedule_p50_cycles_to_known_coassigned": _geometric_quantile(
                schedule_rate,
                0.50,
            ),
            "schedule_status_counts": json.dumps(
                group["schedule_run_status"].value_counts().to_dict(),
                sort_keys=True,
            ),
            "selected_source_availability_counts": json.dumps(
                {
                    str(row["source_endpoint_signature_id"]): int(row["source_observed_count"])
                    for row in selected_sources.to_dict("records")
                },
                sort_keys=True,
            ),
            "schedule_rule_id": SCHEDULE_RULE_ID,
        }
        row["g4_6_case_status"] = _case_status(row)
        rows.append(row)
    return _claim_columns(pd.DataFrame(rows))


def _budget_curve_rows(case_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in case_summary.to_dict("records"):
        baseline_rate = float(case["baseline_known_coassigned_hit_rate"])
        schedule_rate = float(case["schedule_known_coassigned_hit_rate"])
        handle_application_rate = float(case["handle_application_rate"])
        expected_units_per_cycle = float(case["expected_restart_plus_handle_units_per_cycle"])
        for budget in BUDGETS:
            rows.append(
                {
                    "case_id": str(case["case_id"]),
                    "panel_role": str(case["panel_role"]),
                    "expected_gate": str(case["expected_gate"]),
                    "budget_cycles": int(budget),
                    "baseline_hit_probability": _hit_probability(
                        baseline_rate,
                        int(budget),
                    ),
                    "schedule_hit_probability": _hit_probability(
                        schedule_rate,
                        int(budget),
                    ),
                    "expected_selector_evals": float(
                        int(budget) * SELECTOR_EVALS_PER_RESTART
                    ),
                    "expected_handle_units": float(
                        int(budget) * handle_application_rate * HANDLE_UNIT_COST
                    ),
                    "expected_restart_plus_handle_units": float(
                        int(budget) * expected_units_per_cycle
                    ),
                    "schedule_rule_id": SCHEDULE_RULE_ID,
                }
            )
    return _claim_columns(pd.DataFrame(rows))


def _summary(
    *,
    g4_3_dir: Path,
    g4_4_dir: Path,
    g4_5_dir: Path,
    output_dir: Path,
    case_summary: pd.DataFrame,
    source_availability: pd.DataFrame,
    schedule_rows: pd.DataFrame,
) -> dict[str, Any]:
    positives = case_summary[case_summary["panel_role"].eq("positive_holdout")]
    controls = case_summary[~case_summary["panel_role"].eq("positive_holdout")]
    positive_ratios = positives[
        "baseline_over_schedule_restart_plus_handle_unit_ratio"
    ].dropna()
    control_lifts = controls["schedule_probability_lift_vs_baseline"]
    return {
        "schema": "variable_pair_synthetic_g4_6_schedule_accounting_summary.v1",
        "status": ROUTE_EXECUTION_STATUS,
        "g4_3_dir": str(g4_3_dir),
        "g4_4_dir": str(g4_4_dir),
        "g4_5_dir": str(g4_5_dir),
        "output_dir": str(output_dir),
        "case_count": int(len(case_summary)),
        "baseline_run_row_count": int(len(schedule_rows)),
        "source_availability_row_count": int(len(source_availability)),
        "selected_source_count": int(
            source_availability["selector_selected"].astype(bool).sum()
        ),
        "handle_application_row_count": int(schedule_rows["handle_applied"].astype(bool).sum()),
        "positive_case_count": int(len(positives)),
        "control_case_count": int(len(controls)),
        "positive_schedule_pass_count": int(
            positives["g4_6_case_status"]
            .eq("positive_schedule_beats_restart_with_source_accounting")
            .sum()
        ),
        "control_no_added_leak_count": int(
            controls["g4_6_case_status"].eq("control_schedule_no_added_leak").sum()
        ),
        "case_status_counts": case_summary["g4_6_case_status"].value_counts().to_dict(),
        "positive_schedule_hit_rate_min": float(
            positives["schedule_known_coassigned_hit_rate"].min()
        )
        if not positives.empty
        else 0.0,
        "positive_selected_source_discovery_rate_min": float(
            positives["selected_source_discovery_rate"].min()
        )
        if not positives.empty
        else 0.0,
        "positive_cost_adjusted_unit_ratio_min": float(positive_ratios.min())
        if not positive_ratios.empty
        else None,
        "positive_cost_adjusted_unit_ratio_median": float(positive_ratios.median())
        if not positive_ratios.empty
        else None,
        "control_schedule_probability_lift_max": float(control_lifts.max())
        if not control_lifts.empty
        else 0.0,
        "schedule_gate_passed": bool(
            len(positives) > 0
            and len(controls) > 0
            and positives["g4_6_case_status"]
            .eq("positive_schedule_beats_restart_with_source_accounting")
            .all()
            and controls["g4_6_case_status"].eq("control_schedule_no_added_leak").all()
        ),
        "schedule_rule": {
            "schedule_rule_id": SCHEDULE_RULE_ID,
            "restart_unit_cost": RESTART_UNIT_COST,
            "handle_unit_cost": HANDLE_UNIT_COST,
            "selector_evals_per_restart": SELECTOR_EVALS_PER_RESTART,
            "target_hit_short_circuits_handle": True,
            "selected_source_handle_runs_per_cycle": "empirical_selected_source_rate",
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _format_float(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.6g}"


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    case_summary: pd.DataFrame,
    source_availability: pd.DataFrame,
) -> None:
    lines = [
        "# Variable-Pair Synthetic G4.6 Schedule Accounting",
        "",
        f"- status: `{summary['status']}`",
        f"- schedule_gate_passed: {summary['schedule_gate_passed']}",
        f"- positive_schedule_pass_count: {summary['positive_schedule_pass_count']}",
        f"- control_no_added_leak_count: {summary['control_no_added_leak_count']}",
        f"- positive_schedule_hit_rate_min: {summary['positive_schedule_hit_rate_min']}",
        (
            "- positive_selected_source_discovery_rate_min: "
            f"{summary['positive_selected_source_discovery_rate_min']}"
        ),
        (
            "- positive_cost_adjusted_unit_ratio_min: "
            f"{summary['positive_cost_adjusted_unit_ratio_min']}"
        ),
        (
            "- positive_cost_adjusted_unit_ratio_median: "
            f"{summary['positive_cost_adjusted_unit_ratio_median']}"
        ),
        (
            "- control_schedule_probability_lift_max: "
            f"{summary['control_schedule_probability_lift_max']}"
        ),
        f"- case_status_counts: {summary['case_status_counts']}",
        f"- schedule_rule: {summary['schedule_rule']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Case Summary",
    ]
    for row in case_summary.itertuples(index=False):
        lines.append(
            "- "
            f"{row.case_id} ({row.panel_role}): {row.g4_6_case_status}; "
            f"baseline_p={row.baseline_known_coassigned_hit_rate:.3f}, "
            f"source_p={row.selected_source_discovery_rate:.3f}, "
            f"schedule_p={row.schedule_known_coassigned_hit_rate:.3f}, "
            "expected_units baseline/schedule="
            f"{_format_float(row.baseline_expected_restart_units_to_known_coassigned)}/"
            f"{_format_float(row.schedule_expected_restart_plus_handle_units_to_known_coassigned)}, "
            "unit_ratio="
            f"{_format_float(row.baseline_over_schedule_restart_plus_handle_unit_ratio)}, "
            f"status_counts={row.schedule_status_counts}"
        )
    lines.extend(["", "## Selected Source Availability"])
    selected = source_availability[source_availability["selector_selected"].astype(bool)]
    for row in selected.itertuples(index=False):
        lines.append(
            "- "
            f"{row.case_id} {row.source_endpoint_signature_id}: "
            f"count={row.source_observed_count}/{row.baseline_run_count}, "
            f"source_p={row.source_availability_rate:.3f}, "
            f"handle_p={row.handle_known_coassigned_hit_rate:.3f}, "
            f"contribution={row.selected_schedule_contribution_rate:.3f}"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "This is the first accounting gate that includes observed source "
                "availability and one-handle overhead. It is still a fixed-panel "
                "synthetic diagnostic and does not measure wall-clock cost or a "
                "full source-discovery algorithm."
            ),
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    g4_3_dir = Path(args.g4_3_dir)
    g4_4_dir = Path(args.g4_4_dir)
    g4_5_dir = Path(args.g4_5_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_runs = pd.read_csv(g4_3_dir / BASELINE_RUNS_CSV)
    target_table = pd.read_csv(g4_4_dir / TARGET_TABLE_CSV)
    selector_rows = pd.read_csv(g4_5_dir / SELECTOR_SOURCE_ROWS_CSV)

    schedule_rows = _schedule_run_rows(
        baseline_runs=baseline_runs,
        target_table=target_table,
        selector_rows=selector_rows,
    )
    source_availability = _source_availability_rows(
        baseline_runs=baseline_runs,
        selector_rows=selector_rows,
    )
    case_summary = _case_summary(
        schedule_rows=schedule_rows,
        source_availability=source_availability,
        target_table=target_table,
    )
    budget_rows = _budget_curve_rows(case_summary)

    _write_csv(schedule_rows, output_dir / SCHEDULE_RUN_ROWS_CSV)
    _write_csv(source_availability, output_dir / SOURCE_AVAILABILITY_ROWS_CSV)
    _write_csv(case_summary, output_dir / SCHEDULE_CASE_SUMMARY_CSV)
    _write_csv(budget_rows, output_dir / SCHEDULE_BUDGET_CURVE_ROWS_CSV)

    summary = _summary(
        g4_3_dir=g4_3_dir,
        g4_4_dir=g4_4_dir,
        g4_5_dir=g4_5_dir,
        output_dir=output_dir,
        case_summary=case_summary,
        source_availability=source_availability,
        schedule_rows=schedule_rows,
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "variable_pair_synthetic_g4_6_schedule_accounting_config.v1",
        "g4_3_dir": str(g4_3_dir),
        "g4_4_dir": str(g4_4_dir),
        "g4_5_dir": str(g4_5_dir),
        "output_dir": str(output_dir),
        "budgets": list(BUDGETS),
        "schedule_rule": summary["schedule_rule"],
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
        source_availability=source_availability,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g4-3-dir", type=Path, default=DEFAULT_G4_3_DIR)
    parser.add_argument("--g4-4-dir", type=Path, default=DEFAULT_G4_4_DIR)
    parser.add_argument("--g4-5-dir", type=Path, default=DEFAULT_G4_5_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
