#!/usr/bin/env python3
"""Evaluate a target-free source-local selector for the frozen G4.3 handle.

This G4.5 diagnostic reads the G4.3 handle-generalization panel and the G4.4
restart comparison. It freezes one selector/suppression rule for the
``bridge_context_release_without_pair_merge`` handle:

1. the source is handle-eligible and releases at least one bridge;
2. applying the bridge-release initialization keeps the pair separated;
3. the bridge-release initialization is source-neutral in CPM quality;
4. the graph has minimum direct pair support.

The rule uses graph/source-membership features and CPM quality of the proposed
initialization only. It does not read target endpoint signatures or result
outcomes when deciding whether to fire. The audit then checks whether the rule
keeps the positive G4.4 source-conditioned wins while suppressing matched-control
partial-above-restart rows.

It remains a synthetic selector/suppression diagnostic only. It does not
include source discovery cost, wall identification, full-method scheduling,
full NanoClustering replay, quality/cost value, or algorithm-level claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_leiden_cpm_variable_pair_synthetic_demo import BASE_RESULT_DIR, _json_safe, _write_csv
from run_leiden_cpm_variable_pair_synthetic_g4_3_handle_generalization import (
    DEFAULT_OUTPUT_DIR as DEFAULT_G4_3_DIR,
    HANDLE_POLICY_SUMMARY_CSV,
    PANEL_CASES_CSV,
)
from analyze_leiden_cpm_variable_pair_synthetic_g4_4_restart_comparison import (
    BUDGET_CURVE_ROWS_CSV,
    DEFAULT_OUTPUT_DIR as DEFAULT_G4_4_DIR,
    SOURCE_COMPARISON_ROWS_CSV,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_variable_pair_synthetic_g4_5_selector_suppression_v1_20260603"
)

SELECTOR_SOURCE_ROWS_CSV = "variable_pair_synthetic_g4_5_selector_source_rows.csv"
SELECTOR_CASE_SUMMARY_CSV = "variable_pair_synthetic_g4_5_selector_case_summary.csv"
SELECTOR_BUDGET_CURVE_ROWS_CSV = (
    "variable_pair_synthetic_g4_5_selector_budget_curve_rows.csv"
)
SUMMARY_JSON = "variable_pair_synthetic_g4_5_summary.json"
CONFIG_JSON = "variable_pair_synthetic_g4_5_config.json"
REPORT_MD = "variable_pair_synthetic_g4_5_report.md"

HANDLE_POLICY = "bridge_context_release_without_pair_merge"
DIRECT_PAIR_SUPPORT_MIN = 1.0
SOURCE_NEUTRAL_DELTA_ABS_MAX = 1.0e-6
CLAIM_BOUNDARY = (
    "Variable-pair synthetic G4.5 selector/suppression diagnostic only; a "
    "target-free source-local selector is evaluated for the frozen G4.3 "
    "bridge-release handle using graph/source-membership features and local CPM "
    "delta. Source discovery cost, wall identification, full-method scheduling, "
    "full NanoClustering replay, quality/cost value, and algorithm-level claims "
    "remain out of scope."
)
ROUTE_EXECUTION_STATUS = "executed_g4_5_source_local_selector_suppression"
WALL_PROMOTION_STATUS = "not_promoted_selector_suppression_only"
METHOD_STATUS = "selector_gate_not_full_method_claim"


def _claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _selector_decision(row: dict[str, Any]) -> bool:
    return bool(
        bool(row["handle_eligible"])
        and int(row["released_bridge_count"]) > 0
        and not bool(row["initial_pair_coassigned"])
        and bool(row["initial_keeps_pair_relation"])
        and abs(float(row["initial_quality_delta_vs_source"]))
        <= SOURCE_NEUTRAL_DELTA_ABS_MAX
        and float(row["direct_weight"]) >= DIRECT_PAIR_SUPPORT_MIN
    )


def _selector_reason(row: dict[str, Any]) -> str:
    reasons: list[str] = []
    if not bool(row["handle_eligible"]):
        reasons.append("not_handle_eligible")
    if int(row["released_bridge_count"]) <= 0:
        reasons.append("no_released_bridge")
    if bool(row["initial_pair_coassigned"]):
        reasons.append("initial_merges_pair")
    if not bool(row["initial_keeps_pair_relation"]):
        reasons.append("initial_changes_pair_relation")
    if abs(float(row["initial_quality_delta_vs_source"])) > SOURCE_NEUTRAL_DELTA_ABS_MAX:
        reasons.append("release_not_source_neutral")
    if float(row["direct_weight"]) < DIRECT_PAIR_SUPPORT_MIN:
        reasons.append("insufficient_direct_pair_support")
    return "selected" if not reasons else ";".join(reasons)


def _selector_status(row: dict[str, Any]) -> str:
    selected = bool(row["selector_selected"])
    g4_4_status = str(row["g4_4_source_status"])
    panel_role = str(row["panel_role"])
    if selected and g4_4_status == "positive_handle_beats_restart_source_conditioned":
        return "selected_positive_win"
    if not selected and g4_4_status == "positive_handle_beats_restart_source_conditioned":
        return "suppressed_positive_win"
    if selected and panel_role != "positive_holdout":
        return "selected_control_leak"
    if not selected and g4_4_status == "control_handle_partial_above_restart_caveat":
        return "suppressed_control_partial_caveat"
    if not selected and g4_4_status == "control_handle_nonrobust":
        return "suppressed_control_nonrobust"
    return "selector_status_unclassified"


def _selector_source_rows(
    *,
    panel_cases: pd.DataFrame,
    handle_policy_summary: pd.DataFrame,
    source_comparison: pd.DataFrame,
) -> pd.DataFrame:
    panel_cols = [
        "case_id",
        "direct_weight",
        "pair_bridge_weight",
        "bridge_host_weight",
        "host_clique_weight",
    ]
    panel = panel_cases[panel_cols].copy()
    handle = handle_policy_summary[
        handle_policy_summary["handle_policy"].eq(HANDLE_POLICY)
    ].copy()
    handle_cols = [
        "case_id",
        "source_endpoint_signature_id",
        "source_pair_coassigned",
        "initial_pair_coassigned",
        "initial_keeps_pair_relation",
        "initial_quality",
        "initial_cluster_count",
        "initial_coassoc_distance_vs_source",
        "handle_policy_class",
    ]
    rows = (
        source_comparison.merge(panel, on="case_id", how="left")
        .merge(handle[handle_cols], on=["case_id", "source_endpoint_signature_id"], how="left")
        .copy()
    )
    missing = rows[
        rows[["direct_weight", "handle_eligible", "initial_quality_delta_vs_source"]]
        .isna()
        .any(axis=1)
    ]
    if not missing.empty:
        raise ValueError(
            "selector rows missing panel or handle source-local fields: "
            f"{missing[['case_id', 'source_endpoint_signature_id']].to_dict('records')}"
        )
    rows["source_neutral_release"] = (
        rows["initial_quality_delta_vs_source"].astype(float).abs()
        <= SOURCE_NEUTRAL_DELTA_ABS_MAX
    )
    rows["direct_pair_support_floor_passed"] = (
        rows["direct_weight"].astype(float) >= DIRECT_PAIR_SUPPORT_MIN
    )
    rows["selector_selected"] = [
        _selector_decision(row) for row in rows.to_dict("records")
    ]
    rows["selector_suppression_reason"] = [
        _selector_reason(row) for row in rows.to_dict("records")
    ]
    rows["g4_5_selector_status"] = [
        _selector_status(row) for row in rows.to_dict("records")
    ]
    rows["selected_handle_known_hit_rate"] = rows.apply(
        lambda row: float(row["handle_known_coassigned_hit_rate"])
        if bool(row["selector_selected"])
        else 0.0,
        axis=1,
    )
    rows["selected_expected_runs_to_known_coassigned"] = rows.apply(
        lambda row: float(1.0 / row["selected_handle_known_hit_rate"])
        if float(row["selected_handle_known_hit_rate"]) > 0.0
        else None,
        axis=1,
    )
    return _claim_columns(
        rows.sort_values(
            ["case_id", "source_endpoint_signature_id"],
            kind="stable",
        )
    )


def _case_status(group: pd.DataFrame) -> str:
    positive = str(group["panel_role"].iloc[0]) == "positive_holdout"
    selected_count = int(group["selector_selected"].astype(bool).sum())
    if positive:
        if selected_count == len(group) and group["g4_5_selector_status"].eq(
            "selected_positive_win"
        ).all():
            return "positive_wins_retained_all_sources"
        return "positive_win_suppressed_or_not_selected"
    if selected_count == 0:
        return "control_sources_suppressed_all"
    return "control_selector_leak"


def _selector_case_summary(selector_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_id, group in selector_rows.groupby("case_id", sort=True):
        selected = group[group["selector_selected"].astype(bool)]
        suppressed = group[~group["selector_selected"].astype(bool)]
        rows.append(
            {
                "case_id": str(case_id),
                "panel_role": str(group["panel_role"].iloc[0]),
                "expected_gate": str(group["expected_gate"].iloc[0]),
                "source_count": int(len(group)),
                "selector_selected_count": int(len(selected)),
                "selector_suppressed_count": int(len(suppressed)),
                "selected_positive_win_count": int(
                    group["g4_5_selector_status"].eq("selected_positive_win").sum()
                ),
                "suppressed_positive_win_count": int(
                    group["g4_5_selector_status"].eq("suppressed_positive_win").sum()
                ),
                "selected_control_leak_count": int(
                    group["g4_5_selector_status"].eq("selected_control_leak").sum()
                ),
                "suppressed_control_partial_caveat_count": int(
                    group["g4_5_selector_status"]
                    .eq("suppressed_control_partial_caveat")
                    .sum()
                ),
                "suppressed_control_nonrobust_count": int(
                    group["g4_5_selector_status"].eq("suppressed_control_nonrobust").sum()
                ),
                "baseline_known_coassigned_hit_rate": float(
                    group["baseline_known_coassigned_hit_rate"].iloc[0]
                ),
                "handle_known_coassigned_hit_rate_median": float(
                    group["handle_known_coassigned_hit_rate"].median()
                ),
                "selected_handle_known_hit_rate_median": float(
                    selected["handle_known_coassigned_hit_rate"].median()
                )
                if not selected.empty
                else 0.0,
                "initial_quality_delta_vs_source_min": float(
                    group["initial_quality_delta_vs_source"].min()
                ),
                "initial_quality_delta_vs_source_median": float(
                    group["initial_quality_delta_vs_source"].median()
                ),
                "initial_quality_delta_vs_source_max": float(
                    group["initial_quality_delta_vs_source"].max()
                ),
                "selector_reason_counts": json.dumps(
                    group["selector_suppression_reason"].value_counts().to_dict(),
                    sort_keys=True,
                ),
                "selector_status_counts": json.dumps(
                    group["g4_5_selector_status"].value_counts().to_dict(),
                    sort_keys=True,
                ),
                "g4_5_case_status": _case_status(group),
                "selector_rule_id": "neutral_release_with_direct_support_v1",
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _selector_budget_curve(
    *,
    selector_rows: pd.DataFrame,
    budget_curve: pd.DataFrame,
) -> pd.DataFrame:
    selector_cols = [
        "case_id",
        "source_endpoint_signature_id",
        "selector_selected",
        "g4_5_selector_status",
    ]
    rows = budget_curve.merge(
        selector_rows[selector_cols],
        on=["case_id", "source_endpoint_signature_id"],
        how="left",
    )
    if rows["selector_selected"].isna().any():
        raise ValueError("budget rows missing selector decisions")
    rows["selected_handle_hit_probability"] = rows.apply(
        lambda row: float(row["handle_hit_probability"])
        if bool(row["selector_selected"])
        else 0.0,
        axis=1,
    )
    return _claim_columns(rows)


def _summary(
    *,
    g4_3_dir: Path,
    g4_4_dir: Path,
    output_dir: Path,
    selector_rows: pd.DataFrame,
    case_summary: pd.DataFrame,
) -> dict[str, Any]:
    positives = selector_rows[selector_rows["panel_role"].eq("positive_holdout")]
    controls = selector_rows[~selector_rows["panel_role"].eq("positive_holdout")]
    return {
        "schema": "variable_pair_synthetic_g4_5_selector_suppression_summary.v1",
        "status": ROUTE_EXECUTION_STATUS,
        "g4_3_dir": str(g4_3_dir),
        "g4_4_dir": str(g4_4_dir),
        "output_dir": str(output_dir),
        "source_row_count": int(len(selector_rows)),
        "case_count": int(len(case_summary)),
        "selector_selected_count": int(selector_rows["selector_selected"].astype(bool).sum()),
        "selector_suppressed_count": int((~selector_rows["selector_selected"].astype(bool)).sum()),
        "positive_source_count": int(len(positives)),
        "positive_selected_count": int(positives["selector_selected"].astype(bool).sum()),
        "positive_suppressed_count": int((~positives["selector_selected"].astype(bool)).sum()),
        "control_source_count": int(len(controls)),
        "control_selected_count": int(controls["selector_selected"].astype(bool).sum()),
        "control_suppressed_count": int((~controls["selector_selected"].astype(bool)).sum()),
        "suppressed_control_partial_caveat_count": int(
            selector_rows["g4_5_selector_status"]
            .eq("suppressed_control_partial_caveat")
            .sum()
        ),
        "selected_positive_win_count": int(
            selector_rows["g4_5_selector_status"].eq("selected_positive_win").sum()
        ),
        "selector_status_counts": selector_rows[
            "g4_5_selector_status"
        ].value_counts().to_dict(),
        "case_status_counts": case_summary["g4_5_case_status"].value_counts().to_dict(),
        "selector_gate_passed": bool(
            positives["selector_selected"].astype(bool).all()
            and not controls["selector_selected"].astype(bool).any()
        ),
        "selector_rule": {
            "selector_rule_id": "neutral_release_with_direct_support_v1",
            "direct_pair_support_min": DIRECT_PAIR_SUPPORT_MIN,
            "source_neutral_delta_abs_max": SOURCE_NEUTRAL_DELTA_ABS_MAX,
            "requires_handle_eligible": True,
            "requires_released_bridge_count_gt_zero": True,
            "requires_initial_pair_not_coassigned": True,
            "requires_initial_keeps_pair_relation": True,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    case_summary: pd.DataFrame,
    selector_rows: pd.DataFrame,
) -> None:
    lines = [
        "# Variable-Pair Synthetic G4.5 Selector Suppression",
        "",
        f"- status: `{summary['status']}`",
        f"- selector_gate_passed: {summary['selector_gate_passed']}",
        f"- selector_selected_count: {summary['selector_selected_count']}",
        f"- selector_suppressed_count: {summary['selector_suppressed_count']}",
        f"- positive_selected_count: {summary['positive_selected_count']}",
        f"- positive_suppressed_count: {summary['positive_suppressed_count']}",
        f"- control_selected_count: {summary['control_selected_count']}",
        f"- control_suppressed_count: {summary['control_suppressed_count']}",
        f"- suppressed_control_partial_caveat_count: {summary['suppressed_control_partial_caveat_count']}",
        f"- selector_status_counts: {summary['selector_status_counts']}",
        f"- case_status_counts: {summary['case_status_counts']}",
        f"- selector_rule: {summary['selector_rule']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Case Summary",
    ]
    for row in case_summary.itertuples(index=False):
        lines.append(
            "- "
            f"{row.case_id} ({row.panel_role}): {row.g4_5_case_status}; "
            f"selected={row.selector_selected_count}/{row.source_count}, "
            f"delta_median={row.initial_quality_delta_vs_source_median:.6g}, "
            f"reasons={row.selector_reason_counts}"
        )
    lines.extend(["", "## Source Rows"])
    for row in selector_rows.itertuples(index=False):
        lines.append(
            "- "
            f"{row.case_id} {row.source_endpoint_signature_id}: "
            f"{row.g4_5_selector_status}; "
            f"selected={row.selector_selected}, "
            f"reason={row.selector_suppression_reason}, "
            f"direct={row.direct_weight:.6g}, "
            f"delta={row.initial_quality_delta_vs_source:.6g}, "
            f"handle_p={row.handle_known_coassigned_hit_rate:.3f}"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "This selector is target-free, but it is still source-conditioned "
                "and requires source membership plus local CPM scoring. It is not "
                "a full method schedule."
            ),
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    g4_3_dir = Path(args.g4_3_dir)
    g4_4_dir = Path(args.g4_4_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_cases = pd.read_csv(g4_3_dir / PANEL_CASES_CSV)
    handle_policy_summary = pd.read_csv(g4_3_dir / HANDLE_POLICY_SUMMARY_CSV)
    source_comparison = pd.read_csv(g4_4_dir / SOURCE_COMPARISON_ROWS_CSV)
    budget_curve = pd.read_csv(g4_4_dir / BUDGET_CURVE_ROWS_CSV)

    selector_rows = _selector_source_rows(
        panel_cases=panel_cases,
        handle_policy_summary=handle_policy_summary,
        source_comparison=source_comparison,
    )
    case_summary = _selector_case_summary(selector_rows)
    budget_rows = _selector_budget_curve(
        selector_rows=selector_rows,
        budget_curve=budget_curve,
    )
    _write_csv(selector_rows, output_dir / SELECTOR_SOURCE_ROWS_CSV)
    _write_csv(case_summary, output_dir / SELECTOR_CASE_SUMMARY_CSV)
    _write_csv(budget_rows, output_dir / SELECTOR_BUDGET_CURVE_ROWS_CSV)
    summary = _summary(
        g4_3_dir=g4_3_dir,
        g4_4_dir=g4_4_dir,
        output_dir=output_dir,
        selector_rows=selector_rows,
        case_summary=case_summary,
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "variable_pair_synthetic_g4_5_selector_suppression_config.v1",
        "g4_3_dir": str(g4_3_dir),
        "g4_4_dir": str(g4_4_dir),
        "output_dir": str(output_dir),
        "selector_rule": summary["selector_rule"],
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
        selector_rows=selector_rows,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g4-3-dir", type=Path, default=DEFAULT_G4_3_DIR)
    parser.add_argument("--g4-4-dir", type=Path, default=DEFAULT_G4_4_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
