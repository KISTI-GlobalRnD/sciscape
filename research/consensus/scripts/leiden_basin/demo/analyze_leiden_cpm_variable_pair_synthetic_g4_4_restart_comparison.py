#!/usr/bin/env python3
"""Compare the frozen G4.3 handle against same-panel restart baselines.

This G4.4 diagnostic reads the fixed G4.3 synthetic variant/control panel. It
compares source-conditioned first-hit probability for the frozen
``bridge_context_release_without_pair_merge`` handle against ordinary
Leiden+CPM restart discovery of the case-level known coassigned endpoint.

The comparison is intentionally narrow. The handle starts from a known
separated source endpoint, while the restart baseline is case-level target
discovery from the G4.3 baseline run pool. Source discovery cost, wall
identification, full-method scheduling, quality/cost value, full NanoClustering
replay, and algorithm-level claims remain out of scope.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from run_leiden_cpm_variable_pair_synthetic_demo import BASE_RESULT_DIR, _json_safe, _write_csv
from run_leiden_cpm_variable_pair_synthetic_g4_3_handle_generalization import (
    BASELINE_RUNS_CSV,
    DEFAULT_OUTPUT_DIR as DEFAULT_G4_3_DIR,
    ENDPOINT_SUMMARY_CSV,
    HANDLE_POLICY_SUMMARY_CSV,
    HANDLE_RUNS_CSV,
    VARIANT_GATE_ROWS_CSV,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_variable_pair_synthetic_g4_4_restart_comparison_v1_20260603"
)

TARGET_TABLE_CSV = "variable_pair_synthetic_g4_4_target_table.csv"
SOURCE_COMPARISON_ROWS_CSV = (
    "variable_pair_synthetic_g4_4_source_comparison_rows.csv"
)
BUDGET_CURVE_ROWS_CSV = "variable_pair_synthetic_g4_4_budget_curve_rows.csv"
CASE_SUMMARY_CSV = "variable_pair_synthetic_g4_4_case_summary.csv"
SUMMARY_JSON = "variable_pair_synthetic_g4_4_summary.json"
CONFIG_JSON = "variable_pair_synthetic_g4_4_config.json"
REPORT_MD = "variable_pair_synthetic_g4_4_report.md"

HANDLE_POLICY = "bridge_context_release_without_pair_merge"
PAIR_ONLY_POLICY = "pair_relation_only"
BUDGETS = (1, 2, 3, 4, 5, 8, 10, 16)
CLAIM_BOUNDARY = (
    "Variable-pair synthetic G4.4 restart-comparison diagnostic only; the "
    "frozen G4.3 source-conditioned bridge-release handle is compared against "
    "same-panel ordinary Leiden+CPM restart target discovery. Source discovery "
    "cost, wall identification, full-method scheduling, full NanoClustering "
    "replay, quality/cost value, and algorithm-level claims remain out of scope."
)
ROUTE_EXECUTION_STATUS = "executed_g4_4_fixed_panel_restart_comparison"
WALL_PROMOTION_STATUS = "not_promoted_restart_comparison_only"
METHOD_STATUS = "source_conditioned_handle_comparison_not_full_method_claim"


def _claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return None
    return float(value)


def _expected_runs(probability: float) -> float | None:
    if probability <= 0.0:
        return None
    return float(1.0 / probability)


def _hit_probability(probability: float, budget: int) -> float:
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0
    return float(1.0 - (1.0 - probability) ** int(budget))


def _geometric_quantile(probability: float, quantile: float) -> int | None:
    if probability <= 0.0:
        return None
    if probability >= 1.0:
        return 1
    return int(math.ceil(math.log(1.0 - float(quantile)) / math.log(1.0 - probability)))


def _observed_first_hit(
    rows: pd.DataFrame,
    *,
    target_signatures: set[str],
    sort_cols: list[str],
    signature_col: str,
) -> int | None:
    ordered = rows.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    hits = ordered[signature_col].astype(str).isin(target_signatures)
    if not bool(hits.any()):
        return None
    return int(hits.idxmax() + 1)


def _target_table(
    *,
    baseline_runs: pd.DataFrame,
    endpoint_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_id, endpoints in endpoint_summary.groupby("case_id", sort=True):
        baseline = baseline_runs[baseline_runs["case_id"].eq(case_id)]
        targets = endpoints[endpoints["pair_coassigned"].astype(bool)]
        target_signatures = set(targets["endpoint_signature_id"].astype(str))
        known_hit_count = int(
            baseline["endpoint_signature_id"].astype(str).isin(target_signatures).sum()
        )
        pair_hit_count = int(baseline["pair_coassigned"].astype(bool).sum())
        run_count = int(len(baseline))
        rows.append(
            {
                "case_id": str(case_id),
                "panel_role": str(endpoints["panel_role"].iloc[0]),
                "expected_gate": str(endpoints["expected_gate"].iloc[0]),
                "baseline_run_count": run_count,
                "known_coassigned_target_count": int(len(targets)),
                "known_coassigned_target_signatures": ";".join(sorted(target_signatures)),
                "baseline_known_coassigned_hit_count": known_hit_count,
                "baseline_known_coassigned_hit_rate": float(
                    known_hit_count / run_count
                )
                if run_count
                else 0.0,
                "baseline_pair_coassigned_hit_count": pair_hit_count,
                "baseline_pair_coassigned_hit_rate": float(pair_hit_count / run_count)
                if run_count
                else 0.0,
                "baseline_expected_runs_to_known_coassigned": _expected_runs(
                    float(known_hit_count / run_count) if run_count else 0.0
                ),
                "baseline_p50_runs_to_known_coassigned": _geometric_quantile(
                    float(known_hit_count / run_count) if run_count else 0.0,
                    0.50,
                ),
                "baseline_p75_runs_to_known_coassigned": _geometric_quantile(
                    float(known_hit_count / run_count) if run_count else 0.0,
                    0.75,
                ),
                "baseline_p95_runs_to_known_coassigned": _geometric_quantile(
                    float(known_hit_count / run_count) if run_count else 0.0,
                    0.95,
                ),
                "baseline_observed_first_known_hit_in_canonical_pool": _observed_first_hit(
                    baseline,
                    target_signatures=target_signatures,
                    sort_cols=["start_condition", "seed"],
                    signature_col="endpoint_signature_id",
                )
                if target_signatures
                else None,
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _row_status(row: dict[str, Any]) -> str:
    expected_gate = str(row["expected_gate"])
    handle_rate = float(row["handle_known_coassigned_hit_rate"])
    baseline_rate = float(row["baseline_known_coassigned_hit_rate"])
    if expected_gate == "bridge_release_robust_pair_coassignment":
        if handle_rate >= 0.8 and handle_rate > baseline_rate:
            return "positive_handle_beats_restart_source_conditioned"
        return "positive_handle_does_not_beat_restart"
    if handle_rate >= 0.8:
        return "control_handle_robust_unexpected"
    if handle_rate > baseline_rate:
        return "control_handle_partial_above_restart_caveat"
    return "control_handle_nonrobust"


def _source_comparison_rows(
    *,
    target_table: pd.DataFrame,
    handle_policy_summary: pd.DataFrame,
    handle_runs: pd.DataFrame,
) -> pd.DataFrame:
    targets = {
        str(row["case_id"]): row
        for row in target_table.to_dict("records")
    }
    bridge_rows = handle_policy_summary[
        handle_policy_summary["handle_policy"].eq(HANDLE_POLICY)
    ].copy()
    pair_only = handle_policy_summary[
        handle_policy_summary["handle_policy"].eq(PAIR_ONLY_POLICY)
    ].copy()
    pair_only_lookup = {
        (str(row["case_id"]), str(row["source_endpoint_signature_id"])): row
        for row in pair_only.to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    for bridge in bridge_rows.to_dict("records"):
        case_id = str(bridge["case_id"])
        target = targets[case_id]
        target_signatures = set(
            str(target["known_coassigned_target_signatures"]).split(";")
            if str(target["known_coassigned_target_signatures"])
            else []
        )
        source_id = str(bridge["source_endpoint_signature_id"])
        runs = handle_runs[
            handle_runs["case_id"].astype(str).eq(case_id)
            & handle_runs["source_endpoint_signature_id"].astype(str).eq(source_id)
            & handle_runs["handle_policy"].astype(str).eq(HANDLE_POLICY)
        ]
        observed_first = (
            _observed_first_hit(
                runs,
                target_signatures=target_signatures,
                sort_cols=["trace_seed"],
                signature_col="result_endpoint_signature_id",
            )
            if target_signatures
            else None
        )
        pair_row = pair_only_lookup.get((case_id, source_id))
        handle_rate = float(bridge["known_coassigned_endpoint_rate"])
        baseline_rate = float(target["baseline_known_coassigned_hit_rate"])
        handle_expected = _expected_runs(handle_rate)
        baseline_expected = _expected_runs(baseline_rate)
        ratio = (
            float(baseline_expected / handle_expected)
            if baseline_expected is not None and handle_expected is not None
            else None
        )
        row = {
            "case_id": case_id,
            "panel_role": str(bridge["panel_role"]),
            "expected_gate": str(bridge["expected_gate"]),
            "source_endpoint_signature_id": source_id,
            "handle_eligible": bool(bridge["handle_eligible"]),
            "released_bridge_count": int(bridge["released_bridge_count"]),
            "changed_nodes_vs_source": int(bridge["changed_nodes_vs_source"]),
            "initial_quality_delta_vs_source": float(
                bridge["initial_quality_delta_vs_source"]
            ),
            "baseline_known_coassigned_hit_rate": baseline_rate,
            "baseline_pair_coassigned_hit_rate": float(
                target["baseline_pair_coassigned_hit_rate"]
            ),
            "handle_known_coassigned_hit_rate": handle_rate,
            "handle_pair_coassigned_hit_rate": float(bridge["pair_coassigned_rate"]),
            "pair_only_known_coassigned_hit_rate": (
                None
                if pair_row is None
                else float(pair_row["known_coassigned_endpoint_rate"])
            ),
            "pair_only_pair_coassigned_hit_rate": (
                None if pair_row is None else float(pair_row["pair_coassigned_rate"])
            ),
            "baseline_expected_runs_to_known_coassigned": baseline_expected,
            "handle_expected_runs_to_known_coassigned": handle_expected,
            "baseline_over_handle_expected_run_ratio": ratio,
            "baseline_p50_runs_to_known_coassigned": target[
                "baseline_p50_runs_to_known_coassigned"
            ],
            "baseline_p75_runs_to_known_coassigned": target[
                "baseline_p75_runs_to_known_coassigned"
            ],
            "baseline_p95_runs_to_known_coassigned": target[
                "baseline_p95_runs_to_known_coassigned"
            ],
            "handle_p50_runs_to_known_coassigned": _geometric_quantile(
                handle_rate,
                0.50,
            ),
            "handle_p75_runs_to_known_coassigned": _geometric_quantile(
                handle_rate,
                0.75,
            ),
            "handle_p95_runs_to_known_coassigned": _geometric_quantile(
                handle_rate,
                0.95,
            ),
            "handle_observed_first_known_hit": observed_first,
            "baseline_observed_first_known_hit_in_canonical_pool": target[
                "baseline_observed_first_known_hit_in_canonical_pool"
            ],
            "comparison_scope": (
                "source_conditioned_handle_vs_case_level_restart_target_discovery"
            ),
        }
        row["g4_4_source_status"] = _row_status(row)
        rows.append(row)
    return _claim_columns(pd.DataFrame(rows))


def _budget_curve_rows(source_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in source_rows.to_dict("records"):
        for budget in BUDGETS:
            rows.append(
                {
                    "case_id": str(source["case_id"]),
                    "panel_role": str(source["panel_role"]),
                    "expected_gate": str(source["expected_gate"]),
                    "source_endpoint_signature_id": str(
                        source["source_endpoint_signature_id"]
                    ),
                    "budget": int(budget),
                    "baseline_hit_probability": _hit_probability(
                        float(source["baseline_known_coassigned_hit_rate"]),
                        int(budget),
                    ),
                    "handle_hit_probability": _hit_probability(
                        float(source["handle_known_coassigned_hit_rate"]),
                        int(budget),
                    ),
                    "pair_only_hit_probability": _hit_probability(
                        float(source["pair_only_known_coassigned_hit_rate"])
                        if source["pair_only_known_coassigned_hit_rate"] is not None
                        and not pd.isna(source["pair_only_known_coassigned_hit_rate"])
                        else 0.0,
                        int(budget),
                    ),
                    "comparison_scope": str(source["comparison_scope"]),
                }
            )
    return _claim_columns(pd.DataFrame(rows))


def _case_summary(source_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_id, group in source_rows.groupby("case_id", sort=True):
        positive = str(group["expected_gate"].iloc[0]) == "bridge_release_robust_pair_coassignment"
        positive_pass = bool(
            positive
            and group["handle_known_coassigned_hit_rate"].ge(0.8).all()
            and group["handle_known_coassigned_hit_rate"].gt(
                group["baseline_known_coassigned_hit_rate"]
            ).all()
        )
        control_nonrobust = bool(
            not positive and group["handle_known_coassigned_hit_rate"].lt(0.8).all()
        )
        if positive_pass:
            status = "positive_handle_beats_restart_all_sources"
        elif control_nonrobust:
            status = "control_handle_nonrobust_all_sources"
        elif positive:
            status = "positive_handle_comparison_failed"
        else:
            status = "control_handle_unexpected_robust_or_mixed"
        rows.append(
            {
                "case_id": str(case_id),
                "panel_role": str(group["panel_role"].iloc[0]),
                "expected_gate": str(group["expected_gate"].iloc[0]),
                "source_count": int(len(group)),
                "baseline_known_coassigned_hit_rate": float(
                    group["baseline_known_coassigned_hit_rate"].iloc[0]
                ),
                "handle_known_coassigned_hit_rate_min": float(
                    group["handle_known_coassigned_hit_rate"].min()
                ),
                "handle_known_coassigned_hit_rate_median": float(
                    group["handle_known_coassigned_hit_rate"].median()
                ),
                "handle_known_coassigned_hit_rate_max": float(
                    group["handle_known_coassigned_hit_rate"].max()
                ),
                "pair_only_known_coassigned_hit_rate_median": float(
                    group["pair_only_known_coassigned_hit_rate"].fillna(0.0).median()
                ),
                "baseline_expected_runs_to_known_coassigned": _finite_or_none(
                    group["baseline_expected_runs_to_known_coassigned"].iloc[0]
                ),
                "handle_expected_runs_to_known_coassigned_median": _finite_or_none(
                    group["handle_expected_runs_to_known_coassigned"].median()
                ),
                "baseline_over_handle_expected_run_ratio_median": _finite_or_none(
                    group["baseline_over_handle_expected_run_ratio"].median()
                ),
                "source_status_counts": json.dumps(
                    group["g4_4_source_status"].value_counts().to_dict(),
                    sort_keys=True,
                ),
                "g4_4_case_status": status,
                "comparison_scope": str(group["comparison_scope"].iloc[0]),
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _summary(
    *,
    g4_3_dir: Path,
    output_dir: Path,
    target_table: pd.DataFrame,
    source_rows: pd.DataFrame,
    case_summary: pd.DataFrame,
) -> dict[str, Any]:
    positive = case_summary[case_summary["panel_role"].eq("positive_holdout")]
    controls = case_summary[~case_summary["panel_role"].eq("positive_holdout")]
    return {
        "schema": "variable_pair_synthetic_g4_4_restart_comparison_summary.v1",
        "status": ROUTE_EXECUTION_STATUS,
        "g4_3_dir": str(g4_3_dir),
        "output_dir": str(output_dir),
        "target_case_count": int(len(target_table)),
        "source_comparison_count": int(len(source_rows)),
        "case_count": int(len(case_summary)),
        "positive_case_count": int(len(positive)),
        "control_case_count": int(len(controls)),
        "positive_pass_count": int(
            positive["g4_4_case_status"].eq(
                "positive_handle_beats_restart_all_sources"
            ).sum()
        ),
        "positive_fail_count": int(
            (~positive["g4_4_case_status"].eq(
                "positive_handle_beats_restart_all_sources"
            )).sum()
        ),
        "control_nonrobust_count": int(
            controls["g4_4_case_status"].eq(
                "control_handle_nonrobust_all_sources"
            ).sum()
        ),
        "control_unexpected_count": int(
            (~controls["g4_4_case_status"].eq(
                "control_handle_nonrobust_all_sources"
            )).sum()
        ),
        "case_status_counts": case_summary["g4_4_case_status"].value_counts().to_dict(),
        "source_status_counts": source_rows["g4_4_source_status"].value_counts().to_dict(),
        "positive_expected_run_ratio_median": _finite_or_none(
            positive["baseline_over_handle_expected_run_ratio_median"].median()
        )
        if not positive.empty
        else None,
        "control_handle_rate_max": float(
            controls["handle_known_coassigned_hit_rate_max"].max()
        )
        if not controls.empty
        else 0.0,
        "comparison_scope": (
            "source_conditioned_handle_vs_case_level_restart_target_discovery"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    case_summary: pd.DataFrame,
    source_rows: pd.DataFrame,
) -> None:
    lines = [
        "# Variable-Pair Synthetic G4.4 Restart Comparison",
        "",
        f"- status: `{summary['status']}`",
        f"- positive_pass_count: {summary['positive_pass_count']}",
        f"- positive_fail_count: {summary['positive_fail_count']}",
        f"- control_nonrobust_count: {summary['control_nonrobust_count']}",
        f"- control_unexpected_count: {summary['control_unexpected_count']}",
        f"- case_status_counts: {summary['case_status_counts']}",
        f"- source_status_counts: {summary['source_status_counts']}",
        f"- positive_expected_run_ratio_median: {summary['positive_expected_run_ratio_median']}",
        f"- control_handle_rate_max: {summary['control_handle_rate_max']}",
        f"- comparison_scope: {summary['comparison_scope']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Case Summary",
    ]
    for row in case_summary.itertuples(index=False):
        ratio = (
            "NA"
            if pd.isna(row.baseline_over_handle_expected_run_ratio_median)
            else f"{row.baseline_over_handle_expected_run_ratio_median:.6g}"
        )
        baseline_expected = (
            "NA"
            if pd.isna(row.baseline_expected_runs_to_known_coassigned)
            else f"{row.baseline_expected_runs_to_known_coassigned:.6g}"
        )
        handle_expected = (
            "NA"
            if pd.isna(row.handle_expected_runs_to_known_coassigned_median)
            else f"{row.handle_expected_runs_to_known_coassigned_median:.6g}"
        )
        lines.append(
            "- "
            f"{row.case_id} ({row.panel_role}): {row.g4_4_case_status}; "
            f"baseline_p={row.baseline_known_coassigned_hit_rate:.3f}, "
            f"handle_p_med={row.handle_known_coassigned_hit_rate_median:.3f}, "
            f"pair_only_p_med={row.pair_only_known_coassigned_hit_rate_median:.3f}, "
            f"expected_runs baseline/handle={baseline_expected}/{handle_expected}, "
            f"ratio={ratio}"
        )
    lines.extend(["", "## Source Rows"])
    for row in source_rows.itertuples(index=False):
        ratio = (
            "NA"
            if pd.isna(row.baseline_over_handle_expected_run_ratio)
            else f"{row.baseline_over_handle_expected_run_ratio:.6g}"
        )
        lines.append(
            "- "
            f"{row.case_id} {row.source_endpoint_signature_id}: "
            f"{row.g4_4_source_status}; "
            f"baseline_p={row.baseline_known_coassigned_hit_rate:.3f}, "
            f"handle_p={row.handle_known_coassigned_hit_rate:.3f}, "
            f"pair_only_p={row.pair_only_known_coassigned_hit_rate:.3f}, "
            f"ratio={ratio}"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "This comparison is source-conditioned. It does not include the "
                "cost of discovering the source endpoint, selecting the handle, "
                "or running a full method schedule."
            ),
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    g4_3_dir = Path(args.g4_3_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_runs = pd.read_csv(g4_3_dir / BASELINE_RUNS_CSV)
    endpoint_summary = pd.read_csv(g4_3_dir / ENDPOINT_SUMMARY_CSV)
    handle_runs = pd.read_csv(g4_3_dir / HANDLE_RUNS_CSV)
    handle_policy_summary = pd.read_csv(g4_3_dir / HANDLE_POLICY_SUMMARY_CSV)
    # Read for schema provenance: this confirms the G4.3 gate table is present.
    pd.read_csv(g4_3_dir / VARIANT_GATE_ROWS_CSV)

    target_table = _target_table(
        baseline_runs=baseline_runs,
        endpoint_summary=endpoint_summary,
    )
    source_rows = _source_comparison_rows(
        target_table=target_table,
        handle_policy_summary=handle_policy_summary,
        handle_runs=handle_runs,
    )
    budget_rows = _budget_curve_rows(source_rows)
    case_summary = _case_summary(source_rows)
    _write_csv(target_table, output_dir / TARGET_TABLE_CSV)
    _write_csv(source_rows, output_dir / SOURCE_COMPARISON_ROWS_CSV)
    _write_csv(budget_rows, output_dir / BUDGET_CURVE_ROWS_CSV)
    _write_csv(case_summary, output_dir / CASE_SUMMARY_CSV)
    summary = _summary(
        g4_3_dir=g4_3_dir,
        output_dir=output_dir,
        target_table=target_table,
        source_rows=source_rows,
        case_summary=case_summary,
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "variable_pair_synthetic_g4_4_restart_comparison_config.v1",
        "g4_3_dir": str(g4_3_dir),
        "output_dir": str(output_dir),
        "handle_policy": HANDLE_POLICY,
        "pair_only_policy": PAIR_ONLY_POLICY,
        "budgets": list(BUDGETS),
        "comparison_scope": (
            "source_conditioned_handle_vs_case_level_restart_target_discovery"
        ),
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
        source_rows=source_rows,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g4-3-dir", type=Path, default=DEFAULT_G4_3_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
