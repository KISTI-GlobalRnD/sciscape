#!/usr/bin/env python3
"""Design the opportunity-regime gate from G4.3/G4.6 success and G4.7 failure.

This G4.8 artifact reads only materialized G4.3, G4.6, and G4.7 outputs. It
does not run Leiden, tune thresholds, or change the frozen G4.3/G4.5/G4.6
rules. The goal is to turn the G4.7 boundary failure into a sharper next gate:
characterize when the bridge-release schedule has an opportunity surface at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_leiden_cpm_variable_pair_synthetic_demo import (
    BASE_RESULT_DIR,
    _json_safe,
    _write_csv,
)
from run_leiden_cpm_variable_pair_synthetic_g4_3_handle_generalization import (
    PANEL_CASES_CSV,
    VARIANT_GATE_ROWS_CSV,
)
from analyze_leiden_cpm_variable_pair_synthetic_g4_6_schedule_accounting import (
    DEFAULT_OUTPUT_DIR as DEFAULT_G4_6_DIR,
    SCHEDULE_CASE_SUMMARY_CSV,
)
from run_leiden_cpm_variable_pair_synthetic_g4_7_independent_schedule_stress import (
    G4_3_DIRNAME,
    G4_7_CASE_SUMMARY_CSV,
)


DEFAULT_G4_3_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_variable_pair_synthetic_g4_3_handle_generalization_v1_20260603"
)
DEFAULT_G4_6_ACCOUNTING_DIR = DEFAULT_G4_6_DIR
DEFAULT_G4_7_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_variable_pair_synthetic_g4_7_independent_schedule_stress_v1_20260603"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_variable_pair_synthetic_g4_8_opportunity_regime_design_v1_20260603"
)

CASE_ROWS_CSV = "variable_pair_synthetic_g4_8_opportunity_regime_case_rows.csv"
REGIME_SUMMARY_CSV = "variable_pair_synthetic_g4_8_opportunity_regime_summary.csv"
NEXT_GATE_DESIGN_CSV = "variable_pair_synthetic_g4_8_next_gate_design_rows.csv"
SUMMARY_JSON = "variable_pair_synthetic_g4_8_summary.json"
CONFIG_JSON = "variable_pair_synthetic_g4_8_config.json"
REPORT_MD = "variable_pair_synthetic_g4_8_report.md"

CLAIM_BOUNDARY = (
    "Variable-pair synthetic G4.8 opportunity-regime design artifact only; reads "
    "G4.3, G4.6, and G4.7 materialized outputs to classify source-opportunity "
    "regimes. No new Leiden runs, no selector retuning, no wall promotion, no "
    "full NanoClustering replay, no quality/cost value, and no algorithm-level "
    "claims."
)
ROUTE_EXECUTION_STATUS = "not_executed_g4_8_design_only"
WALL_PROMOTION_STATUS = "not_promoted_opportunity_regime_design_only"
METHOD_STATUS = "opportunity_regime_design_not_method_claim"


def _claim_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _load_panel_rows(label: str, g4_3_dir: Path) -> pd.DataFrame:
    cases = pd.read_csv(g4_3_dir / PANEL_CASES_CSV)
    gates = pd.read_csv(g4_3_dir / VARIANT_GATE_ROWS_CSV)
    rows = cases.merge(
        gates,
        on=["case_id", "panel_role", "expected_gate"],
        suffixes=("", "_gate"),
        how="inner",
    )
    rows["evidence_panel"] = label
    return rows


def _merge_schedule_rows(rows: pd.DataFrame, g4_6_dir: Path | None) -> pd.DataFrame:
    if g4_6_dir is None:
        return rows
    schedule_path = Path(g4_6_dir) / SCHEDULE_CASE_SUMMARY_CSV
    if not schedule_path.exists():
        return rows
    schedule = pd.read_csv(schedule_path)
    schedule_cols = [
        "case_id",
        "baseline_known_coassigned_hit_rate",
        "selected_source_count",
        "selected_source_discovery_rate",
        "schedule_known_coassigned_hit_rate",
    ]
    existing = [col for col in schedule_cols if col in rows.columns and col != "case_id"]
    return rows.drop(columns=existing, errors="ignore").merge(
        schedule[schedule_cols],
        on="case_id",
        how="left",
    )


def _load_g4_7_rows(g4_7_dir: Path) -> pd.DataFrame:
    rows = pd.read_csv(g4_7_dir / G4_7_CASE_SUMMARY_CSV)
    g4_3_dir = g4_7_dir / G4_3_DIRNAME
    panel = pd.read_csv(g4_3_dir / PANEL_CASES_CSV)
    gates = pd.read_csv(g4_3_dir / VARIANT_GATE_ROWS_CSV)
    return rows.merge(
        panel[
            [
                "case_id",
                "direct_weight",
                "pair_bridge_weight",
                "bridge_host_weight",
                "host_clique_weight",
                "pair_node_size",
                "note",
            ]
        ],
        on="case_id",
        how="left",
    ).merge(
        gates[
            [
                "case_id",
                "separated_endpoint_count",
                "coassigned_endpoint_count",
            ]
        ],
        on="case_id",
        how="left",
    )


def _combined_case_rows(g4_3_dir: Path, g4_6_dir: Path, g4_7_dir: Path) -> pd.DataFrame:
    base = _merge_schedule_rows(
        _load_panel_rows("g4_3_success_panel", g4_3_dir),
        g4_6_dir,
    )
    stress = _load_g4_7_rows(g4_7_dir)
    stress["evidence_panel"] = "g4_7_independent_stress_panel"
    # Align the stage columns that are only present in the G4.7 case summary.
    for col in [
        "baseline_known_coassigned_hit_rate",
        "selected_source_count",
        "selected_source_discovery_rate",
        "schedule_known_coassigned_hit_rate",
        "g4_7_case_status",
    ]:
        if col not in base.columns:
            base[col] = None
    for col in [
        "separated_endpoint_count",
        "coassigned_endpoint_count",
        "bridge_handle_eligible_source_count",
        "bridge_handle_robust_pair_coassignment_count",
        "pair_relation_only_robust_pair_coassignment_count",
        "bridge_handle_pair_rate_median",
        "gate_passed",
        "g4_3_gate_status",
    ]:
        if col not in stress.columns:
            stress[col] = None
    common_cols = [
        "evidence_panel",
        "case_id",
        "panel_role",
        "expected_gate",
        "direct_weight",
        "pair_bridge_weight",
        "bridge_host_weight",
        "host_clique_weight",
        "pair_node_size",
        "note",
        "baseline_pair_coassigned_run_share",
        "baseline_known_coassigned_hit_rate",
        "separated_endpoint_count",
        "coassigned_endpoint_count",
        "bridge_handle_eligible_source_count",
        "bridge_handle_robust_pair_coassignment_count",
        "pair_relation_only_robust_pair_coassignment_count",
        "bridge_handle_pair_rate_median",
        "selected_source_count",
        "selected_source_discovery_rate",
        "schedule_known_coassigned_hit_rate",
        "gate_passed",
        "g4_3_gate_status",
        "g4_7_case_status",
    ]
    rows = pd.concat([base[common_cols], stress[common_cols]], ignore_index=True)
    rows["endpoint_coexistence"] = (
        rows["separated_endpoint_count"].fillna(0).astype(int).gt(0)
        & rows["coassigned_endpoint_count"].fillna(0).astype(int).gt(0)
    )
    rows["target_saturated"] = rows["baseline_pair_coassigned_run_share"].fillna(0).ge(1.0)
    rows["target_absent"] = rows["baseline_pair_coassigned_run_share"].fillna(0).le(0.0)
    rows["bridge_release_eligible"] = (
        rows["bridge_handle_eligible_source_count"].fillna(0).astype(int).gt(0)
    )
    rows["bridge_release_robust"] = (
        rows["bridge_handle_robust_pair_coassignment_count"].fillna(0).astype(int).gt(0)
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
    rows["opportunity_regime"] = [
        _classify_opportunity(row) for row in rows.to_dict("records")
    ]
    rows["next_gate_role"] = [
        _next_gate_role(row) for row in rows.to_dict("records")
    ]
    return _claim_columns(rows)


def _classify_opportunity(row: dict[str, Any]) -> str:
    role = str(row["panel_role"])
    if bool(row["bridge_release_robust"]) and bool(row["endpoint_coexistence"]):
        return "bridge_release_opportunity_ready"
    if bool(row["pair_only_robust"]):
        return "pair_only_opportunity_not_bridge_release"
    if bool(row["target_saturated"]):
        return "target_saturated_no_source_opportunity"
    if bool(row["target_absent"]):
        if bool(row["bridge_release_eligible"]):
            return "target_absent_bridge_release_no_value"
        return "target_absent_no_bridge_source"
    if role != "positive_holdout" and bool(row["endpoint_coexistence"]):
        return "coexistence_control_suppressed"
    if bool(row["endpoint_coexistence"]) and not bool(row["bridge_release_eligible"]):
        return "coexistence_without_bridge_release_source"
    return "unclassified_opportunity_boundary"


def _next_gate_role(row: dict[str, Any]) -> str:
    regime = str(row["opportunity_regime"])
    if regime == "bridge_release_opportunity_ready":
        return "ready_positive_anchor"
    if regime == "coexistence_control_suppressed":
        return "suppressed_control_anchor"
    if regime == "target_saturated_no_source_opportunity":
        return "target_saturation_boundary"
    if regime == "pair_only_opportunity_not_bridge_release":
        return "competing_pair_only_boundary"
    if regime.startswith("target_absent"):
        return "no_target_boundary"
    return "needs_manual_review"


def _regime_summary(case_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["opportunity_regime", "next_gate_role"]
    for keys, group in case_rows.groupby(group_cols, sort=True):
        opportunity_regime, next_gate_role = keys
        rows.append(
            {
                "opportunity_regime": str(opportunity_regime),
                "next_gate_role": str(next_gate_role),
                "case_count": int(len(group)),
                "positive_count": int(group["panel_role"].eq("positive_holdout").sum()),
                "control_count": int((~group["panel_role"].eq("positive_holdout")).sum()),
                "evidence_panels": ";".join(sorted(group["evidence_panel"].unique())),
                "case_ids": ";".join(sorted(group["case_id"].astype(str))),
                "direct_weight_min": float(group["direct_weight"].min()),
                "direct_weight_max": float(group["direct_weight"].max()),
                "pair_bridge_weight_min": float(group["pair_bridge_weight"].min()),
                "pair_bridge_weight_max": float(group["pair_bridge_weight"].max()),
                "bridge_host_weight_min": float(group["bridge_host_weight"].min()),
                "bridge_host_weight_max": float(group["bridge_host_weight"].max()),
                "host_clique_weight_min": float(group["host_clique_weight"].min()),
                "host_clique_weight_max": float(group["host_clique_weight"].max()),
            }
        )
    return _claim_columns(pd.DataFrame(rows))


def _next_gate_design() -> pd.DataFrame:
    rows = [
        {
            "design_step": "G4.8A",
            "step_name": "opportunity_regime_metric_freeze",
            "purpose": (
                "Freeze source-opportunity metrics before any new panel: endpoint "
                "coexistence, bridge-release eligibility, source-neutral release, "
                "pair-only ambiguity, target saturation, and control leak."
            ),
            "allowed_inputs": "materialized G4.3/G4.6/G4.7 outputs only",
            "blocked_action": "no selector retuning and no new threshold search",
            "pass_read": "metric definitions are auditable and classify every existing case",
        },
        {
            "design_step": "G4.8B",
            "step_name": "fresh_predeclared_regime_panel",
            "purpose": (
                "Build a small panel by regime cell rather than by expected success: "
                "ready positive anchor, suppressed coexistence control, target "
                "saturation boundary, pair-only boundary, and no-target boundary."
            ),
            "allowed_inputs": "G4.8A regime labels and fixed graph construction primitives",
            "blocked_action": "do not tune G4.5 selector on this panel",
            "pass_read": "the frozen schedule only fires in ready cells and remains no-leak elsewhere",
        },
        {
            "design_step": "G4.8C",
            "step_name": "source_discovery_condition_probe",
            "purpose": (
                "After a fresh regime panel, replace observed-source availability with "
                "a predeclared source-discovery condition and measure no-op mass."
            ),
            "allowed_inputs": "frozen G4.8B panel plus frozen schedule",
            "blocked_action": "no wall/pathway or algorithm language",
            "pass_read": "source-discovery condition exposes separated eligible sources before handle use",
        },
    ]
    return _claim_columns(pd.DataFrame(rows))


def _summary(
    *,
    output_dir: Path,
    case_rows: pd.DataFrame,
    regime_summary: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema": "variable_pair_synthetic_g4_8_opportunity_regime_design_summary.v1",
        "status": ROUTE_EXECUTION_STATUS,
        "output_dir": str(output_dir),
        "case_count": int(len(case_rows)),
        "regime_count": int(case_rows["opportunity_regime"].nunique()),
        "regime_counts": case_rows["opportunity_regime"].value_counts().to_dict(),
        "next_gate_role_counts": case_rows["next_gate_role"].value_counts().to_dict(),
        "ready_positive_anchor_count": int(
            case_rows["next_gate_role"].eq("ready_positive_anchor").sum()
        ),
        "target_saturation_boundary_count": int(
            case_rows["next_gate_role"].eq("target_saturation_boundary").sum()
        ),
        "competing_pair_only_boundary_count": int(
            case_rows["next_gate_role"].eq("competing_pair_only_boundary").sum()
        ),
        "suppressed_control_anchor_count": int(
            case_rows["next_gate_role"].eq("suppressed_control_anchor").sum()
        ),
        "regime_summary_row_count": int(len(regime_summary)),
        "recommended_next_gate": (
            "Build G4.8B as a fresh predeclared regime-cell panel after freezing "
            "the G4.8A metrics; do not retune the selector."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    regime_summary: pd.DataFrame,
    case_rows: pd.DataFrame,
    next_gate_design: pd.DataFrame,
) -> None:
    lines = [
        "# Variable-Pair Synthetic G4.8 Opportunity Regime Design",
        "",
        f"- status: `{summary['status']}`",
        f"- regime_counts: {summary['regime_counts']}",
        f"- next_gate_role_counts: {summary['next_gate_role_counts']}",
        f"- recommended_next_gate: {summary['recommended_next_gate']}",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Regime Summary",
    ]
    for row in regime_summary.itertuples(index=False):
        lines.append(
            "- "
            f"{row.opportunity_regime}: role={row.next_gate_role}, "
            f"cases={row.case_count}, positives={row.positive_count}, "
            f"controls={row.control_count}, ids={row.case_ids}"
        )
    lines.extend(["", "## Case Rows"])
    for row in case_rows.itertuples(index=False):
        lines.append(
            "- "
            f"{row.case_id} ({row.evidence_panel}, {row.panel_role}): "
            f"{row.opportunity_regime}; role={row.next_gate_role}; "
            f"pair_share={row.baseline_pair_coassigned_run_share:.3f}, "
            f"separated={_fmt_int(row.separated_endpoint_count)}, "
            f"coassigned={_fmt_int(row.coassigned_endpoint_count)}, "
            f"eligible={_fmt_int(row.bridge_handle_eligible_source_count)}, "
            f"bridge_robust={_fmt_int(row.bridge_handle_robust_pair_coassignment_count)}, "
            f"pair_only_robust={_fmt_int(row.pair_relation_only_robust_pair_coassignment_count)}"
        )
    lines.extend(["", "## Next Gate Design"])
    for row in next_gate_design.itertuples(index=False):
        lines.append(
            "- "
            f"{row.design_step} {row.step_name}: {row.purpose} "
            f"Blocked: {row.blocked_action}."
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "G4.8A is a design and classification artifact. It should be used "
                "to prevent the next step from becoming a selector sweep: the next "
                "new runs should test predeclared opportunity-regime cells."
            ),
            "",
        ]
    )
    (output_dir / REPORT_MD).write_text("\n".join(lines), encoding="utf-8")


def _fmt_int(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return str(int(value))


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    g4_3_dir = Path(args.g4_3_dir)
    g4_6_dir = Path(args.g4_6_dir)
    g4_7_dir = Path(args.g4_7_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_rows = _combined_case_rows(g4_3_dir, g4_6_dir, g4_7_dir)
    regime_summary = _regime_summary(case_rows)
    next_gate_design = _next_gate_design()
    _write_csv(case_rows, output_dir / CASE_ROWS_CSV)
    _write_csv(regime_summary, output_dir / REGIME_SUMMARY_CSV)
    _write_csv(next_gate_design, output_dir / NEXT_GATE_DESIGN_CSV)
    summary = _summary(
        output_dir=output_dir,
        case_rows=case_rows,
        regime_summary=regime_summary,
    )
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "variable_pair_synthetic_g4_8_opportunity_regime_design_config.v1",
        "g4_3_dir": str(g4_3_dir),
        "g4_6_dir": str(g4_6_dir),
        "g4_7_dir": str(g4_7_dir),
        "output_dir": str(output_dir),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        output_dir=output_dir,
        summary=summary,
        regime_summary=regime_summary,
        case_rows=case_rows,
        next_gate_design=next_gate_design,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g4-3-dir", type=Path, default=DEFAULT_G4_3_DIR)
    parser.add_argument("--g4-6-dir", type=Path, default=DEFAULT_G4_6_ACCOUNTING_DIR)
    parser.add_argument("--g4-7-dir", type=Path, default=DEFAULT_G4_7_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
