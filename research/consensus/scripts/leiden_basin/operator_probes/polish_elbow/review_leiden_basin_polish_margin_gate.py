#!/usr/bin/env python3
"""Review W4 polish support-margin bands across the current route-gate panel.

This diagnostic sits between route execution and any possible wall-promotion
rule change. It does not alter `wall_claim_gate_status`. It only records how
far each post-polish assignment is from the support-local target threshold so
route-order-sensitive rows can be separated into boundary-sensitive versus
harder support-loss cases.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
SCRIPT_ROOT = REPO_ROOT / "research/consensus/scripts"
BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_GATE_PANEL_DIR = (
    BASE_RESULT_DIR / "leiden_basin_route_gate_panel_combined_after_clean_distinct_20260528"
)
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_polish_margin_gate_review_20260528"

RUNNER_SOURCES = (
    (
        "initial_replicate",
        BASE_RESULT_DIR / "leiden_basin_uniform_wall_probe_runner_replicate_schedule_20260528",
    ),
    (
        "c0_schedule_debug",
        BASE_RESULT_DIR / "leiden_basin_uniform_wall_probe_runner_schedule_debug_20260528",
    ),
    (
        "expanded_controls",
        BASE_RESULT_DIR / "leiden_basin_uniform_wall_probe_runner_expanded_controls_20260528",
    ),
    (
        "clean_distinct_after_gap_fill",
        BASE_RESULT_DIR / "leiden_basin_uniform_wall_probe_runner_clean_distinct_after_gap_fill_20260528",
    ),
)

PANEL_SUMMARY_CSV = "uniform_route_schedule_claim_panel_summary.csv"
ROUTE_LABEL_CSV = "uniform_route_label_rows.csv"
POLISH_REVERSION_CSV = "uniform_polish_reversion_rows.csv"
OBJECTIVE_WALL_CSV = "uniform_objective_wall_rows.csv"

SCHEDULE_ROWS_CSV = "polish_margin_schedule_rows.csv"
PAIR_GATE_ROWS_CSV = "polish_margin_pair_gate_rows.csv"
SUMMARY_JSON = "polish_margin_gate_review_summary.json"
REPORT_MD = "polish_margin_gate_review_report.md"
CONFIG_JSON = "polish_margin_gate_review_config.json"

ENDPOINT_TAU = 0.02
SAME_SUPPORT_MAX = 0.5
SUPPORT_MARGIN_BAND = 0.05


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if pd.isna(value):
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _objective_summary(objective: pd.DataFrame) -> pd.DataFrame:
    if objective.empty:
        return pd.DataFrame(
            columns=[
                "panel_pair_id",
                "route_schedule",
                "wall_step_count",
                "max_objective_debt",
                "max_objective_recovery",
            ]
        )
    return (
        objective.groupby(["panel_pair_id", "route_schedule"], dropna=False)
        .agg(
            wall_step_count=("wall_step_flag", "sum"),
            max_objective_debt=("objective_debt_from_start", "max"),
            max_objective_recovery=("objective_recovery_from_min", "max"),
        )
        .reset_index()
    )


def _load_runner_schedule_rows(label: str, runner_dir: Path) -> pd.DataFrame:
    labels = _read_csv(runner_dir / ROUTE_LABEL_CSV)
    polish = _read_csv(runner_dir / POLISH_REVERSION_CSV)
    objective = _objective_summary(_read_csv(runner_dir / OBJECTIVE_WALL_CSV))
    if labels.empty or polish.empty:
        return pd.DataFrame()

    label_cols = [
        "panel_pair_id",
        "route_schedule",
        "route_label",
        "wall_assignment_status",
        "support_assignment_status",
        "target_endpoint_distance_final",
        "target_support_distance_final",
        "direct_route_row_count",
    ]
    polish_cols = [
        "panel_pair_id",
        "route_schedule",
        "post_polish_endpoint_assignment",
        "reversion_status",
        "post_polish_support_distance_to_source",
        "post_polish_support_distance_to_target",
        "post_polish_endpoint_distance_to_source",
        "post_polish_endpoint_distance_to_target",
        "post_polish_support_node_count",
    ]
    rows = labels[label_cols].merge(
        polish[polish_cols],
        on=["panel_pair_id", "route_schedule"],
        how="left",
    )
    rows = rows.merge(objective, on=["panel_pair_id", "route_schedule"], how="left")
    rows["runner_source"] = label
    rows["runner_dir"] = _rel(runner_dir)
    return rows


def _margin_band(row: pd.Series) -> str:
    assignment = str(row.get("post_polish_endpoint_assignment", ""))
    support_margin = _safe_float(row.get("post_target_support_margin"))
    endpoint_margin = _safe_float(row.get("post_target_endpoint_margin"))
    if assignment == "target_endpoint":
        if math.isfinite(support_margin) and support_margin <= -SUPPORT_MARGIN_BAND:
            return "target_stable_margin"
        return "target_near_support_boundary"
    if assignment == "source_endpoint":
        return "source_reversion"
    if math.isfinite(endpoint_margin) and endpoint_margin > 0:
        return "endpoint_assignment_loss"
    if math.isfinite(support_margin) and support_margin > SUPPORT_MARGIN_BAND:
        return "support_hard_loss"
    if math.isfinite(support_margin) and support_margin > 0:
        return "support_boundary_loss"
    return "other_or_ambiguous_unclassified"


def _pair_margin_gate(group: pd.DataFrame) -> tuple[str, str]:
    bands = set(group["polish_margin_band"].astype(str))
    route_status = str(group["route_order_sensitivity_status"].iloc[0])
    wall_status = str(group["wall_claim_gate_status"].iloc[0])
    relation = str(group["calibrated_relation"].iloc[0])
    if wall_status == "passes_schedule_invariance_distinct_partial_wall_evidence":
        if bands <= {"target_stable_margin", "target_near_support_boundary"}:
            return (
                "keep_partial_wall_gate_with_margin_context",
                "schedule-invariant distinct gate; margin context is diagnostic only",
            )
    if wall_status == "stable_route_evidence_basin_relation_ambiguous_no_supported_wall_claim":
        return (
            "relation_blocked_keep_as_definition_evidence",
            "route is stable but basin relation is ambiguous",
        )
    if wall_status == "stable_control_no_wall_claim":
        return ("keep_as_same_control", "same-control row remains no-wall")
    if route_status == "route_order_sensitive":
        if "support_boundary_loss" in bands and "support_hard_loss" not in bands:
            return (
                "boundary_sensitive_route_evidence_hold",
                "schedule sensitivity is caused by a near-threshold post-polish support loss",
            )
        if "support_hard_loss" in bands:
            return (
                "support_loss_no_wall_hold",
                "at least one schedule has post-polish support loss beyond the margin band",
            )
        if "endpoint_assignment_loss" in bands or "source_reversion" in bands:
            return (
                "non_support_assignment_loss_hold",
                "schedule sensitivity is not explained by support-margin alone",
            )
        return (
            "route_order_sensitive_unclassified_hold",
            "route labels differ across schedules",
        )
    if relation != "distinct_support_local":
        return ("relation_or_control_hold", "non-distinct relation remains outside wall promotion")
    return ("manual_review_hold", "margin gate could not classify this pair")


def _schedule_rows(gate_panel_dir: Path) -> pd.DataFrame:
    panel = _read_csv(gate_panel_dir / PANEL_SUMMARY_CSV)
    if panel.empty:
        raise FileNotFoundError(gate_panel_dir / PANEL_SUMMARY_CSV)
    frames = [
        _load_runner_schedule_rows(label, runner_dir)
        for label, runner_dir in RUNNER_SOURCES
    ]
    rows = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if rows.empty:
        raise FileNotFoundError("no route label/polish rows found")
    rows = rows.drop_duplicates(
        subset=["panel_pair_id", "route_schedule", "runner_source"],
        keep="first",
    )
    metadata_cols = [
        "panel_pair_id",
        "field",
        "case_id",
        "panel_role",
        "calibrated_relation",
        "support_distance_max",
        "source_output",
        "route_order_sensitivity_status",
        "wall_claim_gate_status",
    ]
    rows = rows.merge(panel[metadata_cols], on="panel_pair_id", how="inner")
    rows["post_target_support_margin"] = (
        pd.to_numeric(rows["post_polish_support_distance_to_target"], errors="coerce")
        - SAME_SUPPORT_MAX
    )
    rows["post_target_endpoint_margin"] = (
        pd.to_numeric(rows["post_polish_endpoint_distance_to_target"], errors="coerce")
        - ENDPOINT_TAU
    )
    rows["pre_label_target_like"] = (
        pd.to_numeric(rows["target_endpoint_distance_final"], errors="coerce").le(ENDPOINT_TAU)
        & pd.to_numeric(rows["target_support_distance_final"], errors="coerce").le(SAME_SUPPORT_MAX)
    )
    rows["post_polish_target_like"] = rows["post_polish_endpoint_assignment"].astype(str).eq(
        "target_endpoint"
    )
    rows["polish_margin_band"] = rows.apply(_margin_band, axis=1)
    public_cols = [
        "panel_pair_id",
        "field",
        "case_id",
        "panel_role",
        "calibrated_relation",
        "support_distance_max",
        "source_output",
        "runner_source",
        "route_schedule",
        "route_order_sensitivity_status",
        "wall_claim_gate_status",
        "route_label",
        "wall_assignment_status",
        "pre_label_target_like",
        "post_polish_target_like",
        "post_polish_endpoint_assignment",
        "polish_margin_band",
        "post_polish_support_distance_to_target",
        "post_target_support_margin",
        "post_polish_endpoint_distance_to_target",
        "post_target_endpoint_margin",
        "wall_step_count",
        "max_objective_debt",
        "max_objective_recovery",
        "direct_route_row_count",
    ]
    return rows[public_cols].sort_values(["field", "panel_pair_id", "route_schedule"])


def _pair_rows(schedule_rows: pd.DataFrame) -> pd.DataFrame:
    pair_rows: list[dict[str, Any]] = []
    for pair_id, group in schedule_rows.groupby("panel_pair_id", dropna=False):
        margin_gate, note = _pair_margin_gate(group)
        pair_rows.append(
            {
                "panel_pair_id": pair_id,
                "field": str(group["field"].iloc[0]),
                "case_id": str(group["case_id"].iloc[0]),
                "panel_role": str(group["panel_role"].iloc[0]),
                "calibrated_relation": str(group["calibrated_relation"].iloc[0]),
                "support_distance_max": _safe_float(group["support_distance_max"].iloc[0]),
                "source_output": str(group["source_output"].iloc[0]),
                "schedule_count": int(len(group)),
                "route_order_sensitivity_status": str(
                    group["route_order_sensitivity_status"].iloc[0]
                ),
                "wall_claim_gate_status": str(group["wall_claim_gate_status"].iloc[0]),
                "post_target_schedule_count": int(group["post_polish_target_like"].sum()),
                "post_non_target_schedule_count": int(
                    len(group) - group["post_polish_target_like"].sum()
                ),
                "polish_margin_bands": "|".join(sorted(set(group["polish_margin_band"]))),
                "post_target_support_margin_min": float(group["post_target_support_margin"].min()),
                "post_target_support_margin_max": float(group["post_target_support_margin"].max()),
                "post_target_support_distance_min": float(
                    group["post_polish_support_distance_to_target"].min()
                ),
                "post_target_support_distance_max": float(
                    group["post_polish_support_distance_to_target"].max()
                ),
                "margin_gate_status": margin_gate,
                "margin_gate_note": note,
                "claim_boundary": (
                    "Margin gate is diagnostic only; it does not alter wall_claim_gate_status."
                ),
            }
        )
    return pd.DataFrame(pair_rows).sort_values(["field", "panel_pair_id"])


def _write_report(path: Path, summary: dict[str, Any], pair_rows: pd.DataFrame) -> None:
    lines = [
        "# Leiden Basin W4 Polish Margin Gate Review",
        "",
        "Status: W4 polish support-margin bands reviewed",
        "Date: 2026-05-28",
        "",
        "This artifact does not change existing wall-claim gates. It records whether route-order sensitivity is near the post-polish support threshold or farther away.",
        "",
        "## Pair Margin Gate",
        "",
        "| pair_id | gate_status | margin_gate | support_margin_max | bands |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for _, row in pair_rows.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["panel_pair_id"]),
                    str(row["wall_claim_gate_status"]),
                    str(row["margin_gate_status"]),
                    f"{row['post_target_support_margin_max']:.6f}",
                    str(row["polish_margin_bands"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- boundary-sensitive route holds: {summary['boundary_sensitive_route_hold_count']}",
            f"- support-loss no-wall holds: {summary['support_loss_no_wall_hold_count']}",
            f"- partial-wall gates kept with margin context: {summary['partial_wall_gate_margin_context_count']}",
            "- Keep existing wall promotion unchanged until this margin rule is validated on more controls.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(gate_panel_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_rows = _schedule_rows(gate_panel_dir)
    pair_rows = _pair_rows(schedule_rows)
    _write_csv(schedule_rows, output_dir / SCHEDULE_ROWS_CSV)
    _write_csv(pair_rows, output_dir / PAIR_GATE_ROWS_CSV)

    status_counts = pair_rows["margin_gate_status"].value_counts().to_dict()
    summary = {
        "status": "polish_margin_gate_review_prepared",
        "date": "2026-05-28",
        "script": _rel(Path(__file__)),
        "gate_panel_dir": _rel(gate_panel_dir),
        "output_dir": _rel(output_dir),
        "same_support_max": SAME_SUPPORT_MAX,
        "endpoint_tau": ENDPOINT_TAU,
        "support_margin_band": SUPPORT_MARGIN_BAND,
        "schedule_row_count": int(len(schedule_rows)),
        "pair_count": int(len(pair_rows)),
        "margin_gate_status_counts": status_counts,
        "boundary_sensitive_route_hold_count": int(
            status_counts.get("boundary_sensitive_route_evidence_hold", 0)
        ),
        "support_loss_no_wall_hold_count": int(
            status_counts.get("support_loss_no_wall_hold", 0)
        ),
        "partial_wall_gate_margin_context_count": int(
            status_counts.get("keep_partial_wall_gate_with_margin_context", 0)
        ),
        "paths": {
            "schedule_rows": _rel(output_dir / SCHEDULE_ROWS_CSV),
            "pair_gate_rows": _rel(output_dir / PAIR_GATE_ROWS_CSV),
            "summary": _rel(output_dir / SUMMARY_JSON),
            "report": _rel(output_dir / REPORT_MD),
        },
        "decision": (
            "Keep existing wall-claim gates unchanged. Treat near-threshold "
            "post-polish support losses as boundary-sensitive route evidence "
            "only after additional validation."
        ),
        "claim_boundary": (
            "Diagnostic W4 margin review only; no basin-quality, cost, or "
            "directed-search claim is made."
        ),
    }
    (output_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / CONFIG_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_report(output_dir / REPORT_MD, summary, pair_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-panel-dir", type=Path, default=DEFAULT_GATE_PANEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.gate_panel_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
