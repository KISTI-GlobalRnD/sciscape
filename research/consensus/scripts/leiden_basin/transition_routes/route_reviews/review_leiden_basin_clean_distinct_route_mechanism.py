#!/usr/bin/env python3
"""Review clean distinct route-gate mechanisms after the gap-fill run.

This artifact compares the two schedule-stable field26 pairs against the two
schedule-sensitive field30 pairs. It stays inside the W1-W6 wall protocol: the
review explains route/schedule/polish behavior and does not rank basin quality
or propose a directed-search operator.
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
DEFAULT_RUNNER_DIR = (
    BASE_RESULT_DIR / "leiden_basin_uniform_wall_probe_runner_clean_distinct_after_gap_fill_20260528"
)
DEFAULT_SUBSET_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_uniform_wall_probe_subset_clean_distinct_after_gap_fill_20260528"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_clean_distinct_route_mechanism_review_20260528"
)

DIRECT_ROUTE_CSV = "uniform_direct_pair_route_rows.csv"
OBJECTIVE_WALL_CSV = "uniform_objective_wall_rows.csv"
SUPPORT_MOVEMENT_CSV = "uniform_support_movement_rows.csv"
POLISH_REVERSION_CSV = "uniform_polish_reversion_rows.csv"
ROUTE_LABEL_CSV = "uniform_route_label_rows.csv"
ROUTE_CLAIM_CSV = "uniform_route_schedule_claim_rows.csv"
SUBSET_CSV = "uniform_wall_probe_subset.csv"

SCHEDULE_ROWS_CSV = "clean_distinct_route_mechanism_schedule_rows.csv"
PAIR_SUMMARY_CSV = "clean_distinct_route_mechanism_pair_summary.csv"
FIELD_CONTRAST_CSV = "clean_distinct_route_mechanism_field_contrast.csv"
SUMMARY_JSON = "clean_distinct_route_mechanism_review_summary.json"
REPORT_MD = "clean_distinct_route_mechanism_review_report.md"
CONFIG_JSON = "clean_distinct_route_mechanism_review_config.json"

ENDPOINT_TAU = 0.02
SAME_SUPPORT_MAX = 0.5


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


def _last_by_schedule(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["panel_pair_id", "route_schedule", *columns])
    ordered = frame.sort_values(["panel_pair_id", "route_schedule", "step_index"])
    return ordered.groupby(["panel_pair_id", "route_schedule"], as_index=False).tail(1)[
        ["panel_pair_id", "route_schedule", *columns]
    ]


def _objective_summary(objective: pd.DataFrame) -> pd.DataFrame:
    if objective.empty:
        return pd.DataFrame()
    return (
        objective.groupby(["panel_pair_id", "route_schedule"], dropna=False)
        .agg(
            objective_row_count=("step_index", "count"),
            wall_step_count=("wall_step_flag", "sum"),
            max_objective_debt=("objective_debt_from_start", "max"),
            max_objective_recovery=("objective_recovery_from_min", "max"),
            min_objective=("objective_value", "min"),
            final_objective=("objective_value", "last"),
        )
        .reset_index()
    )


def _route_summary(route: pd.DataFrame) -> pd.DataFrame:
    if route.empty:
        return pd.DataFrame()
    route = route.copy()
    target_steps = route[route["step_index"].gt(0)].copy()
    total_edits = (
        target_steps.groupby(["panel_pair_id", "route_schedule"], dropna=False)
        .agg(total_edited_node_count=("edited_node_count", "sum"))
        .reset_index()
    )
    last = _last_by_schedule(
        route,
        [
            "step_index",
            "route_scope_node_count",
            "target_group_count",
            "route_completion_status",
        ],
    ).rename(columns={"step_index": "final_step_index"})
    return last.merge(total_edits, on=["panel_pair_id", "route_schedule"], how="left")


def _schedule_rows(runner_dir: Path, subset_dir: Path) -> pd.DataFrame:
    claims = _read_csv(runner_dir / ROUTE_CLAIM_CSV)
    labels = _read_csv(runner_dir / ROUTE_LABEL_CSV)
    polish = _read_csv(runner_dir / POLISH_REVERSION_CSV)
    movement = _read_csv(runner_dir / SUPPORT_MOVEMENT_CSV)
    objective = _read_csv(runner_dir / OBJECTIVE_WALL_CSV)
    route = _read_csv(runner_dir / DIRECT_ROUTE_CSV)
    subset = _read_csv(subset_dir / SUBSET_CSV)
    if claims.empty:
        raise FileNotFoundError(runner_dir / ROUTE_CLAIM_CSV)
    if labels.empty:
        raise FileNotFoundError(runner_dir / ROUTE_LABEL_CSV)

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
    rows = labels[label_cols].copy()

    subset_cols = [
        "panel_pair_id",
        "field",
        "case_id",
        "method",
        "calibrated_relation",
        "support_distance_max",
    ]
    rows = rows.merge(subset[subset_cols], on="panel_pair_id", how="left")

    claim_cols = [
        "panel_pair_id",
        "route_order_sensitivity_status",
        "wall_claim_gate_status",
        "objective_wall_step_count_min",
        "objective_wall_step_count_max",
    ]
    rows = rows.merge(claims[claim_cols], on="panel_pair_id", how="left")

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
    rows = rows.merge(polish[polish_cols], on=["panel_pair_id", "route_schedule"], how="left")
    rows = rows.merge(
        _last_by_schedule(
            movement,
            [
                "support_node_count",
                "support_distance_to_source",
                "support_distance_to_target",
                "endpoint_distance_to_source",
                "endpoint_distance_to_target",
            ],
        ).rename(
            columns={
                "support_node_count": "pre_polish_support_node_count",
                "support_distance_to_source": "pre_polish_support_distance_to_source",
                "support_distance_to_target": "pre_polish_support_distance_to_target",
                "endpoint_distance_to_source": "pre_polish_endpoint_distance_to_source",
                "endpoint_distance_to_target": "pre_polish_endpoint_distance_to_target",
            }
        ),
        on=["panel_pair_id", "route_schedule"],
        how="left",
    )
    rows = rows.merge(_objective_summary(objective), on=["panel_pair_id", "route_schedule"], how="left")
    rows = rows.merge(_route_summary(route), on=["panel_pair_id", "route_schedule"], how="left")

    rows["pre_polish_target_like"] = (
        pd.to_numeric(rows["pre_polish_endpoint_distance_to_target"], errors="coerce").le(ENDPOINT_TAU)
        & pd.to_numeric(rows["pre_polish_support_distance_to_target"], errors="coerce").le(SAME_SUPPORT_MAX)
    )
    rows["post_polish_target_like"] = rows["post_polish_endpoint_assignment"].astype(str).eq(
        "target_endpoint"
    )
    rows["post_polish_support_margin_to_target_threshold"] = (
        pd.to_numeric(rows["post_polish_support_distance_to_target"], errors="coerce")
        - SAME_SUPPORT_MAX
    )
    rows["post_polish_endpoint_margin_to_target_threshold"] = (
        pd.to_numeric(rows["post_polish_endpoint_distance_to_target"], errors="coerce")
        - ENDPOINT_TAU
    )
    rows["polish_assignment_loss"] = rows["pre_polish_target_like"] & ~rows[
        "post_polish_target_like"
    ]
    rows["mechanism_flag"] = rows.apply(_schedule_mechanism_flag, axis=1)
    public_cols = [
        "panel_pair_id",
        "field",
        "method",
        "route_schedule",
        "route_order_sensitivity_status",
        "wall_claim_gate_status",
        "route_label",
        "wall_assignment_status",
        "pre_polish_target_like",
        "post_polish_target_like",
        "post_polish_endpoint_assignment",
        "polish_assignment_loss",
        "mechanism_flag",
        "final_step_index",
        "target_group_count",
        "route_scope_node_count",
        "total_edited_node_count",
        "pre_polish_support_distance_to_target",
        "post_polish_support_distance_to_target",
        "post_polish_support_margin_to_target_threshold",
        "pre_polish_endpoint_distance_to_target",
        "post_polish_endpoint_distance_to_target",
        "post_polish_endpoint_margin_to_target_threshold",
        "wall_step_count",
        "max_objective_debt",
        "max_objective_recovery",
        "support_distance_max",
    ]
    return rows[public_cols].sort_values(["field", "panel_pair_id", "route_schedule"])


def _schedule_mechanism_flag(row: pd.Series) -> str:
    if bool(row.get("post_polish_target_like", False)):
        return "post_polish_target_stable"
    if bool(row.get("polish_assignment_loss", False)):
        target_support_margin = _safe_float(row.get("post_polish_support_margin_to_target_threshold"))
        target_endpoint_margin = _safe_float(row.get("post_polish_endpoint_margin_to_target_threshold"))
        if math.isfinite(target_support_margin) and target_support_margin > 0:
            return "post_polish_support_threshold_loss"
        if math.isfinite(target_endpoint_margin) and target_endpoint_margin > 0:
            return "post_polish_endpoint_threshold_loss"
        return "post_polish_assignment_loss_other"
    if not bool(row.get("pre_polish_target_like", False)):
        return "route_did_not_reach_target_like_state"
    return "unclassified_schedule_mechanism"


def _pair_summary(schedule_rows: pd.DataFrame) -> pd.DataFrame:
    grouped = schedule_rows.groupby("panel_pair_id", dropna=False)
    rows: list[dict[str, Any]] = []
    for pair_id, group in grouped:
        mechanism_flags = sorted(set(group["mechanism_flag"].astype(str)))
        post_target_count = int(group["post_polish_target_like"].sum())
        schedule_count = int(len(group))
        sensitivity = str(group["route_order_sensitivity_status"].iloc[0])
        if sensitivity == "route_order_stable" and post_target_count == schedule_count:
            interpretation = "schedule_invariant_target_polish"
        elif (
            sensitivity == "route_order_sensitive"
            and "post_polish_support_threshold_loss" in mechanism_flags
        ):
            interpretation = "schedule_dependent_post_polish_support_assignment"
        else:
            interpretation = "mixed_or_unclassified_route_mechanism"
        rows.append(
            {
                "panel_pair_id": pair_id,
                "field": str(group["field"].iloc[0]),
                "method": str(group["method"].iloc[0]),
                "route_order_sensitivity_status": sensitivity,
                "wall_claim_gate_status": str(group["wall_claim_gate_status"].iloc[0]),
                "schedule_count": schedule_count,
                "post_target_schedule_count": post_target_count,
                "post_other_schedule_count": int(schedule_count - post_target_count),
                "polish_assignment_loss_count": int(group["polish_assignment_loss"].sum()),
                "mechanism_flags": "|".join(mechanism_flags),
                "mechanism_interpretation": interpretation,
                "target_group_count_min": int(group["target_group_count"].min()),
                "target_group_count_max": int(group["target_group_count"].max()),
                "final_step_count_min": int(group["final_step_index"].min()),
                "final_step_count_max": int(group["final_step_index"].max()),
                "post_target_support_distance_min": float(
                    group["post_polish_support_distance_to_target"].min()
                ),
                "post_target_support_distance_max": float(
                    group["post_polish_support_distance_to_target"].max()
                ),
                "post_target_support_margin_max": float(
                    group["post_polish_support_margin_to_target_threshold"].max()
                ),
                "wall_step_count_min": int(group["wall_step_count"].min()),
                "wall_step_count_max": int(group["wall_step_count"].max()),
                "max_objective_debt_max": float(group["max_objective_debt"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["field", "panel_pair_id"])


def _field_contrast(pair_summary: pd.DataFrame) -> pd.DataFrame:
    if pair_summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (field, method), group in pair_summary.groupby(["field", "method"], dropna=False):
        rows.append(
            {
                "field": field,
                "method": method,
                "pair_count": int(len(group)),
                "schedule_stable_pair_count": int(
                    group["route_order_sensitivity_status"].eq("route_order_stable").sum()
                ),
                "schedule_sensitive_pair_count": int(
                    group["route_order_sensitivity_status"].eq("route_order_sensitive").sum()
                ),
                "partial_wall_gate_count": int(
                    group["wall_claim_gate_status"].eq(
                        "passes_schedule_invariance_distinct_partial_wall_evidence"
                    ).sum()
                ),
                "support_assignment_loss_pair_count": int(
                    group["mechanism_interpretation"].eq(
                        "schedule_dependent_post_polish_support_assignment"
                    ).sum()
                ),
                "post_target_support_distance_max": float(
                    group["post_target_support_distance_max"].max()
                ),
                "interpretations": "|".join(
                    sorted(set(group["mechanism_interpretation"].astype(str)))
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["field", "method"])


def _write_report(
    path: Path,
    summary: dict[str, Any],
    pair_summary: pd.DataFrame,
    field_contrast: pd.DataFrame,
) -> None:
    lines = [
        "# Leiden Basin Clean Distinct Route Mechanism Review",
        "",
        "Status: route mechanism review prepared",
        "Date: 2026-05-28",
        "",
        "This artifact reviews why clean distinct pairs split into schedule-stable and schedule-sensitive route gates. It does not rank basin quality or introduce a directed-search claim.",
        "",
        "## Field Contrast",
        "",
        "| field | method | pairs | stable | sensitive | interpretation |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for _, row in field_contrast.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["field"]),
                    str(row["method"]),
                    str(row["pair_count"]),
                    str(row["schedule_stable_pair_count"]),
                    str(row["schedule_sensitive_pair_count"]),
                    str(row["interpretations"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Pair Mechanisms",
            "",
            "| pair_id | sensitivity | gate | post-target schedules | mechanism | support max |",
            "| --- | --- | --- | ---: | --- | ---: |",
        ]
    )
    for _, row in pair_summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["panel_pair_id"]),
                    str(row["route_order_sensitivity_status"]),
                    str(row["wall_claim_gate_status"]),
                    f"{row['post_target_schedule_count']}/{row['schedule_count']}",
                    str(row["mechanism_interpretation"]),
                    f"{row['post_target_support_distance_max']:.6f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- stable partial-wall pairs: {summary['stable_partial_wall_pair_count']}",
            f"- schedule-sensitive pairs: {summary['schedule_sensitive_pair_count']}",
            "- The field30 failures are post-polish support-assignment losses, not direct-route target reach failures.",
            "- The next method question is whether route labels should inspect polish support-threshold margins before any broader route batch.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(runner_dir: Path, subset_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_rows = _schedule_rows(runner_dir, subset_dir)
    pair_summary = _pair_summary(schedule_rows)
    field_contrast = _field_contrast(pair_summary)

    _write_csv(schedule_rows, output_dir / SCHEDULE_ROWS_CSV)
    _write_csv(pair_summary, output_dir / PAIR_SUMMARY_CSV)
    _write_csv(field_contrast, output_dir / FIELD_CONTRAST_CSV)

    summary = {
        "status": "clean_distinct_route_mechanism_review_prepared",
        "date": "2026-05-28",
        "script": _rel(Path(__file__)),
        "runner_dir": _rel(runner_dir),
        "subset_dir": _rel(subset_dir),
        "output_dir": _rel(output_dir),
        "schedule_row_count": int(len(schedule_rows)),
        "pair_count": int(len(pair_summary)),
        "stable_partial_wall_pair_count": int(
            pair_summary["wall_claim_gate_status"]
            .eq("passes_schedule_invariance_distinct_partial_wall_evidence")
            .sum()
        ),
        "schedule_sensitive_pair_count": int(
            pair_summary["route_order_sensitivity_status"].eq("route_order_sensitive").sum()
        ),
        "post_polish_support_assignment_loss_pair_count": int(
            pair_summary["mechanism_interpretation"]
            .eq("schedule_dependent_post_polish_support_assignment")
            .sum()
        ),
        "field_contrast": field_contrast.to_dict("records"),
        "paths": {
            "schedule_rows": _rel(output_dir / SCHEDULE_ROWS_CSV),
            "pair_summary": _rel(output_dir / PAIR_SUMMARY_CSV),
            "field_contrast": _rel(output_dir / FIELD_CONTRAST_CSV),
            "summary": _rel(output_dir / SUMMARY_JSON),
            "report": _rel(output_dir / REPORT_MD),
        },
        "decision": (
            "Review polish support-threshold assignment losses before broadening "
            "route execution or changing wall-promotion rules."
        ),
        "claim_boundary": (
            "Route-mechanism review only; no basin-quality, cost, or directed-search "
            "claim is made."
        ),
    }
    (output_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / CONFIG_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_report(output_dir / REPORT_MD, summary, pair_summary, field_contrast)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-dir", type=Path, default=DEFAULT_RUNNER_DIR)
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.runner_dir, args.subset_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
