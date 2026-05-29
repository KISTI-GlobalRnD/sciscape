#!/usr/bin/env python3
"""Review the 4-pair Methodology v0 margin-validation panel.

This review combines the original three route schedules from the W4 polish
margin gate with a held-out `target_label_desc` schedule. It validates margin
classes only. It does not change wall-promotion rules, run basin-quality
evaluation, or make directed-search claims.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
SCRIPT_ROOT = REPO_ROOT / "research/consensus/scripts"
_SCRIPT_PATHS = [REPO_ROOT, SCRIPT_ROOT]
_SCRIPT_PATHS.extend(path for path in SCRIPT_ROOT.rglob("*") if path.is_dir())
for _script_path in reversed(_SCRIPT_PATHS):
    _script_path_str = str(_script_path)
    if _script_path_str not in sys.path:
        sys.path.insert(0, _script_path_str)


import pandas as pd

BASE_RESULT_DIR = REPO_ROOT / "research/consensus/results/adaptive_refinement"
DEFAULT_METHODOLOGY_DIR = BASE_RESULT_DIR / "leiden_basin_methodology_v0_margin_validation_20260528"
DEFAULT_MARGIN_REVIEW_DIR = BASE_RESULT_DIR / "leiden_basin_polish_margin_gate_review_20260528"
DEFAULT_HELDOUT_RUNNER_DIRS = (
    BASE_RESULT_DIR / "leiden_basin_margin_validation_runner_initial_target_label_desc_20260529",
    BASE_RESULT_DIR / "leiden_basin_margin_validation_runner_clean_target_label_desc_20260529",
)
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_margin_validation_panel_review_20260529"

VALIDATION_PANEL_CSV = "margin_validation_panel.csv"
PRIOR_SCHEDULE_ROWS_CSV = "polish_margin_schedule_rows.csv"
ROUTE_LABEL_CSV = "uniform_route_label_rows.csv"
POLISH_REVERSION_CSV = "uniform_polish_reversion_rows.csv"
OBJECTIVE_WALL_CSV = "uniform_objective_wall_rows.csv"

SCHEDULE_ROWS_CSV = "margin_validation_schedule_rows.csv"
PAIR_RESULTS_CSV = "margin_validation_pair_results.csv"
SUMMARY_JSON = "margin_validation_panel_review_summary.json"
REPORT_MD = "margin_validation_panel_review_report.md"
CONFIG_JSON = "margin_validation_panel_review_config.json"

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

def _load_heldout_runner_rows(label: str, runner_dir: Path, panel: pd.DataFrame) -> pd.DataFrame:
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
    rows = rows.merge(
        panel[
            [
                "panel_pair_id",
                "field",
                "case_id",
                "panel_role",
                "calibrated_relation",
                "methodology_v0_state",
                "validation_role",
            ]
        ],
        on="panel_pair_id",
        how="inner",
    )
    rows["source_phase"] = "heldout_validation"
    rows["runner_source"] = label
    rows["runner_dir"] = _rel(runner_dir)
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
    return rows

def _schedule_rows(
    methodology_dir: Path,
    margin_review_dir: Path,
    heldout_runner_dirs: tuple[Path, ...],
) -> pd.DataFrame:
    panel = _read_csv(methodology_dir / VALIDATION_PANEL_CSV)
    if panel.empty:
        raise FileNotFoundError(methodology_dir / VALIDATION_PANEL_CSV)
    prior = _read_csv(margin_review_dir / PRIOR_SCHEDULE_ROWS_CSV)
    if prior.empty:
        raise FileNotFoundError(margin_review_dir / PRIOR_SCHEDULE_ROWS_CSV)
    panel_ids = set(panel["panel_pair_id"].astype(str))
    prior = prior[prior["panel_pair_id"].astype(str).isin(panel_ids)].copy()
    prior = prior.merge(
        panel[
            [
                "panel_pair_id",
                "methodology_v0_state",
                "validation_role",
            ]
        ],
        on="panel_pair_id",
        how="left",
    )
    prior["source_phase"] = "methodology_v0_prior"
    prior["runner_dir"] = prior.get("runner_dir", "")
    heldout_frames = [
        _load_heldout_runner_rows(path.name, path, panel)
        for path in heldout_runner_dirs
    ]
    frames = [prior] + [frame for frame in heldout_frames if not frame.empty]
    rows = pd.concat(frames, ignore_index=True, sort=False)
    public_cols = [
        "source_phase",
        "runner_source",
        "runner_dir",
        "panel_pair_id",
        "field",
        "case_id",
        "panel_role",
        "calibrated_relation",
        "methodology_v0_state",
        "validation_role",
        "route_schedule",
        "route_label",
        "wall_assignment_status",
        "support_assignment_status",
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
    for column in public_cols:
        if column not in rows:
            rows[column] = ""
    return rows[public_cols].sort_values(
        ["panel_pair_id", "source_phase", "route_schedule"]
    ).reset_index(drop=True)

def _classify_pair(group: pd.DataFrame) -> tuple[str, str]:
    role = str(group["validation_role"].iloc[0])
    all_bands = set(group["polish_margin_band"].astype(str))
    heldout = group[group["source_phase"].astype(str).eq("heldout_validation")]
    heldout_bands = set(heldout["polish_margin_band"].astype(str))
    heldout_labels = set(heldout["route_label"].astype(str))

    if role == "boundary_sensitive_candidate":
        if "support_hard_loss" in heldout_bands:
            return (
                "boundary_candidate_failed_hard_loss",
                "held-out schedule introduced support-hard-loss; keep no-wall hold",
            )
        if "support_hard_loss" in all_bands:
            return (
                "boundary_candidate_failed_combined_hard_loss",
                "combined schedules include support-hard-loss; keep no-wall hold",
            )
        if "support_boundary_loss" in all_bands:
            return (
                "validated_boundary_sensitive_hold",
                "held-out schedule did not introduce hard support loss; boundary-sensitive class survives",
            )
        return (
            "boundary_candidate_inconclusive_no_boundary_loss",
            "combined schedules do not retain a boundary-loss example",
        )

    if role == "support_loss_contrast":
        if "support_hard_loss" in heldout_bands:
            return (
                "validated_support_loss_contrast",
                "held-out schedule repeats support-hard-loss behavior",
            )
        if "support_hard_loss" in all_bands:
            return (
                "support_loss_contrast_mixed_hold",
                "existing schedules retain hard-loss contrast, but held-out schedule did not repeat it",
            )
        if "direct_route_unassigned" in heldout_labels:
            return (
                "support_loss_contrast_unassigned_without_hard_loss",
                "held-out schedule stays unassigned but not through support-hard-loss",
            )
        return (
            "support_loss_contrast_not_validated",
            "hard-loss contrast was not reproduced",
        )

    return ("not_in_validation_scope", "validation role is outside the 4-pair panel")

def _pair_results(schedule_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair_id, group in schedule_rows.groupby("panel_pair_id", dropna=False):
        status, note = _classify_pair(group)
        heldout = group[group["source_phase"].astype(str).eq("heldout_validation")]
        rows.append(
            {
                "panel_pair_id": pair_id,
                "field": str(group["field"].iloc[0]),
                "case_id": str(group["case_id"].iloc[0]),
                "panel_role": str(group["panel_role"].iloc[0]),
                "calibrated_relation": str(group["calibrated_relation"].iloc[0]),
                "methodology_v0_state": str(group["methodology_v0_state"].iloc[0]),
                "validation_role": str(group["validation_role"].iloc[0]),
                "combined_schedule_count": int(len(group)),
                "heldout_schedule_count": int(len(heldout)),
                "combined_route_labels": "|".join(sorted(set(group["route_label"].astype(str)))),
                "heldout_route_labels": "|".join(sorted(set(heldout["route_label"].astype(str)))),
                "combined_margin_bands": "|".join(
                    sorted(set(group["polish_margin_band"].astype(str)))
                ),
                "heldout_margin_bands": "|".join(
                    sorted(set(heldout["polish_margin_band"].astype(str)))
                ),
                "combined_support_margin_min": float(group["post_target_support_margin"].min()),
                "combined_support_margin_max": float(group["post_target_support_margin"].max()),
                "heldout_support_margin_min": (
                    float(heldout["post_target_support_margin"].min())
                    if not heldout.empty
                    else math.nan
                ),
                "heldout_support_margin_max": (
                    float(heldout["post_target_support_margin"].max())
                    if not heldout.empty
                    else math.nan
                ),
                "validation_status": status,
                "validation_note": note,
                "wall_claim_change": "none",
                "claim_boundary": (
                    "Margin validation only; no wall promotion, basin-quality, "
                    "cost, or directed-search claim is made."
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["validation_role", "field", "panel_pair_id"])

def _write_report(path: Path, summary: dict[str, Any], pair_results: pd.DataFrame) -> None:
    lines = [
        "# Leiden Basin Margin Validation Panel Review",
        "",
        "Status: held-out margin schedule reviewed",
        "Date: 2026-05-29",
        "",
        "This artifact combines the original Methodology v0 schedules with the held-out `target_label_desc` schedule. It validates margin classes only and does not relax wall promotion.",
        "",
        "## Validation Counts",
        "",
        "| validation_status | pairs |",
        "| --- | ---: |",
    ]
    for status, count in sorted(summary["validation_status_counts"].items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "## Pair Results",
            "",
            "| pair_id | role | heldout_margin | combined_margin | status |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in pair_results.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["panel_pair_id"]),
                    str(row["validation_role"]),
                    str(row["heldout_margin_bands"]),
                    str(row["combined_margin_bands"]),
                    str(row["validation_status"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Boundary-sensitive candidates survive the held-out schedule if no support-hard-loss appears.",
            "- Support-loss contrasts are strong only when held-out support-hard-loss repeats.",
            "- Mixed support-loss contrasts stay no-wall holds but should not be used as strong hard-loss validation examples.",
            "- No row changes `wall_claim_gate_status`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run(
    methodology_dir: Path,
    margin_review_dir: Path,
    heldout_runner_dirs: tuple[Path, ...],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_rows = _schedule_rows(methodology_dir, margin_review_dir, heldout_runner_dirs)
    pair_results = _pair_results(schedule_rows)
    _write_csv(schedule_rows, output_dir / SCHEDULE_ROWS_CSV)
    _write_csv(pair_results, output_dir / PAIR_RESULTS_CSV)

    status_counts = pair_results["validation_status"].value_counts().to_dict()
    summary = {
        "status": "margin_validation_panel_review_prepared",
        "date": "2026-05-29",
        "script": _rel(Path(__file__)),
        "methodology_dir": _rel(methodology_dir),
        "margin_review_dir": _rel(margin_review_dir),
        "heldout_runner_dirs": [_rel(path) for path in heldout_runner_dirs],
        "output_dir": _rel(output_dir),
        "schedule_row_count": int(len(schedule_rows)),
        "heldout_schedule_row_count": int(
            schedule_rows["source_phase"].astype(str).eq("heldout_validation").sum()
        ),
        "pair_count": int(len(pair_results)),
        "validation_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "boundary_candidate_validated_count": int(
            status_counts.get("validated_boundary_sensitive_hold", 0)
        ),
        "support_loss_contrast_validated_count": int(
            status_counts.get("validated_support_loss_contrast", 0)
        ),
        "support_loss_contrast_mixed_count": int(
            status_counts.get("support_loss_contrast_mixed_hold", 0)
        ),
        "decision": (
            "Boundary-sensitive route holds survive the held-out schedule; "
            "support-loss contrasts are mixed, with field34 repeating hard loss "
            "and field30 c6-c10 retaining only prior hard-loss evidence."
        ),
        "claim_boundary": (
            "Margin validation only; no wall promotion, basin-quality "
            "evaluation, cost claim, or directed-search claim is made."
        ),
        "paths": {
            "schedule_rows": _rel(output_dir / SCHEDULE_ROWS_CSV),
            "pair_results": _rel(output_dir / PAIR_RESULTS_CSV),
            "summary": _rel(output_dir / SUMMARY_JSON),
            "report": _rel(output_dir / REPORT_MD),
        },
    }
    (output_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / CONFIG_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_report(output_dir / REPORT_MD, summary, pair_results)
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methodology-dir", type=Path, default=DEFAULT_METHODOLOGY_DIR)
    parser.add_argument("--margin-review-dir", type=Path, default=DEFAULT_MARGIN_REVIEW_DIR)
    parser.add_argument(
        "--heldout-runner-dir",
        type=Path,
        action="append",
        default=None,
        help="Held-out runner output directory. Repeat for multiple dirs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    heldout_dirs = (
        tuple(args.heldout_runner_dir)
        if args.heldout_runner_dir
        else DEFAULT_HELDOUT_RUNNER_DIRS
    )
    print(
        json.dumps(
            run(
                args.methodology_dir,
                args.margin_review_dir,
                heldout_dirs,
                args.output_dir,
            ),
            indent=2,
        )
    )

if __name__ == "__main__":
    main()
