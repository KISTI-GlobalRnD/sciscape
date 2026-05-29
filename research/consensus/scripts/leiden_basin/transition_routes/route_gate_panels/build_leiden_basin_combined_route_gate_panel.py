#!/usr/bin/env python3
"""Combine route-schedule claim gates into one wall-panel gate artifact.

The context coverage audit consumes one gate directory at a time. This script
combines the previous expanded-controls route gate with the clean-distinct
after-gap-fill route gate so the full 23-pair panel can be re-audited against a
single current route-gate surface.
"""

from __future__ import annotations

import argparse
import json
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
DEFAULT_EXPANDED_GATE_DIR = (
    BASE_RESULT_DIR / "leiden_basin_uniform_wall_probe_runner_expanded_controls_20260528"
)
DEFAULT_CLEAN_RUNNER_DIR = (
    BASE_RESULT_DIR / "leiden_basin_uniform_wall_probe_runner_clean_distinct_after_gap_fill_20260528"
)
DEFAULT_CLEAN_SUBSET_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_uniform_wall_probe_subset_clean_distinct_after_gap_fill_20260528"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_route_gate_panel_combined_after_clean_distinct_20260528"
)

CLAIM_ROWS_CSV = "uniform_route_schedule_claim_rows.csv"
PANEL_SUMMARY_CSV = "uniform_route_schedule_claim_panel_summary.csv"
PANEL_REPORT_MD = "uniform_route_schedule_claim_panel_report.md"
SUBSET_CSV = "uniform_wall_probe_subset.csv"
SUMMARY_JSON = "uniform_route_schedule_claim_panel_summary.json"
CONFIG_JSON = "uniform_route_schedule_claim_panel_config.json"

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

def _clean_gate_panel(clean_runner_dir: Path, clean_subset_dir: Path) -> pd.DataFrame:
    claims = _read_csv(clean_runner_dir / CLAIM_ROWS_CSV)
    subset = _read_csv(clean_subset_dir / SUBSET_CSV)
    if claims.empty:
        raise FileNotFoundError(clean_runner_dir / CLAIM_ROWS_CSV)
    if subset.empty:
        raise FileNotFoundError(clean_subset_dir / SUBSET_CSV)
    metadata_cols = [
        "panel_pair_id",
        "subset_order",
        "subset_role",
        "panel_role",
        "field",
        "case_id",
        "calibrated_relation",
        "support_distance_max",
    ]
    panel = claims.merge(
        subset[metadata_cols],
        on="panel_pair_id",
        how="left",
        suffixes=("", "_subset"),
    )
    panel["source_output"] = "clean_distinct_after_gap_fill"
    ordered_cols = [
        "panel_pair_id",
        "subset_order",
        "subset_role",
        "panel_role",
        "field",
        "case_id",
        "calibrated_relation",
        "support_distance_max",
        "source_output",
    ] + [column for column in claims.columns if column != "panel_pair_id"]
    return panel[ordered_cols]

def _write_report(path: Path, summary: dict[str, Any], combined: pd.DataFrame) -> None:
    lines = [
        "# Leiden Basin Combined Route-Gate Panel",
        "",
        "Status: expanded-controls and clean-distinct route gates combined",
        "Date: 2026-05-28",
        "",
        "This artifact combines route-schedule claim gates for coverage auditing. It does not rerun routes, rank basins, or promote wall claims beyond each row's predeclared gate status.",
        "",
        "## Gate Counts",
        "",
        "| status | pairs |",
        "| --- | ---: |",
    ]
    for status, count in sorted(summary["wall_claim_gate_status_counts"].items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "## Pair Gate Rows",
            "",
            "| pair_id | source | sensitivity | gate_status |",
            "| --- | --- | --- | --- |",
        ]
    )
    for _, row in combined.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["panel_pair_id"]),
                    str(row.get("source_output", "")),
                    str(row["route_order_sensitivity_status"]),
                    str(row["wall_claim_gate_status"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run(
    expanded_gate_dir: Path,
    clean_runner_dir: Path,
    clean_subset_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    expanded = _read_csv(expanded_gate_dir / PANEL_SUMMARY_CSV)
    if expanded.empty:
        raise FileNotFoundError(expanded_gate_dir / PANEL_SUMMARY_CSV)
    clean = _clean_gate_panel(clean_runner_dir, clean_subset_dir)
    columns = list(expanded.columns)
    for column in clean.columns:
        if column not in columns:
            columns.append(column)
    combined = pd.concat(
        [expanded.reindex(columns=columns), clean.reindex(columns=columns)],
        ignore_index=True,
        sort=False,
    )
    duplicate_ids = combined[combined["panel_pair_id"].astype(str).duplicated(keep=False)]
    if not duplicate_ids.empty:
        duplicates = ", ".join(sorted(set(duplicate_ids["panel_pair_id"].astype(str))))
        raise ValueError(f"duplicate panel_pair_id values across gate panels: {duplicates}")
    combined = combined.sort_values(["field", "case_id", "panel_pair_id"]).reset_index(drop=True)
    _write_csv(combined, output_dir / PANEL_SUMMARY_CSV)

    summary = {
        "status": "combined_route_gate_panel_prepared",
        "date": "2026-05-28",
        "script": _rel(Path(__file__)),
        "expanded_gate_dir": _rel(expanded_gate_dir),
        "clean_runner_dir": _rel(clean_runner_dir),
        "clean_subset_dir": _rel(clean_subset_dir),
        "output_dir": _rel(output_dir),
        "expanded_gate_pair_count": int(len(expanded)),
        "clean_gate_pair_count": int(len(clean)),
        "combined_gate_pair_count": int(len(combined)),
        "route_order_sensitivity_status_counts": (
            combined["route_order_sensitivity_status"].value_counts().to_dict()
        ),
        "wall_claim_gate_status_counts": (
            combined["wall_claim_gate_status"].value_counts().to_dict()
        ),
        "paths": {
            "panel_summary": _rel(output_dir / PANEL_SUMMARY_CSV),
            "summary": _rel(output_dir / SUMMARY_JSON),
            "report": _rel(output_dir / PANEL_REPORT_MD),
        },
        "claim_boundary": (
            "Combined route-gate panel only; no route execution, basin-quality "
            "evaluation, or new wall-promotion rule is introduced."
        ),
    }
    (output_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / CONFIG_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_report(output_dir / PANEL_REPORT_MD, summary, combined)
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded-gate-dir", type=Path, default=DEFAULT_EXPANDED_GATE_DIR)
    parser.add_argument("--clean-runner-dir", type=Path, default=DEFAULT_CLEAN_RUNNER_DIR)
    parser.add_argument("--clean-subset-dir", type=Path, default=DEFAULT_CLEAN_SUBSET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.expanded_gate_dir,
                args.clean_runner_dir,
                args.clean_subset_dir,
                args.output_dir,
            ),
            indent=2,
        )
    )

if __name__ == "__main__":
    main()
