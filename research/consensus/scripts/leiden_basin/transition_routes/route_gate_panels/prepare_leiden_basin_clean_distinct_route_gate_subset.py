#!/usr/bin/env python3
"""Prepare a clean distinct route-gate subset after vanilla context gap fill.

This script converts the post-gap-fill wall-panel coverage rows into the
execution-manifest schema expected by `run_leiden_basin_uniform_direct_pair_routes.py`.
It selects only clean non-field34 distinct pairs whose runner preflight is ready.
No route is executed here and no wall or basin-quality claim is made.
"""

from __future__ import annotations

import argparse
import json
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
DEFAULT_COVERAGE_DIR = (
    BASE_RESULT_DIR / "leiden_basin_wall_panel_context_coverage_after_gap_fill_20260528"
)
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_uniform_wall_probe_subset_clean_distinct_after_gap_fill_20260528"
)

COVERAGE_ROWS_CSV = "wall_panel_context_coverage_rows.csv"
SUBSET_CSV = "uniform_wall_probe_subset.csv"
EXECUTION_MANIFEST_CSV = "uniform_wall_probe_execution_manifest.csv"
SUMMARY_JSON = "uniform_wall_probe_subset_summary.json"
REPORT_MD = "uniform_wall_probe_subset_report.md"
CONFIG_JSON = "uniform_wall_probe_subset_config.json"

RUN_ACTION = "run_w1_w6_route_order_gate"
SUBSET_ROLE = "clean_non_field34_distinct_route_gate"
SELECTION_MODE = "clean_distinct_after_gap_fill"


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


def _select_clean_distinct(coverage: pd.DataFrame) -> pd.DataFrame:
    if coverage.empty:
        raise FileNotFoundError(COVERAGE_ROWS_CSV)
    selected = coverage[coverage["next_action"].astype(str).eq(RUN_ACTION)].copy()
    if selected.empty:
        raise ValueError(f"no coverage rows found with next_action={RUN_ACTION}")

    invalid = selected[
        ~(
            selected["calibrated_relation"].astype(str).eq("distinct_support_local")
            & selected["runner_preflight_status"].astype(str).eq("runner_preflight_ready")
            & selected["field_hygiene_status"].astype(str).eq("standard")
            & selected["existing_route_order_sensitivity_status"].astype(str).eq("not_run")
            & selected["existing_wall_claim_gate_status"].astype(str).eq("not_run")
        )
    ]
    if not invalid.empty:
        bad_pairs = ", ".join(invalid["panel_pair_id"].astype(str).tolist())
        raise ValueError(f"selected rows are not clean distinct route-gate candidates: {bad_pairs}")

    selected["support_distance_max_num"] = pd.to_numeric(
        selected["support_distance_max"],
        errors="coerce",
    )
    return selected.sort_values(
        ["field", "method", "support_distance_max_num", "panel_pair_id"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


def _subset_rows(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for order, (_, row) in enumerate(selected.iterrows(), start=1):
        rows.append(
            {
                "selection_mode": SELECTION_MODE,
                "subset_order": order,
                "subset_role": SUBSET_ROLE,
                "subset_selection_reason": (
                    "post-gap-fill distinct support-local pair with runner preflight ready, "
                    "standard field hygiene, and no existing route gate"
                ),
                "panel_pair_id": row["panel_pair_id"],
                "panel_role": row["panel_role"],
                "source_label": row["source_label"],
                "case_id": row["case_id"],
                "field": row["field"],
                "method": row["method"],
                "candidate_budget": row["candidate_budget"],
                "left_endpoint_identity_id": row["left_endpoint_identity_id"],
                "right_endpoint_identity_id": row["right_endpoint_identity_id"],
                "left_representative_candidate_index": row[
                    "left_representative_candidate_index"
                ],
                "right_representative_candidate_index": row[
                    "right_representative_candidate_index"
                ],
                "calibrated_relation": row["calibrated_relation"],
                "endpoint_distance_min": row["endpoint_distance_min"],
                "endpoint_distance_max": row["endpoint_distance_max"],
                "support_distance_min": row["support_distance_min"],
                "support_distance_max": row["support_distance_max"],
                "has_existing_route_source": False,
                "existing_route_join_candidate": "",
                "existing_route_join_status": "",
                "existing_wall_claim_status": row["existing_wall_claim_gate_status"],
                "direct_route_audit_status": row["existing_route_order_sensitivity_status"],
                "direct_cross_route_rows": "",
                "self_endpoint_route_rows": "",
            }
        )
    return pd.DataFrame(rows)


def _execution_manifest(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for order, (_, row) in enumerate(selected.iterrows(), start=1):
        rows.append(
            {
                "subset_order": order,
                "subset_role": SUBSET_ROLE,
                "panel_pair_id": row["panel_pair_id"],
                "case_id": row["case_id"],
                "left_endpoint_identity_id": row["left_endpoint_identity_id"],
                "right_endpoint_identity_id": row["right_endpoint_identity_id"],
                "left_representative_candidate_index": row[
                    "left_representative_candidate_index"
                ],
                "right_representative_candidate_index": row[
                    "right_representative_candidate_index"
                ],
                "left_endpoint_source_artifact": row["left_endpoint_source_artifact"],
                "right_endpoint_source_artifact": row["right_endpoint_source_artifact"],
                "vanilla_context_dir": row["vanilla_context_dir"],
                "vanilla_context_status": row["vanilla_context_status"],
                "legacy_field34_cc_pathway_status": "not_applicable",
                "uniform_direct_pair_route_status": "missing",
                "execution_readiness": (
                    "vanilla_graph_context_available_boundary_and_uniform_runner_missing"
                ),
                "next_action": "run uniform W1-W6 route-order gate",
                "claim_boundary": (
                    "Execution readiness only; no wall or basin-evaluation claim is made."
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_report(path: Path, summary: dict[str, Any], subset: pd.DataFrame) -> None:
    lines = [
        "# Leiden Basin Clean Distinct Route-Gate Subset",
        "",
        "Status: clean distinct W1-W6 route-gate subset prepared",
        "Date: 2026-05-28",
        "",
        "This artifact selects clean non-field34 distinct pairs for route-order gate execution. It does not run routes, rank basins, or promote wall claims.",
        "",
        "## Selected Pairs",
        "",
        "| order | pair_id | field | relation | support_max |",
        "| ---: | --- | --- | --- | ---: |",
    ]
    for _, row in subset.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["subset_order"]),
                    str(row["panel_pair_id"]),
                    str(row["field"]),
                    str(row["calibrated_relation"]),
                    str(row["support_distance_max"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision Boundary",
            "",
            "- Run W1-W6 route-order gates before any wall promotion.",
            "- Keep boundary-review and field34 hygiene rows outside this execution batch.",
            "- Do not join basin quality or cost fields at this stage.",
            "",
            "## Summary",
            "",
            f"- selected pairs: {summary['selected_pair_count']}",
            f"- selected cases: {summary['selected_case_count']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(coverage_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage = _read_csv(coverage_dir / COVERAGE_ROWS_CSV)
    selected = _select_clean_distinct(coverage)
    subset = _subset_rows(selected)
    manifest = _execution_manifest(selected)

    _write_csv(subset, output_dir / SUBSET_CSV)
    _write_csv(manifest, output_dir / EXECUTION_MANIFEST_CSV)

    summary = {
        "status": "clean_distinct_route_gate_subset_prepared",
        "date": "2026-05-28",
        "script": _rel(Path(__file__)),
        "coverage_dir": _rel(coverage_dir),
        "output_dir": _rel(output_dir),
        "selection_mode": SELECTION_MODE,
        "selection_rule": (
            "next_action=run_w1_w6_route_order_gate, distinct_support_local, "
            "runner_preflight_ready, field_hygiene_status=standard, existing route gate not run"
        ),
        "selected_pair_count": int(len(subset)),
        "selected_case_count": int(subset["case_id"].nunique()),
        "selected_pair_ids": subset["panel_pair_id"].astype(str).tolist(),
        "paths": {
            "subset": _rel(output_dir / SUBSET_CSV),
            "execution_manifest": _rel(output_dir / EXECUTION_MANIFEST_CSV),
            "summary": _rel(output_dir / SUMMARY_JSON),
            "report": _rel(output_dir / REPORT_MD),
        },
        "claim_boundary": (
            "Route-gate subset only; no route execution, wall promotion, or "
            "basin-quality claim is made."
        ),
    }
    (output_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / CONFIG_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_report(output_dir / REPORT_MD, summary, subset)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-dir", type=Path, default=DEFAULT_COVERAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.coverage_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
