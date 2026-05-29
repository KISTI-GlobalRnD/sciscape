#!/usr/bin/env python3
"""Build a conservative v0.1 taxonomy for Leiden basin relations.

This script consumes existing Track C basin-relation artifacts only. It does
not run routes, change thresholds by search, promote wall claims, or evaluate
basin value. The purpose is to split the current ambiguous relation bucket into
explicit boundary-review statuses before any wider wall protocol work.
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
DEFAULT_COVERAGE_DIR = BASE_RESULT_DIR / "leiden_basin_wall_panel_context_coverage_20260528"
DEFAULT_REFINEMENT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_stable_ambiguous_relation_refinement_20260528"
)
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_relation_taxonomy_v01_20260528"

COVERAGE_ROWS_CSV = "wall_panel_context_coverage_rows.csv"
REFINEMENT_ROWS_CSV = "stable_ambiguous_relation_refinement_rows.csv"

TAXONOMY_ROWS_CSV = "basin_relation_taxonomy_rows.csv"
STATUS_SUMMARY_CSV = "basin_relation_taxonomy_status_summary.csv"
BOUNDARY_QUEUE_CSV = "basin_relation_boundary_review_queue.csv"
SUMMARY_JSON = "basin_relation_taxonomy_summary.json"
REPORT_MD = "basin_relation_taxonomy_report.md"
CONFIG_JSON = "basin_relation_taxonomy_config.json"

SAME_SUPPORT_MAX = 0.5
DISTINCT_SUPPORT_MIN = 0.75
NEAR_SAME_MARGIN = 0.02
NEAR_DISTINCT_MARGIN = 0.005

QUALITY_LIKE_TOKENS = (
    "quality",
    "delta_q",
    "material",
    "cost",
    "elapsed",
    "wall_time",
    "p5_eval",
    "operator_success",
    "rank",
)

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

def _status_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    grouped = rows.groupby(
        ["current_calibrated_relation", "relation_taxonomy_v0_1"],
        dropna=False,
    )
    out: list[dict[str, Any]] = []
    for (current_relation, taxonomy_status), group in grouped:
        out.append(
            {
                "current_calibrated_relation": current_relation,
                "relation_taxonomy_v0_1": taxonomy_status,
                "pair_count": int(len(group)),
                "fields": "|".join(sorted(set(group["field"].astype(str)))),
                "wall_promotion_statuses": "|".join(
                    sorted(set(group["wall_promotion_status"].astype(str)))
                ),
            }
        )
    return pd.DataFrame(out).sort_values(
        ["current_calibrated_relation", "relation_taxonomy_v0_1"]
    )

def _boundary_distance_class(value: float) -> tuple[str, str]:
    if not math.isfinite(value):
        return "ambiguous_unknown_metric_hold", "support distance missing"
    if DISTINCT_SUPPORT_MIN - value <= NEAR_DISTINCT_MARGIN:
        return (
            "near_distinct_boundary_pending_membership_check",
            "coverage support distance sits inside the near-distinct boundary band",
        )
    if value - SAME_SUPPORT_MAX <= NEAR_SAME_MARGIN:
        return (
            "near_same_boundary_pending_membership_check",
            "coverage support distance sits inside the near-same boundary band",
        )
    return (
        "middle_ambiguous_support_local_hold",
        "coverage support distance remains in the middle ambiguous zone",
    )

def _cached_boundary_class(refinement_status: str) -> tuple[str, str]:
    if refinement_status == "near_distinct_boundary_requires_definition_choice":
        return (
            "near_distinct_boundary_review_cached",
            "cached full-membership support distance is just below the distinct threshold",
        )
    if refinement_status == "near_same_boundary_requires_definition_choice":
        return (
            "near_same_boundary_review_cached",
            "cached full-membership support distance is just above the same threshold",
        )
    if refinement_status == "distinct_support_local_under_current_rule":
        return (
            "distinct_support_local_after_cached_check",
            "cached full-membership support distance satisfies the distinct rule",
        )
    if refinement_status == "same_support_local_under_current_rule":
        return (
            "same_support_local_after_cached_check",
            "cached full-membership support distance satisfies the same rule",
        )
    if refinement_status == "confirmed_same_observed_basin":
        return (
            "same_endpoint_identity_after_cached_check",
            "cached endpoint membership hashes match",
        )
    return (
        "middle_ambiguous_support_local_hold",
        "cached full-membership support evidence remains between thresholds",
    )

def _wall_promotion_status(taxonomy_status: str, coverage_row: pd.Series) -> str:
    gate_status = str(coverage_row.get("existing_wall_claim_gate_status", "not_run"))
    route_status = str(coverage_row.get("existing_route_order_sensitivity_status", ""))
    runner_status = str(coverage_row.get("runner_context_status", ""))
    hygiene_status = str(coverage_row.get("field_hygiene_status", ""))
    if taxonomy_status.startswith("same_"):
        return "control_no_wall_promotion"
    if taxonomy_status in {
        "near_distinct_boundary_review_cached",
        "near_same_boundary_review_cached",
        "near_distinct_boundary_pending_membership_check",
        "near_same_boundary_pending_membership_check",
    }:
        return "blocked_boundary_review_no_wall_promotion"
    if taxonomy_status in {
        "middle_ambiguous_support_local_hold",
        "ambiguous_unknown_metric_hold",
    }:
        return "blocked_ambiguous_no_wall_promotion"
    if taxonomy_status.startswith("distinct_support_local"):
        if gate_status == "passes_schedule_invariance_distinct_partial_wall_evidence":
            return "relation_allows_existing_partial_wall_gate_not_supported_claim"
        if route_status == "route_order_sensitive" or gate_status == "fails_schedule_invariance_no_supported_wall_claim":
            return "relation_not_blocking_but_route_order_sensitivity_blocks"
        if runner_status == "runnable" and hygiene_status == "standard":
            return "relation_allows_future_route_gate_after_protocol"
        return "relation_not_blocking_but_context_or_hygiene_blocks"
    return "blocked_manual_review_no_wall_promotion"

def _next_taxonomy_action(taxonomy_status: str, coverage_row: pd.Series) -> str:
    if taxonomy_status in {
        "near_distinct_boundary_review_cached",
        "near_same_boundary_review_cached",
    }:
        return "fix_boundary_rule_before_route_promotion"
    if taxonomy_status in {
        "near_distinct_boundary_pending_membership_check",
        "near_same_boundary_pending_membership_check",
    }:
        return "obtain_cached_membership_before_relation_decision"
    if taxonomy_status in {
        "middle_ambiguous_support_local_hold",
        "ambiguous_unknown_metric_hold",
    }:
        return "hold_until_relation_rule_or_stronger_signature"
    if taxonomy_status.startswith("same_"):
        return "keep_as_control"
    return str(coverage_row.get("next_action", "hold_for_manual_review"))

def _build_rows(coverage: pd.DataFrame, refinement: pd.DataFrame) -> pd.DataFrame:
    refinement_lookup = {
        str(row["panel_pair_id"]): row.to_dict()
        for _, row in refinement.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for _, row in coverage.iterrows():
        pair_id = str(row["panel_pair_id"])
        current_relation = str(row["calibrated_relation"])
        support_distance = _safe_float(row.get("support_distance_max"))
        support_distance_source = "coverage_support_distance_max"
        evidence_grade = "calibrated_panel_relation"
        cached_status = ""
        cached_exact_support = math.nan

        if current_relation == "same_endpoint_identity":
            taxonomy_status = "same_endpoint_identity_control"
            reason = "current calibration already identifies the same endpoint identity"
        elif current_relation == "same_support_local":
            taxonomy_status = "same_support_local_control"
            reason = "current calibration already identifies the same support-local zone"
        elif current_relation == "distinct_support_local":
            taxonomy_status = "distinct_support_local_current_rule"
            reason = "current calibration satisfies the provisional distinct support rule"
        elif current_relation == "ambiguous_support_local":
            cached = refinement_lookup.get(pair_id)
            if cached:
                cached_status = str(cached.get("identity_refinement_status", ""))
                cached_exact_support = _safe_float(cached.get("exact_support_distance"))
                taxonomy_status, reason = _cached_boundary_class(cached_status)
                support_distance = cached_exact_support
                support_distance_source = "cached_full_membership_exact_support"
                evidence_grade = str(cached.get("evidence_grade", "cached_full_membership_exact_support"))
            else:
                taxonomy_status, reason = _boundary_distance_class(support_distance)
                evidence_grade = "calibrated_support_proxy_pending_membership_check"
        else:
            taxonomy_status = "manual_review_unrecognized_relation"
            reason = "current relation is not recognized by the v0.1 taxonomy"

        same_margin = support_distance - SAME_SUPPORT_MAX if math.isfinite(support_distance) else math.nan
        distinct_margin = DISTINCT_SUPPORT_MIN - support_distance if math.isfinite(support_distance) else math.nan
        wall_status = _wall_promotion_status(taxonomy_status, row)
        rows.append(
            {
                "panel_pair_id": pair_id,
                "field": str(row["field"]),
                "case_id": str(row["case_id"]),
                "method": str(row["method"]),
                "source_label": str(row["source_label"]),
                "panel_role": str(row["panel_role"]),
                "current_calibrated_relation": current_relation,
                "relation_taxonomy_v0_1": taxonomy_status,
                "taxonomy_reason": reason,
                "support_distance_for_taxonomy": support_distance,
                "support_distance_source": support_distance_source,
                "same_threshold_margin": same_margin,
                "distinct_threshold_margin": distinct_margin,
                "cached_identity_refinement_status": cached_status,
                "evidence_grade": evidence_grade,
                "runner_context_status": str(row.get("runner_context_status", "")),
                "runner_preflight_status": str(row.get("runner_preflight_status", "")),
                "field_hygiene_status": str(row.get("field_hygiene_status", "")),
                "existing_route_order_sensitivity_status": str(
                    row.get("existing_route_order_sensitivity_status", "")
                ),
                "existing_wall_claim_gate_status": str(
                    row.get("existing_wall_claim_gate_status", "")
                ),
                "coverage_next_action": str(row.get("next_action", "")),
                "taxonomy_next_action": _next_taxonomy_action(taxonomy_status, row),
                "wall_promotion_status": wall_status,
                "claim_boundary": (
                    "Basin relation taxonomy only; no route execution, wall promotion, "
                    "or basin evaluation is made."
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "relation_taxonomy_v0_1",
            "support_distance_for_taxonomy",
            "panel_pair_id",
        ],
        ascending=[True, False, True],
    )

def _boundary_queue(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    queue = rows[
        rows["relation_taxonomy_v0_1"].isin(
            {
                "near_distinct_boundary_review_cached",
                "near_same_boundary_review_cached",
                "near_distinct_boundary_pending_membership_check",
                "near_same_boundary_pending_membership_check",
            }
        )
    ].copy()
    if queue.empty:
        return pd.DataFrame()
    cols = [
        "panel_pair_id",
        "field",
        "case_id",
        "method",
        "source_label",
        "current_calibrated_relation",
        "relation_taxonomy_v0_1",
        "support_distance_for_taxonomy",
        "support_distance_source",
        "same_threshold_margin",
        "distinct_threshold_margin",
        "cached_identity_refinement_status",
        "runner_context_status",
        "field_hygiene_status",
        "existing_wall_claim_gate_status",
        "taxonomy_next_action",
        "wall_promotion_status",
    ]
    return queue[cols].sort_values(
        ["relation_taxonomy_v0_1", "distinct_threshold_margin", "same_threshold_margin"]
    )

def _quality_column_leaks(frames: dict[str, pd.DataFrame]) -> list[str]:
    leaks: list[str] = []
    for name, frame in frames.items():
        for column in frame.columns:
            lower = column.lower()
            if any(token in lower for token in QUALITY_LIKE_TOKENS):
                leaks.append(f"{name}:{column}")
    return leaks

def _write_report(
    path: Path,
    summary: dict[str, Any],
    status_summary: pd.DataFrame,
    boundary_queue: pd.DataFrame,
) -> None:
    lines = [
        "# Leiden Basin Relation Taxonomy v0.1",
        "",
        "Status: basin relation taxonomy prepared",
        "Date: 2026-05-28",
        "",
        "This artifact refines the relation labels used before wall cartography. It does not run routes, promote wall claims, rank basins, or evaluate basin value.",
        "",
        "## Rule",
        "",
        f"- same support-local threshold: support distance <= {SAME_SUPPORT_MAX}",
        f"- distinct support-local threshold: support distance >= {DISTINCT_SUPPORT_MIN}",
        f"- near-same boundary band: within {NEAR_SAME_MARGIN} above the same threshold",
        f"- near-distinct boundary band: within {NEAR_DISTINCT_MARGIN} below the distinct threshold",
        "- boundary-review classes are not wall-claim classes.",
        "",
        "## Summary",
        "",
        f"- panel pairs: {summary['panel_pair_count']}",
        f"- cached refinement rows used: {summary['cached_refinement_pair_count']}",
        f"- boundary-review pairs: {summary['boundary_review_pair_count']}",
        f"- wall-promotion eligible pairs newly created here: {summary['new_wall_promotion_pair_count']}",
        "",
        "## Taxonomy Counts",
        "",
        "| current_relation | taxonomy_v0_1 | pairs | fields | wall_promotion_statuses |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for _, row in status_summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["current_calibrated_relation"]),
                    str(row["relation_taxonomy_v0_1"]),
                    str(row["pair_count"]),
                    str(row["fields"]),
                    str(row["wall_promotion_statuses"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Boundary Review Queue",
            "",
            "| pair_id | field | taxonomy_v0_1 | support_distance | source | next_action |",
            "| --- | --- | --- | ---: | --- | --- |",
        ]
    )
    for _, row in boundary_queue.iterrows():
        support = _safe_float(row["support_distance_for_taxonomy"])
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["panel_pair_id"]),
                    str(row["field"]),
                    str(row["relation_taxonomy_v0_1"]),
                    "" if not math.isfinite(support) else f"{support:.6f}",
                    str(row["support_distance_source"]),
                    str(row["taxonomy_next_action"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Keep hard same/distinct thresholds as the current wall-promotion gate.",
            "- Add explicit boundary-review statuses so near-threshold evidence is not hidden inside a generic ambiguous bucket.",
            "- Do not promote any boundary-review row to wall evidence in this artifact.",
            "- The next method step is to decide whether boundary-review rows remain excluded from wall promotion or become a separate basin-relation evidence class with additional membership/signature requirements.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run(coverage_dir: Path, refinement_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage = _read_csv(coverage_dir / COVERAGE_ROWS_CSV)
    refinement = _read_csv(refinement_dir / REFINEMENT_ROWS_CSV)
    if coverage.empty:
        raise FileNotFoundError(coverage_dir / COVERAGE_ROWS_CSV)
    if refinement.empty:
        raise FileNotFoundError(refinement_dir / REFINEMENT_ROWS_CSV)

    rows = _build_rows(coverage, refinement)
    status_summary = _status_summary(rows)
    boundary_queue = _boundary_queue(rows)

    frames = {
        TAXONOMY_ROWS_CSV: rows,
        STATUS_SUMMARY_CSV: status_summary,
        BOUNDARY_QUEUE_CSV: boundary_queue,
    }
    leaks = _quality_column_leaks(frames)
    if leaks:
        raise ValueError("quality-like columns leaked into relation taxonomy outputs: " + ", ".join(leaks))

    summary = {
        "status": "basin_relation_taxonomy_v0_1_prepared",
        "date": "2026-05-28",
        "panel_pair_count": int(len(rows)),
        "cached_refinement_pair_count": int(len(refinement)),
        "taxonomy_status_counts": rows["relation_taxonomy_v0_1"].value_counts().to_dict(),
        "current_relation_counts": rows["current_calibrated_relation"].value_counts().to_dict(),
        "wall_promotion_status_counts": rows["wall_promotion_status"].value_counts().to_dict(),
        "boundary_review_pair_count": int(len(boundary_queue)),
        "cached_boundary_review_pair_count": int(
            boundary_queue["support_distance_source"].eq("cached_full_membership_exact_support").sum()
        )
        if not boundary_queue.empty
        else 0,
        "pending_membership_boundary_pair_count": int(
            boundary_queue["support_distance_source"].ne("cached_full_membership_exact_support").sum()
        )
        if not boundary_queue.empty
        else 0,
        "new_wall_promotion_pair_count": 0,
        "same_support_max": SAME_SUPPORT_MAX,
        "distinct_support_min": DISTINCT_SUPPORT_MIN,
        "near_same_margin": NEAR_SAME_MARGIN,
        "near_distinct_margin": NEAR_DISTINCT_MARGIN,
        "decision": (
            "Use relation_taxonomy_v0_1 as a boundary-aware relation status while "
            "keeping boundary-review rows blocked from wall promotion."
        ),
        "claim_boundary": (
            "Relation taxonomy only; no route execution, wall promotion, basin "
            "quality, cost, ranking, or operator claim is made."
        ),
    }

    for filename, frame in frames.items():
        _write_csv(frame, output_dir / filename)
    (output_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            {
                "script": _rel(Path(__file__)),
                "coverage_dir": _rel(coverage_dir),
                "refinement_dir": _rel(refinement_dir),
                "same_support_max": SAME_SUPPORT_MAX,
                "distinct_support_min": DISTINCT_SUPPORT_MIN,
                "near_same_margin": NEAR_SAME_MARGIN,
                "near_distinct_margin": NEAR_DISTINCT_MARGIN,
                "scope": "basin relation taxonomy only; no route execution",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / REPORT_MD, summary, status_summary, boundary_queue)
    return {"output_dir": _rel(output_dir), **summary}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-dir", type=Path, default=DEFAULT_COVERAGE_DIR)
    parser.add_argument("--refinement-dir", type=Path, default=DEFAULT_REFINEMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.coverage_dir, args.refinement_dir, args.output_dir), indent=2))

if __name__ == "__main__":
    main()
