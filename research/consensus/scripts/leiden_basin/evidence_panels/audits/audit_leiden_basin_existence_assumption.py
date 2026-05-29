#!/usr/bin/env python3
"""Audit Track C's first assumption: multiple meaningful basins exist.

This separates two claims that should not be merged:

1. Existence: a case has multiple endpoint identities with substantial changed
   support and at least one distinct support-local relation.
2. Pathway: there is a route or protocol that can connect distinct basin
   candidates.

The audit does not inspect basin quality/cost, does not run routes, and does
not promote wall claims.
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
DEFAULT_CALIBRATION_DIR = BASE_RESULT_DIR / "leiden_basin_definition_calibration_20260528"
DEFAULT_REMAINING_AUDIT_DIR = BASE_RESULT_DIR / "leiden_basin_remaining_wall_question_audit_20260529"
DEFAULT_FIELD34_AUDIT_DIR = BASE_RESULT_DIR / "leiden_basin_field34_evidence_eligibility_audit_20260529"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_existence_assumption_audit_20260529"

ENDPOINT_ROWS_CSV = "endpoint_identity_rows.csv"
IDENTITY_PAIR_ROWS_CSV = "identity_pair_relation_rows.csv"
WALL_CANDIDATE_ROWS_CSV = "wall_candidate_pair_rows.csv"
REMAINING_ROWS_CSV = "remaining_wall_question_rows.csv"
FIELD34_QUEUE_ROWS_CSV = "field34_queue_projection_rows.csv"

CASE_ROWS_CSV = "basin_existence_case_rows.csv"
PAIR_ROWS_CSV = "basin_existence_pair_rows.csv"
PATHWAY_ROWS_CSV = "pathway_readiness_rows.csv"
SUMMARY_JSON = "basin_existence_assumption_summary.json"
REPORT_MD = "basin_existence_assumption_report.md"
CONFIG_JSON = "basin_existence_assumption_config.json"

MODERATE_SUPPORT_MIN = 100
STRONG_SUPPORT_MIN = 500
CLAIM_BOUNDARY = (
    "Basin-existence assumption audit only; no route execution, wall-promotion "
    "change, basin-quality claim, cost claim, or directed-search claim."
)

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"empty CSV: {path}") from exc

def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

def _safe_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)

def _count(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    return {str(k): int(v) for k, v in frame[column].value_counts(dropna=False).to_dict().items()}

def _endpoint_support(endpoint_rows: pd.DataFrame) -> pd.DataFrame:
    accepted = endpoint_rows[
        endpoint_rows["endpoint_filter_status"].astype(str).str.startswith("accepted")
    ].copy()
    grouped = (
        accepted.groupby(
            ["case_id", "field", "method", "candidate_budget", "endpoint_identity_id"],
            dropna=False,
        )
        .agg(
            endpoint_member_count=("candidate_index", "count"),
            support_node_count=("support_node_count", "max"),
            representative_candidate_index=("representative_candidate_index", "min"),
        )
        .reset_index()
    )
    return grouped

def _existence_pair_rows(calibration_dir: Path) -> pd.DataFrame:
    endpoints = _endpoint_support(_read_csv(calibration_dir / ENDPOINT_ROWS_CSV))
    wall_pairs = _read_csv(calibration_dir / WALL_CANDIDATE_ROWS_CSV).copy()

    left = endpoints.add_prefix("left_")
    right = endpoints.add_prefix("right_")
    rows = wall_pairs.merge(
        left[
            [
                "left_case_id",
                "left_endpoint_identity_id",
                "left_support_node_count",
                "left_endpoint_member_count",
            ]
        ],
        left_on=["case_id", "left_endpoint_identity_id"],
        right_on=["left_case_id", "left_endpoint_identity_id"],
        how="left",
    ).merge(
        right[
            [
                "right_case_id",
                "right_endpoint_identity_id",
                "right_support_node_count",
                "right_endpoint_member_count",
            ]
        ],
        left_on=["case_id", "right_endpoint_identity_id"],
        right_on=["right_case_id", "right_endpoint_identity_id"],
        how="left",
    )
    rows["min_endpoint_support"] = rows[
        ["left_support_node_count", "right_support_node_count"]
    ].min(axis=1)
    rows["support_substance_class"] = rows["min_endpoint_support"].map(
        lambda value: (
            "strong_support_pair"
            if value >= STRONG_SUPPORT_MIN
            else "moderate_support_pair"
            if value >= MODERATE_SUPPORT_MIN
            else "weak_support_pair"
        )
    )
    rows["field_hygiene_class"] = rows["field"].map(
        lambda field: "field34_hygiene_limited" if str(field) == "field34" else "clean_non_field34"
    )
    rows["meaningful_basin_pair_status"] = rows.apply(_meaningful_pair_status, axis=1)
    rows["claim_boundary"] = CLAIM_BOUNDARY

    cols = [
        "case_id",
        "field",
        "method",
        "candidate_budget",
        "left_endpoint_identity_id",
        "right_endpoint_identity_id",
        "support_distance_min",
        "support_distance_max",
        "left_support_node_count",
        "right_support_node_count",
        "min_endpoint_support",
        "support_substance_class",
        "field_hygiene_class",
        "has_route_trace_source",
        "phase2_route_join_status",
        "wall_assignment_status",
        "meaningful_basin_pair_status",
        "claim_boundary",
    ]
    return rows[cols].sort_values(
        ["field", "method", "meaningful_basin_pair_status", "support_distance_max"],
        ascending=[True, True, True, False],
    )

def _meaningful_pair_status(row: pd.Series) -> str:
    if _safe_str(row.get("field")) == "field34":
        return "field34_reference_only_not_clean_meaningful_basin_evidence"
    support = float(row.get("min_endpoint_support", 0) or 0)
    if support >= STRONG_SUPPORT_MIN:
        return "strong_meaningful_distinct_basin_candidate_pair"
    if support >= MODERATE_SUPPORT_MIN:
        return "moderate_meaningful_distinct_basin_candidate_pair"
    return "weak_support_distinct_pair_hold"

def _case_rows(calibration_dir: Path, pair_rows: pd.DataFrame) -> pd.DataFrame:
    endpoints = _endpoint_support(_read_csv(calibration_dir / ENDPOINT_ROWS_CSV))
    identity_pairs = _read_csv(calibration_dir / IDENTITY_PAIR_ROWS_CSV)

    endpoint_stats = (
        endpoints.groupby(["case_id", "field", "method", "candidate_budget"], dropna=False)
        .agg(
            accepted_endpoint_identity_count=("endpoint_identity_id", "nunique"),
            endpoint_support_min=("support_node_count", "min"),
            endpoint_support_median=("support_node_count", "median"),
            endpoint_support_max=("support_node_count", "max"),
        )
        .reset_index()
    )
    relation_counts = (
        identity_pairs.pivot_table(
            index=["case_id"],
            columns="calibrated_relation",
            values="left_endpoint_identity_id",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in [
        "same_support_local",
        "same_endpoint_identity",
        "ambiguous_support_local",
        "distinct_support_local",
    ]:
        if col not in relation_counts:
            relation_counts[col] = 0

    pair_counts = (
        pair_rows.groupby("case_id", dropna=False)
        .agg(
            wall_candidate_pair_count=("left_endpoint_identity_id", "count"),
            strong_meaningful_pair_count=(
                "meaningful_basin_pair_status",
                lambda s: int((s == "strong_meaningful_distinct_basin_candidate_pair").sum()),
            ),
            moderate_meaningful_pair_count=(
                "meaningful_basin_pair_status",
                lambda s: int((s == "moderate_meaningful_distinct_basin_candidate_pair").sum()),
            ),
            field34_reference_pair_count=(
                "meaningful_basin_pair_status",
                lambda s: int(
                    (s == "field34_reference_only_not_clean_meaningful_basin_evidence").sum()
                ),
            ),
        )
        .reset_index()
    )
    rows = endpoint_stats.merge(relation_counts, on="case_id", how="left").merge(
        pair_counts,
        on="case_id",
        how="left",
    )
    count_cols = [
        "same_support_local",
        "same_endpoint_identity",
        "ambiguous_support_local",
        "distinct_support_local",
        "wall_candidate_pair_count",
        "strong_meaningful_pair_count",
        "moderate_meaningful_pair_count",
        "field34_reference_pair_count",
    ]
    rows[count_cols] = rows[count_cols].fillna(0).astype(int)
    rows["multi_basin_existence_status"] = rows.apply(_case_existence_status, axis=1)
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values(["field", "method", "case_id"]).reset_index(drop=True)

def _case_existence_status(row: pd.Series) -> str:
    if _safe_str(row.get("field")) == "field34":
        return "field34_reference_only_hygiene_limited"
    if int(row.get("strong_meaningful_pair_count", 0)) > 0:
        return "strong_candidate_multi_basin_existence_evidence"
    if int(row.get("moderate_meaningful_pair_count", 0)) > 0:
        return "moderate_candidate_multi_basin_existence_evidence"
    if int(row.get("distinct_support_local", 0)) > 0:
        return "weak_or_low_support_distinct_candidate_hold"
    if int(row.get("ambiguous_support_local", 0)) > 0:
        return "ambiguous_basin_relation_only"
    return "no_multi_basin_evidence_under_current_gate"

def _pathway_rows(
    *,
    remaining_audit_dir: Path,
    field34_audit_dir: Path,
) -> pd.DataFrame:
    remaining = _read_csv(remaining_audit_dir / REMAINING_ROWS_CSV)
    field34 = _read_csv(field34_audit_dir / FIELD34_QUEUE_ROWS_CSV)
    non_field34_rows = []
    for _, row in remaining.iterrows():
        non_field34_rows.append(
            {
                "panel_pair_id": row["panel_pair_id"],
                "field": row["field"],
                "case_id": row["case_id"],
                "source_surface": "current_23_pair_non_field34",
                "route_label_interpretation_v0": row.get("route_label_interpretation_v0", ""),
                "pathway_readiness_status": _pathway_status_from_remaining(row),
                "route_execution_decision": row.get("route_execution_decision", ""),
                "wall_promotion_decision": row.get("wall_promotion_decision", ""),
                "pathway_interpretation": _pathway_interpretation_from_remaining(row),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    field34_rows = []
    for _, row in field34.iterrows():
        field34_rows.append(
            {
                "panel_pair_id": row["panel_pair_id"],
                "field": row["field"],
                "case_id": row["case_id"],
                "source_surface": "current_23_pair_field34",
                "route_label_interpretation_v0": row.get("route_label_interpretation_v0", ""),
                "pathway_readiness_status": "field34_pathway_not_ready_hygiene_limited",
                "route_execution_decision": row.get("route_execution_status_after_hygiene", ""),
                "wall_promotion_decision": row.get("wall_promotion_status_after_hygiene", ""),
                "pathway_interpretation": (
                    "Field34 remains reference/hold/filtered evidence, not a pathway "
                    "execution source under current gates."
                ),
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(non_field34_rows + field34_rows).sort_values(
        ["source_surface", "pathway_readiness_status", "panel_pair_id"]
    )

def _pathway_status_from_remaining(row: pd.Series) -> str:
    cls = _safe_str(row.get("remaining_wall_question_class"))
    if cls == "protocol_reference_only":
        return "pathway_protocol_reference_not_operational"
    if cls == "route_uncertainty_reference_only":
        return "pathway_uncertainty_reference_not_operational"
    if cls in {
        "closed_by_current_boundary_rule",
        "closed_by_cached_pending_membership_boundary_review",
        "middle_ambiguous_definition_hold",
    }:
        return "pathway_blocked_by_basin_relation_definition"
    if cls == "no_wall_contrast_reference_only":
        return "pathway_negative_contrast_reference"
    if cls == "same_or_identity_control_only":
        return "pathway_control_only"
    if cls == "unrun_distinct_candidate_found":
        return "pathway_candidate_requires_manual_review"
    return "pathway_not_ready"

def _pathway_interpretation_from_remaining(row: pd.Series) -> str:
    status = _pathway_status_from_remaining(row)
    if status == "pathway_protocol_reference_not_operational":
        return "Route protocol evidence exists, but it is not a supported pathway method."
    if status == "pathway_uncertainty_reference_not_operational":
        return "Route behavior is boundary-sensitive and cannot support a pathway claim."
    if status == "pathway_blocked_by_basin_relation_definition":
        return "Pathway testing is premature because basin relation is not accepted."
    if status == "pathway_negative_contrast_reference":
        return "This row is useful as a no-wall or support-loss contrast."
    if status == "pathway_control_only":
        return "Control relation; not a pathway target."
    if status == "pathway_candidate_requires_manual_review":
        return "Potential route candidate found; requires a predeclared mechanism question."
    return "No pathway claim follows under current gates."

def _summary(
    *,
    case_rows: pd.DataFrame,
    pair_rows: pd.DataFrame,
    pathway_rows: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    non_field34_cases = case_rows[case_rows["field"].ne("field34")]
    non_field34_pairs = pair_rows[pair_rows["field"].ne("field34")]
    strong_case_count = int(
        non_field34_cases["multi_basin_existence_status"]
        .eq("strong_candidate_multi_basin_existence_evidence")
        .sum()
    )
    moderate_case_count = int(
        non_field34_cases["multi_basin_existence_status"]
        .eq("moderate_candidate_multi_basin_existence_evidence")
        .sum()
    )
    strong_pair_count = int(
        non_field34_pairs["meaningful_basin_pair_status"]
        .eq("strong_meaningful_distinct_basin_candidate_pair")
        .sum()
    )
    moderate_pair_count = int(
        non_field34_pairs["meaningful_basin_pair_status"]
        .eq("moderate_meaningful_distinct_basin_candidate_pair")
        .sum()
    )
    executable_pathway_count = int(
        pathway_rows["pathway_readiness_status"].eq("pathway_candidate_requires_manual_review").sum()
    )
    return {
        "status": "basin_existence_assumption_audit_prepared",
        "date": "2026-05-29",
        "script": _rel(Path(__file__).resolve()),
        "output_dir": _rel(output_dir),
        "support_thresholds": {
            "moderate_support_min": MODERATE_SUPPORT_MIN,
            "strong_support_min": STRONG_SUPPORT_MIN,
        },
        "case_count": int(len(case_rows)),
        "non_field34_case_count": int(len(non_field34_cases)),
        "strong_candidate_multi_basin_case_count": strong_case_count,
        "moderate_candidate_multi_basin_case_count": moderate_case_count,
        "clean_non_field34_distinct_pair_count": int(len(non_field34_pairs)),
        "strong_meaningful_distinct_pair_count": strong_pair_count,
        "moderate_meaningful_distinct_pair_count": moderate_pair_count,
        "pathway_surface_row_count": int(len(pathway_rows)),
        "pathway_candidate_requires_manual_review_count": executable_pathway_count,
        "pathway_readiness_status_counts": _count(pathway_rows, "pathway_readiness_status"),
        "hypothesis_1_basin_existence_status": (
            "supported_as_candidate_multi_basin_existence_evidence_not_final_basin_definition"
            if strong_pair_count or moderate_pair_count
            else "not_supported_under_current_gate"
        ),
        "hypothesis_2_pathway_status": (
            "not_operational_under_current_gates"
            if executable_pathway_count == 0
            else "candidate_requires_manual_review_before_route"
        ),
        "decision": (
            "Current data support the first assumption only as candidate evidence: "
            "multiple clean non-field34 support-local basin candidates exist under "
            "declared thresholds. The pathway methodology is not yet operational."
        ),
        "next_step": (
            "Do not run route batches from this audit. Either formalize the basin "
            "existence definition, or reopen pathway work by declaring a new "
            "precommitted panel and wall/pathway evidence requirement."
        ),
        "paths": {
            "case_rows": _rel(output_dir / CASE_ROWS_CSV),
            "pair_rows": _rel(output_dir / PAIR_ROWS_CSV),
            "pathway_rows": _rel(output_dir / PATHWAY_ROWS_CSV),
            "summary": _rel(output_dir / SUMMARY_JSON),
            "report": _rel(output_dir / REPORT_MD),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }

def _write_report(
    path: Path,
    summary: dict[str, Any],
    case_rows: pd.DataFrame,
    pathway_rows: pd.DataFrame,
) -> None:
    lines = [
        "# Basin Existence Assumption Audit",
        "",
        "Date: 2026-05-29",
        "",
        "## Scope",
        "",
        "This artifact separates Track C's first assumption from the pathway claim.",
        "It asks whether multiple meaningful basin candidates exist under declared",
        "support-local gates, and whether current route/pathway evidence is ready.",
        "",
        "## Decision",
        "",
        str(summary["decision"]),
        "",
        "## Hypothesis Status",
        "",
        f"- H1 basin existence: `{summary['hypothesis_1_basin_existence_status']}`",
        f"- H2 pathway between basins: `{summary['hypothesis_2_pathway_status']}`",
        "",
        "## Basin Existence Counts",
        "",
        f"- non-field34 cases: `{summary['non_field34_case_count']}`",
        f"- strong candidate multi-basin cases: `{summary['strong_candidate_multi_basin_case_count']}`",
        f"- moderate candidate multi-basin cases: `{summary['moderate_candidate_multi_basin_case_count']}`",
        f"- strong meaningful distinct pairs: `{summary['strong_meaningful_distinct_pair_count']}`",
        f"- moderate meaningful distinct pairs: `{summary['moderate_meaningful_distinct_pair_count']}`",
        "",
        "## Case Status",
        "",
        "| case_id | field | method | budget | endpoint identities | distinct pairs | strong pairs | moderate pairs | status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in case_rows.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.case_id),
                    str(row.field),
                    str(row.method),
                    str(row.candidate_budget),
                    str(row.accepted_endpoint_identity_count),
                    str(row.distinct_support_local),
                    str(row.strong_meaningful_pair_count),
                    str(row.moderate_meaningful_pair_count),
                    str(row.multi_basin_existence_status),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Pathway Readiness",
            "",
            "| status | rows |",
            "| --- | ---: |",
        ]
    )
    for status, count in summary["pathway_readiness_status_counts"].items():
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "Next step: " + str(summary["next_step"]),
            "",
            "Claim boundary: " + CLAIM_BOUNDARY,
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")

def run(
    *,
    calibration_dir: Path,
    remaining_audit_dir: Path,
    field34_audit_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows = _existence_pair_rows(calibration_dir)
    case_rows = _case_rows(calibration_dir, pair_rows)
    pathway_rows = _pathway_rows(
        remaining_audit_dir=remaining_audit_dir,
        field34_audit_dir=field34_audit_dir,
    )
    summary = _summary(
        case_rows=case_rows,
        pair_rows=pair_rows,
        pathway_rows=pathway_rows,
        output_dir=output_dir,
    )
    _write_csv(case_rows, output_dir / CASE_ROWS_CSV)
    _write_csv(pair_rows, output_dir / PAIR_ROWS_CSV)
    _write_csv(pathway_rows, output_dir / PATHWAY_ROWS_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            {
                "calibration_dir": _rel(calibration_dir),
                "remaining_audit_dir": _rel(remaining_audit_dir),
                "field34_audit_dir": _rel(field34_audit_dir),
                "output_dir": _rel(output_dir),
                "moderate_support_min": MODERATE_SUPPORT_MIN,
                "strong_support_min": STRONG_SUPPORT_MIN,
                "claim_boundary": CLAIM_BOUNDARY,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / REPORT_MD, summary, case_rows, pathway_rows)
    return summary

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--remaining-audit-dir", type=Path, default=DEFAULT_REMAINING_AUDIT_DIR)
    parser.add_argument("--field34-audit-dir", type=Path, default=DEFAULT_FIELD34_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser

def main() -> int:
    args = build_parser().parse_args()
    summary = run(
        calibration_dir=args.calibration_dir,
        remaining_audit_dir=args.remaining_audit_dir,
        field34_audit_dir=args.field34_audit_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
