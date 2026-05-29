#!/usr/bin/env python3
"""Review the relation boundary rule for route-stable blocked Leiden basin rows.

The review focuses on the three highest-priority rows that have stable route
evidence but are blocked from wall promotion by a near-threshold basin relation.
It tests whether the current boundary-review rule should be relaxed. It does
not run routes, change wall-promotion gates, or inspect basin quality/cost.
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
DEFAULT_BLOCKER_TRIAGE_DIR = BASE_RESULT_DIR / "leiden_basin_route_label_blocker_triage_20260529"
DEFAULT_STABLE_REFINEMENT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_stable_ambiguous_relation_refinement_20260528"
)
DEFAULT_TAXONOMY_DIR = BASE_RESULT_DIR / "leiden_basin_relation_taxonomy_v01_20260528"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_relation_boundary_rule_review_20260529"

RELATION_QUEUE_CSV = "relation_definition_queue.csv"
STABLE_REFINEMENT_CSV = "stable_ambiguous_relation_refinement_rows.csv"
TAXONOMY_ROWS_CSV = "basin_relation_taxonomy_rows.csv"

REVIEW_ROWS_CSV = "relation_boundary_rule_review_rows.csv"
COUNTERFACTUALS_CSV = "relation_boundary_rule_counterfactuals.csv"
OPTIONS_CSV = "relation_boundary_rule_options.csv"
SUMMARY_JSON = "relation_boundary_rule_review_summary.json"
REPORT_MD = "relation_boundary_rule_review_report.md"
CONFIG_JSON = "relation_boundary_rule_review_config.json"

REVIEW_VERSION = "relation_boundary_rule_review_20260529"
SAME_SUPPORT_MAX = 0.5
DISTINCT_SUPPORT_MIN = 0.75
CLAIM_BOUNDARY = (
    "Boundary-rule review only; no route execution, wall-promotion change, "
    "basin-quality claim, cost claim, or directed-search claim."
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


def _count(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    return {str(k): int(v) for k, v in frame[column].value_counts(dropna=False).to_dict().items()}


def _safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _current_classification(distance: float) -> str:
    if distance <= SAME_SUPPORT_MAX:
        return "same_support_local"
    if distance >= DISTINCT_SUPPORT_MIN:
        return "distinct_support_local"
    return "boundary_review_ambiguous_support_local"


def _epsilon_classification(distance: float, epsilon: float, two_sided: bool) -> str:
    if distance <= SAME_SUPPORT_MAX:
        return "same_support_local"
    if distance >= DISTINCT_SUPPORT_MIN:
        return "distinct_support_local"
    if two_sided and distance <= SAME_SUPPORT_MAX + epsilon:
        return "same_support_local_epsilon_snap"
    if distance >= DISTINCT_SUPPORT_MIN - epsilon:
        return "distinct_support_local_epsilon_snap"
    return "boundary_review_ambiguous_support_local"


def _boundary_status(row: pd.Series) -> str:
    band = str(row.get("ambiguous_band", ""))
    same_margin = _safe_float(row.get("same_threshold_margin"))
    distinct_margin = _safe_float(row.get("distinct_threshold_margin"))
    if band == "near_distinct":
        return f"near_distinct_below_threshold_margin_{distinct_margin:.6g}"
    if band == "near_same":
        return f"near_same_above_threshold_margin_{same_margin:.6g}"
    return "boundary_review"


def _review_decision(row: pd.Series) -> str:
    return "keep_boundary_review_no_route_promotion"


def _review_rationale(row: pd.Series) -> str:
    band = str(row.get("ambiguous_band", ""))
    if band == "near_distinct":
        return (
            "Exact membership evidence shows a different endpoint identity, but "
            "the support-local distance remains just below the predeclared "
            "distinct threshold; route stability cannot snap the relation."
        )
    if band == "near_same":
        return (
            "Exact membership evidence shows a different endpoint identity, but "
            "the support-local distance remains just above the predeclared same "
            "threshold; route stability cannot snap the relation."
        )
    return "The row remains boundary-reviewed under the current relation rule."


def _next_evidence(row: pd.Series) -> str:
    band = str(row.get("ambiguous_band", ""))
    if band == "near_distinct":
        return (
            "predeclare a boundary-band rule across all near-distinct rows, or "
            "add stronger membership/signature evidence before route promotion"
        )
    if band == "near_same":
        return (
            "predeclare a near-same boundary rule and prove it does not turn "
            "control-like rows into wall candidates"
        )
    return "keep in relation review until a global boundary rule is accepted"


def _review_rows(
    blocker_triage_dir: Path,
    stable_refinement_dir: Path,
    taxonomy_dir: Path,
) -> pd.DataFrame:
    relation_queue = _read_csv(blocker_triage_dir / RELATION_QUEUE_CSV)
    stable = _read_csv(stable_refinement_dir / STABLE_REFINEMENT_CSV)
    taxonomy = _read_csv(taxonomy_dir / TAXONOMY_ROWS_CSV)

    target_ids = set(
        relation_queue[
            relation_queue["relation_queue_status"].eq("route_evidence_relation_blocked")
        ]["panel_pair_id"].astype(str)
    )
    if not target_ids:
        raise ValueError("no route_evidence_relation_blocked rows found")

    stable_cols = [
        "panel_pair_id",
        "field",
        "case_id",
        "method",
        "ambiguous_band",
        "left_candidate_index",
        "right_candidate_index",
        "left_endpoint_identity_id",
        "right_endpoint_identity_id",
        "membership_hash_match",
        "left_support_node_count_exact",
        "right_support_node_count_exact",
        "exact_support_intersection",
        "exact_support_union",
        "exact_support_distance",
        "same_threshold_margin",
        "distinct_threshold_margin",
        "left_right_label_aligned_changed_node_count",
        "left_right_label_aligned_changed_fraction",
        "support_union_endpoint_distance",
        "stable_route_evidence_status",
        "identity_refinement_status",
        "route_promotion_status",
        "evidence_grade",
    ]
    taxonomy_cols = [
        "panel_pair_id",
        "panel_role",
        "current_calibrated_relation",
        "relation_taxonomy_v0_1",
        "taxonomy_next_action",
        "wall_promotion_status",
        "taxonomy_reason",
    ]
    queue_cols = [
        "panel_pair_id",
        "route_label_interpretation_v0",
        "relation_queue_status",
        "triage_action",
        "runner_preflight_status",
        "hygiene_blocker_status",
    ]
    rows = stable[stable["panel_pair_id"].astype(str).isin(target_ids)][stable_cols].merge(
        taxonomy[taxonomy_cols],
        on="panel_pair_id",
        how="left",
    )
    rows = rows.merge(relation_queue[queue_cols], on="panel_pair_id", how="left")
    rows["current_hard_gate_classification"] = rows["exact_support_distance"].map(
        _current_classification
    )
    rows["epsilon_0p001_distinct_only_classification"] = rows["exact_support_distance"].map(
        lambda value: _epsilon_classification(value, epsilon=0.001, two_sided=False)
    )
    rows["epsilon_0p01_two_sided_classification"] = rows["exact_support_distance"].map(
        lambda value: _epsilon_classification(value, epsilon=0.01, two_sided=True)
    )
    rows["endpoint_identity_status"] = rows["membership_hash_match"].map(
        lambda match: "same_membership_hash" if bool(match) else "different_membership_hash"
    )
    rows["support_local_boundary_status"] = rows.apply(_boundary_status, axis=1)
    rows["boundary_rule_review_decision"] = rows.apply(_review_decision, axis=1)
    rows["route_promotion_status_after_review"] = "blocked_until_predeclared_relation_rule"
    rows["wall_promotion_status_after_review"] = "no_wall_promotion"
    rows["review_rationale"] = rows.apply(_review_rationale, axis=1)
    rows["next_evidence_required"] = rows.apply(_next_evidence, axis=1)
    rows["review_version"] = REVIEW_VERSION
    rows["claim_boundary"] = CLAIM_BOUNDARY
    rows["source_relation_queue_artifact"] = _rel(blocker_triage_dir / RELATION_QUEUE_CSV)
    rows["source_stable_refinement_artifact"] = _rel(
        stable_refinement_dir / STABLE_REFINEMENT_CSV
    )
    rows["source_taxonomy_artifact"] = _rel(taxonomy_dir / TAXONOMY_ROWS_CSV)

    preferred_cols = [
        "panel_pair_id",
        "field",
        "case_id",
        "method",
        "panel_role",
        "ambiguous_band",
        "current_calibrated_relation",
        "relation_taxonomy_v0_1",
        "route_label_interpretation_v0",
        "stable_route_evidence_status",
        "endpoint_identity_status",
        "membership_hash_match",
        "left_endpoint_identity_id",
        "right_endpoint_identity_id",
        "left_support_node_count_exact",
        "right_support_node_count_exact",
        "exact_support_intersection",
        "exact_support_union",
        "exact_support_distance",
        "same_threshold_margin",
        "distinct_threshold_margin",
        "left_right_label_aligned_changed_node_count",
        "left_right_label_aligned_changed_fraction",
        "support_union_endpoint_distance",
        "current_hard_gate_classification",
        "epsilon_0p001_distinct_only_classification",
        "epsilon_0p01_two_sided_classification",
        "support_local_boundary_status",
        "identity_refinement_status",
        "evidence_grade",
        "boundary_rule_review_decision",
        "route_promotion_status_after_review",
        "wall_promotion_status_after_review",
        "review_rationale",
        "next_evidence_required",
        "review_version",
        "claim_boundary",
        "source_relation_queue_artifact",
        "source_stable_refinement_artifact",
        "source_taxonomy_artifact",
    ]
    return rows[[col for col in preferred_cols if col in rows.columns]].sort_values(
        ["ambiguous_band", "field", "panel_pair_id"]
    ).reset_index(drop=True)


def _counterfactuals(rows: pd.DataFrame) -> pd.DataFrame:
    policies = [
        (
            "current_hard_gate",
            "Keep same <= 0.5 and distinct >= 0.75; middle rows stay boundary review.",
            "current_hard_gate_classification",
            "accepted",
        ),
        (
            "epsilon_0p001_distinct_only",
            "Snap near-distinct rows within 0.001 below 0.75 to distinct.",
            "epsilon_0p001_distinct_only_classification",
            "rejected",
        ),
        (
            "epsilon_0p01_two_sided",
            "Snap rows within 0.01 of either same or distinct threshold.",
            "epsilon_0p01_two_sided_classification",
            "rejected",
        ),
        (
            "route_stability_override",
            "Let stable route evidence override an ambiguous basin relation.",
            "",
            "rejected",
        ),
    ]
    out: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        for policy_id, policy_description, source_col, status in policies:
            if policy_id == "route_stability_override":
                classification = "route_promoted_despite_ambiguous_relation"
            else:
                classification = getattr(row, source_col)
            out.append(
                {
                    "panel_pair_id": row.panel_pair_id,
                    "ambiguous_band": row.ambiguous_band,
                    "exact_support_distance": row.exact_support_distance,
                    "policy_id": policy_id,
                    "policy_description": policy_description,
                    "counterfactual_classification": classification,
                    "policy_review_status": status,
                    "review_reason": (
                        "Accepted because it preserves the predeclared boundary-review class."
                        if status == "accepted"
                        else "Rejected because it changes basin relation from threshold proximity or route evidence rather than a predeclared relation rule."
                    ),
                }
            )
    return pd.DataFrame(out)


def _options(counterfactuals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy_id, frame in counterfactuals.groupby("policy_id", sort=False):
        status = str(frame["policy_review_status"].iloc[0])
        counts = frame["counterfactual_classification"].value_counts().to_dict()
        if policy_id == "current_hard_gate":
            decision = "accept_for_current_methodology"
            reason = (
                "It keeps basin relation independent from route evidence and preserves "
                "the predeclared same/distinct thresholds."
            )
        elif policy_id == "epsilon_0p001_distinct_only":
            decision = "reject_for_now"
            reason = (
                "It would promote the two near-distinct route-stable rows only because "
                "they are close to 0.75; this is threshold snapping after seeing route evidence."
            )
        elif policy_id == "epsilon_0p01_two_sided":
            decision = "reject_for_now"
            reason = (
                "It would also snap the near-same row to same-control. That may be useful "
                "as a future boundary-band hypothesis, but it must be predeclared and tested broadly."
            )
        else:
            decision = "reject"
            reason = "Route stability is wall/route evidence; it cannot define basin relation."
        rows.append(
            {
                "policy_id": policy_id,
                "policy_review_status": status,
                "policy_decision": decision,
                "counterfactual_counts": json.dumps(counts, sort_keys=True),
                "reason": reason,
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _summary(rows: pd.DataFrame, counterfactuals: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    accepted_policy = "current_hard_gate"
    promoted = int(rows["wall_promotion_status_after_review"].ne("no_wall_promotion").sum())
    return {
        "status": "relation_boundary_rule_review_prepared",
        "date": "2026-05-29",
        "script": "research/consensus/scripts/review_leiden_basin_relation_boundary_rule.py",
        "output_dir": _rel(output_dir),
        "review_version": REVIEW_VERSION,
        "reviewed_pair_count": int(len(rows)),
        "ambiguous_band_counts": _count(rows, "ambiguous_band"),
        "current_hard_gate_classification_counts": _count(
            rows, "current_hard_gate_classification"
        ),
        "epsilon_0p001_distinct_only_counts": _count(
            rows, "epsilon_0p001_distinct_only_classification"
        ),
        "epsilon_0p01_two_sided_counts": _count(
            rows, "epsilon_0p01_two_sided_classification"
        ),
        "boundary_rule_review_decision_counts": _count(rows, "boundary_rule_review_decision"),
        "accepted_policy": accepted_policy,
        "promoted_wall_claim_count": promoted,
        "decision": (
            "Keep the current hard same/distinct relation gate and boundary-review "
            "class. Do not snap near-threshold rows or let route stability define "
            "basin relation. The three route-stable rows remain blocked from wall promotion."
        ),
        "next_step": (
            "Review pending-membership relation checks before any route batch; "
            "field34 hygiene remains a separate blocker."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "paths": {
            "review_rows": _rel(output_dir / REVIEW_ROWS_CSV),
            "counterfactuals": _rel(output_dir / COUNTERFACTUALS_CSV),
            "options": _rel(output_dir / OPTIONS_CSV),
            "summary": _rel(output_dir / SUMMARY_JSON),
            "report": _rel(output_dir / REPORT_MD),
        },
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rendered_rows = [
        ["" if pd.isna(value) else str(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    widths = [
        max(len(str(column)), *(len(row[index]) for row in rendered_rows))
        for index, column in enumerate(columns)
    ]
    header = "| " + " | ".join(
        str(column).ljust(widths[index]) for index, column in enumerate(columns)
    ) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(columns))) + " |"
        for row in rendered_rows
    ]
    return "\n".join([header, separator, *body])


def _report(rows: pd.DataFrame, options: pd.DataFrame, summary: dict[str, Any]) -> str:
    lines = [
        "# Relation Boundary Rule Review",
        "",
        "Date: 2026-05-29",
        "",
        "## Scope",
        "",
        "This review targets the three route-stable rows that remain blocked by",
        "near-threshold basin relation. It compares the current hard gate against",
        "threshold snapping and route-stability override policies.",
        "",
        "## Decision",
        "",
        str(summary["decision"]),
        "",
        "Claim boundary: " + CLAIM_BOUNDARY,
        "",
        "## Reviewed Rows",
        "",
    ]
    row_cols = [
        "panel_pair_id",
        "ambiguous_band",
        "exact_support_distance",
        "same_threshold_margin",
        "distinct_threshold_margin",
        "current_hard_gate_classification",
        "boundary_rule_review_decision",
    ]
    lines.append(_markdown_table(rows[row_cols]))
    lines.extend(["", "## Policy Options", ""])
    option_cols = [
        "policy_id",
        "policy_decision",
        "counterfactual_counts",
        "reason",
    ]
    lines.append(_markdown_table(options[option_cols]))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The exact membership hashes differ, so these are not label-namespace",
            "duplicates. But the support-local distances remain inside the",
            "predeclared middle zone. The relation rule should therefore keep them",
            "as boundary-review rows until a boundary-band rule is accepted across",
            "the broader relation surface. Route stability remains route evidence,",
            "not basin-relation evidence.",
            "",
            "Next step: " + str(summary["next_step"]),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocker-triage-dir", type=Path, default=DEFAULT_BLOCKER_TRIAGE_DIR)
    parser.add_argument(
        "--stable-refinement-dir",
        type=Path,
        default=DEFAULT_STABLE_REFINEMENT_DIR,
    )
    parser.add_argument("--taxonomy-dir", type=Path, default=DEFAULT_TAXONOMY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _review_rows(args.blocker_triage_dir, args.stable_refinement_dir, args.taxonomy_dir)
    counterfactuals = _counterfactuals(rows)
    options = _options(counterfactuals)
    summary = _summary(rows, counterfactuals, output_dir)
    config = {
        "blocker_triage_dir": _rel(args.blocker_triage_dir),
        "stable_refinement_dir": _rel(args.stable_refinement_dir),
        "taxonomy_dir": _rel(args.taxonomy_dir),
        "output_dir": _rel(output_dir),
        "same_support_max": SAME_SUPPORT_MAX,
        "distinct_support_min": DISTINCT_SUPPORT_MIN,
        "review_version": REVIEW_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
    }

    _write_csv(rows, output_dir / REVIEW_ROWS_CSV)
    _write_csv(counterfactuals, output_dir / COUNTERFACTUALS_CSV)
    _write_csv(options, output_dir / OPTIONS_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / REPORT_MD).write_text(_report(rows, options, summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
