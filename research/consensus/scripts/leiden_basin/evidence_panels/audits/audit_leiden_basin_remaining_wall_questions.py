#!/usr/bin/env python3
"""Audit remaining non-field34 wall-evidence questions for Leiden basin Track C.

This closes the current blocker sequence after relation-boundary review,
pending-membership cache materialization, and field34 hygiene. It asks whether
any non-field34 wall-evidence question remains executable under the fixed basin
relation gates. It does not run routes, change wall promotion, or inspect basin
quality, cost, ranking, or operator success.
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
DEFAULT_TRIAGE_DIR = BASE_RESULT_DIR / "leiden_basin_route_label_blocker_triage_20260529"
DEFAULT_CURRENT_REVIEW_DIR = BASE_RESULT_DIR / "leiden_basin_current_results_review_20260529"
DEFAULT_BOUNDARY_REVIEW_DIR = BASE_RESULT_DIR / "leiden_basin_relation_boundary_rule_review_20260529"
DEFAULT_PENDING_REVIEW_DIR = (
    BASE_RESULT_DIR / "leiden_basin_pending_membership_relation_review_after_cache_materialization_20260529"
)
DEFAULT_FIELD34_AUDIT_DIR = BASE_RESULT_DIR / "leiden_basin_field34_evidence_eligibility_audit_20260529"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_remaining_wall_question_audit_20260529"

TRIAGE_ROWS_CSV = "route_label_blocker_triage_rows.csv"
CURRENT_PAIR_ROWS_CSV = "current_pair_state_ledger.csv"
BOUNDARY_REVIEW_ROWS_CSV = "relation_boundary_rule_review_rows.csv"
PENDING_REVIEW_ROWS_CSV = "pending_membership_relation_review_rows.csv"
FIELD34_QUEUE_ROWS_CSV = "field34_queue_projection_rows.csv"

REMAINING_ROWS_CSV = "remaining_wall_question_rows.csv"
DECISION_COUNTS_CSV = "remaining_wall_question_decision_counts.csv"
SUMMARY_JSON = "remaining_wall_question_summary.json"
REPORT_MD = "remaining_wall_question_report.md"
CONFIG_JSON = "remaining_wall_question_config.json"

REVIEW_VERSION = "remaining_wall_question_audit_20260529"
CLAIM_BOUNDARY = (
    "Remaining wall-question audit only; no route execution, wall-promotion "
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


def _decision_for_row(
    row: pd.Series,
    *,
    boundary_lookup: dict[str, dict[str, Any]],
    pending_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    panel_pair_id = _safe_str(row.get("panel_pair_id"))
    route_label = _safe_str(row.get("route_label_interpretation_v0"))
    relation_queue = _safe_str(row.get("relation_queue_status"))
    primary = _safe_str(row.get("primary_blocker_class"))
    taxonomy = _safe_str(row.get("relation_taxonomy_v0_1"))
    relation = _safe_str(row.get("calibrated_relation"))
    runner = _safe_str(row.get("runner_preflight_status"))

    if primary == "basin_relation_definition_blocker":
        if relation_queue == "route_evidence_relation_blocked":
            boundary = boundary_lookup.get(panel_pair_id, {})
            return {
                "remaining_wall_question_class": "closed_by_current_boundary_rule",
                "decision_basis": _safe_str(
                    boundary.get("boundary_rule_review_decision", "boundary_review_missing")
                ),
                "route_execution_decision": "not_recommended",
                "wall_promotion_decision": "no_wall_promotion",
                "next_allowed_work": "reopen_boundary_band_definition_only_if_explicit",
                "decision_reason": (
                    "Stable route evidence exists, but the accepted hard relation gate "
                    "keeps the pair in boundary review."
                ),
            }
        if relation_queue == "pending_membership_relation_check":
            pending = pending_lookup.get(panel_pair_id, {})
            return {
                "remaining_wall_question_class": "closed_by_cached_pending_membership_boundary_review",
                "decision_basis": _safe_str(
                    pending.get("review_decision", "pending_membership_review_missing")
                ),
                "route_execution_decision": "not_recommended",
                "wall_promotion_decision": "no_wall_promotion",
                "next_allowed_work": "reopen_boundary_band_definition_only_if_explicit",
                "decision_reason": (
                    "Full memberships are cached, but the row remains inside the "
                    "predeclared middle zone under the hard relation gate."
                ),
            }
        if "middle_ambiguous" in taxonomy:
            return {
                "remaining_wall_question_class": "middle_ambiguous_definition_hold",
                "decision_basis": "middle_ambiguous_support_local_hold",
                "route_execution_decision": "not_recommended",
                "wall_promotion_decision": "no_wall_promotion",
                "next_allowed_work": "stronger_relation_evidence_or_definition_redesign",
                "decision_reason": (
                    "The pair is not a current wall candidate because its basin relation "
                    "is still middle ambiguous and no accepted wall route is attached."
                ),
            }
        return {
            "remaining_wall_question_class": "relation_definition_hold",
            "decision_basis": relation_queue or taxonomy,
            "route_execution_decision": "not_recommended",
            "wall_promotion_decision": "no_wall_promotion",
            "next_allowed_work": "relation_definition_review_only",
            "decision_reason": "Basin relation is not accepted as distinct under the current gate.",
        }

    if route_label == "partial_wall_protocol_evidence":
        return {
            "remaining_wall_question_class": "protocol_reference_only",
            "decision_basis": "partial_wall_protocol_evidence",
            "route_execution_decision": "not_recommended",
            "wall_promotion_decision": "no_wall_promotion",
            "next_allowed_work": "write_protocol_constraints_or_predeclare_extra_wall_requirement",
            "decision_reason": (
                "The row is a conservative partial-wall protocol reference, not a "
                "supported wall claim."
            ),
        }

    if route_label == "boundary_sensitive_route_uncertainty":
        return {
            "remaining_wall_question_class": "route_uncertainty_reference_only",
            "decision_basis": "validated_boundary_sensitive_route_uncertainty",
            "route_execution_decision": "not_recommended",
            "wall_promotion_decision": "no_wall_promotion",
            "next_allowed_work": "retain_uncertainty_class_as_reference",
            "decision_reason": (
                "Held-out validation supports uncertainty wording, not wall promotion "
                "or another route batch."
            ),
        }

    if route_label in {"hard_support_loss_no_wall_contrast", "mixed_support_loss_no_wall_hold"}:
        return {
            "remaining_wall_question_class": "no_wall_contrast_reference_only",
            "decision_basis": route_label,
            "route_execution_decision": "not_recommended",
            "wall_promotion_decision": "no_wall_promotion",
            "next_allowed_work": "retain_as_no_wall_contrast",
            "decision_reason": "The row is already a no-wall contrast or hold.",
        }

    if route_label == "same_control_no_wall" or relation in {"same_support_local", "same_endpoint_identity"}:
        return {
            "remaining_wall_question_class": "same_or_identity_control_only",
            "decision_basis": route_label or relation,
            "route_execution_decision": "not_recommended",
            "wall_promotion_decision": "no_wall_promotion",
            "next_allowed_work": "retain_as_control",
            "decision_reason": "Same/control relation blocks wall evidence by design.",
        }

    if (
        relation == "distinct_support_local"
        and runner == "runner_preflight_ready"
        and not route_label
    ):
        return {
            "remaining_wall_question_class": "unrun_distinct_candidate_found",
            "decision_basis": "distinct_runner_ready_without_route_label",
            "route_execution_decision": "candidate_requires_manual_mechanism_question",
            "wall_promotion_decision": "no_wall_promotion",
            "next_allowed_work": "manual_mechanism_review_before_route",
            "decision_reason": (
                "A runner-ready distinct non-field34 row without route evidence exists; "
                "this should be reviewed before closure."
            ),
        }

    return {
        "remaining_wall_question_class": "manual_review_hold",
        "decision_basis": primary or route_label or relation,
        "route_execution_decision": "not_recommended",
        "wall_promotion_decision": "no_wall_promotion",
        "next_allowed_work": "manual_review_only",
        "decision_reason": "No accepted route or wall action follows from the current gates.",
    }


def _remaining_rows(
    *,
    triage_dir: Path,
    current_review_dir: Path,
    boundary_review_dir: Path,
    pending_review_dir: Path,
) -> pd.DataFrame:
    triage = _read_csv(triage_dir / TRIAGE_ROWS_CSV)
    current = _read_csv(current_review_dir / CURRENT_PAIR_ROWS_CSV)
    boundary = _read_csv(boundary_review_dir / BOUNDARY_REVIEW_ROWS_CSV)
    pending = _read_csv(pending_review_dir / PENDING_REVIEW_ROWS_CSV)

    boundary_lookup = boundary.set_index("panel_pair_id").to_dict("index")
    pending_lookup = pending.set_index("panel_pair_id").to_dict("index")

    current_cols = [
        "panel_pair_id",
        "support_distance_max",
        "route_order_sensitivity_status",
        "wall_claim_gate_status",
        "current_review_status",
        "next_action",
        "review_comment",
    ]
    rows = triage[triage["field"].astype(str).ne("field34")].copy()
    rows = rows.merge(
        current[current_cols],
        on="panel_pair_id",
        how="left",
        suffixes=("", "_current"),
    )
    decisions = [
        _decision_for_row(row, boundary_lookup=boundary_lookup, pending_lookup=pending_lookup)
        for _, row in rows.iterrows()
    ]
    decision_frame = pd.DataFrame(decisions)
    rows = pd.concat([rows.reset_index(drop=True), decision_frame], axis=1)
    rows["is_executable_route_candidate"] = rows["remaining_wall_question_class"].eq(
        "unrun_distinct_candidate_found"
    )
    rows["is_wall_promotion_candidate"] = rows["wall_promotion_decision"].ne("no_wall_promotion")
    rows["review_version"] = REVIEW_VERSION
    rows["claim_boundary"] = CLAIM_BOUNDARY
    rows["source_triage_artifact"] = _rel(triage_dir / TRIAGE_ROWS_CSV)
    rows["source_current_review_artifact"] = _rel(current_review_dir / CURRENT_PAIR_ROWS_CSV)
    rows["source_boundary_review_artifact"] = _rel(boundary_review_dir / BOUNDARY_REVIEW_ROWS_CSV)
    rows["source_pending_review_artifact"] = _rel(pending_review_dir / PENDING_REVIEW_ROWS_CSV)

    preferred_cols = [
        "panel_pair_id",
        "field",
        "case_id",
        "panel_role",
        "calibrated_relation",
        "relation_taxonomy_v0_1",
        "runner_preflight_status",
        "route_label_interpretation_v0",
        "wall_claim_gate_status",
        "primary_blocker_class",
        "relation_queue_status",
        "wall_evidence_question_status",
        "current_review_status",
        "remaining_wall_question_class",
        "decision_basis",
        "route_execution_decision",
        "wall_promotion_decision",
        "is_executable_route_candidate",
        "is_wall_promotion_candidate",
        "next_allowed_work",
        "decision_reason",
        "review_comment",
        "review_version",
        "claim_boundary",
        "source_triage_artifact",
        "source_current_review_artifact",
        "source_boundary_review_artifact",
        "source_pending_review_artifact",
    ]
    return rows[preferred_cols].sort_values(
        ["is_executable_route_candidate", "remaining_wall_question_class", "panel_pair_id"],
        ascending=[False, True, True],
    )


def _decision_counts(rows: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["remaining_wall_question_class", "route_execution_decision", "wall_promotion_decision"]
    return (
        rows.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values(["remaining_wall_question_class", "row_count"])
        .reset_index(drop=True)
    )


def _summary(
    *,
    remaining_rows: pd.DataFrame,
    decision_counts: pd.DataFrame,
    field34_rows: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "status": "remaining_wall_question_audit_prepared",
        "date": "2026-05-29",
        "script": "research/consensus/scripts/audit_leiden_basin_remaining_wall_questions.py",
        "output_dir": _rel(output_dir),
        "non_field34_pair_count": int(len(remaining_rows)),
        "field34_queue_pair_count": int(len(field34_rows)),
        "non_field34_executable_route_candidate_count": int(
            remaining_rows["is_executable_route_candidate"].sum()
        ),
        "non_field34_wall_promotion_candidate_count": int(
            remaining_rows["is_wall_promotion_candidate"].sum()
        ),
        "field34_route_gate_candidate_count": int(
            field34_rows["field34_fixture_decision"]
            .eq("hygiene_pass_route_gate_candidate_with_field34_caution")
            .sum()
        ),
        "field34_immediate_route_execution_count": int(
            field34_rows["route_execution_status_after_hygiene"].eq("ready").sum()
        ),
        "field34_promoted_wall_claim_count": int(
            field34_rows["wall_promotion_status_after_hygiene"].ne("no_wall_promotion").sum()
        ),
        "remaining_wall_question_class_counts": _count(
            remaining_rows,
            "remaining_wall_question_class",
        ),
        "primary_blocker_counts": _count(remaining_rows, "primary_blocker_class"),
        "decision_counts": decision_counts.to_dict(orient="records"),
        "decision": (
            "Under the fixed current gates, there is no executable non-field34 "
            "wall-evidence route candidate and no wall-promotion candidate. Field34 "
            "is also closed as reference/hold/filtered evidence. The current Track C "
            "cycle should not continue with another route batch unless the "
            "boundary-band definition is explicitly reopened."
        ),
        "next_step": (
            "Close the current cycle as basin-definition and wall-protocol evidence, "
            "or explicitly reopen the basin-relation boundary rule as a definition "
            "problem before any route execution."
        ),
        "paths": {
            "remaining_rows": _rel(output_dir / REMAINING_ROWS_CSV),
            "decision_counts": _rel(output_dir / DECISION_COUNTS_CSV),
            "summary": _rel(output_dir / SUMMARY_JSON),
            "report": _rel(output_dir / REPORT_MD),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    path: Path,
    summary: dict[str, Any],
    decision_counts: pd.DataFrame,
    rows: pd.DataFrame,
) -> None:
    def cell(value: Any) -> str:
        text = _safe_str(value)
        return "" if text == "nan" else text

    lines = [
        "# Remaining Wall-Question Audit",
        "",
        "Date: 2026-05-29",
        "",
        "## Scope",
        "",
        "This artifact audits whether any non-field34 wall-evidence question remains",
        "executable under the fixed current basin-relation gates. It also carries the",
        "field34 hygiene closure forward as a separate non-executable condition.",
        "",
        "## Decision",
        "",
        str(summary["decision"]),
        "",
        "## Non-Field34 Decision Counts",
        "",
        "| remaining class | route execution | wall promotion | rows |",
        "| --- | --- | --- | ---: |",
    ]
    for row in decision_counts.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.remaining_wall_question_class),
                    str(row.route_execution_decision),
                    str(row.wall_promotion_decision),
                    str(row.row_count),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Non-Field34 Pair Decisions",
            "",
            "| panel_pair_id | route label | remaining class | decision |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in rows.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    cell(row.panel_pair_id),
                    cell(row.route_label_interpretation_v0),
                    cell(row.remaining_wall_question_class),
                    cell(row.route_execution_decision),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Summary Counts",
            "",
            f"- non-field34 pairs: `{summary['non_field34_pair_count']}`",
            f"- executable non-field34 route candidates: `{summary['non_field34_executable_route_candidate_count']}`",
            f"- non-field34 wall-promotion candidates: `{summary['non_field34_wall_promotion_candidate_count']}`",
            f"- field34 route-gate candidates: `{summary['field34_route_gate_candidate_count']}`",
            f"- field34 immediate route executions: `{summary['field34_immediate_route_execution_count']}`",
            f"- field34 promoted wall claims: `{summary['field34_promoted_wall_claim_count']}`",
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
    triage_dir: Path,
    current_review_dir: Path,
    boundary_review_dir: Path,
    pending_review_dir: Path,
    field34_audit_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    remaining_rows = _remaining_rows(
        triage_dir=triage_dir,
        current_review_dir=current_review_dir,
        boundary_review_dir=boundary_review_dir,
        pending_review_dir=pending_review_dir,
    )
    field34_rows = _read_csv(field34_audit_dir / FIELD34_QUEUE_ROWS_CSV)
    decision_counts = _decision_counts(remaining_rows)
    summary = _summary(
        remaining_rows=remaining_rows,
        decision_counts=decision_counts,
        field34_rows=field34_rows,
        output_dir=output_dir,
    )

    _write_csv(remaining_rows, output_dir / REMAINING_ROWS_CSV)
    _write_csv(decision_counts, output_dir / DECISION_COUNTS_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            {
                "triage_dir": _rel(triage_dir),
                "current_review_dir": _rel(current_review_dir),
                "boundary_review_dir": _rel(boundary_review_dir),
                "pending_review_dir": _rel(pending_review_dir),
                "field34_audit_dir": _rel(field34_audit_dir),
                "output_dir": _rel(output_dir),
                "claim_boundary": CLAIM_BOUNDARY,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / REPORT_MD, summary, decision_counts, remaining_rows)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triage-dir", type=Path, default=DEFAULT_TRIAGE_DIR)
    parser.add_argument("--current-review-dir", type=Path, default=DEFAULT_CURRENT_REVIEW_DIR)
    parser.add_argument("--boundary-review-dir", type=Path, default=DEFAULT_BOUNDARY_REVIEW_DIR)
    parser.add_argument("--pending-review-dir", type=Path, default=DEFAULT_PENDING_REVIEW_DIR)
    parser.add_argument("--field34-audit-dir", type=Path, default=DEFAULT_FIELD34_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run(
        triage_dir=args.triage_dir,
        current_review_dir=args.current_review_dir,
        boundary_review_dir=args.boundary_review_dir,
        pending_review_dir=args.pending_review_dir,
        field34_audit_dir=args.field34_audit_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
