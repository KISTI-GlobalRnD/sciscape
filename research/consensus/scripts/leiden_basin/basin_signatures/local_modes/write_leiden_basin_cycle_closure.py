#!/usr/bin/env python3
"""Write the current Track C Leiden basin cycle closure artifact.

The closure is deliberately scoped to the current 23-pair wall surface and its
blocker chain. It does not claim that the full 206-pair calibration universe has
no wall structure. It does not run routes or inspect basin quality/cost.
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

DEFAULT_CALIBRATION_DIR = BASE_RESULT_DIR / "leiden_basin_definition_calibration_20260528"
DEFAULT_CURRENT_REVIEW_DIR = BASE_RESULT_DIR / "leiden_basin_current_results_review_20260529"
DEFAULT_TRIAGE_DIR = BASE_RESULT_DIR / "leiden_basin_route_label_blocker_triage_20260529"
DEFAULT_BOUNDARY_REVIEW_DIR = BASE_RESULT_DIR / "leiden_basin_relation_boundary_rule_review_20260529"
DEFAULT_PENDING_REVIEW_DIR = (
    BASE_RESULT_DIR / "leiden_basin_pending_membership_relation_review_after_cache_materialization_20260529"
)
DEFAULT_FIELD34_AUDIT_DIR = BASE_RESULT_DIR / "leiden_basin_field34_evidence_eligibility_audit_20260529"
DEFAULT_REMAINING_AUDIT_DIR = BASE_RESULT_DIR / "leiden_basin_remaining_wall_question_audit_20260529"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_cycle_closure_writeup_20260529"

CLOSURE_SUMMARY_JSON = "track_c_cycle_closure_summary.json"
CLOSURE_REPORT_MD = "track_c_cycle_closure_report.md"
EVIDENCE_ROWS_CSV = "track_c_cycle_closure_evidence_rows.csv"
REOPEN_CONDITIONS_CSV = "track_c_cycle_reopen_conditions.csv"
CONFIG_JSON = "track_c_cycle_closure_config.json"

CLAIM_BOUNDARY = (
    "Cycle-closure write-up only; scoped to the current 23-pair wall surface "
    "and blocker chain. No route execution, wall-promotion change, basin-quality "
    "claim, cost claim, or directed-search claim."
)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _evidence_row(
    *,
    stage: str,
    artifact: Path,
    key_counts: dict[str, Any],
    decision: str,
    interpretation: str,
    closure_role: str,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "artifact": _rel(artifact),
        "key_counts": json.dumps(key_counts, sort_keys=True),
        "decision": decision,
        "interpretation": interpretation,
        "closure_role": closure_role,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _evidence_rows(
    *,
    calibration_dir: Path,
    current_review_dir: Path,
    triage_dir: Path,
    boundary_review_dir: Path,
    pending_review_dir: Path,
    field34_audit_dir: Path,
    remaining_audit_dir: Path,
) -> pd.DataFrame:
    calibration = _read_json(calibration_dir / "basin_definition_calibration_summary.json")
    current = _read_json(current_review_dir / "current_results_review_summary.json")
    triage = _read_json(triage_dir / "route_label_blocker_triage_summary.json")
    boundary = _read_json(boundary_review_dir / "relation_boundary_rule_review_summary.json")
    pending = _read_json(pending_review_dir / "pending_membership_relation_review_summary.json")
    field34 = _read_json(field34_audit_dir / "field34_evidence_eligibility_summary.json")
    remaining = _read_json(remaining_audit_dir / "remaining_wall_question_summary.json")

    rows = [
        _evidence_row(
            stage="basin_definition_calibration",
            artifact=calibration_dir / "basin_definition_calibration_summary.json",
            key_counts={
                "identity_pair_rows": calibration["identity_pair_rows"],
                "wall_candidate_pair_rows": calibration["wall_candidate_pair_rows"],
                "route_join_candidate_pair_rows": calibration["route_join_candidate_pair_rows"],
                "ambiguous_identity_pair_rows": calibration["ambiguous_identity_pair_rows"],
            },
            decision=(
                "Use the same/distinct/ambiguous relation gate as a calibration "
                "surface, not as wall evidence."
            ),
            interpretation=(
                "The full calibration universe contains many wall candidates, but "
                "only a narrow route-evidence surface is currently instrumented."
            ),
            closure_role="defines_scope_boundary_not_closure_of_full_universe",
        ),
        _evidence_row(
            stage="current_23_pair_review",
            artifact=current_review_dir / "current_results_review_summary.json",
            key_counts={
                "pair_count": current["pair_count"],
                "route_gate_group_counts": current["route_gate_group_counts"],
                "hygiene_blocker_status_counts": current["hygiene_blocker_status_counts"],
            },
            decision=current["decision"],
            interpretation=(
                "The current 23-pair surface has references, blockers, controls, "
                "and no wall-promotion change."
            ),
            closure_role="freezes_current_wall_surface",
        ),
        _evidence_row(
            stage="route_label_blocker_triage",
            artifact=triage_dir / "route_label_blocker_triage_summary.json",
            key_counts={
                "pair_count": triage["pair_count"],
                "immediate_route_execution_count": triage["immediate_route_execution_count"],
                "relation_definition_queue_count": triage["relation_definition_queue_count"],
                "field34_hygiene_queue_count": triage["field34_hygiene_queue_count"],
                "wall_evidence_question_hold_queue_count": triage[
                    "wall_evidence_question_hold_queue_count"
                ],
            },
            decision=triage["decision"],
            interpretation="The next blockers were definition and hygiene, not execution.",
            closure_role="establishes_no_immediate_route_execution",
        ),
        _evidence_row(
            stage="relation_boundary_rule_review",
            artifact=boundary_review_dir / "relation_boundary_rule_review_summary.json",
            key_counts={
                "reviewed_pair_count": boundary["reviewed_pair_count"],
                "accepted_policy": boundary["accepted_policy"],
                "current_hard_gate_classification_counts": boundary[
                    "current_hard_gate_classification_counts"
                ],
                "promoted_wall_claim_count": boundary["promoted_wall_claim_count"],
            },
            decision=boundary["decision"],
            interpretation=(
                "Stable route evidence cannot snap basin relation under the accepted "
                "hard gate."
            ),
            closure_role="closes_route_stable_boundary_review_rows",
        ),
        _evidence_row(
            stage="pending_membership_relation_review",
            artifact=pending_review_dir / "pending_membership_relation_review_summary.json",
            key_counts={
                "reviewed_pair_count": pending["reviewed_pair_count"],
                "full_membership_cache_status_counts": pending[
                    "full_membership_cache_status_counts"
                ],
                "current_hard_gate_classification_counts": pending[
                    "current_hard_gate_classification_counts"
                ],
                "immediate_route_execution_count": pending["immediate_route_execution_count"],
                "promoted_wall_claim_count": pending["promoted_wall_claim_count"],
            },
            decision=pending["decision"],
            interpretation=(
                "Full membership evidence closes the cache gap but keeps both rows "
                "inside boundary review."
            ),
            closure_role="closes_pending_membership_blocker",
        ),
        _evidence_row(
            stage="field34_evidence_eligibility",
            artifact=field34_audit_dir / "field34_evidence_eligibility_summary.json",
            key_counts={
                "endpoint_row_count": field34["endpoint_row_count"],
                "method_count": field34["method_count"],
                "queue_row_count": field34["queue_row_count"],
                "route_gate_candidate_count": field34["route_gate_candidate_count"],
                "immediate_route_execution_count": field34["immediate_route_execution_count"],
                "promoted_wall_claim_count": field34["promoted_wall_claim_count"],
            },
            decision=field34["decision"],
            interpretation=(
                "Field34 remains diagnostic/reference evidence, not a clean "
                "calibration or route-gate source."
            ),
            closure_role="closes_field34_hygiene_blocker",
        ),
        _evidence_row(
            stage="remaining_wall_question_audit",
            artifact=remaining_audit_dir / "remaining_wall_question_summary.json",
            key_counts={
                "non_field34_pair_count": remaining["non_field34_pair_count"],
                "non_field34_executable_route_candidate_count": remaining[
                    "non_field34_executable_route_candidate_count"
                ],
                "non_field34_wall_promotion_candidate_count": remaining[
                    "non_field34_wall_promotion_candidate_count"
                ],
                "field34_route_gate_candidate_count": remaining[
                    "field34_route_gate_candidate_count"
                ],
            },
            decision=remaining["decision"],
            interpretation=(
                "The current wall-question surface has no executable route candidate "
                "under the fixed gates."
            ),
            closure_role="current_cycle_closure_decision",
        ),
    ]
    return pd.DataFrame(rows)


def _reopen_conditions() -> pd.DataFrame:
    rows = [
        {
            "condition_id": "R1",
            "condition": "Reopen basin-relation boundary band",
            "required_precommitment": (
                "Define a new same/ambiguous/distinct rule before route execution, "
                "including how near-same and near-distinct cases are treated."
            ),
            "allowed_work": "definition audit, counterfactual relation table, sensitivity report",
            "forbidden_shortcut": "using route stability or quality/cost to snap relation labels post hoc",
            "status": "allowed_reopen_as_definition_problem_only",
        },
        {
            "condition_id": "R2",
            "condition": "Build a new non-field34 panel from the 206 wall-candidate universe",
            "required_precommitment": (
                "Predeclare sampling, graph-context requirements, and wall-evidence "
                "requirements; treat it as a new panel, not as continuation of the "
                "closed 23-pair cycle."
            ),
            "allowed_work": "panel construction and preflight audit",
            "forbidden_shortcut": "claiming the current 23-pair closure covers all 206 candidates",
            "status": "future_work_if_new_panel_is_declared",
        },
        {
            "condition_id": "R3",
            "condition": "Upgrade protocol references into stronger wall evidence",
            "required_precommitment": (
                "Define extra wall evidence such as objective debt, failed direct path, "
                "support incompatibility, or polish reversion before promotion."
            ),
            "allowed_work": "wall-evidence requirement design and artifact contract",
            "forbidden_shortcut": "promoting partial-wall protocol rows to supported wall claims",
            "status": "allowed_as_protocol_design_not_execution",
        },
        {
            "condition_id": "R4",
            "condition": "Evaluate basin quality or cost",
            "required_precommitment": (
                "Require accepted basin relation and supported wall evidence first."
            ),
            "allowed_work": "none in the current closed cycle",
            "forbidden_shortcut": "joining quality/cost to define basins or walls",
            "status": "blocked_until_wall_evidence_exists",
        },
    ]
    return pd.DataFrame(rows)


def _summary(
    *,
    evidence_rows: pd.DataFrame,
    reopen_conditions: pd.DataFrame,
    remaining_audit_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    remaining = _read_json(remaining_audit_dir / "remaining_wall_question_summary.json")
    return {
        "status": "track_c_cycle_closure_writeup_prepared",
        "date": "2026-05-29",
        "script": _rel(Path(__file__).resolve()),
        "output_dir": _rel(output_dir),
        "cycle_scope": "current_23_pair_wall_surface_and_blocker_chain",
        "closed_claim": (
            "Under the fixed current gates, the current Track C wall-evidence "
            "cycle has no executable route candidate and no wall-promotion candidate."
        ),
        "not_claimed": [
            "No claim that the full 206-pair calibration universe has no walls.",
            "No basin-quality or cost claim.",
            "No directed-search or operator-success claim.",
            "No route execution or wall-promotion change.",
        ],
        "non_field34_executable_route_candidate_count": remaining[
            "non_field34_executable_route_candidate_count"
        ],
        "non_field34_wall_promotion_candidate_count": remaining[
            "non_field34_wall_promotion_candidate_count"
        ],
        "field34_route_gate_candidate_count": remaining["field34_route_gate_candidate_count"],
        "evidence_row_count": int(len(evidence_rows)),
        "reopen_condition_count": int(len(reopen_conditions)),
        "recommended_next_work": (
            "Write the Track C closure as basin-definition and wall-protocol evidence. "
            "Only reopen by changing the basin-relation boundary definition or by "
            "declaring a new precommitted non-field34 panel from the broader universe."
        ),
        "paths": {
            "summary": _rel(output_dir / CLOSURE_SUMMARY_JSON),
            "report": _rel(output_dir / CLOSURE_REPORT_MD),
            "evidence_rows": _rel(output_dir / EVIDENCE_ROWS_CSV),
            "reopen_conditions": _rel(output_dir / REOPEN_CONDITIONS_CSV),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_report(
    path: Path,
    *,
    summary: dict[str, Any],
    evidence_rows: pd.DataFrame,
    reopen_conditions: pd.DataFrame,
) -> None:
    lines = [
        "# Track C Cycle Closure Write-up",
        "",
        "Date: 2026-05-29",
        "",
        "## Closure Claim",
        "",
        str(summary["closed_claim"]),
        "",
        "This closes the current 23-pair wall surface and blocker chain. It does",
        "not close the full 206-pair calibration universe.",
        "",
        "## What Is Not Claimed",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["not_claimed"])
    lines.extend(
        [
            "",
            "## Evidence Chain",
            "",
            "| stage | closure role | key counts | decision |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in evidence_rows.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.stage),
                    str(row.closure_role),
                    str(row.key_counts).replace("|", "/"),
                    str(row.decision).replace("|", "/"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Reopen Conditions",
            "",
            "| id | condition | required precommitment | status |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in reopen_conditions.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.condition_id),
                    str(row.condition),
                    str(row.required_precommitment).replace("|", "/"),
                    str(row.status),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Recommended Next Work",
            "",
            str(summary["recommended_next_work"]),
            "",
            "Claim boundary: " + CLAIM_BOUNDARY,
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    calibration_dir: Path,
    current_review_dir: Path,
    triage_dir: Path,
    boundary_review_dir: Path,
    pending_review_dir: Path,
    field34_audit_dir: Path,
    remaining_audit_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_rows = _evidence_rows(
        calibration_dir=calibration_dir,
        current_review_dir=current_review_dir,
        triage_dir=triage_dir,
        boundary_review_dir=boundary_review_dir,
        pending_review_dir=pending_review_dir,
        field34_audit_dir=field34_audit_dir,
        remaining_audit_dir=remaining_audit_dir,
    )
    reopen_conditions = _reopen_conditions()
    summary = _summary(
        evidence_rows=evidence_rows,
        reopen_conditions=reopen_conditions,
        remaining_audit_dir=remaining_audit_dir,
        output_dir=output_dir,
    )

    _write_csv(evidence_rows, output_dir / EVIDENCE_ROWS_CSV)
    _write_csv(reopen_conditions, output_dir / REOPEN_CONDITIONS_CSV)
    (output_dir / CLOSURE_SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(
            {
                "calibration_dir": _rel(calibration_dir),
                "current_review_dir": _rel(current_review_dir),
                "triage_dir": _rel(triage_dir),
                "boundary_review_dir": _rel(boundary_review_dir),
                "pending_review_dir": _rel(pending_review_dir),
                "field34_audit_dir": _rel(field34_audit_dir),
                "remaining_audit_dir": _rel(remaining_audit_dir),
                "output_dir": _rel(output_dir),
                "claim_boundary": CLAIM_BOUNDARY,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir / CLOSURE_REPORT_MD,
        summary=summary,
        evidence_rows=evidence_rows,
        reopen_conditions=reopen_conditions,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--current-review-dir", type=Path, default=DEFAULT_CURRENT_REVIEW_DIR)
    parser.add_argument("--triage-dir", type=Path, default=DEFAULT_TRIAGE_DIR)
    parser.add_argument("--boundary-review-dir", type=Path, default=DEFAULT_BOUNDARY_REVIEW_DIR)
    parser.add_argument("--pending-review-dir", type=Path, default=DEFAULT_PENDING_REVIEW_DIR)
    parser.add_argument("--field34-audit-dir", type=Path, default=DEFAULT_FIELD34_AUDIT_DIR)
    parser.add_argument("--remaining-audit-dir", type=Path, default=DEFAULT_REMAINING_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run(
        calibration_dir=args.calibration_dir,
        current_review_dir=args.current_review_dir,
        triage_dir=args.triage_dir,
        boundary_review_dir=args.boundary_review_dir,
        pending_review_dir=args.pending_review_dir,
        field34_audit_dir=args.field34_audit_dir,
        remaining_audit_dir=args.remaining_audit_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
