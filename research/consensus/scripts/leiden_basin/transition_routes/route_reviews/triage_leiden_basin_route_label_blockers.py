#!/usr/bin/env python3
"""Triage Track C blockers after freezing route-label interpretation v0.

This script consumes the reconciled 23-pair surface plus the frozen 11-pair
route-label interpretation. It separates the next blockers into basin-relation
definition, field34 hygiene, and wall-evidence question holds. It does not run
routes, relax wall-promotion gates, or inspect basin quality/cost.
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
DEFAULT_CURRENT_REVIEW_DIR = BASE_RESULT_DIR / "leiden_basin_current_results_review_20260529"
DEFAULT_ROUTE_LABEL_DIR = BASE_RESULT_DIR / "leiden_basin_route_label_interpretation_v0_20260529"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_route_label_blocker_triage_20260529"

CURRENT_PAIR_STATE_CSV = "current_pair_state_ledger.csv"
ROUTE_LABEL_ROWS_CSV = "route_label_interpretation_rows.csv"

TRIAGE_ROWS_CSV = "route_label_blocker_triage_rows.csv"
RELATION_QUEUE_CSV = "relation_definition_queue.csv"
FIELD34_QUEUE_CSV = "field34_hygiene_queue.csv"
WALL_QUESTION_QUEUE_CSV = "wall_evidence_question_hold_queue.csv"
COUNTS_CSV = "route_label_blocker_triage_counts.csv"
SUMMARY_JSON = "route_label_blocker_triage_summary.json"
REPORT_MD = "route_label_blocker_triage_report.md"
CONFIG_JSON = "route_label_blocker_triage_config.json"

TRIAGE_VERSION = "route_label_blocker_triage_20260529"
CLAIM_BOUNDARY = (
    "Blocker triage only; no new route execution, wall-promotion change, "
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

def _as_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)

def _blocker_tags(row: pd.Series) -> str:
    tags: list[str] = []
    if _as_str(row.get("relation_blocker_status")) == "ambiguous_relation_blocks_wall_promotion":
        tags.append("relation_definition")
    if _as_str(row.get("hygiene_blocker_status")) == "field34_hygiene_review_required":
        tags.append("field34_hygiene")
    route_label = _as_str(row.get("route_label_interpretation_v0"))
    if route_label:
        tags.append(f"route_label:{route_label}")
    if _as_str(row.get("route_gate_group")) == "not_run":
        tags.append("route_not_run")
    if "missing" in _as_str(row.get("runner_preflight_status")):
        tags.append("runner_context_missing")
    if not tags:
        tags.append("no_active_blocker_tag")
    return "|".join(tags)

def _relation_queue_status(row: pd.Series) -> str:
    taxonomy = _as_str(row.get("relation_taxonomy_v0_1"))
    route_label = _as_str(row.get("route_label_interpretation_v0"))
    if _as_str(row.get("hygiene_blocker_status")) == "field34_hygiene_review_required":
        return "field34_hygiene_before_relation_review"
    if route_label == "relation_blocked_route_evidence":
        return "route_evidence_relation_blocked"
    if "pending_membership_check" in taxonomy:
        return "pending_membership_relation_check"
    if "boundary_review_cached" in taxonomy:
        return "cached_boundary_relation_review"
    if "middle_ambiguous" in taxonomy:
        return "middle_ambiguous_relation_hold"
    return ""

def _primary_blocker_class(row: pd.Series) -> str:
    status = _as_str(row.get("current_review_status"))
    route_label = _as_str(row.get("route_label_interpretation_v0"))

    if status == "field34_hygiene_blocked":
        return "field34_hygiene_blocker"
    if route_label == "relation_blocked_route_evidence":
        return "basin_relation_definition_blocker"
    if status == "ambiguous_relation_pending":
        return "basin_relation_definition_blocker"
    if route_label == "partial_wall_protocol_evidence":
        return "wall_protocol_evidence_hold"
    if route_label == "boundary_sensitive_route_uncertainty":
        return "route_uncertainty_hold"
    if route_label in {
        "hard_support_loss_no_wall_contrast",
        "mixed_support_loss_no_wall_hold",
    }:
        return "no_wall_contrast_hold"
    if route_label == "same_control_no_wall" or status == "same_control_no_wall":
        return "control_hold"
    return "not_currently_actionable"

def _priority(row: pd.Series) -> int:
    primary = _as_str(row.get("primary_blocker_class"))
    route_label = _as_str(row.get("route_label_interpretation_v0"))
    taxonomy = _as_str(row.get("relation_taxonomy_v0_1"))
    panel_role = _as_str(row.get("panel_role"))

    if primary == "basin_relation_definition_blocker" and route_label == "relation_blocked_route_evidence":
        return 1
    if primary == "field34_hygiene_blocker" and _as_str(row.get("calibrated_relation")) == "distinct_support_local":
        return 2
    if primary == "basin_relation_definition_blocker" and "pending_membership_check" in taxonomy:
        return 2
    if primary == "basin_relation_definition_blocker":
        return 3
    if primary == "field34_hygiene_blocker":
        return 3 if "ambiguous" in panel_role else 4
    if primary in {"wall_protocol_evidence_hold", "route_uncertainty_hold", "no_wall_contrast_hold"}:
        return 5
    if primary == "control_hold":
        return 6
    return 7

def _triage_action(row: pd.Series) -> str:
    primary = _as_str(row.get("primary_blocker_class"))
    route_label = _as_str(row.get("route_label_interpretation_v0"))
    taxonomy = _as_str(row.get("relation_taxonomy_v0_1"))

    if primary == "basin_relation_definition_blocker":
        if route_label == "relation_blocked_route_evidence":
            return "review_boundary_rule_before_any_route_promotion"
        if "pending_membership_check" in taxonomy:
            return "obtain_or_link_membership_evidence_for_relation_rule"
        return "hold_until_relation_rule_or_stronger_signature"
    if primary == "field34_hygiene_blocker":
        return "audit_field34_zero_support_duplicate_and_tiny_support_endpoints"
    if primary == "wall_protocol_evidence_hold":
        return "retain_as_protocol_reference_define_extra_wall_evidence_before_promotion"
    if primary == "route_uncertainty_hold":
        return "retain_as_boundary_sensitive_route_uncertainty_reference"
    if primary == "no_wall_contrast_hold":
        if route_label == "hard_support_loss_no_wall_contrast":
            return "retain_as_hard_support_loss_no_wall_contrast_with_hygiene_note"
        return "retain_as_mixed_no_wall_contrast_not_strong_hard_loss_example"
    if primary == "control_hold":
        return "retain_as_no_wall_control"
    return "manual_review_only_no_route_execution"

def _triage_rationale(row: pd.Series) -> str:
    primary = _as_str(row.get("primary_blocker_class"))
    if primary == "basin_relation_definition_blocker":
        return "Basin relation remains ambiguous or boundary-reviewed, so route evidence cannot become wall evidence."
    if primary == "field34_hygiene_blocker":
        return "Field34 zero-support, duplicate, or tiny-support endpoint risk must be audited before route-gate use."
    if primary == "wall_protocol_evidence_hold":
        return "The row is useful protocol evidence but not a supported wall claim under the frozen rule."
    if primary == "route_uncertainty_hold":
        return "Held-out validation supports a boundary-sensitive route uncertainty class, not wall evidence."
    if primary == "no_wall_contrast_hold":
        return "The row is a no-wall contrast or hold, so it should not trigger wall promotion."
    if primary == "control_hold":
        return "Control relation blocks wall promotion by design."
    return "No current route/wall action is justified without a sharper mechanism question."

def _allowed_next_work(row: pd.Series) -> str:
    primary = _as_str(row.get("primary_blocker_class"))
    if primary == "basin_relation_definition_blocker":
        return "membership/signature relation evidence; boundary-rule review"
    if primary == "field34_hygiene_blocker":
        return "field34 metric hygiene audit; endpoint support-source audit"
    if primary == "wall_protocol_evidence_hold":
        return "write protocol-evidence constraints; design predeclared wall-evidence requirement"
    if primary == "route_uncertainty_hold":
        return "route taxonomy wording; uncertainty-class examples"
    if primary == "no_wall_contrast_hold":
        return "negative-control wording; no-wall contrast examples"
    if primary == "control_hold":
        return "control retention only"
    return "manual review only"

def _forbidden_next_work(row: pd.Series) -> str:
    return (
        "no wider route batch; no wall promotion; no basin-quality/cost join; "
        "no directed-search or operator-success claim"
    )

def _wall_question_status(row: pd.Series) -> str:
    primary = _as_str(row.get("primary_blocker_class"))
    if primary == "wall_protocol_evidence_hold":
        return "protocol_reference_no_immediate_execution"
    if primary == "route_uncertainty_hold":
        return "uncertainty_reference_no_immediate_execution"
    if primary == "no_wall_contrast_hold":
        return "no_wall_contrast_reference_no_immediate_execution"
    return ""

def _triage_rows(current_review_dir: Path, route_label_dir: Path) -> pd.DataFrame:
    current = _read_csv(current_review_dir / CURRENT_PAIR_STATE_CSV)
    route = _read_csv(route_label_dir / ROUTE_LABEL_ROWS_CSV)
    route_cols = [
        "panel_pair_id",
        "route_label_interpretation_version",
        "route_label_interpretation_v0",
        "route_label_interpretation_group",
        "wall_promotion_status_v0",
        "wall_claim_change_v0",
        "quality_cost_join_status",
        "route_label_next_method_action_v0",
        "allowed_claim",
        "forbidden_claim",
        "route_label_claim_boundary_v0",
    ]
    rows = current.merge(route[route_cols], on="panel_pair_id", how="left")
    for column in route_cols:
        if column != "panel_pair_id":
            rows[column] = rows[column].fillna("")
    rows["route_label_interpretation_version"] = rows[
        "route_label_interpretation_version"
    ].replace("", TRIAGE_VERSION + "_not_in_route_label_surface")
    rows["blocker_tags"] = rows.apply(_blocker_tags, axis=1)
    rows["relation_queue_status"] = rows.apply(_relation_queue_status, axis=1)
    rows["primary_blocker_class"] = rows.apply(_primary_blocker_class, axis=1)
    rows["blocker_priority"] = rows.apply(_priority, axis=1)
    rows["triage_action"] = rows.apply(_triage_action, axis=1)
    rows["triage_rationale"] = rows.apply(_triage_rationale, axis=1)
    rows["allowed_next_work"] = rows.apply(_allowed_next_work, axis=1)
    rows["forbidden_next_work"] = rows.apply(_forbidden_next_work, axis=1)
    rows["wall_evidence_question_status"] = rows.apply(_wall_question_status, axis=1)
    rows["immediate_route_execution_status"] = "blocked_or_not_recommended"
    rows["triage_version"] = TRIAGE_VERSION
    rows["claim_boundary"] = CLAIM_BOUNDARY
    rows["source_current_review_artifact"] = _rel(current_review_dir / CURRENT_PAIR_STATE_CSV)
    rows["source_route_label_artifact"] = _rel(route_label_dir / ROUTE_LABEL_ROWS_CSV)

    preferred_cols = [
        "panel_pair_id",
        "field",
        "case_id",
        "panel_role",
        "calibrated_relation",
        "relation_taxonomy_v0_1",
        "field_hygiene_status",
        "runner_context_status",
        "runner_preflight_status",
        "current_review_status",
        "route_gate_group",
        "route_label_interpretation_v0",
        "route_label_interpretation_group",
        "wall_promotion_status_v0",
        "wall_claim_gate_status",
        "relation_blocker_status",
        "hygiene_blocker_status",
        "blocker_tags",
        "primary_blocker_class",
        "relation_queue_status",
        "wall_evidence_question_status",
        "blocker_priority",
        "triage_action",
        "triage_rationale",
        "allowed_next_work",
        "forbidden_next_work",
        "immediate_route_execution_status",
        "triage_version",
        "claim_boundary",
        "review_comment",
        "allowed_claim",
        "forbidden_claim",
        "source_current_review_artifact",
        "source_route_label_artifact",
    ]
    return rows[[col for col in preferred_cols if col in rows.columns]].sort_values(
        ["blocker_priority", "primary_blocker_class", "field", "panel_pair_id"],
        na_position="last",
    ).reset_index(drop=True)

def _counts(rows: pd.DataFrame) -> pd.DataFrame:
    count_rows: list[dict[str, Any]] = []
    for column in (
        "primary_blocker_class",
        "relation_queue_status",
        "wall_evidence_question_status",
        "hygiene_blocker_status",
        "route_label_interpretation_v0",
        "immediate_route_execution_status",
    ):
        for value, count in _count(rows, column).items():
            count_rows.append({"count_type": column, "value": value, "count": count})
    return pd.DataFrame(count_rows)

def _relation_queue(rows: pd.DataFrame) -> pd.DataFrame:
    queue = rows[rows["relation_blocker_status"].eq("ambiguous_relation_blocks_wall_promotion")]
    cols = [
        "panel_pair_id",
        "field",
        "case_id",
        "panel_role",
        "calibrated_relation",
        "relation_taxonomy_v0_1",
        "route_label_interpretation_v0",
        "relation_queue_status",
        "blocker_priority",
        "triage_action",
        "triage_rationale",
        "hygiene_blocker_status",
        "runner_preflight_status",
        "claim_boundary",
    ]
    return queue[[col for col in cols if col in queue.columns]].sort_values(
        ["blocker_priority", "field", "panel_pair_id"]
    )

def _field34_queue(rows: pd.DataFrame) -> pd.DataFrame:
    queue = rows[rows["hygiene_blocker_status"].eq("field34_hygiene_review_required")]
    cols = [
        "panel_pair_id",
        "field",
        "case_id",
        "panel_role",
        "calibrated_relation",
        "relation_taxonomy_v0_1",
        "route_label_interpretation_v0",
        "primary_blocker_class",
        "blocker_priority",
        "triage_action",
        "triage_rationale",
        "wall_evidence_question_status",
        "claim_boundary",
    ]
    return queue[[col for col in cols if col in queue.columns]].sort_values(
        ["blocker_priority", "calibrated_relation", "panel_pair_id"]
    )

def _wall_question_queue(rows: pd.DataFrame) -> pd.DataFrame:
    queue = rows[rows["wall_evidence_question_status"].ne("")]
    cols = [
        "panel_pair_id",
        "field",
        "case_id",
        "panel_role",
        "route_label_interpretation_v0",
        "route_label_interpretation_group",
        "wall_evidence_question_status",
        "blocker_priority",
        "triage_action",
        "allowed_next_work",
        "forbidden_next_work",
        "claim_boundary",
    ]
    return queue[[col for col in cols if col in queue.columns]].sort_values(
        ["blocker_priority", "field", "panel_pair_id"]
    )

def _summary(
    rows: pd.DataFrame,
    relation_queue: pd.DataFrame,
    field34_queue: pd.DataFrame,
    wall_queue: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    immediate_route_count = int(
        rows["immediate_route_execution_status"].astype(str).eq("ready").sum()
    )
    return {
        "status": "route_label_blocker_triage_prepared",
        "date": "2026-05-29",
        "script": _rel(Path(__file__).resolve()),
        "output_dir": _rel(output_dir),
        "triage_version": TRIAGE_VERSION,
        "pair_count": int(len(rows)),
        "primary_blocker_class_counts": _count(rows, "primary_blocker_class"),
        "relation_queue_status_counts": _count(relation_queue, "relation_queue_status"),
        "wall_evidence_question_status_counts": _count(
            wall_queue, "wall_evidence_question_status"
        ),
        "hygiene_blocker_status_counts": _count(rows, "hygiene_blocker_status"),
        "route_label_interpretation_counts": _count(rows, "route_label_interpretation_v0"),
        "relation_definition_queue_count": int(len(relation_queue)),
        "field34_hygiene_queue_count": int(len(field34_queue)),
        "wall_evidence_question_hold_queue_count": int(len(wall_queue)),
        "immediate_route_execution_count": immediate_route_count,
        "decision": (
            "Do not broaden route execution. The next work is relation-definition "
            "review first, field34 hygiene audit second, and only written "
            "wall-evidence requirement design for frozen protocol/uncertainty/no-wall rows."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "paths": {
            "triage_rows": _rel(output_dir / TRIAGE_ROWS_CSV),
            "relation_queue": _rel(output_dir / RELATION_QUEUE_CSV),
            "field34_queue": _rel(output_dir / FIELD34_QUEUE_CSV),
            "wall_question_queue": _rel(output_dir / WALL_QUESTION_QUEUE_CSV),
            "counts": _rel(output_dir / COUNTS_CSV),
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

def _report(
    rows: pd.DataFrame,
    relation_queue: pd.DataFrame,
    field34_queue: pd.DataFrame,
    wall_queue: pd.DataFrame,
    summary: dict[str, Any],
) -> str:
    lines = [
        "# Route-Label Blocker Triage",
        "",
        "Date: 2026-05-29",
        "",
        "## Scope",
        "",
        "This artifact applies the frozen route-label interpretation v0 to the",
        "current 23-pair surface. It separates remaining blockers without running",
        "new routes, changing wall-promotion rules, or inspecting basin quality/cost.",
        "",
        "## Decision",
        "",
        str(summary["decision"]),
        "",
        "Claim boundary: " + CLAIM_BOUNDARY,
        "",
        "## Primary Blockers",
        "",
    ]
    for key, count in summary["primary_blocker_class_counts"].items():
        lines.append(f"- `{key}`: {count}")
    lines.extend(
        [
            f"- relation-definition queue rows: {summary['relation_definition_queue_count']}",
            f"- field34 hygiene queue rows: {summary['field34_hygiene_queue_count']}",
            f"- wall-evidence question hold rows: {summary['wall_evidence_question_hold_queue_count']}",
            f"- immediate route execution rows: {summary['immediate_route_execution_count']}",
            "",
            "## Relation Definition Queue",
            "",
        ]
    )
    relation_cols = [
        "panel_pair_id",
        "relation_taxonomy_v0_1",
        "route_label_interpretation_v0",
        "relation_queue_status",
        "blocker_priority",
    ]
    lines.append(_markdown_table(relation_queue[relation_cols]))
    lines.extend(["", "## Field34 Hygiene Queue", ""])
    field34_cols = [
        "panel_pair_id",
        "calibrated_relation",
        "route_label_interpretation_v0",
        "primary_blocker_class",
        "blocker_priority",
    ]
    lines.append(_markdown_table(field34_queue[field34_cols]))
    lines.extend(["", "## Wall-Evidence Question Hold Queue", ""])
    wall_cols = [
        "panel_pair_id",
        "route_label_interpretation_v0",
        "wall_evidence_question_status",
        "triage_action",
    ]
    lines.append(_markdown_table(wall_queue[wall_cols]))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The current route surface does not justify a broader route batch. The",
            "highest-priority work is the three route-stable but relation-blocked",
            "rows, followed by pending-membership relation checks and field34 metric",
            "hygiene. Frozen protocol and uncertainty rows should be retained as",
            "examples or constraints, not promoted to wall evidence.",
            "",
        ]
    )
    return "\n".join(lines)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-review-dir", type=Path, default=DEFAULT_CURRENT_REVIEW_DIR)
    parser.add_argument("--route-label-dir", type=Path, default=DEFAULT_ROUTE_LABEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _triage_rows(args.current_review_dir, args.route_label_dir)
    relation_queue = _relation_queue(rows)
    field34_queue = _field34_queue(rows)
    wall_queue = _wall_question_queue(rows)
    counts = _counts(rows)
    summary = _summary(rows, relation_queue, field34_queue, wall_queue, output_dir)
    config = {
        "current_review_dir": _rel(args.current_review_dir),
        "route_label_dir": _rel(args.route_label_dir),
        "output_dir": _rel(output_dir),
        "triage_version": TRIAGE_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
    }

    _write_csv(rows, output_dir / TRIAGE_ROWS_CSV)
    _write_csv(relation_queue, output_dir / RELATION_QUEUE_CSV)
    _write_csv(field34_queue, output_dir / FIELD34_QUEUE_CSV)
    _write_csv(wall_queue, output_dir / WALL_QUESTION_QUEUE_CSV)
    _write_csv(counts, output_dir / COUNTS_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / REPORT_MD).write_text(
        _report(rows, relation_queue, field34_queue, wall_queue, summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
