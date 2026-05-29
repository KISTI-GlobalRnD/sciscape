#!/usr/bin/env python3
"""Freeze Route-label interpretation v0 for Leiden basin cartography.

This artifact sits after Methodology v0 and the held-out margin-validation
review. It assigns conservative interpretation labels to the current 11-pair
route-gate surface. It does not run new routes, change wall-promotion rules, or
join quality/cost fields.
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
DEFAULT_METHODOLOGY_DIR = BASE_RESULT_DIR / "leiden_basin_methodology_v0_margin_validation_20260528"
DEFAULT_MARGIN_VALIDATION_DIR = (
    BASE_RESULT_DIR / "leiden_basin_margin_validation_panel_review_20260529"
)
DEFAULT_CURRENT_REVIEW_DIR = BASE_RESULT_DIR / "leiden_basin_current_results_review_20260529"
DEFAULT_OUTPUT_DIR = BASE_RESULT_DIR / "leiden_basin_route_label_interpretation_v0_20260529"

METHODOLOGY_ROWS_CSV = "methodology_v0_route_gate_decision_rows.csv"
VALIDATION_RESULTS_CSV = "margin_validation_pair_results.csv"
CURRENT_PAIR_STATE_CSV = "current_pair_state_ledger.csv"

INTERPRETATION_ROWS_CSV = "route_label_interpretation_rows.csv"
RULES_CSV = "route_label_interpretation_rules.csv"
COUNTS_CSV = "route_label_interpretation_counts.csv"
SUMMARY_JSON = "route_label_interpretation_summary.json"
REPORT_MD = "route_label_interpretation_report.md"
CONFIG_JSON = "route_label_interpretation_config.json"

INTERPRETATION_VERSION = "route_label_interpretation_v0_20260529"
CLAIM_BOUNDARY = (
    "Route-label interpretation only; no basin-quality claim, cost claim, "
    "directed-search claim, or wall-promotion change."
)


RULES: tuple[dict[str, str], ...] = (
    {
        "rule_id": "R1",
        "methodology_v0_state": "partial_wall_gate_conservative",
        "margin_validation_status": "any_or_empty",
        "route_label_interpretation_v0": "partial_wall_protocol_evidence",
        "route_label_interpretation_group": "conservative_partial_wall_gate",
        "wall_promotion_status_v0": "no_wall_promotion",
        "next_method_action": "retain_as_protocol_evidence",
        "allowed_claim": (
            "Schedule-invariant distinct route protocol evidence under the "
            "current W1-W6 gate."
        ),
        "forbidden_claim": (
            "Do not use as a supported wall claim, basin-quality result, "
            "cost result, or directed-search success claim."
        ),
        "rule_note": "The gate is retained but not strengthened by margin context.",
    },
    {
        "rule_id": "R2",
        "methodology_v0_state": "relation_blocked_definition_evidence",
        "margin_validation_status": "any_or_empty",
        "route_label_interpretation_v0": "relation_blocked_route_evidence",
        "route_label_interpretation_group": "definition_blocked",
        "wall_promotion_status_v0": "no_wall_promotion",
        "next_method_action": "fix_relation_rule_or_add_stronger_membership_evidence",
        "allowed_claim": (
            "Stable route evidence exists, but the basin relation remains "
            "ambiguous under the current definition."
        ),
        "forbidden_claim": (
            "Do not promote route-stable ambiguous rows to wall evidence."
        ),
        "rule_note": "The bottleneck is basin relation, not route execution.",
    },
    {
        "rule_id": "R3",
        "methodology_v0_state": "same_control_no_wall",
        "margin_validation_status": "any_or_empty",
        "route_label_interpretation_v0": "same_control_no_wall",
        "route_label_interpretation_group": "control_no_wall",
        "wall_promotion_status_v0": "no_wall_promotion",
        "next_method_action": "retain_as_control",
        "allowed_claim": "Same-control no-wall reference row.",
        "forbidden_claim": "Do not use a same-control row as wall evidence.",
        "rule_note": "Control relation blocks wall promotion.",
    },
    {
        "rule_id": "R4",
        "methodology_v0_state": "boundary_sensitive_margin_validation_candidate",
        "margin_validation_status": "validated_boundary_sensitive_hold",
        "route_label_interpretation_v0": "boundary_sensitive_route_uncertainty",
        "route_label_interpretation_group": "validated_boundary_uncertainty",
        "wall_promotion_status_v0": "no_wall_promotion",
        "next_method_action": "freeze_as_route_uncertainty_class",
        "allowed_claim": (
            "Held-out margin validation did not introduce hard support loss; "
            "the row is a boundary-sensitive route uncertainty class."
        ),
        "forbidden_claim": (
            "Do not convert near-threshold W4 support assignment into wall "
            "evidence or basin quality evidence."
        ),
        "rule_note": "This is the main new label unlocked by held-out validation.",
    },
    {
        "rule_id": "R5",
        "methodology_v0_state": "boundary_sensitive_margin_validation_candidate",
        "margin_validation_status": "any_or_empty",
        "route_label_interpretation_v0": "boundary_sensitive_pending_validation",
        "route_label_interpretation_group": "pending_boundary_uncertainty",
        "wall_promotion_status_v0": "no_wall_promotion",
        "next_method_action": "run_or_review_predeclared_margin_validation",
        "allowed_claim": "Boundary-sensitive candidate awaiting validation.",
        "forbidden_claim": "Do not change route interpretation before validation.",
        "rule_note": "Fallback rule for future rows without held-out validation.",
    },
    {
        "rule_id": "R6",
        "methodology_v0_state": "support_loss_no_wall_contrast",
        "margin_validation_status": "validated_support_loss_contrast",
        "route_label_interpretation_v0": "hard_support_loss_no_wall_contrast",
        "route_label_interpretation_group": "strong_no_wall_contrast",
        "wall_promotion_status_v0": "no_wall_promotion",
        "next_method_action": "retain_as_repeated_hard_loss_contrast",
        "allowed_claim": (
            "Held-out validation repeats hard post-polish target-support loss; "
            "the row is a strong no-wall contrast."
        ),
        "forbidden_claim": (
            "Do not describe hard support loss as a failed better-basin result."
        ),
        "rule_note": "Strong contrast, not wall evidence.",
    },
    {
        "rule_id": "R7",
        "methodology_v0_state": "support_loss_no_wall_contrast",
        "margin_validation_status": "support_loss_contrast_mixed_hold",
        "route_label_interpretation_v0": "mixed_support_loss_no_wall_hold",
        "route_label_interpretation_group": "mixed_no_wall_contrast",
        "wall_promotion_status_v0": "no_wall_promotion",
        "next_method_action": "retain_no_wall_but_do_not_use_as_strong_hard_loss_example",
        "allowed_claim": (
            "Prior schedules retain hard-loss evidence, but the held-out "
            "schedule is near-boundary; keep the row as a mixed no-wall hold."
        ),
        "forbidden_claim": (
            "Do not use this row as a repeated hard-loss exemplar or wall claim."
        ),
        "rule_note": "Mixed contrast should weaken, not strengthen, the claim.",
    },
    {
        "rule_id": "R8",
        "methodology_v0_state": "support_loss_no_wall_contrast",
        "margin_validation_status": "any_or_empty",
        "route_label_interpretation_v0": "support_loss_no_wall_contrast_pending_validation",
        "route_label_interpretation_group": "pending_no_wall_contrast",
        "wall_promotion_status_v0": "no_wall_promotion",
        "next_method_action": "run_or_review_predeclared_margin_validation",
        "allowed_claim": "Support-loss no-wall contrast awaiting validation.",
        "forbidden_claim": "Do not strengthen contrast before validation.",
        "rule_note": "Fallback rule for future contrast rows without validation.",
    },
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


def _rule_frame() -> pd.DataFrame:
    rules = pd.DataFrame(RULES)
    rules["claim_boundary"] = CLAIM_BOUNDARY
    return rules


def _match_rule(row: pd.Series, rules: pd.DataFrame) -> pd.Series:
    state = str(row.get("methodology_v0_state", ""))
    validation = str(row.get("margin_validation_status", ""))
    exact = rules[
        (rules["methodology_v0_state"].astype(str).eq(state))
        & (rules["margin_validation_status"].astype(str).eq(validation))
    ]
    if exact.empty:
        exact = rules[
            (rules["methodology_v0_state"].astype(str).eq(state))
            & (rules["margin_validation_status"].astype(str).eq("any_or_empty"))
        ]
    if exact.empty:
        return pd.Series(
            {
                "rule_id": "UNMATCHED",
                "route_label_interpretation_v0": "unmatched_route_label_interpretation",
                "route_label_interpretation_group": "unmatched",
                "wall_promotion_status_v0": "no_wall_promotion",
                "next_method_action": "manual_review_required",
                "allowed_claim": "No automatic interpretation rule matched.",
                "forbidden_claim": "Do not use unmatched rows for claims.",
                "rule_note": "Add an explicit rule before using this row.",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return exact.iloc[0]


def _interpretation_rows(
    methodology_dir: Path,
    margin_validation_dir: Path,
    current_review_dir: Path,
) -> pd.DataFrame:
    methodology = _read_csv(methodology_dir / METHODOLOGY_ROWS_CSV)
    validation = _read_csv(margin_validation_dir / VALIDATION_RESULTS_CSV)
    current = _read_csv(current_review_dir / CURRENT_PAIR_STATE_CSV)
    rules = _rule_frame()

    validation_cols = [
        "panel_pair_id",
        "validation_status",
        "heldout_margin_bands",
        "combined_margin_bands",
        "validation_note",
    ]
    current_cols = [
        "panel_pair_id",
        "current_review_status",
        "route_gate_group",
        "relation_blocker_status",
        "hygiene_blocker_status",
        "review_comment",
    ]
    rows = methodology.merge(
        validation[validation_cols].rename(
            columns={
                "validation_status": "margin_validation_status",
                "validation_note": "margin_validation_note",
            }
        ),
        on="panel_pair_id",
        how="left",
    )
    rows = rows.merge(current[current_cols], on="panel_pair_id", how="left")
    rows = rows.rename(
        columns={
            "next_method_action": "methodology_next_method_action",
            "claim_boundary": "methodology_claim_boundary",
            "claim_boundary_v0": "methodology_claim_boundary_v0",
        }
    )
    rows["margin_validation_status"] = rows["margin_validation_status"].fillna("")
    rows["heldout_margin_bands"] = rows["heldout_margin_bands"].fillna("")
    rows["combined_margin_bands"] = rows["combined_margin_bands"].fillna("")
    rows["margin_validation_note"] = rows["margin_validation_note"].fillna("")

    matched = rows.apply(lambda row: _match_rule(row, rules), axis=1).reset_index(drop=True)
    matched = matched.rename(
        columns={
            "next_method_action": "route_label_next_method_action_v0",
            "claim_boundary": "route_label_claim_boundary_v0",
        }
    )
    matched_cols = [
        "rule_id",
        "route_label_interpretation_v0",
        "route_label_interpretation_group",
        "wall_promotion_status_v0",
        "route_label_next_method_action_v0",
        "allowed_claim",
        "forbidden_claim",
        "rule_note",
        "route_label_claim_boundary_v0",
    ]
    rows = pd.concat(
        [rows.reset_index(drop=True), matched[matched_cols].reset_index(drop=True)],
        axis=1,
    )
    rows["route_label_interpretation_version"] = INTERPRETATION_VERSION
    rows["wall_claim_change_v0"] = "none"
    rows["quality_cost_join_status"] = "deferred"
    rows["source_methodology_artifact"] = _rel(methodology_dir / METHODOLOGY_ROWS_CSV)
    rows["source_margin_validation_artifact"] = _rel(margin_validation_dir / VALIDATION_RESULTS_CSV)
    rows["source_current_review_artifact"] = _rel(current_review_dir / CURRENT_PAIR_STATE_CSV)

    preferred_cols = [
        "panel_pair_id",
        "field",
        "case_id",
        "panel_role",
        "calibrated_relation",
        "support_distance_max",
        "route_order_sensitivity_status",
        "wall_claim_gate_status",
        "route_labels",
        "polish_margin_bands",
        "margin_gate_status",
        "methodology_v0_state",
        "validation_role",
        "margin_validation_status",
        "heldout_margin_bands",
        "combined_margin_bands",
        "rule_id",
        "route_label_interpretation_version",
        "route_label_interpretation_v0",
        "route_label_interpretation_group",
        "wall_promotion_status_v0",
        "wall_claim_change_v0",
        "quality_cost_join_status",
        "current_review_status",
        "route_gate_group",
        "relation_blocker_status",
        "hygiene_blocker_status",
        "methodology_next_method_action",
        "route_label_next_method_action_v0",
        "allowed_claim",
        "forbidden_claim",
        "rule_note",
        "methodology_claim_boundary",
        "methodology_claim_boundary_v0",
        "route_label_claim_boundary_v0",
        "methodology_v0_rationale",
        "margin_validation_note",
        "review_comment",
        "source_methodology_artifact",
        "source_margin_validation_artifact",
        "source_current_review_artifact",
    ]
    rows = rows[[col for col in preferred_cols if col in rows.columns]]
    return rows.sort_values(
        ["route_label_interpretation_group", "field", "panel_pair_id"],
        na_position="last",
    ).reset_index(drop=True)


def _counts(rows: pd.DataFrame) -> pd.DataFrame:
    count_rows: list[dict[str, Any]] = []
    for column in (
        "route_label_interpretation_v0",
        "route_label_interpretation_group",
        "wall_promotion_status_v0",
        "methodology_v0_state",
        "margin_validation_status",
    ):
        for value, count in _count(rows, column).items():
            count_rows.append({"count_type": column, "value": value, "count": count})
    return pd.DataFrame(count_rows)


def _summary(rows: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    promoted = int(
        rows["wall_promotion_status_v0"].astype(str).ne("no_wall_promotion").sum()
    )
    return {
        "status": "route_label_interpretation_v0_prepared",
        "date": "2026-05-29",
        "script": _rel(Path(__file__).resolve()),
        "output_dir": _rel(output_dir),
        "interpretation_version": INTERPRETATION_VERSION,
        "pair_count": int(len(rows)),
        "route_label_interpretation_counts": _count(rows, "route_label_interpretation_v0"),
        "route_label_interpretation_group_counts": _count(rows, "route_label_interpretation_group"),
        "methodology_v0_state_counts": _count(rows, "methodology_v0_state"),
        "margin_validation_status_counts": _count(rows, "margin_validation_status"),
        "wall_promotion_status_counts": _count(rows, "wall_promotion_status_v0"),
        "promoted_wall_claim_count": promoted,
        "decision": (
            "Use Route-label interpretation v0 as the next Track C route "
            "surface. Boundary-sensitive rows are uncertainty, partial-wall "
            "rows are protocol evidence, relation-blocked rows stay definition "
            "evidence, support-loss rows stay no-wall contrasts."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "paths": {
            "rows": _rel(output_dir / INTERPRETATION_ROWS_CSV),
            "rules": _rel(output_dir / RULES_CSV),
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
        max(
            len(str(column)),
            *(len(row[index]) for row in rendered_rows),
        )
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


def _report(rows: pd.DataFrame, rules: pd.DataFrame, summary: dict[str, Any]) -> str:
    lines = [
        "# Route-Label Interpretation v0",
        "",
        "Date: 2026-05-29",
        "",
        "## Scope",
        "",
        "This artifact freezes the interpretation of the current 11-pair route-gate",
        "surface after the held-out margin-validation review. It does not run new",
        "routes, change wall-promotion rules, or inspect basin quality/cost.",
        "",
        "## Decision",
        "",
        str(summary["decision"]),
        "",
        "Claim boundary: " + CLAIM_BOUNDARY,
        "",
        "## Counts",
        "",
    ]
    for key, counts in summary["route_label_interpretation_counts"].items():
        lines.append(f"- `{key}`: {counts}")
    lines.extend(
        [
            f"- promoted wall claims: {summary['promoted_wall_claim_count']}",
            "",
            "## Interpretation Rules",
            "",
        ]
    )
    for row in rules.itertuples(index=False):
        lines.extend(
            [
                f"### {row.rule_id}: `{row.route_label_interpretation_v0}`",
                "",
                f"- methodology state: `{row.methodology_v0_state}`",
                f"- margin validation status: `{row.margin_validation_status}`",
                f"- group: `{row.route_label_interpretation_group}`",
                f"- wall promotion: `{row.wall_promotion_status_v0}`",
                f"- next action: `{row.next_method_action}`",
                f"- allowed claim: {row.allowed_claim}",
                f"- forbidden claim: {row.forbidden_claim}",
                "",
            ]
        )
    lines.extend(["## Pair-Level Interpretation", ""])
    table_cols = [
        "panel_pair_id",
        "methodology_v0_state",
        "margin_validation_status",
        "route_label_interpretation_v0",
        "wall_promotion_status_v0",
    ]
    lines.append(_markdown_table(rows[table_cols]))
    lines.extend(
        [
            "",
            "## Next Method Action",
            "",
            "Do not broaden route execution from this artifact. The next method work",
            "is to use these frozen labels when deciding whether the remaining blocker",
            "is basin relation, field34 hygiene, or a genuinely new wall-evidence",
            "question.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methodology-dir", type=Path, default=DEFAULT_METHODOLOGY_DIR)
    parser.add_argument(
        "--margin-validation-dir",
        type=Path,
        default=DEFAULT_MARGIN_VALIDATION_DIR,
    )
    parser.add_argument("--current-review-dir", type=Path, default=DEFAULT_CURRENT_REVIEW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rules = _rule_frame()
    rows = _interpretation_rows(
        methodology_dir=args.methodology_dir,
        margin_validation_dir=args.margin_validation_dir,
        current_review_dir=args.current_review_dir,
    )
    counts = _counts(rows)
    summary = _summary(rows, output_dir)
    config = {
        "methodology_dir": _rel(args.methodology_dir),
        "margin_validation_dir": _rel(args.margin_validation_dir),
        "current_review_dir": _rel(args.current_review_dir),
        "output_dir": _rel(output_dir),
        "interpretation_version": INTERPRETATION_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
    }

    _write_csv(rows, output_dir / INTERPRETATION_ROWS_CSV)
    _write_csv(rules, output_dir / RULES_CSV)
    _write_csv(counts, output_dir / COUNTS_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / REPORT_MD).write_text(_report(rows, rules, summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
