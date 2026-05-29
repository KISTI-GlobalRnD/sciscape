#!/usr/bin/env python3
"""Prepare Methodology v0 route-gate decisions and a margin validation panel.

This script consumes the W4 polish margin gate review and freezes the current
conservative interpretation as a reusable methodology artifact. It does not
rerun routes, evaluate basin quality, or change wall-promotion gates.
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
DEFAULT_MARGIN_REVIEW_DIR = BASE_RESULT_DIR / "leiden_basin_polish_margin_gate_review_20260528"
DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR / "leiden_basin_methodology_v0_margin_validation_20260528"
)

PAIR_GATE_ROWS_CSV = "polish_margin_pair_gate_rows.csv"
DECISION_ROWS_CSV = "methodology_v0_route_gate_decision_rows.csv"
STATE_COUNTS_CSV = "methodology_v0_state_counts.csv"
VALIDATION_PANEL_CSV = "margin_validation_panel.csv"
SUMMARY_JSON = "methodology_v0_margin_validation_summary.json"
REPORT_MD = "methodology_v0_margin_validation_report.md"
CONFIG_JSON = "methodology_v0_margin_validation_config.json"

FORBIDDEN_USE = (
    "Do not use this row for basin quality/cost ranking, directed-search "
    "success claims, or wall-promotion relaxation."
)

COMMON_VALIDATION_QUESTION = (
    "Do near-threshold post-polish support losses behave differently from "
    "support-hard-loss no-wall holds under predeclared repeat polish/schedule "
    "validation?"
)

STATE_RULES: dict[str, dict[str, Any]] = {
    "keep_partial_wall_gate_with_margin_context": {
        "methodology_v0_state": "partial_wall_gate_conservative",
        "validation_role": "reference_partial_wall_gate",
        "include_in_margin_validation_panel": False,
        "next_method_action": (
            "Keep as current protocol evidence. Margin values are context only."
        ),
        "validation_question": "No margin validation needed before retaining this gate.",
        "pass_condition": "Existing schedule-invariant distinct partial-wall gate remains unchanged.",
        "fail_condition": "Only revisit if the upstream route-gate artifact changes.",
        "claim_boundary": (
            "Supported only as current partial-wall protocol evidence; not a "
            "quality or directed-search claim."
        ),
        "methodology_v0_rationale": (
            "The pair already passes the current schedule-invariant distinct "
            "partial-wall gate. Margin context must not strengthen the claim."
        ),
    },
    "relation_blocked_keep_as_definition_evidence": {
        "methodology_v0_state": "relation_blocked_definition_evidence",
        "validation_role": "relation_definition_reference",
        "include_in_margin_validation_panel": False,
        "next_method_action": (
            "Use for basin-relation refinement. Do not promote to wall evidence."
        ),
        "validation_question": "Resolve basin relation before any route or wall interpretation.",
        "pass_condition": "Relation rule stays ambiguous or is separately refined.",
        "fail_condition": "A future relation rule reclassifies the pair as distinct or same.",
        "claim_boundary": (
            "Definition evidence only. Stable route evidence is blocked by basin relation."
        ),
        "methodology_v0_rationale": (
            "Stable route evidence is not enough while the basin relation is ambiguous."
        ),
    },
    "boundary_sensitive_route_evidence_hold": {
        "methodology_v0_state": "boundary_sensitive_margin_validation_candidate",
        "validation_role": "boundary_sensitive_candidate",
        "include_in_margin_validation_panel": True,
        "next_method_action": (
            "Run only a narrow predeclared margin-validation repeat before "
            "changing route-label interpretation."
        ),
        "validation_question": COMMON_VALIDATION_QUESTION,
        "pass_condition": (
            "Repeated schedules/polish checks keep failures inside the "
            "support-margin band without introducing support-hard-loss rows."
        ),
        "fail_condition": (
            "Any repeated evidence shows support-hard-loss behavior or relation "
            "instability; keep as no-wall hold."
        ),
        "claim_boundary": (
            "Boundary-sensitive route evidence only. It cannot become wall "
            "evidence without a validated margin rule and unchanged relation gate."
        ),
        "methodology_v0_rationale": (
            "The route reaches a target-like pre-polish state, but W4 polish "
            "sometimes lands just beyond the support threshold."
        ),
    },
    "support_loss_no_wall_hold": {
        "methodology_v0_state": "support_loss_no_wall_contrast",
        "validation_role": "support_loss_contrast",
        "include_in_margin_validation_panel": True,
        "next_method_action": (
            "Use as the hard-loss contrast in the margin-validation panel."
        ),
        "validation_question": COMMON_VALIDATION_QUESTION,
        "pass_condition": (
            "Repeated checks continue to show support-hard-loss behavior; no "
            "wall-promotion rule changes."
        ),
        "fail_condition": (
            "Repeated checks collapse into only near-boundary loss, forcing the "
            "margin panel to stay inconclusive."
        ),
        "claim_boundary": (
            "No-wall hold and negative contrast for margin validation; not a "
            "failed basin-quality result."
        ),
        "methodology_v0_rationale": (
            "At least one schedule loses post-polish target support beyond the "
            "predeclared diagnostic margin band."
        ),
    },
    "keep_as_same_control": {
        "methodology_v0_state": "same_control_no_wall",
        "validation_role": "same_control_reference",
        "include_in_margin_validation_panel": False,
        "next_method_action": "Keep as a same-control no-wall row.",
        "validation_question": "No margin validation needed for same-control status.",
        "pass_condition": "Same-control relation remains unchanged.",
        "fail_condition": "A future relation rule no longer treats the pair as same-control.",
        "claim_boundary": "Control row only. It cannot support wall existence.",
        "methodology_v0_rationale": (
            "Same-control rows are outside distinct-basin wall promotion."
        ),
    },
}

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

def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    counts = frame[column].value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in counts.items()}

def _decision_fields(row: pd.Series) -> dict[str, Any]:
    status = str(row.get("margin_gate_status", ""))
    if status not in STATE_RULES:
        raise ValueError(f"unsupported margin_gate_status: {status}")
    rule = STATE_RULES[status]
    return {
        "methodology_v0_state": rule["methodology_v0_state"],
        "validation_role": rule["validation_role"],
        "include_in_margin_validation_panel": bool(rule["include_in_margin_validation_panel"]),
        "validation_question": rule["validation_question"],
        "pass_condition": rule["pass_condition"],
        "fail_condition": rule["fail_condition"],
        "forbidden_use": FORBIDDEN_USE,
        "next_method_action": rule["next_method_action"],
        "claim_boundary_v0": rule["claim_boundary"],
        "methodology_v0_rationale": rule["methodology_v0_rationale"],
    }

def _decision_rows(pair_rows: pd.DataFrame) -> pd.DataFrame:
    decisions = pair_rows.copy()
    decision_fields = decisions.apply(_decision_fields, axis=1, result_type="expand")
    decisions = pd.concat([decisions, decision_fields], axis=1)
    ordered_cols = [
        "panel_pair_id",
        "field",
        "case_id",
        "panel_role",
        "calibrated_relation",
        "support_distance_max",
        "source_output",
        "schedule_count",
        "route_order_sensitivity_status",
        "wall_claim_gate_status",
        "post_target_schedule_count",
        "post_non_target_schedule_count",
        "polish_margin_bands",
        "post_target_support_margin_min",
        "post_target_support_margin_max",
        "post_target_support_distance_min",
        "post_target_support_distance_max",
        "margin_gate_status",
        "methodology_v0_state",
        "validation_role",
        "include_in_margin_validation_panel",
        "validation_question",
        "pass_condition",
        "fail_condition",
        "forbidden_use",
        "next_method_action",
        "margin_gate_note",
        "methodology_v0_rationale",
        "claim_boundary",
        "claim_boundary_v0",
    ]
    for column in ordered_cols:
        if column not in decisions:
            decisions[column] = ""
    return decisions[ordered_cols].sort_values(["field", "panel_pair_id"]).reset_index(drop=True)

def _state_counts(decisions: pd.DataFrame) -> pd.DataFrame:
    counts = (
        decisions.groupby(["methodology_v0_state", "validation_role"], dropna=False)
        .size()
        .reset_index(name="pair_count")
        .sort_values(["methodology_v0_state", "validation_role"])
    )
    return counts

def _validation_panel(decisions: pd.DataFrame) -> pd.DataFrame:
    panel = decisions[decisions["include_in_margin_validation_panel"].astype(bool)].copy()
    panel["validation_order"] = panel["validation_role"].map(
        {
            "boundary_sensitive_candidate": 1,
            "support_loss_contrast": 2,
        }
    )
    panel["predeclared_repeat_scope"] = (
        "Repeat only the fixed W1-W6 route schedules/polish checks needed to "
        "compare support-boundary loss against support-hard-loss behavior."
    )
    panel["promotion_rule"] = (
        "No row in this panel may promote a wall claim during validation."
    )
    ordered_cols = [
        "validation_order",
        "panel_pair_id",
        "field",
        "case_id",
        "panel_role",
        "calibrated_relation",
        "methodology_v0_state",
        "validation_role",
        "route_order_sensitivity_status",
        "wall_claim_gate_status",
        "polish_margin_bands",
        "post_target_support_margin_min",
        "post_target_support_margin_max",
        "post_target_support_distance_min",
        "post_target_support_distance_max",
        "validation_question",
        "pass_condition",
        "fail_condition",
        "predeclared_repeat_scope",
        "promotion_rule",
        "forbidden_use",
        "next_method_action",
    ]
    return (
        panel[ordered_cols]
        .sort_values(["validation_order", "field", "panel_pair_id"])
        .reset_index(drop=True)
    )

def _write_report(
    path: Path,
    summary: dict[str, Any],
    decisions: pd.DataFrame,
    validation_panel: pd.DataFrame,
) -> None:
    lines = [
        "# Leiden Basin Methodology v0 Margin Validation",
        "",
        "Status: Methodology v0 route-gate decisions frozen",
        "Date: 2026-05-28",
        "",
        "This artifact converts the W4 polish margin review into a conservative method decision table. It does not rerun routes, inspect basin quality, or relax wall promotion.",
        "",
        "## State Counts",
        "",
        "| methodology_v0_state | pairs |",
        "| --- | ---: |",
    ]
    state_counts = summary["methodology_v0_state_counts"]
    for state, count in sorted(state_counts.items()):
        lines.append(f"| {state} | {count} |")
    lines.extend(
        [
            "",
            "## Margin Validation Panel",
            "",
            "| pair_id | role | state | support_margin_max | bands |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for _, row in validation_panel.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["panel_pair_id"]),
                    str(row["validation_role"]),
                    str(row["methodology_v0_state"]),
                    f"{row['post_target_support_margin_max']:.6f}",
                    str(row["polish_margin_bands"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Keep the 3 existing distinct partial-wall gates unchanged.",
            "- Keep ambiguous-relation rows blocked from wall promotion.",
            "- Use only the 4-row margin panel to validate boundary-sensitive route interpretation.",
            "- Do not treat margin validation as basin quality, cost, or directed-search evidence.",
            "",
            "## All Route-Gate Decisions",
            "",
            "| pair_id | margin_gate | methodology_v0_state | validation_role |",
            "| --- | --- | --- | --- |",
        ]
    )
    for _, row in decisions.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["panel_pair_id"]),
                    str(row["margin_gate_status"]),
                    str(row["methodology_v0_state"]),
                    str(row["validation_role"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run(margin_review_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows = _read_csv(margin_review_dir / PAIR_GATE_ROWS_CSV)
    decisions = _decision_rows(pair_rows)
    counts = _state_counts(decisions)
    validation_panel = _validation_panel(decisions)

    _write_csv(decisions, output_dir / DECISION_ROWS_CSV)
    _write_csv(counts, output_dir / STATE_COUNTS_CSV)
    _write_csv(validation_panel, output_dir / VALIDATION_PANEL_CSV)

    state_count_map = _value_counts(decisions, "methodology_v0_state")
    role_count_map = _value_counts(decisions, "validation_role")
    validation_role_count_map = _value_counts(validation_panel, "validation_role")
    summary = {
        "status": "methodology_v0_margin_validation_prepared",
        "date": "2026-05-28",
        "script": _rel(Path(__file__)),
        "margin_review_dir": _rel(margin_review_dir),
        "output_dir": _rel(output_dir),
        "input_pair_count": int(len(pair_rows)),
        "decision_pair_count": int(len(decisions)),
        "validation_panel_pair_count": int(len(validation_panel)),
        "methodology_v0_state_counts": state_count_map,
        "validation_role_counts": role_count_map,
        "validation_panel_role_counts": validation_role_count_map,
        "validation_question": COMMON_VALIDATION_QUESTION,
        "paths": {
            "decision_rows": _rel(output_dir / DECISION_ROWS_CSV),
            "state_counts": _rel(output_dir / STATE_COUNTS_CSV),
            "validation_panel": _rel(output_dir / VALIDATION_PANEL_CSV),
            "summary": _rel(output_dir / SUMMARY_JSON),
            "report": _rel(output_dir / REPORT_MD),
        },
        "decision": (
            "Freeze current wall gates unchanged. Treat boundary-sensitive "
            "margin rows as validation candidates only, with support-loss rows "
            "as contrasts."
        ),
        "claim_boundary": (
            "Methodology v0 preparation only; no route rerun, basin-quality "
            "evaluation, cost claim, directed-search claim, or wall-promotion "
            "rule change is made."
        ),
    }
    (output_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / CONFIG_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_report(output_dir / REPORT_MD, summary, decisions, validation_panel)
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margin-review-dir", type=Path, default=DEFAULT_MARGIN_REVIEW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.margin_review_dir, args.output_dir), indent=2))

if __name__ == "__main__":
    main()
