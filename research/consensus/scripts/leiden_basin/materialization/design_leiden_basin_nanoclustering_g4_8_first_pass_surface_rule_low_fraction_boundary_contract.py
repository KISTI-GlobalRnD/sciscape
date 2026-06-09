#!/usr/bin/env python3
"""Design the 001/007 low-fraction schedule-boundary audit contract.

This contract follows the surface-rule gap-fill trace audit. It does not reopen
the 15 screened gaps and does not seek new basin examples. It only checks
whether the previous 001/007 negative guards were artifacts of stopping the
bridge-fraction readout at 0.5.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from audit_leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_gap_fill_trace import (
    DEFAULT_OUTPUT_DIR as DEFAULT_GAP_FILL_AUDIT_DIR,
    GATE_MATRIX_CSV as GAP_FILL_AUDIT_GATE_MATRIX_CSV,
    SUMMARY_JSON as GAP_FILL_AUDIT_SUMMARY_JSON,
    UPDATED_PAIR_ROWS_CSV as GAP_FILL_AUDIT_UPDATED_PAIR_ROWS_CSV,
)
from design_leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_gap_fill_contract import (
    DEFAULT_OUTPUT_DIR as DEFAULT_GAP_FILL_CONTRACT_DIR,
    ROUTE_PLAN_ROWS_CSV as GAP_FILL_ROUTE_PLAN_ROWS_CSV,
)
from run_leiden_basin_nanoclustering_role_local_route_pilot import (
    BASE_RESULT_DIR,
    _json_safe,
    _read_csv,
    _write_csv,
)
from surface_claim_schema_adapter import (
    surface_claim_count_dict as _count_dict,
    surface_claim_gate_row as _gate_row,
    surface_claim_json_dump as _json_dump,
)


DEFAULT_OUTPUT_DIR = (
    BASE_RESULT_DIR
    / "leiden_basin_nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_contract_gamma1e5_20260609"
)

PAIR_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_contract_pair_rows.csv"
)
ROUTE_PLAN_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_contract_route_plan_rows.csv"
)
ACCEPTANCE_RULE_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_contract_acceptance_rule_rows.csv"
)
DECISION_ROWS_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_contract_decision_rows.csv"
)
GATE_MATRIX_CSV = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_contract_gate_matrix.csv"
)
SUMMARY_JSON = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_contract_summary.json"
)
CONFIG_JSON = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_contract_config.json"
)
REPORT_MD = (
    "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_contract_report.md"
)

RUN_STATUS = "designed_nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_contract"
ROUTE_EXECUTION_STATUS = "design_only_not_executed"
WALL_PROMOTION_STATUS = "not_promoted_low_fraction_boundary_contract_only"
METHOD_STATUS = "surface_rule_low_fraction_boundary_contract_not_method"
CLAIM_BOUNDARY = (
    "NanoClustering G4.8 first-pass 001/007 low-fraction schedule-boundary "
    "contract only; it tests whether the prior 0.5 lower-bound negative guard "
    "was schedule-bound. It keeps screened gaps closed and does not promote "
    "wall, pathway, panel-generality, quality/cost, full-replay, or method claims."
)

CANDIDATE_IDS = ("local_pair_001", "local_pair_007")
LOW_FRACTIONS = (0.5, 0.375, 0.25, 0.125, 0.0)
ROUTE_FAMILY = "first_pass_surface_rule_low_fraction_boundary_scan"
EXPECTED_ROUTE_PLAN_ROWS = 30

ACCEPTANCE_RULES: tuple[dict[str, str], ...] = (
    {
        "rule_id": "LF1",
        "rule_group": "scope",
        "rule_question": "Are only 001/007 reopened?",
        "acceptance_requirement": "execution_pair_ids == {local_pair_001, local_pair_007}",
        "claim_effect": "prevents the 15 screened gaps from becoming a new sweep",
    },
    {
        "rule_id": "LF2",
        "rule_group": "schedule_boundary",
        "rule_question": "Is this only a 0.5 lower-bound artifact check?",
        "acceptance_requirement": "bridge_fraction in {0.5,0.375,0.25,0.125,0.0}",
        "claim_effect": "tests the previous schedule boundary without changing starts or guards",
    },
    {
        "rule_id": "LF3",
        "rule_group": "reinforced_negative",
        "rule_question": "What reinforces the negative guard?",
        "acceptance_requirement": "no low-fraction target-like or finite single-side signal",
        "claim_effect": "may strengthen the negative guard only",
    },
    {
        "rule_id": "LF4",
        "rule_group": "late_collapse",
        "rule_question": "What indicates a late target collapse?",
        "acceptance_requirement": "target-like state appears below 0.5 without a finite single-side band",
        "claim_effect": "reclassifies the row as a late-collapse guard, not 016-like",
    },
    {
        "rule_id": "LF5",
        "rule_group": "single_side_signal",
        "rule_question": "What reopens positive diagnostic scrutiny?",
        "acceptance_requirement": "single-side band appears below 0.5 under the same starts/seeds",
        "claim_effect": "opens diagnostic-candidate wording only after a separate audit",
    },
    {
        "rule_id": "LF6",
        "rule_group": "independent_readout",
        "rule_question": "Does the trace remain a local independent-fraction readout?",
        "acceptance_requirement": "no warm-start pathway or wall claim is inferred from the scan",
        "claim_effect": "blocks pathway wording",
    },
    {
        "rule_id": "LF7",
        "rule_group": "claim_boundary",
        "rule_question": "Are method, wall, pathway, quality, replay, and generality claims closed?",
        "acceptance_requirement": "all promotion flags remain false",
        "claim_effect": "claim promotion blocked",
    },
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 80) -> str:
    cols = [column for column in columns if column in frame.columns]
    if not cols:
        return "_No matching columns._"
    visible = frame[cols].head(max_rows)
    if visible.empty:
        return "_No rows._"

    def cell(value: Any) -> str:
        if isinstance(value, (dict, list, tuple, set)):
            return _json_dump(value).replace("|", "\\|")
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in visible.itertuples(index=False):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def _pair_rows(audit_pair_rows: pd.DataFrame) -> pd.DataFrame:
    rows = audit_pair_rows[
        audit_pair_rows["local_pair_id"].astype(str).isin(CANDIDATE_IDS)
    ].copy()
    rows["next_contract_role"] = "low_fraction_boundary_candidate"
    rows["next_contract_action"] = "execute_low_fraction_schedule_boundary_audit"
    rows["next_contract_reason"] = (
        "prior 001/007 negative guard stops at bridge fraction 0.5"
    )
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows.sort_values("local_pair_id", kind="mergesort").reset_index(drop=True)


def _route_plan_rows(upstream_route_plan: pd.DataFrame) -> pd.DataFrame:
    starts = (
        upstream_route_plan[
            upstream_route_plan["local_pair_id"].astype(str).isin(CANDIDATE_IDS)
        ][
            [
                "local_pair_id",
                "next_contract_role",
                "start_condition",
                "start_condition_macro_role",
                "start_condition_expected_validation_pass",
            ]
        ]
        .drop_duplicates()
        .sort_values(["local_pair_id", "start_condition"], kind="mergesort")
    )
    route_rows: list[dict[str, Any]] = []
    for start in starts.itertuples(index=False):
        pair_id = str(start.local_pair_id)
        for index, fraction in enumerate(LOW_FRACTIONS, start=1):
            route_rows.append(
                {
                    "route_contract_id": (
                        f"surface_low_fraction_boundary_{pair_id.replace('local_pair_', '')}_"
                        f"{start.start_condition}_lf{index:02d}"
                    ),
                    "local_pair_id": pair_id,
                    "next_contract_role": "low_fraction_boundary_candidate",
                    "route_family": ROUTE_FAMILY,
                    "start_condition": str(start.start_condition),
                    "start_condition_macro_role": str(start.start_condition_macro_role),
                    "start_condition_expected_validation_pass": bool(
                        start.start_condition_expected_validation_pass
                    ),
                    "bridge_fraction": float(fraction),
                    "fraction_order": int(index),
                    "positive_recurrence_claim_allowed_after_contract": False,
                    "panel_generality_claim_allowed_after_contract": False,
                    "wall_claim_allowed_after_contract": False,
                    "pathway_claim_allowed_after_contract": False,
                    "method_claim_allowed_after_contract": False,
                    "quality_cost_claim_allowed_after_contract": False,
                    "route_execution_status": ROUTE_EXECUTION_STATUS,
                    "wall_promotion_status": WALL_PROMOTION_STATUS,
                    "method_status": METHOD_STATUS,
                    "run_status": RUN_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
    return pd.DataFrame(route_rows)


def _acceptance_rule_rows() -> pd.DataFrame:
    rows = pd.DataFrame(ACCEPTANCE_RULES)
    rows["route_execution_status"] = ROUTE_EXECUTION_STATUS
    rows["wall_promotion_status"] = WALL_PROMOTION_STATUS
    rows["method_status"] = METHOD_STATUS
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _decision_rows() -> pd.DataFrame:
    rows = pd.DataFrame(
        [
            {
                "decision_id": "D1",
                "decision": "test_schedule_boundary_not_generality",
                "rationale": (
                    "001/007 are already scoreable negatives under the 0.5 stop; "
                    "the remaining doubt is whether that stop hid a lower-fraction state."
                ),
            },
            {
                "decision_id": "D2",
                "decision": "do_not_open_screened_gaps",
                "rationale": (
                    "The 15 screened gaps do not answer the schedule-boundary artifact "
                    "question and would turn the step back into a broad sweep."
                ),
            },
            {
                "decision_id": "D3",
                "decision": "reuse_same_starts_and_seeds",
                "rationale": (
                    "Changing starts would mix the boundary check with a new local "
                    "route search."
                ),
            },
            {
                "decision_id": "D4",
                "decision": "classify_three_outcomes",
                "rationale": (
                    "No signal reinforces the guard; target-only signal becomes a "
                    "late-collapse guard; single-side signal reopens diagnostic scrutiny."
                ),
            },
            {
                "decision_id": "D5",
                "decision": "block_pathway_wording",
                "rationale": (
                    "The planned execution is an independent-fraction local readout, "
                    "not a warm-start route through a wall."
                ),
            },
        ]
    )
    rows["run_status"] = RUN_STATUS
    rows["claim_boundary"] = CLAIM_BOUNDARY
    return rows


def _gate_matrix(
    *,
    audit_summary: dict[str, Any],
    audit_gates: pd.DataFrame,
    pair_rows: pd.DataFrame,
    route_plan_rows: pd.DataFrame,
    acceptance_rule_rows: pd.DataFrame,
) -> pd.DataFrame:
    fraction_values = sorted(route_plan_rows["bridge_fraction"].astype(float).unique())
    rows = [
        _gate_row(
            "G1_gap_fill_negative_context_ready",
            "Did the upstream gap-fill audit pass and classify 001/007 as negatives?",
            {
                "upstream_failed_gates": audit_summary.get("failed_gates"),
                "audit_gate_status_counts": _count_dict(audit_gates["gate_status"]),
                "surface_rule_classes": _count_dict(pair_rows["surface_rule_class"]),
            },
            "upstream audit passed and 001/007 are negative guards",
            bool(audit_summary.get("failed_gates") == [])
            and bool(audit_gates["gate_status"].astype(str).eq("pass").all())
            and bool(
                pair_rows["surface_rule_class"]
                .astype(str)
                .eq("gap_fill_scoreable_negative_no_recurrence_guard")
                .all()
            ),
        ),
        _gate_row(
            "G2_scope_only_001_007",
            "Are only 001/007 opened?",
            sorted(route_plan_rows["local_pair_id"].astype(str).unique()),
            "candidate ids equal 001/007",
            set(route_plan_rows["local_pair_id"].astype(str)) == set(CANDIDATE_IDS),
        ),
        _gate_row(
            "G3_low_fraction_schedule_fixed",
            "Is the route plan exactly the low-fraction schedule?",
            {
                "route_row_count": int(len(route_plan_rows)),
                "fraction_values": fraction_values,
            },
            "six starts times five fractions, including 0.5 anchor and 0.0 endpoint",
            int(len(route_plan_rows)) == EXPECTED_ROUTE_PLAN_ROWS
            and fraction_values == sorted(LOW_FRACTIONS),
        ),
        _gate_row(
            "G4_same_start_scope_reused",
            "Does the route plan reuse the existing allowed starts?",
            route_plan_rows.groupby("local_pair_id")["start_condition"]
            .nunique()
            .astype(int)
            .to_dict(),
            "001 has two starts and 007 has four starts",
            route_plan_rows.groupby("local_pair_id")["start_condition"]
            .nunique()
            .astype(int)
            .to_dict()
            == {"local_pair_001": 2, "local_pair_007": 4},
        ),
        _gate_row(
            "G5_acceptance_rules_predeclared",
            "Are the three outcome classes and claim boundary predeclared?",
            sorted(acceptance_rule_rows["rule_group"].astype(str).unique()),
            "scope, schedule, negative, late-collapse, single-side, independent readout, claims",
            {
                "scope",
                "schedule_boundary",
                "reinforced_negative",
                "late_collapse",
                "single_side_signal",
                "independent_readout",
                "claim_boundary",
            }.issubset(set(acceptance_rule_rows["rule_group"].astype(str))),
        ),
        _gate_row(
            "G6_no_claim_promotion",
            "Are all promotion flags closed?",
            {
                "wall_flags": _count_dict(route_plan_rows["wall_claim_allowed_after_contract"]),
                "pathway_flags": _count_dict(
                    route_plan_rows["pathway_claim_allowed_after_contract"]
                ),
                "method_flags": _count_dict(
                    route_plan_rows["method_claim_allowed_after_contract"]
                ),
            },
            "all route-plan promotion flags false",
            bool(route_plan_rows["wall_claim_allowed_after_contract"].eq(False).all())
            and bool(route_plan_rows["pathway_claim_allowed_after_contract"].eq(False).all())
            and bool(route_plan_rows["method_claim_allowed_after_contract"].eq(False).all())
            and bool(route_plan_rows["quality_cost_claim_allowed_after_contract"].eq(False).all()),
        ),
    ]
    return pd.DataFrame(rows)


def _summary(
    *,
    output_dir: Path,
    gap_fill_contract_dir: Path,
    gap_fill_audit_dir: Path,
    pair_rows: pd.DataFrame,
    route_plan_rows: pd.DataFrame,
    acceptance_rule_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> dict[str, Any]:
    failed_gates = list(
        gate_matrix.loc[gate_matrix["gate_status"].ne("pass"), "gate_id"].astype(str)
    )
    return {
        "schema": "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_contract_summary.v1",
        "status": RUN_STATUS,
        "output_dir": str(output_dir),
        "gap_fill_contract_dir": str(gap_fill_contract_dir),
        "gap_fill_audit_dir": str(gap_fill_audit_dir),
        "candidate_pair_ids": list(CANDIDATE_IDS),
        "route_plan_row_count": int(len(route_plan_rows)),
        "fraction_count": len(LOW_FRACTIONS),
        "low_fractions": list(LOW_FRACTIONS),
        "acceptance_rule_count": int(len(acceptance_rule_rows)),
        "decision_row_count": int(len(decision_rows)),
        "gate_status_counts": _count_dict(gate_matrix["gate_status"]),
        "failed_gates": failed_gates,
        "route_execution_opened": False,
        "wall_claim_allowed_after_contract": False,
        "pathway_claim_allowed_after_contract": False,
        "panel_generality_claim_allowed_after_contract": False,
        "method_claim_allowed_after_contract": False,
        "quality_cost_claim_allowed_after_contract": False,
        "interpretation": (
            "This contract only tests whether the 001/007 negative guard was an "
            "artifact of stopping at bridge fraction 0.5."
        ),
        "recommended_next_gate": (
            "Execute the 30 route rows over 8 seeds, then classify each pair as "
            "reinforced negative, late target collapse, or low-fraction single-side "
            "diagnostic candidate. Keep pathway and generality claims closed."
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _report(
    *,
    summary: dict[str, Any],
    pair_rows: pd.DataFrame,
    route_plan_rows: pd.DataFrame,
    acceptance_rule_rows: pd.DataFrame,
    decision_rows: pd.DataFrame,
    gate_matrix: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# NanoClustering G4.8 001/007 Low-Fraction Boundary Contract",
            "",
            f"- status: `{summary['status']}`",
            f"- candidate_pair_ids: {summary['candidate_pair_ids']}",
            f"- route_plan_row_count: {summary['route_plan_row_count']}",
            f"- low_fractions: {summary['low_fractions']}",
            f"- gate_status_counts: {summary['gate_status_counts']}",
            f"- failed_gates: {summary['failed_gates']}",
            f"- interpretation: {summary['interpretation']}",
            f"- recommended_next_gate: {summary['recommended_next_gate']}",
            f"- claim_boundary: {CLAIM_BOUNDARY}",
            "",
            "## Pair Rows",
            "",
            _markdown_table(
                pair_rows,
                [
                    "local_pair_id",
                    "surface_rule_class",
                    "generalization_status",
                    "promotion_status",
                    "next_contract_action",
                    "next_contract_reason",
                ],
            ),
            "",
            "## Route Plan Rows",
            "",
            _markdown_table(
                route_plan_rows,
                [
                    "route_contract_id",
                    "local_pair_id",
                    "start_condition",
                    "start_condition_macro_role",
                    "bridge_fraction",
                    "fraction_order",
                ],
            ),
            "",
            "## Acceptance Rules",
            "",
            _markdown_table(
                acceptance_rule_rows,
                [
                    "rule_id",
                    "rule_group",
                    "rule_question",
                    "acceptance_requirement",
                    "claim_effect",
                ],
            ),
            "",
            "## Decisions",
            "",
            _markdown_table(decision_rows, ["decision_id", "decision", "rationale"]),
            "",
            "## Gate Matrix",
            "",
            _markdown_table(
                gate_matrix,
                ["gate_id", "gate_status", "observed", "minimum_or_rule", "question"],
            ),
            "",
            "## Boundary",
            "",
            "This is a schedule-boundary design contract, not a pathway or method result.",
            "",
        ]
    )


def run(
    *,
    gap_fill_contract_dir: Path,
    gap_fill_audit_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    audit_summary = _read_json(gap_fill_audit_dir / GAP_FILL_AUDIT_SUMMARY_JSON)
    audit_gates = _read_csv(gap_fill_audit_dir / GAP_FILL_AUDIT_GATE_MATRIX_CSV)
    audit_pair_rows = _read_csv(gap_fill_audit_dir / GAP_FILL_AUDIT_UPDATED_PAIR_ROWS_CSV)
    upstream_route_plan = _read_csv(gap_fill_contract_dir / GAP_FILL_ROUTE_PLAN_ROWS_CSV)

    pair_rows = _pair_rows(audit_pair_rows)
    route_plan_rows = _route_plan_rows(upstream_route_plan)
    acceptance_rule_rows = _acceptance_rule_rows()
    decision_rows = _decision_rows()
    gate_matrix = _gate_matrix(
        audit_summary=audit_summary,
        audit_gates=audit_gates,
        pair_rows=pair_rows,
        route_plan_rows=route_plan_rows,
        acceptance_rule_rows=acceptance_rule_rows,
    )
    summary = _summary(
        output_dir=output_dir,
        gap_fill_contract_dir=gap_fill_contract_dir,
        gap_fill_audit_dir=gap_fill_audit_dir,
        pair_rows=pair_rows,
        route_plan_rows=route_plan_rows,
        acceptance_rule_rows=acceptance_rule_rows,
        decision_rows=decision_rows,
        gate_matrix=gate_matrix,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(pair_rows, output_dir / PAIR_ROWS_CSV)
    _write_csv(route_plan_rows, output_dir / ROUTE_PLAN_ROWS_CSV)
    _write_csv(acceptance_rule_rows, output_dir / ACCEPTANCE_RULE_ROWS_CSV)
    _write_csv(decision_rows, output_dir / DECISION_ROWS_CSV)
    _write_csv(gate_matrix, output_dir / GATE_MATRIX_CSV)
    (output_dir / SUMMARY_JSON).write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    config = {
        "schema": "nanoclustering_g4_8_first_pass_surface_rule_low_fraction_boundary_contract_config.v1",
        "gap_fill_contract_dir": str(gap_fill_contract_dir),
        "gap_fill_audit_dir": str(gap_fill_audit_dir),
        "output_dir": str(output_dir),
        "candidate_pair_ids": list(CANDIDATE_IDS),
        "low_fractions": list(LOW_FRACTIONS),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (output_dir / CONFIG_JSON).write_text(
        json.dumps(_json_safe(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / REPORT_MD).write_text(
        _report(
            summary=summary,
            pair_rows=pair_rows,
            route_plan_rows=route_plan_rows,
            acceptance_rule_rows=acceptance_rule_rows,
            decision_rows=decision_rows,
            gate_matrix=gate_matrix,
        ),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-fill-contract-dir", type=Path, default=DEFAULT_GAP_FILL_CONTRACT_DIR)
    parser.add_argument("--gap-fill-audit-dir", type=Path, default=DEFAULT_GAP_FILL_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        gap_fill_contract_dir=args.gap_fill_contract_dir,
        gap_fill_audit_dir=args.gap_fill_audit_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
